"""
ComfyUI Custom Node: Qwen 3.5 4B Text Encoder for Anima 2B

Provides a CLIP-compatible loader for the Qwen 3.5 4B hybrid (Mamba2 + Attention)
text encoder used with the Anima 2B diffusion model (cosmos-qwen3.5).

CRITICAL: This model IS the official Qwen3.5-4B text backbone (same architecture,
same vocab_size=248320). It requires the Qwen3.5 tokenizer, NOT the Qwen3 tokenizer.
The Qwen3 tokenizer has vocab=151936, leaving 96K embedding rows untouched.

Also includes the Qwen3.5 Vision Transformer (ViT) for image-conditioned generation.
The ViT extracts visual features from reference images and injects them into the
text encoder's token sequence, enabling style transfer, character consistency, and
image-guided generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import math
from typing import Optional, Tuple

import comfy.sd
import comfy.ops
import comfy.model_management
import comfy.utils
import comfy.sd1_clip
import comfy.text_encoders.hunyuan_video
from comfy.ldm.common_dit import rms_norm
from comfy.supported_models_base import ClipTarget
import safetensors.torch as safetensors_torch

logger = logging.getLogger(__name__)

# Path to this node's directory (for bundled tokenizer files)
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
QWEN35_TOKENIZER_DIR = os.path.join(NODE_DIR, "qwen35_tokenizer")


# ============================================================================
# Model Architecture Components
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm with learnable scale."""
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class ExpRMSNorm(nn.Module):
    """
    RMSNorm with exp(weight) parameterization.

    Used for the late normalization layer where learned weights are near-zero
    (mean~-0.003). Standard RMSNorm would interpret these as "scale to ~0",
    collapsing all information. With exp(weight), near-zero means exp(0)~1,
    i.e. "nearly identity normalization" with tiny learned perturbations.

    Evidence:
    - All internal RMSNorms have weights centered 0.04-1.11 (normal scaling)
    - ONLY the late norm has weights at -0.003 (different parameterization)
    - Direct weight: diversity=0.003, cross=0.999 (COLLAPSED)
    - exp(weight): diversity=0.821, cross=0.689 (PRESERVED)
    """
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, torch.exp(self.weight.float()).to(x.dtype), self.eps)


class BiasAdd(nn.Module):
    """Simple module that adds a learnable bias."""
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x + self.bias


class SSMBlock(nn.Module):
    """
    Mamba2-style Selective State Space Model block (reference: state-spaces/mamba).

    Architecture verified against reference Mamba2:
    - in_proj_qkv: Linear(2560, 8192) -> conv1d -> silu -> split into x(4096), B(2048), C(2048)
      where d_ssm=4096, ngroups*d_state=2048 each
    - in_proj_z: Linear(2560, 4096) -> gate z that BYPASSES conv1d
    - in_proj_b: Linear(2560, 32) -> per-group B bias / additional modulation
    - in_proj_a: Linear(2560, 32) -> per-group C bias / additional modulation
    - A_log: [32] -> state transition (nheads=32)
    - dt_bias: [32] -> discretization timestep bias (nheads=32)
    - conv1d: depthwise Conv1d(8192, 8192, kernel=4)
    - norm: RMSNorm(128) -> per-head norm (head_dim=128)
    - out_proj: Linear(4096, 2560) -> d_ssm -> hidden_size

    Dimensions: d_ssm=4096, nheads=32, head_dim=128, ngroups=32, d_state=64
    conv_dim = d_ssm + 2*ngroups*d_state = 4096 + 2*32*64 = 8192
    """
    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32,
                 d_gate=4096, conv_kernel=4, norm_dim=128,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.hidden_size = hidden_size
        self.d_inner = d_inner  # conv channels: d_ssm + 2*ngroups*d_state
        self.n_groups = n_groups  # also nheads (1 head per group)
        self.d_ssm = d_gate  # 4096 = nheads * head_dim
        self.head_dim = d_gate // n_groups  # 128
        self.d_state = (d_inner - d_gate) // (2 * n_groups)  # 64

        self.in_proj_qkv = ops.Linear(hidden_size, d_inner, bias=False, device=device, dtype=dtype)
        self.in_proj_z = ops.Linear(hidden_size, d_gate, bias=False, device=device, dtype=dtype)
        self.in_proj_a = ops.Linear(hidden_size, n_groups, bias=False, device=device, dtype=dtype)
        self.in_proj_b = ops.Linear(hidden_size, n_groups, bias=False, device=device, dtype=dtype)

        # Use ops.Conv1d for auto device/dtype casting (matches comfy.ops behavior)
        self.conv1d = ops.Conv1d(
            d_inner, d_inner, conv_kernel, groups=d_inner,
            padding=conv_kernel - 1, bias=False, device=device, dtype=dtype
        )

        self.out_proj = ops.Linear(d_gate, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm = RMSNorm(norm_dim, device=device, dtype=dtype)

        self.A_log = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))

    def _ssm_scan(self, x, B_state, C_state, dt_input, D_input):
        """
        Multi-head SSM scan with d_state > 1 (proper Mamba2 recurrence).

        x: [B, L, nheads, head_dim]  (the SSM input, nheads=32, head_dim=128)
        B_state: [B, L, ngroups, d_state]  (input matrix, ngroups=32, d_state=64)
        C_state: [B, L, ngroups, d_state]  (output matrix, ngroups=32, d_state=64)
        dt_input: [B, L, nheads]  (input-dependent discretization step)
        D_input: [B, L, nheads]   (input-dependent skip connection)

        SSM recurrence per head n (in group g):
            dt = softplus(dt_input + dt_bias)
            dA = exp(dt * A)
            h[n] = dA[n] * h[n] + dt[n] * (B[g] outer x[n])
            y[n] = (C[g] . h[n]) + D[n] * x[n]  (skip connection)

        State shape: [batch, nheads, head_dim, d_state]
        Returns: [B, L, nheads, head_dim]
        """
        batch, seq_len, nheads, head_dim = x.shape
        d_state = B_state.shape[-1]
        device = x.device
        compute_dtype = torch.float32

        # Move params to device
        A = -torch.exp(self.A_log.to(device=device).float())  # [nheads] (negative)
        dt_bias = self.dt_bias.to(device=device).float()  # [nheads]

        # State: [batch, nheads, head_dim, d_state]
        h = torch.zeros(batch, nheads, head_dim, d_state, device=device, dtype=compute_dtype)
        outputs = []

        x_f = x.float()
        B_f = B_state.float()  # [B, L, ngroups, d_state]
        C_f = C_state.float()  # [B, L, ngroups, d_state]
        dt_f = dt_input.float()  # [B, L, nheads]
        D_f = D_input.float()   # [B, L, nheads]

        for t in range(seq_len):
            x_t = x_f[:, t]  # [B, nheads, head_dim]
            B_t = B_f[:, t]  # [B, ngroups, d_state]
            C_t = C_f[:, t]  # [B, ngroups, d_state]

            # Input-dependent dt: softplus(dt_input + dt_bias) [B, nheads]
            dt_t = F.softplus(dt_f[:, t] + dt_bias)  # [B, nheads]

            # Discretize: dA = exp(A * dt) per head per batch
            dA_t = torch.exp(dt_t * A.unsqueeze(0))  # [B, nheads]

            # dBx = dt * outer(x_t, B_t): [B, nheads, head_dim, d_state]
            dt_expanded = dt_t.unsqueeze(-1).unsqueeze(-1)  # [B, nheads, 1, 1]
            dBx = dt_expanded * torch.einsum('bnh,bns->bnhs', x_t, B_t)

            # State update: h = dA * h + dBx
            dA_expanded = dA_t.unsqueeze(-1).unsqueeze(-1)  # [B, nheads, 1, 1]
            h = dA_expanded * h + dBx

            # Output: y_t = einsum(h, C_t) over d_state + D * x (skip)
            y_t = torch.einsum('bnhs,bns->bnh', h, C_t)  # [B, nheads, head_dim]
            y_t = y_t + D_f[:, t].unsqueeze(-1) * x_t  # D skip connection

            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(x.dtype)  # [B, L, nheads, head_dim]

    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape

        # === Gate (bypasses conv1d, reference Mamba2 pattern) ===
        z = self.in_proj_z(hidden_states)  # [B, L, 4096] - the gate

        # === xBC goes through conv1d ===
        xBC = self.in_proj_qkv(hidden_states)  # [B, L, 8192]

        # in_proj_b -> input-dependent dt (time step for selective SSM)
        # in_proj_a -> input-dependent D (skip connection, no separate D param exists)
        dt_input = self.in_proj_b(hidden_states)  # [B, L, 32] (nheads)
        D_input = self.in_proj_a(hidden_states)   # [B, L, 32] (nheads)

        # Causal 1D convolution + activation
        xBC_conv = xBC.transpose(1, 2)  # [B, 8192, L]
        xBC_conv = self.conv1d(xBC_conv)[..., :seq_len]
        xBC_conv = F.silu(xBC_conv.transpose(1, 2))  # [B, L, 8192]

        # Split conv output: x(d_ssm=4096), B_conv(ngroups*d_state=2048), C_conv(2048)
        x, B_conv, C_conv = torch.split(
            xBC_conv,
            [self.d_ssm, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1
        )

        # Reshape for SSM
        x = x.reshape(batch, seq_len, self.n_groups, self.head_dim)  # [B, L, 32, 128]
        B_state = B_conv.reshape(batch, seq_len, self.n_groups, self.d_state)  # [B, L, 32, 64]
        C_state = C_conv.reshape(batch, seq_len, self.n_groups, self.d_state)  # [B, L, 32, 64]

        # SSM scan (with input-dependent dt and D)
        y = self._ssm_scan(x, B_state, C_state, dt_input, D_input)  # [B, L, 32, 128]

        # Per-head RMSNorm (norm_dim=128=head_dim)
        y = self.norm(y)

        # Reshape and apply gating: y = norm(y) * silu(z) (RMSNormGated pattern)
        y = y.reshape(batch, seq_len, -1)  # [B, L, 4096]
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)


class GatedSelfAttention(nn.Module):
    """
    Self-attention with gated Q projection.

    q_proj outputs Q(4096) + gate(4096) = 8192:
    - 16 attention heads with 256 head_dim
    - 4 KV heads with 256 head_dim (GQA ratio 4)
    - After attention: [B, L, 4096] gated by silu(gate) -> o_proj
    """
    def __init__(self, hidden_size=2560, num_heads=16, num_kv_heads=4,
                 head_dim=256, rope_theta=1000000.0,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_heads // num_kv_heads
        self.inner_dim = num_heads * head_dim  # 4096

        self.q_proj = ops.Linear(hidden_size, 2 * self.inner_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = ops.Linear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ops.Linear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_dim, hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        B, L, _ = hidden_states.shape

        # Q projection with gate
        qg = self.q_proj(hidden_states)  # [B, L, 8192]
        q, gate = qg.chunk(2, dim=-1)  # [B, L, 4096] each

        # Reshape to heads
        q = q.view(B, L, self.num_heads, self.head_dim)  # [B, L, 16, 256]
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)  # [B, L, 4, 256]
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)  # [B, L, 4, 256]

        # Per-head norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose for attention: [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

        # GQA: expand K, V
        k = k.repeat_interleave(self.gqa_ratio, dim=1)  # [B, 16, L, 256]
        v = v.repeat_interleave(self.gqa_ratio, dim=1)  # [B, 16, L, 256]

        # Attention (ensure mask dtype matches query)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(dtype=q.dtype)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attention_mask is None)
        )  # [B, 16, L, 256]

        # Reshape and gate
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.inner_dim)  # [B, L, 4096]
        attn_out = attn_out * F.silu(gate)

        return self.o_proj(attn_out)


def _apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings."""
    # x: [B, H, L, D]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _precompute_freqs_cis(head_dim, max_seq_len, theta=1000000.0, device=None, dtype=None):
    """Precompute RoPE frequencies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    # Duplicate for full head_dim
    cos = cos.repeat(1, 1, 1, 2)  # [1, 1, L, D]
    sin = sin.repeat(1, 1, 1, 2)  # [1, 1, L, D]
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


class MLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    """
    A single transformer block that uses either SSM or self-attention.
    """
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 use_ssm=True, has_mlp=True,
                 device=None, dtype=None, ops=None):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp

        self.input_layernorm = RMSNorm(hidden_size, device=device, dtype=dtype)

        if use_ssm:
            self.linear_attn = SSMBlock(
                hidden_size=hidden_size,
                device=device, dtype=dtype, ops=ops
            )
        else:
            self.self_attn = GatedSelfAttention(
                hidden_size=hidden_size,
                device=device, dtype=dtype, ops=ops
            )

        if has_mlp:
            self.post_attention_layernorm = RMSNorm(hidden_size, device=device, dtype=dtype)
            self.mlp = MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=device, dtype=dtype, ops=ops
            )

    def forward(self, x, attention_mask=None, freqs_cis=None):
        # Pre-norm + attention/SSM
        residual = x
        x_norm = self.input_layernorm(x)

        if self.use_ssm:
            x = residual + self.linear_attn(x_norm)
        else:
            x = residual + self.self_attn(x_norm, attention_mask=attention_mask, freqs_cis=freqs_cis)

        # Pre-norm + MLP
        if self.has_mlp:
            residual = x
            x = residual + self.mlp(self.post_attention_layernorm(x))

        return x


# ============================================================================
# Full Qwen 3.5 Hybrid Model
# ============================================================================

class Qwen35HybridModel(nn.Module):
    """
    Qwen 3.5 4B Hybrid Text Encoder.

    This is the official Qwen3.5-4B text backbone (confirmed by matching config.json
    from Qwen/Qwen3.5-4B: same vocab_size=248320, hidden_size=2560, 32 layers,
    same linear_attention/full_attention pattern every 4 layers).

    32 layers with alternating SSM/attention:
    - SSM layers: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30
    - Self-attention layers: 3,7,11,15,19,23,27,31
    - Layer 31: self-attention only (no MLP)
    - Final norm: Linear(2560->1024) + ExpRMSNorm(1024) + SiLU + Linear(1024->1024)

    The late norm uses exp(weight) parameterization for RMSNorm. The learned weights
    are near-zero (~-0.003), which with exp() gives scale ~ 0.997 ~ 1.0 (near-identity).
    Standard w*norm would collapse all tokens to the same vector (diversity=0.003).
    With exp(w)*norm, token diversity is preserved (diversity=0.82).
    """
    SELF_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE = 248320
    OUTPUT_DIM = 1024
    HEAD_DIM = 256  # For RoPE
    ROPE_THETA = 1000000.0
    DEFAULT_OUTPUT_SCALE = 1.0  # Raw output; use slider to experiment

    # Path to calibration parameters (computed by calibrate.py)
    CALIBRATION_FILE = os.path.join(NODE_DIR, "calibration_params.safetensors")

    # Path to Procrustes alignment matrix (computed by compute_alignment.py)
    ALIGNMENT_FILE = os.path.join(NODE_DIR, "rotation_matrix.safetensors")

    def __init__(self, config_dict=None, dtype=None, device=None, operations=None):
        super().__init__()
        if config_dict is None:
            config_dict = {}
        ops = operations or comfy.ops.disable_weight_init

        self.num_layers = self.NUM_LAYERS
        self.dtype = dtype

        # Token embeddings
        self.embed_tokens = ops.Embedding(
            self.VOCAB_SIZE, self.HIDDEN_SIZE, device=device, dtype=dtype
        )

        # Transformer blocks
        self.layers = nn.ModuleList()
        for i in range(self.NUM_LAYERS):
            use_ssm = (i not in self.SELF_ATTN_LAYERS)
            has_mlp = (i != 31)  # Layer 31 has no MLP
            self.layers.append(HybridBlock(
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                use_ssm=use_ssm,
                has_mlp=has_mlp,
                device=device, dtype=dtype, ops=ops
            ))

        # Output projection: Linear(2560->1024) + ExpRMSNorm + SiLU + Linear(1024->1024)
        self.norm = nn.Sequential(
            ops.Linear(self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
            ExpRMSNorm(self.OUTPUT_DIM, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
        )

        # Output scaling (set externally via config_dict or after construction)
        self._output_scale = config_dict.get("output_scale", self.DEFAULT_OUTPUT_SCALE)

        # Per-dimension affine calibration (computed by calibrate.py)
        self._calibration_scale = None  # [1024]
        self._calibration_bias = None   # [1024]
        self._use_calibration = config_dict.get("use_calibration", False)
        if self._use_calibration:
            self._load_calibration()

        # Procrustes rotation alignment (computed by compute_alignment.py)
        # Rotates 4B embedding directions to match 0.6B's concept space
        # This is a full 1024x1024 orthogonal rotation — preserves distances
        # but re-orients spatial/pose concepts to match what the adapter expects
        self._rotation_matrix = None   # [1024, 1024]
        self._rotation_mean_4b = None  # [1024]
        self._rotation_mean_06b = None # [1024]
        self._use_alignment = config_dict.get("use_alignment", False)
        self._alignment_strength = config_dict.get("alignment_strength", 1.0)  # 0..1 blend
        if self._use_alignment:
            self._load_alignment()

        # Pending visual embeddings for vision-text encoding
        # Set externally before forward(), cleared after
        self._pending_visual_embeds = None
        self._pending_vision_weight = 1.0   # Scale factor applied AFTER norm projection
        self._pending_vision_mode = "add"   # "add", "concat", or "replace_padding"

    def _load_calibration(self):
        """Load per-dimension affine calibration parameters."""
        if os.path.exists(self.CALIBRATION_FILE):
            try:
                cal = safetensors_torch.load_file(self.CALIBRATION_FILE)
                self._calibration_scale = cal["scale"].float()  # [1024]
                self._calibration_bias = cal["bias"].float()    # [1024]
                logger.info(f"[Qwen3.5-Anima] Loaded calibration: scale mean={self._calibration_scale.mean():.3f}, bias mean={self._calibration_bias.mean():.3f}")
            except Exception as e:
                logger.warning(f"[Qwen3.5-Anima] Failed to load calibration: {e}")
                self._use_calibration = False
        else:
            logger.warning(f"[Qwen3.5-Anima] Calibration file not found: {self.CALIBRATION_FILE}")
            self._use_calibration = False

    def _load_alignment(self):
        """Load Procrustes rotation matrix for 4B→0.6B concept alignment."""
        if os.path.exists(self.ALIGNMENT_FILE):
            try:
                data = safetensors_torch.load_file(self.ALIGNMENT_FILE)
                self._rotation_matrix = data["rotation"].float()     # [1024, 1024]
                self._rotation_mean_4b = data["mean_4b"].float()     # [1024]
                self._rotation_mean_06b = data["mean_06b"].float()   # [1024]
                # Use slogdet for numerical stability (det() on 1024x1024 float32 underflows to 0)
                sign, logabsdet = torch.linalg.slogdet(self._rotation_matrix.double())
                logger.info(
                    f"[Qwen3.5-Anima] Loaded Procrustes alignment: "
                    f"R shape={self._rotation_matrix.shape}, "
                    f"det={sign.item():+.0f}, "
                    f"mean_4b L2={self._rotation_mean_4b.norm():.1f}, "
                    f"mean_06b L2={self._rotation_mean_06b.norm():.1f}"
                )
            except Exception as e:
                logger.warning(f"[Qwen3.5-Anima] Failed to load alignment: {e}")
                self._use_alignment = False
        else:
            logger.warning(f"[Qwen3.5-Anima] Alignment file not found: {self.ALIGNMENT_FILE}")
            self._use_alignment = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=None, **kwargs):
        # Get embeddings
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(input_ids, out_dtype=dtype or torch.float32)

        # NOTE: Visual features (from ViT) are NOT fed through the backbone.
        # The backbone was trained on text only — visual prefixes would be
        # treated as noise, corrupting both visual and text representations.
        # Instead, visual features are projected through the norm (2560→1024)
        # separately and prepended to the output AFTER backbone processing.

        seq_len = x.shape[1]

        # Precompute RoPE frequencies for self-attention layers
        freqs_cis = _precompute_freqs_cis(
            self.HEAD_DIM, seq_len,
            theta=self.ROPE_THETA,
            device=x.device, dtype=x.dtype
        )

        # Build causal attention mask for self-attention layers
        attn_mask = None
        if attention_mask is not None:
            mask_fill = torch.finfo(x.dtype).min / 4
            causal = torch.empty(
                seq_len, seq_len, dtype=x.dtype, device=x.device
            ).fill_(mask_fill).triu_(1)
            pad_mask = 1.0 - attention_mask.to(x.dtype).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            ).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            pad_mask = pad_mask.masked_fill(pad_mask.to(torch.bool), mask_fill)
            attn_mask = causal + pad_mask
        elif seq_len > 1:
            mask_fill = torch.finfo(x.dtype).min / 4
            attn_mask = torch.empty(
                seq_len, seq_len, dtype=x.dtype, device=x.device
            ).fill_(mask_fill).triu_(1)

        # Process through layers (text only)
        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=attn_mask, freqs_cis=freqs_cis)

            # Capture intermediate output if requested
            if intermediate_output is not None:
                if isinstance(intermediate_output, int) and i == intermediate_output:
                    intermediate = x.clone()
                elif isinstance(intermediate_output, list) and i in intermediate_output:
                    if intermediate is None:
                        intermediate = {}
                    intermediate[i] = x.clone()

        # ── Visual injection in 2560-dim space (BEFORE norm) ──────────────
        # For "add" mode, inject here so the norm pipeline processes the
        # combined text+visual signal.  Different images perturb the hidden
        # states in different *directions*, and those directional differences
        # survive the norm even though magnitudes get normalised.
        # For "concat" / "replace_padding", we project visual tokens through
        # the full norm separately (they exist as distinct tokens).
        n_visual = 0
        _vis_for_post_norm = None  # visual embeds held for concat/replace_padding
        if self._pending_visual_embeds is not None:
            visual = self._pending_visual_embeds.to(device=x.device, dtype=x.dtype)
            n_visual = visual.shape[1]
            mode = self._pending_vision_mode
            weight = self._pending_vision_weight

            if mode == "add":
                # Mean-pool 196 ViT patches → single 2560-dim style vector
                style_vec_2560 = visual.mean(dim=1, keepdim=True)  # [B, 1, 2560]
                # Normalise style vector to match text hidden-state magnitude.
                # Raw ViT style is ~4× smaller than text (L2 3.5 vs 15),
                # so without scaling the perturbation is negligible after norm.
                # With this, weight=1.0 means "visual same magnitude as text".
                text_scale = x.norm(dim=-1).mean().clamp(min=1e-6)
                style_scale = style_vec_2560.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                style_vec_2560 = style_vec_2560 * (text_scale / style_scale)
                # Add as residual to every text hidden state before norm
                x = x + weight * style_vec_2560
                n_visual = 0  # no extra tokens in output
            else:
                # Keep visual embeds for projection after norm
                _vis_for_post_norm = (visual, n_visual, mode, weight)

        # Apply output projection (Linear -> ExpRMSNorm -> SiLU -> Linear)
        # Maps 2560-dim hidden states to 1024-dim adapter space
        x = self.norm(x)

        # Procrustes alignment: rotation + optional bias shift.
        # Decomposed into two parts:
        #   1. Rotation: R @ (x - m4b) — always applied when alignment is on.
        #      This re-orients concept directions (e.g. "from side") to match
        #      what the 0.6B-trained adapter expects.  Preserves norms.
        #   2. Bias shift: re-center from m4b to m06b (blended by alignment_strength).
        #      m06b has L2=70 vs m4b L2=5, so full shift dramatically changes
        #      output magnitude.  strength=0 keeps 4B's own scale, strength=1
        #      shifts fully to 0.6B's characteristic bias.
        if self._use_alignment and self._rotation_matrix is not None:
            R = self._rotation_matrix.to(device=x.device, dtype=x.dtype)
            m4b = self._rotation_mean_4b.to(device=x.device, dtype=x.dtype)
            m06b = self._rotation_mean_06b.to(device=x.device, dtype=x.dtype)
            alpha = self._alignment_strength
            # Always rotate (fixes concept directions)
            x_rotated = torch.einsum('ij,...j->...i', R, x - m4b)
            # Blend the re-centering: (1-α)*m4b + α*m06b
            x = x_rotated + (1.0 - alpha) * m4b + alpha * m06b

        # Per-dimension affine calibration
        if self._use_calibration and self._calibration_scale is not None:
            cal_scale = self._calibration_scale.to(device=x.device, dtype=x.dtype)
            cal_bias = self._calibration_bias.to(device=x.device, dtype=x.dtype)
            x = x * cal_scale + cal_bias

        # Additional uniform scaling
        if self._output_scale != 1.0:
            x = x * self._output_scale

        # ── Post-norm visual injection (concat / replace_padding) ─────────
        if _vis_for_post_norm is not None:
            visual, n_visual, mode, weight = _vis_for_post_norm

            # Project visual tokens through full norm (same as text)
            visual_projected = self.norm(visual)  # [B, N_vis, 1024]

            # Procrustes rotation on visual tokens too (same decomposition)
            if self._use_alignment and self._rotation_matrix is not None:
                R = self._rotation_matrix.to(device=visual_projected.device, dtype=visual_projected.dtype)
                m4b = self._rotation_mean_4b.to(device=visual_projected.device, dtype=visual_projected.dtype)
                m06b = self._rotation_mean_06b.to(device=visual_projected.device, dtype=visual_projected.dtype)
                alpha = self._alignment_strength
                vp_rotated = torch.einsum('ij,...j->...i', R, visual_projected - m4b)
                visual_projected = vp_rotated + (1.0 - alpha) * m4b + alpha * m06b

            if mode == "concat":
                if weight != 1.0:
                    visual_projected = visual_projected * weight
                x = torch.cat([visual_projected, x], dim=1)

            elif mode == "replace_padding":
                B, T, D = x.shape
                tok_norms = x.norm(dim=-1)  # [B, T]
                non_pad = (tok_norms[0] > 1.0).nonzero(as_tuple=True)[0]
                first_pad = (non_pad[-1].item() + 1) if len(non_pad) > 0 else 0
                n_pad_slots = T - first_pad

                if n_pad_slots > 0:
                    if weight != 1.0:
                        visual_projected = visual_projected * weight
                    if n_visual <= n_pad_slots:
                        x[:, first_pad:first_pad + n_visual, :] = visual_projected
                    else:
                        chunk_size = n_visual // n_pad_slots
                        for s in range(n_pad_slots):
                            start = s * chunk_size
                            end = min(start + chunk_size, n_visual) if s < n_pad_slots - 1 else n_visual
                            x[:, first_pad + s, :] = visual_projected[:, start:end, :].mean(dim=1)
                n_visual = 0

            else:
                logger.warning(f"[Qwen3.5-Vision] Unknown vision mode '{mode}', falling back to 'add'")
                style_vec = visual_projected.mean(dim=1, keepdim=True)
                x = x + weight * style_vec
                n_visual = 0

        self._last_n_visual = n_visual

        if intermediate is not None:
            return x, intermediate
        return x, None


# ============================================================================
# Vision Transformer (ViT) for Qwen 3.5 4B
# ============================================================================
# Architecture: 24 blocks, hidden=1024, intermediate=4096, 16 heads, head_dim=64
# PatchEmbed: Conv3d(3, 1024, (2,16,16)) — temporal_patch_size=2, patch_size=16
# PatchMerger: spatial_merge_size=2, output_dim=2560 (matches text backbone hidden)
# Position: learned 48×48 grid + 2D rotary embeddings
# Weights extracted from Qwen/Qwen3.5-4B (visual.* keys)

# CLIP-standard image normalization
VIT_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
VIT_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def _rotate_half(x):
    """Rotates half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Apply 2D rotary position embeddings to Q and K in the ViT."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()  # [..., 1, dim]
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for the ViT (per spatial dimension)."""
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, max_grid_size):
        """Returns frequency table: [max_grid_size, dim//2]"""
        seq = torch.arange(max_grid_size, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)  # [max_grid_size, dim//2]
        return freqs


class ViTAttention(nn.Module):
    """
    Multi-head attention for ViT with fused QKV and 2D RoPE.

    Architecture (from weights):
    - qkv: Linear(1024, 3072, bias=True) — fused Q+K+V
    - proj: Linear(1024, 1024, bias=True) — output projection
    - 16 heads, head_dim=64
    """
    def __init__(self, hidden_size=1024, num_heads=16, device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 64
        self.scale = self.head_dim ** -0.5

        self.qkv = ops.Linear(hidden_size, 3 * hidden_size, bias=True, device=device, dtype=dtype)
        self.proj = ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)

    def forward(self, hidden_states, position_embeddings=None):
        """
        Args:
            hidden_states: [B, N, C] or [N, C] (flattened batch)
            position_embeddings: (cos, sin) tuple for 2D RoPE
        """
        # Handle both batched [B, N, C] and flattened [N, C] inputs
        if hidden_states.dim() == 2:
            seq_len, _ = hidden_states.shape
            is_flat = True
        else:
            is_flat = False
            B, seq_len, _ = hidden_states.shape

        # Fused QKV projection
        qkv = self.qkv(hidden_states)
        if is_flat:
            qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)  # each [N, heads, dim]
        else:
            qkv = qkv.reshape(B, seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # each [B, N, heads, dim]

        # Apply 2D RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Transpose for attention: [..., heads, N, dim]
        if is_flat:
            q = q.transpose(0, 1).unsqueeze(0)  # [1, heads, N, dim]
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
        else:
            q = q.transpose(1, 2)  # [B, heads, N, dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # Scaled dot-product attention (non-causal for ViT)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Reshape back
        if is_flat:
            attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        else:
            attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, -1)

        return self.proj(attn_out)


class ViTMLP(nn.Module):
    """
    MLP for ViT blocks: Linear(1024→4096) + GELU + Linear(4096→1024).
    Note: Uses GELU (not SwiGLU) and HAS bias (unlike the text backbone).
    """
    def __init__(self, hidden_size=1024, intermediate_size=4096,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.linear_fc1 = ops.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(intermediate_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class ViTBlock(nn.Module):
    """
    Single ViT transformer block: LayerNorm + Attention + residual + LayerNorm + MLP + residual.
    Uses standard LayerNorm (not RMSNorm) with bias.
    """
    def __init__(self, hidden_size=1024, intermediate_size=4096, num_heads=16,
                 device=None, dtype=None, ops=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.attn = ViTAttention(hidden_size=hidden_size, num_heads=num_heads,
                                 device=device, dtype=dtype, ops=ops)
        self.mlp = ViTMLP(hidden_size=hidden_size, intermediate_size=intermediate_size,
                          device=device, dtype=dtype, ops=ops)

    def forward(self, hidden_states, position_embeddings=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class ViTPatchMerger(nn.Module):
    """
    Merges 2×2 spatial patch groups into single tokens.

    Input: [N, 1024] where N patches are in block-shuffled order
           (consecutive 4 patches form a 2×2 spatial block)
    Output: [N/4, 2560] where 2560 = text backbone hidden_size

    Architecture (from weights):
    - norm: LayerNorm(1024) — applied per-patch before grouping
    - linear_fc1: Linear(4096, 4096) — 4096 = 1024 * 2 * 2
    - GELU
    - linear_fc2: Linear(4096, 2560)
    """
    MERGE_SIZE = 2  # spatial_merge_size

    def __init__(self, hidden_size=1024, out_hidden_size=2560,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        merged_dim = hidden_size * self.MERGE_SIZE * self.MERGE_SIZE  # 4096
        self.hidden_size = merged_dim
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(merged_dim, merged_dim, bias=True, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.linear_fc2 = ops.Linear(merged_dim, out_hidden_size, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        """x: [N, 1024] in block-shuffled order → [N/4, 2560]"""
        x = self.norm(x)  # per-patch norm: [N, 1024]
        x = x.view(-1, self.hidden_size)  # group 4 patches: [N/4, 4096]
        x = self.linear_fc2(self.act(self.linear_fc1(x)))  # [N/4, 2560]
        return x


class Qwen35ViT(nn.Module):
    """
    Qwen 3.5 Vision Transformer.

    Processes images into visual embeddings (dim=2560) that can be injected
    into the Qwen 3.5 text backbone.

    Architecture:
    - 24 transformer blocks (hidden=1024, intermediate=4096, 16 heads)
    - Conv3d patch embedding (temporal=2, spatial=16)
    - Learned position embedding (48×48 grid with bilinear interpolation)
    - 2D rotary position embeddings
    - PatchMerger (2×2 spatial merge → 2560 output dim)

    Output: [B, num_merged_patches, 2560] (matches text backbone hidden_size)

    For a 448×448 image: 28×28 = 784 patches → 196 merged tokens → [1, 196, 2560]
    """
    NUM_BLOCKS = 24
    HIDDEN_SIZE = 1024
    INTERMEDIATE_SIZE = 4096
    NUM_HEADS = 16
    HEAD_DIM = 64  # 1024 // 16
    PATCH_SIZE = 16
    TEMPORAL_PATCH_SIZE = 2
    SPATIAL_MERGE_SIZE = 2
    OUT_HIDDEN_SIZE = 2560  # matches text backbone
    NUM_GRID_PER_SIDE = 48  # sqrt(2304) for position embedding
    NUM_POSITION_EMBEDDINGS = 2304  # 48 * 48
    ROPE_THETA = 10000.0
    IN_CHANNELS = 3

    # Path to ViT weights
    VIT_WEIGHTS_FILE = os.path.join(NODE_DIR, "qwen35_vit.safetensors")

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations or comfy.ops.disable_weight_init

        # Patch embedding: Conv3d(3, 1024, (2, 16, 16))
        self.patch_embed_proj = ops.Conv3d(
            self.IN_CHANNELS, self.HIDDEN_SIZE,
            kernel_size=(self.TEMPORAL_PATCH_SIZE, self.PATCH_SIZE, self.PATCH_SIZE),
            stride=(self.TEMPORAL_PATCH_SIZE, self.PATCH_SIZE, self.PATCH_SIZE),
            bias=True, device=device, dtype=dtype
        )

        # Learned position embedding: Embedding(2304, 1024)
        self.pos_embed = nn.Embedding(
            self.NUM_POSITION_EMBEDDINGS, self.HIDDEN_SIZE, device=device, dtype=dtype
        )

        # 2D Rotary position embedding
        self.rotary_pos_emb = VisionRotaryEmbedding(self.HEAD_DIM // 2, theta=self.ROPE_THETA)

        # 24 transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                num_heads=self.NUM_HEADS,
                device=device, dtype=dtype, ops=ops
            )
            for _ in range(self.NUM_BLOCKS)
        ])

        # Patch merger (2×2 → single token, outputs 2560)
        self.merger = ViTPatchMerger(
            hidden_size=self.HIDDEN_SIZE,
            out_hidden_size=self.OUT_HIDDEN_SIZE,
            device=device, dtype=dtype, ops=ops
        )

    def _compute_position_embeddings(self, h_patches, w_patches, device, dtype):
        """
        Compute learned + rotary position embeddings for a grid of patches.

        Patches are in block-shuffled order (groups of 4 = 2×2 spatial blocks).

        Args:
            h_patches: number of patch rows
            w_patches: number of patch columns
            device, dtype: target device and dtype

        Returns:
            pos_embeds: [N, hidden_size] learned position embeddings (block-shuffled order)
            rope_cos_sin: (cos, sin) tuple for 2D rotary embeddings
        """
        merge = self.SPATIAL_MERGE_SIZE
        h_blocks = h_patches // merge
        w_blocks = w_patches // merge

        # === Learned position embedding with bilinear interpolation ===
        # Map patch grid positions to the 48×48 learned grid
        h_idxs = torch.linspace(0, self.NUM_GRID_PER_SIDE - 1, h_patches, device=device)
        w_idxs = torch.linspace(0, self.NUM_GRID_PER_SIDE - 1, w_patches, device=device)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clamp(max=self.NUM_GRID_PER_SIDE - 1)
        w_ceil = (w_floor + 1).clamp(max=self.NUM_GRID_PER_SIDE - 1)

        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()

        # Four corner indices for bilinear interpolation
        base_h = h_floor * self.NUM_GRID_PER_SIDE
        base_h_ceil = h_ceil * self.NUM_GRID_PER_SIDE

        idx_00 = (base_h[None].T + w_floor[None]).flatten()         # top-left
        idx_01 = (base_h[None].T + w_ceil[None]).flatten()          # top-right
        idx_10 = (base_h_ceil[None].T + w_floor[None]).flatten()    # bottom-left
        idx_11 = (base_h_ceil[None].T + w_ceil[None]).flatten()     # bottom-right

        w_00 = ((1 - dh)[None].T * (1 - dw)[None]).flatten()
        w_01 = ((1 - dh)[None].T * dw[None]).flatten()
        w_10 = (dh[None].T * (1 - dw)[None]).flatten()
        w_11 = (dh[None].T * dw[None]).flatten()

        # Look up and interpolate
        idx_all = torch.stack([idx_00, idx_01, idx_10, idx_11]).long()
        w_all = torch.stack([w_00, w_01, w_10, w_11]).to(dtype=dtype)

        all_embeds = self.pos_embed(idx_all.to(device))  # [4, h*w, hidden]
        pos_embed = (all_embeds * w_all.unsqueeze(-1)).sum(0)  # [h*w, hidden]

        # Rearrange from row-major to block-shuffled order (for merger compatibility)
        # [h*w, C] → [h_blocks, merge, w_blocks, merge, C] → [h_blocks, w_blocks, merge, merge, C]
        pos_embed = pos_embed.view(h_patches, w_patches, -1)
        pos_embed = pos_embed.view(h_blocks, merge, w_blocks, merge, -1)
        pos_embed = pos_embed.permute(0, 2, 1, 3, 4)  # [h_blocks, w_blocks, merge, merge, C]
        pos_embed = pos_embed.reshape(-1, self.HIDDEN_SIZE)  # [N, C] block-shuffled

        # === 2D Rotary position embeddings ===
        max_hw = max(h_patches, w_patches)
        freq_table = self.rotary_pos_emb(max_hw)  # [max_hw, head_dim//4]

        # Compute (row, col) coordinates in block-shuffled order
        block_rows = torch.arange(h_blocks, device=device)
        block_cols = torch.arange(w_blocks, device=device)
        intra_row = torch.arange(merge, device=device)
        intra_col = torch.arange(merge, device=device)

        row_idx = block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge + intra_col[None, None, None, :]

        row_idx = row_idx.expand(h_blocks, w_blocks, merge, merge).reshape(-1)
        col_idx = col_idx.expand(h_blocks, w_blocks, merge, merge).reshape(-1)

        # Look up rotary frequencies for each spatial dimension
        row_freqs = freq_table[row_idx]  # [N, head_dim//4]
        col_freqs = freq_table[col_idx]  # [N, head_dim//4]
        emb = torch.cat([row_freqs, col_freqs], dim=-1)  # [N, head_dim//2]
        emb = torch.cat([emb, emb], dim=-1)  # [N, head_dim] (duplicate for cos/sin)

        rope_cos = emb.cos().to(dtype=dtype)
        rope_sin = emb.sin().to(dtype=dtype)

        return pos_embed, (rope_cos, rope_sin)

    def _preprocess_image(self, image, target_size=None):
        """
        Preprocess a ComfyUI IMAGE tensor for the ViT.

        Args:
            image: [B, H, W, 3] float tensor in [0, 1] (ComfyUI format)
            target_size: (H, W) to resize to, or None for auto

        Returns:
            patches: [total_patches, 3, 2, 16, 16] ready for Conv3d
            h_patches, w_patches: number of patch rows/columns
        """
        # ComfyUI: [B, H, W, 3] → [B, 3, H, W]
        x = image.permute(0, 3, 1, 2).float()
        B = x.shape[0]

        # Resize to target size if specified, ensuring multiples of patch_size * merge_size = 32
        if target_size is not None:
            th, tw = target_size
        else:
            # Round to nearest multiple of 32
            th = max(32, (x.shape[2] + 15) // 32 * 32)
            tw = max(32, (x.shape[3] + 15) // 32 * 32)

        if x.shape[2] != th or x.shape[3] != tw:
            x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False)
            x = x.clamp(0, 1)

        # Normalize with CLIP mean/std
        mean = torch.tensor(VIT_IMAGE_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(VIT_IMAGE_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Duplicate temporal dimension: [B, 3, H, W] → [B, 3, 2, H, W]
        x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        # Extract patches in block-shuffled order
        h_patches = th // self.PATCH_SIZE
        w_patches = tw // self.PATCH_SIZE
        merge = self.SPATIAL_MERGE_SIZE
        h_blocks = h_patches // merge
        w_blocks = w_patches // merge

        # Unfold into patches: [B, 3, 2, H, W] → [B, 3, 2, h_patches, 16, w_patches, 16]
        x = x.view(B, 3, 2, h_patches, self.PATCH_SIZE, w_patches, self.PATCH_SIZE)

        # Rearrange to block-shuffled order
        # First: [B, 3, 2, h_blocks, merge, 16, w_blocks, merge, 16]
        x = x.view(B, 3, 2, h_blocks, merge, self.PATCH_SIZE, w_blocks, merge, self.PATCH_SIZE)
        # Permute: [B, h_blocks, w_blocks, merge, merge, 3, 2, 16, 16]
        x = x.permute(0, 3, 6, 4, 7, 1, 2, 5, 8)
        # Flatten batch + spatial: [B * h_blocks * w_blocks * merge * merge, 3, 2, 16, 16]
        patches = x.reshape(-1, 3, 2, self.PATCH_SIZE, self.PATCH_SIZE)

        return patches, h_patches, w_patches, B

    def forward(self, image, target_size=None):
        """
        Process an image through the ViT.

        Args:
            image: [B, H, W, 3] ComfyUI IMAGE tensor (float, [0,1])
            target_size: (H, W) tuple, default auto-rounds to mult of 32

        Returns:
            visual_features: [B, num_merged_patches, 2560]
                where num_merged_patches = (H/32) * (W/32)
        """
        # Preprocess: extract patches in block-shuffled order
        patches, h_patches, w_patches, B = self._preprocess_image(image, target_size)

        # Cast patches to model dtype for Conv3d compatibility
        target_dtype = self.patch_embed_proj.weight.dtype
        patches = patches.to(dtype=target_dtype)

        # Patch embedding: Conv3d reduces each patch to a vector
        # patches: [total_patches, 3, 2, 16, 16] → [total_patches, 1024]
        hidden_states = self.patch_embed_proj(patches).view(-1, self.HIDDEN_SIZE)

        # Compute position embeddings
        pos_embeds, rope_cos_sin = self._compute_position_embeddings(
            h_patches, w_patches, hidden_states.device, hidden_states.dtype
        )

        # Total patches per image (in block-shuffled order)
        n_patches = h_patches * w_patches

        # Add learned position embeddings (repeated for batch)
        # hidden_states: [B * n_patches, C], pos_embeds: [n_patches, C]
        pos_embeds = pos_embeds.repeat(B, 1)  # [B * n_patches, C]
        hidden_states = hidden_states + pos_embeds

        # Prepare rotary embeddings (repeated for batch)
        cos, sin = rope_cos_sin
        cos = cos.repeat(B, 1)  # [B * n_patches, head_dim]
        sin = sin.repeat(B, 1)

        # Process through transformer blocks
        # Working with flat [total_patches, C] format
        for block in self.blocks:
            hidden_states = block(hidden_states, position_embeddings=(cos, sin))

        # Merge patches: [B * n_patches, 1024] → [B * n_patches/4, 2560]
        merged = self.merger(hidden_states)

        # Reshape to batch: [B, n_merged, 2560]
        n_merged = n_patches // (self.SPATIAL_MERGE_SIZE ** 2)
        merged = merged.view(B, n_merged, self.OUT_HIDDEN_SIZE)

        return merged

    def load_weights(self, weights_path=None):
        """Load ViT weights from safetensors file."""
        path = weights_path or self.VIT_WEIGHTS_FILE
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ViT weights not found: {path}\n"
                f"Run extract_vit.py to download and extract weights from Qwen/Qwen3.5-4B."
            )

        sd = safetensors_torch.load_file(path)

        # Map weight keys: visual.X → self.X
        # Rename patch_embed.proj → patch_embed_proj
        mapped = {}
        for k, v in sd.items():
            # Strip "visual." prefix
            key = k
            if key.startswith("visual."):
                key = key[len("visual."):]

            # Rename patch_embed.proj → patch_embed_proj
            key = key.replace("patch_embed.proj.", "patch_embed_proj.")

            # Rename attn.qkv → attn.qkv, attn.proj → attn.proj (already correct)
            # Rename mlp.linear_fc1 → mlp.linear_fc1 (already correct)
            # Rename merger.linear_fc1 → merger.linear_fc1 (already correct)
            # Rename merger.norm → merger.norm (already correct)

            mapped[key] = v

        # Load with strict=False to handle any unexpected keys
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if missing:
            logger.warning(f"[Qwen3.5-ViT] Missing keys: {missing}")
        if unexpected:
            logger.warning(f"[Qwen3.5-ViT] Unexpected keys: {unexpected}")

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"[Qwen3.5-ViT] Loaded {param_count:,} parameters from {path}")
        return self


# ============================================================================
# CLIP Wrappers
# ============================================================================

class Qwen35_4BClipModel(comfy.sd1_clip.SDClipModel):
    """SDClipModel wrapper for the Qwen 3.5 4B hybrid model."""
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None,
                 attention_mask=True, model_options={}):
        # Qwen3.5 pad token: <|endoftext|> = 151643 (same as Qwen3)
        # but eos_token_id in Qwen3.5 config is 248044
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config={},
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=Qwen35HybridModel,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options
        )


class AnimaQwen35TEModel(comfy.sd1_clip.SD1ClipModel):
    """
    Text encoder model wrapping Qwen 3.5 4B for Anima.
    Produces Qwen embeddings (1024-dim) + T5 token IDs for the LLM adapter.
    """
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(
            device=device, dtype=dtype,
            name="qwen35_4b",
            clip_model=Qwen35_4BClipModel,
            model_options=model_options
        )

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        # Attach T5 token IDs and weights for the Anima LLM adapter
        if "t5xxl" in token_weight_pairs:
            out[2]["t5xxl_ids"] = torch.tensor(
                list(map(lambda a: a[0], token_weight_pairs["t5xxl"][0])),
                dtype=torch.int
            )
            out[2]["t5xxl_weights"] = torch.tensor(
                list(map(lambda a: a[1], token_weight_pairs["t5xxl"][0]))
            )
        return out


def te_factory(dtype_llama=None, llama_quantization_metadata=None):
    """Factory to create AnimaQwen35TEModel with specific dtype/quantization."""
    class AnimaQwen35TEModel_(AnimaQwen35TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return AnimaQwen35TEModel_


# ============================================================================
# Qwen 3.5 Tokenizer (vocab=248320, NOT the Qwen3 tokenizer with vocab=151936)
# ============================================================================

class Qwen35Tokenizer:
    """
    Tokenizer for the Qwen 3.5 text encoder (vocab_size=248320).

    CRITICAL: nightknocker's 4B encoder IS the official Qwen3.5-4B text backbone.
    The official config (Qwen/Qwen3.5-4B/config.json) confirms:
    - vocab_size: 248320 (NOT 151936 like Qwen3)
    - model_type: qwen3_5_text
    - Same architecture: hidden=2560, 32 layers, linear_attn/full_attn pattern

    Using the wrong tokenizer (Qwen3, vocab=151936) means:
    - Different BPE merge rules produce different token boundaries
    - Every token ID maps to the wrong embedding row
    - 96,384 trained embedding rows (151936-248319) are never accessed
    - The model receives garbled input it was never trained on

    The Qwen3.5 tokenizer files (vocab.json, merges.txt) must be placed in
    the 'qwen35_tokenizer/' subdirectory of this custom node, OR downloaded
    automatically from Qwen/Qwen3.5-4B on HuggingFace.
    """
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = self._find_tokenizer()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=False
        )
        self.embedding_size = 1024  # Output dim of the late norm
        self.embedding_key = 'qwen35_4b'

        # Tokens per chunk (ComfyUI pads/truncates to this)
        self.max_length = 1024

        # Qwen3.5 special tokens
        self.pad_token_id = 151643  # <|endoftext|>
        self.eos_token_id = 248044  # Qwen3.5 eos

        empty = self.tokenizer.encode("")
        self.tokens_to_skip = set(empty)  # BOS/EOS tokens added by tokenizer

    def _find_tokenizer(self):
        """
        Find Qwen3.5 tokenizer files. Priority:
        1. Bundled qwen35_tokenizer/ directory in the custom node folder
        2. ComfyUI's text_encoders directory (if user placed files there)
        3. Fall back to Qwen3 tokenizer with a warning (wrong but functional)
        """
        # Option 1: Bundled with custom node
        if os.path.isdir(QWEN35_TOKENIZER_DIR):
            has_vocab = os.path.exists(os.path.join(QWEN35_TOKENIZER_DIR, "vocab.json"))
            has_tokenizer_json = os.path.exists(os.path.join(QWEN35_TOKENIZER_DIR, "tokenizer.json"))
            if has_vocab or has_tokenizer_json:
                logger.info(f"[Qwen3.5-Anima] Using bundled Qwen3.5 tokenizer from: {QWEN35_TOKENIZER_DIR}")
                return QWEN35_TOKENIZER_DIR

        # Option 2: Try to use HuggingFace model ID directly (auto-download)
        # This will download ~10MB of tokenizer files on first use
        try:
            logger.info("[Qwen3.5-Anima] Qwen3.5 tokenizer not found locally, attempting to load from Qwen/Qwen3.5-4B...")
            return "Qwen/Qwen3.5-4B"
        except Exception:
            pass

        # Option 3: Fall back to Qwen3 tokenizer (WRONG but at least it runs)
        qwen3_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "comfy", "text_encoders", "qwen25_tokenizer"
        )
        qwen3_path = os.path.normpath(qwen3_path)
        if os.path.isdir(qwen3_path):
            logger.warning(
                "[Qwen3.5-Anima] WARNING: Falling back to Qwen3 tokenizer (vocab=151936). "
                "This is WRONG for the Qwen3.5 4B model (vocab=248320). "
                "Image quality will be degraded. "
                "Please download the Qwen3.5 tokenizer files from Qwen/Qwen3.5-4B on HuggingFace "
                "and place them in: " + QWEN35_TOKENIZER_DIR
            )
            return qwen3_path

        raise FileNotFoundError(
            f"Cannot find Qwen3.5 tokenizer. Please download vocab.json, merges.txt, and "
            f"tokenizer.json from https://huggingface.co/Qwen/Qwen3.5-4B and place them in: "
            f"{QWEN35_TOKENIZER_DIR}"
        )

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        """Tokenize text into token IDs with weights."""
        # Encode with the Qwen3.5 tokenizer
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Build token-weight pairs (ComfyUI format)
        # Each chunk is a list of (token_id, weight [, word_id]) tuples
        pairs = []
        for i, tid in enumerate(token_ids):
            if return_word_ids:
                pairs.append((tid, 1.0, i))
            else:
                pairs.append((tid, 1.0))

        # Pad to max_length with pad tokens
        while len(pairs) < self.max_length:
            if return_word_ids:
                pairs.append((self.pad_token_id, 1.0, len(pairs)))
            else:
                pairs.append((self.pad_token_id, 1.0))

        # Truncate if needed
        pairs = pairs[:self.max_length]

        return [pairs]  # List of chunks (single chunk for now)

    def untokenize(self, token_weight_pair):
        """Convert token IDs back to text."""
        ids = [t[0] for t in token_weight_pair if t[0] != self.pad_token_id]
        return self.tokenizer.decode(ids)

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)


class AnimaQwen35Tokenizer:
    """
    Dual tokenizer for Anima + Qwen 3.5:
    - Qwen 3.5 tokenizer (vocab=248320) for the main text encoder
    - T5 tokenizer for the LLM adapter's target IDs
    """
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        # Use the CORRECT Qwen3.5 tokenizer (vocab=248320)
        self.qwen35_4b = Qwen35Tokenizer(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data
        )

        # T5 tokenizer for the LLM adapter (same as standard Anima)
        from comfy.text_encoders.anima import T5XXLTokenizer
        self.t5xxl = T5XXLTokenizer(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data
        )

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = {}
        qwen_ids = self.qwen35_4b.tokenize_with_weights(text, return_word_ids, **kwargs)
        # Set weights to 1.0 (Anima convention)
        out["qwen35_4b"] = [
            [(k[0], 1.0, k[2]) if return_word_ids else (k[0], 1.0) for k in inner_list]
            for inner_list in qwen_ids
        ]
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.t5xxl.untokenize(token_weight_pair)

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return self.qwen35_4b.decode(token_ids, **kwargs)


# ============================================================================
# ComfyUI Custom Node
# ============================================================================

class LoadQwen35AnimaCLIP:
    """
    Loads the Qwen 3.5 4B hybrid text encoder for use with Anima 2B.

    Place your qwen35_4b.safetensors in ComfyUI/models/text_encoders/

    IMPORTANT: This node requires the Qwen3.5 tokenizer (vocab=248320).
    On first use it will attempt to download it from Qwen/Qwen3.5-4B on HuggingFace.
    Alternatively, manually download vocab.json, merges.txt, and tokenizer.json from
    https://huggingface.co/Qwen/Qwen3.5-4B and place them in the 'qwen35_tokenizer/'
    subdirectory next to this file.
    """
    @classmethod
    def INPUT_TYPES(cls):
        te_files = []
        te_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))), "models", "text_encoders")
        if os.path.isdir(te_dir):
            for f in os.listdir(te_dir):
                if f.endswith(".safetensors") and "qwen35" in f.lower():
                    te_files.append(f)
        if not te_files:
            # Fallback: show all safetensors in text_encoders
            if os.path.isdir(te_dir):
                for f in os.listdir(te_dir):
                    if f.endswith(".safetensors"):
                        te_files.append(f)
        if not te_files:
            te_files = ["qwen35_4b.safetensors"]

        # Check if calibration file exists
        cal_file = os.path.join(NODE_DIR, "calibration_params.safetensors")
        cal_available = os.path.exists(cal_file)

        return {
            "required": {
                "clip_name": (sorted(te_files),),
            },
            "optional": {
                "use_calibration": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply per-dimension affine calibration to align 4B output distribution with 0.6B. Requires calibration_params.safetensors (run calibrate.py to generate).",
                }),
                "use_alignment": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Procrustes rotation to align 4B spatial/pose concept directions with 0.6B. Requires rotation_matrix.safetensors (run compute_alignment.py to generate). Helps with 'from side', 'from behind', and other viewpoint tags.",
                }),
                "alignment_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Controls the bias shift strength. The rotation (fixing concept directions like 'from side') is always applied when alignment is on. "
                               "This slider blends the distribution center: 0=keep 4B's own magnitude (L2~10), 1=shift to 0.6B's magnitude (L2~70). "
                               "Try 0.0 first (rotation only) and increase if poses/viewpoints still need help.",
                }),
                "output_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Additional uniform scale factor applied AFTER calibration (if enabled). Usually leave at 1.0 when calibration is on.",
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/Anima"
    TITLE = "Load Qwen3.5 CLIP (Anima)"

    def load_clip(self, clip_name, use_calibration=True, use_alignment=False, alignment_strength=1.0, output_scale=1.0):
        import folder_paths
        import safetensors.torch

        # Find the file
        te_dir = os.path.join(folder_paths.models_dir, "text_encoders")
        clip_path = os.path.join(te_dir, clip_name)

        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Text encoder not found: {clip_path}")

        logger.info(f"[Qwen3.5-Anima] Loading text encoder from: {clip_path}")
        logger.info(f"[Qwen3.5-Anima] use_calibration={use_calibration}, use_alignment={use_alignment}, alignment_strength={alignment_strength}, output_scale={output_scale}")

        # Load state dict
        sd = safetensors.torch.load_file(clip_path)

        # Detect dtype and quantization from checkpoint
        detect = {}
        for norm_key in ["model.norm.weight", "model.layers.0.input_layernorm.weight",
                         "norm.1.weight", "layers.0.input_layernorm.weight"]:
            if norm_key in sd:
                detect["dtype_llama"] = sd[norm_key].dtype
                break

        quant = comfy.utils.detect_layer_quantization(sd, "")
        if quant is not None:
            detect["llama_quantization_metadata"] = quant

        # Build ClipTarget
        te_class = te_factory(**detect)
        clip_target = ClipTarget(AnimaQwen35Tokenizer, te_class)

        # Count parameters for memory estimation
        param_count = sum(
            torch.tensor(v.shape).prod().item() for v in sd.values()
        )

        # Create CLIP object
        clip = comfy.sd.CLIP(
            target=clip_target,
            state_dict=[sd],
            parameters=param_count,
        )

        # Apply settings to the loaded model
        try:
            inner_model = clip.cond_stage_model.qwen35_4b.transformer
            inner_model._output_scale = output_scale
            inner_model._use_calibration = use_calibration
            inner_model._use_alignment = use_alignment
            inner_model._alignment_strength = alignment_strength
            if use_calibration:
                inner_model._load_calibration()
            if use_alignment:
                inner_model._load_alignment()
            logger.info(
                f"[Qwen3.5-Anima] Applied output_scale={output_scale}, "
                f"calibration={'ON' if use_calibration else 'OFF'}, "
                f"alignment={'ON' if use_alignment else 'OFF'}"
            )
        except AttributeError as e:
            logger.warning(f"[Qwen3.5-Anima] Could not set parameters: {e}")

        logger.info(f"[Qwen3.5-Anima] Text encoder loaded successfully ({param_count:,} parameters)")
        return (clip,)


class LoadQwen35ViT:
    """
    Loads the Qwen 3.5 Vision Transformer for image-conditioned generation with Anima 2B.

    The ViT extracts visual features from reference images which are then
    injected into the text encoder's token sequence via the Qwen35VisionEncode node.

    Requires qwen35_vit.safetensors (run extract_vit.py to generate from Qwen/Qwen3.5-4B).
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Look for ViT weights in the custom node directory
        vit_files = []
        if os.path.exists(os.path.join(NODE_DIR, "qwen35_vit.safetensors")):
            vit_files.append("qwen35_vit.safetensors")

        # Also check text_encoders directory
        te_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))), "models", "text_encoders")
        if os.path.isdir(te_dir):
            for f in os.listdir(te_dir):
                if "vit" in f.lower() and f.endswith(".safetensors"):
                    vit_files.append(f)

        if not vit_files:
            vit_files = ["qwen35_vit.safetensors"]

        return {
            "required": {
                "vit_name": (sorted(set(vit_files)),),
            }
        }

    RETURN_TYPES = ("QWEN35_VIT",)
    RETURN_NAMES = ("vit",)
    FUNCTION = "load_vit"
    CATEGORY = "loaders/Anima"
    TITLE = "Load Qwen3.5 ViT (Anima)"

    def load_vit(self, vit_name):
        # Find the weights file
        vit_path = os.path.join(NODE_DIR, vit_name)
        if not os.path.exists(vit_path):
            import folder_paths
            te_dir = os.path.join(folder_paths.models_dir, "text_encoders")
            vit_path = os.path.join(te_dir, vit_name)

        if not os.path.exists(vit_path):
            raise FileNotFoundError(
                f"ViT weights not found: {vit_name}\n"
                f"Run extract_vit.py to download and extract weights from Qwen/Qwen3.5-4B."
            )

        logger.info(f"[Qwen3.5-ViT] Loading ViT from: {vit_path}")

        # Create and load the ViT model
        vit = Qwen35ViT(device="cpu", dtype=torch.bfloat16)
        vit.load_weights(vit_path)
        vit.eval()

        logger.info(f"[Qwen3.5-ViT] ViT loaded successfully")
        return (vit,)


class Qwen35VisionEncode:
    """
    Encode text + image into conditioning for Anima 2B using Qwen 3.5 vision.

    This node:
    1. Processes the reference image through the Qwen 3.5 ViT
    2. Injects visual features as a prefix to text token embeddings
    2. Projects ViT output through the text encoder's norm (2560→1024) — bypassing
       the 32-layer text backbone which was never trained on visual tokens
    3. Injects visual information into the text conditioning
    4. Returns CONDITIONING for cross-attention

    Injection modes:
    - **add** (recommended): Mean-pools visual patches into a single style vector
      in 2560-dim space, magnitude-normalised to match text hidden states, then
      added as a residual BEFORE the norm projection.  The combined text+visual
      signal is projected through the full norm pipeline together, so directional
      differences between images survive normalisation. weight=1.0 means the visual
      perturbation has the same magnitude as the text hidden states.
    - **concat**: Prepends all visual tokens (projected through full norm) before
      text tokens. WARNING: the diffusion model's cross-attention may put 80%+
      attention on visual tokens, drowning out the text prompt.
    - **replace_padding**: Overwrites trailing padding tokens with projected visual
      tokens. Keeps token count unchanged. A middle ground.

    Architecture note: Visual features bypass the text backbone entirely.
    The backbone (24 SSM + 8 attention layers) was trained on text only.
    For 'add' mode, visual injection happens at the 2560-dim level before
    the norm projection (Linear 2560→1024 + ExpRMSNorm + SiLU + Linear).
    For other modes, visual tokens are projected through the full norm.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vit": ("QWEN35_VIT",),
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                    "tooltip": "Text prompt. Visual features from the image will be injected via the selected mode."}),
            },
            "optional": {
                "mode": (["add", "replace_padding", "concat"], {
                    "default": "add",
                    "tooltip": "How to inject visual features:\n"
                               "- add: pool ViT patches into a style vector, add to every text token (best for style/character transfer)\n"
                               "- replace_padding: overwrite padding tokens with visual features (middle ground)\n"
                               "- concat: prepend all visual tokens before text (can drown out text prompt)",
                }),
                "image_size": ("INT", {
                    "default": 448,
                    "min": 64,
                    "max": 768,
                    "step": 32,
                    "tooltip": "Resize image to this size (both H and W, must be multiple of 32). "
                               "448 = 196 visual tokens, 224 = 49 tokens. Larger = more detail but slower.",
                }),
                "vision_weight": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Scale factor for visual influence. "
                               "For 'add' mode (pre-norm): 0.1=subtle, 0.3=moderate, 1.0=strong (visual same magnitude as text). "
                               "For 'concat'/'replace_padding': scales projected token magnitudes.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/Anima"
    TITLE = "Qwen3.5 Vision Encode (Anima)"

    def encode(self, clip, vit, image, text, mode="add", image_size=448, vision_weight=0.3):
        if clip is None:
            raise RuntimeError("CLIP input is None. Connect a Qwen3.5 CLIP loader.")
        if vit is None:
            raise RuntimeError("ViT input is None. Connect a Qwen3.5 ViT loader.")

        # Ensure image_size is multiple of 32 (patch_size * merge_size = 16 * 2)
        image_size = max(64, (image_size // 32) * 32)
        target_size = (image_size, image_size)

        # Step 1: Process image through ViT
        logger.info(f"[Qwen3.5-Vision] Processing image {image.shape} at {image_size}x{image_size}, "
                     f"mode={mode}, weight={vision_weight}")
        device = comfy.model_management.get_torch_device()

        vit.to(device)
        with torch.no_grad():
            visual_embeds = vit(image.to(device), target_size=target_size)
            # visual_embeds: [B, N_img, 2560]
        vit.to("cpu")  # offload ViT after use

        n_visual = visual_embeds.shape[1]
        logger.info(f"[Qwen3.5-Vision] ViT produced {n_visual} visual tokens (dim={visual_embeds.shape[-1]})")

        # Step 2: Store visual embeddings + injection settings
        # The forward() method will:
        #   - Process text through all 32 backbone layers (text-only)
        #   - Project visual features through norm only (2560→1024)
        #   - Inject via selected mode (add/concat/replace_padding)
        try:
            inner_model = clip.cond_stage_model.qwen35_4b.transformer
        except AttributeError:
            raise RuntimeError(
                "Could not access the Qwen 3.5 text encoder. "
                "Make sure the CLIP input is from a 'Load Qwen3.5 CLIP (Anima)' node."
            )

        inner_model._pending_visual_embeds = visual_embeds.to(
            comfy.model_management.intermediate_device()
        )
        inner_model._pending_vision_weight = vision_weight
        inner_model._pending_vision_mode = mode

        try:
            # Step 3: Encode text (forward will handle visual injection)
            tokens = clip.tokenize(text)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        finally:
            inner_model._pending_visual_embeds = None
            inner_model._pending_vision_weight = 1.0
            inner_model._pending_vision_mode = "add"

        # Log output shape
        if conditioning and len(conditioning) > 0:
            cond_shape = conditioning[0][0].shape
            logger.info(f"[Qwen3.5-Vision] Output: {cond_shape} (mode={mode})")

        return (conditioning,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadQwen35AnimaCLIP": LoadQwen35AnimaCLIP,
    "LoadQwen35ViT": LoadQwen35ViT,
    "Qwen35VisionEncode": Qwen35VisionEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwen35AnimaCLIP": "Load Qwen3.5 CLIP (Anima)",
    "LoadQwen35ViT": "Load Qwen3.5 ViT (Anima)",
    "Qwen35VisionEncode": "Qwen3.5 Vision Encode (Anima)",
}

logger.info("[Qwen3.5-Anima] Custom node loaded successfully")
