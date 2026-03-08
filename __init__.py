"""
ComfyUI Custom Node: Qwen 3.5 4B Text Encoder for Anima 2B

Provides a CLIP-compatible loader for the Qwen 3.5 4B hybrid (Mamba2 + Attention)
text encoder used with the Anima 2B diffusion model (cosmos-qwen3.5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import math

import comfy.sd
import comfy.ops
import comfy.model_management
import comfy.utils
import comfy.sd1_clip
import comfy.text_encoders.hunyuan_video
from comfy.ldm.common_dit import rms_norm
from comfy.supported_models_base import ClipTarget

logger = logging.getLogger(__name__)


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


class BiasAdd(nn.Module):
    """Simple module that adds a learnable bias."""
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x + self.bias


class SSMBlock(nn.Module):
    """
    Mamba2-style Selective State Space Model block.

    Architecture (from weight analysis):
    - in_proj_qkv: Linear(2560, 8192) -> conv1d -> split into x(4096) + z(4096)
    - in_proj_b: Linear(2560, 32) -> B coefficients for SSM
    - in_proj_a: Linear(2560, 32) -> C coefficients for SSM
    - in_proj_z: Linear(2560, 4096) -> external gate
    - A_log: [32] -> state transition diagonal
    - dt_bias: [32] -> discretization timestep bias
    - conv1d: depthwise Conv1d(8192, 8192, kernel=4)
    - norm: RMSNorm(128) -> per-group norm (32 groups * 128 dim = 4096)
    - out_proj: Linear(4096, 2560)
    """
    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32,
                 d_gate=4096, conv_kernel=4, norm_dim=128,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.hidden_size = hidden_size
        self.d_inner = d_inner
        self.n_groups = n_groups
        self.d_gate = d_gate
        self.head_dim = d_gate // n_groups  # 128

        self.in_proj_qkv = ops.Linear(hidden_size, d_inner, bias=False, device=device, dtype=dtype)
        self.in_proj_z = ops.Linear(hidden_size, d_gate, bias=False, device=device, dtype=dtype)
        self.in_proj_a = ops.Linear(hidden_size, n_groups, bias=False, device=device, dtype=dtype)  # C projection
        self.in_proj_b = ops.Linear(hidden_size, n_groups, bias=False, device=device, dtype=dtype)  # B projection

        # Use ops.Conv1d for auto device/dtype casting (matches comfy.ops behavior)
        self.conv1d = ops.Conv1d(
            d_inner, d_inner, conv_kernel, groups=d_inner,
            padding=conv_kernel - 1, bias=False, device=device, dtype=dtype
        )

        self.out_proj = ops.Linear(d_gate, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm = RMSNorm(norm_dim, device=device, dtype=dtype)

        self.A_log = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))

    def _ssm_scan(self, x, B_coeff, C_coeff):
        """
        Simple recurrent SSM scan.
        x: [B, L, n_groups, head_dim]
        B_coeff: [B, L, n_groups]
        C_coeff: [B, L, n_groups]
        Returns: [B, L, n_groups, head_dim]
        """
        batch, seq_len, n_groups, head_dim = x.shape
        device = x.device
        compute_dtype = torch.float32

        # Explicitly move params to input device (handles dynamic VRAM loading)
        A = -torch.exp(self.A_log.to(device=device).float())  # [n_groups]
        dt = F.softplus(self.dt_bias.to(device=device).float())  # [n_groups]

        # Discretize: dA = exp(A * dt)
        dA = torch.exp(dt * A)  # [n_groups]

        h = torch.zeros(batch, n_groups, head_dim, device=device, dtype=compute_dtype)
        outputs = []

        x_f = x.float()
        B_f = B_coeff.float()
        C_f = C_coeff.float()
        dA_expanded = dA.unsqueeze(0).unsqueeze(-1)  # [1, n_groups, 1]

        for t in range(seq_len):
            # h = dA * h + dB * x_t
            h = dA_expanded * h + (dt * B_f[:, t]).unsqueeze(-1) * x_f[:, t]
            # y = C * h
            y_t = C_f[:, t].unsqueeze(-1) * h
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(x.dtype)  # [B, L, n_groups, head_dim]

    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape

        # Input projections
        xz = self.in_proj_qkv(hidden_states)  # [B, L, 8192]
        B_coeff = self.in_proj_b(hidden_states)  # [B, L, 32]
        C_coeff = self.in_proj_a(hidden_states)  # [B, L, 32]

        # Causal 1D convolution (ops.Conv1d handles device/dtype auto-casting)
        xz_conv = xz.transpose(1, 2)  # [B, 8192, L]
        xz_conv = self.conv1d(xz_conv)[..., :seq_len]
        xz_conv = xz_conv.transpose(1, 2)  # [B, L, 8192]

        # Split into SSM input and inner gate
        x, z_inner = xz_conv.chunk(2, dim=-1)  # each [B, L, 4096]
        x = F.silu(x)

        # Reshape for SSM: [B, L, n_groups, head_dim]
        x = x.reshape(batch, seq_len, self.n_groups, self.head_dim)

        # SSM scan
        y = self._ssm_scan(x, B_coeff, C_coeff)  # [B, L, 32, 128]

        # Per-group norm
        y = self.norm(y)

        # Reshape back
        y = y.reshape(batch, seq_len, -1)  # [B, L, 4096]

        # Gate with inner z from conv split
        y = y * F.silu(z_inner)

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

    32 layers with alternating SSM/attention:
    - SSM layers: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30
    - Self-attention layers: 3,7,11,15,19,23,27,31
    - Layer 31: self-attention only (no MLP)
    - Final norm: Linear(2560->1024) + RMSNorm(1024) + SiLU + BiasAdd(1024)
    """
    SELF_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE = 248320
    OUTPUT_DIM = 1024
    HEAD_DIM = 256  # For RoPE
    ROPE_THETA = 1000000.0

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

        # Output projection norm: Linear(2560->1024) + RMSNorm + SiLU + Linear(1024->1024)
        self.norm = nn.Sequential(
            ops.Linear(self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
            RMSNorm(self.OUTPUT_DIM, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
        )

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

        seq_len = x.shape[1]

        # Precompute RoPE frequencies for self-attention layers
        freqs_cis = _precompute_freqs_cis(
            self.HEAD_DIM, seq_len,
            theta=self.ROPE_THETA,
            device=x.device, dtype=x.dtype
        )

        # Build causal attention mask for self-attention layers
        # Use finfo.min/4 like ComfyUI's Llama2_ does (avoids NaN from -inf in softmax)
        attn_mask = None
        if attention_mask is not None:
            # Convert padding mask to causal attention mask
            # attention_mask: [B, L] with 0 for padding
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
            # No padding mask, but still need causal mask for seq_len > 1
            mask_fill = torch.finfo(x.dtype).min / 4
            attn_mask = torch.empty(
                seq_len, seq_len, dtype=x.dtype, device=x.device
            ).fill_(mask_fill).triu_(1)

        # Process through layers
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

        # Apply output projection norm
        x = self.norm(x)

        if intermediate is not None:
            return x, intermediate
        return x, None


# ============================================================================
# CLIP Wrappers
# ============================================================================

class Qwen35_4BClipModel(comfy.sd1_clip.SDClipModel):
    """SDClipModel wrapper for the Qwen 3.5 4B hybrid model."""
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None,
                 attention_mask=True, model_options={}):
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


class AnimaQwen35Tokenizer:
    """
    Dual tokenizer for Anima + Qwen 3.5:
    - Qwen tokenizer for the main text encoder
    - T5 tokenizer for the LLM adapter's target IDs
    """
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        # Reuse existing Qwen3 tokenizer (compatible vocab)
        from comfy.text_encoders.anima import Qwen3Tokenizer, T5XXLTokenizer
        self.qwen35_4b = Qwen3Tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        # Override the embedding_size to match our model's output dim
        self.qwen35_4b.embedding_size = 1024
        self.qwen35_4b.embedding_key = 'qwen35_4b'
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

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

        return {
            "required": {
                "clip_name": (sorted(te_files),),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/Anima"
    TITLE = "Load Qwen3.5 CLIP (Anima)"

    def load_clip(self, clip_name):
        import folder_paths
        import safetensors.torch

        # Find the file
        te_dir = os.path.join(folder_paths.models_dir, "text_encoders")
        clip_path = os.path.join(te_dir, clip_name)

        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Text encoder not found: {clip_path}")

        logger.info(f"[Qwen3.5-Anima] Loading text encoder from: {clip_path}")

        # Load state dict
        sd = safetensors.torch.load_file(clip_path)

        # Detect dtype and quantization from checkpoint
        detect = {}
        # Try standard prefix first, then bare keys
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

        logger.info(f"[Qwen3.5-Anima] Text encoder loaded successfully ({param_count:,} parameters)")
        return (clip,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadQwen35AnimaCLIP": LoadQwen35AnimaCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwen35AnimaCLIP": "Load Qwen3.5 CLIP (Anima)",
}

logger.info("[Qwen3.5-Anima] Custom node loaded successfully")
