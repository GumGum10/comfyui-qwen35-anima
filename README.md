# ComfyUI Qwen 3.5 4B Text Encoder for Anima 2B

A custom ComfyUI node that adds support for the **Qwen 3.5 4B** hybrid (Mamba2 + Attention) text encoder for use with the **Anima 2B** diffusion model.

The base Anima 2B ships with a Qwen 3 0.6B text encoder. This node enables the larger Qwen 3.5 4B variant from [cosmos-qwen3.5](https://huggingface.co/nightknocker/cosmos-qwen3.5/tree/main/4b), which uses a hybrid SSM/attention architecture for improved text understanding.

## Architecture

Qwen 3.5 4B is **not** a standard transformer — it's a hybrid model alternating between Mamba2-style selective state space (SSM) blocks and gated self-attention:

- **32 layers** total: 24 SSM + 8 self-attention (at positions 3, 7, 11, 15, 19, 23, 27, 31)
- **Hidden size**: 2560, **Output dim**: 1024 (matching Anima's expected embedding size)
- **Vocab**: 248,320 tokens
- **FP8 quantized** (F8_E4M3) weights with BF16 norms

## Installation

1. Clone this repo into your ComfyUI `custom_nodes` directory:

   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/GumGum10/comfyui-qwen35-anima.git
   ```

2. Download `qwen35_4b.safetensors` from [nightknocker/cosmos-qwen3.5 (4b)](https://huggingface.co/nightknocker/cosmos-qwen3.5/tree/main/4b) and place it in:

   ```
   ComfyUI/models/text_encoders/qwen35_4b.safetensors
   ```

3. Restart ComfyUI.

## Usage

1. Add the **Load Qwen3.5 CLIP (Anima)** node (found under `loaders/Anima`)
2. Select your `qwen35_4b.safetensors` file
3. Connect the CLIP output to a **CLIPTextEncode** node
4. Use with the Anima 2B diffusion model as usual

## Requirements

- ComfyUI (tested on v0.16.3+)
- An Anima 2B checkpoint (e.g. `animaFp8_preview.safetensors`)
- The Qwen 3.5 4B text encoder weights

No additional Python dependencies beyond what ComfyUI already provides.

## Updates

### v0.3.0 — ExpRMSNorm Late Norm Fix (2026-03-09)

**Critical discovery**: The late normalization layer uses `exp(weight)` parameterization, not standard `weight * norm`. The learned weights are near-zero (~-0.003), which with standard RMSNorm collapsed ALL token information into identical vectors (diversity = 0.003, cross-prompt similarity = 0.999). With `exp(weight)`, near-zero means `exp(0) ≈ 1` — preserving token identity.

| Metric | Before (standard RMSNorm) | After (ExpRMSNorm) |
|--------|--------------------------|---------------------|
| Output std | 0.018 | **0.324** (18x) |
| L2/token | 0.58 | **10.37** (18x) |
| Token diversity | 0.003 | **0.821** (274x) |
| Cross-prompt similarity | 0.999 (collapsed) | **0.689** (distinguishable) |

Evidence: All 64 internal RMSNorm layers have weights centered 0.04–1.11 (normal scaling). ONLY the late norm has weights at -0.003 — a fundamentally different parameterization.

### v0.2.0 — Mamba2 SSM Architecture Rewrite (2026-03-09)

Fixed 5 critical bugs in the SSM block based on the [reference Mamba2 implementation](https://github.com/state-spaces/mamba):

1. **Conv output split**: Was `x(4096) + z(4096)`, now correctly `x(4096) + B(2048) + C(2048)` matching `d_ssm + 2 * ngroups * d_state`
2. **in_proj_z gate**: Was UNUSED (~240M parameters ignored), now gates SSM output as `y * silu(z)` bypassing conv1d
3. **SSM d_state**: Was 1 (scalar B/C), now 64 with proper outer-product state recurrence
4. **Input-dependent dt**: Was static `softplus(dt_bias)`, now `softplus(in_proj_b(h) + dt_bias)` making the SSM truly selective
5. **D skip connection**: Was non-existent, now `y += in_proj_a(h) * x` per reference Mamba2

### v0.1.0 — Initial Release (2026-03-09)

- Complete custom node for Qwen 3.5 4B hybrid (Mamba2 + Attention) text encoder
- Full weight loading (426/426 keys, 4.14B parameters)
- ComfyUI CLIP-compatible with Anima 2B diffusion models

## Known Limitations

- **Tokenizer mismatch**: The model has a 248,320-token vocabulary (151,936 Qwen3 + 96,384 extra tokens in 3 blocks of 32,128 = T5 vocab size). We currently use the standard Qwen3 tokenizer, leaving the extra embedding rows untouched. nightknocker's private tokenizer likely maps differently.
- **Closed training pipeline**: The original author (nightknocker) has stated that reverse-engineered implementations may not perfectly match the private inference code. Weights were trained coupled to his specific code path.

## Credits

- **Anima 2B**: [circlestone-labs](https://huggingface.co/circlestone-labs/Anima)
- **Qwen 3.5 4B for Anima**: [nightknocker/cosmos-qwen3.5](https://huggingface.co/nightknocker/cosmos-qwen3.5)

## License

MIT
