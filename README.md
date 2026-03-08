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

## Credits

- **Anima 2B**: [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
- **Qwen 3.5 4B for Anima**: [nightknocker/cosmos-qwen3.5](https://huggingface.co/nightknocker/cosmos-qwen3.5)

## License

MIT
