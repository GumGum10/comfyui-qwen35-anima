# Qwen 3.5 4B Text Encoder with Anima 2B — Working but Underperforming vs Base 0.6B

## Summary

I built a ComfyUI custom node to support the **Qwen 3.5 4B** hybrid (Mamba2 + Attention) text encoder from [nightknocker/cosmos-qwen3.5](https://huggingface.co/nightknocker/cosmos-qwen3.5/tree/main/4b) with Anima 2B.

The node loads, runs, and generates images — no errors — but the results are consistently **worse** than the base Qwen 3 0.6B text encoder that ships with Anima.

## Screenshots

| Qwen 3.5 4B | Original TE (0.6B) |
|:---:|:---:|
| ![qwen35](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/xYBfmUA29u9tC6P6l2T5V.png) | ![original](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/78UM9HWfOMntNp3j7BYcO.png) |
| ![qwen35](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/wdRW1uMwX3JqcFAF6k2rP.png) | ![original](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/X0N_8E_LDDrGbq0PF3QD_.png) |
| ![qwen35](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/8T6c3iWMlB8GyQpG5QZaW.png) | ![original](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/V78AZowx0_zoCrLYzZvKy.png) |
| ![qwen35](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/yjWAAjIVhU_rVujyoEg6A.png) | ![original](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/f94gFg3MZdjuZ8d7VQD9s.png) |
| ![qwen35](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/8vh8ksxjkPPNohSz8-W7v.png) | ![original](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/JWR0Q5tfLQ3exaKtS5jh-.png) |
| ![qwen35](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/Dz3Yp27isTI9MoBfEX0j5.png) | ![original](https://cdn-uploads.huggingface.co/production/uploads/63820cd163e3fab40c8b41ca/ordABg4GvKRpzZcHf-qB6.png) |

## Details

- The Qwen 3.5 4B model is a hybrid SSM/attention architecture (not a standard transformer), so it required a full custom implementation rather than a simple config swap.
- Weights load correctly (426/426 tensors, 0 missing/0 unexpected), forward pass produces valid 1024-dim embeddings with no NaN/Inf.
- The output norms and value ranges look reasonable (~0.58 L2 norm per token, values in [-0.034, 0.033]).
- Despite all this, generated images are noticeably worse in quality/coherence compared to the stock 0.6B encoder.

## Possible Explanations

- **The Anima LLM adapter may have been trained specifically against Qwen 3 0.6B embeddings.** Even though both encoders output 1024-dim vectors, the embedding space distributions likely differ. The adapter's learned mapping from text embeddings → diffusion conditioning may not generalize to a different encoder without fine-tuning.
- **The SSM recurrence implementation may behave differently at inference** compared to the original training framework (e.g. chunked parallel scan vs. sequential). Subtle numerical differences could accumulate across 24 SSM layers.
- **FP8 quantization artifacts** — the 4B weights are stored in F8_E4M3 which introduces more quantization noise than the 0.6B's native precision.

## Repo

Custom node: [https://github.com/GumGum10/comfyui-qwen35-anima](https://github.com/GumGum10/comfyui-qwen35-anima)

If anyone has ideas on why the larger encoder underperforms, or knows whether the LLM adapter needs retraining for different text encoders, I'd appreciate the input.
