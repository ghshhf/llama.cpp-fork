# Model Conversion Framework

This directory contains a modular framework for converting models from Hugging Face
format to [GGUF](https://github.com/ggerganov/gguf/blob/master/README.md) format,
compatible with llama.cpp inference.

## Architecture

```
conversion/
├── base.py              # Core framework (138KB) — base classes, shared logic
├── __init__.py          # Package entry point
├── llama.py             # LLaMA / LLaMA 2 / LLaMA 3 / LLaMA 4
├── gemma.py             # Gemma / Gemma 2 / Gemma 3
├── qwen.py              # Qwen / Qwen2 / Qwen3
├── deepseek.py          # DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1
├── phi.py               # Phi-2 / Phi-3 / Phi-4
├── mistral.py           # Mistral / Mixtral
├── ...
└── (40+ model modules)
```

Each model module inherits from `ConversionBase` and implements model-specific
logic for:

- **Tensor mapping** — Hugging Face → GGUF tensor name translation
- **Architecture detection** — Identify model family from config
- **Quantization support** — Which quantization methods apply
- **Special handling** — MoE, multimodal, vision encoders, etc.

## Supported Models

| Model Family | Module | Notes |
|---|---|---|
| LLaMA 1/2/3/4 | `llama.py` | Including Llama 4 Scout/Maverick |
| Gemma 1/2/3 | `gemma.py` | Google Gemma |
| Qwen 1.5/2/2.5/3 | `qwen.py` | Alibaba Qwen |
| DeepSeek V2/V3/R1 | `deepseek.py` | MoE architecture |
| Phi 2/3/4 | `phi.py` | Microsoft Phi |
| Mistral/Mixtral | `mistral.py` | Sparse MoE |
| GPT-OSS | `gpt_oss.py` | MXFP4 native format |
| LLaVA / Pixtral | `llava.py`, `pixtral.py` | Vision-language models |
| Janus / Dream | `januspro.py`, `dream.py` | Multimodal |
| BERT / T5 | `bert.py`, `t5.py` | Encoder models |
| RWKV | `rwkv.py` | Recurrent architecture |
| And 30+ more... | | See module list above |

## Usage

```sh
# Convert a model from Hugging Face to GGUF
python convert_hf_to_gguf.py model_repo_id

# Example: convert Gemma 3
python convert_hf_to_gguf.py google/gemma-3-1b-it

# Example: convert DeepSeek-R1
python convert_hf_to_gguf.py deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

## Extending the Framework

To add support for a new model family:

1. Create a new module in this directory (e.g. `my_model.py`)
2. Inherit from `ConversionBase` in `base.py`
3. Implement required methods: `tensor_mapping()`, `architecture()`, `quantization_supported()`
4. Register the model in `base.py`'s model dispatch table

See `llama.py` or `gemma.py` as reference implementations.

## Key Improvements Over Upstream

| Aspect | Upstream (`convert_hf_to_gguf.py`) | This Framework |
|---|---|---|
| Structure | Single 12,890-line monolith | 40+ focused modules |
| Extensibility | Modify one huge file | Add one module per model |
| Type safety | Minimal | Full `__future__` annotations |
| Error handling | Generic | Model-specific validation |
| Testing | Per-script | Per-module isolation |
