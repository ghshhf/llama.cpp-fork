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

## Model-Specific Notes

### Phi-4 (`phi.py`)

Phi-4 (`microsoft/phi-4`) reuses the `Phi3ForCausalLM` HF architecture and is
converted by `Phi3MiniModel`, the same class used for earlier Phi-3 models.
A few things differ from those earlier models and are handled specially:

- **Tokenizer.** Phi-4 ships a `GPT2Tokenizer` (a tiktoken-style BPE vocab)
  instead of the SentencePiece tokenizer used by prior Phi-3 checkpoints.
  `Phi3MiniModel.set_vocab()` reads `tokenizer_class` out of
  `tokenizer_config.json` and dispatches to `_set_vocab_gpt2()` when it's
  `"GPT2Tokenizer"`, falling back to the legacy `tokenizer.model`
  SentencePiece loader otherwise. See
  [ggml-org/llama.cpp#10814](https://github.com/ggml-org/llama.cpp/issues/10814)
  for the original report.
- **Sliding window.** Phi-4's `config.json` sets `sliding_window: null`
  (full attention, default context 16384). `set_gguf_parameters()` writes
  a sliding window of `0` in that case, which is also how the converter
  distinguishes Phi-4 from earlier sliding-window Phi-3 variants.
- **RoPE.** Phi-4 does not use long/short RoPE factor scaling
  (`rope_scaling: null`, `rope_theta: 250000`), so
  `generate_extra_tensors()` is a no-op for it. The same converter also
  supports long-context Phi-3 models (e.g. `Phi-3-medium-128k-instruct`)
  whose `rope_scaling` provides `long_factor`/`short_factor` arrays
  (`"longrope"`/`"su"` or `"yarn"` types); these are written as
  `rope_factors_long`/`rope_factors_short` tensors.
- **Phi-4 multimodal.** The Phi-4 multimodal vision tower uses a separate
  architecture string, `Phi4ForCausalLMV`, which is registered on *both*
  `Phi3MiniModel` (its text backbone) and `Phi4VisionMmprojModel` (its
  vision projector, exported with `--mmproj`). These don't collide because
  `ModelBase` keys registrations by text vs. mmproj model type.

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
