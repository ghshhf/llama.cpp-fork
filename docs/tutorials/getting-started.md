# Getting Started with llama.cpp

This guide walks you through installing llama.cpp, downloading a model, and running your first chat session — no prior experience required.

## What is llama.cpp?

llama.cpp is a lightweight, dependency-free C/C++ program for running large language models locally on your own hardware (CPU, Apple Silicon, or GPU). It works with models in the **GGUF** file format.

## 1. Installing llama.cpp

Pick whichever method fits your platform. You don't need to build from source unless you want a custom build (e.g. specific GPU backend flags).

### macOS / Linux (Homebrew)

```bash
brew install llama.cpp
```

### Windows (winget)

```powershell
winget install llama.cpp
```

### Docker

No install needed — just pull and run an image. Useful if you don't want anything on your host system.

```bash
# CPU only
docker pull ghcr.io/ggml-org/llama.cpp:server

# NVIDIA GPU (requires the NVIDIA Container Toolkit)
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
```

You'll use these images later when running the server (see [Running your first inference](#3-running-your-first-inference)).

### Build from source

Best if you want the latest features, a specific GPU backend, or a build tuned for your CPU.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# CPU-only build
cmake -B build
cmake --build build --config Release

# Or, for NVIDIA GPU support:
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

The compiled binaries (`llama-cli`, `llama-server`, etc.) will be in `build/bin/`.

### Verify your install

```bash
llama-cli --version
llama-server --version
```

If these print a version number, you're ready to go.

## 2. Downloading a model

llama.cpp uses models in **GGUF** format. The easiest way to get one is to let llama.cpp download it directly from Hugging Face using the `-hf` flag — no manual downloading required.

```bash
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

This downloads the model (if not already cached) and starts an interactive session in one step. The format is:

```
-hf <user>/<repo>[:quant]
```

For example, to pick a specific quantization level:

```bash
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF:Q4_K_M
```

If you omit the quant, llama.cpp picks a sensible default (typically `Q4_K_M` if available).

**Notes:**
- Downloaded models are cached in your standard Hugging Face cache directory, so other tools that use the HF cache can reuse them.
- If you already have a `.gguf` file downloaded manually, you can point directly to it instead:

```bash
llama-cli -m /path/to/model.gguf
```

**Choosing a model size:** Start small. A 1B–3B parameter model (like `gemma-3-1b-it-GGUF`) will run comfortably on almost any laptop and is a great way to confirm your setup works before downloading something larger.

## 3. Running your first inference

### Option A: Interactive CLI chat

The simplest way to talk to a model:

```bash
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

This launches an interactive, conversational session in your terminal. Type a message and press enter; type `/bye` or press `Ctrl+C` to exit.

If you only want a single one-off completion rather than a chat session, use `-p` with a prompt and add `-no-cnv` to disable conversation mode:

```bash
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -no-cnv -p "Explain recursion in one sentence."
```

### Option B: Local HTTP server

If you want an OpenAI-compatible API you can call from code or other tools:

```bash
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

By default this starts a server at `http://127.0.0.1:8080`, which also serves a built-in web chat UI at that same URL — open it in your browser to chat right away.

Test it from the command line with curl:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is an LLM?"}
    ]
  }'
```

### Option C: Docker

```bash
docker run -it --rm \
  -v ~/models:/models \
  ghcr.io/ggml-org/llama.cpp:light \
  -m /models/your-model.gguf \
  -p "What is an LLM?"
```

Or run the server in Docker:

```bash
docker run -d --name llama-server \
  -p 8080:8080 \
  -v ~/models:/models \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/your-model.gguf --host 0.0.0.0
```

## 4. Basic chat formatting and system prompts

Most instruction-tuned models (anything with `-it`, `-instruct`, or `-chat` in the name) expect a **chat template** — a specific structure that wraps your messages with role markers. llama.cpp detects and applies this automatically for GGUF models that embed a chat template, so in most cases you don't need to do anything extra.

### Setting a system prompt

A system prompt lets you steer the model's behavior or persona for the whole conversation. With the CLI:

```bash
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF \
  --system-prompt "You are a concise assistant that answers in bullet points."
```

With the server, pass it as part of the `messages` array in your API request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a concise assistant that answers in bullet points."},
      {"role": "user", "content": "What is an LLM?"}
    ]
  }'
```

### Troubleshooting odd or unstructured output

If responses look garbled, ignore instructions, or never stop generating:

- Confirm the model is an **instruct/chat-tuned** variant, not a base completion model.
- Make sure you're running in conversational mode (the default for `llama-cli`, unless you passed `-no-cnv`).
- Try the server's built-in web UI first — it applies chat formatting automatically and is a good way to sanity-check the model before wiring it into your own code.

## 5. Next steps

Now that you have basic inference working, here's where to go next:

- **Embeddings** — llama.cpp can also generate text embeddings for search/RAG use cases via `llama-embedding` or the server's `/v1/embeddings` endpoint.
- **Server deep dive** — explore additional `llama-server` flags for context size (`--ctx-size`), GPU layer offloading (`--n-gpu-layers`/`-ngl`), continuous batching (`--cont-batching`), and multiple concurrent requests (`--parallel`).
- **Multimodal models** — some GGUF models support image or audio input via the server's multimodal support.
- **Quantization** — learn how different quantization levels (Q4_K_M, Q5_K_M, Q8_0, etc.) trade off model size, speed, and quality, so you can pick the right one for your hardware.
- **GPU backends** — if you have a GPU, look into building with `-DGGML_CUDA=ON` (NVIDIA), `-DGGML_HIP=ON` (AMD ROCm), or Vulkan/SYCL support for a significant speed boost over CPU.
- **Editor integrations** — try [llama.vscode](https://github.com/ggml-org/llama.vscode) or [llama.vim](https://github.com/ggml-org/llama.vim) for local AI code completion.

## Getting help

- Full documentation and flag reference: the project's [GitHub repository](https://github.com/ggml-org/llama.cpp)
- Build guide for other hardware (ROCm, SYCL, RISC-V, etc.): `docs/build.md` in the repository
- Docker image reference: `docs/docker.md` in the repository