# Fork vs Upstream: What's Different

This document provides a detailed comparison between this fork and upstream [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Summary

| Category | Upstream | This Fork |
|---|---|---|
| Core inference | ✅ | ✅ (identical, auto-synced nightly) |
| Code quality tooling | ❌ | ✅ `.clang-tidy`, `.clang-format`, `pre-commit` |
| CI for code quality | ❌ | ✅ clang-tidy, flake8, mypy, pre-commit |
| Tutorials | 0 | 9 comprehensive guides |
| Model conversion | Monolithic 12K-line script | Modular 40+ model framework |
| Dockerfiles | Minimal | 11 for all major backends |
| Security hardening | Partial | Complete (`weights_only=True` everywhere) |
| Documentation governance | Basic | CODEOWNERS, SECURITY.md, CHANGELOG |
| Auto-sync | N/A | ✅ Daily CI rebase onto upstream |

## Code Quality Improvements

### C++ Static Analysis

| Item | Detail |
|---|---|
| `.clang-tidy` | 50+ checks: bugprone, security, performance, misc |
| `.clang-format` | C/C++ code formatting rules |
| CI enforcement | clang-tidy runs on every PR, checks modified files only |
| Violations fixed | `bugprone-branch-clone`, `performance-no-int-to-ptr`, `misc-include-cleaner`, `misc-use-internal-linkage` |

### Python Code Quality

| Item | Detail |
|---|---|
| `.flake8` | Max line length 125, strict rules |
| `mypy.ini` | Type checking configuration |
| `pyrightconfig.json` | Static type analysis |
| `pyproject.toml` | Strict dependency versions |
| `pre-commit` | Trailing whitespace, EOF, YAML, flake8 |
| CI enforcement | flake8 + mypy run on every PR |

### Security Hardening

| File | Change |
|---|---|
| `tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py` | Added `weights_only=True` |
| `tools/mtmd/legacy-models/glmedge-convert-image-encoder-to-gguf.py` | Added `weights_only=True` |
| `tools/mtmd/legacy-models/llava_surgery.py` | Added `weights_only=True` |
| `tools/mtmd/legacy-models/llava_surgery_v2.py` | Added `weights_only=True` |
| `tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py` | Added `weights_only=True` |
| `tools/tts/convert_pt_to_hf.py` | Added `weights_only=True` |
| `examples/model-conversion/scripts/utils/check-nmse.py` | Narrowed `except:` to specific types |

## Documentation Improvements

### Tutorials (9 new documents)

| Tutorial | Lines | Description |
|---|---|---|
| `add-model.md` | 212 | How to add a new model to llama.cpp |
| `agentic-coding.md` | 247 | Building agentic coding workflows |
| `embeddings.md` | 193 | Using embedding models for semantic search |
| `gpt-oss.md` | 177 | Running gpt-oss with llama.cpp |
| `hf-endpoints.md` | 170 | Hugging Face Inference Endpoints integration |
| `kv-cache-reuse.md` | 187 | KV cache reuse for multi-turn conversations |
| `multi-prefix-slots.md` | 109 | Multi-prefix slot management |
| `ttft-tbt.md` | 141 | Measuring TTFT and TBT performance |
| `webui.md` | 120 | Using the new WebUI of llama.cpp |

### Conversion Framework Documentation

| Item | Detail |
|---|---|
| Architecture docs | `conversion/README.md` with module overview |
| Supported models | 40+ model families documented |
| Extension guide | How to add a new model converter |
| Comparison table | vs upstream monolithic `convert_hf_to_gguf.py` |

## Build System Improvements

### CMake

| Change | Detail |
|---|---|
| Version range | 3.14...3.30 (upstream: 3.28 upper bound) |
| MSVC warnings | Replaced `# todo` with actual `/W4` flags |
| OpenVINO globs | Added `CONFIGURE_DEPENDS` for incremental builds |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | Documented in CONTRIBUTING.md |

### DevOps

| Item | Count | Details |
|---|---|---|
| Dockerfiles | 11 | CUDA, Vulkan, ROCm, OpenVINO, CANN, MUSA, ZenDNN, s390x, CPU |
| Nix configs | 10 | Reproducible builds (apps, devshells, docker, jetson) |
| RPM specs | 2 | Generic and CUDA builds |
| Custom GitHub Actions | 8 | OpenVINO, SpaceMIT, Vulkan, CUDA/ROCm Windows setup |

### CI/CD Workflows

| Workflow | Purpose |
|---|---|
| `sync-upstream.yml` | Daily auto-sync with upstream |
| `clang-tidy.yml` | C++ code quality checks |
| `pre-commit.yml` | Git hook enforcement |
| `python-lint.yml` | flake8 + mypy checks |
| + 35 upstream workflows | Build, test, release for all backends |

## Governance

| Document | Purpose |
|---|---|
| `CODEOWNERS` | 122 lines, per-directory maintainer assignments |
| `SECURITY.md` | Vulnerability reporting, secure usage guidelines |
| `CONTRIBUTING.md` | Fork-specific contribution guidelines |
| `CHANGELOG.md` | Track all fork-specific changes |

## What We Don't Change

This fork **does not modify** core inference logic. All changes are in:

- Build configuration and CI/CD
- Documentation and tutorials
- Python conversion scripts (security + modularity)
- Code quality tooling and enforcement
- DevOps infrastructure (Dockerfiles, Nix, RPM)

Core files like `llama-kv-cache.cpp`, `llama-model.cpp`, `ggml.c` are only touched for **code quality fixes** (NOLINT comments, static keywords, include fixes) — no algorithmic changes.
