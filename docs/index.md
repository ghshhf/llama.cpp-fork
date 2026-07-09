---
title: llama.cpp-fork
description: Upstream-first llama.cpp with code quality, docs, and build-system fixes
---

# llama.cpp-fork

> **Upstream-first edition of [llama.cpp](https://github.com/ggml-org/llama.cpp) — rebased nightly onto the latest upstream, with code quality checks, expanded documentation, and a modular model conversion framework.**

[![CI](https://github.com/ghshhf/llama.cpp-fork/actions/workflows/build-cpu.yml/badge.svg)](https://github.com/ghshhf/llama.cpp-fork/actions)
[![clang-tidy](https://github.com/ghshhf/llama.cpp-fork/actions/workflows/clang-tidy.yml/badge.svg)](https://github.com/ghshhf/llama.cpp-fork/actions/workflows/clang-tidy.yml)
[![Python lint](https://github.com/ghshhf/llama.cpp-fork/actions/workflows/python-lint.yml/badge.svg)](https://github.com/ghshhf/llama.cpp-fork/actions/workflows/python-lint.yml)
[![Discussions](https://img.shields.io/github/discussions/ghshhf/llama.cpp-fork)](https://github.com/ghshhf/llama.cpp-fork/discussions)

## What Makes This Fork Different

### 🔄 Auto-Synced Upstream
Every day, CI automatically rebases our custom commits onto the latest upstream code. You get all upstream improvements plus our enhancements.

### 🔧 Code Quality Tooling
- **clang-tidy**: 50+ checks (bugprone, security, performance, misc) with CI enforcement
- **clang-format**: C/C++ formatting rules
- **flake8 + mypy**: Python linting and type checking
- **pre-commit hooks**: Enforced on every push

### 📚 9 Tutorials
From embeddings to agentic coding, KV-cache reuse to WebUI — we have guides that upstream doesn't.

### 🏗️ Modular Conversion Framework
40+ model-specific converter modules replacing the monolithic 12K-line script. Easy to extend, easy to maintain.

### 📦 11 Dockerfiles
CUDA, Vulkan, ROCm, OpenVINO, CANN, MUSA, ZenDNN, s390x, CPU — all with multi-stage builds and BuildKit caching.

### 🔒 Security Hardening
`weights_only=True` in all `torch.load()` calls. Narrowed exception handling. Security policy documented.

## Quick Links

- 📖 [Tutorials](docs/tutorials/) — 9 comprehensive guides
- 🔧 [Conversion Framework](conversion/README.md) — 40+ model modules
- 🤝 [Contributing](CONTRIBUTING.md) — How to contribute
- 📋 [Changelog](CHANGELOG.md) — What changed and when
- ⚖️ [Fork vs Upstream](docs/fork-vs-upstream.md) — Detailed comparison
- 🔒 [Security Policy](SECURITY.md)

## Supported Backends

| Backend | CI Status |
|---|---|
| CPU (x86/ARM/RISC-V) | ✅ |
| CUDA (NVIDIA) | ✅ |
| ROCm (AMD) | ✅ |
| Vulkan | ✅ |
| Apple Metal | ✅ |
| SYCL (Intel) | ✅ |
| OpenVINO (Intel) | ✅ |
| OpenCL | ✅ |
| WebGPU | ✅ |
| CANN (Huawei) | ✅ |
| MUSA (Moore Threads) | ✅ |
| Hexagon (Qualcomm) | ✅ |
| s390x (IBM) | ✅ |

## Stats

- Based on upstream `ggml-org/llama.cpp` commit `cb295bf`
- Auto-sync: daily via CI
- 40+ model conversion modules
- 9 tutorials
- 11 Dockerfiles
- 48+ CI workflows

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ghshhf/llama.cpp-fork&type=Timeline)](https://star-history.com/#ghshhf/llama.cpp-fork&Timeline)

---

*This fork is maintained independently. Core inference logic is identical to upstream — all changes are in code quality, documentation, build system, and security.*
