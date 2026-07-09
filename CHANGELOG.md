# Changelog

All notable changes to this fork will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions correspond to the upstream llama.cpp release they are based on.
Fork-specific changes are listed under each version.

## [Unreleased]

### Added
- Auto-sync CI workflow that rebases fork commits onto upstream nightly (`sync-upstream.yml`)
- Clang-tidy code quality check CI workflow
- Conversion framework README with architecture documentation
- Fork feature banner in main README highlighting key improvements
- GitHub Topics: `cuda`, `gguf`, `inference`, `llama-cpp`, `llm`, `vulkan`
- GitHub Discussions enabled for community interaction

### Changed
- PR #1 merged: rebased onto upstream master (262 commits) + restored 9 lost tutorials

## [2026-07-09] - Based on upstream cb295bf

### Added
- 9 tutorial documents: add-model, agentic-coding, embeddings, gpt-oss, hf-endpoints, kv-cache-reuse, multi-prefix-slots, ttft-tbt, webui
- Modular model conversion framework (`conversion/`): 40+ model-specific converter modules
- 11 Dockerfiles for CUDA, Vulkan, ROCm, OpenVINO, CANN, MUSA, ZenDNN, s390x, CPU
- Nix build configurations (`.devops/nix/`)
- Custom GitHub Actions: OpenVINO setup, SpaceMIT setup, Vulkan setup, CUDA/ROCm Windows setup
- CODEOWNERS file with per-directory maintainer assignments
- SECURITY.md with vulnerability reporting policy
- Code quality configuration: `.clang-format`, `.clang-tidy`, `.flake8`, `pre-commit-config.yaml`, `pyproject.toml`, `mypy.ini`, `pyrightconfig.json`, `ty.toml`
- Expanded `.github/labeler.yml` with backend-specific labels
- 5 structured issue templates

### Changed
- CMake minimum version raised to 3.14, maximum to 3.30
- MSVC warning flags: replaced `# todo` placeholder with actual `/W4` flags
- OpenVINO CMake: added `CONFIGURE_DEPENDS` to `file(GLOB_RECURSE)` for incremental builds
- `convert_hf_to_gguf_update.py`: fixed path resolution to use `__file__` instead of CWD
- `convert_hf_to_gguf_update.py`: changed logging from DEBUG to INFO
- `scripts/compare-logprobs.py`, `scripts/gen-unicode-data.py`, `scripts/server-test-model.py`: added shebang and encoding headers

### Security
- Added `weights_only=True` to all `torch.load()` calls in 6 Python files
- Narrowed bare `except:` to specific exception types in conversion scripts

## [2026-06-13] - Initial fork

Forked from `ggml-org/llama.cpp` at commit `4988f6e`.
