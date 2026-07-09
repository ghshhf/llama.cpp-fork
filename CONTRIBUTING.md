# Contributing to llama.cpp-fork

Thank you for your interest in contributing! This fork focuses on **code quality, documentation, and build-system improvements** on top of the latest upstream [llama.cpp](https://github.com/ggml-org/llama.cpp).

## How This Fork Works

This fork is **auto-synced nightly** with upstream `ggml-org/llama.cpp`. The sync workflow rebases our custom commits on top of the latest upstream code daily.

- **Upstream** (`ggml-org/llama.cpp`): Core inference engine, performance, new backends
- **This fork**: Code quality tooling, documentation, build fixes, security hardening on top of upstream

## Before You Start

1. Check that the change you want to make aligns with the fork's goals (code quality, docs, build system)
2. If it's a core inference change, it belongs in [upstream llama.cpp](https://github.com/ggml-org/llama.cpp) instead
3. Search existing issues and PRs to avoid duplicates

## Development Setup

### Prerequisites

- CMake >= 3.14
- A C++ compiler with C++17 support (GCC 9+, Clang 10+, MSVC 2019+)
- Python 3.10+ with dependencies: `pip install -r requirements.txt`
- `clang-tidy` and `clang-format` for C++ code quality checks
- `pre-commit` for git hooks: `pip install pre-commit && pre-commit install`

### Building

```sh
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build build --parallel

# Or use the Makefile shortcut
make
```

### Generate compile_commands.json

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cp build/compile_commands.json .
```

This file is required for `clang-tidy` analysis.

## Code Quality Standards

This fork enforces strict code quality checks. All contributions must pass:

### C++ Code

- **Formatting**: Follow `.clang-format` rules. Run `clang-format -i file.cpp` before committing.
- **Static Analysis**: Pass `.clang-tidy` checks (bugprone, readability, clang-analyzer, performance, portability, misc).
- **No bare `except:`**: Always catch specific exception types.
- **No raw `torch.load()`**: Always use `torch.load(path, weights_only=True)` in Python scripts.

### Python Code

- **Linting**: Pass `flake8` (max line length 125)
- **Type checking**: Pass `mypy` and `pyright` checks
- **Shebang**: All executable scripts must start with `#!/usr/bin/env python3`
- **Encoding**: Add `# -*- coding: utf-8 -*-` header if non-ASCII characters are used

### Git Hooks

Run `pre-commit run --all-files` before pushing. The CI will also run these checks on every PR.

## Commit Message Format

```
type: short description (50 chars or less)

Longer description if needed. Explain what and why, not how.

Fixes #123
```

Types: `ci`, `docs`, `fix`, `feat`, `refactor`, `perf`, `test`, `chore`

## Pull Request Process

1. Fork the repo and create a feature branch from `master`
2. Make your changes following the code quality standards above
3. Run `pre-commit run --all-files` and fix any issues
4. Ensure CI passes on your PR
5. Submit a PR with a clear description of what changed and why

### PR Requirements

- [ ] Code follows `.clang-format` style
- [ ] `clang-tidy` passes for modified files
- [ ] `flake8` passes for modified Python files
- [ ] `pre-commit` hooks pass
- [ ] Commit messages follow the format above
- [ ] CHANGELOG.md updated if the change is user-facing

## Reporting Issues

Use the [issue templates](/.github/ISSUE_TEMPLATE/) when filing bugs:
- Compilation errors → `010-bug-compilation.yml`
- Runtime/results bugs → `011-bug-results.yml`
- Enhancements → `020-enhancement.yml`
- Research questions → `030-research.yml`

## Security

See [SECURITY.md](SECURITY.md) for our security policy. Do not report vulnerabilities as public issues.

## Code of Conduct

Be respectful and constructive. We're all here to make llama.cpp better for everyone.

## Questions?

Open a [Discussion](https://github.com/ghshhf/llama.cpp-fork/discussions) — we're happy to help!
