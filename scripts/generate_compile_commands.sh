#!/usr/bin/env bash
# Generate compile_commands.json for clang-tidy analysis
# Usage: bash scripts/generate_compile_commands.sh

set -e

BUILD_DIR="${1:-build}"

echo "Configuring CMake with compile_commands.json export..."
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGGML_BACKEND_DL=ON

echo "Copying compile_commands.json to project root..."
cp "$BUILD_DIR/compile_commands.json" .

echo "Done! compile_commands.json is ready for clang-tidy analysis."
echo "Run: clang-tidy --config-file=.clang-tidy -p . <source_file.cpp>"
