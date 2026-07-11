#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llama-model-advisor - Intelligent model & quantization recommendation tool.

Analyzes your hardware and recommends the best model + quantization
for running llama.cpp locally.

Usage:
    python scripts/llama-model-advisor.py
    python scripts/llama-model-advisor.py --model qwen
    python scripts/llama-model-advisor.py --ram 16 --gpu none
    python scripts/llama-model-advisor.py --export
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def detect_ram_gb():
    """Detect available RAM in GB."""
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // (1024 * 1024)
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            return int(out) // (1024**3)
        elif platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True,
            ).strip()
            return int(out) // (1024**3)
    except Exception:
        pass
    return 8


def detect_cpu_cores():
    """Detect logical CPU cores."""
    return os.cpu_count() or 4


def detect_gpu():
    """
    Detect GPU type: 'cuda', 'rocm', 'metal', 'vulkan', 'opencl', 'none'.
    """
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True)
            return "cuda"
        except Exception:
            pass

    if shutil.which("rocm-smi"):
        try:
            subprocess.run(["rocm-smi"], capture_output=True, check=True)
            return "rocm"
        except Exception:
            pass

    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            if "Metal" in out or "Apple" in out:
                return "metal"
        except Exception:
            pass

    if shutil.which("vulkaninfo"):
        try:
            subprocess.run(
                ["vulkaninfo", "--summary"], capture_output=True, check=True
            )
            return "vulkan"
        except Exception:
            pass

    return "none"


def get_hw_profile():
    """Return full hardware profile."""
    return {
        "cpu_cores": detect_cpu_cores(),
        "ram_gb": detect_ram_gb(),
        "gpu": detect_gpu(),
        "platform": platform.system(),
    }


QUANT_ORDER = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]


def recommend_quant(ram_gb, model, gpu):
    """Pick the best quantization level based on available RAM."""
    for q_name in reversed(QUANT_ORDER):
        if q_name == "F16":
            needed = model["ram_f16"]
        elif q_name in ("Q8_0",):
            needed = model["ram_q8"]
        else:
            needed = model["ram_q4"]
        if ram_gb >= needed:
            return q_name
    return "Q2_K"


MODEL_DB = [
    {"id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "name": "Qwen2.5-1.5B", "params_b": 1.5,
     "ram_q4": 1.5, "ram_q8": 2.5, "ram_f16": 4.5, "gpu": "any",
     "tags": ["qwen", "chinese", "fast", "cpu-friendly"],
     "desc": "通义千问 1.5B，适合低配设备，中文能力强"},
    {"id": "Qwen/Qwen2.5-3B-Instruct-GGUF", "name": "Qwen2.5-3B", "params_b": 3,
     "ram_q4": 2.5, "ram_q8": 4.5, "ram_f16": 7.5, "gpu": "any",
     "tags": ["qwen", "chinese", "fast", "cpu-friendly"],
     "desc": "通义千问 3B，性价比极高，中文表现好"},
    {"id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "name": "Qwen2.5-7B", "params_b": 7,
     "ram_q4": 5.5, "ram_q8": 9.5, "ram_f16": 15, "gpu": "cuda,rocm,vulkan",
     "tags": ["qwen", "chinese", "balanced"],
     "desc": "通义千问 7B，平衡之选，中文能力优秀"},
    {"id": "Qwen/Qwen2.5-14B-Instruct-GGUF", "name": "Qwen2.5-14B", "params_b": 14,
     "ram_q4": 9, "ram_q8": 16, "ram_f16": 26, "gpu": "cuda,rocm,vulkan",
     "tags": ["qwen", "chinese", "high-quality"],
     "desc": "通义千问 14B，高质量中文推理，需 GPU 或大内存"},
    {"id": "Qwen/Qwen2.5-32B-Instruct-GGUF", "name": "Qwen2.5-32B", "params_b": 32,
     "ram_q4": 18, "ram_q8": 32, "ram_f16": 52, "gpu": "cuda,rocm",
     "tags": ["qwen", "chinese", "high-quality"],
     "desc": "通义千问 32B，旗舰级中文能力，需大显存/内存"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "name": "DeepSeek-R1-1.5B", "params_b": 1.5,
     "ram_q4": 1.5, "ram_q8": 2.5, "ram_f16": 4.5, "gpu": "any",
     "tags": ["deepseek", "chinese", "reasoning", "fast"],
     "desc": "DeepSeek R1 蒸馏 1.5B，推理能力强，适合低配"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF", "name": "DeepSeek-R1-7B", "params_b": 7,
     "ram_q4": 5.5, "ram_q8": 9.5, "ram_f16": 15, "gpu": "cuda,rocm,vulkan",
     "tags": ["deepseek", "chinese", "reasoning"],
     "desc": "DeepSeek R1 蒸馏 7B，推理能力强，需适量内存"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF", "name": "DeepSeek-R1-14B", "params_b": 14,
     "ram_q4": 9, "ram_q8": 16, "ram_f16": 26, "gpu": "cuda,rocm",
     "tags": ["deepseek", "chinese", "reasoning", "high-quality"],
     "desc": "DeepSeek R1 蒸馏 14B，旗舰推理，需大内存"},
    {"id": "THUDM/glm-4-9b-chat-GGUF", "name": "GLM-4-9B", "params_b": 9,
     "ram_q4": 6.5, "ram_q8": 11, "ram_f16": 18, "gpu": "cuda,rocm,vulkan",
     "tags": ["glm", "chinese", "balanced"],
     "desc": "智谱 GLM-4-9B，多语言能力强"},
    {"id": "THUDM/chatglm3-6b-GGUF", "name": "ChatGLM3-6B", "params_b": 6,
     "ram_q4": 4.5, "ram_q8": 7.5, "ram_f16": 12, "gpu": "any",
     "tags": ["glm", "chinese", "balanced"],
     "desc": "ChatGLM3-6B，轻量中文对话模型"},
    {"id": "01-ai/Yi-1.5-9B-Chat-GGUF", "name": "Yi-1.5-9B", "params_b": 9,
     "ram_q4": 6.5, "ram_q8": 11, "ram_f16": 18, "gpu": "cuda,rocm,vulkan",
     "tags": ["yi", "chinese", "balanced"],
     "desc": "Yi-1.5-9B，中文能力强"},
    {"id": "internlm/internlm2_5-7b-chat-GGUF", "name": "InternLM2.5-7B", "params_b": 7,
     "ram_q4": 5.5, "ram_q8": 9.5, "ram_f16": 15, "gpu": "cuda,rocm,vulkan",
     "tags": ["internlm", "chinese", "balanced"],
     "desc": "书生 InternLM2.5-7B"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct-GGUF", "name": "Llama-3.1-8B", "params_b": 8,
     "ram_q4": 5.5, "ram_q8": 9.5, "ram_f16": 16, "gpu": "cuda,rocm,vulkan",
     "tags": ["llama", "english", "balanced", "meta"],
     "desc": "Meta Llama 3.1 8B，国际主流模型"},
    {"id": "meta-llama/Llama-3.1-70B-Instruct-GGUF", "name": "Llama-3.1-70B", "params_b": 70,
     "ram_q4": 35, "ram_q8": 70, "ram_f16": 140, "gpu": "cuda,rocm",
     "tags": ["llama", "english", "high-quality", "meta"],
     "desc": "Meta Llama 3.1 70B，需大显存/内存"},
    {"id": "mistralai/Mistral-Nemo-12B-Instruct-GGUF", "name": "Mistral-Nemo-12B", "params_b": 12,
     "ram_q4": 7.5, "ram_q8": 13, "ram_f16": 24, "gpu": "cuda,rocm,vulkan",
     "tags": ["mistral", "english", "balanced"],
     "desc": "Mistral Nemo 12B，128k 上下文"},
]


def recommend(hw, tag_filter=None):
    """Return sorted list of (model, quant, est_tps) or (model, None, reason)."""
    results = []
    ram = hw["ram_gb"]
    gpu = hw["gpu"]

    for m in MODEL_DB:
        if tag_filter and tag_filter not in m["tags"]:
            continue

        if ram < m["ram_q4"]:
            results.append((m, None,
                f"Not enough RAM (need >= {m['ram_q4']:.1f} GB @ Q4)"))
            continue

        quant = recommend_quant(ram, m, gpu)
        cores = hw["cpu_cores"]
        params = m["params_b"]
        if gpu in ("cuda", "rocm"):
            est_tps = max(1, int(params * 50 / max(1, cores / 4)))
        else:
            est_tps = max(1, int(params * 10 / max(1, cores / 4)))

        results.append((m, quant, est_tps))

    results.sort(key=lambda x: (x[1] is None, x[0]["params_b"]))
    return results


def format_hw(hw):
    lines = [
        f"  C
