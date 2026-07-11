#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gguf-info - Enhanced GGUF model file inspector

Provides deep analysis of GGUF model files including:
- Model metadata (architecture, name, context length, etc.)
- Tensor statistics by quantization type
- Parameter count estimation
- Layer breakdown
- File statistics
- Multiple output formats (text table, JSON)

Usage:
    python scripts/gguf-info.py model.gguf
    python scripts/gguf-info.py model.gguf --json
    python scripts/gguf-info.py model.gguf --tensors
    python scripts/gguf-info.py model.gguf --layers
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Allow running from scripts/ directory
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402


# ──────────────────────────────────────────────
# Quantization type display names
# ──────────────────────────────────────────────

QUANT_NAMES = {
    GGMLQuantizationType.F16:     "F16",
    GGMLQuantizationType.Q4_0:    "Q4_0",
    GGMLQuantizationType.Q4_1:    "Q4_1",
    GGMLQuantizationType.Q5_0:    "Q5_0",
    GGMLQuantizationType.Q5_1:    "Q5_1",
    GGMLQuantizationType.Q8_0:    "Q8_0",
    GGMLQuantizationType.Q2_K:    "Q2_K",
    GGMLQuantizationType.Q3_K:    "Q3_K",
    GGMLQuantizationType.Q4_K:    "Q4_K",
    GGMLQuantizationType.Q5_K:    "Q5_K",
    GGMLQuantizationType.Q6_K:    "Q6_K",
    GGMLQuantizationType.Q8_K:    "Q8_K",
    GGMLQuantizationType.IQ2_XXS: "IQ2_XXS",
    GGMLQuantizationType.IQ2_XS:  "IQ2_XS",
    GGMLQuantizationType.IQ3_XXS: "IQ3_XXS",
    GGMLQuantizationType.IQ1_S:   "IQ1_S",
    GGMLQuantizationType.IQ3_S:   "IQ3_S",
    GGMLQuantizationType.IQ2_S:   "IQ2_S",
    GGMLQuantizationType.IQ4_XS:  "IQ4_XS",
    GGMLQuantizationType.IQ1_M:   "IQ1_M",
    GGMLQuantizationType.BF16:    "BF16",
}

QUANT_DESCRIPTIONS = {
    "F16":     "16-bit float (full precision)",
    "Q4_0":    "4-bit uniform (no superblock)",
    "Q4_1":    "4-bit uniform (with offset)",
    "Q5_0":    "5-bit uniform (no superblock)",
    "Q5_1":    "5-bit uniform (with offset)",
    "Q8_0":    "8-bit uniform (no superblock)",
    "Q2_K":    "2-bit K-quant (superblock)",
    "Q3_K":    "3-bit K-quant (superblock)",
    "Q4_K":    "4-bit K-quant (superblock)",
    "Q5_K":    "5-bit K-quant (superblock)",
    "Q6_K":    "6-bit K-quant (superblock)",
    "Q8_K":    "8-bit K-quant (superblock)",
    "IQ2_XXS": "2-bit improved (extra extra small)",
    "IQ2_XS":  "2-bit improved (extra small)",
    "IQ3_XXS": "3-bit improved (extra extra small)",
    "IQ1_S":   "1-bit improved (small)",
    "IQ3_S":   "3-bit improved (small)",
    "IQ2_S":   "2-bit improved (small)",
    "IQ4_XS":  "4-bit improved (extra small)",
    "IQ1_M":   "1-bit improved (medium)",
    "BF16":    "Brain float 16-bit",
}


# ──────────────────────────────────────────────
# Metadata key -> human-readable label mapping
# ──────────────────────────────────────────────

METADATA_LABELS = {
    "general.architecture":       "Architecture",
    "general.name":               "Model Name",
    "general.author":             "Author",
    "general.description":        "Description",
    "general.url":                "URL",
    "general.license":            "License",
    "general.source_url":         "Source URL",
    "general.file_type":          "File Type",
    "general.quantization_version": "Quant Version",
    "general.alignment":          "Alignment",
    "llm.vocab_size":             "Vocab Size",
    "llm.context_length":         "Context Length",
    "llm.embedding_length":       "Embedding Dim",
    "llm.block_count":            "Layers",
    "llm.feed_forward_length":    "FFN Dim",
    "llm.use_parallel_residual":  "Parallel Residual",
    "llm.tensor_data_layout":     "Tensor Layout",
    "llm.rope.dimension_count":   "RoPE Dim Count",
    "llm.rope.freq_base":         "RoPE Freq Base",
    "llm.rope.scaling_type":      "RoPE Scaling",
    "llm.rope.scaling_factor":    "RoPE Factor",
    "llm.rope.scaling_orig_ctx_len": "RoPE Orig Ctx",
    "llm.rope.scaling_finetuned": "RoPE Finetuned",
    "llm.attention.head_count":           "Attention Heads",
    "llm.attention.head_count_kv":       "KV Heads",
    "llm.attention.max_alibi_bias":      "Max ALiBi Bias",
    "llm.attention.clamp_kqv":           "Clamp KQV",
    "llm.attention.layer_norm_epsilon":  "LayerNorm Eps",
    "llm.attention.layer_norm_rms_epsilon": "RMSNorm Eps",
    "llm.ssm.conv_kernel":      "SSM Conv Kernel",
    "llm.ssm.inner_size":       "SSM Inner Size",
    "llm.ssm.state_size":       "SSM State Size",
    "llm.ssm.time_step_rank":   "SSM Time Step Rank",
    "llm.ssm.group_count":      "SSM Group Count",
    "tokenizer.ggml.model":     "Tokenizer Model",
    "tokenizer.ggml.pre":       "Tokenizer Pre",
    "tokenizer.ggml.bos_id":    "BOS Token ID",
    "tokenizer.ggml.eos_id":    "EOS Token ID",
    "tokenizer.ggml.unk_id":    "UNK Token ID",
    "tokenizer.ggml.sep_id":    "SEP Token ID",
    "tokenizer.ggml.pad_id":    "PAD Token ID",
    "tokenizer.ggml.mask_id":   "MASK Token ID",
    "split.no":                 "Split No",
    "split.count":              "Split Count",
    "split.tensors.count":      "Split Tensors",
}


# ──────────────────────────────────────────────
# Core analysis functions
# ──────────────────────────────────────────────

def analyze_metadata(reader):
    """Extract and organize model metadata."""
    info = {}
    for key, field in reader.fields.items():
        label = METADATA_LABELS.get(key, key)
        try:
            value = field.contents()
            if isinstance(value, list) and len(value) > 10:
                value = "[" + str(len(value)) + " items]"
            info[label] = value
        except Exception:
            info[label] = "<unreadable>"
    return info


def analyze_tensors(reader, file_path):
    """Analyze tensor statistics."""
    stats = {
        "total_tensors": len(reader.tensors),
        "total_elements": 0,
        "tensor_bytes": 0,
        "by_type": defaultdict(lambda: {"count": 0, "elements": 0, "bytes": 0}),
        "by_prefix": defaultdict(lambda: {"count": 0, "elements": 0, "bytes": 0}),
        "layers": {},
        "largest_tensors": [],
    }

    for tensor in reader.tensors:
        qtype = tensor.tensor_type
        qname = QUANT_NAMES.get(qtype, "UNKNOWN(" + str(qtype.value) + ")")
        elements = tensor.n_elements
        n_bytes = tensor.n_bytes

        stats["total_elements"] += elements
        stats["tensor_bytes"] += n_bytes

        # By quantization type
        qt = stats["by_type"][qname]
        qt["count"] += 1
        qt["elements"] += elements
        qt["bytes"] += n_bytes

        # By tensor prefix (layer grouping)
        name = tensor.name
        prefix = name.split(".")[0] if "." in name else name
        pt = stats["by_prefix"][prefix]
        pt["count"] += 1
        pt["elements"] += elements
        pt["bytes"] += n_bytes

        # Layer extraction (e.g. "blk.12.attn_q.weight" -> layer 12)
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part.startswith("blk") or part.startswith("layer"):
                try:
                    layer_num = int(parts[i + 1]) if i + 1 < len(parts) else -1
                    layer_key = part + "." + str(layer_num)
                    if layer_key not in stats["layers"]:
                        stats["layers"][layer_key] = {
                            "count": 0, "elements": 0, "bytes": 0, "tensors": []
                        }
                    lk = stats["layers"][layer_key]
                    lk["count"] += 1
                    lk["elements"] += elements
                    lk["bytes"] += n_bytes
                    lk["tensors"].append(name)
                except (ValueError, IndexError):
                    pass
                break

        # Track largest tensors
        stats["largest_tensors"].append((name, qname, elements, n_bytes))

    # Sort largest tensors by bytes
    stats["largest_tensors"].sort(key=lambda x: x[3], reverse=True)
    stats["largest_tensors"] = stats["largest_tensors"][:10]

    # Compute file size
    stats["file_size"] = os.path.getsize(file_path)
    stats["header_size"] = reader.data_offset
    stats["data_percentage"] = (stats["tensor_bytes"] / stats["file_size"] * 100) if stats["file_size"] > 0 else 0

    return stats


def estimate_params(stats):
    """Estimate model parameters from tensor analysis."""
    total_elements = stats["total_elements"]
    params_b = total_elements / 1e9

    return {
        "total_elements": total_elements,
        "estimated_params_b": round(params_b, 2),
        "estimated_params": "~" + str(round(params_b, 1)) + "B",
    }


# ──────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────

def format_bytes(n):
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return str(round(n, 1)) + " " + unit
        n /= 1024.0
    return str(round(n, 1)) + " PB"


def format_number(n):
    """Format number with commas."""
    return "{:,}".format(n)


def print_text_report(metadata, tensor_stats, param_info, show_tensors, show_layers):
    """Print human-readable report."""
    # Header
    print("=" * 60)
    print("  GGUF Model Inspector")
    print("=" * 60)

    # Model info
    print("")
    print("--- Model Information ---")
    for label, value in metadata.items():
        if value is not None and value != "":
            print("  " + "{:<30s}".format(label) + ": " + str(value))

    # File statistics
    print("")
    print("--- File Statistics ---")
    print("  File size            : " + format_bytes(tensor_stats["file_size"]))
    print("  Header size          : " + format_bytes(tensor_stats["header_size"]))
    print("  Tensor data          : " + format_bytes(tensor_stats["tensor_bytes"]))
    print("  Data percentage      : " + str(round(tensor_stats["data_percentage"], 1)) + "%")
    print("  Total tensors        : " + format_number(tensor_stats["total_tensors"]))
    print("  Total elements       : " + format_number(tensor_stats["total_elements"]))

    # Parameter estimate
    print("")
    print("--- Parameters ---")
    print("  Estimated parameters : " + param_info["estimated_params"] + " (" + str(param_info["estimated_params_b"]) + "B)")

    # Tensor type breakdown
    print("")
    print("--- Tensors by Quantization Type ---")
    print("  " + "{:<12} {:>6} {:>12} {:>10}  {}".format("Type", "Count", "Elements", "Bytes", "Description"))
    print("  " + "{:<12} {:>6} {:>12} {:>10}  {}".format("-" * 12, "-" * 6, "-" * 12, "-" * 10, "-" * 30))

    for qname in sorted(tensor_stats["by_type"].keys()):
        qt = tensor_stats["by_type"][qname]
        desc = QUANT_DESCRIPTIONS.get(qname, "")
        print("  " + "{:<12} {:>6} {:>12} {:>10}  {}".format(
            qname, qt["count"], format_number(qt["elements"]), format_bytes(qt["bytes"]), desc))

    # Tensor prefix breakdown
    print("")
    print("--- Tensors by Prefix ---")
    print("  " + "{:<20} {:>6} {:>12} {:>10}".format("Prefix", "Count", "Elements", "Bytes"))
    print("  " + "{:<20} {:>6} {:>12} {:>10}".format("-" * 20, "-" * 6, "-" * 12, "-" * 10))

    for prefix in sorted(tensor_stats["by_prefix"].keys()):
        pt = tensor_stats["by_prefix"][prefix]
        print("  " + "{:<20} {:>6} {:>12} {:>10}".format(
            prefix, pt["count"], format_number(pt["elements"]), format_bytes(pt["bytes"])))

    # Largest tensors
    if show_tensors and tensor_stats["largest_tensors"]:
        print("")
        print("--- Top 10 Largest Tensors ---")
        print("  " + "{:<50} {:<10} {:>12} {:>10}".format("Name", "Type", "Elements", "Bytes"))
        print("  " + "{:<50} {:<10} {:>12} {:>10}".format("-" * 50, "-" * 10, "-" * 12, "-" * 10))
        for name, qname, elements, n_bytes in tensor_stats["largest_tensors"]:
            print("  " + "{:<50} {:<10} {:>12} {:>10}".format(
                name, qname, format_number(elements), format_bytes(n_bytes)))

    # Layer breakdown
    if show_layers and tensor_stats["layers"]:
        print("")
        print("--- Layer Breakdown ---")
        print("  " + "{:<15} {:>7} {:>12} {:>10}".format("Layer", "Tensors", "Elements", "Bytes"))
        print("  " + "{:<15} {:>7} {:>12} {:>10}".format("-" * 15, "-" * 7, "-" * 12, "-" * 10))

        for layer_key in sorted(tensor_stats["layers"].keys()):
            lk = tensor_stats["layers"][layer_key]
            print("  " + "{:<15} {:>7} {:>12} {:>10}".format(
                layer_key, lk["count"], format_number(lk["elements"]), format_bytes(lk["bytes"])))

    print("")
    print("=" * 60)


def print_json_report(metadata, tensor_stats, param_info):
    """Print JSON report."""
    by_type = {k: dict(v) for k, v in tensor_stats["by_type"].items()}
    by_prefix = {k: dict(v) for k, v in tensor_stats["by_prefix"].items()}
    layers = {k: dict(v) for k, v in tensor_stats["layers"].items()}

    report = {
        "model_info": metadata,
        "file_statistics": {
            "file_size_bytes": tensor_stats["file_size"],
            "file_size_human": format_bytes(tensor_stats["file_size"]),
            "header_size_bytes": tensor_stats["header_size"],
            "tensor_data_bytes": tensor_stats["tensor_bytes"],
            "data_percentage": round(tensor_stats["data_percentage"], 2),
            "total_tensors": tensor_stats["total_tensors"],
            "total_elements": tensor_stats["total_elements"],
        },
        "parameters": param_info,
        "tensors_by_type": by_type,
        "tensors_by_prefix": by_prefix,
        "layers": layers,
        "largest_tensors": [
            {"name": name, "type": qname, "elements": elements, "bytes": n_bytes}
            for name, qname, elements, n_bytes in tensor_stats["largest_tensors"]
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced GGUF model file inspector - deep analysis of model files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.gguf                    # Full text report
  %(prog)s model.gguf --json             # JSON output
  %(prog)s model.gguf --tensors          # Show top 10 largest tensors
  %(prog)s model.gguf --layers           # Show per-layer breakdown
  %(prog)s model.gguf --json --layers    # JSON with layer detail
        """,
    )
    parser.add_argument("gguf_file", help="Path to GGUF model file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--tensors", action="store_true", help="Show largest tensors")
    parser.add_argument("--layers", action="store_true", help="Show per-layer breakdown")
    args = parser.parse_args()

    gguf_path = Path(args.gguf_file)
    if not gguf_path.exists():
        print("Error: File not found: " + str(gguf_path), file=sys.stderr)
        sys.exit(1)

    print("Inspecting: " + str(gguf_path), file=sys.stderr)

    reader = GGUFReader(gguf_path)

    metadata = analyze_metadata(reader)
    tensor_stats = analyze_tensors(reader, str(gguf_path))
    param_info = estimate_params(tensor_stats)

    if args.json:
        print_json_report(metadata, tensor_stats, param_info)
    else:
        print_text_report(metadata, tensor_stats, param_info, args.tensors, args.layers)


if __name__ == "__main__":
    main()
