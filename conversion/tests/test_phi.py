#!/usr/bin/env python3
"""Tests for the Phi-2 / Phi-3 / Phi-4 GGUF converter (conversion/phi.py).

These tests exercise ``conversion.phi`` without requiring an actual
Hugging Face checkout on disk: ``ModelBase`` accepts a pre-built
``hparams`` dict and, when no ``*.safetensors`` / ``*.bin`` files are
present in ``dir_model``, indexes zero tensors. That is enough to drive
``set_gguf_parameters()``, ``generate_extra_tensors()``, ``set_vocab()``
and tensor-name mapping in isolation.

The hyperparameters used below are taken from the real
``config.json`` files published for the models they claim to represent
(see the comment above each dict), so a regression here should track a
real conversion regression.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

# Necessary to load the local `conversion` and `gguf` packages when this
# file is run outside of an installed environment.
if (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))

import gguf  # noqa: E402
from conversion.base import ModelBase, ModelType  # noqa: E402
from conversion.phi import (  # noqa: E402
    Phi2Model,
    Phi3MiniModel,
    Phi4VisionMmprojModel,
    PhiMoeModel,
)


def make_model(model_cls, hparams: dict, dir_model: Path | None = None):
    """Instantiate a converter model without touching real weight files."""
    tmpdir = dir_model if dir_model is not None else Path(tempfile.mkdtemp())
    return model_cls(
        dir_model=tmpdir,
        ftype=gguf.LlamaFileType.MOSTLY_F16,
        fname_out=tmpdir / "out.gguf",
        hparams=hparams,
        dry_run=True,
    )


def kv(model, key: str):
    """Look up a value that was written to the model's GGUFWriter."""
    for split in model.gguf_writer.kv_data:
        if key in split:
            return split[key].value
    raise KeyError(key)


class TestPhiArchitectureRegistration(unittest.TestCase):
    """The Phi-4 text model shares its HF architecture name with the
    Phi-4 multimodal projector, so both must resolve correctly depending
    on whether a TEXT or MMPROJ model is being requested."""

    def test_phi3_arch_resolves_to_text_model(self):
        cls = ModelBase.from_model_architecture("Phi3ForCausalLM", ModelType.TEXT)
        self.assertIs(cls, Phi3MiniModel)

    def test_phi4_causal_lmv_resolves_to_text_model(self):
        cls = ModelBase.from_model_architecture("Phi4ForCausalLMV", ModelType.TEXT)
        self.assertIs(cls, Phi3MiniModel)

    def test_phi4_causal_lmv_resolves_to_mmproj_model(self):
        cls = ModelBase.from_model_architecture("Phi4ForCausalLMV", ModelType.MMPROJ)
        self.assertIs(cls, Phi4VisionMmprojModel)

    def test_phi2_arch_resolves_to_phi2_model(self):
        cls = ModelBase.from_model_architecture("PhiForCausalLM", ModelType.TEXT)
        self.assertIs(cls, Phi2Model)

    def test_phimoe_arch_resolves_to_phimoe_model(self):
        cls = ModelBase.from_model_architecture("PhiMoEForCausalLM", ModelType.TEXT)
        self.assertIs(cls, PhiMoeModel)


class TestPhi4DenseConversion(unittest.TestCase):
    """microsoft/phi-4 (14B, dense, non-scaled RoPE).

    Values below mirror the published config.json:
    https://huggingface.co/microsoft/phi-4/blob/main/config.json
    """

    def setUp(self):
        self.hparams = dict(
            architectures=["Phi3ForCausalLM"],
            hidden_size=5120,
            intermediate_size=17920,
            max_position_embeddings=16384,
            num_attention_heads=40,
            num_hidden_layers=40,
            num_key_value_heads=10,
            original_max_position_embeddings=16384,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=250000,
            sliding_window=None,
            vocab_size=100352,
        )
        self.model = make_model(Phi3MiniModel, self.hparams)

    def test_block_count(self):
        self.assertEqual(self.model.block_count, 40)

    def test_set_gguf_parameters(self):
        self.model.set_gguf_parameters()
        self.assertEqual(kv(self.model, "phi3.embedding_length"), 5120)
        self.assertEqual(kv(self.model, "phi3.feed_forward_length"), 17920)
        self.assertEqual(kv(self.model, "phi3.attention.head_count"), 40)
        self.assertEqual(kv(self.model, "phi3.attention.head_count_kv"), 10)
        self.assertEqual(kv(self.model, "phi3.context_length"), 16384)
        self.assertEqual(kv(self.model, "phi3.rope.scaling.original_context_length"), 16384)
        self.assertEqual(kv(self.model, "phi3.rope.freq_base"), 250000)
        # rope_dims = int(partial_rotary_factor * n_embd) // n_head = int(1.0 * 5120) // 40
        self.assertEqual(kv(self.model, "phi3.rope.dimension_count"), 128)

    def test_null_sliding_window_written_as_zero(self):
        # A zero sliding_window is how the converter distinguishes Phi-4
        # (full attention) from earlier sliding-window Phi-3 variants.
        self.model.set_gguf_parameters()
        self.assertEqual(kv(self.model, "phi3.attention.sliding_window"), 0)

    def test_no_extra_tensors_without_rope_long_short_factors(self):
        # Phi-4's config.json has rope_scaling: null, so no
        # rope_factors_long/short tensors should be generated.
        self.assertEqual(list(self.model.generate_extra_tensors()), [])


class TestPhi3LongRopeScaling(unittest.TestCase):
    """Long-context Phi-3 variants (e.g. Phi-3-medium-128k-instruct) reuse
    the same Phi3MiniModel converter but exercise the long/short RoPE
    factor path in generate_extra_tensors(), which Phi-4 itself does not
    use but which the shared converter must still handle correctly."""

    def base_hparams(self, rope_scaling: dict | None) -> dict:
        return dict(
            architectures=["Phi3ForCausalLM"],
            hidden_size=5120,
            intermediate_size=17920,
            max_position_embeddings=131072,
            num_attention_heads=40,
            num_hidden_layers=40,
            num_key_value_heads=10,
            original_max_position_embeddings=4096,
            rms_norm_eps=1e-05,
            rope_theta=250000,
            sliding_window=131072,
            vocab_size=100352,
            rope_scaling=rope_scaling,
        )

    def test_longrope_factors_are_emitted(self):
        half = 64  # rope_dims(128) / 2
        hparams = self.base_hparams({
            "type": "longrope",
            "long_factor": [1.0 + i * 0.01 for i in range(half)],
            "short_factor": [1.0] * half,
        })
        model = make_model(Phi3MiniModel, hparams)
        model.set_gguf_parameters()
        extra = dict(model.generate_extra_tensors())
        self.assertIn("rope_factors_long.weight", extra)
        self.assertIn("rope_factors_short.weight", extra)
        self.assertEqual(tuple(extra["rope_factors_long.weight"].shape), (half,))
        self.assertEqual(tuple(extra["rope_factors_short.weight"].shape), (half,))

    def test_yarn_scaling_type_is_accepted(self):
        half = 64
        hparams = self.base_hparams({
            "type": "yarn",
            "long_factor": [1.0] * half,
            "short_factor": [1.0] * half,
        })
        model = make_model(Phi3MiniModel, hparams)
        model.set_gguf_parameters()
        extra = dict(model.generate_extra_tensors())
        self.assertIn("rope_factors_long.weight", extra)

    def test_missing_rope_type_raises(self):
        half = 64
        hparams = self.base_hparams({
            "long_factor": [1.0] * half,
            "short_factor": [1.0] * half,
        })
        model = make_model(Phi3MiniModel, hparams)
        model.set_gguf_parameters()
        with self.assertRaises(KeyError):
            list(model.generate_extra_tensors())

    def test_mismatched_factor_lengths_raises(self):
        hparams = self.base_hparams({
            "type": "longrope",
            "long_factor": [1.0] * 64,
            "short_factor": [1.0] * 32,
        })
        model = make_model(Phi3MiniModel, hparams)
        model.set_gguf_parameters()
        with self.assertRaises(ValueError):
            list(model.generate_extra_tensors())

    def test_unsupported_scaling_type_raises(self):
        half = 64
        hparams = self.base_hparams({
            "type": "linear",
            "long_factor": [1.0] * half,
            "short_factor": [1.0] * half,
        })
        model = make_model(Phi3MiniModel, hparams)
        model.set_gguf_parameters()
        with self.assertRaises(NotImplementedError):
            list(model.generate_extra_tensors())


class TestPhi4VocabDispatch(unittest.TestCase):
    """Phi-4 is the first Phi-3-architecture model to ship a GPT2Tokenizer
    instead of the SentencePiece tokenizer earlier Phi-3 models use
    (see https://github.com/ggml-org/llama.cpp/issues/10814). Phi3MiniModel
    must route to the correct vocab loader based on tokenizer_config.json."""

    def hparams(self) -> dict:
        return dict(
            architectures=["Phi3ForCausalLM"],
            hidden_size=5120,
            intermediate_size=17920,
            max_position_embeddings=16384,
            num_attention_heads=40,
            num_hidden_layers=40,
            num_key_value_heads=10,
            original_max_position_embeddings=16384,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=250000,
            sliding_window=None,
            vocab_size=100352,
        )

    def test_gpt2_tokenizer_class_dispatches_to_set_vocab_gpt2(self):
        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GPT2Tokenizer"})
        )
        model = make_model(Phi3MiniModel, self.hparams(), dir_model=tmpdir)
        with patch.object(model, "_set_vocab_gpt2") as mock_gpt2:
            model.set_vocab()
        mock_gpt2.assert_called_once()

    def test_missing_sentencepiece_model_raises_with_no_tokenizer_config(self):
        # Older Phi-3 models (and a Phi-4 conversion missing
        # tokenizer_config.json) fall through to the SentencePiece path,
        # which requires tokenizer.model to be present.
        tmpdir = Path(tempfile.mkdtemp())
        model = make_model(Phi3MiniModel, self.hparams(), dir_model=tmpdir)
        with self.assertRaises(ValueError):
            model.set_vocab()

    def test_non_gpt2_tokenizer_class_falls_through_to_sentencepiece(self):
        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "LlamaTokenizer"})
        )
        model = make_model(Phi3MiniModel, self.hparams(), dir_model=tmpdir)
        with patch.object(model, "_set_vocab_gpt2") as mock_gpt2:
            with self.assertRaises(ValueError):
                model.set_vocab()
        mock_gpt2.assert_not_called()


class TestPhi2Conversion(unittest.TestCase):
    """microsoft/phi-2 config.json:
    https://huggingface.co/microsoft/phi-2/blob/main/config.json
    """

    def setUp(self):
        self.hparams = dict(
            architectures=["PhiForCausalLM"],
            hidden_size=2560,
            intermediate_size=10240,
            max_position_embeddings=2048,
            num_attention_heads=32,
            num_hidden_layers=32,
            num_key_value_heads=32,
            layer_norm_eps=1e-05,
            partial_rotary_factor=0.4,
            rope_scaling=None,
            rope_theta=10000.0,
            vocab_size=51200,
        )
        self.model = make_model(Phi2Model, self.hparams)

    def test_set_gguf_parameters(self):
        self.model.set_gguf_parameters()
        self.assertEqual(kv(self.model, "phi2.embedding_length"), 2560)
        # Phi2Model derives the FFN size as 4 * n_embd rather than reading
        # intermediate_size directly; this happens to match phi-2's real
        # intermediate_size of 10240.
        self.assertEqual(kv(self.model, "phi2.feed_forward_length"), 10240)
        self.assertEqual(kv(self.model, "phi2.attention.head_count"), 32)
        self.assertEqual(kv(self.model, "phi2.attention.head_count_kv"), 32)
        # rope_dims = int(partial_rotary_factor * n_embd) // n_head
        #           = int(0.4 * 2560) // 32 = 32
        self.assertEqual(kv(self.model, "phi2.rope.dimension_count"), 32)
        self.assertEqual(kv(self.model, "tokenizer.ggml.add_bos_token"), False)


class TestPhiMoeExpertMerging(unittest.TestCase):
    """PhiMoE (e.g. Phi-3.5-MoE) receives per-expert tensors one at a time
    from the HF checkpoint and must buffer + merge them into a single
    stacked GGUF tensor per weight matrix."""

    def setUp(self):
        self.n_experts = 4
        self.hparams = dict(
            architectures=["PhiMoEForCausalLM"],
            hidden_size=128,
            intermediate_size=256,
            max_position_embeddings=4096,
            num_attention_heads=8,
            num_hidden_layers=2,
            num_key_value_heads=8,
            original_max_position_embeddings=4096,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            sliding_window=None,
            vocab_size=32000,
            num_local_experts=self.n_experts,
            num_experts_per_tok=2,
        )
        self.model = make_model(PhiMoeModel, self.hparams)

    def _expert_tensor(self, w_name: str) -> torch.Tensor:
        # down_proj (w2) has the transposed shape of gate/up (w1/w3).
        return torch.randn(32, 16) if w_name == "w2" else torch.randn(16, 32)

    def test_experts_are_merged_into_stacked_tensors(self):
        bid = 0
        outputs: list[tuple[str, torch.Tensor]] = []
        for xid in range(self.n_experts):
            for w_name in ["w1", "w2", "w3"]:
                name = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                outputs.extend(self.model.modify_tensors(self._expert_tensor(w_name), name, bid))

        names = {name for name, _ in outputs}
        self.assertEqual(names, {
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ffn_up_exps.weight",
        })
        shapes = {name: tuple(t.shape) for name, t in outputs}
        self.assertEqual(shapes["blk.0.ffn_gate_exps.weight"], (self.n_experts, 16, 32))
        self.assertEqual(shapes["blk.0.ffn_down_exps.weight"], (self.n_experts, 32, 16))
        self.assertEqual(shapes["blk.0.ffn_up_exps.weight"], (self.n_experts, 16, 32))

    def test_prepare_tensors_raises_on_unprocessed_experts(self):
        # Simulate a truncated checkpoint: only 2 of 3 weights arrive for
        # a single expert, so the buffered entry is never flushed.
        bid = 0
        name = f"model.layers.{bid}.block_sparse_moe.experts.0.w1.weight"
        list(self.model.modify_tensors(self._expert_tensor("w1"), name, bid))
        with self.assertRaises(ValueError):
            self.model.prepare_tensors()


if __name__ == "__main__":
    unittest.main()