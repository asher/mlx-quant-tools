"""Unit tests for the attn-protect-quantize predicate.

Run with pytest:
  pytest tests/test_attn_protect_predicate.py -v
"""

from __future__ import annotations

import types
import unittest

from mlx_quant_tools.cli import attn_protect_quantize as apq


def make(**overrides):
    """Build a predicate with sensible defaults; overrides go straight through."""
    base = dict(bits=4, group_size=64, attn_protect_mode="bf16")
    base.update(overrides)
    return apq.make_attn_protect_predicate(**base)


# Realistic module paths drawn from Qwen3 (self_attn.*), MoE-VL (linear_attn.*,
# attn.out_proj), and shared structure (lm_head, mlp.*).
PATHS = {
    # Rule 1 — linear_attn (full-attn variant from MoE-VL repacks)
    "linear_attn_qkv": "model.layers.0.linear_attn.in_proj_qkv",
    "linear_attn_z": "model.layers.0.linear_attn.in_proj_z",
    "linear_attn_out": "model.layers.5.linear_attn.out_proj",
    # Rule 2 — attn.out_proj on the visual tower (NOT self_attn.o_proj!)
    "vis_attn_out": "visual.blocks.0.attn.out_proj",
    # Rule 3 — q/k/v/o_proj under self_attn
    "self_attn_q": "model.layers.0.self_attn.q_proj",
    "self_attn_k": "model.layers.7.self_attn.k_proj",
    "self_attn_v": "model.layers.0.self_attn.v_proj",
    "self_attn_o": "model.layers.0.self_attn.o_proj",
    # Rule 4 — lm_head
    "lm_head": "lm_head",
    "lm_head_nested": "model.lm_head",
    # Default — MLP and other quantizable modules
    "mlp_gate": "model.layers.0.mlp.gate_proj",
    "mlp_up": "model.layers.0.mlp.up_proj",
    "mlp_down": "model.layers.0.mlp.down_proj",
    "expert_gate": "model.layers.0.mlp.experts.0.gate_proj",
    # Tied-embedding output projection (Qwen3-0.6B uses this layout)
    "embed_tokens": "model.embed_tokens",
    "embed_tokens_alt": "transformer.wte",
    # Edge cases — names that LOOK like floor-rule targets but shouldn't match
    "fake_q_proj": "model.something.q_proj",  # not under self_attn
    "fake_attn_out": "model.layers.0.self_attn.out_proj",  # has self_ prefix
    "fake_lm_head_substr": "model.layers.0.head_lm",  # contains 'lm_head'? no
    "fake_lm_head_word": "model.lm_head_lora",  # not the exact tail
    # VLM tower paths (--protect-vlm). The proj/o_proj sub-paths inside a
    # vision tower would otherwise be caught by Rule 3; the protect-vlm rule
    # must override them.
    "vlm_vision_tower_qkv": "vision_tower.encoder.layers.0.self_attn.q_proj",
    "vlm_vision_tower_proj": "vision_tower.encoder.layers.0.mlp.down_proj",
    "vlm_vision_model_attn": "vision_model.blocks.3.attn.proj",
    "vlm_visual_root": "visual.blocks.0.attn.proj",
    "vlm_audio_tower": "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear",
    "vlm_mm_projector": "multi_modal_projector.linear_1",
    "vlm_merger": "model.merger.linear_fc1",
    "vlm_connector": "model.connector.linear_fc2",
    "vlm_embed_vision": "embed_vision.embedding_projection",  # Gemma-4 projector
    "vlm_embed_audio": "embed_audio.embedding_projection",
    # Edge case: 'merger'/'connector' must require a trailing dot to avoid
    # accidentally matching unrelated names like `model.layers.0.merger_norm`.
    "fake_merger_substr": "model.layers.0.merger_proj",  # no '.' after 'merger'
}


class PredicateRulesTest(unittest.TestCase):
    """Default config (bf16 protect, bits=4, gs=64) — exhaustive coverage."""

    def setUp(self):
        self.p = make()

    def assertVerdict(self, path: str, expected):
        got = self.p(path, None)
        self.assertEqual(got, expected, f"path={path!r}")

    def test_rule1_linear_attn_skipped(self):
        for key in ("linear_attn_qkv", "linear_attn_z", "linear_attn_out"):
            self.assertVerdict(PATHS[key], False)

    def test_rule2_attn_out_proj_skipped(self):
        self.assertVerdict(PATHS["vis_attn_out"], False)

    def test_rule3_qkvo_floored_to_8bit(self):
        for key in ("self_attn_q", "self_attn_k", "self_attn_v", "self_attn_o"):
            self.assertVerdict(PATHS[key], {"bits": 8, "group_size": 64})

    def test_rule4_lm_head_8bit(self):
        self.assertVerdict(PATHS["lm_head"], {"bits": 8, "group_size": 64})
        self.assertVerdict(PATHS["lm_head_nested"], {"bits": 8, "group_size": 64})

    def test_default_passthrough(self):
        for key in ("mlp_gate", "mlp_up", "mlp_down", "expert_gate"):
            self.assertVerdict(PATHS[key], True)

    def test_edge_cases_do_not_match_floor_rules(self):
        # q_proj outside self_attn → default
        self.assertVerdict(PATHS["fake_q_proj"], True)
        # self_attn.out_proj must NOT trigger Rule 2 (Rule 2 is for non-self attn)
        # Note: it also won't trigger Rule 3 (out_proj != o_proj). Default.
        self.assertVerdict(PATHS["fake_attn_out"], True)
        # 'head_lm' isn't 'lm_head'
        self.assertVerdict(PATHS["fake_lm_head_substr"], True)
        # 'lm_head_lora' tail isn't an exact 'lm_head' match
        self.assertVerdict(PATHS["fake_lm_head_word"], True)


class GroupSizeHonoredTest(unittest.TestCase):
    """No rule hardcodes group_size=64 — all four honor --group-size."""

    def test_group_size_32(self):
        p = make(group_size=32)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 32})
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 32})

    def test_group_size_128(self):
        p = make(group_size=128)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 128})
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 128})

    def test_group_size_propagates_to_8bit_protect_mode(self):
        p = make(attn_protect_mode="8bit", group_size=128)
        self.assertEqual(p(PATHS["linear_attn_qkv"], None), {"bits": 8, "group_size": 128})
        self.assertEqual(p(PATHS["vis_attn_out"], None), {"bits": 8, "group_size": 128})


class AttnProtectModeTest(unittest.TestCase):
    """--attn-protect-mode flips Rule 1 / Rule 2; other rules unchanged."""

    def test_bf16_skips_protect_layers(self):
        p = make(attn_protect_mode="bf16")
        self.assertIs(p(PATHS["linear_attn_qkv"], None), False)
        self.assertIs(p(PATHS["vis_attn_out"], None), False)

    def test_8bit_quantizes_protect_layers_at_floor(self):
        p = make(attn_protect_mode="8bit")
        self.assertEqual(p(PATHS["linear_attn_qkv"], None), {"bits": 8, "group_size": 64})
        self.assertEqual(p(PATHS["vis_attn_out"], None), {"bits": 8, "group_size": 64})

    def test_other_rules_unchanged_in_8bit_mode(self):
        bf16 = make(attn_protect_mode="bf16")
        eight = make(attn_protect_mode="8bit")
        for key in ("self_attn_q", "lm_head", "mlp_down"):
            self.assertEqual(
                bf16(PATHS[key], None),
                eight(PATHS[key], None),
                f"non-protect path {key!r} should be unchanged across modes",
            )


class BitsBoundaryTest(unittest.TestCase):
    """Floor logic: max(--bits, 8) for Rule 3; lm_head always 8."""

    def test_bits_4_qkvo_floored_to_8(self):
        p = make(bits=4)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})

    def test_bits_8_qkvo_no_op(self):
        p = make(bits=8)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})

    def test_bits_8_lm_head_no_op(self):
        p = make(bits=8)
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 64})

    def test_bits_8_protect_rules_still_apply_bf16(self):
        p = make(bits=8, attn_protect_mode="bf16")
        self.assertIs(p(PATHS["linear_attn_qkv"], None), False)
        self.assertIs(p(PATHS["vis_attn_out"], None), False)

    def test_bits_8_protect_rules_8bit_mode(self):
        p = make(bits=8, attn_protect_mode="8bit")
        self.assertEqual(p(PATHS["linear_attn_qkv"], None), {"bits": 8, "group_size": 64})

    def test_bits_6_qkvo_still_floored_to_8(self):
        p = make(bits=6)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})


class AblationFlagsTest(unittest.TestCase):
    """Each --no-* flag disables exactly one rule; others remain."""

    def test_quantize_linear_attn(self):
        p = make(quantize_linear_attn=True)
        self.assertIs(p(PATHS["linear_attn_qkv"], None), True)
        self.assertIs(p(PATHS["vis_attn_out"], None), False)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 64})

    def test_quantize_attn_out(self):
        p = make(quantize_attn_out=True)
        self.assertIs(p(PATHS["vis_attn_out"], None), True)
        self.assertIs(p(PATHS["linear_attn_qkv"], None), False)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 64})

    def test_no_attn_floor(self):
        p = make(no_attn_floor=True)
        self.assertIs(p(PATHS["self_attn_q"], None), True)
        self.assertIs(p(PATHS["linear_attn_qkv"], None), False)
        self.assertIs(p(PATHS["vis_attn_out"], None), False)
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 64})

    def test_no_lm_head_floor(self):
        p = make(no_lm_head_floor=True)
        self.assertIs(p(PATHS["lm_head"], None), True)
        self.assertIs(p(PATHS["linear_attn_qkv"], None), False)
        self.assertIs(p(PATHS["vis_attn_out"], None), False)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})

    def test_all_ablations_together_matches_default_uniform(self):
        p = make(
            quantize_linear_attn=True,
            quantize_attn_out=True,
            no_attn_floor=True,
            no_lm_head_floor=True,
        )
        for path in PATHS.values():
            self.assertIs(p(path, None), True, f"path={path!r}")


class FloorTiedEmbedTest(unittest.TestCase):
    """--floor-tied-embed extends Rule 4 to embed_tokens for tied-embed models."""

    def test_default_off_leaves_embed_at_default(self):
        p = make()
        self.assertIs(p(PATHS["embed_tokens"], None), True)
        self.assertIs(p(PATHS["embed_tokens_alt"], None), True)

    def test_on_floors_embed_to_8bit(self):
        p = make(floor_tied_embed=True)
        self.assertEqual(p(PATHS["embed_tokens"], None), {"bits": 8, "group_size": 64})
        self.assertEqual(p(PATHS["embed_tokens_alt"], None), {"bits": 8, "group_size": 64})

    def test_honors_group_size(self):
        p = make(floor_tied_embed=True, group_size=32)
        self.assertEqual(p(PATHS["embed_tokens"], None), {"bits": 8, "group_size": 32})

    def test_disabled_by_no_lm_head_floor(self):
        p = make(floor_tied_embed=True, no_lm_head_floor=True)
        self.assertIs(p(PATHS["embed_tokens"], None), True)
        self.assertIs(p(PATHS["lm_head"], None), True)

    def test_does_not_affect_non_embed_paths(self):
        p = make(floor_tied_embed=True)
        self.assertIs(p(PATHS["linear_attn_qkv"], None), False)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 64})
        self.assertIs(p(PATHS["mlp_down"], None), True)


class ProtectVlmTest(unittest.TestCase):
    """--protect-vlm extends Rule 1 to vision/audio tower and projector paths."""

    VLM_PATHS = (
        "vlm_vision_tower_qkv",
        "vlm_vision_tower_proj",
        "vlm_vision_model_attn",
        "vlm_visual_root",
        "vlm_audio_tower",
        "vlm_mm_projector",
        "vlm_merger",
        "vlm_connector",
        "vlm_embed_vision",
        "vlm_embed_audio",
    )

    def test_default_off_leaves_vlm_paths_at_default(self):
        p = make()
        self.assertIs(p(PATHS["vlm_vision_tower_proj"], None), True)
        self.assertIs(p(PATHS["vlm_audio_tower"], None), True)
        self.assertIs(p(PATHS["vlm_mm_projector"], None), True)
        self.assertIs(p(PATHS["vlm_embed_vision"], None), True)
        self.assertIs(p(PATHS["vlm_embed_audio"], None), True)
        self.assertEqual(
            p(PATHS["vlm_vision_tower_qkv"], None),
            {"bits": 8, "group_size": 64},
        )

    def test_on_skips_vlm_paths_in_bf16_mode(self):
        p = make(protect_vlm=True)
        for key in self.VLM_PATHS:
            self.assertIs(
                p(PATHS[key], None),
                False,
                f"VLM path {key!r} should be skipped under --protect-vlm bf16",
            )

    def test_on_8bit_mode_floors_vlm_paths(self):
        p = make(protect_vlm=True, attn_protect_mode="8bit")
        for key in self.VLM_PATHS:
            self.assertEqual(
                p(PATHS[key], None),
                {"bits": 8, "group_size": 64},
                f"VLM path {key!r} should get 8-bit floor under --protect-vlm 8bit",
            )

    def test_on_overrides_rule3_inside_vlm_towers(self):
        p = make(protect_vlm=True)
        self.assertIs(p(PATHS["vlm_vision_tower_qkv"], None), False)
        self.assertIs(p(PATHS["vlm_vision_model_attn"], None), False)

    def test_on_does_not_affect_language_model_paths(self):
        p = make(protect_vlm=True)
        self.assertIs(p(PATHS["linear_attn_qkv"], None), False)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})
        self.assertEqual(p(PATHS["lm_head"], None), {"bits": 8, "group_size": 64})
        self.assertIs(p(PATHS["mlp_down"], None), True)

    def test_merger_requires_trailing_dot(self):
        p = make(protect_vlm=True)
        self.assertIs(p(PATHS["fake_merger_substr"], None), True)


class BoostsMapTest(unittest.TestCase):
    """Per-tensor `boosts` map (Tier 2) overrides every other rule."""

    def test_default_no_boosts_unchanged(self):
        p = make()
        self.assertIs(p(PATHS["mlp_gate"], None), True)
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 8, "group_size": 64})

    def test_boost_overrides_default_mlp(self):
        p = make(boosts={PATHS["mlp_gate"]: {"bits": 5, "group_size": 64}})
        self.assertEqual(p(PATHS["mlp_gate"], None), {"bits": 5, "group_size": 64})
        self.assertIs(p(PATHS["mlp_up"], None), True)

    def test_boost_overrides_attention_floor(self):
        p = make(boosts={PATHS["self_attn_q"]: {"bits": 6, "group_size": 64}})
        self.assertEqual(p(PATHS["self_attn_q"], None), {"bits": 6, "group_size": 64})
        self.assertEqual(p(PATHS["self_attn_k"], None), {"bits": 8, "group_size": 64})

    def test_boost_returns_a_copy(self):
        boosts = {PATHS["mlp_gate"]: {"bits": 5, "group_size": 64}}
        p = make(boosts=boosts)
        v = p(PATHS["mlp_gate"], None)
        v["bits"] = 99
        v2 = p(PATHS["mlp_gate"], None)
        self.assertEqual(v2["bits"], 5)

    def test_empty_boosts_dict_equivalent_to_none(self):
        p_none = make(boosts=None)
        p_empty = make(boosts={})
        for path in PATHS.values():
            self.assertEqual(p_none(path, None), p_empty(path, None), f"path={path!r}")


class OutputPathTest(unittest.TestCase):
    """default_output_path suffixes assemble correctly."""

    def _args(self, **kw):
        defaults = dict(
            bits=4,
            group_size=64,
            attn_protect_mode="bf16",
            with_dwq=False,
            floor_tied_embed_effective=False,
        )
        defaults.update(kw)
        return types.SimpleNamespace(**defaults)

    def test_default_4bit_bf16(self):
        out = apq.default_output_path("Qwen/Qwen3.6-4B", self._args())
        self.assertEqual(out.name, "Qwen3.6-4B-AP4bit")

    def test_8bit_mode_suffix(self):
        out = apq.default_output_path("Qwen/Qwen3.6-4B", self._args(attn_protect_mode="8bit"))
        self.assertEqual(out.name, "Qwen3.6-4B-AP4bit-8bit")

    def test_dwq_suffix(self):
        out = apq.default_output_path("Qwen/Qwen3.6-4B", self._args(with_dwq=True))
        self.assertEqual(out.name, "Qwen3.6-4B-AP4bit-dwq")

    def test_group_size_suffix_only_when_nondefault(self):
        out64 = apq.default_output_path("model", self._args(group_size=64))
        self.assertEqual(out64.name, "model-AP4bit")
        out32 = apq.default_output_path("model", self._args(group_size=32))
        self.assertEqual(out32.name, "model-AP4bit-gs32")

    def test_local_path_basename(self):
        out = apq.default_output_path("/Users/me/llm/mlx/qwen3.6-27b/", self._args())
        self.assertEqual(out.name, "qwen3.6-27b-AP4bit")

    def test_full_suffix_combo(self):
        out = apq.default_output_path(
            "Qwen/Qwen3.6-27B",
            self._args(bits=4, attn_protect_mode="8bit", with_dwq=True, group_size=32),
        )
        self.assertEqual(out.name, "Qwen3.6-27B-AP4bit-8bit-dwq-gs32")

    def test_tied8_suffix_only_when_effective(self):
        out = apq.default_output_path(
            "Qwen/Qwen3-4B", self._args(floor_tied_embed_effective=False)
        )
        self.assertEqual(out.name, "Qwen3-4B-AP4bit")
        out = apq.default_output_path(
            "Qwen/Qwen3-0.6B", self._args(floor_tied_embed_effective=True)
        )
        self.assertEqual(out.name, "Qwen3-0.6B-AP4bit-tied8")
        out = apq.default_output_path(
            "Qwen/Qwen3-0.6B",
            self._args(floor_tied_embed_effective=True, attn_protect_mode="8bit", with_dwq=True),
        )
        self.assertEqual(out.name, "Qwen3-0.6B-AP4bit-tied8-8bit-dwq")


if __name__ == "__main__":
    unittest.main(verbosity=2)
