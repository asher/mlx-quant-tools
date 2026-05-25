"""K-quant in-process quantizer for MLX checkpoints.

Drives `mx.quantize(mode="kquant", kquant_type=...)` per-tensor over an HF
model, swaps in `nn.QuantizedLinear(mode="kquant", ...)` modules, and saves
a load-ready checkpoint. Replaces the external `llama-quantize` round-trip
for K-quant production work.

Output format
-------------
- safetensors with uint8 wire-byte tensors (one per quantizable Linear)
- updated config.json with `quantization_config.mode = "kquant"` and a
  `per_tensor` map of {tensor_path: kquant_type}
- optional Markdown + JSON validation report

Usage examples
--------------
  # Default codec across the whole model:
  mqt-quantize-kquant Qwen/Qwen3-0.6B --codec q4_k -o /tmp/qwen3-q4k

  # Preset (attention/lm_head boosted, FFN at base codec):
  mqt-quantize-kquant Qwen/Qwen3-0.6B --preset q4_k_m -o /tmp/qwen3-q4km

  # With imatrix (importance-weighted encoding):
  mqt-quantize-kquant Qwen/Qwen3-0.6B --codec q4_k --imatrix imat.dat \\
        -o /tmp/qwen3-q4k-imat

  # Plan-only:
  mqt-quantize-kquant Qwen/Qwen3-0.6B --preset q4_k_m --dry-run

  # Validate post-quantization round-trip:
  mqt-quantize-kquant Qwen/Qwen3-0.6B --codec q4_k -o /tmp/out --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_quant_tools.model_role_classifier import (
    TensorRole,
    classify_tensors,
    extract_layer_idx,
)

# Codec geometry (group_size, bits, bytes_per_block, weights_per_block).
# Source of truth: mlx::core::kquant_codec_by_name (mlx/primitives.cpp).
# Mirrored from run-gguf-kquant.py --- kept in sync manually.
CODEC_GEOMETRY: dict[str, tuple[int, int, int, int]] = {
    "q8_0": (32, 8, 34, 32),
    "q4_k": (256, 4, 144, 256),
    "q5_k": (256, 5, 176, 256),
    "q6_k": (256, 6, 210, 256),
    "q3_k": (256, 3, 110, 256),
    "q2_k": (256, 2, 84, 256),
}

# Codecs supported by the MLX encoder (mx.quantize(mode="kquant")). All six
# K-codecs in the table above plus Q8_0. Other GGUF codecs (Q4_0/Q4_1/Q5_0/
# Q5_1) decode fine but have no MLX encoder --- skip them in presets.
ENCODER_CODECS = set(CODEC_GEOMETRY.keys())

# Bf16 pass-through role: never quantized regardless of preset. Embedding
# lookup tables and small per-token gates (routers) stay full-precision;
# quantizing them is known to hurt quality at the byte budgets K-quant is
# targeting.
#
# LINEAR_ATTN is intentionally NOT in this set --- by default it's quantized
# at the preset's base codec, with `_xl` variants picking per-submodule
# codecs (out_proj->Q8_0, in_proj_qkv->Q6_K, in_proj_z->Q5_K) to match
# Unsloth's UD pattern. Until 2026-05-16 it WAS in this set, which caused
# `q4_k_m` to silently keep ~11 GB of Mamba projections at bf16 on
# Qwen3.6-27B --- making `q4_k_m` (25 GB) larger than `q4_k_xl` (21 GB).
# AP-protected presets opt linear_attn back into bf16 via
# `_PRESET_BF16_PASSTHROUGH`.
_BF16_PASSTHROUGH_ROLES = frozenset(
    {
        TensorRole.EMBEDDING,
        TensorRole.EMBEDDING_PER_LAYER,
        TensorRole.MOE_ROUTER,
    }
)


# ---------- preset -> role -> codec maps ----------

# Each preset gives every role a codec (or None for bf16 pass-through).
# Roles unmatched by a path fall back to the `default` slot. Preset
# semantics mirror llama-quantize's templates (`src/llama-quant.cpp`):
#   - `_s` / `_m` / `_xl`: progressively heavier protection within a base
#     codec family. `_s` is the leanest; `_m` adds llama-quantize's
#     `use_more_bits` selective bump for FFN down_proj (32 of 64 layers
#     on Qwen3.6-27B); `_xl` adds Unsloth-style Mamba per-submodule bumps.
#   - `_ap` / `_ap_xl`: attn-protect variants --- linear_attn bf16 (Rule 1),
#     optionally attention_output bf16 (VLM cross-attn Rule 1a).
#
# Every preset's role map is uniform-per-role. Per-layer bumps go through
# `_PATH_BUMPS` (suffix match) or `_LAYER_POSITION_BUMPS` (suffix + layer
# index predicate). Resolution order in resolve_codec_map: explicit
# overrides > path bumps > layer-position bumps > explicit preset role
# entry > per-preset BF16 passthrough > global BF16 passthrough > base
# default.
#
# Preset summary (codec -> attn_v / ffn_down / lm_head / embed_tokens):
#   q4_k_s:    Q4_K / Q4_K / Q6_K / Q4_K     attn_v->Q6_K
#   q4_k_m:    Q4_K / use_more_bits->Q6_K / Q6_K / Q4_K     attn_v->Q6_K
#                + linear_attn.out_proj->Q5_K
#   q4_k_xl:   q4_k_m + down_proj uniform Q6_K + Unsloth Mamba per-submodule
#              (out_proj->Q8_0, in_proj_qkv->Q6_K, in_proj_z->Q5_K).
#   q4_k_ap:   q4_k_m with linear_attn bf16 (Rule 1) + ATTENTION_OUTPUT
#              bf16 (Rule 1a, cross-attn protection on VLMs/audio).
#   q4_k_ap_xl: q4_k_m + down_proj uniform Q6_K, linear_attn all bf16.
#   q4_k_moe:  q4_k_m + routed_expert->Q4_K, shared_expert->Q5_K. MoE-aware.
#   q5_k_*:    Q5_K parallel of the above. attn_v stays Q6_K, lm_head Q6_K.
#              Mamba.out_proj->Q6_K in `_m`; `_xl` Mamba: out_proj->Q8_0,
#              in_proj_qkv->Q6_K, in_proj_z->Q6_K.
#   q5_k_moe:  q5_k_m + routed_expert->Q5_K, shared_expert->Q6_K.
#   q3_k_m:    Q3_K / use_more_bits->Q5_K / Q5_K / Q3_K     attn_v->Q5_K
#                + linear_attn.out_proj->Q5_K
#   q2_k:      Q2_K / Q2_K / Q3_K / Q2_K  ATTENTION_OUTPUT->Q3_K,
#              attn_v->Q3_K, linear_attn.out_proj->Q3_K. Matches
#              llama-quantize Q2_K's selective Q3_K bumps on sensitive
#              tensors (Q2_K is the floor codec but not used uniformly).
#   q6_k:      Pure Q6_K. Top of the K-quant ladder; no `_m`/`_xl`
#              variants because there's no higher K-quant codec to bump
#              into. Linear_attn stays at Q6_K base.
#   q8_k_xl:   Q8_0 base + attn K/V bf16 (UD-Q8_K_XL universal protection).
#              Note: keeps attn_v bf16 (overrides the v_proj bump) --- K/V
#              are the most sensitive to quantization downstream of softmax.
#              Skips the per-layer dynamic bumps that need imatrix
#              sensitivity analysis.
# `_m` presets match llama-quantize's templates (see src/llama-quant.cpp):
#   - attn_v: uniform bump to base+1 codec (most-sensitive attn projection)
#   - attn_q/k/o: base codec (no bump in llama-quantize Q*_K_M)
#   - ffn_down: bumped via _LAYER_POSITION_BUMPS using `use_more_bits`
#   - lm_head: base+1 codec
#   - embed_tokens: base codec (quantized; not bf16 pass-through)
#   - linear_attn.out_proj: base+1 codec (Mamba-aware Unsloth extension)
#   - linear_attn.*: base codec (other Mamba projections)
#   - norms: bf16 (handled outside the preset --- not classifiable Linears)
#
# `_s` presets are the `_m` template MINUS the ffn_down layer-position bump.
# `_ap*` presets keep linear_attn / attention_output bf16 (see
# _PRESET_BF16_PASSTHROUGH).
# `_xl` presets layer per-submodule Mamba bumps over the `_m` base via path
# bumps; `_xxl` if we ever ship one would extend further.
_PRESETS: dict[str, dict[str, str | None]] = {
    # -- Q4 family --
    "q4_k_s": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_m": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_xl": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_xl_ap8": {
        # AP8: every attention-family linear floors at Q8_0 (self_attn q/k/v/o
        # via this role entry; linear_attn submodules via path bumps).
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q8_0",
        TensorRole.ATTENTION_OUTPUT.value: "q8_0",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_ap_xl": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_ap": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: None,
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
    },
    "q4_k_moe": {
        "default": "q4_k",
        TensorRole.ATTENTION_QKVO.value: "q4_k",
        TensorRole.ATTENTION_OUTPUT.value: "q4_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q4_k",
        TensorRole.SHARED_EXPERT.value: "q5_k",
        TensorRole.ROUTED_EXPERT.value: "q4_k",
    },
    # -- Q5 family --
    "q5_k_s": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
    },
    "q5_k_m": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
    },
    "q5_k_xl": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
    },
    "q5_k_ap_xl": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
    },
    "q5_k_moe": {
        "default": "q5_k",
        TensorRole.ATTENTION_QKVO.value: "q5_k",
        TensorRole.ATTENTION_OUTPUT.value: "q5_k",
        TensorRole.LM_HEAD.value: "q6_k",
        TensorRole.EMBEDDING.value: "q5_k",
        TensorRole.SHARED_EXPERT.value: "q6_k",
        TensorRole.ROUTED_EXPERT.value: "q5_k",
    },
    # -- Q3 family --
    "q3_k_m": {
        "default": "q3_k",
        TensorRole.ATTENTION_QKVO.value: "q3_k",
        TensorRole.ATTENTION_OUTPUT.value: "q3_k",
        TensorRole.LM_HEAD.value: "q5_k",
        TensorRole.EMBEDDING.value: "q3_k",
    },
    # -- Q6 family --
    "q6_k_m": {
        # Q6_K base; lm_head + Mamba.out_proj bumped to Q8_0. No llama-quantize
        # analog (llama-quant Q6_K is uniform) --- modeled on q4_k_m's "M" shape.
        "default": "q6_k",
        TensorRole.ATTENTION_QKVO.value: "q6_k",
        TensorRole.ATTENTION_OUTPUT.value: "q6_k",
        TensorRole.LM_HEAD.value: "q8_0",
        TensorRole.EMBEDDING.value: "q6_k",
    },
    "q6_k_xl": {
        # Mirrors Unsloth UD-Q6_K_XL: attn q/k/v -> Q8_0 (input projections);
        # self_attn.o_proj stays at Q6_K (path bump back); ffn_down selective
        # Q8_0 via use_more_bits; embed + lm_head Q8_0.
        "default": "q6_k",
        TensorRole.ATTENTION_QKVO.value: "q8_0",
        TensorRole.ATTENTION_OUTPUT.value: "q6_k",
        TensorRole.LM_HEAD.value: "q8_0",
        TensorRole.EMBEDDING.value: "q8_0",
    },
    "q6_k_xl_ap8": {
        # q6_k_xl base + uniform Q8_0 floor on EVERY attention path
        # (self_attn.q/k/v/o via role; linear_attn via path bumps below).
        "default": "q6_k",
        TensorRole.ATTENTION_QKVO.value: "q8_0",
        TensorRole.ATTENTION_OUTPUT.value: "q8_0",
        TensorRole.LM_HEAD.value: "q8_0",
        TensorRole.EMBEDDING.value: "q8_0",
    },
    "q6_k_xl_ap16": {
        # q6_k_xl base + bf16 floor on every attention path. Role-level None
        # forces bf16 on self_attn; path bumps below force it on linear_attn.
        "default": "q6_k",
        TensorRole.ATTENTION_QKVO.value: None,
        TensorRole.ATTENTION_OUTPUT.value: None,
        TensorRole.LM_HEAD.value: "q8_0",
        TensorRole.EMBEDDING.value: "q8_0",
    },
    # -- Top / floor --
    "q6_k": {"default": "q6_k"},
    "q2_k": {
        # llama-quantize Q2_K is not pure --- sensitive tensors get Q3_K bumps.
        "default": "q2_k",
        TensorRole.ATTENTION_QKVO.value: "q2_k",
        TensorRole.ATTENTION_OUTPUT.value: "q3_k",
        TensorRole.LM_HEAD.value: "q3_k",
        TensorRole.EMBEDDING.value: "q2_k",
    },
    "q8_k_xl": {"default": "q8_0"},
    # Pure uniform Q8_0 --- no bumps, no bf16 anywhere (incl. embed). Upper
    # bound for the K-quant family before bf16 protection kicks in.
    "q8": {"default": "q8_0", TensorRole.EMBEDDING.value: "q8_0"},
    "q8_xl_ap16": {
        # Q8_0 base + bf16 floor on every attention path (parallel to
        # q6_k_xl_ap16). Different from q8_k_xl which only bf16s K/V.
        "default": "q8_0",
        TensorRole.ATTENTION_QKVO.value: None,
        TensorRole.ATTENTION_OUTPUT.value: None,
    },
}

# attn_v gets uniform Q6_K (or one rung up) in every llama-quantize _S/_M/_XL
# template --- encoded as a path bump so it overrides the role-level codec
# without needing a separate ATTENTION_V_PROJ role.
_ATTN_V_BUMP = {".self_attn.v_proj": "q6_k"}
# linear_attn.out_proj at base+1 --- Mamba-aware extension matching Unsloth GGUFs.
_MAMBA_OUT_BUMP_Q5K = {".linear_attn.out_proj": "q5_k"}
_MAMBA_OUT_BUMP_Q6K = {".linear_attn.out_proj": "q6_k"}

# Path-suffix bumps applied AFTER role resolution but BEFORE explicit
# --per-tensor-codecs overrides. Used for codec choices that depend on
# sub-tensor identity (gate vs up vs down inside an FFN, or per-submodule
# routing inside Mamba `linear_attn`). Path bumps OVERRIDE the BF16
# passthrough rules --- that's how the `_xl` variants pick per-submodule
# codecs on linear_attn. A None value forces bf16 pass-through (per
# UD-Q8_K_XL recipe).
_MAMBA_LINEAR_ATTN_XL = {
    ".linear_attn.out_proj": "q8_0",
    ".linear_attn.in_proj_qkv": "q6_k",
    ".linear_attn.in_proj_z": "q5_k",
}
_MAMBA_LINEAR_ATTN_Q5_XL = {
    ".linear_attn.out_proj": "q8_0",
    ".linear_attn.in_proj_qkv": "q6_k",
    ".linear_attn.in_proj_z": "q6_k",
}
# Uniform Q8_0 floor on every linear_attn submodule --- the "AP8" variant.
# Heavier than the Unsloth per-submodule mix above, but provides a clean
# uniform-precision floor for benchmarking AP8 vs AP16 (bf16) tradeoffs.
_MAMBA_LINEAR_ATTN_AP8 = {
    ".linear_attn.out_proj": "q8_0",
    ".linear_attn.in_proj_qkv": "q8_0",
    ".linear_attn.in_proj_z": "q8_0",
    ".linear_attn.in_proj_a": "q8_0",
    ".linear_attn.in_proj_b": "q8_0",
}
_PATH_BUMPS: dict[str, dict[str, str | None]] = {
    # Q4 family: attn_v + linear_attn.out_proj uniformly bumped to next rung
    # (Q6_K and Q5_K respectively) per llama-quantize Q4_K_* + Unsloth Mamba
    # extension. `_xl` adds down_proj uniform Q6_K AND per-submodule Mamba.
    "q4_k_s": {".self_attn.v_proj": "q6_k", **_MAMBA_OUT_BUMP_Q5K},
    "q4_k_m": {".self_attn.v_proj": "q6_k", **_MAMBA_OUT_BUMP_Q5K},
    "q4_k_moe": {".self_attn.v_proj": "q6_k", **_MAMBA_OUT_BUMP_Q5K},
    "q4_k_xl": {".self_attn.v_proj": "q6_k", ".mlp.down_proj": "q6_k", **_MAMBA_LINEAR_ATTN_XL},
    "q4_k_xl_ap8": {".mlp.down_proj": "q6_k", **_MAMBA_LINEAR_ATTN_AP8},
    "q4_k_ap": {".self_attn.v_proj": "q6_k"},
    "q4_k_ap_xl": {".self_attn.v_proj": "q6_k", ".mlp.down_proj": "q6_k"},
    # Q5 family: attn_v + linear_attn.out_proj at Q6_K.
    "q5_k_s": {".self_attn.v_proj": "q6_k", **_MAMBA_OUT_BUMP_Q6K},
    "q5_k_m": {".self_attn.v_proj": "q6_k", **_MAMBA_OUT_BUMP_Q6K},
    "q5_k_moe": {".self_attn.v_proj": "q6_k", **_MAMBA_OUT_BUMP_Q6K},
    "q5_k_xl": {".self_attn.v_proj": "q6_k", ".mlp.down_proj": "q6_k", **_MAMBA_LINEAR_ATTN_Q5_XL},
    "q5_k_ap_xl": {".self_attn.v_proj": "q6_k", ".mlp.down_proj": "q6_k"},
    # Q6 family. q6_k_m: minimal Mamba protection only. q6_k_xl: Unsloth UD
    # pattern --- q/k/v already Q8_0 via role; self_attn.o_proj already Q6_K via
    # role override above; Mamba per-submodule mix lifted relative to Q4_K_XL
    # (z bumped to Q8 since base is Q6).
    "q6_k_m": {".linear_attn.out_proj": "q8_0"},
    "q6_k_xl": {
        ".self_attn.o_proj": "q6_k",
        ".linear_attn.out_proj": "q8_0",
        ".linear_attn.in_proj_qkv": "q8_0",
        ".linear_attn.in_proj_z": "q8_0",
    },
    "q6_k_xl_ap8": {**_MAMBA_LINEAR_ATTN_AP8},  # all linear_attn at Q8_0
    # No path bumps for q6_k_xl_ap16 --- LINEAR_ATTN passthrough below makes
    # all linear_attn submodules bf16. self_attn already bf16 via role.
    # Q3 family: attn_v + linear_attn.out_proj at Q5_K (one rung up from Q3_K).
    "q3_k_m": {".self_attn.v_proj": "q5_k", ".linear_attn.out_proj": "q5_k"},
    # Q2_K: attn_v + linear_attn.out_proj at Q3_K (one rung up from floor).
    "q2_k": {".self_attn.v_proj": "q3_k", ".linear_attn.out_proj": "q3_k"},
    # Q8_K_XL: K/V projections explicitly bf16 --- Unsloth UD-Q8_K_XL universal
    # protection (K/V are the most sensitive to quantization downstream of
    # softmax). Note this overrides any v_proj bump.
    "q8_k_xl": {".self_attn.k_proj": None, ".self_attn.v_proj": None},
}

# Layer-position bumps applied AFTER path bumps and AFTER explicit role-to-codec
# preset entries, but BEFORE passthrough and base default. Replicates
# llama-quantize's `use_more_bits` ffn_down selection (see llama-quant.cpp).
# Map: preset -> suffix -> (bumped_codec, rule_name). The rule is a callable
# `(i_layer, n_layers) -> bool`; True means apply the bumped codec.


def _use_more_bits(i_layer: int, n_layers: int) -> bool:
    """llama-quantize's selective bump rule for ffn_down in _M variants.

    True for layers in the first/last 1/8 + every 3rd among the middle.
    For n_layers=64 this hits 8 + 8 + 16 = 32 of 64 layers --- matches the
    32/32 Q4_K/Q6_K mix in Unsloth's Qwen3.6-27B Q4_K_M GGUF.
    """
    return (
        i_layer < n_layers // 8
        or i_layer >= 7 * n_layers // 8
        or (i_layer - n_layers // 8) % 3 == 2
    )


_LAYER_POSITION_RULES = {
    "use_more_bits": _use_more_bits,
}

_LAYER_POSITION_BUMPS: dict[str, dict[str, tuple[str, str]]] = {
    "q4_k_m": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q4_k_moe": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q5_k_m": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q5_k_moe": {".mlp.down_proj": ("q6_k", "use_more_bits")},
    "q3_k_m": {".mlp.down_proj": ("q5_k", "use_more_bits")},
    # q6_k_xl: selective ffn_down -> Q8_0 via use_more_bits (matches Unsloth
    # UD-Q6_K_XL having ~30% of ffn_down at Q8_0). AP variants inherit the
    # same ffn_down selectivity since the AP floor only changes attention.
    "q6_k_xl": {".mlp.down_proj": ("q8_0", "use_more_bits")},
    "q6_k_xl_ap8": {".mlp.down_proj": ("q8_0", "use_more_bits")},
    "q6_k_xl_ap16": {".mlp.down_proj": ("q8_0", "use_more_bits")},
}

# Per-preset extra BF16 passthrough roles. Merged with the global set in
# `resolve_codec_map`. The AP family uses this to keep linear_attn (and
# any other AP-protected role) at bf16 while still selecting K-quant
# codecs for the rest of the model. Empty / missing -> preset uses the
# global passthrough only.
_PRESET_BF16_PASSTHROUGH: dict[str, frozenset[TensorRole]] = {
    "q4_k_ap": frozenset({TensorRole.LINEAR_ATTN}),
    "q4_k_ap_xl": frozenset({TensorRole.LINEAR_ATTN}),
    "q5_k_ap_xl": frozenset({TensorRole.LINEAR_ATTN}),
    "q6_k_xl_ap16": frozenset({TensorRole.LINEAR_ATTN}),
    "q8_xl_ap16": frozenset({TensorRole.LINEAR_ATTN}),
}


def resolve_codec_map(
    role_map: dict[str, TensorRole],
    *,
    preset: str | None,
    default_codec: str | None,
    overrides: dict[str, str] | None,
) -> dict[str, str | None]:
    """Resolve {path: codec_or_None} for every classified path.

    Priority (highest to lowest):
      1. `overrides[path]` --- explicit per-tensor codec
      2. preset path-suffix bump (e.g., q4_k_xl bumps `.mlp.down_proj` and
         the linear_attn submodules --- these can override the BF16
         passthrough role)
      3. role-based BF16 passthrough (embedding, MoE router, linear_attn)
      4. role-specific preset entry
      5. preset `default`
      6. `--codec` (default_codec) argument
    """
    overrides = overrides or {}
    if preset is not None:
        if preset not in _PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choices: {sorted(_PRESETS)}")
        preset_map = _PRESETS[preset]
    else:
        preset_map = {}
    base_default = preset_map.get("default", default_codec)
    path_bumps = _PATH_BUMPS.get(preset, {}) if preset is not None else {}
    preset_passthrough = (
        _PRESET_BF16_PASSTHROUGH.get(preset, frozenset()) if preset is not None else frozenset()
    )
    effective_passthrough = _BF16_PASSTHROUGH_ROLES | preset_passthrough
    lpos_bumps = _LAYER_POSITION_BUMPS.get(preset, {}) if preset is not None else {}
    # n_layers used by the layer-position rules. Inferred from `.layers.<N>.`
    # in role_map paths so we don't need to thread config in. Hybrid models
    # (Mamba + attn) use a single shared layer numbering, matching the
    # llama-quantize convention of i_layer over the full transformer stack.
    n_layers = 0
    if lpos_bumps:
        for path in role_map:
            idx = extract_layer_idx(path)
            if idx is not None and idx + 1 > n_layers:
                n_layers = idx + 1

    out: dict[str, str | None] = {}
    for path, role in role_map.items():
        if path in overrides:
            out[path] = overrides[path]
            continue
        # 1. Path bumps run FIRST so they can override the BF16 passthrough
        #    default (per-submodule Mamba routing in `_xl` presets).
        matched_bump = False
        for suffix, bumped in path_bumps.items():
            if path.endswith(suffix):
                out[path] = bumped
                matched_bump = True
                break
        if matched_bump:
            continue
        # 2. Layer-position bumps (llama-quantize use_more_bits etc.).
        matched_lpos = False
        for suffix, (lpos_codec, rule_name) in lpos_bumps.items():
            if path.endswith(suffix):
                i_layer = extract_layer_idx(path)
                if (
                    i_layer is not None
                    and n_layers > 0
                    and _LAYER_POSITION_RULES[rule_name](i_layer, n_layers)
                ):
                    out[path] = lpos_codec
                    matched_lpos = True
                break
        if matched_lpos:
            continue
        # 3. Explicit per-role preset entry (any value including None for
        #    bf16). Overrides BF16 passthrough --- that's how AP variants can
        #    keep ATTENTION_OUTPUT bf16 while quantizing everything else.
        if role.value in preset_map:
            out[path] = preset_map[role.value]
            continue
        # 4. BF16 passthrough roles (embedding, MoE router, plus per-preset
        #    extras like linear_attn for AP variants).
        if role in effective_passthrough:
            out[path] = None
            continue
        # 5. Preset's base default (or --codec arg).
        out[path] = base_default
    return out


# ---------- imatrix loading ----------


def load_imatrix(path: str | Path) -> dict[str, np.ndarray]:
    """Load an imatrix file (legacy `.dat` or GGUF) into {name: ndarray}.

    Each ndarray is float32, shape (K,). Tensor names follow the GGUF
    convention as written by `llama-imatrix`; the caller maps them to
    the HF model via `gguf_name_remap`.
    """

    path = Path(path)
    if path.suffix == ".gguf":
        return _load_imatrix_gguf(path)
    return _load_imatrix_dat(path)


def _load_imatrix_dat(path: Path) -> dict[str, np.ndarray]:
    """Legacy llama-imatrix .dat binary format.

    Layout (little-endian, all int32 sizes):
      int32  n_entries
      for each entry:
        int32  name_len
        bytes  name (utf-8, name_len bytes)
        int32  ncall
        int32  nval
        float32[nval] sum_x2_over_ncall
    """
    import numpy as np

    out: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        data = f.read()
    pos = 0
    (n_entries,) = np.frombuffer(data, dtype=np.int32, count=1, offset=pos)
    pos += 4
    for _ in range(int(n_entries)):
        (name_len,) = np.frombuffer(data, dtype=np.int32, count=1, offset=pos)
        pos += 4
        name = data[pos : pos + int(name_len)].decode("utf-8")
        pos += int(name_len)
        # ncall, nval
        _ncall, nval = np.frombuffer(data, dtype=np.int32, count=2, offset=pos)
        pos += 8
        nval = int(nval)
        vals = np.frombuffer(data, dtype=np.float32, count=nval, offset=pos).copy()
        pos += nval * 4
        out[name] = vals
    return out


def _load_imatrix_gguf(path: Path) -> dict[str, np.ndarray]:
    """GGUF imatrix format. Requires the `gguf` package (pip install gguf)."""
    import gguf  # type: ignore[import-not-found]
    import numpy as np

    reader = gguf.GGUFReader(path, "r")
    out: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        out[tensor.name] = np.asarray(tensor.data, dtype=np.float32).copy()
    return out


def map_imatrix_to_hf(
    imatrix: dict[str, np.ndarray],
    hf_paths: list[str],
    arch_string: str | None = None,
) -> dict[str, np.ndarray]:
    """Resolve imatrix tensor names against the HF module paths we'll quantize.

    Tries identity match first (HF-format keys with `.weight` suffix --- what
    `calibrate-mlx-imatrix.py` emits). If that finds nothing, falls back to
    GGUF->HF remap via `gguf_name_remap.parse_gguf_name(arch, name)` (what
    `llama-imatrix` emits).
    """
    hf_set = set(hf_paths)
    identity = {
        k.removesuffix(".weight"): v
        for k, v in imatrix.items()
        if k.removesuffix(".weight") in hf_set
    }
    if identity:
        return identity
    if arch_string is None:
        return {}
    try:
        from mlx_quant_tools.gguf_name_remap import RemapDecision, parse_gguf_name
    except ImportError:
        return {}
    out: dict[str, np.ndarray] = {}
    for gguf_name, vec in imatrix.items():
        decision = parse_gguf_name(arch_string, gguf_name)
        if decision.kind != RemapDecision.KIND_MAP or decision.hf_name is None:
            continue
        hf_path = decision.hf_name.removesuffix(".weight")
        if hf_path in hf_set:
            out[hf_path] = vec
    return out


# ---------- module swap (bf16 source -> kquant-mode QuantizedLinear) ----------


def make_kquant_linear(
    in_dims: int,
    out_dims: int,
    bias: bool,
    codec: str,
    *,
    weight_bytes: mx.array,
    bias_val: mx.array | None = None,
):
    """Construct a kquant-mode `nn.QuantizedLinear` and populate its
    wire-byte weight. Bypasses `QuantizedLinear.__init__`'s `mx.quantize`
    call (which already does the right thing for kquant --- but we go
    around it to avoid re-encoding when the caller already has bytes).
    """
    import mlx.core as mx
    import mlx.nn as nn

    gs, bits, _, _ = CODEC_GEOMETRY[codec]
    layer = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(layer)
    layer.group_size = gs
    layer.bits = bits
    layer.mode = "kquant"
    layer.kquant_type = codec
    layer.weight = weight_bytes
    layer.scales = mx.zeros((1,), dtype=mx.uint8)
    layer.biases = None
    if bias and bias_val is not None:
        layer.bias = bias_val
    layer.freeze()
    return layer


def make_kquant_switch_linear(
    num_experts: int,
    out_dims: int,
    in_dims: int,
    bias: bool,
    codec: str,
    *,
    weight_bytes: mx.array,
    bias_val: mx.array | None = None,
):
    """MoE counterpart to make_kquant_linear. Mirrors KQuantSwitchLinear
    in run-gguf-kquant.py --- just enough structure for safetensors save +
    later in-place reload via install_kquant_modules.
    """
    import mlx.core as mx
    import mlx.nn as nn

    gs, bits, _, _ = CODEC_GEOMETRY[codec]
    layer = nn.Module.__new__(nn.Module)
    nn.Module.__init__(layer)
    layer.group_size = gs
    layer.bits = bits
    layer.mode = "kquant"
    layer.kquant_type = codec
    layer.weight = weight_bytes
    layer.scales = mx.zeros((1,), dtype=mx.uint8)
    layer.biases = None
    if bias and bias_val is not None:
        layer.bias = bias_val
    layer.freeze()
    return layer


def encode_module(
    module,
    codec: str,
    *,
    imatrix_vec: np.ndarray | None = None,
):
    """Run `mx.quantize(mode="kquant")` on `module.weight` and return a
    populated kquant module. Handles both Linear (2D weight) and
    SwitchLinear (3D weight, leading axis = num_experts). Bias preserved
    when present.
    """
    import mlx.core as mx

    w = module.weight
    if w.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise TypeError(f"weight must be float-typed; got {w.dtype} on {type(module).__name__}")
    is_switch = w.ndim == 3
    if is_switch:
        n_experts, out_dims, in_dims = w.shape
    else:
        out_dims, in_dims = w.shape
    gs, bits, _, _ = CODEC_GEOMETRY[codec]
    if in_dims % gs != 0:
        raise ValueError(f"in_dims={in_dims} not divisible by group_size={gs} for codec {codec!r}")
    imatrix_arg = None
    if imatrix_vec is not None:
        if imatrix_vec.shape != (in_dims,):
            raise ValueError(f"imatrix shape {imatrix_vec.shape} does not match in_dims={in_dims}")
        imatrix_arg = mx.array(imatrix_vec, dtype=mx.float32)
    wq, _ = mx.quantize(
        w,
        group_size=gs,
        bits=bits,
        mode="kquant",
        kquant_type=codec,
        imatrix=imatrix_arg,
    )
    mx.eval(wq)
    has_bias = getattr(module, "bias", None) is not None
    bias_val = getattr(module, "bias", None) if has_bias else None
    if is_switch:
        return make_kquant_switch_linear(
            num_experts=n_experts,
            out_dims=out_dims,
            in_dims=in_dims,
            bias=has_bias,
            codec=codec,
            weight_bytes=wq,
            bias_val=bias_val,
        )
    return make_kquant_linear(
        in_dims=in_dims,
        out_dims=out_dims,
        bias=has_bias,
        codec=codec,
        weight_bytes=wq,
        bias_val=bias_val,
    )


def swap_modules(
    model,
    codec_map: dict[str, str | None],
    *,
    imatrix_by_path: dict[str, np.ndarray] | None = None,
    progress: bool = True,
) -> dict[str, str]:
    """Walk `model.named_modules()` and replace each Linear matching
    `codec_map[path]` with a kquant-mode QuantizedLinear. Returns the
    final {path: codec} map (omits skipped/bf16 paths).
    """

    # Pre-compute a path -> module lookup so we can install replacements
    # via setattr on the parent.
    def parent_and_leaf(path: str):
        parts = path.split(".")
        obj = model
        for p in parts[:-1]:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj, parts[-1]

    encoded: dict[str, str] = {}
    n_total = sum(1 for c in codec_map.values() if c is not None)
    n_done = 0
    for path, module in list(model.named_modules()):
        if path not in codec_map:
            continue
        codec = codec_map[path]
        if codec is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim not in (2, 3):
            # Norms, biases, scalar params, etc. left untouched.
            continue
        imatrix_vec = (imatrix_by_path or {}).get(path)
        new_layer = encode_module(module, codec, imatrix_vec=imatrix_vec)
        parent, leaf = parent_and_leaf(path)
        setattr(parent, leaf, new_layer)
        encoded[path] = codec
        n_done += 1
        if progress and (n_done % 25 == 0 or n_done == n_total):
            print(f"[INFO] encoded {n_done}/{n_total} tensors", file=sys.stderr)
    return encoded


# ---------- validation ----------


def validate_roundtrip(
    model,
    src_weights: dict[str, np.ndarray],
    encoded: dict[str, str],
) -> list[dict]:
    """Per-tensor round-trip: dequantize the kquant module's weight bytes
    and compare to the original bf16 source. Returns one dict per tensor:
      {path, codec, max_abs_err, mean_abs_err, rel_max, rel_mean}
    """
    import mlx.core as mx
    import numpy as np

    results: list[dict] = []
    for path, codec in encoded.items():
        gs, bits, _, _ = CODEC_GEOMETRY[codec]
        # Resolve the kquant module --- same path traversal used above.
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        wq = obj.weight
        deq = mx.dequantize(
            wq,
            scales=obj.scales,
            biases=None,
            group_size=gs,
            bits=bits,
            mode="kquant",
            kquant_type=codec,
            dtype=mx.float32,
        )
        mx.eval(deq)
        y = np.asarray(deq).astype(np.float32)
        src = src_weights[path]
        diff = np.abs(y - src)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        denom = max(1e-8, float(np.max(np.abs(src))))
        results.append(
            {
                "path": path,
                "codec": codec,
                "max_abs_err": max_abs,
                "mean_abs_err": mean_abs,
                "rel_max": max_abs / denom,
                "rel_mean": mean_abs / denom,
            }
        )
    return results


# ---------- report ----------


def write_report(
    out_dir: Path,
    codec_map: dict[str, str | None],
    encoded: dict[str, str],
    imatrix_coverage: float | None,
    validation: list[dict] | None,
) -> None:
    """Emit `quantize-kquant-report.{md,json}` next to the saved model."""
    # Per-codec count.
    by_codec: dict[str, int] = {}
    for c in encoded.values():
        by_codec[c] = by_codec.get(c, 0) + 1
    n_bf16 = sum(1 for v in codec_map.values() if v is None)

    report_json = {
        "n_quantized": len(encoded),
        "n_bf16_passthrough": n_bf16,
        "by_codec": dict(sorted(by_codec.items())),
        "per_tensor": dict(sorted(encoded.items())),
    }
    if imatrix_coverage is not None:
        report_json["imatrix_coverage"] = imatrix_coverage
    if validation is not None:
        report_json["validation"] = {
            "max_rel_max": max((r["rel_max"] for r in validation), default=0.0),
            "mean_rel_mean": (sum(r["rel_mean"] for r in validation) / max(1, len(validation))),
            "per_tensor": validation,
        }
    (out_dir / "quantize-kquant-report.json").write_text(json.dumps(report_json, indent=2))

    md = ["# quantize-kquant report", ""]
    md.append(f"- quantized tensors: {len(encoded)}")
    md.append(f"- bf16 pass-through: {n_bf16}")
    md.append("")
    md.append("## By codec")
    md.append("")
    md.append("| codec | count |")
    md.append("|---|---:|")
    for codec, n in sorted(by_codec.items()):
        md.append(f"| {codec} | {n} |")
    md.append("")
    if imatrix_coverage is not None:
        md.append(f"## imatrix coverage: {imatrix_coverage:.1%}")
        md.append("")
        if imatrix_coverage < 0.9:
            md.append("> WARN: coverage below 90% --- many tensors are falling")
            md.append("> back to the self-derived weight path (av_x + |x|).")
            md.append("")
    if validation is not None:
        worst = sorted(validation, key=lambda r: -r["rel_max"])[:10]
        md.append("## validation (top 10 by rel_max)")
        md.append("")
        md.append("| path | codec | rel_max | rel_mean |")
        md.append("|---|---|---:|---:|")
        for r in worst:
            md.append(
                f"| `{r['path']}` | {r['codec']} | {r['rel_max']:.4f} | {r['rel_mean']:.4f} |"
            )
        md.append("")
    (out_dir / "quantize-kquant-report.md").write_text("\n".join(md))


# ---------- CLI ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mqt-quantize-kquant", description="K-quant in-process quantizer for MLX checkpoints."
    )
    p.add_argument(
        "model",
        help="HF repo id or local checkpoint path (text-only models only in v0).",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output directory. Required unless --dry-run.",
    )
    g = p.add_argument_group("codec selection")
    g.add_argument(
        "--codec",
        choices=sorted(ENCODER_CODECS),
        help="Default codec for every quantizable tensor (when no preset).",
    )
    g.add_argument(
        "--preset",
        choices=sorted(_PRESETS),
        help=(
            "Codec preset. Boosts attention/lm_head to a higher codec; FFN "
            "stays at base. Overrides --codec for matched roles."
        ),
    )
    g.add_argument(
        "--per-tensor-codecs",
        help=(
            "Path to a JSON file of {tensor_path: codec} overrides. "
            "Highest priority (overrides preset and --codec)."
        ),
    )
    g.add_argument(
        "--imatrix",
        help="Path to an imatrix .dat or .gguf file (importance weights).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved codec map and exit (no quantization).",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help=(
            "After quantization, round-trip every tensor through "
            "mx.dequantize and report per-tensor error."
        ),
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing quantize-kquant-report.{md,json}.",
    )
    return p


def _check_coverage(
    model,
    role_map: dict[str, TensorRole],
    codec_map: dict[str, str | None],
) -> None:
    """Catch silent classifier gaps and unexpectedly large bf16 fallouts.

    Two failure modes this surfaces:
      1. Module path with `.to_quantized` not in role_map -> likely a
         missing role-classifier regex (e.g. switch_mlp before the fix).
      2. Quantizable mass that ends up bf16 exceeds 30% of model bytes
         AND isn't explained by linear_attn (which is intentionally
         protected on Mamba+attn hybrids per the lab's AP Rule 1).
    """
    classified = set(role_map)
    quantizable_paths: list[tuple[str, int]] = []
    unclassified: list[str] = []
    for path, module in model.named_modules():
        if not hasattr(module, "to_quantized"):
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        numel = 1
        for s in weight.shape:
            numel *= s
        quantizable_paths.append((path, numel))
        if path not in classified:
            unclassified.append(path)

    if unclassified:
        print(
            f"[WARN] {len(unclassified)} module(s) with .to_quantized are not "
            f"classified by model_role_classifier --- they will be left bf16. "
            f"This usually indicates a missing pattern in classify_path. "
            f"Examples: {unclassified[:3]}",
            file=sys.stderr,
        )

    total_mass = sum(n for _, n in quantizable_paths)
    if total_mass == 0:
        return
    bf16_mass = 0
    linear_attn_mass = 0
    for path, numel in quantizable_paths:
        codec = codec_map.get(path)
        if codec is not None:
            continue
        bf16_mass += numel
        role = role_map.get(path)
        if role is not None and role.value == "linear_attn":
            linear_attn_mass += numel
    bf16_frac = bf16_mass / total_mass
    explained_frac = linear_attn_mass / total_mass
    if bf16_frac > 0.30:
        unexplained = bf16_frac - explained_frac
        print(
            f"[WARN] {bf16_frac:.0%} of quantizable mass is bf16 pass-through "
            f"({explained_frac:.0%} linear_attn). "
            f"Unexplained bf16 fraction: {unexplained:.0%}. "
            f"Check for unclassified routed experts or other missed roles.",
            file=sys.stderr,
        )


def main() -> None:
    args = build_parser().parse_args()

    if args.preset is None and args.codec is None:
        sys.exit("specify --preset or --codec")
    if not args.dry_run and args.output is None:
        sys.exit("--output is required unless --dry-run")

    overrides: dict[str, str] | None = None
    if args.per_tensor_codecs is not None:
        overrides = json.loads(Path(args.per_tensor_codecs).read_text())
        bad = {p: c for p, c in overrides.items() if c not in ENCODER_CODECS}
        if bad:
            sys.exit(f"--per-tensor-codecs contains unknown codecs: {sorted(set(bad.values()))}")

    import mlx.core as mx
    from mlx_lm.utils import load, save

    print(f"[INFO] loading {args.model}", file=sys.stderr)
    model, tokenizer, src_config = load(args.model, return_config=True)

    role_map = classify_tensors(model)
    codec_map = resolve_codec_map(
        role_map,
        preset=args.preset,
        default_codec=args.codec,
        overrides=overrides,
    )

    _check_coverage(model, role_map, codec_map)

    imatrix_by_path: dict[str, np.ndarray] | None = None
    imatrix_coverage: float | None = None
    if args.imatrix is not None:
        print(f"[INFO] loading imatrix {args.imatrix}", file=sys.stderr)
        raw = load_imatrix(args.imatrix)
        arch_string = (
            src_config.get("model_type")
            if isinstance(src_config, dict)
            else getattr(src_config, "model_type", None)
        )
        # Map GGUF->HF names; restrict to paths we're about to quantize.
        quant_paths = [p for p, c in codec_map.items() if c is not None]
        imatrix_by_path = map_imatrix_to_hf(raw, quant_paths, arch_string=arch_string)
        imatrix_coverage = len(imatrix_by_path) / max(1, len(quant_paths))
        print(
            f"[INFO] imatrix coverage: {len(imatrix_by_path)}/{len(quant_paths)} "
            f"({imatrix_coverage:.1%})",
            file=sys.stderr,
        )
        if imatrix_coverage < 0.9:
            print(
                "[WARN] imatrix coverage below 90% --- affected tensors will "
                "fall back to self-derived weights.",
                file=sys.stderr,
            )

    if args.dry_run:
        print("# resolved codec map")
        for path, codec in sorted(codec_map.items()):
            role = role_map.get(path, "?")
            tag = "bf16" if codec is None else codec
            has_imat = imatrix_by_path is not None and path in imatrix_by_path
            imat_tag = " +imatrix" if has_imat else ""
            print(f"{tag:6} {role.value if hasattr(role, 'value') else role:25} {path}{imat_tag}")
        return

    # Stash source weights (as fp32 for clean numpy round-trip) so
    # --validate can compare round-trip error per tensor.
    src_weights: dict[str, mx.array] = {}
    if args.validate:
        import numpy as np

        for path, module in model.named_modules():
            if path not in codec_map or codec_map[path] is None:
                continue
            w = getattr(module, "weight", None)
            if w is None or w.ndim not in (2, 3):
                continue
            w_fp32 = w.astype(mx.float32)
            mx.eval(w_fp32)
            src_weights[path] = np.asarray(w_fp32).copy()

    encoded = swap_modules(model, codec_map, imatrix_by_path=imatrix_by_path)
    print(
        f"[INFO] encoded {len(encoded)} tensors; "
        f"{sum(1 for v in codec_map.values() if v is None)} pass-through",
        file=sys.stderr,
    )

    # Embed the kquant recipe in config.json so loaders can interpret the
    # safetensors uint8 tensors. mlx_lm.utils.save reads quantization_config
    # off the config dict; we drop a `mode = "kquant"` next to a per_tensor
    # map. The mlx-quant-lab load path uses run-gguf-kquant.py's helpers.
    src_config = dict(src_config)
    src_config["quantization_config"] = {
        "mode": "kquant",
        "per_tensor": encoded,
    }

    validation: list[dict] | None = None
    if args.validate:
        print("[INFO] running round-trip validation", file=sys.stderr)
        validation = validate_roundtrip(model, src_weights, encoded)
        max_rel = max(r["rel_max"] for r in validation)
        print(f"[INFO] worst rel error: {max_rel:.4f}", file=sys.stderr)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save(out_dir, args.model, model, tokenizer, src_config, donate_model=True)
    print(f"[INFO] saved to {out_dir}", file=sys.stderr)

    if not args.no_report:
        write_report(out_dir, codec_map, encoded, imatrix_coverage, validation)
        print(f"[INFO] wrote {out_dir}/quantize-kquant-report.{{md,json}}", file=sys.stderr)


if __name__ == "__main__":
    main()
