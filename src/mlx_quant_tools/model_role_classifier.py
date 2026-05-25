"""Pure tensor role classifier for transformer / MoE / VLM checkpoints.

Walks `model.named_modules()` and labels each quantizable module by its
semantic role (attention QKVO, MoE router, FFN projection, etc). Pure
classification — no policy. Callers map roles to (bits, group_size, codec)
per their recipe.

Public API
----------
- `TensorRole`           — string enum of role labels.
- `classify_path(path)`  — single path → leaf role (or None).
- `classify_tensors(model)` — `model.named_modules()` → {path: role}.
- `is_vlm_tower(path)`   — orthogonal hierarchical check; True for any
                           path inside a vision/audio tower or projector.
- `extract_layer_idx(path)` — `.layers.<N>.` index extractor.
- `extract_layer_types(config)` — pull `layer_types` from HF config.

Role conventions
----------------
- Roles are leaf-level — VLM_TOWER is only returned for paths *inside* a
  VLM tower that do not match any specific inner role. For inner paths
  like `vision_tower.encoder.layer.0.self_attn.q_proj`, the role is
  ATTENTION_QKVO and `is_vlm_tower(path) == True`.
- An unmatched quantizable path returns role=None. Caller decides the
  default policy.
"""

from __future__ import annotations

import re
from enum import Enum


class TensorRole(str, Enum):
    LINEAR_ATTN = "linear_attn"
    ATTENTION_OUTPUT = "attention_output"  # `attn.out_proj` (non-self-attn)
    ATTENTION_QKVO = "attention_qkvo"  # `self_attn.{q,k,v,o}_proj`
    LM_HEAD = "lm_head"
    EMBEDDING = "embedding"  # `embed_tokens` / `wte`
    EMBEDDING_PER_LAYER = "embedding_per_layer"  # Gemma-4 PLE
    VLM_TOWER = "vlm_tower"  # leaf-only when no inner match
    MOE_ROUTER = "moe_router"  # router gates, MoE only
    FFN = "ffn"  # `mlp.{gate,up,down}_proj`
    SHARED_EXPERT = "shared_expert"  # `shared_expert.{...}_proj`
    ROUTED_EXPERT = "routed_expert"  # per-expert or SwitchGLU MoE


# Regex patterns. Anchored end-of-path ($) where the role is identifiable
# by the leaf module name; segment-prefix anchored where the role is
# determined by an ancestor segment (VLM tower).

_LINEAR_ATTN = re.compile(r"\.linear_attn\.[^.]+$")

# Rule 2 explicitly excludes `self_attn.out_proj` via the negative
# lookbehind; that case is owned by ATTENTION_QKVO.
_ATTN_OUT_PROJ = re.compile(r"(?<!self_)\battn\.out_proj$")

_QKVO = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")

# Per-layer index extractor for sliding-vs-full layer-type overrides.
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")

_LM_HEAD = re.compile(r"(^|\.)lm_head$")

# `\bembed_tokens\b` does NOT match `embed_tokens_per_layer` — `_` is a
# word character so the trailing `\b` fails between `s` and `_`. The
# per-layer variant has its own matcher.
_EMBED_TOKENS = re.compile(r"\b(embed_tokens|wte|word_embeddings)\b")
_EMBED_PER_LAYER = re.compile(r"\bembed_tokens_per_layer\b")

# Segment-prefix matcher: any path containing one of these tower segments
# as a name component. Used hierarchically — a VLM tower's *inner* paths
# may also match leaf-role patterns (e.g. its own self_attn.q_proj); the
# `is_vlm_tower` predicate is orthogonal to `classify_path`.
_VLM_TOWER = re.compile(
    r"(^|\.)("
    r"vision_tower|vision_model|visual"
    r"|audio_tower|audio_model"
    r"|multi_modal_projector|vl_connector"
    r"|merger|connector"
    r"|embed_vision|embed_audio"
    r")\."
)

# MoE routers and per-expert gates.
#   - `router.proj`        Gemma-4 MoE router
#   - `mlp.gate`           Qwen3-MoE router (end-of-path; does NOT match
#                          `mlp.gate_proj`, the dense SwiGLU gate)
#   - `shared_expert_gate` Qwen3-MoE per-token shared-expert mix scalar
_MOE_ROUTER = re.compile(
    r"(\.router\.proj"
    r"|\.mlp\.gate"
    r"|\.shared_expert_gate)$"
)

# Dense MLP and shared-expert dense projections — kept as separate
# patterns so the caller can apply different codecs per role. Both end
# in `(gate|up|down)_proj` but differ in the parent segment.
_FFN = re.compile(r"\.mlp\.(gate|up|down)_proj$")
_SHARED_EXPERT = re.compile(r"\.shared_expert\.(gate|up|down)_proj$")

# Routed experts — three layouts:
#   - SwitchGLU/SwitchLinear (`experts.switch_glu.{...}_proj`) packs all
#     experts in one 3D tensor (leading axis = num_experts). Older mlx-lm.
#   - SwitchMLP (`mlp.switch_mlp.{...}_proj`) — 3D tensor too, leading axis
#     = num_experts. Used by Qwen3-MoE / Qwen3.6-MoE in mlx-lm 0.31+.
#   - Per-expert nn.Linear (`mlp.experts.<i>.{...}_proj`).
_ROUTED_EXPERTS = re.compile(
    r"\.experts\.switch_glu\.(gate|up|down)_proj$"
    r"|\.mlp\.switch_mlp\.(gate|up|down)_proj$"
    r"|\.mlp\.experts\.\d+\.(gate|up|down)_proj$"
)


def classify_path(path: str) -> TensorRole | None:
    """Map a single module path to its leaf TensorRole, or None.

    Returns the most-specific leaf role. Does NOT check VLM hierarchy —
    use `is_vlm_tower(path)` for that orthogonal predicate.
    """
    if _LINEAR_ATTN.search(path):
        return TensorRole.LINEAR_ATTN
    if _ATTN_OUT_PROJ.search(path):
        return TensorRole.ATTENTION_OUTPUT
    if _ROUTED_EXPERTS.search(path):
        return TensorRole.ROUTED_EXPERT
    if _MOE_ROUTER.search(path):
        return TensorRole.MOE_ROUTER
    if _QKVO.search(path):
        return TensorRole.ATTENTION_QKVO
    if _LM_HEAD.search(path):
        return TensorRole.LM_HEAD
    if _EMBED_PER_LAYER.search(path):
        return TensorRole.EMBEDDING_PER_LAYER
    if _EMBED_TOKENS.search(path):
        return TensorRole.EMBEDDING
    if _SHARED_EXPERT.search(path):
        return TensorRole.SHARED_EXPERT
    if _FFN.search(path):
        return TensorRole.FFN
    # Fall-through: VLM tower path with no leaf-role match.
    if _VLM_TOWER.search(path):
        return TensorRole.VLM_TOWER
    return None


def is_vlm_tower(path: str) -> bool:
    """Return True iff the path lies inside a vision/audio tower or
    multi-modal projector. Orthogonal to `classify_path` — a VLM
    tower's `self_attn.q_proj` is both ATTENTION_QKVO and is_vlm_tower.
    """
    return bool(_VLM_TOWER.search(path))


def classify_tensors(model) -> dict[str, TensorRole]:
    """Walk `model.named_modules()` and return {path: role} for every
    quantizable module (those exposing `to_quantized`).

    Only paths with a non-None role are included. Caller's default
    policy applies to omitted paths.
    """
    result: dict[str, TensorRole] = {}
    for path, module in model.named_modules():
        if not hasattr(module, "to_quantized"):
            continue
        role = classify_path(path)
        if role is not None:
            result[path] = role
    return result


def extract_layer_idx(path: str) -> int | None:
    """Extract the integer layer index from a path like
    `model.layers.7.self_attn.q_proj`. None when no `.layers.N.` segment.
    """
    m = _LAYER_IDX_RE.search(path)
    if m is None:
        return None
    return int(m.group(1))


def extract_layer_types(config) -> list[str] | None:
    """Read `layer_types` from a HF config dict or AutoConfig namespace.

    Used by callers that need to distinguish sliding vs full attention
    layers (Gemma-3/4 layouts). Returns None when the config lacks this
    field.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("layer_types")
    return getattr(config, "layer_types", None)
