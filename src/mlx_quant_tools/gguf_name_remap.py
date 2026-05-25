"""GGUF tensor name → HF tensor name remap (pure string/decision logic).

Shared by ``mqt-dequant-gguf``, ``mqt-run-gguf``, and other GGUF tools.
This module is intentionally numpy/mlx-free: it only inspects names and
decides what to do. Callers own the array manipulation (dequant, layout
transforms, wire-byte splits, etc.).
"""

from __future__ import annotations

import re

# Naming policy / why each lookup layer exists:
#
# - GGUF stores tensors with names like `blk.0.attn_q.weight`. HF/MLX checkpoints
#   want `model.layers.0.self_attn.q_proj.weight`. Remap is per-arch.
# - `gguf-py.constants.TENSOR_NAMES` provides the canonical GGUF→enum direction
#   for most tensors. We invert it in `_gguf_to_enum()`. Then `CANONICAL_HF`
#   maps the enum to the HF target template.
# - Some archs (gemma-4-MoE) use names that diverge from HF stock or have
#   `.scale` tensors that the universal Unsloth-UD `.scale`-skip would drop.
#   `ARCH_PRIORITY_OVERRIDES` runs FIRST so an arch can claim those before
#   the universal skip / canonical lookup.
# - Some archs (gemma-4 dual-FFN norms) have tensors not in upstream
#   gguf-py templates — `EXTRA_OVERRIDES` is the per-arch fallback.
# - Tensors not mapped by any TensorNameMap entry — hard-fail (genuine unknown).


# GGUF arch string → MODEL_ARCH enum used for TENSOR_NAMES lookup.
# gguf-py at the version pinned in vllm-mlx has no GEMMA4 enum yet; gemma-4 GGUFs
# advertise general.architecture='gemma4' but reuse GEMMA3's tensor templates plus
# a few extra dual-FFN norms (handled via EXTRA_OVERRIDES below).
ARCH_ALIAS = {
    "gemma4": "GEMMA3",
    "gemma3": "GEMMA3",
    "gemma3n": "GEMMA3N",
    "qwen3": "QWEN3",
    "qwen3moe": "QWEN3MOE",
    "qwen35": "QWEN35",
    "qwen35moe": "QWEN35MOE",
    "llama": "LLAMA",
    # llama.cpp's 'mistral3' arch (Ministral-3 / Mistral-Small-3.1) uses the
    # canonical Llama tensor layout (q/k permuted at convert time, ffn_norm
    # = post_attention_layernorm). The LLAMA arch overrides + qk_permute
    # transform apply unchanged.
    "mistral3": "LLAMA",
    "nemotron_h_moe": "NEMOTRON_H_MOE",
}

# HF stock name (with `model.` prefix and `{bid}` placeholder) per MODEL_TENSOR.
# Names follow the de-facto HF Llama-family layout. mlx_lm-specific renames
# (e.g., `experts.switch_glu.*`) are NOT applied here; the loader emits an
# HF-format checkpoint that downstream tooling (`mlx_lm.convert`,
# `mlx_vlm.utils.convert`) can transform if/when needed.
CANONICAL_HF = {
    "TOKEN_EMBD": "model.embed_tokens.weight",
    "OUTPUT": "lm_head.weight",
    "OUTPUT_NORM": "model.norm.weight",
    "ROPE_FREQS": None,  # never serialized; skip
    "ATTN_NORM": "model.layers.{bid}.input_layernorm.weight",
    "ATTN_POST_NORM": "model.layers.{bid}.post_attention_layernorm.weight",
    "ATTN_Q": "model.layers.{bid}.self_attn.q_proj.weight",
    "ATTN_K": "model.layers.{bid}.self_attn.k_proj.weight",
    "ATTN_V": "model.layers.{bid}.self_attn.v_proj.weight",
    "ATTN_OUT": "model.layers.{bid}.self_attn.o_proj.weight",
    "ATTN_Q_NORM": "model.layers.{bid}.self_attn.q_norm.weight",
    "ATTN_K_NORM": "model.layers.{bid}.self_attn.k_norm.weight",
    "FFN_NORM": "model.layers.{bid}.post_attention_layernorm.weight",
    "FFN_PRE_NORM": "model.layers.{bid}.pre_feedforward_layernorm.weight",
    "FFN_POST_NORM": "model.layers.{bid}.post_feedforward_layernorm.weight",
    "FFN_GATE": "model.layers.{bid}.mlp.gate_proj.weight",
    "FFN_UP": "model.layers.{bid}.mlp.up_proj.weight",
    "FFN_DOWN": "model.layers.{bid}.mlp.down_proj.weight",
    "FFN_GATE_INP": "model.layers.{bid}.mlp.gate.weight",  # MoE router
    "FFN_GATE_EXP": "model.layers.{bid}.mlp.experts.gate_proj.weight",
    "FFN_UP_EXP": "model.layers.{bid}.mlp.experts.up_proj.weight",
    "FFN_DOWN_EXP": "model.layers.{bid}.mlp.experts.down_proj.weight",
    "FFN_GATE_UP_EXP": "model.layers.{bid}.mlp.experts.gate_up_proj.weight",  # fused (gemma-4)
    # Gemma-3/4 per-layer features:
    "LAYER_OUT_SCALE": "model.layers.{bid}.layer_scalar",
    "PER_LAYER_TOKEN_EMBD": "model.embed_tokens_per_layer.weight",
    "PER_LAYER_MODEL_PROJ": "model.per_layer_model_projection.weight",
    "PER_LAYER_PROJ_NORM": "model.per_layer_projection_norm.weight",
    "PER_LAYER_INP_GATE": "model.layers.{bid}.per_layer_input_gate.weight",
    "PER_LAYER_PROJ": "model.layers.{bid}.per_layer_projection.weight",
    "PER_LAYER_POST_NORM": "model.layers.{bid}.post_per_layer_input_norm.weight",
    # Qwen3.5/3.6 hybrid Mamba+attn (linear_attn-housed) features:
    "ATTN_QKV": "model.layers.{bid}.linear_attn.in_proj_qkv.weight",
    "ATTN_GATE": "model.layers.{bid}.linear_attn.in_proj_z.weight",
    "SSM_IN": "model.layers.{bid}.linear_attn.in_proj.weight",
    "SSM_A": "model.layers.{bid}.linear_attn.A_log",
    "SSM_ALPHA": "model.layers.{bid}.linear_attn.in_proj_a.weight",
    "SSM_BETA": "model.layers.{bid}.linear_attn.in_proj_b.weight",
    "SSM_CONV1D": "model.layers.{bid}.linear_attn.conv1d.weight",
    # GGUF SSM_DT is stored as a 1D bias ('blk.{N}.ssm_dt.bias', shape (48,)).
    # MLX names it `dt_bias`, not `dt_proj.weight` — it's a bias vector, not a projection.
    "SSM_DT": "model.layers.{bid}.linear_attn.dt_bias",
    "SSM_NORM": "model.layers.{bid}.linear_attn.norm.weight",
    "SSM_OUT": "model.layers.{bid}.linear_attn.out_proj.weight",
}

# Per-arch overrides for tensors not in `gguf.constants.TENSOR_NAMES` —
# typically architecture quirks not yet upstreamed in gguf-py.
#
# Each entry is a regex matching the GGUF name (sans `.weight` suffix; see
# parse_gguf_name) -> HF format string with optional {bid} placeholder.
# Order matters: the first matching pattern wins.
EXTRA_OVERRIDES: dict[str, list[tuple[re.Pattern, str | None]]] = {
    "GEMMA3": [
        # Gemma-4 dual-FFN extra norms (not in upstream gguf-py templates):
        (
            re.compile(r"^blk\.(\d+)\.post_ffw_norm_1$"),
            "model.layers.{bid}.post_feedforward_layernorm_1.weight",
        ),
        (
            re.compile(r"^blk\.(\d+)\.post_ffw_norm_2$"),
            "model.layers.{bid}.post_feedforward_layernorm_2.weight",
        ),
        (
            re.compile(r"^blk\.(\d+)\.pre_ffw_norm_2$"),
            "model.layers.{bid}.pre_feedforward_layernorm_2.weight",
        ),
        (
            re.compile(r"^blk\.(\d+)\.layer_output_scale$"),
            "model.layers.{bid}.layer_scalar",
        ),  # note: no .weight suffix in MLX
    ],
}

# Arch-priority overrides — checked BEFORE the universal `.scale`-skip and
# canonical TENSOR_NAMES lookup. Lets an arch (a) redirect a tensor to a name
# that diverges from the HF/Llama stock layout (e.g., gemma-4-MoE's
# `router.proj` vs HF-stock `mlp.gate`) and (b) keep architectural `.scale`
# tensors that the universal Unsloth-UD skip would otherwise drop.
#
# Entry: (pattern, hf_format_string, transform). transform=`passthrough` for
# pure renames; `moe_split_gate_up` for fused gate_up split; etc.
#
# For gemma-4-MoE specifically, MLX's mlx-vlm gemma-4 implementation uses:
#   model.layers.{N}.experts.switch_glu.{gate,up,down}_proj.weight
#   model.layers.{N}.router.proj.weight     (GGUF: blk.{N}.ffn_gate_inp.weight)
#   model.layers.{N}.router.scale           (GGUF: blk.{N}.ffn_gate_inp.scale)
#   model.layers.{N}.router.per_expert_scale  (GGUF: blk.{N}.ffn_down_exps.scale)
#
# Other `.scale` tensors in Unsloth UD GGUFs are recipe metadata and continue
# to be skipped by `_is_unsloth_ud_scale`.
#
# For LLAMA arch, FFN_NORM is shadowed by FFN_PRE_NORM in the global reverse-map
# (TENSOR_NAMES collision: both enums use `blk.{bid}.ffn_norm`). The override
# below claims `blk.{N}.ffn_norm.weight` first so Llama gets the right HF target
# (post_attention_layernorm); without it, Llama would get pre_feedforward_layernorm,
# which silently produces a misnamed checkpoint.
ARCH_PRIORITY_OVERRIDES: dict[str, list[tuple[re.Pattern, str | None, str]]] = {
    # FFN_NORM/FFN_PRE_NORM TENSOR_NAMES collision: both enums use
    # `blk.{bid}.ffn_norm`, so the global reverse-map's resolution depends on
    # gguf-py enum ordering. Both archs that consume `ffn_norm` get an
    # explicit override below so a gguf-py version bump can't silently flip
    # the mapping.
    "GEMMA3": [
        # Router (architectural .scale must be claimed before universal skip).
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
            "model.layers.{bid}.router.proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.scale$"),
            "model.layers.{bid}.router.scale",
            "passthrough",
        ),
        # Per-expert routing scale: Unsloth UD GGUFs ship this under the
        # `ffn_down_exps.scale` name (verified bit-exact against the upstream
        # `router.per_expert_scale` bf16 tensor). Must be claimed before the
        # universal `.scale` skip; without it the student loads with the
        # tensor missing and routing is silently mis-scaled.
        (
            re.compile(r"^blk\.(\d+)\.ffn_down_exps\.scale$"),
            "model.layers.{bid}.router.per_expert_scale",
            "passthrough",
        ),
        # MoE expert weights — namespace divergence: HF stock uses `mlp.experts.*`,
        # mlx-vlm gemma-4 uses `experts.switch_glu.*`. Override the canonical map
        # so --out-dir produces a directly mlx_lm.load-able checkpoint.
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_up_exps\.weight$"),
            "model.layers.{bid}.experts.switch_glu.gate_up_proj.weight",
            "moe_split_gate_up",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
            "model.layers.{bid}.experts.switch_glu.down_proj.weight",
            "passthrough",
        ),
        # FFN_PRE_NORM is what gemma-3/4 wants for `blk.{bid}.ffn_norm`. Pin it
        # explicitly so the resolution doesn't drift on a gguf-py upgrade.
        (
            re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
            "model.layers.{bid}.pre_feedforward_layernorm.weight",
            "passthrough",
        ),
    ],
    "LLAMA": [
        # FFN_NORM (post_attention_layernorm) — Llama has no separate pre-FFN norm;
        # the post-attn norm IS the pre-FFN norm.
        (
            re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
            "model.layers.{bid}.post_attention_layernorm.weight",
            "passthrough",
        ),
    ],
    "QWEN3": [
        (
            re.compile(r"^blk\.(\d+)\.ffn_norm\.weight$"),
            "model.layers.{bid}.post_attention_layernorm.weight",
            "passthrough",
        ),
    ],
    "QWEN35MOE": [
        # Routed experts — GGUF has separate gate/up/down (not fused like gemma-4).
        # Model-native path is mlp.switch_mlp.*, not mlp.experts.* (which would
        # need qwen3_5_moe.Model.sanitize to rename, but we build TextModel
        # directly to avoid the language_model. prefix).
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$"),
            "model.layers.{bid}.mlp.switch_mlp.gate_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
            "model.layers.{bid}.mlp.switch_mlp.up_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
            "model.layers.{bid}.mlp.switch_mlp.down_proj.weight",
            "passthrough",
        ),
        # Shared expert — not in gguf-py TENSOR_NAMES.
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_shexp\.weight$"),
            "model.layers.{bid}.mlp.shared_expert.gate_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
            "model.layers.{bid}.mlp.shared_expert.up_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
            "model.layers.{bid}.mlp.shared_expert.down_proj.weight",
            "passthrough",
        ),
        # Shared expert gate: 1D [hidden_size] in GGUF → nn.Linear(dim, 1)
        # expects [1, hidden_size]. Needs unsqueeze.
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_inp_shexp\.weight$"),
            "model.layers.{bid}.mlp.shared_expert_gate.weight",
            "gate_1d_unsqueeze",
        ),
        # Router — explicit claim for self-containment.
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
            "model.layers.{bid}.mlp.gate.weight",
            "passthrough",
        ),
    ],
    "NEMOTRON_H_MOE": [
        # nemotron_h uses backbone.layers prefix and unified `mixer` module name
        # for all block types (attention, mamba, MoE, dense MLP). Every tensor
        # must be claimed here because CANONICAL_HF uses model.layers which is
        # wrong for this arch.
        #
        # Global tensors
        (re.compile(r"^token_embd\.weight$"), "backbone.embeddings.weight", "passthrough"),
        (re.compile(r"^output_norm\.weight$"), "backbone.norm_f.weight", "passthrough"),
        (re.compile(r"^output\.weight$"), "lm_head.weight", "passthrough"),
        # Per-layer norm (single norm per block, regardless of block type)
        (
            re.compile(r"^blk\.(\d+)\.attn_norm\.weight$"),
            "backbone.layers.{bid}.norm.weight",
            "passthrough",
        ),
        # Attention (Q/K permuted by convert_hf_to_gguf via GraniteHybridModel)
        (
            re.compile(r"^blk\.(\d+)\.attn_q\.weight$"),
            "backbone.layers.{bid}.mixer.q_proj.weight",
            "qk_permute",
        ),
        (
            re.compile(r"^blk\.(\d+)\.attn_k\.weight$"),
            "backbone.layers.{bid}.mixer.k_proj.weight",
            "qk_permute",
        ),
        (
            re.compile(r"^blk\.(\d+)\.attn_v\.weight$"),
            "backbone.layers.{bid}.mixer.v_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.attn_output\.weight$"),
            "backbone.layers.{bid}.mixer.o_proj.weight",
            "passthrough",
        ),
        # Mamba2 SSM
        (
            re.compile(r"^blk\.(\d+)\.ssm_in\.weight$"),
            "backbone.layers.{bid}.mixer.in_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ssm_conv1d\.weight$"),
            "backbone.layers.{bid}.mixer.conv1d.weight",
            "conv1d_unsqueeze",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ssm_conv1d\.bias$"),
            "backbone.layers.{bid}.mixer.conv1d.bias",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ssm_dt\.bias$"),
            "backbone.layers.{bid}.mixer.dt_bias",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ssm_a$"),
            "backbone.layers.{bid}.mixer.A_log",
            "ssm_a_to_a_log",
        ),
        (re.compile(r"^blk\.(\d+)\.ssm_d$"), "backbone.layers.{bid}.mixer.D", "flatten"),
        (
            re.compile(r"^blk\.(\d+)\.ssm_norm\.weight$"),
            "backbone.layers.{bid}.mixer.norm.weight",
            "flatten",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ssm_out\.weight$"),
            "backbone.layers.{bid}.mixer.out_proj.weight",
            "passthrough",
        ),
        # MoE router + expert correction bias
        (
            re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$"),
            "backbone.layers.{bid}.mixer.gate.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.exp_probs_b\.bias$"),
            "backbone.layers.{bid}.mixer.gate.e_score_correction_bias",
            "passthrough",
        ),
        # MoE routed experts (stacked: shape [intermediate, hidden, n_experts])
        (
            re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$"),
            "backbone.layers.{bid}.mixer.switch_mlp.fc1.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$"),
            "backbone.layers.{bid}.mixer.switch_mlp.fc2.weight",
            "passthrough",
        ),
        # MoE latent projections (dimensionality reduction before/after experts)
        (
            re.compile(r"^blk\.(\d+)\.ffn_latent_down\.weight$"),
            "backbone.layers.{bid}.mixer.fc1_latent_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_latent_up\.weight$"),
            "backbone.layers.{bid}.mixer.fc2_latent_proj.weight",
            "passthrough",
        ),
        # MoE shared expert
        (
            re.compile(r"^blk\.(\d+)\.ffn_up_shexp\.weight$"),
            "backbone.layers.{bid}.mixer.shared_experts.up_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_down_shexp\.weight$"),
            "backbone.layers.{bid}.mixer.shared_experts.down_proj.weight",
            "passthrough",
        ),
        # Dense MLP (non-MoE layers, if present in the pattern)
        (
            re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
            "backbone.layers.{bid}.mixer.up_proj.weight",
            "passthrough",
        ),
        (
            re.compile(r"^blk\.(\d+)\.ffn_down\.weight$"),
            "backbone.layers.{bid}.mixer.down_proj.weight",
            "passthrough",
        ),
    ],
}

# GGUF tensor-name prefixes that indicate vision/audio tower content. These
# are skipped (not hard-failed); loading them into MLX needs mlx_vlm.utils
# plumbing well outside this loader's scope.
VLM_PREFIXES = (
    "v.",  # vision tower (per gguf TENSOR_NAMES convention: V_*)
    "v_enc.",
    "vision_tower.",
    "mm.",  # multi-modal projector
    "a.",  # audio tower (A_*)
    "audio_tower.",
    "resampler.",
)


# Unsloth UD `.scale` tensors are metadata for Unsloth's MLX recipe (per-tensor
# additional scaling on top of the base K-quant). They have no HF/MLX-stock
# counterpart; skip with warning.
def _is_unsloth_ud_scale(name: str) -> bool:
    return name.endswith(".scale")


# Module-level cache: {gguf_name_template: MODEL_TENSOR_enum}. The reverse of
# `gguf.constants.TENSOR_NAMES` is global — TENSOR_NAMES is a single dict[enum,
# name] shared across archs, so there's nothing arch-specific to cache. Per-arch
# routing is handled by ARCH_PRIORITY_OVERRIDES (which runs first) and CANONICAL_HF
# (the post-lookup HF/MLX target naming). Built lazily on first call to keep the
# `from gguf.constants import TENSOR_NAMES` import cost out of module-load time.
_GGUF_TO_ENUM_CACHE: dict[str, object] | None = None


def _gguf_to_enum() -> dict[str, object]:
    global _GGUF_TO_ENUM_CACHE
    if _GGUF_TO_ENUM_CACHE is None:
        from gguf.constants import TENSOR_NAMES

        _GGUF_TO_ENUM_CACHE = {name: enum for enum, name in TENSOR_NAMES.items()}
    return _GGUF_TO_ENUM_CACHE


class RemapDecision:
    __slots__ = ("kind", "hf_name", "transform", "reason", "bid")
    KIND_MAP = "map"  # produce HF tensor (possibly with transform)
    KIND_SKIP = "skip"  # log warning + drop from output
    KIND_FAIL = "fail"  # hard error: unrecognized tensor

    def __init__(
        self,
        kind: str,
        *,
        hf_name: str | None = None,
        transform: str = "passthrough",
        reason: str = "",
        bid: int | None = None,
    ):
        self.kind = kind
        self.hf_name = hf_name
        self.transform = (
            transform  # passthrough | qk_permute | moe_split_gate_up | conv1d_unsqueeze
        )
        self.reason = reason
        self.bid = bid


def parse_gguf_name(arch_string: str, gguf_name: str) -> RemapDecision:
    """Decide what to do with a single GGUF tensor name."""
    # VLM/audio tower skip — universal, no arch lookup needed.
    if any(gguf_name.startswith(p) for p in VLM_PREFIXES):
        return RemapDecision(RemapDecision.KIND_SKIP, reason="vision/audio tower (prefix match)")

    arch_alias = ARCH_ALIAS.get(arch_string)

    # Arch-priority overrides — run BEFORE the universal .scale skip so an
    # arch can claim architectural .scale tensors (e.g., gemma-4-MoE router).
    if arch_alias is not None:
        for pat, hf_fmt, transform in ARCH_PRIORITY_OVERRIDES.get(arch_alias, []):
            m = pat.match(gguf_name)
            if m:
                if hf_fmt is None:
                    return RemapDecision(
                        RemapDecision.KIND_SKIP, reason=f"arch-priority drops {gguf_name!r}"
                    )
                bid = int(m.group(1)) if m.groups() else None
                hf_name = hf_fmt.format(bid=bid) if bid is not None else hf_fmt
                return RemapDecision(
                    RemapDecision.KIND_MAP, hf_name=hf_name, transform=transform, bid=bid
                )

    # Universal Unsloth-UD .scale skip (after arch-priority claims).
    if _is_unsloth_ud_scale(gguf_name):
        return RemapDecision(RemapDecision.KIND_SKIP, reason="Unsloth UD .scale metadata tensor")

    # Strip the trailing ".weight" / ".bias" if present, since
    # `TENSOR_NAMES` templates omit it.
    base = gguf_name
    for s in (".weight", ".bias"):
        if base.endswith(s):
            base = base[: -len(s)]
            break

    if arch_alias is None:
        return RemapDecision(
            RemapDecision.KIND_SKIP,
            reason=f"arch {arch_string!r} has no remap support; --no-remap or extend ARCH_ALIAS",
        )

    # Try the canonical TENSOR_NAMES reverse map. The templates use {bid};
    # match by pattern.
    rev = _gguf_to_enum()
    bid: int | None = None
    matched_enum = None
    for tmpl, enum in rev.items():
        if "{bid}" in tmpl:
            # Replace {bid} with capture group, anchor full match.
            pat = "^" + re.escape(tmpl).replace(r"\{bid\}", r"(\d+)") + "$"
            m = re.match(pat, base)
            if m:
                bid = int(m.group(1))
                matched_enum = enum
                break
        else:
            if tmpl == base:
                matched_enum = enum
                break

    if matched_enum is not None:
        canonical = CANONICAL_HF.get(matched_enum.name)
        if canonical is None:
            # Mapped to an enum we deliberately drop (e.g., ROPE_FREQS).
            return RemapDecision(
                RemapDecision.KIND_SKIP, reason=f"{matched_enum.name} not serialized to HF/MLX"
            )
        hf_name = canonical.format(bid=bid) if bid is not None else canonical
        # If the GGUF name didn't carry .weight (e.g., scalar tensors) and the HF
        # form does, that's fine — caller can still write under hf_name.
        transform = "passthrough"
        if matched_enum.name in ("ATTN_Q", "ATTN_K") and arch_alias == "LLAMA":
            transform = "qk_permute"
        if matched_enum.name == "FFN_GATE_UP_EXP":
            transform = "moe_split_gate_up"
        # Qwen3.5/3.6 hybrid Mamba: GGUF stores conv1d as (out_ch, kernel),
        # MLX Conv1d expects (out_ch, kernel, in_ch_per_group=1) for depthwise.
        if matched_enum.name == "SSM_CONV1D":
            transform = "conv1d_unsqueeze"
        # GGUF stores SSM_A as -exp(A_log); MLX model expects raw A_log.
        if matched_enum.name == "SSM_A":
            transform = "ssm_a_to_a_log"
        return RemapDecision(RemapDecision.KIND_MAP, hf_name=hf_name, transform=transform, bid=bid)

    # Per-arch overrides for tensors not in upstream TENSOR_NAMES.
    for pat, hf_fmt in EXTRA_OVERRIDES.get(arch_alias, []):
        m = pat.match(base)
        if m:
            if hf_fmt is None:
                return RemapDecision(RemapDecision.KIND_SKIP, reason=f"override drops {base!r}")
            bid = int(m.group(1)) if m.groups() else None
            hf_name = hf_fmt.format(bid=bid) if bid is not None else hf_fmt
            return RemapDecision(
                RemapDecision.KIND_MAP, hf_name=hf_name, transform="passthrough", bid=bid
            )

    # Genuine unknown — hard-fail so we extend the override table.
    return RemapDecision(
        RemapDecision.KIND_FAIL, reason=f"no remap entry for {gguf_name!r} (arch={arch_string})"
    )


# GGUF metadata + arch detection
def _read_string_field(reader, field_name: str) -> str | None:
    """Read a GGUF string KV field; return None if missing."""
    f = reader.fields.get(field_name)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def detect_arch(reader) -> str:
    """Read `general.architecture` from GGUF metadata. Returns the raw string
    (e.g., "gemma4", "gemma3", "qwen3", "qwen3moe", "llama"). Caller maps to
    a TensorNameMap arch enum.
    """
    arch = _read_string_field(reader, "general.architecture")
    if arch is None:
        raise ValueError("GGUF missing 'general.architecture' KV field — can't detect arch")
    return arch
