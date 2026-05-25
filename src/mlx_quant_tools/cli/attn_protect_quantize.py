"""Attention-protected mixed-precision quantizer for MLX checkpoints.

Tier 1: floors-only recipe. No sensitivity scoring, no GPTQ.

The five floor rules:
  Rule 1: linear_attn.<any direct child>                  → policy from --attn-protect-mode
  Rule 2: attn.out_proj  (NOT self_attn.o_proj)           → policy from --attn-protect-mode
  Rule 3: self_attn.{q,k,v,o}_proj                        → max(--bits, 8), --group-size
  Rule 4: lm_head                                         → 8-bit, --group-size
  Rule 5: MoE routers/gates                               → policy from --attn-protect-mode
          (router.proj, mlp.gate, shared_expert_gate)
  Default: every other quantizable module gets (--bits, --group-size).

Output is a standard mlx-lm safetensors checkpoint, validatable with
``mqt-inspect-recipe``.

Usage examples:
  mqt-quantize Qwen/Qwen3.6-4B --bits 4
  mqt-quantize /path/to/model --bits 4 --attn-protect-mode 8bit
  mqt-quantize /path/to/model --bits 4 --with-dwq
  mqt-quantize Qwen/Qwen3.6-4B --bits 4 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from mlx_quant_tools.model_role_classifier import (
    TensorRole,
    classify_path,
    extract_layer_idx,
    is_vlm_tower,
)

# ---------- progress logging ----------


def info(msg: str) -> None:
    """Status messages go to stderr so stdout stays clean for --dry-run output."""
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


# ---------- predicate ----------

# Tensor-role patterns live in `model_role_classifier.py` (imported above).
# That module is the single source of truth for matching module paths to
# semantic roles (LINEAR_ATTN, ATTENTION_QKVO, MOE_ROUTER, etc.); this file
# owns only the *policy* — which role gets which (bits, group_size).
#
# Roles relevant to the Tier 2 MLP-boost candidate set: FFN and
# SHARED_EXPERT. Routed experts (a separate role) are excluded from the
# candidate set — too numerous to score within budget.

# Affine-quant scale/bias storage precision (mlx-lm defaults to half-precision
# for both). Used by `_affine_bpw` to model the metadata overhead per group.
_AFFINE_SCALE_BITS = 16
_AFFINE_BIAS_BITS = 16
# mlx.metallib precompiled affine kernels span these group sizes (all bits in
# {2,3,4,5,6,8}). gs-refine candidates are only emitted when the halved group
# size lands inside this set.
_SUPPORTED_AFFINE_GROUP_SIZES = (16, 32, 64, 128)


def _affine_bpw(
    bits: int,
    group_size: int,
    *,
    scale_bits: int = _AFFINE_SCALE_BITS,
    bias_bits: int = _AFFINE_BIAS_BITS,
) -> float:
    """Bits-per-weight of an affine-quantized recipe at (bits, group_size).

    Models the on-disk cost: weight bits per element plus per-group scale and
    bias amortized across the group. q4 + gs=64 → 4 + 32/64 = 4.5; q4 + gs=32
    → 4 + 32/32 = 5.0; gs-refine `delta_bpw` between those is 0.5, matching
    Track 4 §3's quoted ~0.5 for gs=32. Pure arithmetic; no mlx import.
    """
    return float(bits) + (scale_bits + bias_bits) / float(group_size)


PredicateVerdict = bool | dict

# Roles whose paths are eligible for `bf16_shared_mlp` (the Tier 2 candidate
# set). The classifier separates dense FFN from shared-expert MLPs as
# distinct roles, but the predicate treats them uniformly.
_BOOST_CANDIDATE_ROLES = (TensorRole.FFN, TensorRole.SHARED_EXPERT)


def make_attn_protect_predicate(
    *,
    bits: int,
    group_size: int,
    attn_protect_mode: str,
    quantize_linear_attn: bool = False,
    quantize_attn_out: bool = False,
    no_attn_floor: bool = False,
    no_lm_head_floor: bool = False,
    quantize_moe_router: bool = False,
    floor_tied_embed: bool = False,
    protect_vlm: bool = False,
    bf16_routed_experts: bool = False,
    bf16_embed_tokens: bool = False,
    bf16_attn_floor: bool = False,
    bf16_shared_mlp: bool = False,
    local_attn_mode: str | None = None,
    layer_types: list[str] | None = None,
    boosts: dict | None = None,
) -> Callable[[str, object], PredicateVerdict]:
    """Build the per-module predicate consumed by mlx_lm.utils.quantize_model.

    Signature matches mlx-lm 0.31's `quant_predicate(path, module) -> bool|dict`.
    The module argument is unused by the floors-only recipe — the path string
    is the entire input.

    Naming convention: every ablation kwarg is positive ("do the thing
    that's normally protected against"). All default to False, meaning full
    attention protection is on. Setting a `quantize_*` flag to True turns
    OFF the corresponding rule's protection and forces base-bits quantization
    on those modules; setting a `no_*_floor` flag to True removes the 8-bit
    floor on those modules.

    `floor_tied_embed` (when True) extends Rule 4's 8-bit floor to the
    `embed_tokens` module — for tied-embedding models where embed_tokens
    serves as both input lookup and output projection. The caller is
    responsible for only setting this when the model actually has tied
    embeddings; the predicate will apply the verdict unconditionally.
    Disabled by `--no-lm-head-floor` (logically the same Rule 4).

    `protect_vlm` (when True) adds a VLM tower rule that fires before the
    other rules: vision/audio tower and multi-modal projector paths receive
    the same verdict as Rule 1 (driven by `attn_protect_mode`). This rule
    must run first because VLM tower modules contain their own self_attn /
    mlp / o_proj sub-paths that would otherwise be matched by Rule 3 or the
    default-quantize fallthrough.

    Rule 5 (MoE routers/gates) is on by default; ablate with
    `quantize_moe_router=True`. It's a no-op on dense models since the
    matched paths only exist in MoE architectures.

    `boosts` (Tier 2) is an optional per-tensor exact-match map
    `{path: {"bits": N, "group_size": M}}` produced by sensitivity
    scoring. Boosts are checked *first*, ahead of every rule, so they
    can override even an attention-floor verdict (e.g. an MLP tensor
    promoted from base bits to base+1). The expected use case is per-
    layer MLP boosts at +1 bit; other promotions are mechanically
    supported but not part of the v1 allocator.
    """
    if attn_protect_mode == "bf16":
        protect_verdict: PredicateVerdict = False
    elif attn_protect_mode == "8bit":
        protect_verdict = {"bits": max(bits, 8), "group_size": group_size}
    else:
        raise ValueError(f"unknown attn_protect_mode: {attn_protect_mode!r}")

    qkvo_verdict = {"bits": max(bits, 8), "group_size": group_size}
    lm_head_verdict = {"bits": 8, "group_size": group_size}
    # P3.1: per-layer-type override for QKVO floor. Only fires when
    # `local_attn_mode` is set AND the model's config exposes `layer_types`
    # AND the matched module's layer index maps to "sliding_attention".
    # Validation:
    #   - bf16     → no-op (same as default attn_protect_mode=bf16)
    #   - 8bit     → 8-bit affine on sliding QKVO (vs bf16 floor)
    #   - base     → drop sliding QKVO to base bits (joins DWQ pool at q4)
    if local_attn_mode is None:
        local_verdict: PredicateVerdict | None = None
    elif local_attn_mode == "bf16":
        local_verdict = False
    elif local_attn_mode == "8bit":
        local_verdict = {"bits": max(bits, 8), "group_size": group_size}
    elif local_attn_mode == "base":
        local_verdict = {"bits": bits, "group_size": group_size}
    else:
        raise ValueError(f"unknown local_attn_mode: {local_attn_mode!r}")
    boosts = boosts or {}

    def predicate(path: str, module: object) -> PredicateVerdict:
        if path in boosts:
            return dict(boosts[path])
        # VLM is orthogonal to leaf roles — a tower's `self_attn.q_proj` is
        # both is_vlm_tower=True and role=ATTENTION_QKVO. When --protect-vlm
        # is on, every tower-internal path gets `protect_verdict` regardless
        # of its leaf role.
        if protect_vlm and is_vlm_tower(path):
            return protect_verdict
        role = classify_path(path)
        if not quantize_linear_attn and role == TensorRole.LINEAR_ATTN:
            return protect_verdict
        if not quantize_attn_out and role == TensorRole.ATTENTION_OUTPUT:
            return protect_verdict
        if not quantize_moe_router and role == TensorRole.MOE_ROUTER:
            return protect_verdict
        if bf16_routed_experts and role == TensorRole.ROUTED_EXPERT:
            return False
        if bf16_embed_tokens and role in (
            TensorRole.LM_HEAD,
            TensorRole.EMBEDDING,
            TensorRole.EMBEDDING_PER_LAYER,
        ):
            return False
        if bf16_attn_floor and role == TensorRole.ATTENTION_QKVO:
            return False
        if bf16_shared_mlp and role in _BOOST_CANDIDATE_ROLES:
            return False
        # P3.1: per-layer-type QKVO override. Checked before the standard
        # attn-floor rule so it can override even bf16. Only fires on
        # sliding-attention layers when the override is configured; full-
        # attention layers fall through to the standard floor below.
        if (
            local_verdict is not None
            and layer_types is not None
            and role == TensorRole.ATTENTION_QKVO
        ):
            idx = extract_layer_idx(path)
            if (
                idx is not None
                and 0 <= idx < len(layer_types)
                and layer_types[idx] == "sliding_attention"
            ):
                return local_verdict
        if not no_attn_floor and role == TensorRole.ATTENTION_QKVO:
            return qkvo_verdict
        if not no_lm_head_floor and role == TensorRole.LM_HEAD:
            return lm_head_verdict
        if not no_lm_head_floor and floor_tied_embed and role == TensorRole.EMBEDDING:
            return lm_head_verdict
        # PLE input lookup (E2B/E4B only). Gated by the same flag as lm_head
        # since it's a vocabulary-sized table read once per token. Independent
        # of `floor_tied_embed` (this table is not tied to anything).
        if not no_lm_head_floor and role == TensorRole.EMBEDDING_PER_LAYER:
            return lm_head_verdict
        return True

    return predicate


# ---------- Tier 2: KLD-recovery sensitivity scoring ----------

_KLD_HELPERS_MOD = None


def _load_kld_helpers():
    """Import the KLD scoring helpers from the package. Used by the sensitivity
    loop to reuse the existing teacher top-K cache + KLD computation without
    subprocess round-trips."""
    global _KLD_HELPERS_MOD
    if _KLD_HELPERS_MOD is not None:
        return _KLD_HELPERS_MOD
    from mlx_quant_tools.cli import score_kld as _mod

    _KLD_HELPERS_MOD = _mod
    return _mod


# Boost candidate types. Each MLP/expert tensor can spawn one of each; the
# allocator picks at most one per tensor by `recovery / bpw_cost` density.
_BOOST_TYPE_BIT_BUMP = "bit-bump"  # (base_bits, base_gs) → (base_bits+1, base_gs)
_BOOST_TYPE_GS_REFINE = "gs-refine"  # (base_bits, base_gs) → (base_bits, base_gs/2)


def _candidate_id(path: str, candidate_type: str) -> str:
    """Stable id for a (path, candidate_type) pair. Used as the dict key for
    `recovery` and `pre_quantized_boosts` so the same path can carry multiple
    candidate scores."""
    return f"{path}|{candidate_type}"


def find_mlp_boost_candidates(
    model,
    *,
    base_bits: int,
    base_group_size: int,
    candidate_types: tuple[str, ...] = (_BOOST_TYPE_BIT_BUMP, _BOOST_TYPE_GS_REFINE),
    bit_bump_group_size: int | None = None,
) -> list[dict]:
    """Walk model.named_modules() and build the Tier 2 candidate list.

    For every quantizable dense-MLP projection (classifier roles FFN or
    SHARED_EXPERT — excluding routed experts, attention/lm_head floors,
    VLM towers, MoE routers), emit one candidate dict per type in
    `candidate_types`. Each
    candidate carries everything the scorer and allocator need:

        {
          "id": "<path>|<candidate_type>",
          "path": str,
          "module": nn.Module,           # bf16 source module
          "candidate_type": str,         # "bit-bump" | "gs-refine"
          "base_bits": int,
          "base_group_size": int,
          "target_bits": int,
          "target_group_size": int,
          "delta_bpw": float,            # cost vs base, computed via _affine_bpw
        }

    `bit_bump_group_size` (default None → `base_group_size`) overrides the
    bit-bump candidate's target group size. Preserves the legacy
    `--mlp-boost-group-size` semantics where a +1-bit boost can simultaneously
    refine groups; orthogonal to the gs-refine candidate type.

    A `gs-refine` candidate is generated only when `base_group_size // 2`
    lands inside the supported affine group sizes; otherwise that path emits
    only its `bit-bump` candidate (if requested).

    Must be called BEFORE the model is AP-quantized — sensitivity scoring
    needs the bf16 modules to construct the boosted versions. After
    AP-quantize runs, the matched modules will have been replaced with
    `QuantizedLinear` instances at base bits and the boost-construction
    path no longer applies.
    """
    if bit_bump_group_size is None:
        bit_bump_group_size = base_group_size
    base_bpw = _affine_bpw(base_bits, base_group_size)

    candidates: list[dict] = []
    for path, module in model.named_modules():
        if classify_path(path) not in _BOOST_CANDIDATE_ROLES:
            continue
        if not hasattr(module, "to_quantized"):
            continue
        for ctype in candidate_types:
            if ctype == _BOOST_TYPE_BIT_BUMP:
                target_bits = base_bits + 1
                target_gs = bit_bump_group_size
            elif ctype == _BOOST_TYPE_GS_REFINE:
                target_gs = base_group_size // 2
                if target_gs not in _SUPPORTED_AFFINE_GROUP_SIZES:
                    continue
                target_bits = base_bits
            else:
                raise ValueError(f"unknown candidate type: {ctype!r}")
            delta_bpw = _affine_bpw(target_bits, target_gs) - base_bpw
            candidates.append(
                {
                    "id": _candidate_id(path, ctype),
                    "path": path,
                    "module": module,
                    "candidate_type": ctype,
                    "base_bits": base_bits,
                    "base_group_size": base_group_size,
                    "target_bits": target_bits,
                    "target_group_size": target_gs,
                    "delta_bpw": delta_bpw,
                }
            )
    return candidates


def _walk_to_parent(model, path: str):
    """Return (parent_module, leaf_name) for a dotted attribute path. Numeric
    segments index into list-typed children (e.g. `model.layers.0` →
    `model.layers[0]`)."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def _get_module(model, path: str):
    parent, leaf = _walk_to_parent(model, path)
    return parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)


def _set_module(model, path: str, new_module) -> None:
    parent, leaf = _walk_to_parent(model, path)
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def score_mlp_sensitivity(
    *,
    model,
    candidates: list[dict],
    teacher_path: str,
    calibration_data: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    pre_quantized_boosts: dict | None = None,
) -> dict:
    """KLD-recovery scoring loop. Returns:
        {
          "baseline_kld": float,            # mean KLD with no boosts
          "recovery": {cand_id: float},     # nats recovered if candidate fires
          "tensor_params": {path: int},     # parameter count per tensor
          "boosted_modules": {cand_id: nn.Module},
          "calibration": {...},             # spec used (for provenance)
          "cache_dir": str,                 # where teacher logits cache lives
        }

    `model` must be the AP-quantized model (already in memory, post-quantize).
    `candidates` is the list returned by `find_mlp_boost_candidates(...)` —
    one dict per (path, candidate_type), each carrying its own target_bits
    and target_group_size. Same path may appear in multiple entries (one per
    candidate type); recovery is keyed by `id` (`f"{path}|{candidate_type}"`)
    to disambiguate.

    Algorithm:
      1. Pre-quantize each candidate at its own target (bits, group_size).
      2. Ensure teacher top-K cache exists at the calibration spec; the
         existing `score-mlx-kld.py` teacher cache amortizes the pass across
         this scoring loop and any later KLD scoring on the same teacher.
      3. Baseline forward: full AP-quantized model vs teacher → mean KLD.
      4. For each candidate: swap in its boosted module → forward → KLD →
         restore. `recovery[cand_id] = baseline - tensor_kld`.

    The dual framing (recovery, not perturbation) directly answers the
    allocator's question: "what's the KLD value of boosting this tensor at
    this candidate type?" Measuring at the actual boost level avoids
    systematically overestimating the gain that the allocator can deliver.
    """
    import mlx.core as mx

    helpers = _load_kld_helpers()

    # 1. Pre-quantize each candidate at its own (target_bits, target_group_size)
    # (or accept pre-computed).
    #
    # `pre_quantized_boosts` lets the caller hand in modules that were
    # quantized from the bf16 source *before* the source was freed. Used by
    # the post-DWQ-baseline flow (--mlp-boost-baseline-from): we pre-quantize
    # while bf16 is still loaded, then drop bf16 and load the DWQ'd baseline
    # as the model to score against. Without this, sensitivity would have
    # to hold both the bf16 source and the baseline simultaneously (~80 GB
    # peak on a 26B target), or re-load bf16 just to do the pre-quantize step.
    # `pre_quantized_boosts` keys are candidate ids (`{path}|{candidate_type}`).
    tensor_params: dict = {}
    for cand in candidates:
        path = cand["path"]
        if path not in tensor_params:
            tensor_params[path] = int(cand["module"].weight.size)
    if pre_quantized_boosts is not None:
        info(f"Using {len(pre_quantized_boosts)} pre-computed boosted modules")
        boosted_modules = pre_quantized_boosts
    else:
        info(
            f"Pre-quantizing {len(candidates)} MLP candidates "
            f"(per-candidate targets) for the recovery loop"
        )
        boosted_modules: dict = {}
        for cand in candidates:
            boosted = cand["module"].to_quantized(
                group_size=cand["target_group_size"],
                bits=cand["target_bits"],
            )
            # Force eval so the bf16 source can be freed once AP-quantize replaces it.
            mx.eval(boosted.parameters())
            boosted_modules[cand["id"]] = boosted
        mx.eval(boosted_modules)

    # 2. Ensure teacher top-K cache.
    # Sensitivity scoring is a within-loop delta ranking, not an absolute
    # publication number — protocol-bias terms (top-K floor, score window)
    # add equally to every recipe and cancel in the per-tensor delta. We
    # pin top_k=128 here to keep cache size bounded (~770 MB per teacher at
    # 1M tokens vs ~196 GB at the new K=32768 publishing default) and to
    # preserve existing sensitivity caches across the v2 protocol switch.
    info(
        f"Ensuring teacher cache: data={calibration_data}, "
        f"samples={num_samples}, seq={max_seq_len}, seed={seed}"
    )
    cache_dir, manifest, _tokenizer = helpers.ensure_teacher_topk_cache(
        teacher_path=teacher_path,
        dataset_name=calibration_data,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed,
        batch_size=1,
        top_k=128,
    )

    model.eval()

    # 3. Baseline forward.
    info("Sensitivity baseline: forward AP-quantized model vs teacher cache")
    baseline = helpers.score_loaded_student(model, cache_dir, manifest)
    baseline_kld = float(baseline["kld"]["mean"])
    info(f"Baseline mean KLD: {baseline_kld:.4f} nats")

    # 4. Recovery loop.
    recovery: dict = {}
    n = len(candidates)
    for i, cand in enumerate(candidates, start=1):
        cid = cand["id"]
        path = cand["path"]
        info(
            f"[{i}/{n}] scoring {cid} "
            f"({cand['candidate_type']} → bits={cand['target_bits']} "
            f"gs={cand['target_group_size']})"
        )
        original = _get_module(model, path)
        _set_module(model, path, boosted_modules[cid])
        try:
            metrics = helpers.score_loaded_student(model, cache_dir, manifest)
            tensor_kld = float(metrics["kld"]["mean"])
        finally:
            _set_module(model, path, original)
        recovery[cid] = baseline_kld - tensor_kld
        info(f"     KLD={tensor_kld:.4f}, recovery={recovery[cid]:+.4f} nats")

    return {
        "baseline_kld": baseline_kld,
        "recovery": recovery,
        "tensor_params": tensor_params,
        "boosted_modules": boosted_modules,  # caller permanently swaps in the
        # subset chosen by allocate_mlp_boosts
        "calibration": {
            "data": calibration_data,
            "samples": num_samples,
            "max_seq_len": max_seq_len,
            "seed": seed,
        },
        "cache_dir": str(cache_dir),
    }


# ---------- Tier 2: MLP boost allocator ----------


def allocate_mlp_boosts(
    *,
    candidates: list[dict],
    recovery: dict,
    tensor_params: dict,
    bit_budget: float,
    recovery_noise_floor: float = 0.002,
) -> dict:
    """Greedy `recovery / bpw_cost` allocator over heterogeneous candidates.

    Returns `{path: verdict}` where verdict is `{"bits": ..., "group_size": ...}`
    consumed by `make_attn_protect_predicate(boosts=...)`. At most one boost
    per path (densest candidate per path wins); lower-density candidates for
    the same path are skipped after the path is assigned.

    Inputs:
      `candidates`         — list returned by `find_mlp_boost_candidates(...)`.
                             Each carries its own target_bits, target_group_size,
                             and delta_bpw (cost vs base).
      `recovery`           — `{cand_id: kld_recovery_in_nats}` from sensitivity.
                             Must be defined for every candidate id.
      `tensor_params`      — `{path: parameter_count}` for each candidate path.
      `bit_budget`         — total *extra* bits the allocator may spend across
                             all selected boosts (real-valued because gs-refine
                             carries a fractional delta_bpw):
                                 bit_budget = mlp_boost_budget_bpw
                                            × total_quantizable_params
      `recovery_noise_floor` — Track 4 §3 caveat: when absolute recovery is
                             ~noise (~0.002 nats), density `recovery/cost` is
                             dominated by the noise term and the smaller-cost
                             variant wins systematically. Skip candidates with
                             recovery below this floor. Default 0.002 nats.

    Greedy density-sort is provably optimal under a knapsack interpretation
    even with heterogeneous costs (Track 4 §3). Tie-break by recovery
    descending, then by candidate id for determinism.
    """
    if bit_budget <= 0:
        return {}

    ranked = []
    for cand in candidates:
        cid = cand["id"]
        path = cand["path"]
        score = recovery.get(cid)
        if score is None or score <= 0.0:
            continue
        if score < recovery_noise_floor:
            continue
        params = tensor_params.get(path)
        if params is None or params <= 0:
            continue
        delta_bpw = float(cand["delta_bpw"])
        if delta_bpw <= 0.0:
            continue
        cost = params * delta_bpw
        density = score / cost
        ranked.append((density, score, cid, cand, cost))
    # Sort: highest density first, then highest recovery, then candidate id.
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))

    boosts: dict = {}
    spent = 0.0
    for _density, _score, _cid, cand, cost in ranked:
        path = cand["path"]
        if path in boosts:
            # Densest candidate for this path already won. Skip lower-density
            # same-path candidates without consuming budget.
            continue
        if spent + cost > bit_budget:
            # Doesn't fit. Don't mark the path taken — a smaller-cost candidate
            # for the same path may still squeeze in later in the ranking.
            continue
        boosts[path] = {
            "bits": cand["target_bits"],
            "group_size": cand["target_group_size"],
            "candidate_type": cand["candidate_type"],
        }
        spent += cost
    return boosts


# ---------- output path ----------


def default_output_path(model_path: str, args: argparse.Namespace) -> Path:
    """Derive `./<basename>-AP<bits>bit[-8bit][-dwq][-gs<N>]`.

    `model_path` is the user's --model arg; for HF ids like `Qwen/Qwen3.6-4B`
    we use the part after the slash.
    """
    base = model_path.rstrip("/").split("/")[-1]
    suffix = f"-AP{args.bits}bit"
    # `tied8` only added when the flag actually applied (model had tied
    # embeddings); the resolution happens in main() and is forwarded here
    # as args.floor_tied_embed_effective.
    if getattr(args, "floor_tied_embed_effective", False):
        suffix += "-tied8"
    if args.attn_protect_mode == "8bit":
        suffix += "-8bit"
    if getattr(args, "quantize_moe_router", False):
        suffix += "-noroute"
    if getattr(args, "bf16_routed_experts", False):
        suffix += "-experts-bf16"
    if getattr(args, "bf16_embed_tokens", False):
        suffix += "-embed-bf16"
    if getattr(args, "bf16_attn_floor", False):
        suffix += "-attn-bf16"
    if getattr(args, "bf16_shared_mlp", False):
        suffix += "-mlp-bf16"
    local_attn = getattr(args, "local_attn_mode", None)
    if local_attn is not None:
        # bf16 is the no-op default; suffix only when the override changes
        # behavior. "8bit" becomes "-localattn8" and "base" becomes
        # "-localattnB" so the recipe is unambiguous in the output dir name.
        if local_attn == "8bit":
            suffix += "-localattn8"
        elif local_attn == "base":
            suffix += "-localattnB"
    if getattr(args, "with_mlp_boosts", False):
        suffix += "-mlpboost"
        # P4.1 step 0: encode the candidate-type mode for non-default modes.
        # bit-bump preserves the legacy `-mlpboost` suffix so existing rollup
        # rows remain matchable; gs-refine and mixed get explicit tags.
        cand_mode = getattr(args, "mlp_boost_candidates", _BOOST_TYPE_BIT_BUMP)
        if cand_mode == _BOOST_TYPE_GS_REFINE:
            suffix += "-gsr"
        elif cand_mode == "both":
            suffix += "-mix"
    if args.with_dwq:
        suffix += "-dwq"
    if args.group_size != 64:
        suffix += f"-gs{args.group_size}"
    boost_gs = getattr(args, "mlp_boost_group_size", None)
    if (
        getattr(args, "with_mlp_boosts", False)
        and boost_gs is not None
        and boost_gs != args.group_size
    ):
        suffix += f"-bgs{boost_gs}"
    return Path.cwd() / f"{base}{suffix}"


# ---------- dry-run printer ----------


def print_recipe(model, predicate: Callable[[str, object], PredicateVerdict]) -> None:
    """Walk model.named_modules() and print the predicate verdict per module.

    Mirrors mlx_lm's wrapped_predicate filtering: only modules with
    `to_quantized` are eligible to be quantized, so we only report on those.
    """
    rows: list[tuple[str, str]] = []
    for path, module in model.named_modules():
        if not hasattr(module, "to_quantized"):
            continue
        verdict = predicate(path, module)
        if verdict is False:
            cell = "skip (bf16)"
        elif verdict is True:
            cell = "default"
        elif isinstance(verdict, dict):
            b = verdict.get("bits")
            g = verdict.get("group_size")
            cell = f"bits={b}, group_size={g}"
        else:
            cell = repr(verdict)
        rows.append((path, cell))

    width = max((len(p) for p, _ in rows), default=20)
    print(f"{'module path':<{width}}  verdict")
    print(f"{'-' * width}  {'-' * 30}")
    for path, cell in rows:
        print(f"{path:<{width}}  {cell}")
    print(f"\n{len(rows)} quantizable modules.")


# ---------- recipe provenance ----------


def _git_sha_of(path: Path) -> str:
    """Return a short git SHA describing the script's checkout, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def write_recipe_provenance(
    out_dir: Path,
    args: argparse.Namespace,
    *,
    boosts: dict | None = None,
    boost_meta: dict | None = None,
    dwq_completed: bool = False,
) -> None:
    """Write `<out_dir>/attn-protect-recipe.json` describing the resolved CLI.

    The sub-object schema is locked by `docs/kld-scoring-plan.md` so that the
    scorer can inline it verbatim into its own `recipe` field.

    `dwq_completed` controls whether `with_dwq=True` is recorded for this run.
    The recipe is written *twice* in DWQ runs — once before the cascade with
    `dwq_completed=False` so the on-disk recipe matches AP-only weights if the
    cascade dies (Python exception, Metal C++ uncaught_exception, SIGKILL,
    OOM-kill), and again after the cascade completes successfully with
    `dwq_completed=True`. The companion `.dwq-in-flight` marker file (written
    around the cascade) is the forensic indicator of "DWQ requested, did not
    complete" — distinguishable from "DWQ never requested" by its presence.

    `with_dwq` and `with_mlp_boosts` reflect *effective* behavior, not the
    requested flags. The cascade and the boost allocator are no-ops in some
    flag combos (e.g. `--with-dwq --bits 8`, or `--with-mlp-boosts` with the
    allocator returning an empty map under a 0.0 budget), so the recipe must
    record what's actually in the weights. Same convention as
    `floor_tied_embed` via `floor_tied_embed_effective`.

    Sub-objects:
      `dwq`         — present iff with_dwq=true; calibration spec.
      `mlp_boosts`  — present iff with_mlp_boosts=true and at least one
                      tensor was actually boosted; carries the calibration
                      spec, the bpw budget, and the resolved per-tensor
                      boost map. Memory levers like `--dwq-target-dir` are
                      deliberately excluded — they don't change the weights.
    """
    dwq_effective = bool(args.with_dwq) and args.bits < 8 and dwq_completed
    # Variant C: weights inherit DWQ from the loaded baseline on every un-boosted
    # tensor (boosted tensors are fresh 5-bit-from-bf16, un-DWQ'd). Set with_dwq
    # to reflect the state of the dominant tensor population — sensitivity_phase
    # in the mlp_boosts sub-object disambiguates. No cascade runs on this branch
    # so dwq_completed is irrelevant here.
    if getattr(args, "mlp_boost_baseline_from", None) is not None:
        dwq_effective = True
    boosts_effective = bool(getattr(args, "with_mlp_boosts", False)) and bool(boosts)
    payload = {
        "tool": "attn-protect-quantize",
        "tool_version": _git_sha_of(Path(__file__).resolve()),
        "source_model": args.model,
        "bits": args.bits,
        "group_size": args.group_size,
        "attn_protect_mode": args.attn_protect_mode,
        "with_dwq": dwq_effective,
        "with_mlp_boosts": boosts_effective,
        "floor_tied_embed": bool(getattr(args, "floor_tied_embed_effective", False)),
        "protect_vlm": bool(args.protect_vlm),
        "quantize_linear_attn": bool(args.quantize_linear_attn),
        "quantize_attn_out": bool(args.quantize_attn_out),
        "no_attn_floor": bool(args.no_attn_floor),
        "no_lm_head_floor": bool(args.no_lm_head_floor),
        "quantize_moe_router": bool(args.quantize_moe_router),
        "bf16_routed_experts": bool(getattr(args, "bf16_routed_experts", False)),
        "bf16_embed_tokens": bool(getattr(args, "bf16_embed_tokens", False)),
        "bf16_attn_floor": bool(getattr(args, "bf16_attn_floor", False)),
        "bf16_shared_mlp": bool(getattr(args, "bf16_shared_mlp", False)),
        "local_attn_mode": getattr(args, "local_attn_mode", None),
    }
    if dwq_effective:
        payload["dwq"] = {
            "calibration_data": args.calibration_data,
            "samples": args.calibration_samples,
            "seed": args.calibration_seed,
            "batch_size": args.calibration_batch_size,
            "max_seq_len": args.calibration_max_seq_len,
            "split": getattr(args, "calibration_split", "train"),
            "text_column": getattr(args, "calibration_text_column", None),
        }
    if boosts_effective:
        # Per-boost entries carry `candidate_type` ("bit-bump" | "gs-refine")
        # so the rollup can attribute each tensor's verdict to a candidate.
        # Old recipe.json files (no candidate_type) remain readable.
        type_counts: dict[str, int] = {}
        for v in boosts.values():
            ctype = v.get("candidate_type", _BOOST_TYPE_BIT_BUMP)
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        payload["mlp_boosts"] = {
            "budget_bpw": float(args.mlp_boost_budget_bpw),
            "candidate_mode": getattr(args, "mlp_boost_candidates", _BOOST_TYPE_BIT_BUMP),
            "applied_count": len(boosts),
            "applied_by_type": type_counts,
            "recovery_noise_floor": float(getattr(args, "mlp_boost_noise_floor", 0.002)),
            "calibration": {
                "data": args.mlp_boost_calibration_data,
                "samples": args.mlp_boost_calibration_samples,
                "max_seq_len": args.mlp_boost_calibration_max_seq_len,
                "seed": args.mlp_boost_calibration_seed,
            },
            "boosts": dict(sorted(boosts.items())),
        }
        if getattr(args, "mlp_boost_baseline_from", None) is not None:
            # Variant C: sensitivity was measured against the loaded baseline,
            # not against a freshly AP-quantized model. Record the lineage so
            # the rollup can distinguish post-DWQ-sensitivity rows from
            # standard pre-DWQ-sensitivity ones.
            payload["mlp_boosts"]["baseline_from"] = str(args.mlp_boost_baseline_from)
            payload["mlp_boosts"]["sensitivity_phase"] = "post_dwq"
        if boost_meta:
            # Optional extras: baseline_kld, candidate_count, bit_budget.
            payload["mlp_boosts"].update(boost_meta)
    (out_dir / "attn-protect-recipe.json").write_text(json.dumps(payload, indent=2))


def _mark_recipe_dwq_failed(out_dir: Path, reason: str) -> None:
    """Rewrite recipe.json after a DWQ-cascade failure so it matches the
    AP-only weights actually on disk.

    Adds `dwq_failed=true` and the failure reason; clears the `dwq` sub-object
    and flips `with_dwq` to false. Schema-additive (no version bump): consumers
    that don't know about `dwq_failed` see a normal AP-only recipe, which is
    the correct interpretation of the saved weights.

    `reason` is a free-form string. Caller passes either the formatted Python
    exception or a description of how the failure was detected (e.g. stale
    marker file from a prior crashed process).
    """
    recipe = out_dir / "attn-protect-recipe.json"
    if not recipe.exists():
        return
    payload = json.loads(recipe.read_text())
    payload["with_dwq"] = False
    payload["dwq"] = None
    payload["dwq_failed"] = True
    payload["dwq_failure_reason"] = reason[:500]
    recipe.write_text(json.dumps(payload, indent=2))


# Marker file written around the DWQ cascade. Its presence after the cascade
# returns control to Python means the cascade died via a path that bypasses
# Python's exception machinery (Metal C++ uncaught_exception, SIGKILL, OOM
# reaper, machine reboot, ...). The recipe.json is intentionally written with
# with_dwq=False *before* the cascade and rewritten with with_dwq=True *after*
# success, so the recipe alone is the source of truth for "what's in the
# weights"; the marker is forensic evidence of "DWQ was requested but did
# not complete cleanly".
_DWQ_MARKER_NAME = ".dwq-in-flight"


def _write_dwq_marker(out_dir: Path, args: argparse.Namespace) -> Path:
    marker = out_dir / _DWQ_MARKER_NAME
    marker.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at_unix": time.time(),
                "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "calibration_data": args.calibration_data,
                "samples": args.calibration_samples,
                "seed": args.calibration_seed,
                "batch_size": args.calibration_batch_size,
                "max_seq_len": args.calibration_max_seq_len,
                "split": getattr(args, "calibration_split", "train"),
                "text_column": getattr(args, "calibration_text_column", None),
            },
            indent=2,
        )
    )
    return marker


def _clear_dwq_marker(marker: Path) -> None:
    try:
        marker.unlink()
    except FileNotFoundError:
        pass


def _reconcile_stale_dwq_marker(out_dir: Path) -> None:
    """If a `.dwq-in-flight` marker survived from a prior process, that prior
    DWQ cascade did not complete cleanly. Rewrite recipe.json to reflect the
    AP-only weights actually on disk and remove the marker. No-op if no
    marker exists."""
    marker = out_dir / _DWQ_MARKER_NAME
    if not marker.exists():
        return
    try:
        info = json.loads(marker.read_text())
        prior_pid = info.get("pid", "<unknown>")
        prior_iso = info.get("started_at_iso", "<unknown>")
        reason = (
            f"DWQ marker survived a prior process (pid={prior_pid}, "
            f"started_at={prior_iso}); cascade died via a path that bypasses "
            "Python exceptions (Metal C++ uncaught_exception / SIGKILL / OOM "
            "reaper). Recipe reconciled post-hoc."
        )
    except (json.JSONDecodeError, OSError):
        reason = "DWQ marker survived a prior process; cascade did not complete."
    _mark_recipe_dwq_failed(out_dir, reason)
    _clear_dwq_marker(marker)


def write_sensitivity_provenance(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    sensitivity: dict,
    bit_budget: int,
) -> None:
    """Write `<out_dir>/attn-protect-sensitivity.json` with the *full* recovery
    scores (all candidates, not just the boosted subset).

    Persisted alongside the checkpoint so:
      - the same scoring pass can be re-allocated against a different bpw
        budget without rerunning (the loop is the expensive part), and
      - rollups + post-hoc analyses can correlate per-tensor recovery with
        whatever empirical KLD the boosted checkpoint ultimately delivers.
    """
    payload = {
        "tool": "attn-protect-quantize",
        "tool_version": _git_sha_of(Path(__file__).resolve()),
        "source_model": args.model,
        "base_bits": args.bits,
        "boost_to_bits": args.bits + 1,
        "group_size": args.group_size,
        "calibration": sensitivity["calibration"],
        "baseline_kld": sensitivity["baseline_kld"],
        "bit_budget": bit_budget,
        "budget_bpw": float(args.mlp_boost_budget_bpw),
        "candidate_count": len(sensitivity["recovery"]),
        "tensor_params": dict(sorted(sensitivity["tensor_params"].items())),
        "recovery": dict(sorted(sensitivity["recovery"].items())),
        "cache_dir": sensitivity["cache_dir"],
    }
    (out_dir / "attn-protect-sensitivity.json").write_text(json.dumps(payload, indent=2))


# ---------- multimodal detection + save ----------


def is_multimodal(config: dict) -> bool:
    """A config is multimodal iff it has a non-empty vision_config or audio_config.

    Both Gemma-4-E (vision+audio) and Qwen2.5-VL (vision only) populate these
    sub-objects on the source HF config; mlx-lm's text-only model classes
    don't model the towers and reject the weights at load_weights(strict=True).
    Detecting these keys is the deterministic dispatch signal — if either
    sub-config is a non-empty dict, the load/save path goes through mlx-vlm.
    """
    for key in ("vision_config", "audio_config"):
        sub = config.get(key)
        if isinstance(sub, dict) and sub:
            return True
    return False


def save_multimodal(
    out_dir: Path,
    src_path: Path,
    model,
    processor,
    config: dict,
) -> None:
    """Save a multimodal checkpoint via mlx-vlm conventions.

    Mirrors mlx_vlm.convert.convert's save block: write the (now-quantized)
    weights via save_weights, copy auxiliary *.py / *.json files from the
    source snapshot, copy any sub-directories (chat templates etc.), let
    the processor write its tokenizer/processor files, and finally write
    the updated config.json on top.
    """
    from mlx_vlm.utils import save_config, save_weights

    save_weights(out_dir, model, donate_weights=True)

    # Copy auxiliary files from the source snapshot. Skip the safetensors
    # index — save_weights already wrote a fresh one for the quantized
    # shards. Skip config.json — save_config will write the updated one.
    for pattern in ("*.py", "*.json"):
        for f in src_path.glob(pattern):
            if f.name in ("model.safetensors.index.json", "config.json"):
                continue
            shutil.copy(f, out_dir)

    for item in src_path.iterdir():
        if item.is_dir():
            dest = out_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    processor.save_pretrained(str(out_dir))
    save_config(config, config_path=out_dir / "config.json")


# ---------- DWQ cascade ----------


def _wrap_with_cache_injection(model):
    """Runtime-subclass `model` so `model(batch)` injects a fresh KV cache and
    unwraps `LanguageModelOutput` back to a raw mx.array.

    Two effects, both of which mlx_lm.dwq depends on for VLMs:

      1. Cache injection. gemma-4 (and likely other mlx-vlm models) emit
         garbage logits when `__call__` is invoked with `cache=None` —
         RoPE / sliding-window position bookkeeping happens via the cache
         object. mlx_vlm.generate always sets one up; a single-shot full-
         sequence forward must do the same. (See commit 5e38e58 for the
         scorer-side fix that surfaced this; same root cause applies to
         dwq's teacher and student forwards.)

      2. Output unwrap. mlx-vlm's top-level `__call__` returns
         `LanguageModelOutput(logits=...)`; mlx_lm.dwq treats the return
         value as a raw array (`mx.stop_gradient(logits, ...)`,
         `mx.take_along_axis(logits, ids, ...)` etc.). Unwrapping inside
         the wrapped `__call__` keeps dwq's loop unchanged.

    Text-only models don't need either fix and aren't wrapped — we only
    call this on `is_multimodal(config)` checkpoints.
    """
    from mlx_lm.models import cache as cache_mod

    orig_cls = type(model)
    orig_call = orig_cls.__call__

    def call_with_cache(self, *args, **kwargs):
        if kwargs.get("cache") is None:
            target = self.language_model if hasattr(self, "language_model") else self
            kwargs["cache"] = cache_mod.make_prompt_cache(target, max_kv_size=None)
        out = orig_call(self, *args, **kwargs)
        return out.logits if hasattr(out, "logits") else out

    model.__class__ = type(
        f"CacheInjecting{orig_cls.__name__}",
        (orig_cls,),
        {"__call__": call_with_cache},
    )
    return model


def _patch_dwq_for_multimodal():
    """Context manager: swap mlx_lm.quant.dwq's `load` and `save` symbols for
    VLM-aware versions, and restore them on exit.

    Why patch the names bound inside dwq rather than `mlx_lm.utils.load` / `save`:
    `mlx_lm.quant.dwq` does `from mlx_lm.utils import load, save` at import
    time, so dwq has its own bindings. Patching `mlx_lm.utils.load` would
    not redirect dwq's calls. Patching `dwq.load` / `dwq.save` does.

    Both patched versions dispatch on `is_multimodal(config)`. Text-only
    sources fall through to the original mlx_lm helpers (so a multimodal
    teacher with a text-only student, or the inverse, still works one
    side at a time — though in practice we only run this for multimodal
    teacher + multimodal student).

    The patched `load` also wraps the returned model with cache injection
    so dwq's training loop can call `model(batch)` unchanged.
    """
    import contextlib

    from mlx_lm.quant import dwq as _dwq
    from mlx_lm.utils import load as _mlx_lm_load
    from mlx_lm.utils import save as _mlx_lm_save
    from mlx_vlm.utils import fetch_from_hub, get_model_path

    def vlm_aware_load(path_or_repo, *, lazy=False, return_config=False, **kw):
        local = get_model_path(path_or_repo)
        cfg = json.loads((local / "config.json").read_text())
        if not is_multimodal(cfg):
            return _mlx_lm_load(path_or_repo, lazy=lazy, return_config=return_config, **kw)
        model, config, processor = fetch_from_hub(local, lazy=lazy)
        _wrap_with_cache_injection(model)
        # dwq.main treats the second return value as `tokenizer` and uses it
        # only for `save(...)` at the end — the actual text tokenizer used by
        # `load_data` is fetched separately via `load_tokenizer(args.model)`.
        # Returning the processor lines up with `save_multimodal`'s expected
        # signature.
        if return_config:
            return model, processor, config
        return model, processor

    def vlm_aware_save(dst, src, model, tokenizer_or_processor, config, donate_model=True):
        local = get_model_path(src)
        cfg = json.loads((local / "config.json").read_text())
        if not is_multimodal(cfg):
            return _mlx_lm_save(
                dst,
                src,
                model,
                tokenizer_or_processor,
                config,
                donate_model=donate_model,
            )
        save_multimodal(Path(dst), local, model, tokenizer_or_processor, config)

    @contextlib.contextmanager
    def _ctx():
        saved_load = _dwq.load
        saved_save = _dwq.save
        try:
            _dwq.load = vlm_aware_load
            _dwq.save = vlm_aware_save
            yield
        finally:
            _dwq.load = saved_load
            _dwq.save = saved_save

    return _ctx()


def _patch_dwq_iterate_batches_seed():
    """Context manager: pre-seed numpy before each `iterate_batches` call so
    `seed=0` produces a deterministic batch order.

    Upstream `mlx_lm.tuner.trainer.iterate_batches` has `if seed:
    np.random.seed(seed)` which treats `seed=0` as falsy and skips seeding.
    The generator then uses whatever the global numpy state happens to be.
    That's harmless without target caching — every call within the same
    process inherits the same drifting state — but with `target_dir`,
    `compute_dwq_targets` writes shard `idx` for one batch, then
    `validate()` later reads shard `idx` expecting the same batch. The two
    iterations sit on opposite sides of `compute_dwq_targets`'s permutation
    consumption, so the cached target ends up bound to the wrong batch and
    `take_along_axis(student_logits, teacher_ids)` blows up on mismatched
    sequence lengths.

    Fix: wrap dwq's bound `iterate_batches` and call `np.random.seed(seed)`
    ourselves before delegating, so the falsy-`if` branch becomes a no-op
    on a known-seeded state. Non-zero seeds are pass-through (the upstream
    branch already seeds correctly).
    """
    import contextlib

    import numpy as np
    from mlx_lm.quant import dwq as _dwq

    saved = _dwq.iterate_batches

    def patched(*args, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            np.random.seed(seed)
        return saved(*args, **kwargs)

    @contextlib.contextmanager
    def _ctx():
        _dwq.iterate_batches = patched
        try:
            yield
        finally:
            _dwq.iterate_batches = saved

    return _ctx()


def _split_calibration_data(spec: str) -> tuple[str, str | None]:
    """Split a `--calibration-data` value into (hf_path, subset_name).

    Accepts the HF subset-config syntax `path:name` (e.g.
    `Salesforce/wikitext:wikitext-103-raw-v1`) so corpora that require a
    `name=` kwarg to `datasets.load_dataset` are passable from the CLI.
    Plain `path` values (`allenai/tulu-3-sft-mixture`) return
    `(spec, None)` and route through the unmodified dwq load path.
    """
    path, sep, name = spec.partition(":")
    return (path, name) if sep and name else (spec, None)


def _patch_dwq_load_data(
    subset_name: str,
    split: str = "train",
    text_column: str | None = None,
):
    """Context manager: swap mlx_lm.quant.dwq.load_data so the `hf_dataset`
    dict it builds includes `config={"name": subset_name}` and any non-default
    split / text-column overrides.

    `mlx_lm.tuner.datasets.load_custom_hf_dataset` reads `ds.get("config", {})`
    and unpacks it as `**hf_config` into `datasets.load_dataset(...)`, which
    is how HF subset configs (wikitext-103-raw-v1, sample-10BT, etc.) get
    plumbed through. The same dict can carry `train_split` and `text_feature`
    so corpora with non-standard splits (Ultra-FineWeb's `en`/`zh`) or
    non-standard text columns (Ultra-FineWeb's `content`) work without code
    changes upstream. dwq's stock `load_data` doesn't surface these knobs,
    so we replace the function with a copy that does. No-op-ish defaults
    when subset_name is None and split=="train" and text_column is None.
    """
    import contextlib
    import types as _types

    import numpy as np
    from mlx_lm.quant import dwq as _dwq
    from mlx_lm.tuner.datasets import load_dataset as _load_dataset

    def load_data_with_subset(
        tokenizer,
        data_path,
        num_samples,
        max_seq_length,
        num_valid_samples=32,
    ):
        ds_dict = {
            "path": data_path,
            "train_split": split,
            "valid_split": f"{split}[:1]",
            "config": {"name": subset_name},
        }
        if text_column is not None:
            ds_dict["text_feature"] = text_column
        args = _types.SimpleNamespace(
            hf_dataset=ds_dict,
            train=True,
            test=False,
        )
        dataset = _load_dataset(args, tokenizer)[0]
        perm = np.random.permutation(len(dataset))
        train_perm = perm[:num_samples].tolist()
        valid_perm = perm[num_samples : num_samples + num_valid_samples].tolist()

        def process(idx):
            tokens, offset = dataset.process(dataset[idx])
            return (tokens[:max_seq_length], offset)

        return [process(i) for i in train_perm], [process(i) for i in valid_perm]

    @contextlib.contextmanager
    def _ctx():
        saved = _dwq.load_data
        try:
            _dwq.load_data = load_data_with_subset
            yield
        finally:
            _dwq.load_data = saved

    return _ctx()


def _dwq_target_dir_for_spec(
    *,
    teacher_model: str,
    data_path: str,
    num_samples: int,
    max_seq_len: int,
    batch_size: int,
    seed: int,
    subset_name: str | None = None,
    split: str = "train",
    text_column: str | None = None,
) -> Path:
    """Stable per-(teacher × calibration-spec) directory under
    `~/.mlx-dwq-targets/<hash>/` that mlx_lm.dwq can populate with
    pre-computed teacher logits + topk indices.

    The hash mirrors `score-mlx-kld.py`'s teacher-cache scheme so the two
    caches sit side-by-side and behave the same way: same teacher + same
    spec → same directory → reused across runs (cheap sweep iteration);
    different spec → different directory (no false-cache-hit risk).

    Two memory benefits even on a cold target-dir:
      1. mlx_lm.dwq computes targets to disk once, then `del`s the teacher
         model before training begins — frees ~50GB on a 26B teacher,
         ~70GB on a 35B+ teacher, regardless of train-loop batch/seq.
      2. Subsequent runs against the same teacher × calibration spec
         skip the teacher load entirely.

    Disk cost is bounded: top-1024 logits + indices, fp16 + int32, per
    batch. At seq=2048 × samples=2048 × batch=1: ~33GB per spec.
    """
    payload_parts = [
        teacher_model,
        data_path,
        str(num_samples),
        str(max_seq_len),
        str(batch_size),
        str(seed),
    ]
    # Append the new fields only when non-default so cache hashes stay
    # back-compatible with pre-flag runs (no subset, split=train, no
    # text_column override).
    if subset_name is not None or split != "train" or text_column is not None:
        payload_parts.extend(
            [
                subset_name or "",
                split,
                text_column or "",
            ]
        )
    payload = "|".join(payload_parts)
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return Path.home() / ".mlx-dwq-targets" / h


def run_dwq_cascade(
    *,
    teacher_model: str,
    student_dir: Path,
    calibration_data: str,
    num_samples: int,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    target_dir: Path | None,
    split: str = "train",
    text_column: str | None = None,
) -> None:
    """Invoke mlx_lm.dwq programmatically by constructing argv + calling main.

    Cascades on top of the just-saved AP-quantized checkpoint, refining
    per-group scales via KL distillation against the full-precision teacher.

    `calibration_data` accepts `path:name` to plumb HF subset configs (see
    `_split_calibration_data` / `_patch_dwq_load_data`). The subset, when
    present, is stripped from the argv passed to dwq.main and reinjected
    via the load_data patch.

    `batch_size` and `max_seq_len` directly control the dwq train loop's
    memory footprint. mlx_lm.dwq's defaults (4 / 1025) work for small
    teachers but blow the unified-memory budget on 26B+ multimodal teachers
    where the bf16 teacher alone is ~50GB and the train step holds teacher
    forward + student forward + student backward simultaneously.

    `target_dir` (default-on, derived from `_dwq_target_dir_for_spec`) is
    the bigger memory lever: dwq computes teacher targets to disk once,
    then deletes the teacher model before training. Removes the teacher
    weights from peak memory entirely; subsequent runs against the same
    spec skip the teacher load.

    Multimodal teacher: dwq's load/save are routed through mlx-vlm and the
    forward calls get a fresh KV cache injected (see `_wrap_with_cache_injection`
    and `_patch_dwq_for_multimodal`). Text-only teacher: the cascade runs
    bit-identical to the pre-multimodal-dispatch path.
    """
    import contextlib

    from mlx_lm.quant import dwq
    from mlx_vlm.utils import get_model_path

    teacher_path = get_model_path(teacher_model)
    teacher_cfg = json.loads((teacher_path / "config.json").read_text())
    multimodal = is_multimodal(teacher_cfg)

    data_path, subset_name = _split_calibration_data(calibration_data)

    if target_dir is None:
        target_dir = _dwq_target_dir_for_spec(
            teacher_model=teacher_model,
            data_path=data_path,
            num_samples=num_samples,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            seed=seed,
            subset_name=subset_name,
            split=split,
            text_column=text_column,
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "mlx_lm.dwq",
        "--model",
        teacher_model,
        "--quantized-model",
        str(student_dir),
        "--mlx-path",
        str(student_dir),
        "--data-path",
        data_path,
        "--num-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--max-seq-length",
        str(max_seq_len),
        "--target-dir",
        str(target_dir),
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        with contextlib.ExitStack() as stack:
            stack.enter_context(_patch_dwq_iterate_batches_seed())
            if multimodal:
                info("DWQ cascade: multimodal dispatch (mlx-vlm load/save + cache injection)")
                stack.enter_context(_patch_dwq_for_multimodal())
            if subset_name is not None or split != "train" or text_column is not None:
                bits = []
                if subset_name is not None:
                    bits.append(f"subset={subset_name!r}")
                if split != "train":
                    bits.append(f"split={split!r}")
                if text_column is not None:
                    bits.append(f"text_column={text_column!r}")
                info(f"DWQ cascade: HF dataset overrides ({', '.join(bits)})")
                stack.enter_context(
                    _patch_dwq_load_data(
                        subset_name,
                        split=split,
                        text_column=text_column,
                    )
                )
            dwq.main()
    finally:
        sys.argv = saved


# ---------- argument parsing ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mqt-quantize",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="HF id or local path of the source (full-precision) model")

    g = p.add_argument_group("recipe options")
    g.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Base bits for non-floored layers (default: 4). "
        "At bits >= 8, rules 3 and 4 become no-ops.",
    )
    g.add_argument(
        "--group-size",
        type=int,
        default=64,
        choices=[32, 64, 128],
        help="Group size for affine quantization (default: 64)",
    )
    g.add_argument(
        "--attn-protect-mode",
        choices=["bf16", "8bit"],
        default="bf16",
        help="Bit policy for Rule 1 (linear_attn.*) and Rule 2 (attn.out_proj). "
        "bf16 (default): unquantized. 8bit: max(--bits, 8) affine.",
    )

    g = p.add_argument_group("DWQ cascade")
    g.add_argument(
        "--with-dwq",
        action="store_true",
        help="Cascade mlx_lm.dwq on the quantized output (default: off)",
    )
    g.add_argument(
        "--calibration-data",
        default="allenai/tulu-3-sft-mixture",
        help="Calibration corpus for DWQ. Default matches mlx_lm.dwq's "
        "own default (in-distribution for instruct deployment; "
        "requires a chat template on the teacher). Accepts "
        "`path:name` for HF subset configs, e.g. "
        "`Salesforce/wikitext:wikitext-103-raw-v1`.",
    )
    g.add_argument(
        "--calibration-samples",
        type=int,
        default=2048,
        help="Number of calibration samples (default: 2048, mlx_lm.dwq's "
        "own default; matches the published-DWQ-numbers protocol). "
        "Lower to 512 for sweep iteration — empirically only "
        "~0.03 nats worse on a 26B teacher per the gemma-4-26B "
        "anchor in docs/attn-protect-quantize-plan.md.",
    )
    g.add_argument(
        "--calibration-seed",
        type=int,
        default=0,
        help="Seed for calibration sample selection (default: 0). DWQ-only.",
    )
    g.add_argument(
        "--calibration-batch-size",
        type=int,
        default=4,
        help="DWQ train batch size (default: 4, mlx_lm.dwq's own default). "
        "Lower this on 26B+ teachers to fit unified memory; the train "
        "step holds teacher fwd + student fwd + student bwd at once.",
    )
    g.add_argument(
        "--calibration-max-seq-len",
        type=int,
        default=1025,
        help="DWQ max sequence length (default: 1025, mlx_lm.dwq's own default). "
        "Memory scales linearly; halving cuts activation memory ~2x.",
    )
    g.add_argument(
        "--calibration-split",
        default="train",
        help="HF dataset split name for calibration data (default: train). "
        "Override for corpora with non-standard splits, e.g. "
        "`openbmb/Ultra-FineWeb` uses split=`en` (English) / `zh`.",
    )
    g.add_argument(
        "--calibration-text-column",
        default=None,
        help="HF dataset column name to read calibration text from. "
        "Default: auto-detect — mlx_lm.tuner.datasets routes "
        "`messages` (chat) / `prompt`+`completion` (sft) / `text` (raw). "
        "Override only for corpora with a non-standard text column, "
        "e.g. `openbmb/Ultra-FineWeb` uses column=`content`.",
    )
    g.add_argument(
        "--dwq-target-dir",
        type=Path,
        default=None,
        help="Override the auto-derived ~/.mlx-dwq-targets/<hash>/ "
        "directory for pre-computed teacher logits. The default "
        "auto-derives a stable per-(teacher × calibration-spec) "
        "path so sweep iterations cache-hit. Passing an empty "
        "/ non-existent dir starts a fresh teacher pass.",
    )

    g = p.add_argument_group("ablation flags (research-only; default = full attention protection)")
    g.add_argument(
        "--quantize-linear-attn",
        action="store_true",
        help="Ablation: quantize linear_attn at base bits (disables Rule 1 protection)",
    )
    g.add_argument(
        "--quantize-attn-out",
        action="store_true",
        help="Ablation: quantize attn.out_proj at base bits (disables Rule 2 protection)",
    )
    g.add_argument(
        "--no-attn-floor",
        action="store_true",
        help="Ablation: q/k/v/o get base bits, no 8-bit floor (disables Rule 3)",
    )
    g.add_argument(
        "--no-lm-head-floor",
        action="store_true",
        help="Ablation: lm_head gets base bits, no 8-bit floor (disables Rule 4). "
        "Also disables --floor-tied-embed (logically the same rule).",
    )
    g.add_argument(
        "--quantize-moe-router",
        action="store_true",
        help="Ablation: quantize MoE router.proj / mlp.gate / "
        "shared_expert_gate at base bits (disables Rule 5). "
        "Research-only — router weight noise corrupts top-K "
        "expert routing.",
    )
    g.add_argument(
        "--floor-tied-embed",
        action="store_true",
        help="For tied-embedding models (no separate lm_head), "
        "apply the Rule 4 8-bit floor to embed_tokens instead. "
        "Silently ignored on models with untied embeddings. "
        "Default off; flip via KLD A/B once score-mlx-kld.py exists.",
    )
    g.add_argument(
        "--protect-vlm",
        action="store_true",
        help="For multimodal checkpoints, keep vision/audio tower and "
        "multi-modal projector modules at the --attn-protect-mode "
        "verdict (default: bf16). No-op on text-only models. "
        "Required to produce a working VLM quant; load/save are "
        "automatically routed through mlx-vlm when the source "
        "config has vision_config or audio_config.",
    )
    g.add_argument(
        "--bf16-routed-experts",
        action="store_true",
        help="Diagnostic (P2.5.0): leave routed-expert weight tensors "
        "at bf16 (skip quantization) on MoE models. Matches "
        "SwitchLinear-packed `experts.switch_glu.{gate,up,down}_proj` "
        "and per-expert `mlp.experts.<i>.{gate,up,down}_proj`. "
        "Bounds the cold-expert-quantization-variance lever from "
        "above: KLD vs bf16 teacher with this flag set tells you "
        "the absolute ceiling of any per-expert bf16-cold scheme. "
        "Heavy on disk (most params are experts on MoE) — for "
        "diagnostic builds only.",
    )
    g.add_argument(
        "--bf16-embed-tokens",
        action="store_true",
        help="Diagnostic: leave embed_tokens (and lm_head when present) "
        "at bf16. Targets the dominant single-tensor 8-bit cost on "
        "tied-embed large-vocab models (V=262k Gemma-4 = 738M params "
        "in one tensor; on tied models this same tensor is the "
        "lm_head matmul, so 8-bit error is spent twice per token). "
        "Use to test whether the elevated AP8bit ceiling on a given "
        "checkpoint is dominated by embed/lm_head quantization vs "
        "attention or shared-MLP quantization.",
    )
    g.add_argument(
        "--bf16-attn-floor",
        action="store_true",
        help="Diagnostic: skip quantization on the q/k/v/o attention "
        "projections that Rule 3 normally floors at 8-bit, leaving "
        "them at bf16. Distinct from --no-attn-floor (which sends "
        "QKVO to base bits). Use as a non-MoE-specific ablation "
        "to localize whether the elevated AP8bit ceiling on a "
        "given checkpoint is from attention vs shared-MLP "
        "quantization.",
    )
    g.add_argument(
        "--bf16-shared-mlp",
        action="store_true",
        help="Diagnostic: skip quantization on shared-dense MLP "
        "projections (the same tensor set Tier 2 mlp_boosts "
        "would target — `mlp.{gate,up,down}_proj` and Qwen-style "
        "`shared_expert.{gate,up,down}_proj`). On gemma-4-MoE "
        "this is the parallel dense MLP that runs unconditionally "
        "every layer alongside the routed experts; bf16'ing it "
        "completes the AP8bit ceiling-elevation diagnostic chain "
        "(experts/embed/attn/shared-MLP). Heavy for non-MoE "
        "checkpoints where this is the primary MLP path.",
    )
    g.add_argument(
        "--local-attn-mode",
        choices=["bf16", "8bit", "base"],
        default=None,
        help="P3.1: per-layer-type override for QKVO floor on architectures "
        "with sliding/full attention split (Gemma-3, Gemma-4 — the "
        "config exposes `layer_types`). 'bf16' is the default attn-floor "
        "behavior (no-op). '8bit' drops sliding-attention QKVO from bf16 "
        "to 8-bit affine while keeping global (full_attention) layers at "
        "bf16 — saves ~0.22 bpw on gemma-4-26B-A4B-it (25 sliding × 28.8M "
        "vs 5 global × 48.9M params). 'base' drops sliding QKVO to base "
        "bits, joining the DWQ pool. No-op when config has no layer_types "
        "or no sliding_attention entries.",
    )

    g = p.add_argument_group("Tier 2: MLP boost allocator")
    g.add_argument(
        "--with-mlp-boosts",
        action="store_true",
        help="Score per-tensor sensitivity (KLD recovery vs the bf16 "
        "teacher) and selectively boost dense MLP projections "
        "from base bits to base+1 bits within --mlp-boost-budget-bpw. "
        "Adds a calibration pass; reuses score-mlx-kld.py's "
        "teacher cache. See docs/attn-protect-quantize-plan.md "
        "for the trigger rule (within-model q4-AP-flat-dwq vs "
        "q8-AP-flat ceiling > 0.10 nats).",
    )
    g.add_argument(
        "--mlp-boost-budget-bpw",
        type=float,
        default=0.05,
        help="Maximum extra bits-per-weight the MLP-boost allocator "
        "may spend on top of the AP baseline (default: 0.05, "
        "matching UD-MLX-4bit's empirical bpw delta on Qwen3.6-27B). "
        "Pass a larger value to allow more boosts; 0 disables.",
    )
    g.add_argument(
        "--mlp-boost-group-size",
        type=int,
        default=None,
        choices=[32, 64, 128],
        help="Group size for boost-tensor pre-quantize and verdicts. "
        "Defaults to --group-size if unset. Smaller boost groups "
        "(e.g. --group-size 64 --mlp-boost-group-size 32) selectively "
        "fine-grain only the boosted tensors at minimal bpw cost — "
        "isolates the per-tensor finer-scale lever from a uniform "
        "smaller-base-group choice.",
    )
    g.add_argument(
        "--mlp-boost-calibration-data",
        default="wikitext-2-raw-v1",
        help="Calibration corpus for sensitivity scoring (default: "
        "wikitext-2-raw-v1, matching score-mlx-kld.py's default "
        "so the teacher top-K cache amortizes across both passes). "
        "Accepts `path:name` for HF subset configs and chat-format "
        "corpora (`messages` column) — those auto-render through "
        "the teacher's chat template.",
    )
    g.add_argument(
        "--mlp-boost-calibration-samples",
        type=int,
        default=16,
        help="Number of calibration samples for sensitivity scoring "
        "(default: 16). Sensitivity *ranking* is robust to small "
        "sample counts; absolute KLD precision matters less than "
        "ordering. Cost scales linearly: ~25 sec/sample on a 26B MoE.",
    )
    g.add_argument(
        "--mlp-boost-calibration-max-seq-len",
        type=int,
        default=1024,
        help="Sequence length for sensitivity calibration (default: 1024).",
    )
    g.add_argument(
        "--mlp-boost-calibration-seed",
        type=int,
        default=123,
        help="Calibration seed for sensitivity scoring (default: 123, "
        "matching score-mlx-kld.py's default for cache parity).",
    )
    g.add_argument(
        "--mlp-boost-baseline-from",
        type=Path,
        default=None,
        help="Path to an existing AP-quantized + DWQ'd checkpoint to use as "
        "the sensitivity-scoring baseline (post-DWQ sensitivity). "
        "Pre-quantized 5-bit boost candidates are derived from --model "
        "(bf16 source) but recovery is measured against the loaded "
        "baseline, so the score answers 'what's the marginal value of "
        "boosting this tensor *on top of* DWQ?' Boosts then replace the "
        "matching 4-bit DWQ'd modules in the baseline in place; no "
        "second DWQ pass — the boosted tensors stay un-DWQ'd. "
        "Requires --with-mlp-boosts; conflicts with --with-dwq "
        "(would re-DWQ the baseline and undo the existing refinement).",
    )
    g.add_argument(
        "--mlp-boost-candidates",
        choices=[_BOOST_TYPE_BIT_BUMP, _BOOST_TYPE_GS_REFINE, "both"],
        default="both",
        help="Which Tier 2 candidate types the allocator scores per tensor "
        "(default: both). 'bit-bump' = +1 bit at the same group_size "
        "(legacy behavior; ~1.0 bpw cost). 'gs-refine' = same bits at "
        "halved group_size (~0.5 bpw cost on q4 gs=64→gs=32). 'both' "
        "scores each tensor under both rules and picks the densest "
        "candidate per tensor. P4.1 step 0.",
    )
    g.add_argument(
        "--mlp-boost-noise-floor",
        type=float,
        default=0.002,
        help="Track 4 §3 noise floor: skip allocator candidates whose "
        "measured KLD recovery is below this many nats (default: "
        "0.002). Above the floor, density `recovery/cost` ranking "
        "is well-conditioned; below it, the smaller-cost variant "
        "wins systematically on noise.",
    )

    g = p.add_argument_group("output")
    g.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: ./<basename>-AP<bits>bit[-8bit][-dwq][-gs<N>][-mlpboost]",
    )
    g.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Computation dtype while quantizing (default: bfloat16)",
    )

    g = p.add_argument_group("operational")
    g.add_argument(
        "--dry-run", action="store_true", help="Print the per-module recipe without quantizing"
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    import mlx.core as mx
    from mlx.utils import tree_map_with_path
    from mlx_lm.utils import quantize_model
    from mlx_vlm.utils import get_model_path

    # Resolve the source to a local snapshot path (handles HF download for
    # both text-only and VLM models). mlx_vlm.utils.get_model_path pulls
    # the full snapshot when needed, which is what the multimodal save
    # path requires for copying auxiliary files.
    src_path = get_model_path(args.model)
    early_config = json.loads((src_path / "config.json").read_text())

    multimodal = is_multimodal(early_config)
    if multimodal and not args.protect_vlm:
        sys.exit(
            f"Source {args.model!r} is multimodal "
            f"(config has non-empty vision_config/audio_config) but "
            f"--protect-vlm was not set.\n"
            f"Pass --protect-vlm to produce a working VLM quant; without it, "
            f"vision/audio tower modules would be quantized along with the "
            f"language model, which corrupts the multimodal forward pass."
        )

    # Variant C (post-DWQ sensitivity) gating. Surface the validation up
    # front so the user catches mistakes before a 38-min sensitivity loop.
    if args.mlp_boost_baseline_from is not None:
        if not args.with_mlp_boosts:
            sys.exit("--mlp-boost-baseline-from requires --with-mlp-boosts")
        if args.with_dwq:
            sys.exit(
                "--mlp-boost-baseline-from cannot combine with --with-dwq: a "
                "second DWQ pass would refine and overwrite the existing "
                "DWQ'd scales/biases on the un-boosted tensors of the "
                "baseline, defeating the point of starting from one."
            )
        if args.out is None:
            sys.exit(
                "--mlp-boost-baseline-from requires an explicit --out: the "
                "default output path doesn't encode 'sensitivity scored "
                "post-DWQ', so the resulting checkpoint would be ambiguous "
                "with a normal Tier 2 run."
            )
        if not args.mlp_boost_baseline_from.exists():
            sys.exit(
                f"--mlp-boost-baseline-from path does not exist: {args.mlp_boost_baseline_from}"
            )

    if multimodal and args.with_dwq:
        # Multimodal DWQ dispatch is wired up via _patch_dwq_for_multimodal()
        # in run_dwq_cascade: dwq.load/save route through mlx-vlm, and the
        # teacher/student forwards get a fresh KV cache injected (otherwise
        # gemma-4 emits garbage logits and KL distillation trains against
        # noise — same issue as score-mlx-kld step 4).
        info("Multimodal DWQ dispatch active (mlx-vlm load/save + cache injection)")

    tied = bool(early_config.get("tie_word_embeddings", False))
    if args.floor_tied_embed and not tied:
        info("--floor-tied-embed has no effect: model has untied embeddings (separate lm_head)")
    args.floor_tied_embed_effective = args.floor_tied_embed and tied

    # Resolve the boost group_size default (None → match base --group-size).
    # When the two differ, the boost map encodes its own per-tensor group_size
    # (the predicate consumes the verdict verbatim) and the AP-quantize step
    # still uses --group-size for everything else.
    if args.mlp_boost_group_size is None:
        args.mlp_boost_group_size = args.group_size
    if args.with_mlp_boosts and args.mlp_boost_group_size > args.group_size:
        info(
            f"WARNING: --mlp-boost-group-size {args.mlp_boost_group_size} > "
            f"--group-size {args.group_size}: coarser groups on boosted tensors "
            f"than on the base wastes bpw vs the base. Continuing anyway."
        )

    out_dir = args.out or default_output_path(args.model, args)
    if not args.dry_run and out_dir.exists():
        sys.exit(f"Output path already exists: {out_dir}\nDelete it or pick a different --out.")

    if multimodal:
        info(f"Loading {args.model} via mlx-vlm (multimodal)")
        from mlx_vlm.utils import fetch_from_hub

        model, config, processor = fetch_from_hub(src_path, lazy=True)
        tokenizer_or_processor = processor
    else:
        info(f"Loading {args.model} via mlx-lm")
        from mlx_lm.utils import load

        model, tokenizer, config = load(args.model, return_config=True, lazy=True)
        tokenizer_or_processor = tokenizer

    dtype = getattr(mx, args.dtype)
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def set_dtype(k, v):
        if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
            return v.astype(dtype)
        return v

    model.update(tree_map_with_path(set_dtype, model.parameters()))

    # P3.1: layer_types lives at config root on text-only models, under
    # `text_config` on multimodal Gemma-4 (vision/audio sub-configs above it).
    # Resolve once here; predicate factory is a no-op when layer_types is None.
    _text_cfg = early_config.get("text_config", early_config) or {}
    _layer_types = _text_cfg.get("layer_types")

    predicate = make_attn_protect_predicate(
        bits=args.bits,
        group_size=args.group_size,
        attn_protect_mode=args.attn_protect_mode,
        quantize_linear_attn=args.quantize_linear_attn,
        quantize_attn_out=args.quantize_attn_out,
        no_attn_floor=args.no_attn_floor,
        no_lm_head_floor=args.no_lm_head_floor,
        quantize_moe_router=args.quantize_moe_router,
        floor_tied_embed=args.floor_tied_embed_effective,
        protect_vlm=args.protect_vlm,
        bf16_routed_experts=args.bf16_routed_experts,
        bf16_embed_tokens=args.bf16_embed_tokens,
        bf16_attn_floor=args.bf16_attn_floor,
        bf16_shared_mlp=args.bf16_shared_mlp,
        local_attn_mode=args.local_attn_mode,
        layer_types=_layer_types,
    )

    if args.dry_run:
        print_recipe(model, predicate)
        return

    # Tier 2: capture pre-quantize candidate references BEFORE quantize_model
    # mutates the module tree. score_mlp_sensitivity will use the bf16
    # versions to construct boosted modules at each candidate's target
    # (bits, group_size); AP-quantize then runs in place and replaces the
    # originals with QuantizedLinear at base bits.
    boost_candidates: list[dict] = []
    total_quant_params = 0
    if args.with_mlp_boosts and args.bits < 8:
        cand_mode = args.mlp_boost_candidates
        if cand_mode == "both":
            candidate_types = (_BOOST_TYPE_BIT_BUMP, _BOOST_TYPE_GS_REFINE)
        else:
            candidate_types = (cand_mode,)
        boost_candidates = find_mlp_boost_candidates(
            model,
            base_bits=args.bits,
            base_group_size=args.group_size,
            candidate_types=candidate_types,
            bit_bump_group_size=args.mlp_boost_group_size,
        )
        n_paths = len({c["path"] for c in boost_candidates})
        info(
            f"Tier 2: found {len(boost_candidates)} candidates "
            f"({n_paths} paths × types={candidate_types})"
        )
        # Walk the bf16 model and count weights on modules that the predicate
        # will quantize (predicate returns dict or True; False means bf16 skip).
        # Computed pre-quantize because `to_quantized` is gone after the call.
        for path, m in model.named_modules():
            if not (hasattr(m, "to_quantized") and hasattr(m, "weight")):
                continue
            verdict = predicate(path, m)
            if verdict is not False:
                total_quant_params += int(m.weight.size)
    elif args.with_mlp_boosts:
        info("--with-mlp-boosts skipped: nothing meaningful to boost at bits>=8")

    pre_quantized_boosts: dict | None = None
    if args.mlp_boost_baseline_from is not None:
        # Variant C: pre-quantize boost candidates from bf16 source NOW (while
        # bf16 weights are still in memory), then drop the bf16 source and load
        # the DWQ'd baseline as the model to score against. The pre-computed
        # boosted modules survive the source teardown — they're standalone
        # QuantizedLinears with no reference back to the bf16 graph.
        info(
            f"Variant C: pre-quantizing {len(boost_candidates)} candidates "
            f"(per-candidate targets, bf16 still loaded)"
        )
        pre_quantized_boosts = {}
        for cand in boost_candidates:
            boosted = cand["module"].to_quantized(
                group_size=cand["target_group_size"],
                bits=cand["target_bits"],
            )
            mx.eval(boosted.parameters())
            pre_quantized_boosts[cand["id"]] = boosted
        mx.eval(pre_quantized_boosts)

        # Free bf16 source. Loading the (~16 GB) baseline on top of the
        # (~50 GB) bf16 source on a 26B target would OOM the M5 Max.
        info("Freeing bf16 source; loading post-DWQ baseline as scoring model")
        del model
        del tokenizer_or_processor
        import gc

        gc.collect()
        mx.metal.clear_cache()

        baseline_path = args.mlp_boost_baseline_from
        if multimodal:
            from mlx_vlm.utils import fetch_from_hub

            model, new_config, processor = fetch_from_hub(baseline_path, lazy=False)
            tokenizer_or_processor = processor
        else:
            from mlx_lm.utils import load

            model, tokenizer, new_config = load(
                baseline_path,
                return_config=True,
                lazy=False,
            )
            tokenizer_or_processor = tokenizer
        # Sanity: the baseline must already be quantized; we'll be swapping
        # boosted modules in *on top of* its existing quantization map.
        if "quantization" not in new_config:
            sys.exit(
                f"--mlp-boost-baseline-from {baseline_path} has no "
                "'quantization' entry in config.json — not a quantized "
                "checkpoint."
            )
    else:
        info(
            f"Quantizing (bits={args.bits}, group_size={args.group_size}, "
            f"attn_protect_mode={args.attn_protect_mode}, "
            f"multimodal={multimodal})"
        )
        model, new_config = quantize_model(
            model,
            config,
            group_size=args.group_size,
            bits=args.bits,
            quant_predicate=predicate,
        )

    # Tier 2: score sensitivity, allocate boosts, apply them in-place.
    boosts: dict = {}
    sensitivity: dict | None = None
    # bit_budget is real-valued — gs-refine carries fractional delta_bpw (~0.5
    # at q4 gs=64→gs=32). Allocator math is in bpw·params units throughout.
    bit_budget = float(args.mlp_boost_budget_bpw) * float(total_quant_params)
    if boost_candidates:
        info(
            f"Tier 2: bit_budget={bit_budget:.0f} bits "
            f"(={args.mlp_boost_budget_bpw:.4f} bpw × {total_quant_params:,} "
            f"quantized params)"
        )
        sensitivity = score_mlp_sensitivity(
            model=model,
            candidates=boost_candidates,
            teacher_path=args.model,
            calibration_data=args.mlp_boost_calibration_data,
            num_samples=args.mlp_boost_calibration_samples,
            max_seq_len=args.mlp_boost_calibration_max_seq_len,
            seed=args.mlp_boost_calibration_seed,
            pre_quantized_boosts=pre_quantized_boosts,
        )
        boosts = allocate_mlp_boosts(
            candidates=boost_candidates,
            recovery=sensitivity["recovery"],
            tensor_params=sensitivity["tensor_params"],
            bit_budget=bit_budget,
            recovery_noise_floor=float(args.mlp_boost_noise_floor),
        )
        type_summary = {}
        for v in boosts.values():
            t = v.get("candidate_type", _BOOST_TYPE_BIT_BUMP)
            type_summary[t] = type_summary.get(t, 0) + 1
        info(
            f"Tier 2: allocator selected {len(boosts)} of "
            f"{len(boost_candidates)} candidates "
            f"(by type: {type_summary})"
        )

        # Apply boosts: swap in the higher-precision modules and update the
        # config so mlx-lm's save records the per-tensor bits correctly.
        # `candidate_type` is metadata for our recipe.json — strip it before
        # writing to qcfg, which mlx-lm serializes into config.json.
        if boosts:
            qcfg = new_config.setdefault("quantization", {})
            for path, verdict in boosts.items():
                cand_id = _candidate_id(path, verdict["candidate_type"])
                _set_module(model, path, sensitivity["boosted_modules"][cand_id])
                qcfg[path] = {k: v for k, v in verdict.items() if k != "candidate_type"}

    info(f"Saving to {out_dir}")
    if multimodal:
        save_multimodal(out_dir, src_path, model, tokenizer_or_processor, new_config)
    else:
        from mlx_lm.utils import save

        save(out_dir, args.model, model, tokenizer_or_processor, new_config)
    write_recipe_provenance(
        out_dir,
        args,
        boosts=boosts or None,
        boost_meta=(
            {
                "baseline_kld": sensitivity["baseline_kld"],
                "candidate_count": len(boost_candidates),
                "bit_budget": bit_budget,
            }
            if sensitivity
            else None
        ),
    )
    if sensitivity is not None:
        write_sensitivity_provenance(
            out_dir,
            args=args,
            sensitivity=sensitivity,
            bit_budget=bit_budget,
        )

    if args.with_dwq:
        if args.bits >= 8:
            info("--with-dwq skipped: nothing meaningful to refine at bits>=8")
        else:
            info(
                f"Cascading DWQ (samples={args.calibration_samples}, data={args.calibration_data})"
            )
            marker = _write_dwq_marker(out_dir, args)
            try:
                run_dwq_cascade(
                    teacher_model=args.model,
                    student_dir=out_dir,
                    calibration_data=args.calibration_data,
                    num_samples=args.calibration_samples,
                    seed=args.calibration_seed,
                    batch_size=args.calibration_batch_size,
                    max_seq_len=args.calibration_max_seq_len,
                    target_dir=args.dwq_target_dir,
                    split=args.calibration_split,
                    text_column=args.calibration_text_column,
                )
            except Exception as exc:
                # DWQ failed mid-cascade via Python exception. Reconcile the
                # recipe to AP-only and clear the marker before re-raising.
                # Note: the recipe was already written with with_dwq=False at
                # save time (dwq_completed defaults to False); this call adds
                # the dwq_failed=true forensic flag.
                _mark_recipe_dwq_failed(out_dir, f"{type(exc).__name__}: {exc}")
                _clear_dwq_marker(marker)
                raise
            # Cascade completed successfully. Rewrite the recipe with
            # with_dwq=True (now that we know the cascade actually ran to
            # completion) and remove the marker. If we crash between here and
            # the rewrite, the marker survives and the next quantize against
            # this directory will reconcile it; the worst outcome is a recipe
            # with with_dwq=False on weights that *did* receive DWQ — under-
            # reporting rather than over-reporting, which is the safer
            # direction.
            write_recipe_provenance(
                out_dir,
                args,
                boosts=boosts or None,
                boost_meta=(
                    {
                        "baseline_kld": sensitivity["baseline_kld"],
                        "candidate_count": len(boost_candidates),
                        "bit_budget": bit_budget,
                    }
                    if sensitivity
                    else None
                ),
                dwq_completed=True,
            )
            _clear_dwq_marker(marker)

    info(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
