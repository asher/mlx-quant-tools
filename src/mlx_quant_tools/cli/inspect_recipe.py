#!/usr/bin/env python
"""Inspect an MLX checkpoint to extract its full quantization recipe.

Outputs:
  - A human-readable Markdown report (stdout by default; --md FILE to a file)
  - A complete JSON dump (--json FILE) suitable for programmatic reuse

What it captures (everything you need to reimplement):
  - Global quant defaults (bits, group_size, mode) from config.json
  - Per-module bits and group_size for every quantized tensor
  - Tensors NOT in the quantization config (these are "protected" — kept at
    full precision); their actual on-disk dtype (bf16/fp16/fp32)
  - Aggregated bit distribution by structural role (embed, lm_head, attn.*,
    mlp.*, experts.*, router, norm, visual.*)
  - Per-layer recipe (which layer indices got which bits where)
  - Effective bits-per-weight (bpw) per module and for the whole model,
    counting scales/biases overhead
  - "Outlier" modules whose settings deviate from the defaults — the recipe's
    actual editorial decisions

Usage:
  inspect-mlx-quant-recipe.py <mlx-model-dir> [--md out.md] [--json out.json]
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open

# ---------- role classification ----------

# Order matters: more specific patterns first.
_ROLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("embedding", re.compile(r"\b(embed_tokens|wte|word_embeddings)\b")),
    ("lm_head", re.compile(r"(^|\.)lm_head$")),
    ("attn.q_proj", re.compile(r"\.q_proj$")),
    ("attn.k_proj", re.compile(r"\.k_proj$")),
    ("attn.v_proj", re.compile(r"\.v_proj$")),
    ("attn.o_proj", re.compile(r"\.o_proj$")),
    ("attn.qkv_proj", re.compile(r"\.(qkv_proj|qkv)$")),
    ("attn.out_proj", re.compile(r"\.(out_proj|attn\.proj)$")),
    ("linear_attn.in_proj_qkv", re.compile(r"\.linear_attn\.in_proj_qkv$")),
    ("linear_attn.in_proj_z", re.compile(r"\.linear_attn\.in_proj_z$")),
    ("linear_attn.out_proj", re.compile(r"\.linear_attn\.out_proj$")),
    ("linear_attn.conv1d", re.compile(r"\.linear_attn\.conv1d$")),
    ("linear_attn.dt_proj", re.compile(r"\.linear_attn\.dt_proj$")),
    ("linear_attn.norm", re.compile(r"\.linear_attn\.norm")),
    ("linear_attn.other", re.compile(r"\.linear_attn\.")),
    ("mlp.gate_proj", re.compile(r"\.mlp\.gate_proj$")),
    ("mlp.up_proj", re.compile(r"\.mlp\.up_proj$")),
    ("mlp.down_proj", re.compile(r"\.mlp\.down_proj$")),
    ("mlp.gate_up_proj", re.compile(r"\.mlp\.gate_up_proj$")),
    (
        "experts.gate_up",
        re.compile(r"\.experts\.(switch_mlp\.)?(gate_up_proj|gate_proj|up_proj)$"),
    ),
    ("experts.down", re.compile(r"\.experts\.(switch_mlp\.)?down_proj$")),
    ("router", re.compile(r"\.(router|gate)\.(weight|w[12])$|\.router$|\.gate$")),
    ("shared_expert", re.compile(r"\.shared_expert(\.|_gate)")),
    ("norm", re.compile(r"(\.norm$|\.layernorm$|_norm$|_layernorm$|\.rmsnorm$)")),
    ("vlm.merger", re.compile(r"(^|\.)(merger|connector|multi_modal_projector|vl_connector)\.")),
    ("vlm.patch_embed", re.compile(r"\.patch_embed")),
    ("vlm.visual", re.compile(r"(^|\.)(visual|vision_tower|vision_model)\.")),
    ("vlm.audio", re.compile(r"(^|\.)(audio_tower|audio_model)\.")),
    # Gemma-4-E uses `embed_vision` / `embed_audio` for the multimodal
    # projector; classify these alongside the merger family.
    ("vlm.projector", re.compile(r"(^|\.)(embed_vision|embed_audio)\.")),
]


def classify_role(base: str) -> str:
    for role, pat in _ROLE_PATTERNS:
        if pat.search(base):
            return role
    return "other"


# Extract integer layer index from a module path, or None.
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index(base: str) -> int | None:
    m = _LAYER_RE.search(base)
    return int(m.group(1)) if m else None


# ---------- safetensors metadata walk ----------


def collect_tensor_metadata(model_dir: Path) -> dict[str, dict]:
    """Return {tensor_name: {dtype, shape, nbytes, shard}} for every tensor."""
    index_path = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"

    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif single.exists():
        shards = ["model.safetensors"]
    else:
        sys.exit(f"No safetensors found in {model_dir}")

    meta: dict[str, dict] = {}
    for shard in shards:
        path = model_dir / shard
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                slice_view = f.get_slice(key)
                shape = list(slice_view.get_shape())
                dtype = slice_view.get_dtype()
                # Header reports byte size accurately without reading the data.
                # Compute it from shape + dtype since safetensors doesn't expose
                # tensor nbytes directly via the slice API.
                meta[key] = {
                    "dtype": dtype,
                    "shape": shape,
                    "shard": shard,
                    "nbytes": _compute_nbytes(shape, dtype),
                }
    return meta


_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _compute_nbytes(shape: list[int], dtype: str) -> int:
    elem_bytes = _DTYPE_BYTES.get(dtype, 0)
    if not elem_bytes:
        return 0
    return math.prod(shape) * elem_bytes


# ---------- module grouping ----------


def group_into_modules(meta: dict[str, dict]) -> dict[str, dict]:
    """Group .weight + .scales + .biases triples under a shared base name.

    Returns {base: {"weight": tensor_meta, "scales": tensor_meta?, "biases": tensor_meta?}}.
    Tensors without one of these three suffixes (e.g. layernorms exposed as
    just `<name>.weight`) still produce a base entry with weight only.
    """
    modules: dict[str, dict] = {}
    for name, info in meta.items():
        for suffix in (".weight", ".scales", ".biases"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                kind = suffix[1:]  # strip leading dot
                modules.setdefault(base, {})[kind] = info
                break
        else:
            # No quantization-related suffix — store as standalone for completeness.
            modules.setdefault(name, {"weight": info})
    return modules


# ---------- recipe extraction ----------


def derive_module_recipe(
    base: str,
    parts: dict,
    quant_cfg: dict,
    default_bits: int | None,
    default_group_size: int | None,
) -> dict:
    """Determine bits, group_size, dtype, param count, effective bpw."""
    weight = parts.get("weight")
    scales = parts.get("scales")
    biases = parts.get("biases")

    is_quantized = scales is not None
    declared = quant_cfg.get(base) if isinstance(quant_cfg.get(base), dict) else None

    info: dict[str, Any] = {
        "base": base,
        "role": classify_role(base),
        "layer": layer_index(base),
        "quantized": is_quantized,
        "declared_in_config": declared is not None,
        "declared_bits": declared["bits"] if declared else None,
        "declared_group_size": declared["group_size"] if declared else None,
        "weight_dtype": weight["dtype"] if weight else None,
        "weight_shape": weight["shape"] if weight else None,
        "scales_dtype": scales["dtype"] if scales else None,
        "scales_shape": scales["shape"] if scales else None,
        "biases_dtype": biases["dtype"] if biases else None,
        "biases_shape": biases["shape"] if biases else None,
    }

    total_bytes = sum(p["nbytes"] for p in parts.values() if p)
    info["total_bytes"] = total_bytes

    # Param count: derived from the unquantized shape. For a quantized linear
    # layer, the on-disk weight is packed along the last axis; the original
    # in_features is reconstructable from scales.shape[-1] * group_size, or
    # equivalently weight.shape[-1] * 32 / bits.
    param_count = None
    inferred_bits = None
    if is_quantized and weight and scales:
        gs = info["declared_group_size"] or default_group_size or 64
        in_features = scales["shape"][-1] * gs
        out_dims = weight["shape"][:-1]
        param_count = math.prod(out_dims) * in_features
        # Sanity: derive bits independently from the packed shape.
        try:
            inferred_bits = weight["shape"][-1] * 32 / in_features
        except ZeroDivisionError:
            inferred_bits = None
    elif weight:
        param_count = math.prod(weight["shape"])

    info["param_count"] = param_count
    info["inferred_bits"] = inferred_bits
    info["effective_bpw"] = (total_bytes * 8 / param_count) if param_count else None
    return info


# ---------- aggregation ----------


def aggregate_by_role(modules: list[dict]) -> dict[str, dict]:
    by_role: dict[str, dict] = {}
    for m in modules:
        role = m["role"]
        slot = by_role.setdefault(
            role,
            {
                "module_count": 0,
                "param_count": 0,
                "total_bytes": 0,
                "bit_distribution": Counter(),
                "dtype_distribution": Counter(),
                "examples": [],
            },
        )
        slot["module_count"] += 1
        slot["param_count"] += m["param_count"] or 0
        slot["total_bytes"] += m["total_bytes"]
        bit_key = m["declared_bits"] if m["quantized"] else f"unquantized:{m['weight_dtype']}"
        slot["bit_distribution"][str(bit_key)] += 1
        slot["dtype_distribution"][m["weight_dtype"] or "?"] += 1
        if len(slot["examples"]) < 3:
            slot["examples"].append(m["base"])
    for role, slot in by_role.items():
        slot["effective_bpw"] = (
            (slot["total_bytes"] * 8 / slot["param_count"]) if slot["param_count"] else None
        )
    return by_role


def per_layer_breakdown(modules: list[dict]) -> dict[int, list[dict]]:
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for m in modules:
        if m["layer"] is not None:
            by_layer[m["layer"]].append(m)
    return dict(sorted(by_layer.items()))


# ---------- markdown rendering ----------


def render_markdown(report: dict) -> str:
    out: list[str] = []
    p = out.append

    p(f"# MLX Quantization Recipe — `{report['model_name']}`\n")

    # Global config
    qd = report["quant_defaults"]
    p("## Global defaults\n")
    p(f"- `bits`: **{qd['bits']}**")
    p(f"- `group_size`: **{qd['group_size']}**")
    p(f"- `mode`: **{qd['mode']}**")
    p(f"- architecture: `{report.get('architecture')}`")
    p(f"- model_type: `{report.get('model_type')}`\n")

    # Top-line stats
    s = report["totals"]
    p("## Totals\n")
    p(
        f"- modules with weights: **{s['module_count']}** "
        f"(quantized: {s['quantized_count']}, unquantized: {s['unquantized_count']})"
    )
    p(f"- total parameters: **{s['param_count']:,}**")
    p(f"- total on-disk bytes: **{s['total_bytes'] / 1e9:.2f} GB**")
    p(f"- model effective bpw: **{s['effective_bpw']:.3f}**\n")

    # Role breakdown
    p("## Bit allocation by structural role\n")
    p("| Role | Modules | Params | Effective bpw | Bit distribution | Weight dtype(s) |")
    p("| --- | ---: | ---: | ---: | --- | --- |")
    role_rows = sorted(report["by_role"].items(), key=lambda kv: -kv[1]["param_count"])
    for role, slot in role_rows:
        bits = ", ".join(f"{k}: {v}" for k, v in slot["bit_distribution"].most_common())
        dtypes = ", ".join(f"{k}: {v}" for k, v in slot["dtype_distribution"].most_common())
        bpw = f"{slot['effective_bpw']:.3f}" if slot["effective_bpw"] else "—"
        p(
            f"| `{role}` | {slot['module_count']} "
            f"| {slot['param_count']:,} | {bpw} "
            f"| {bits} | {dtypes} |"
        )
    p("")

    # Protected tensors
    p("## Tensors kept at full precision (not in quantization config)\n")
    protected = report["protected_tensors"]
    if not protected:
        p("_None — every weight tensor was quantized._\n")
    else:
        p(
            "These weights survive in their on-disk dtype"
            " because the recipe deliberately omitted them"
            " from the quantization plan.\n"
        )
        p("| Module | Role | dtype | Shape | Params |")
        p("| --- | --- | --- | --- | ---: |")
        for m in sorted(protected, key=lambda x: -(x["param_count"] or 0)):
            shape = "×".join(str(d) for d in (m["weight_shape"] or []))
            p(
                f"| `{m['base']}` | `{m['role']}` "
                f"| {m['weight_dtype']} | {shape} "
                f"| {m['param_count'] or 0:,} |"
            )
        p("")

    # Per-layer recipe
    by_layer = report["by_layer"]
    if by_layer:
        p("## Per-layer recipe\n")
        p(
            "Each row is one transformer block; columns show"
            " the bit width assigned to each sub-module."
            " `—` means the module isn't in this layer;"
            " `fp` means kept at full precision.\n"
        )

        # Discover the columns we need based on what shows up in any layer
        columns = set()
        for ms in by_layer.values():
            for m in ms:
                columns.add(m["role"])
        col_order = [
            r
            for r in [
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "attn.o_proj",
                "attn.qkv_proj",
                "attn.out_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "mlp.gate_up_proj",
                "experts.gate_up",
                "experts.down",
                "router",
                "shared_expert",
                "norm",
            ]
            if r in columns
        ] + sorted(
            c
            for c in columns
            if c
            not in {
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "attn.o_proj",
                "attn.qkv_proj",
                "attn.out_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "mlp.gate_up_proj",
                "experts.gate_up",
                "experts.down",
                "router",
                "shared_expert",
                "norm",
            }
        )

        p("| Layer | " + " | ".join(f"`{c}`" for c in col_order) + " |")
        p("| ---: | " + " | ".join("---" for _ in col_order) + " |")
        for layer_idx, ms in by_layer.items():
            row_cells = []
            by_role_in_layer: dict[str, list[dict]] = defaultdict(list)
            for m in ms:
                by_role_in_layer[m["role"]].append(m)
            for role in col_order:
                cell_modules = by_role_in_layer.get(role, [])
                if not cell_modules:
                    row_cells.append("—")
                else:
                    bits = []
                    for m in cell_modules:
                        if m["quantized"]:
                            bits.append(str(m["declared_bits"]))
                        else:
                            bits.append("fp")
                    row_cells.append("/".join(bits) if len(set(bits)) > 1 else bits[0])
            p(f"| {layer_idx} | " + " | ".join(row_cells) + " |")
        p("")

    # Outliers
    p("## Outlier modules (deviating from the global default)\n")
    outliers = report["outliers"]
    if not outliers:
        p("_No deviations — every module uses the global default._\n")
    else:
        p(
            f"{len(outliers)} modules differ from the"
            f" default `(bits={qd['bits']},"
            f" group_size={qd['group_size']})`.\n"
        )
        # Cluster by (declared_bits, declared_group_size)
        clusters: dict[tuple, list[str]] = defaultdict(list)
        for m in outliers:
            key = (
                m["declared_bits"] if m["quantized"] else f"unquantized:{m['weight_dtype']}",
                m["declared_group_size"] if m["quantized"] else None,
            )
            clusters[key].append(m["base"])
        p("| bits | group_size | count | example modules |")
        p("| --- | --- | ---: | --- |")
        for (b, gs), bases in sorted(clusters.items(), key=lambda x: -len(x[1])):
            example = ", ".join(f"`{b_}`" for b_ in bases[:3])
            if len(bases) > 3:
                example += f", … (+{len(bases) - 3} more)"
            p(f"| {b} | {gs} | {len(bases)} | {example} |")
        p("")

    p("## Reproduction sketch\n")
    p("Pseudo-code outline of what to feed `mlx_lm.quantize` (or your own quantizer):\n")
    p("```python")
    p("quantization_config = {")
    p(f"    'bits': {qd['bits']},")
    p(f"    'group_size': {qd['group_size']},")
    p(f"    'mode': '{qd['mode']}',")
    p("    # per-module overrides (the editorial recipe):")
    cluster_summary: dict[tuple, list[str]] = defaultdict(list)
    for m in report["outliers"]:
        if m["quantized"]:
            cluster_summary[(m["declared_bits"], m["declared_group_size"])].append(m["base"])
    for (b, gs), bases in sorted(cluster_summary.items(), key=lambda x: -len(x[1])):
        p(f"    # {len(bases)} modules at bits={b}, group_size={gs}, e.g. {bases[0]}")
    if report["protected_tensors"]:
        p("    # tensors below this line are SKIPPED (kept at full precision):")
        for m in report["protected_tensors"][:5]:
            p(f"    #   {m['base']}  ({m['weight_dtype']}, {m['role']})")
        if len(report["protected_tensors"]) > 5:
            p(f"    #   … ({len(report['protected_tensors']) - 5} more)")
    p("}")
    p("```\n")

    return "\n".join(out)


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model_dir", type=Path)
    ap.add_argument(
        "--md",
        type=Path,
        default=None,
        help="Write Markdown report to this file (default: stdout)",
    )
    ap.add_argument("--json", type=Path, default=None, help="Write full JSON dump to this file")
    args = ap.parse_args()

    md = args.model_dir
    if not md.is_dir():
        sys.exit(f"Not a directory: {md}")

    config = json.loads((md / "config.json").read_text())
    quant_cfg = config.get("quantization", {})
    if not isinstance(quant_cfg, dict):
        sys.exit("config.json has no 'quantization' object — is this an unquantized checkpoint?")

    default_bits = quant_cfg.get("bits")
    default_group_size = quant_cfg.get("group_size")
    default_mode = quant_cfg.get("mode")

    meta = collect_tensor_metadata(md)
    modules_raw = group_into_modules(meta)
    modules = [
        derive_module_recipe(base, parts, quant_cfg, default_bits, default_group_size)
        for base, parts in sorted(modules_raw.items())
        if "weight" in parts
    ]

    by_role = aggregate_by_role(modules)
    by_layer = per_layer_breakdown(modules)

    quantized = [m for m in modules if m["quantized"]]
    unquantized = [m for m in modules if not m["quantized"]]
    # "Protected" = weight tensor exists, no scales sibling, AND not declared
    # in the quantization config. (A non-declared module is what the recipe
    # author meant to exclude.)
    protected = [m for m in unquantized if not m["declared_in_config"]]

    # Outliers: declared modules whose (bits, group_size) deviate from defaults.
    outliers = [
        m
        for m in modules
        if m["declared_in_config"]
        and (m["declared_bits"] != default_bits or m["declared_group_size"] != default_group_size)
    ]
    # Plus protected (non-declared) modules — those are also editorial decisions.
    for m in protected:
        outliers.append(m)

    total_params = sum((m["param_count"] or 0) for m in modules)
    total_bytes = sum(m["total_bytes"] for m in modules)
    effective_bpw = (total_bytes * 8 / total_params) if total_params else 0

    report = {
        "model_name": md.name,
        "model_path": str(md),
        "architecture": (config.get("architectures") or [None])[0],
        "model_type": config.get("model_type"),
        "quant_defaults": {
            "bits": default_bits,
            "group_size": default_group_size,
            "mode": default_mode,
        },
        "totals": {
            "module_count": len(modules),
            "quantized_count": len(quantized),
            "unquantized_count": len(unquantized),
            "param_count": total_params,
            "total_bytes": total_bytes,
            "effective_bpw": effective_bpw,
        },
        "by_role": by_role,
        "by_layer": by_layer,
        "modules": modules,
        "protected_tensors": protected,
        "outliers": outliers,
    }

    md_text = render_markdown(report)
    if args.md:
        args.md.write_text(md_text)
    else:
        print(md_text)

    if args.json:
        # Counters → plain dicts; tuple keys → strings for JSON safety.
        def to_jsonable(o):
            if isinstance(o, Counter):
                return dict(o)
            if isinstance(o, dict):
                return {str(k): to_jsonable(v) for k, v in o.items()}
            if isinstance(o, list):
                return [to_jsonable(x) for x in o]
            return o

        args.json.write_text(json.dumps(to_jsonable(report), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
