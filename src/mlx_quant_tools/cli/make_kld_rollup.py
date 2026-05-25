#!/usr/bin/env python
"""Aggregate per-checkpoint kld-vs-<teacher>.json files into rollup tables.

Walks a glob of kld-vs-*.json files, groups by teacher path, sorts each
group by mean KLD ascending, and emits

    <out-dir>/<teacher-basename>-rollup.md
    <out-dir>/<teacher-basename>-rollup.json

per teacher group. The MD is human-readable; the JSON preserves each run's
full per-checkpoint schema verbatim.

This tool only reads schema_version=2 records and skips anything else with
a warning.

Usage:
  mqt-make-kld-rollup --pattern '/path/to/models/*/kld-vs-*.json'
  mqt-make-kld-rollup --pattern '/path/to/models/gemma-4-E2B*/kld-vs-*.json'
  mqt-make-kld-rollup --teacher google/gemma-4-E2B \\
      --md results/gemma-4-e2b-rollup.md \\
      --json results/gemma-4-e2b-rollup.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


# ---------- loading ----------


def load_runs(patterns: str | list[str]) -> list[dict]:
    """Walk pattern(s), parse JSON, drop records that don't match the locked schema.

    Accepts a single glob string or a list of globs.

    De-duplicates by (student.path, teacher.path, calibration spec): if both
    a live sidecar and a preserved one point at the same scoring run, keep the
    live one (first in the patterns list).
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    seen_paths: set[str] = set()
    paths: list[str] = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.expanduser(pat))):
            if p in seen_paths:
                continue
            seen_paths.add(p)
            paths.append(p)
    runs: list[dict] = []
    seen_keys: set[tuple] = set()
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warn(f"skipping {p}: {e}")
            continue
        if d.get("schema_version") != SCHEMA_VERSION:
            warn(
                f"skipping {p}: schema_version {d.get('schema_version')!r} (want {SCHEMA_VERSION})"
            )
            continue
        key = (
            d.get("student", {}).get("path"),
            d.get("teacher", {}).get("path"),
            _calibration_key(d.get("calibration", {})),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        d["_source_path"] = p
        runs.append(d)
    return runs


def group_by_teacher(runs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["teacher"]["path"], []).append(r)
    return groups


# ---------- rendering ----------


def _teacher_slug(path: str) -> str:
    return Path(path).name.lower()


def _calibration_key(c: dict) -> tuple:
    return (c.get("corpus"), c.get("num_samples"), c.get("max_seq_len"), c.get("seed"))


def _fmt(v: Any, kind: str = "auto") -> str:
    """Render a value for the MD table. Missing/None → em-dash."""
    if v is None:
        return "—"
    if kind == "bool":
        return "yes" if v else "no"
    if kind == "gb":
        return f"{v / 1e9:.2f} GB"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "kld":
        return f"{v:.4f}"
    if kind == "bpw":
        return f"{v:.3f}"
    return str(v)


def _short_tool(recipe: dict) -> str:
    t = recipe.get("tool")
    if t == "attn-protect-quantize":
        return "AP"
    if t in (None, "unknown"):
        return "?"
    return str(t)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def render_md(teacher_path: str, runs: list[dict]) -> str:
    """One Markdown rollup, sorted by mean KLD ascending.

    Columns are tight on purpose — the JSON sibling carries the full schema
    for any consumer that needs more.
    """
    out: list[str] = []
    teacher = runs[0]["teacher"]
    out.append(f"# KLD rollup: `{Path(teacher_path).name}`")
    out.append("")
    out.append(f"- teacher: `{teacher_path}`")
    out.append(f"- precision: `{teacher.get('precision')}`")
    out.append(f"- runs: **{len(runs)}**")
    out.append(f"- generated: {_now_iso()}")
    out.append("")

    cals = sorted({_calibration_key(r["calibration"]) for r in runs})
    if len(cals) == 1:
        corpus, n, sl, seed = cals[0]
        out.append(
            f"All runs share calibration: corpus=`{corpus}`, "
            f"num_samples={n}, max_seq_len={sl}, seed={seed}."
        )
    else:
        out.append(
            f"**Mixed calibration ({len(cals)} distinct specs) — KLD numbers across "
            f"calibration boundaries are not directly comparable.** "
            f"The `cal` column groups rows by calibration; see the legend below."
        )
    out.append("")

    cal_index = {k: i + 1 for i, k in enumerate(cals)}
    runs_sorted = sorted(runs, key=lambda r: r["kld"].get("mean", float("inf")))

    headers = [
        "student",
        "tool",
        "bits",
        "gs",
        "mode",
        "dwq",
        "boosts",
        "tied8",
        "vlm",
        "bpw",
        "size",
        "cal",
        "mean KLD",
        "p50",
        "p95",
        "p99",
        "p99.9",
        "top-1",
        "top-5",
    ]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in runs_sorted:
        s = r["student"]
        rcp = r["recipe"]
        kld = r["kld"]
        ag = r["agreement"]
        cal_cell = f"#{cal_index[_calibration_key(r['calibration'])]}" if len(cals) > 1 else "—"
        # Render the boost cell as the count of boosted tensors (the most
        # decision-relevant single-cell summary). The full per-tensor map
        # lives in the JSON sibling and the on-disk recipe.json.
        mlp_boosts = rcp.get("mlp_boosts") or {}
        boost_count = mlp_boosts.get("applied_count")
        if boost_count is None and rcp.get("with_mlp_boosts"):
            # Fallback: mlp_boosts present but applied_count missing on an
            # older schema — count the per-tensor map directly.
            boost_count = len(mlp_boosts.get("boosts", {}) or {})
        boosts_cell = _fmt(boost_count) if rcp.get("with_mlp_boosts") else "no"
        row = [
            f"`{Path(s['path']).name}`",
            _short_tool(rcp),
            _fmt(rcp.get("bits")),
            _fmt(rcp.get("group_size")),
            _fmt(rcp.get("attn_protect_mode")),
            _fmt(rcp.get("with_dwq"), "bool"),
            boosts_cell,
            _fmt(rcp.get("floor_tied_embed"), "bool"),
            _fmt(rcp.get("protect_vlm"), "bool"),
            _fmt(s.get("effective_bpw"), "bpw"),
            _fmt(s.get("size_bytes"), "gb"),
            cal_cell,
            _fmt(kld.get("mean"), "kld"),
            _fmt(kld.get("p50"), "kld"),
            _fmt(kld.get("p95"), "kld"),
            _fmt(kld.get("p99"), "kld"),
            _fmt(kld.get("p999"), "kld"),
            _fmt(ag.get("top1"), "pct"),
            _fmt(ag.get("top5"), "pct"),
        ]
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    if len(cals) > 1:
        out.append("## Calibration legend")
        out.append("")
        out.append("| ID | corpus | samples | seq_len | seed |")
        out.append("|---|---|---:|---:|---:|")
        for key, idx in cal_index.items():
            corpus, n, sl, seed = key
            out.append(f"| #{idx} | `{corpus}` | {n} | {sl} | {seed} |")
        out.append("")

    out.append("## Sources")
    out.append("")
    for r in runs_sorted:
        out.append(f"- `{r['_source_path']}`")
    out.append("")
    return "\n".join(out)


def render_json(teacher_path: str, runs: list[dict]) -> dict:
    """Machine-readable rollup: full per-run records sorted by mean KLD ascending."""
    runs_sorted = sorted(runs, key=lambda r: r["kld"].get("mean", float("inf")))
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher": runs[0]["teacher"],
        "generated": _now_iso(),
        "num_runs": len(runs),
        "runs": [{k: v for k, v in r.items() if k != "_source_path"} for r in runs_sorted],
        "sources": [r["_source_path"] for r in runs_sorted],
    }


# ---------- argument parsing ----------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mqt-make-kld-rollup",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob for kld-vs-*.json files (repeatable, required). "
        "Example: '/path/to/models/*/kld-vs-*.json'",
    )
    p.add_argument(
        "--teacher",
        default=None,
        help="Substring filter on teacher.path; emit only the matching group(s). "
        "Required when using --md/--json overrides.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write <teacher>-rollup.{md,json} (default: results/)",
    )
    p.add_argument(
        "--md",
        type=Path,
        default=None,
        help=(
            "Explicit MD path (single-teacher mode; requires --teacher to match exactly one group)"
        ),
    )
    p.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help=(
            "Explicit JSON path (single-teacher mode;"
            " requires --teacher to match exactly one group)"
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.pattern:
        patterns = args.pattern
    else:
        sys.exit(
            "no --pattern given. Pass one or more glob patterns, e.g.:\n"
            "  mqt-make-kld-rollup --pattern '~/models/*/kld-vs-*.json'"
        )

    runs = load_runs(patterns)
    if not runs:
        sys.exit(f"no kld-vs-*.json files matched {patterns!r}")

    info(f"loaded {len(runs)} runs")
    groups = group_by_teacher(runs)

    if args.teacher:
        matched = {k: v for k, v in groups.items() if args.teacher in k}
        if not matched:
            sys.exit(
                f"no teacher path contains {args.teacher!r}; available: {sorted(groups.keys())}"
            )
        groups = matched

    if (args.md or args.json_path) and len(groups) != 1:
        sys.exit("--md/--json require --teacher to select exactly one group")

    for teacher_path, group_runs in groups.items():
        slug = _teacher_slug(teacher_path)
        md_path = args.md or args.out_dir / f"{slug}-rollup.md"
        json_path = args.json_path or args.out_dir / f"{slug}-rollup.json"

        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        md_path.write_text(render_md(teacher_path, group_runs))
        json_path.write_text(json.dumps(render_json(teacher_path, group_runs), indent=2) + "\n")
        info(f"{teacher_path} → {md_path} + {json_path} ({len(group_runs)} runs)")


if __name__ == "__main__":
    main()
