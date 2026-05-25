"""Load a GGUF K-quant file and generate text using MLX's kquant kernels.

End-to-end driver for the K-quant kernel stack: GGUF → in-memory model with
per-tensor `nn.QuantizedLinear(mode="kquant")` and a `KQuantEmbedding` that
gathers wire bytes then dequantizes (avoiding the silent-corruption bug in
`mx.dequantize` on outputs > INT_MAX elements).

Config and tokenizer are synthesized from GGUF metadata via
`gguf_config_synth` and `gguf_tokenizer`. For native MLX checkpoints
use `--mlx-source` instead.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from gguf import GGUFReader
from mlx_lm.sample_utils import make_sampler

from mlx_quant_tools.gguf_runtime import (
    KQuantEmbedding,
    load_gguf_via_mx,
    load_kquant_model,
    print_inventory,
    remap_arrays,
)

# Benign filler used to synthesize fixed-length prompts. Three pangrams + one
# generic English sentence — tokenizes to ~40 tokens on most BPE/SentencePiece
# vocabs, so we can build any target length by repeat+truncate.
_BENCH_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
    "Performance benchmarking measures throughput across diverse workloads. "
)


def _parse_int_list(s: str, *, flag: str) -> list[int]:
    """Parse a comma-separated list of positive ints (e.g. '16,32,64').
    Single values are accepted (returns a one-element list)."""
    try:
        out = [int(x) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"{flag} expects comma-separated integers, got {s!r}"
        ) from e
    if not out or any(n <= 0 for n in out):
        raise argparse.ArgumentTypeError(f"{flag} values must be positive integers, got {s!r}")
    return out


def synthesize_prompt_ids(tokenizer, n_tokens: int) -> list[int]:
    """Return exactly `n_tokens` token IDs by repeating filler text.

    Prepends BOS if the tokenizer has one (most models expect this for the
    first token of any prompt; absence shifts attention patterns).
    """
    base = tokenizer.encode(_BENCH_FILLER, add_special_tokens=False)
    if not base:
        raise RuntimeError("tokenizer.encode returned empty for the filler text")
    out: list[int] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        out.append(bos)
    while len(out) < n_tokens:
        out.extend(base)
    return out[:n_tokens]


def _run_one(model, tokenizer, prompt_ids, *, max_tokens, sampler):
    """One generation. Returns (final GenerationResponse, wall_seconds)."""
    final = None
    t0 = time.perf_counter()
    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        final = resp
    return final, time.perf_counter() - t0


def _summarize_runs(runs, *, key: str):
    """Reduce a list of (response, wall) samples to a steady-state summary.

    Drops the first sample (cold-cache / pipeline-compile cost on first
    use of a shape) and returns the median of the rest. Reports min/max
    of the same effective set so thermal/throughput drift is visible —
    e.g. on Apple Silicon, sustained GPU load downclocks the chip and
    decode tps decays monotonically; reporting `max` alone overstates
    the steady-state number users will see.

    `key` is "prompt_tps" or "generation_tps".
    Returns (repr_response, median_wall_s, stats_dict) or None if empty.
    """
    if not runs:
        return None
    effective = runs[1:] if len(runs) >= 2 else runs
    sorted_runs = sorted(effective, key=lambda x: getattr(x[0], key))
    mid = len(sorted_runs) // 2
    repr_resp = sorted_runs[mid][0]
    tps = [getattr(r[0], key) for r in sorted_runs]
    walls = sorted(w for _, w in effective)
    return (
        repr_resp,
        walls[len(walls) // 2],
        {
            "median": tps[mid],
            "min": tps[0],
            "max": tps[-1],
            "n_used": len(effective),
            "n_dropped": len(runs) - len(effective),
        },
    )


def _shape_label(*, n_runs: int, warmup: bool) -> str:
    """Per-shape progress label that matches the actual warmup state."""
    return ("compile-warmup + " if warmup else "no warmup, ") + f"{n_runs} timed"


def run_benchmark(
    model, tokenizer, lengths, *, max_tokens, sampler, runs: int = 3, warmup: bool = True
):
    """Sweep prompt lengths. Per length: optional compile-warmup pass +
    `runs` timed runs. Returns list of (n_prompt_tokens, repr_response,
    median_wall_s, stats_dict). Reporting uses median of trailing
    samples (drops cold first run) — see `_summarize_runs`.
    """
    if warmup:
        # Tiny prompt, tiny decode — exercises the full inference graph once
        # so subsequent shapes get a populated kernel cache.
        warm_ids = synthesize_prompt_ids(tokenizer, 64)
        print("[bench] global warmup pass ...", flush=True)
        _run_one(model, tokenizer, warm_ids, max_tokens=4, sampler=sampler)

    results = []
    for n in lengths:
        prompt_ids = synthesize_prompt_ids(tokenizer, n)
        actual_n = len(prompt_ids)
        # First call at a new shape triggers Metal pipeline compile.
        # Per-shape compile-warmup is gated on the same `warmup` flag as
        # the global pass so `--bench-no-warmup` is unambiguous.
        print(
            f"[bench] prompt_tokens={actual_n} max_tokens={max_tokens} "
            f"({_shape_label(n_runs=runs, warmup=warmup)}) ...",
            flush=True,
        )
        if warmup:
            _run_one(model, tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler)
        timed = []
        for i in range(runs):
            resp, wall = _run_one(
                model, tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler
            )
            if resp is None:
                continue
            timed.append((resp, wall))
            print(
                f"  run {i + 1}: prompt_tps={resp.prompt_tps:.1f}  "
                f"generation_tps={resp.generation_tps:.1f}  "
                f"wall={wall:.2f}s"
            )
        summary = _summarize_runs(timed, key="prompt_tps")
        if summary is None:
            continue
        repr_resp, median_wall, stats = summary
        results.append((actual_n, repr_resp, median_wall, stats))
    return results


def _format_spread(stats: dict) -> str:
    """e.g. '[18.7-19.3 / 4]' — min-max range and effective sample count."""
    return f"[{stats['min']:.1f}-{stats['max']:.1f} / {stats['n_used']}]"


def print_bench_table(results) -> None:
    print()
    print("=== prompt processing benchmark (median of trailing samples) ===")
    print(
        f"  {'n_prompt':>9}  {'prefill tps':>12}  {'decode tps':>12}  "
        f"{'peak GB':>8}  {'spread / N':>16}  {'median wall':>11}"
    )
    for n, r, median_wall, stats in results:
        print(
            f"  {n:>9}  {r.prompt_tps:>12.1f}  {r.generation_tps:>12.1f}  "
            f"{r.peak_memory:>8.2f}  {_format_spread(stats):>16}  "
            f"{median_wall:>10.2f}s"
        )


def run_benchmark_split(
    model, tokenizer, lengths, *, decode_tokens_list, sampler, runs: int = 3, warmup: bool = True
):
    """Apples-to-apples bench mirroring llama-bench's pp/tg split.

    Two independent measurement phases:
      - Prefill (pp_N): per length N, run stream_generate(prompt=N tokens,
        max_tokens=1) and read prompt_tps. The single decode token is
        ignored — the model still has to emit one logit but the timing we
        keep is the prefill phase only.
      - Decode (tg_M): per M in decode_tokens_list, run stream_generate(
        prompt=[BOS], max_tokens=M) and read generation_tps. Decode tps is
        independent of prompt length, so each M is measured once with empty
        cache — matching llama-bench's tg_M which also starts from empty.

    Returns (prefill_results, decode_results):
      prefill_results: list of (n_prompt, repr_response, median_wall_s, stats)
      decode_results:  list of (n_gen,    repr_response, median_wall_s, stats)
    """
    if warmup:
        warm_ids = synthesize_prompt_ids(tokenizer, 64)
        print("[bench-split] global warmup pass ...", flush=True)
        _run_one(model, tokenizer, warm_ids, max_tokens=4, sampler=sampler)

    # --- decode-only timing (sweep over decode_tokens_list) ---
    # Seed with a non-terminal token so the model generates freely.
    # Qwen3.x HF tokenizers have bos_token_id=None; falling back to
    # eos_token_id would seed with <|im_end|> and stop after ~9 tokens.
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = tokenizer.encode("A", add_special_tokens=False)[0]
    decode_seed = [bos_id]
    decode_results = []
    for decode_tokens in decode_tokens_list:
        print(
            f"[bench-split] decode-only: {decode_tokens} tokens "
            f"({_shape_label(n_runs=runs, warmup=warmup)}) ...",
            flush=True,
        )
        if warmup:
            _run_one(model, tokenizer, decode_seed, max_tokens=decode_tokens, sampler=sampler)
        decode_runs = []
        for i in range(runs):
            resp, wall = _run_one(
                model, tokenizer, decode_seed, max_tokens=decode_tokens, sampler=sampler
            )
            if resp is None:
                continue
            decode_runs.append((resp, wall))
            print(f"  decode run {i + 1}: gen_tps={resp.generation_tps:.1f}  wall={wall:.2f}s")
        summary = _summarize_runs(decode_runs, key="generation_tps")
        if summary is not None:
            repr_resp, median_wall, stats = summary
            decode_results.append((decode_tokens, repr_resp, median_wall, stats))

    # --- prefill-only timing (one shape per length) ---
    prefill_results = []
    for n in lengths:
        prompt_ids = synthesize_prompt_ids(tokenizer, n)
        actual_n = len(prompt_ids)
        print(
            f"[bench-split] prefill: prompt_tokens={actual_n} "
            f"({_shape_label(n_runs=runs, warmup=warmup)}) ...",
            flush=True,
        )
        if warmup:
            _run_one(model, tokenizer, prompt_ids, max_tokens=1, sampler=sampler)
        timed = []
        for i in range(runs):
            resp, wall = _run_one(model, tokenizer, prompt_ids, max_tokens=1, sampler=sampler)
            if resp is None:
                continue
            timed.append((resp, wall))
            print(f"  prefill run {i + 1}: prompt_tps={resp.prompt_tps:.1f}  wall={wall:.2f}s")
        summary = _summarize_runs(timed, key="prompt_tps")
        if summary is not None:
            repr_resp, median_wall, stats = summary
            prefill_results.append((actual_n, repr_resp, median_wall, stats))

    return prefill_results, decode_results


def print_bench_table_split(prefill_results, decode_results) -> None:
    print()
    print("=== bench-split: prefill (median of trailing samples) ===")
    print(
        f"  {'n_prompt':>9}  {'prefill tps':>12}  {'peak GB':>8}  "
        f"{'spread / N':>16}  {'median wall':>11}"
    )
    for n, r, median_wall, stats in prefill_results:
        print(
            f"  {n:>9}  {r.prompt_tps:>12.1f}  {r.peak_memory:>8.2f}  "
            f"{_format_spread(stats):>16}  {median_wall:>10.2f}s"
        )
    print()
    print("=== bench-split: decode (median of trailing samples) ===")
    if not decode_results:
        print("  (no decode runs completed)")
        return
    print(
        f"  {'n_gen':>9}  {'decode tps':>12}  {'peak GB':>8}  "
        f"{'spread / N':>16}  {'median wall':>11}"
    )
    for n_gen, r, median_wall, stats in decode_results:
        print(
            f"  {n_gen:>9}  {r.generation_tps:>12.1f}  "
            f"{r.peak_memory:>8.2f}  {_format_spread(stats):>16}  "
            f"{median_wall:>10.2f}s"
        )


# ---------------------------------------------------------------------------
# --profile-layers: per-layer-type wallclock split for prefill
# ---------------------------------------------------------------------------

# Bucket order is canonical for printing.
_PROFILE_BUCKETS = ("attn", "ssm", "mlp", "other")


def _classify_layer_attr(attr: str) -> str:
    """Map a transformer-layer child-attribute name to a profiling bucket.

    Order matters:
      1. `norm` is checked first so that `post_attention_layernorm` and
         `pre_feedforward_layernorm` don't get pulled into attn / mlp by
         the substring rules below — those layers do norm work, not
         attention/MLP work, and conflating them inflates the bucket.
      2. SSM keywords are checked before generic `attn` because some
         hybrid models name their linear-attention block `linear_attn`
         (which would otherwise match the generic `attn` rule).
    """
    n = attr.lower()
    if "norm" in n:
        return "other"
    if any(k in n for k in ("linear_attn", "gated_delta", "mamba", "ssm")):
        return "ssm"
    if "attn" in n or "attention" in n:
        return "attn"
    if any(k in n for k in ("mlp", "moe", "expert", "feed_forward", "block_sparse")):
        return "mlp"
    return "other"


class _TimedProxy(nn.Module):
    """Drop-in replacement for a transformer-layer sub-module that records
    wallclock per call. Forwards `__call__` to the wrapped child after
    forcing `mx.eval()` on its output, so the timer captures the full GPU
    cost of that sub-module rather than just kernel-dispatch overhead.

    Implementation note: monkey-patching `instance.__call__` is shadowed by
    Python's class-level `__call__` lookup, so we wrap by attribute swap on
    the parent layer (`setattr(layer, attr, proxy)`).
    """

    def __init__(
        self, inner: nn.Module, bucket: str, totals: dict[str, float], counts: dict[str, int]
    ) -> None:
        super().__init__()
        self._inner = inner
        self._bucket = bucket
        self._totals = totals
        self._counts = counts

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._inner(*args, **kwargs)
        # Force a barrier so timing isn't async-distorted. Async kernel
        # launch in mlx means without this the timed scope would only cover
        # graph-construction overhead.
        if isinstance(out, mx.array):
            mx.eval(out)
        elif isinstance(out, (tuple, list)):
            arrs = [o for o in out if isinstance(o, mx.array)]
            if arrs:
                mx.eval(*arrs)
        dt = time.perf_counter() - t0
        self._totals[self._bucket] += dt
        self._counts[self._bucket] += 1
        return out


def _find_transformer_layers(model: nn.Module):
    """Return the list of transformer layer modules and a label for the
    container path used (just for logging). Handles the common mlx_lm
    wrappers — `model.model.layers`, `model.layers`, and the VLM
    `model.language_model.model.layers` variant — without per-arch
    hardcoding."""
    candidates = [
        ("model.model", getattr(getattr(model, "model", None), "layers", None)),
        ("model", getattr(model, "layers", None)),
        (
            "model.language_model.model",
            getattr(
                getattr(getattr(model, "language_model", None), "model", None),
                "layers",
                None,
            ),
        ),
    ]
    for label, layers in candidates:
        if layers is not None:
            return layers, label
    return None, None


def _install_layer_profiler(
    model: nn.Module,
) -> tuple[dict[str, float], dict[str, int], callable, int]:
    """Wrap each direct child `nn.Module` of every transformer layer with a
    `_TimedProxy`. Returns `(totals_seconds, counts, uninstall, n_wrapped)`.
    `uninstall()` restores the original attributes."""
    totals: dict[str, float] = {b: 0.0 for b in _PROFILE_BUCKETS}
    counts: dict[str, int] = {b: 0 for b in _PROFILE_BUCKETS}
    originals: list[tuple[nn.Module, str, nn.Module]] = []

    layers, _label = _find_transformer_layers(model)
    if layers is None:
        return totals, counts, (lambda: None), 0

    for layer in layers:
        # mlx's `nn.Module` subclasses `dict`; submodules (and arrays/lists)
        # are stored as dict entries via `__setattr__`. Iterate the dict
        # keys to find children — `dir()` would also surface class-level
        # descriptors (e.g. the read-only `state` property) and writing
        # back would crash on the no-deleter chain.
        for attr in list(layer.keys()):
            if attr.startswith("_"):
                continue
            child = layer[attr]
            if not isinstance(child, nn.Module):
                continue
            if isinstance(child, _TimedProxy):
                continue
            bucket = _classify_layer_attr(attr)
            proxy = _TimedProxy(child, bucket, totals, counts)
            setattr(layer, attr, proxy)
            originals.append((layer, attr, child))

    def uninstall() -> None:
        for parent, attr, orig in originals:
            setattr(parent, attr, orig)

    return totals, counts, uninstall, len(originals)


def _make_timed_fn(orig_fn, bucket, totals, counts):
    """Wrap a plain function with eval-fenced timing.

    Evals all array inputs before timing starts (so upstream lazy ops don't
    pollute the measurement), then evals all array outputs after the call.
    """

    def timed(*args, **kwargs):
        in_arrs = [a for a in args if isinstance(a, mx.array)]
        in_arrs.extend(v for v in kwargs.values() if isinstance(v, mx.array))
        if in_arrs:
            mx.eval(*in_arrs)
        t0 = time.perf_counter()
        result = orig_fn(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            out_arrs = [o for o in result if isinstance(o, mx.array)]
            if out_arrs:
                mx.eval(*out_arrs)
        elif isinstance(result, mx.array):
            mx.eval(result)
        dt = time.perf_counter() - t0
        totals[bucket] += dt
        counts[bucket] += 1
        return result

    return timed


def _install_detail_profiler(model: nn.Module):
    """Wrap grandchildren of each transformer layer for per-projection timing.

    Instead of timing `layer.mlp` as one unit, times `mlp.gate_proj`,
    `mlp.up_proj`, `mlp.down_proj` individually.  Norms stay at depth 1.

    Also detects non-module function calls in SSM paths (e.g.
    ``gated_delta_update``) and wraps them so their cost isn't misattributed
    to the next proxied sub-module.

    Returns ``(totals, counts, uninstall, n_wrapped, bucket_order)``.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    originals: list[tuple[nn.Module, str, nn.Module]] = []
    fn_patches: list[tuple[object, str, object]] = []
    bucket_order: list[str] = []

    layers, _label = _find_transformer_layers(model)
    if layers is None:
        return totals, counts, (lambda: None), 0, bucket_order

    seen_buckets: set[str] = set()
    patched_modules: set[int] = set()

    _SSM_FN_HOOKS = {
        "gated_delta_update": "ssm.recurrence",
    }

    for layer in layers:
        for attr in list(layer.keys()):
            if attr.startswith("_"):
                continue
            child = layer[attr]
            if not isinstance(child, nn.Module):
                continue
            parent_bucket = _classify_layer_attr(attr)
            child_keys = [
                k
                for k in child.keys()
                if not k.startswith("_") and isinstance(child[k], nn.Module)
            ]

            if parent_bucket in ("attn", "ssm", "mlp") and child_keys:
                for sub_attr in child_keys:
                    grandchild = child[sub_attr]
                    if not isinstance(grandchild, nn.Module):
                        continue
                    if isinstance(grandchild, _TimedProxy):
                        continue
                    bucket = f"{parent_bucket}.{sub_attr}"
                    totals.setdefault(bucket, 0.0)
                    counts.setdefault(bucket, 0)
                    if bucket not in seen_buckets:
                        bucket_order.append(bucket)
                        seen_buckets.add(bucket)
                    proxy = _TimedProxy(grandchild, bucket, totals, counts)
                    setattr(child, sub_attr, proxy)
                    originals.append((child, sub_attr, grandchild))

                if parent_bucket == "ssm":
                    child_mod = sys.modules.get(type(child).__module__)
                    if child_mod and id(child_mod) not in patched_modules:
                        for fn_name, fn_bucket in _SSM_FN_HOOKS.items():
                            orig_fn = getattr(child_mod, fn_name, None)
                            if orig_fn is None:
                                continue
                            totals.setdefault(fn_bucket, 0.0)
                            counts.setdefault(fn_bucket, 0)
                            if fn_bucket not in seen_buckets:
                                try:
                                    norm_idx = bucket_order.index("ssm.norm")
                                    bucket_order.insert(norm_idx, fn_bucket)
                                except ValueError:
                                    bucket_order.append(fn_bucket)
                                seen_buckets.add(fn_bucket)
                            wrapped = _make_timed_fn(orig_fn, fn_bucket, totals, counts)
                            setattr(child_mod, fn_name, wrapped)
                            fn_patches.append((child_mod, fn_name, orig_fn))
                        patched_modules.add(id(child_mod))
            else:
                bucket = parent_bucket
                totals.setdefault(bucket, 0.0)
                counts.setdefault(bucket, 0)
                if bucket not in seen_buckets:
                    bucket_order.append(bucket)
                    seen_buckets.add(bucket)
                if isinstance(child, _TimedProxy):
                    continue
                proxy = _TimedProxy(child, bucket, totals, counts)
                setattr(layer, attr, proxy)
                originals.append((layer, attr, child))

    def uninstall():
        for parent, attr, orig in originals:
            setattr(parent, attr, orig)
        for mod, attr, orig in fn_patches:
            setattr(mod, attr, orig)

    return totals, counts, uninstall, len(originals), bucket_order


def profile_prefill_detail(
    model: nn.Module, tokenizer, lengths: list[int], *, runs: int = 2
) -> dict[int, dict]:
    """Fine-grained prefill profiling with per-projection timing."""
    from mlx_lm.models import cache as cache_mod

    summary: dict[int, dict] = {}
    for n in lengths:
        prompt_ids = synthesize_prompt_ids(tokenizer, n)
        actual_n = len(prompt_ids)
        all_runs: list[dict] = []
        bucket_order = []
        for run_i in range(runs):
            totals, counts, uninstall, n_wrapped, bucket_order = _install_detail_profiler(model)
            try:
                cache = cache_mod.make_prompt_cache(model, max_kv_size=None)
                token_ids = mx.array([prompt_ids], dtype=mx.int32)
                t0 = time.perf_counter()
                out = model(token_ids, cache=cache)
                if hasattr(out, "logits"):
                    out = out.logits
                mx.eval(out)
                wall = time.perf_counter() - t0
            finally:
                uninstall()
            row = dict(totals)
            row["_wall"] = wall
            row["_counts"] = dict(counts)
            row["_wrapped"] = n_wrapped
            row["_bucket_order"] = bucket_order
            all_runs.append(row)
        if len(all_runs) >= 3:
            all_runs.sort(key=lambda r: sum(v for k, v in r.items() if not k.startswith("_")))
            pick = all_runs[len(all_runs) // 2]
        else:
            pick = min(
                all_runs, key=lambda r: sum(v for k, v in r.items() if not k.startswith("_"))
            )
        summary[actual_n] = pick
    return summary


def print_detail_profile_table(summary: dict[int, dict]) -> None:
    print()
    print("=== prefill detail wallclock (per-projection breakdown) ===")
    for n, row in summary.items():
        bucket_order = row.get("_bucket_order", sorted(k for k in row if not k.startswith("_")))
        wall_ms = row["_wall"] * 1000
        instrumented_ms = sum(row.get(b, 0) for b in bucket_order) * 1000

        groups: dict[str, list[tuple[str, float]]] = {}
        for b in bucket_order:
            ms = row.get(b, 0) * 1000
            if "." in b:
                parent = b.split(".")[0]
            else:
                parent = b
            groups.setdefault(parent, []).append((b, ms))

        print(
            f"  n_prompt={n}  wall={wall_ms:.1f} ms  "
            f"instrumented={instrumented_ms:.1f} ms  "
            f"gap={wall_ms - instrumented_ms:.1f} ms"
        )
        print()
        for parent, entries in groups.items():
            parent_total = sum(ms for _, ms in entries)
            pct = (100 * parent_total / instrumented_ms) if instrumented_ms > 0 else 0
            print(f"  {parent:>8} total: {parent_total:>8.1f} ms ({pct:>5.1f}%)")
            cnt = row.get("_counts", {})
            for bucket, ms in entries:
                label = bucket.split(".")[-1] if "." in bucket else bucket
                n_calls = cnt.get(bucket, 0)
                per_call = ms / n_calls if n_calls else 0
                print(
                    f"    {label:<20} {ms:>8.1f} ms  "
                    f"({n_calls:>4} calls, {per_call:>6.2f} ms/call)"
                )
            print()


def profile_prefill(
    model: nn.Module, tokenizer, lengths: list[int], *, runs: int = 2
) -> dict[int, dict[str, float]]:
    """Run prefill-only at each prompt length under the layer profiler.

    Returns `{n_prompt: {bucket: seconds, ..., "total": seconds,
    "wall": seconds}}`. For each length we run a single forward pass per
    `runs` and report the best (minimum total instrumented time) sample.
    The forced `mx.eval()` inside `_TimedProxy` makes per-bucket sums
    additive within a single run; the host-side wallclock from
    `time.perf_counter()` is a sanity-check upper bound that includes the
    final eval barrier.
    """
    from mlx_lm.models import cache as cache_mod

    summary: dict[int, dict[str, float]] = {}
    for n in lengths:
        prompt_ids = synthesize_prompt_ids(tokenizer, n)
        actual_n = len(prompt_ids)
        best: dict[str, float] | None = None
        for run_i in range(runs):
            totals, counts, uninstall, n_wrapped = _install_layer_profiler(model)
            try:
                cache = cache_mod.make_prompt_cache(model, max_kv_size=None)
                token_ids = mx.array([prompt_ids], dtype=mx.int32)
                t0 = time.perf_counter()
                out = model(token_ids, cache=cache)
                if hasattr(out, "logits"):
                    out = out.logits
                mx.eval(out)
                wall = time.perf_counter() - t0
            finally:
                uninstall()
            row = {b: totals[b] for b in _PROFILE_BUCKETS}
            row["total"] = sum(row.values())
            row["wall"] = wall
            row["_counts"] = dict(counts)
            row["_wrapped"] = n_wrapped
            if best is None or row["total"] < best["total"]:
                best = row
        summary[actual_n] = best  # type: ignore[assignment]
    return summary


def print_profile_table(summary: dict[int, dict[str, float]]) -> None:
    print()
    print("=== prefill layer-bucket wallclock (best of N runs per row) ===")
    header = (
        f"  {'n_prompt':>9}  "
        f"{'attn ms':>10}  {'ssm ms':>10}  {'mlp ms':>10}  {'other ms':>10}  "
        f"{'sum ms':>10}  {'wall ms':>10}"
    )
    print(header)
    for n, row in summary.items():
        total_ms = row["total"] * 1000
        parts = []
        for b in _PROFILE_BUCKETS:
            ms = row[b] * 1000
            pct = (100 * ms / total_ms) if total_ms > 0 else 0.0
            parts.append(f"{ms:>6.1f} ({pct:>4.1f}%)")
        print(
            f"  {n:>9}  {parts[0]:>10}  {parts[1]:>10}  {parts[2]:>10}  "
            f"{parts[3]:>10}  {total_ms:>10.1f}  {row['wall'] * 1000:>10.1f}"
        )
    if summary:
        any_row = next(iter(summary.values()))
        print(
            f"  (instrumented {any_row.get('_wrapped', 0)} layer sub-modules; "
            f"counts/run={any_row.get('_counts', {})})"
        )


def _summarize_tps_samples(samples: list[float]) -> dict | None:
    """Same trailing-K-median rule as `_summarize_runs`, but for raw tps
    sample lists (used by the llama.cpp side which doesn't have a
    response object — only sample tps from the JSON)."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None
    effective = samples[1:] if len(samples) >= 2 else samples
    sorted_samples = sorted(effective)
    return {
        "median": sorted_samples[len(sorted_samples) // 2],
        "min": sorted_samples[0],
        "max": sorted_samples[-1],
        "n_used": len(effective),
        "n_dropped": len(samples) - len(effective),
    }


def run_llama_bench(
    gguf_path, lengths, decode_tokens_list, runs, *, binary: str, no_warmup: bool = False
):
    """Invoke llama.cpp's llama-bench with parameters matching our --bench
    split-mode run. Returns (prefill_stats, decode_stats) where each maps
    n_prompt or n_gen → stats dict (median/min/max/n_used).

    `lengths` and `decode_tokens_list` are each passed as comma-separated
    `-p` / `-n` to llama-bench, which treats them independently (no cross-
    product) — one record per (n_prompt=Ni, n_gen=0) and one per
    (n_prompt=0, n_gen=Mj).

    Apples-to-apples settings:
      -fa 0   no flash-attn (mlx_lm models don't use flash-attn here)
      -ngl 99 full Metal offload (default; matches MLX's all-GPU path)
      -o json parseable output

    Per-rep tps samples come from llama-bench's `samples_ts` JSON field.
    We print each sample (so output is comparable to the MLX side's
    per-run lines), then summarize via the same trailing-K-median rule
    that the MLX side uses — so the side-by-side ratio reflects
    steady-state on both sides, not cold-cache best-of-N.

    Errors:
      Raises FileNotFoundError if the binary is missing (caught upstream).
      Raises subprocess.CalledProcessError if llama-bench exits non-zero.
    """
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            f"llama-bench binary not found at {binary}. "
            f"Build llama.cpp first or pass --llama-bench-bin <path>."
        )
    cmd = [
        binary,
        "-m",
        str(gguf_path),
        "-p",
        ",".join(str(n) for n in lengths),
        "-n",
        ",".join(str(n) for n in decode_tokens_list),
        "-r",
        str(runs),
        "-o",
        "json",
        "-fa",
        "0",
    ]
    if no_warmup:
        cmd.append("--no-warmup")
    print(f"\n[vs-llama-bench] invoking: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    records = json.loads(proc.stdout)
    prefill: dict[int, dict] = {}
    decode: dict[int, dict] = {}
    for rec in records:
        np_, ng = rec.get("n_prompt", 0), rec.get("n_gen", 0)
        samples = rec.get("samples_ts")
        if not samples:
            avg = rec.get("avg_ts")
            samples = [avg] if avg is not None else []
        # Per-run echo so output is comparable with MLX side.
        if np_ > 0 and ng == 0:
            label, tps_unit = f"prefill (n_prompt={np_})", "prompt_tps"
        elif ng > 0 and np_ == 0:
            label, tps_unit = f"decode (n_gen={ng})", "gen_tps"
        else:
            continue
        print(f"[vs-llama-bench] {label}:", flush=True)
        for i, s in enumerate(samples, 1):
            if s is None:
                continue
            print(f"  llama run {i}: {tps_unit}={s:.1f}")
        stats = _summarize_tps_samples(samples)
        if stats is None:
            continue
        if np_ > 0 and ng == 0:
            prefill[np_] = stats
        elif ng > 0 and np_ == 0:
            decode[ng] = stats
    return prefill, decode


def _split_tuples_to_ref_dict(results) -> dict[int, dict]:
    """Convert run_benchmark_split output to the ref-dict shape used by
    print_comparison. Drops repr_resp + median_wall — only the stats summary
    matters for the comparison ratio."""
    return {n: stats for n, _, _, stats in results}


def _bench_source_label(*, gguf_path: str | None = None, mlx_path: str | None = None) -> str:
    """Concise descriptive label for a model source — used in the legend
    line of the comparison table (full name, no truncation)."""
    if gguf_path:
        return Path(gguf_path).stem  # drop .gguf
    if mlx_path:
        p = Path(mlx_path)
        return p.name or p.parent.name
    return "?"


def _shorten_pair(a: str, b: str, *, max_len: int = 14) -> tuple[str, str]:
    """Strip the longest shared prefix between two labels (only on '-' /
    '_' / '.' boundaries — never mid-token), then truncate each to max_len.
    Keeps column headers readable for sibling-named sources like
    'Qwen3.6-27B-Q4_K-pure' / 'Qwen3.6-27B-mlx-community-4bit'."""
    if a == b:
        return a[:max_len], b[:max_len]
    boundary = set("-_.")
    common = 0
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            break
        if ca in boundary:
            common = i + 1
    sa, sb = a[common:] or a, b[common:] or b
    if len(sa) > max_len:
        sa = sa[: max_len - 1] + "…"
    if len(sb) > max_len:
        sb = sb[: max_len - 1] + "…"
    return sa, sb


def print_comparison(
    prefill_results,
    decode_results,
    ref_prefill: dict[int, dict],
    ref_decode: dict[int, dict],
    *,
    primary_label: str = "MLX",
    ref_label: str = "llama.cpp",
) -> None:
    """Side-by-side comparison table.

    Full labels appear in a legend block above the table; column headers
    use a shortened-and-prefix-stripped form so the columns stay narrow.
    Ratio column is always primary/ref."""
    short_p, short_r = _shorten_pair(primary_label, ref_label)
    print()
    print("=== Comparison (median of trailing samples) ===")
    print(f"  A: {primary_label}")
    print(f"  B: {ref_label}")
    print()
    cell_w = 28  # accommodates "12345.6 [12345.6-12345.6 / 9]"
    p_hdr = f"A: {short_p}"
    r_hdr = f"B: {short_r}"
    ratio_hdr = "A/B"
    ratio_w = 8
    print("Prefill:")
    print(f"  {'n_prompt':>9}  {p_hdr:>{cell_w}}  {r_hdr:>{cell_w}}  {ratio_hdr:>{ratio_w}}")
    for n, r, _, primary_stats in prefill_results:
        primary_tps = r.prompt_tps
        primary_cell = f"{primary_tps:.1f} {_format_spread(primary_stats)}"
        s = ref_prefill.get(n)
        if s is None:
            ref_cell, ratio_str = "(no data)", "-"
        else:
            ref_tps = s["median"]
            ref_cell = f"{ref_tps:.1f} {_format_spread(s)}"
            ratio_str = f"{primary_tps / ref_tps:.2f}x"
        print(f"  {n:>9}  {primary_cell:>{cell_w}}  {ref_cell:>{cell_w}}  {ratio_str:>{ratio_w}}")
    print()
    print("Decode:")
    if not decode_results:
        print("  (no primary decode result)")
        return
    print(f"  {'n_gen':>9}  {p_hdr:>{cell_w}}  {r_hdr:>{cell_w}}  {ratio_hdr:>{ratio_w}}")
    for n_gen, r, _, primary_stats in decode_results:
        primary_tps = r.generation_tps
        primary_cell = f"{primary_tps:.1f} {_format_spread(primary_stats)}"
        s = ref_decode.get(n_gen)
        if s is None:
            ref_cell, ratio_str = "(no data)", "-"
        else:
            ref_tps = s["median"]
            ref_cell = f"{ref_tps:.1f} {_format_spread(s)}"
            ratio_str = f"{primary_tps / ref_tps:.2f}x"
        print(
            f"  {n_gen:>9}  {primary_cell:>{cell_w}}  {ref_cell:>{cell_w}}  {ratio_str:>{ratio_w}}"
        )


# ---------------------------------------------------------------------------
# Group-of-8 cross-N-row layout transform (experimental)
# ---------------------------------------------------------------------------


def _repack_q4k_interleave_np(data_np, K, group_size):
    """Repack Q4_K uint8 buffer from standard to group-of-N interleaved layout.

    Standard: each row is [hdr(16B) qs(128B)] × blocks_per_row, contiguous.
    Interleaved: G adjacent rows share each superblock's cache lines:
        [G×hdr(G*16B), 4×(G×qs_pair)(G*32B)].
    """
    G = group_size
    N = data_np.shape[0]
    blocks_per_row = K // 256
    assert data_np.shape[1] == blocks_per_row * 144
    assert N % G == 0

    sb = data_np.reshape(N, blocks_per_row, 144)
    hdr = sb[:, :, :16]
    qs = sb[:, :, 16:]

    hdr_g = hdr.reshape(N // G, G, blocks_per_row, 16)
    qs_g = qs.reshape(N // G, G, blocks_per_row, 4, 32)

    hdr_i = hdr_g.transpose(0, 2, 1, 3).reshape(N // G, blocks_per_row, G * 16)
    qs_i = qs_g.transpose(0, 2, 3, 1, 4).reshape(N // G, blocks_per_row, G * 128)

    out = np.concatenate([hdr_i, qs_i], axis=2)
    return out.reshape(N, blocks_per_row * 144)


def repack_model_q4k_interleave(model, group_size):
    """Walk model leaves and repack all Q4_K weight tensors to interleaved layout.

    Skips KQuantEmbedding modules: embedding lookup uses mx.dequantize
    (single-row gather), which doesn't have interleaved addressing.
    """
    G = group_size
    count = 0
    skipped = 0
    for path, module in model.named_modules():
        if not hasattr(module, "kquant_type") or module.kquant_type != "q4_k":
            continue
        if isinstance(module, KQuantEmbedding):
            skipped += 1
            continue
        w = module.weight
        if w.ndim != 2:
            continue
        N, bpr = w.shape
        if N % G != 0:
            print(f"  [g{G}] skip {path}: N={N} not divisible by {G}")
            continue
        K = (bpr // 144) * 256
        w_np = np.array(w)
        w_ig = _repack_q4k_interleave_np(w_np, K, G)
        module.weight = mx.array(w_ig)
        count += 1
    if skipped:
        print(f"  [g{G}] skipped {skipped} KQuantEmbedding module(s)")
    mx.eval(model.parameters())
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "gguf",
        nargs="?",
        default=None,
        help="Path to GGUF file. Optional when --mlx-source is "
        "given (still required for --vs-llama-bench, which "
        "needs a GGUF as the llama.cpp reference).",
    )
    ap.add_argument(
        "--mlx-source",
        default=None,
        help="Load a native MLX checkpoint from this path (HF id "
        "or local mlx_lm-format directory) instead of a GGUF "
        "file. Useful for A/B-ing native MLX quantization "
        "(e.g. UD-MLX-4bit) against KQuant or llama.cpp on "
        "the same architecture. When combined with a `gguf` "
        "positional + --vs-llama-bench, runs MLX-native vs "
        "llama.cpp (the GGUF is used only for llama-bench).",
    )
    ap.add_argument(
        "--prompt",
        default="Hello, world!",
        help="Generation prompt (default: 'Hello, world!'). "
        "Ignored if --prompt-tokens or --bench is set.",
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="Synthesize a prompt of exactly N tokens from filler "
        "text (skips chat template, no BOS prepend beyond what "
        "the tokenizer dictates). Useful for prefill timing.",
    )
    ap.add_argument(
        "--bench",
        default="",
        help="Comma-separated prompt lengths in tokens "
        "(e.g., 128,512,2048,8192). Sweeps prompt lengths and "
        "prints a prefill/decode tps table. "
        "Combine with --max-tokens to set the decode length "
        "per run (default 16 in --bench mode).",
    )
    ap.add_argument(
        "--bench-runs",
        type=int,
        default=3,
        help="Number of timed runs per prompt length (default 3); "
        "the best run by prefill tps is reported (Metal "
        "pipeline-state warmup is the dominant noise source).",
    )
    ap.add_argument(
        "--bench-no-warmup",
        action="store_true",
        help="Skip the global warmup pass before benchmarking. "
        "Without it, the first length will report cold-cache "
        "numbers and bias low.",
    )
    ap.add_argument(
        "--bench-mode",
        choices=["combined", "split"],
        default="split",
        help="split (default): separate prefill-only (max_tokens=1) "
        "and decode-only (BOS prompt) timings — apples-to-apples "
        "with llama-bench's pp/tg split. "
        "combined: one stream_generate per trial — realistic "
        "prompt+decode workload.",
    )
    ap.add_argument(
        "--vs-llama-bench",
        action="store_true",
        help="After --bench, also run llama.cpp's llama-bench "
        "with matching parameters and print a side-by-side "
        "comparison. Implies --bench-mode split.",
    )
    ap.add_argument(
        "--vs-mlx-source",
        default=None,
        help="After --bench, load this MLX-native checkpoint and "
        "re-run the same bench, then print a side-by-side "
        "comparison vs the primary source (gguf/kquant or "
        "--mlx-source). Useful for kernel-isolation A/Bs "
        "(e.g. kquant Q4_K vs flat affine 4-bit on the same "
        "model). Mutually exclusive with --vs-llama-bench. "
        "Implies --bench-mode split.",
    )
    ap.add_argument(
        "--llama-bench-bin",
        default="llama-bench",
        help="Path to llama-bench binary (default: llama-bench on $PATH).",
    )
    ap.add_argument(
        "--max-tokens",
        type=str,
        default="100",
        help="Decode-token count. Single int (default 100) for "
        "non-bench paths and combined-bench. In --bench-mode "
        "split, accepts a comma-separated list to sweep "
        "decode lengths (e.g., '16,32,64'); --vs-llama-bench "
        "passes the same list to llama-bench's -n.",
    )
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument(
        "--arch", default=None, help="Override architecture detection (gemma3/gemma4/qwen3/...)."
    )
    ap.add_argument(
        "--target-prefix", default="", help="Prepend this prefix to all remapped tensor names"
    )
    ap.add_argument(
        "--no-remap", action="store_true", help="Skip GGUF→HF name remap (raw GGUF names)"
    )
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Pass the prompt verbatim, even if the tokenizer "
        "has a chat template. Use for base/non-instruct models.",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="Load + show inventory; skip model build + generation",
    )
    ap.add_argument(
        "--fail-on-unknown",
        action="store_true",
        help="Hard-fail if any GGUF tensor has no remap entry",
    )
    ap.add_argument(
        "--profile-detail",
        action="store_true",
        help="Fine-grained prefill profiling: breaks down each "
        "layer bucket (attn/ssm/mlp) into per-projection "
        "timing (gate_proj, in_proj_qkv, etc). Shows where "
        "time is spent within each module type.",
    )
    ap.add_argument(
        "--profile-layers",
        action="store_true",
        help="Run prefill at each --bench length (default 1024 "
        "if --bench omitted) under a layer-bucket profiler. "
        "Wraps each transformer layer's child sub-modules "
        "with a timer, classifies them as attn / ssm / mlp / "
        "other, and prints a per-bucket wallclock split. "
        "The forced mx.eval inside each wrapper inflates "
        "absolute wallclock vs unprofiled --bench; use the "
        "table for relative bucket attribution, not for "
        "tps comparisons.",
    )
    ap.add_argument(
        "--interleave-group",
        type=int,
        default=0,
        metavar="N",
        help="Repack Q4_K weight tensors to group-of-N interleaved "
        "layout (4 or 8) for reduced cache-line amplification. "
        "Requires MLX built with KQ_INTERLEAVE_GROUP=N.",
    )
    args = ap.parse_args()

    # Source validation: exactly one of {gguf positional, --mlx-source} must
    # supply the model. --vs-llama-bench additionally needs a GGUF for the
    # llama.cpp reference even under --mlx-source.
    if not args.gguf and not args.mlx_source:
        ap.error("Either gguf positional or --mlx-source is required.")
    if args.mlx_source and args.report_only:
        ap.error("--report-only only applies to GGUF files (no --mlx-source).")
    if args.mlx_source and args.vs_llama_bench and not args.gguf:
        ap.error(
            "--vs-llama-bench with --mlx-source needs a gguf positional "
            "(used as the llama.cpp reference)."
        )
    if args.gguf and args.mlx_source and not args.vs_llama_bench:
        ap.error(
            "gguf positional + --mlx-source is only valid with "
            "--vs-llama-bench (otherwise the GGUF is unused — the model "
            "is loaded from --mlx-source)."
        )
    if args.vs_llama_bench and args.vs_mlx_source:
        ap.error(
            "--vs-llama-bench and --vs-mlx-source are mutually exclusive (one comparison per run)."
        )
    if args.vs_mlx_source and not args.bench:
        ap.error(
            "--vs-mlx-source requires --bench (it re-runs the same bench "
            "on the secondary checkpoint)."
        )
    # --report-only short-circuits before model build.
    if args.report_only:
        t0 = time.perf_counter()
        arrays, kquant_meta, arch_meta = load_gguf_via_mx(args.gguf)
        print(
            f"[mx.load] {len(arrays)} arrays, {len(kquant_meta)} kquant "
            f"({time.perf_counter() - t0:.2f}s)"
        )
        arch = args.arch or arch_meta
        if arch is None:
            print(
                "FATAL: could not determine arch from GGUF metadata; pass --arch", file=sys.stderr
            )
            return 1
        print(f"[arch] {arch}")
        reader_for_meta = GGUFReader(args.gguf, "r")

        def _read_first_int_ro(rdr, key):
            f = rdr.fields.get(key)
            return int(f.parts[f.data[0]][0]) if f else None

        def _read_first_nonzero_int_ro(rdr, key):
            f = rdr.fields.get(key)
            if f is None:
                return None
            for i in f.data:
                v = int(f.parts[i][0])
                if v > 0:
                    return v
            return int(f.parts[f.data[0]][0])

        n_head = _read_first_int_ro(reader_for_meta, f"{arch}.attention.head_count")
        n_head_kv = _read_first_nonzero_int_ro(reader_for_meta, f"{arch}.attention.head_count_kv")
        _, hf_kquant_meta, stats = remap_arrays(
            arrays,
            kquant_meta,
            arch,
            no_remap=args.no_remap,
            target_prefix=args.target_prefix,
            fail_on_unknown=args.fail_on_unknown,
            n_head=n_head,
            n_head_kv=n_head_kv,
        )
        print_inventory(arch, kquant_meta, hf_kquant_meta, stats)
        return 0

    # Model load: native MLX checkpoint via mlx_lm.load, or GGUF→kquant
    # pipeline via load_kquant_model.
    if args.mlx_source:
        print(f"[mlx-native] loading {args.mlx_source}")
        t0 = time.perf_counter()
        model, tokenizer = mlx_lm.load(args.mlx_source)
        print(f"[mlx-native] loaded in {time.perf_counter() - t0:.1f}s")
        config = None
    else:
        # Full pipeline: load → remap → build (from synth config) → sanitize
        # → install → load_weights → tokenizer (from GGUF metadata).
        model, config, tokenizer = load_kquant_model(
            args.gguf,
            arch=args.arch,
            target_prefix=args.target_prefix,
            no_remap=args.no_remap,
            fail_on_unknown=args.fail_on_unknown,
        )

    if args.interleave_group:
        G = args.interleave_group
        t0 = time.perf_counter()
        n = repack_model_q4k_interleave(model, G)
        print(f"[g{G}] repacked {n} Q4_K tensors ({time.perf_counter() - t0:.1f}s)")

    sampler = make_sampler(temp=args.temp)

    # --profile-detail / --profile-layers: prefill-only sweep with wallclock
    # split. Independent of --bench; if --bench is provided we reuse the
    # length list (it's the natural sweep), otherwise default to 1024.
    if args.profile_detail or args.profile_layers:
        if args.bench:
            try:
                lengths = _parse_int_list(args.bench, flag="--bench")
            except argparse.ArgumentTypeError as e:
                print(str(e), file=sys.stderr)
                return 2
        else:
            lengths = [1024]
        print(
            f"[profile] prefill layer-bucket sweep: lengths={lengths}  "
            f"runs/length={args.bench_runs}"
        )
        # Warmup once at the largest length to populate Metal kernel cache;
        # otherwise the first profiled run absorbs JIT compile time.
        if not args.bench_no_warmup:
            warm_n = max(lengths)
            print(f"[profile] warmup forward pass at n_prompt={warm_n} ...", flush=True)
            from mlx_lm.models import cache as cache_mod

            warm_ids = synthesize_prompt_ids(tokenizer, warm_n)
            warm_cache = cache_mod.make_prompt_cache(model, max_kv_size=None)
            warm_out = model(mx.array([warm_ids], dtype=mx.int32), cache=warm_cache)
            mx.eval(warm_out.logits if hasattr(warm_out, "logits") else warm_out)
        if args.profile_detail:
            summary = profile_prefill_detail(model, tokenizer, lengths, runs=args.bench_runs)
            print_detail_profile_table(summary)
        else:
            summary = profile_prefill(model, tokenizer, lengths, runs=args.bench_runs)
            print_profile_table(summary)
        return 0

    # Bench mode: sweep synthesized prompts of declared lengths, print a table.
    # Skips chat templating entirely (we want the model to see exactly N tokens).
    if args.bench:
        try:
            lengths = _parse_int_list(args.bench, flag="--bench")
        except argparse.ArgumentTypeError as e:
            print(str(e), file=sys.stderr)
            return 2
        # In --bench mode, --max-tokens default of "100" means "use 16" (small
        # decode keeps the prefill measurement honest); otherwise honor user.
        decode_arg = "16" if args.max_tokens == "100" else args.max_tokens
        try:
            decode_tokens_list = _parse_int_list(decode_arg, flag="--max-tokens")
        except argparse.ArgumentTypeError as e:
            print(str(e), file=sys.stderr)
            return 2
        bench_mode = args.bench_mode
        if args.vs_llama_bench and bench_mode != "split":
            print(
                "[vs-llama-bench] forcing --bench-mode split for apples-to-apples comparison",
                file=sys.stderr,
            )
            bench_mode = "split"
        if args.vs_mlx_source and bench_mode != "split":
            print(
                "[vs-mlx-source] forcing --bench-mode split for apples-to-apples comparison",
                file=sys.stderr,
            )
            bench_mode = "split"
        if bench_mode != "split" and len(decode_tokens_list) > 1:
            print(
                "--max-tokens accepts a comma-separated sweep only with "
                "--bench-mode split (combined mode runs one prompt+decode "
                "trial per length).",
                file=sys.stderr,
            )
            return 2
        print(
            f"[bench] mode={bench_mode}  prompt lengths: {lengths}  "
            f"decode tokens: {decode_tokens_list}  "
            f"runs/length: {args.bench_runs}  "
            f"warmup: {not args.bench_no_warmup}"
        )
        if bench_mode == "split":
            prefill_results, decode_results = run_benchmark_split(
                model,
                tokenizer,
                lengths,
                decode_tokens_list=decode_tokens_list,
                sampler=sampler,
                runs=args.bench_runs,
                warmup=not args.bench_no_warmup,
            )
            print_bench_table_split(prefill_results, decode_results)
            if args.vs_llama_bench:
                try:
                    llama_prefill, llama_decode = run_llama_bench(
                        args.gguf,
                        lengths,
                        decode_tokens_list,
                        args.bench_runs,
                        binary=args.llama_bench_bin,
                        no_warmup=args.bench_no_warmup,
                    )
                except FileNotFoundError as e:
                    print(f"[vs-llama-bench] {e}", file=sys.stderr)
                    return 1
                except subprocess.CalledProcessError as e:
                    print(
                        f"[vs-llama-bench] llama-bench failed (exit={e.returncode}):\n{e.stderr}",
                        file=sys.stderr,
                    )
                    return 1
                # MLX/llama labels distinguish the runtimes (both sides are
                # running the same GGUF), not the model — keep them short.
                if args.mlx_source:
                    mlx_label = "MLX (native)"
                else:
                    mlx_label = "MLX (kquant)"
                print_comparison(
                    prefill_results,
                    decode_results,
                    llama_prefill,
                    llama_decode,
                    primary_label=mlx_label,
                    ref_label="llama.cpp",
                )
            elif args.vs_mlx_source:
                # Free primary before loading secondary — both are 27B-class
                # checkpoints and holding both at once approaches OOM.
                primary_label = _bench_source_label(
                    gguf_path=args.gguf if not args.mlx_source else None, mlx_path=args.mlx_source
                )
                ref_label = _bench_source_label(mlx_path=args.vs_mlx_source)
                print(f"[vs-mlx-source] freeing primary, loading {ref_label}", flush=True)
                del model
                gc.collect()
                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
                t0 = time.perf_counter()
                ref_model, ref_tokenizer = mlx_lm.load(args.vs_mlx_source)
                print(f"[vs-mlx-source] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
                ref_prefill_t, ref_decode_t = run_benchmark_split(
                    ref_model,
                    ref_tokenizer,
                    lengths,
                    decode_tokens_list=decode_tokens_list,
                    sampler=sampler,
                    runs=args.bench_runs,
                    warmup=not args.bench_no_warmup,
                )
                print_bench_table_split(ref_prefill_t, ref_decode_t)
                print_comparison(
                    prefill_results,
                    decode_results,
                    _split_tuples_to_ref_dict(ref_prefill_t),
                    _split_tuples_to_ref_dict(ref_decode_t),
                    primary_label=primary_label,
                    ref_label=ref_label,
                )
        else:
            results = run_benchmark(
                model,
                tokenizer,
                lengths,
                max_tokens=decode_tokens_list[0],
                sampler=sampler,
                runs=args.bench_runs,
                warmup=not args.bench_no_warmup,
            )
            print_bench_table(results)
        return 0

    # Non-bench paths take a single max_tokens value.
    try:
        decode_tokens_list = _parse_int_list(args.max_tokens, flag="--max-tokens")
    except argparse.ArgumentTypeError as e:
        print(str(e), file=sys.stderr)
        return 2
    if len(decode_tokens_list) > 1:
        print(
            "--max-tokens accepts a comma-separated sweep only with --bench --bench-mode split.",
            file=sys.stderr,
        )
        return 2
    max_tokens = decode_tokens_list[0]

    # Single-prompt synth mode: skip chat template, decode small to focus on prefill.
    if args.prompt_tokens > 0:
        prompt_ids = synthesize_prompt_ids(tokenizer, args.prompt_tokens)
        print(
            f"[generate] synthesized prompt: {len(prompt_ids)} tokens, "
            f"max_tokens={max_tokens} temp={args.temp}"
        )
        print()
        mlx_lm.generate(
            model,
            tokenizer,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=True,
        )
        return 0

    # Apply the chat template if the tokenizer has one and --no-chat-template
    # was not set. Instruct-tuned models (gemma-4-it, qwen3-it) require this;
    # without it they tend to spam control tokens.
    prompt = args.prompt
    if not args.no_chat_template and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"[generate] applied chat template ({len(args.prompt)} → {len(prompt)} chars)")

    print(f"[generate] max_tokens={max_tokens} temp={args.temp}")
    print()
    mlx_lm.generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
