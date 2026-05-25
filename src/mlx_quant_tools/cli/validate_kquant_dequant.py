"""Bit-exact validation: `mx.dequantize(mode="kquant")` vs `gguf.quants` reference.

Walks a GGUF, dequantizes each quantized tensor twice --- once with the upstream
`gguf.quants.dequantize` numpy reference, once through MLX's Metal kquant
kernel --- and compares. Reports per-codec pass/fail counts.

Usage:
    mqt-validate-dequant <gguf> [--limit-tensors N] \\
        [--codecs q4_k,q8_0,...] [--verbose]

Exits non-zero if any tensor fails.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, quants

# Codec geometry --- source of truth: mlx::core::kquant_codec_by_name in
# mlx/primitives.cpp; mirrored in load_gguf_kquant.py:KQUANT_CODECS.
# (weights_per_block, bytes_per_block, group_size, bits)
KQUANT_CODECS: dict[GGMLQuantizationType, tuple[int, int, int, int, str]] = {
    GGMLQuantizationType.Q4_0: (32, 18, 32, 4, "q4_0"),
    GGMLQuantizationType.Q4_1: (32, 20, 32, 4, "q4_1"),
    GGMLQuantizationType.Q5_0: (32, 22, 32, 5, "q5_0"),
    GGMLQuantizationType.Q5_1: (32, 24, 32, 5, "q5_1"),
    GGMLQuantizationType.Q8_0: (32, 34, 32, 8, "q8_0"),
    GGMLQuantizationType.Q2_K: (256, 84, 256, 2, "q2_k"),
    GGMLQuantizationType.Q3_K: (256, 110, 256, 3, "q3_k"),
    GGMLQuantizationType.Q4_K: (256, 144, 256, 4, "q4_k"),
    GGMLQuantizationType.Q5_K: (256, 176, 256, 5, "q5_k"),
    GGMLQuantizationType.Q6_K: (256, 210, 256, 6, "q6_k"),
}

ATOL_LOOSE = 1e-3
RTOL_LOOSE = 1e-3


def _validate_one(tensor, *, verbose: bool) -> tuple[str, str, str]:
    """Validate one tensor. Returns (codec, status, detail).

    status is one of: 'bit_exact', 'loose', 'fail', 'unsupported'.
    """
    geom = KQUANT_CODECS.get(tensor.tensor_type)
    if geom is None:
        return (tensor.tensor_type.name, "unsupported", "")
    wpb, bpb, gs, bits, codec = geom

    # Reference path: gguf.quants on the raw wire bytes.
    raw = np.ascontiguousarray(tensor.data, dtype=np.uint8)
    ref = quants.dequantize(raw, tensor.tensor_type).astype(np.float32)

    # MLX path: pack the wire bytes as uint8 [..., bytes_per_row], call
    # mx.dequantize with a vestigial scales placeholder.
    logical_shape = [int(d) for d in tensor.shape][::-1]
    last_dim = logical_shape[-1]
    if last_dim % wpb != 0:
        return (codec, "fail", f"last dim {last_dim} not divisible by weights_per_block {wpb}")
    packed_shape = list(logical_shape)
    packed_shape[-1] = (last_dim // wpb) * bpb
    w = mx.array(raw.reshape(packed_shape))
    scales = mx.zeros((1,), dtype=mx.uint8)
    out = mx.dequantize(
        w,
        scales,
        group_size=gs,
        bits=bits,
        mode="kquant",
        kquant_type=codec,
    )
    mx.eval(out)
    mlx_arr = np.array(out).astype(np.float32).reshape(ref.shape)

    if np.array_equal(ref, mlx_arr):
        if verbose:
            print(f"  {tensor.name:<60} {codec:<5} bit-exact (shape={tuple(ref.shape)})")
        return (codec, "bit_exact", "")

    diff = np.abs(ref - mlx_arr)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(ref) + 1e-12)).max())
    if np.allclose(ref, mlx_arr, atol=ATOL_LOOSE, rtol=RTOL_LOOSE):
        detail = f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        if verbose:
            print(f"  {tensor.name:<60} {codec:<5} LOOSE  {detail}")
        return (codec, "loose", detail)

    detail = f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
    print(f"  FAIL {tensor.name:<55} {codec:<5} {detail}", file=sys.stderr)
    return (codec, "fail", detail)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="mqt-validate-dequant",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("gguf", help="Path to GGUF file")
    ap.add_argument(
        "--limit-tensors",
        type=int,
        default=0,
        help="Stop after validating N quantized tensors (0 = all)",
    )
    ap.add_argument(
        "--codecs",
        default="",
        help="Comma-separated codec allowlist (e.g., q4_k,q8_0). Default: all 10 kquant codecs.",
    )
    ap.add_argument("--verbose", action="store_true", help="Print one line per tensor")
    args = ap.parse_args()

    allow = {c.strip().lower() for c in args.codecs.split(",") if c.strip()}

    reader = GGUFReader(args.gguf, "r")
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"bit_exact": 0, "loose": 0, "fail": 0}
    )
    # Track quantized tensors whose codec is NOT in our K-quant kernel stack
    # (e.g., IQ4_XS / IQ3_XXS --- Unsloth UD packings sometimes mix these in
    # alongside K-quants). These are silently invisible to the validator
    # otherwise, but they will break run-gguf-kquant.py end-to-end.
    unsupported_counts: dict[str, int] = defaultdict(int)
    # Non-quantized passthrough tensors (F32 / F16 / BF16) --- informational.
    passthrough_count = 0
    fail_examples: list[tuple[str, str, str]] = []
    n_validated = 0
    n_skipped = 0

    PASSTHROUGH_TYPES = {"F32", "F16", "BF16"}

    for t in reader.tensors:
        if t.tensor_type not in KQUANT_CODECS:
            tname = t.tensor_type.name
            if tname in PASSTHROUGH_TYPES:
                passthrough_count += 1
            else:
                unsupported_counts[tname] += 1
            continue
        codec = KQUANT_CODECS[t.tensor_type][4]
        if allow and codec not in allow:
            n_skipped += 1
            continue
        if args.limit_tensors and n_validated >= args.limit_tensors:
            break

        codec, status, detail = _validate_one(t, verbose=args.verbose)
        if status == "unsupported":
            continue
        counts[codec][status] += 1
        if status == "fail":
            fail_examples.append((t.name, codec, detail))
        n_validated += 1

    print()
    print(f"=== validate-kquant-dequant: {args.gguf} ===")
    print(f"validated: {n_validated} tensors  (skipped by --codecs filter: {n_skipped})")
    if passthrough_count:
        print(f"passthrough (F32/F16/BF16, not validated): {passthrough_count}")
    print()
    print(f"  {'codec':<6} {'bit-exact':>10} {'loose':>8} {'fail':>6}")
    total = {"bit_exact": 0, "loose": 0, "fail": 0}
    for codec in sorted(counts):
        c = counts[codec]
        for k in total:
            total[k] += c[k]
        print(f"  {codec:<6} {c['bit_exact']:>10} {c['loose']:>8} {c['fail']:>6}")
    print(f"  {'-' * 6} {'-' * 10} {'-' * 8} {'-' * 6}")
    print(f"  {'total':<6} {total['bit_exact']:>10} {total['loose']:>8} {total['fail']:>6}")

    if fail_examples:
        print()
        print(f"FAIL examples ({min(5, len(fail_examples))} of {len(fail_examples)}):")
        for name, codec, detail in fail_examples[:5]:
            print(f"  {name}  {codec}  {detail}")

    if unsupported_counts:
        print()
        print("WARNING: GGUF contains quantized tensors with codecs NOT in the")
        print("MLX K-quant kernel stack. These cannot be loaded by load_gguf_kquant")
        print("or run-gguf-kquant; they will be silently zero-init'd, producing")
        print("garbage inference. Re-pack the GGUF without these codecs:")
        for codec_name, n in sorted(unsupported_counts.items()):
            print(f"  {codec_name:<10} {n:>4} tensor(s)")

    # Hard-fail on either real validation failures or unsupported codecs in
    # the file --- a UD-style pack with a few IQ4_XS tensors will pass tensor
    # validation but break the end-to-end pipeline; loud signal beats subtle.
    return 0 if (total["fail"] == 0 and not unsupported_counts) else 1


if __name__ == "__main__":
    sys.exit(main())
