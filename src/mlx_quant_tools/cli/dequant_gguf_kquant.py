"""Load a GGUF K-quant file, dequant to bf16, and (optionally) write a safetensors
checkpoint loadable by mlx_lm / mqt-score-kld.

What this does
  1. Open a GGUF (typically an Unsloth UD-*-XL artifact with a mix of
     Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0 / F32 / BF16 tensors).
  2. Dequant each tensor to fp32 (bit-exact reproduction of llama.cpp's
     `dequantize_row_*`), then cast to bf16.
  3. Apply a per-architecture name + layout remap so the output uses the
     HF/MLX tensor names the upstream HF repo would have. Vision/audio tower
     tensors are skipped with a warning (mlx_vlm-side concern).
  4. Optionally write a sharded safetensors checkpoint plus copy
     config.json + tokenizer files from an upstream HF repo, producing a
     directory directly loadable by mlx_lm.load.

Bit-exactness contract: `--validate` compares our dequant against
`gguf.quants` reference dequant via bytewise float32 equality.

Usage:
  mqt-dequant-gguf <gguf> [--validate] [--dry-run]
                           [--out-dir DIR] [--hf-source REPO_ID]
                           [--arch ARCH] [--no-remap]
                           [--max-shard-gb N] [--limit-tensors N]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mlx_quant_tools.gguf_name_remap import (
    RemapDecision,
    _read_string_field,
    detect_arch,
    parse_gguf_name,
)

# ---------------------------------------------------------------------------
# Codec on-disk constants. See ggml/src/ggml-common.h in llama.cpp.
# ---------------------------------------------------------------------------
QK_K = 256
K_SCALE_SIZE = 12
Q4_K_BLOCK_BYTES = 2 * 2 + K_SCALE_SIZE + QK_K // 2  # 144
Q5_K_BLOCK_BYTES = 2 * 2 + K_SCALE_SIZE + QK_K // 2 + QK_K // 8  # 176
Q6_K_BLOCK_BYTES = 2 + QK_K // 16 + 3 * QK_K // 4  # 210

QK5_1 = 32
Q5_1_BLOCK_BYTES = 2 * 2 + 4 + QK5_1 // 2  # 24

QK8_0 = 32
Q8_0_BLOCK_BYTES = 2 + QK8_0  # 34


# ---------------------------------------------------------------------------
# K-quant dequant --- pure-numpy port of llama.cpp's dequantize_row_* functions.
# Validation contract: bit-exact match vs gguf.quants reference.
# ---------------------------------------------------------------------------


def _unpack_scale_min_q4k(scales12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the 12-byte `scales[K_SCALE_SIZE]` field into 8 sub-block
    (6-bit scale, 6-bit min) pairs. Reference: `get_scale_min_k4` in
    ggml-quants.c:818-825.
    """
    s = scales12.astype(np.uint8)
    n = s.shape[0]
    sc = np.empty((n, 8), dtype=np.uint8)
    mn = np.empty((n, 8), dtype=np.uint8)
    sc[:, 0:4] = s[:, 0:4] & 0x3F
    mn[:, 0:4] = s[:, 4:8] & 0x3F
    sc[:, 4:8] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, mn


def dequantize_q4_k(raw: bytes, n_elements: int) -> np.ndarray:
    """Pure-numpy port of `dequantize_row_q4_K` (ggml-quants.c:1467-1489)."""
    assert n_elements % QK_K == 0, f"n_elements {n_elements} not divisible by QK_K={QK_K}"
    n_blocks = n_elements // QK_K
    expected = n_blocks * Q4_K_BLOCK_BYTES
    assert len(raw) == expected, f"q4_K: got {len(raw)} bytes, expected {expected}"

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q4_K_BLOCK_BYTES)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    sc8, mn8 = _unpack_scale_min_q4k(blocks[:, 4:16])
    qs = blocks[:, 16 : 16 + QK_K // 2]

    sub_scale = d[:, None] * sc8.astype(np.float32)
    sub_min = dmin[:, None] * mn8.astype(np.float32)

    qs = qs.reshape(n_blocks, 4, 32)
    low_nib = (qs & 0x0F).astype(np.float32)
    high_nib = (qs >> 4).astype(np.float32)
    sub_q = np.stack([low_nib, high_nib], axis=2).reshape(n_blocks, 8, 32)

    out = sub_scale[:, :, None] * sub_q - sub_min[:, :, None]
    return out.reshape(n_blocks * QK_K).astype(np.float32)


def dequantize_q5_k(raw: bytes, n_elements: int) -> np.ndarray:
    """Pure-numpy port of `dequantize_row_q5_K` (ggml-quants.c:1669-1694)."""
    assert n_elements % QK_K == 0
    n_blocks = n_elements // QK_K
    expected = n_blocks * Q5_K_BLOCK_BYTES
    assert len(raw) == expected, f"q5_K: got {len(raw)} bytes, expected {expected}"

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q5_K_BLOCK_BYTES)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    sc8, mn8 = _unpack_scale_min_q4k(blocks[:, 4:16])
    qh = blocks[:, 16:48]
    qs = blocks[:, 48 : 48 + QK_K // 2]

    sub_scale = d[:, None] * sc8.astype(np.float32)
    sub_min = dmin[:, None] * mn8.astype(np.float32)

    qs_g = qs.reshape(n_blocks, 4, 32)
    low_nib = (qs_g & 0x0F).astype(np.uint8)
    high_nib = (qs_g >> 4).astype(np.uint8)
    low_bits = np.stack([low_nib, high_nib], axis=2).reshape(n_blocks, 8, 32)

    bit_sel = np.arange(8, dtype=np.uint8).reshape(1, 8, 1)
    high_bit = (qh[:, None, :] >> bit_sel) & 0x01

    q5 = (low_bits | (high_bit << 4)).astype(np.float32)
    out = sub_scale[:, :, None] * q5 - sub_min[:, :, None]
    return out.reshape(n_blocks * QK_K).astype(np.float32)


def dequantize_q6_k(raw: bytes, n_elements: int) -> np.ndarray:
    """Pure-numpy port of `dequantize_row_q6_K` (ggml-quants.c:1877-1906)."""
    assert n_elements % QK_K == 0
    n_blocks = n_elements // QK_K
    expected = n_blocks * Q6_K_BLOCK_BYTES
    assert len(raw) == expected, f"q6_K: got {len(raw)} bytes, expected {expected}"

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q6_K_BLOCK_BYTES)
    ql = blocks[:, 0 : QK_K // 2]
    qh = blocks[:, QK_K // 2 : QK_K // 2 + QK_K // 4]
    scales = blocks[:, QK_K // 2 + QK_K // 4 : QK_K // 2 + QK_K // 4 + QK_K // 16]
    d = blocks[:, -2:].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    sc16 = scales.view(np.int8).astype(np.float32)

    ql_h = ql.reshape(n_blocks, 2, 64)
    qh_h = qh.reshape(n_blocks, 2, 32)
    sc_h = sc16.reshape(n_blocks, 2, 8)

    out = np.empty((n_blocks, 2, 128), dtype=np.float32)
    is_idx = np.arange(32) // 16
    for half_idx in range(2):
        ql_half = ql_h[:, half_idx, :]
        qh_half = qh_h[:, half_idx, :]
        sc_half = sc_h[:, half_idx, :]
        ql_lo = ql_half[:, 0:32]
        ql_lo32 = ql_half[:, 32:64]
        q1 = ((ql_lo & 0x0F) | (((qh_half >> 0) & 0x03) << 4)).astype(np.int8) - np.int8(32)
        q2 = ((ql_lo32 & 0x0F) | (((qh_half >> 2) & 0x03) << 4)).astype(np.int8) - np.int8(32)
        q3 = ((ql_lo >> 4) | (((qh_half >> 4) & 0x03) << 4)).astype(np.int8) - np.int8(32)
        q4 = ((ql_lo32 >> 4) | (((qh_half >> 6) & 0x03) << 4)).astype(np.int8) - np.int8(32)
        for is_off, qq, out_slice in (
            (0, q1, slice(0, 32)),
            (2, q2, slice(32, 64)),
            (4, q3, slice(64, 96)),
            (6, q4, slice(96, 128)),
        ):
            scl = sc_half[:, is_off + is_idx]
            d_eff = d[:, None] * scl
            out[:, half_idx, out_slice] = d_eff * qq.astype(np.float32)

    return out.reshape(n_blocks * QK_K).astype(np.float32)


# ---------------------------------------------------------------------------
# Flat-legacy codecs --- ports of llama.cpp's dequant routines.
# Sources: ggml/src/ggml-common.h, ggml/src/ggml-quants.c in llama.cpp.
# ---------------------------------------------------------------------------


def dequantize_q5_1(raw: bytes, n_elements: int) -> np.ndarray:
    """Pure-numpy port of `dequantize_row_q5_1` (ggml-quants.c:464-489).

    Block: 24 bytes, 32 weights. Layout: d (fp16, [0:2]), m (fp16, [2:4]),
    qh (uint32, [4:8]), qs (16 bytes low-nibbles, [8:24]).
    Sign convention: y = q5*d + m (Q5_1 stores +m, unlike Q4_K which stores -m).
    """
    assert n_elements % QK5_1 == 0, f"n_elements {n_elements} not divisible by QK5_1={QK5_1}"
    n_blocks = n_elements // QK5_1
    expected = n_blocks * Q5_1_BLOCK_BYTES
    assert len(raw) == expected, f"q5_1: got {len(raw)} bytes, expected {expected}"

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q5_1_BLOCK_BYTES)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    m = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qh = blocks[:, 4:8].copy().view(np.uint32).reshape(n_blocks, 1)  # 32-bit high-bit field
    qs = blocks[:, 8:24]  # (n_blocks, 16) low nibbles

    j = np.arange(QK5_1 // 2, dtype=np.uint32).reshape(1, QK5_1 // 2)  # j in [0..15]
    # xh_0[j] = bit j of qh, in position 4 (i.e., 0x10). y[j+0]  uses qh bit j.
    # xh_1[j] = bit (j+16) of qh, in position 4.        y[j+16] uses qh bit (j+16).
    xh_0 = ((qh >> j) << np.uint32(4)) & np.uint32(0x10)  # (n_blocks, 16)
    xh_1 = (qh >> (j + np.uint32(12))) & np.uint32(0x10)  # (n_blocks, 16)

    x0 = (qs.astype(np.uint32) & np.uint32(0x0F)) | xh_0  # in [0..31]
    x1 = (qs.astype(np.uint32) >> np.uint32(4)) | xh_1  # in [0..31]

    # Output ordering: y[0..15] = x0; y[16..31] = x1 (per the C kernel).
    out = np.empty((n_blocks, QK5_1), dtype=np.float32)
    out[:, 0:16] = x0.astype(np.float32) * d + m
    out[:, 16:32] = x1.astype(np.float32) * d + m
    return out.reshape(n_blocks * QK5_1)


def dequantize_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    """Pure-numpy port of `dequantize_row_q8_0` (ggml-quants.c:491-505).

    Block: 34 bytes, 32 weights. Layout: d (fp16, [0:2]), qs (32 x signed int8, [2:34]).
    Symmetric: y[j] = qs[j] * d.
    """
    assert n_elements % QK8_0 == 0, f"n_elements {n_elements} not divisible by QK8_0={QK8_0}"
    n_blocks = n_elements // QK8_0
    expected = n_blocks * Q8_0_BLOCK_BYTES
    assert len(raw) == expected, f"q8_0: got {len(raw)} bytes, expected {expected}"

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q8_0_BLOCK_BYTES)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = blocks[:, 2 : 2 + QK8_0].view(np.int8).astype(np.float32)  # (n_blocks, 32)
    return (qs * d).reshape(n_blocks * QK8_0)


# ---------------------------------------------------------------------------
# Codec dispatch
# ---------------------------------------------------------------------------

# GGMLQuantizationType enum values (verified):
#   0 F32, 1 F16, 7 Q5_1, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, 30 BF16
DEQUANT_FUNCS = {
    7: dequantize_q5_1,
    8: dequantize_q8_0,
    12: dequantize_q4_k,
    13: dequantize_q5_k,
    14: dequantize_q6_k,
}

BLOCK_BYTES = {
    7: Q5_1_BLOCK_BYTES,
    8: Q8_0_BLOCK_BYTES,
    12: Q4_K_BLOCK_BYTES,
    13: Q5_K_BLOCK_BYTES,
    14: Q6_K_BLOCK_BYTES,
}

# Pass-through codecs that don't need block-byte slicing.
PASSTHROUGH_TYPES = {0, 1, 30}  # F32, F16, BF16


def _gguf_block_size_for(qtype_id: int, qtype_name: str) -> int:
    """Return the block-element count (32 for legacy Q-, 256 for K-quants)."""
    if qtype_id in (12, 13, 14):
        return QK_K
    if qtype_id in (7, 8):
        return 32
    raise ValueError(f"no block-size mapping for {qtype_name} ({qtype_id})")


def _get_raw_bytes(reader_tensor) -> bytes:
    """Flatten a `ReaderTensor.data` (which gguf reshapes to (n_rows, row_bytes))
    back to a contiguous byte buffer of exactly the size the codec expects.
    """
    flat = np.ascontiguousarray(reader_tensor.data).reshape(-1).view(np.uint8)
    return flat.tobytes()


def dequantize_tensor(reader_tensor) -> np.ndarray:
    """Dispatch to the right dequant for a `ReaderTensor`. Returns a float32
    numpy array of length `n_elements` (flat --- caller reshapes).
    """
    qtype = reader_tensor.tensor_type
    qtype_id = int(qtype)
    n_elements = int(reader_tensor.n_elements)

    if qtype_id in DEQUANT_FUNCS:
        raw = _get_raw_bytes(reader_tensor)
        return DEQUANT_FUNCS[qtype_id](raw, n_elements)

    # Pass-through codecs (F32, F16, BF16). View raw bytes through the dtype.
    flat = np.ascontiguousarray(reader_tensor.data).reshape(-1).view(np.uint8)
    if qtype_id == 0:  # F32
        return flat.view(np.float32).copy()
    if qtype_id == 1:  # F16
        return flat.view(np.float16).astype(np.float32)
    if qtype_id == 30:  # BF16
        # numpy lacks native bf16; re-interpret as uint16 and shift into fp32.
        bf = flat.view(np.uint16).astype(np.uint32)
        return (bf << 16).view(np.float32).copy()

    raise NotImplementedError(
        f"unsupported tensor_type {qtype.name} ({qtype_id}) on {reader_tensor.name!r}; "
        f"supported: Q4_K, Q5_K, Q6_K, Q5_1, Q8_0, F32, F16, BF16"
    )


# ---------------------------------------------------------------------------
# --validate: bit-exact gate vs gguf.quants reference dequant
# ---------------------------------------------------------------------------


def validate_against_reference(reader_tensor) -> tuple[bool, str]:
    """Return (ok, detail). `ok=True` <-> our dequant matches `gguf.quants`
    bit-for-bit (uint32 view of the float32 result), NaN-equal.
    """
    from gguf import GGMLQuantizationType
    from gguf.quants import Q4_K, Q5_1, Q5_K, Q6_K, Q8_0

    cls_map = {
        GGMLQuantizationType.Q4_K: (Q4_K, Q4_K_BLOCK_BYTES),
        GGMLQuantizationType.Q5_K: (Q5_K, Q5_K_BLOCK_BYTES),
        GGMLQuantizationType.Q6_K: (Q6_K, Q6_K_BLOCK_BYTES),
        GGMLQuantizationType.Q5_1: (Q5_1, Q5_1_BLOCK_BYTES),
        GGMLQuantizationType.Q8_0: (Q8_0, Q8_0_BLOCK_BYTES),
    }
    qtype = reader_tensor.tensor_type
    if qtype not in cls_map:
        return True, "skip (not a quantized codec)"

    qtype_id = int(qtype)
    n_elements = int(reader_tensor.n_elements)
    raw = _get_raw_bytes(reader_tensor)
    ours = DEQUANT_FUNCS[qtype_id](raw, n_elements)

    cls, block_bytes = cls_map[qtype]
    n_blocks = n_elements // _gguf_block_size_for(qtype_id, qtype.name)
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, block_bytes)
    ref = cls.dequantize_blocks(blocks).reshape(-1).astype(np.float32)

    ok = (ours.shape == ref.shape) and np.array_equal(ours.view(np.uint32), ref.view(np.uint32))
    if ok:
        return True, "bit-exact"
    max_abs = float(np.max(np.abs(ours - ref)))
    return False, f"MISMATCH: max_abs_diff={max_abs:.3e}"


# Per-arch name + layout remap is in `gguf_name_remap.py` (shared with
# `run-gguf-kquant.py`). The canonical entry point is `parse_gguf_name`
# imported above; layout transforms (numpy-side) stay below.


# ---------------------------------------------------------------------------
# Layout transforms
# ---------------------------------------------------------------------------


def apply_qk_permute(arr: np.ndarray, n_head: int, n_head_kv: int | None = None) -> np.ndarray:
    """LLAMA-only Q/K permute. Inverts `convert_hf_to_gguf.py::LlamaModel.permute`
    (line 2885).

    The forward permute is reshape(N, 2, D) -> swapaxes(1,2); the inverse
    is reshape(N, D, 2) -> swapaxes(1,2).  NOT self-inverse when D != 2.

    For Q_proj on a GQA model (n_head_kv != n_head): pass n_head_kv=None (or n_head)
    so the leading axis count is n_head. For K_proj: pass the actual n_head_kv ---
    the upstream reference replaces n_head with n_head_kv before the reshape, since
    K_proj has shape (n_head_kv * head_dim, hidden), not (n_head * head_dim, hidden).
    """
    if n_head_kv is not None and n_head_kv != n_head:
        n_head = n_head_kv
    n_out, hidden = arr.shape
    head_dim = n_out // n_head
    return arr.reshape(n_head, head_dim // 2, 2, hidden).swapaxes(1, 2).reshape(n_out, hidden)


def split_fused_gate_up_exps(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a fused MoE `ffn_gate_up_exps` tensor into separate gate/up.

    Input shape (after GGUF axis reversal): (n_experts, 2 * intermediate, hidden).
    The fused axis is the second one --- the first half is gate, the second half is up.
    (Convention: matches HF gate_up_proj packed format.)
    """
    if arr.ndim != 3:
        raise ValueError(f"split_fused_gate_up_exps expected 3D tensor, got shape {arr.shape}")
    n_experts, two_intermediate, hidden = arr.shape
    if two_intermediate % 2 != 0:
        raise ValueError(f"fused axis must be even, got {two_intermediate}")
    half = two_intermediate // 2
    gate = arr[:, :half, :]  # (n_experts, intermediate, hidden)
    up = arr[:, half:, :]  # (n_experts, intermediate, hidden)
    return gate, up


# GGUF metadata + arch detection (`detect_arch`, `_read_string_field`) moved
# to `gguf_name_remap.py`; imported above.


# ---------------------------------------------------------------------------
# Inventory output
# ---------------------------------------------------------------------------


def _gb(n_bytes: int) -> float:
    return n_bytes / 1e9


def print_inventory(
    reader, tensors, *, header: str = "Tensor inventory", gguf_path: str = ""
) -> None:
    """Print a per-codec rollup table. Used by --dry-run + as the default
    pre-load summary."""
    print(f"\n=== {header} ===")
    if gguf_path:
        print(f"  GGUF: {gguf_path}")
    print(
        f"  alignment={reader.alignment}, byte_order={reader.byte_order!r}, "
        f"n_tensors={len(tensors)}"
    )
    arch = _read_string_field(reader, "general.architecture") or "<missing>"
    print(f"  general.architecture: {arch!r}")

    counts: Counter = Counter()
    bytes_by: Counter = Counter()
    for t in tensors:
        counts[t.tensor_type.name] += 1
        bytes_by[t.tensor_type.name] += int(t.n_bytes)

    print(f"\n  {'codec':<10} {'#tensors':>9} {'GB raw':>9}")
    for name, c in sorted(counts.items(), key=lambda kv: -bytes_by[kv[0]]):
        print(f"  {name:<10} {c:>9} {_gb(bytes_by[name]):>9.3f}")
    print()


# ---------------------------------------------------------------------------
# bf16 cast + sharded safetensors writer
# ---------------------------------------------------------------------------


def _gguf_shape_to_hf(t) -> tuple[int, ...]:
    """GGUF stores axes in reverse vs HF/PyTorch. The dequanted flat array of
    length n_elements reshapes to `tuple(reversed(t.shape))` to recover HF
    layout. (GGUF reports shape with `d0` fastest-varying.)
    """
    return tuple(reversed([int(d) for d in t.shape]))


def _to_bf16_mx(np_fp32: np.ndarray):
    """Convert a fp32 numpy array to an mx.array in bf16. Native MLX path ---
    avoids the safetensors uint16-view trick.
    """
    import mlx.core as mx

    return mx.array(np_fp32).astype(mx.bfloat16)


class ShardedSafetensorsWriter:
    """Greedy bin-pack writer. Each tensor is added one at a time; when the
    in-flight shard exceeds `max_shard_bytes`, it's flushed to disk and a
    new shard begins. Oversized single tensors get their own shard.
    """

    def __init__(self, out_dir: Path, max_shard_bytes: int):
        self.out_dir = out_dir
        self.max_shard_bytes = max_shard_bytes
        self._cur: dict[str, object] = {}
        self._cur_bytes = 0
        self._shards: list[dict[str, str]] = []  # list of name -> shard_filename
        self._total_bytes = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, mx_arr) -> None:
        """Add a single tensor. Flushes the current shard before adding if
        needed."""
        # mx.array bytes ~ size * itemsize (bf16=2 bytes/elem).
        n_bytes = int(mx_arr.size) * 2  # bf16
        if self._cur and self._cur_bytes + n_bytes > self.max_shard_bytes:
            self._flush()
        self._cur[name] = mx_arr
        self._cur_bytes += n_bytes
        self._total_bytes += n_bytes

    def _flush(self) -> None:
        if not self._cur:
            return
        import mlx.core as mx

        idx = len(self._shards) + 1  # 1-based
        # Filename will be patched at finalize once we know N total.
        tmp_name = f".dequant-gguf-kquant.shard-{idx:05d}.safetensors"
        out_path = self.out_dir / tmp_name
        # save_safetensors realizes lazy mx.arrays internally before writing,
        # so dropping `self._cur` after this point releases the source arrays.
        # The `format=mlx` metadata tells mlx-vlm's load path that this
        # checkpoint already uses native MLX key conventions and should bypass
        # the HuggingFace-format sanitize transform (which would otherwise
        # add a spurious inner `model.` prefix and fail the load).
        mx.save_safetensors(str(out_path), self._cur, metadata={"format": "mlx"})
        self._shards.append({"path": str(out_path), "names": list(self._cur.keys())})
        self._cur = {}
        self._cur_bytes = 0
        print(f"  wrote shard {idx}: {tmp_name}  ({len(self._shards[-1]['names'])} tensors)")

    def finalize(self) -> None:
        self._flush()
        n_shards = len(self._shards)
        if n_shards == 0:
            print("WARN: no tensors to write --- empty output.")
            return
        weight_map: dict[str, str] = {}
        for i, s in enumerate(self._shards, start=1):
            new_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
            new_path = self.out_dir / new_name
            os.rename(s["path"], new_path)
            for n in s["names"]:
                weight_map[n] = new_name
        index = {
            "metadata": {"total_size": self._total_bytes},
            "weight_map": weight_map,
        }
        with open(self.out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        print(
            f"  wrote model.safetensors.index.json --- "
            f"{len(weight_map)} tensors across {n_shards} shards "
            f"({_gb(self._total_bytes):.2f} GB total)"
        )


# ---------------------------------------------------------------------------
# HF source asset copy (--hf-source)
# ---------------------------------------------------------------------------

# Files we *try* to copy from the upstream HF repo (skip-if-missing).
_HF_ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
)


def copy_hf_assets(repo_id: str, out_dir: Path) -> None:
    """Fetch config + tokenizer files from upstream HF into out_dir.
    Strips `quantization_config` from config.json (the upstream is bf16; our
    output is bf16). Never fetches model weights.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    print(f"\n=== HF source assets ({repo_id}) ===")
    fetched: list[str] = []
    for fname in _HF_ASSET_FILES:
        try:
            local = hf_hub_download(repo_id=repo_id, filename=fname)
        except EntryNotFoundError:
            continue
        except Exception as e:  # network / 404-like / gated-403
            print(f"  skip {fname}: {e}")
            continue
        dest = out_dir / fname
        if fname == "config.json":
            with open(local) as f:
                cfg = json.load(f)
            cfg.pop("quantization_config", None)
            with open(dest, "w") as f:
                json.dump(cfg, f, indent=2)
        else:
            shutil.copy(local, dest)
        fetched.append(fname)
    print(f"  fetched: {fetched}")


def copy_local_assets(source_dir: Path, out_dir: Path) -> None:
    """Copy config + tokenizer files from a local directory into out_dir
    (e.g., from an existing MLX checkpoint dir). Same file set as
    `copy_hf_assets`, same `quantization_config`-stripping post-process on
    config.json. Useful when the upstream HF repo is gated and we already
    have a local mirror.
    """
    print(f"\n=== Local source assets ({source_dir}) ===")
    if not source_dir.is_dir():
        print(f"  WARN: source dir {source_dir} not found --- skipping asset copy")
        return
    fetched: list[str] = []
    for fname in _HF_ASSET_FILES:
        src = source_dir / fname
        if not src.is_file():
            continue
        dest = out_dir / fname
        if fname == "config.json":
            with open(src) as f:
                cfg = json.load(f)
            cfg.pop("quantization_config", None)
            with open(dest, "w") as f:
                json.dump(cfg, f, indent=2)
        else:
            shutil.copy(src, dest)
        fetched.append(fname)
    print(f"  copied: {fetched}")


# ---------------------------------------------------------------------------
# Main / driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="mqt-dequant-gguf",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "gguf", help="Path to a GGUF file (e.g., unsloth/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf)"
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Bit-exact compare each quantized tensor against"
            " gguf.quants reference. Hard-fails on mismatch."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse metadata + show inventory only; no dequant, no load.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="If set, write sharded bf16 safetensors + index.json to this directory.",
    )
    ap.add_argument(
        "--hf-source",
        type=str,
        default=None,
        help=(
            "HF repo id whose config + tokenizer are copied"
            " into --out-dir (e.g., google/gemma-4-E2B-it)."
            " Gated repos (incl. all Google Gemma) need"
            " huggingface-cli login + license accepted on"
            " the repo page."
        ),
    )
    ap.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Local directory containing config + tokenizer files to copy into --out-dir. "
        "Sidesteps HF gating when an upstream MLX/HF mirror already exists locally. "
        "Mutually exclusive with --hf-source.",
    )
    ap.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override architecture detection. Values: gemma3, gemma4, qwen3, qwen3moe, llama.",
    )
    ap.add_argument(
        "--no-remap",
        action="store_true",
        help="Disable name/layout remap; preserve raw GGUF tensor names + layouts in output.",
    )
    ap.add_argument(
        "--target-prefix",
        type=str,
        default=None,
        help=(
            "REQUIRED for VLM targets. Optional name prefix"
            " that replaces the leading 'model.' in output"
            " tensor names. The loader produces HF-stock"
            " names like 'model.layers.0.self_attn.q_proj"
            ".weight'; this flag retargets them at an"
            " mlx-vlm or mlx-lm checkpoint convention. "
            "For gemma-4-26B-A4B-it (mlx-lm-style VLM):"
            " pass '--target-prefix language_model.model.'"
            " to produce 'language_model.model.layers.0.*'."
            " For gemma-4-E2B-it (mlx-vlm style): pass"
            " '--target-prefix model.language_model.'."
            " For text-only targets, leave unset."
            " lm_head.weight (when present) is also"
            " retargeted so it lives in the same namespace."
        ),
    )
    ap.add_argument(
        "--max-shard-gb",
        type=float,
        default=5.0,
        help="Max bytes per safetensors shard (default 5.0 GB, matches mlx-lm).",
    )
    ap.add_argument(
        "--limit-tensors", type=int, default=None, help="Stop after N tensors (development aid)."
    )
    args = ap.parse_args()

    from gguf import GGUFReader

    if not os.path.isfile(args.gguf):
        print(f"FATAL: GGUF not found: {args.gguf}", file=sys.stderr)
        return 2

    reader = GGUFReader(args.gguf, "r")
    if reader.byte_order != "I":
        print(
            f"FATAL: GGUF byte_order={reader.byte_order!r}, expected 'I' (intel-LE). "
            f"Big-endian / non-LE GGUFs are not supported.",
            file=sys.stderr,
        )
        return 2

    tensors = list(reader.tensors)
    if args.limit_tensors is not None:
        tensors = tensors[: args.limit_tensors]

    print_inventory(reader, tensors, header="GGUF inventory", gguf_path=args.gguf)

    arch_string = args.arch or detect_arch(reader)
    print(f"  arch (effective): {arch_string!r}")

    if args.dry_run:
        # Confirm codec set + show remap coverage (HF target name preview).
        unsupported = [
            t
            for t in tensors
            if int(t.tensor_type) not in DEQUANT_FUNCS
            and int(t.tensor_type) not in PASSTHROUGH_TYPES
        ]
        if unsupported:
            print(f"WARN: {len(unsupported)} tensors have unsupported codecs:")
            for t in unsupported[:10]:
                print(f"    {t.name!r:50} {t.tensor_type.name}")
            if len(unsupported) > 10:
                print(f"    ... (+{len(unsupported) - 10} more)")
        else:
            print(
                "OK: all tensors use supported codecs"
                " (Q4_K, Q5_K, Q6_K, Q5_1, Q8_0,"
                " F32, F16, BF16)."
            )

        # Remap preview
        if not args.no_remap:
            print("\n=== Remap preview ===")
            n_map = n_skip = n_fail = 0
            transforms: Counter = Counter()
            skip_reasons: Counter = Counter()
            fail_reasons: list[str] = []
            for t in tensors:
                d = parse_gguf_name(arch_string, t.name)
                if d.kind == RemapDecision.KIND_MAP:
                    n_map += 1
                    transforms[d.transform] += 1
                elif d.kind == RemapDecision.KIND_SKIP:
                    n_skip += 1
                    skip_reasons[d.reason] += 1
                else:
                    n_fail += 1
                    fail_reasons.append(f"{t.name!r}: {d.reason}")
            print(f"  mapped: {n_map}  (transforms: {dict(transforms)})")
            print(f"  skipped: {n_skip}  (reasons:")
            for r, c in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
                print(f"      {c} x {r}")
            print("  )")
            if fail_reasons:
                print(f"  UNMAPPED (will hard-fail without --no-remap): {n_fail}")
                for r in fail_reasons[:20]:
                    print(f"    {r}")
                if len(fail_reasons) > 20:
                    print(f"    ... (+{len(fail_reasons) - 20} more)")
                return 1
        return 0

    # Default + --validate + --out-dir path: walk every tensor, dequant,
    # remap, optionally write to safetensors.
    out_dir = Path(args.out_dir) if args.out_dir else None
    writer = None
    if out_dir is not None:
        max_bytes = int(args.max_shard_gb * 1e9)
        writer = ShardedSafetensorsWriter(out_dir, max_bytes)

    n_ok = 0
    n_validated = 0
    n_failed = 0
    n_skipped = 0
    n_emitted = 0
    saw_gemma4_moe_router = False
    saw_gemma4_per_expert_scale = False

    # Optional final prefix retargeting: leading 'model.' -> '<prefix>'.
    # 'lm_head.weight' (untied) is also retargeted so it lives under the same
    # nested namespace.
    target_prefix = args.target_prefix

    def _retarget(name: str) -> str:
        if target_prefix is None:
            return name
        if name.startswith("model."):
            return target_prefix + name[len("model.") :]
        if name == "lm_head.weight":
            return target_prefix + name
        return name

    print(
        f"\n=== Pass ({'with --validate' if args.validate else 'codec-only'}"
        f"{', writing safetensors' if writer is not None else ''}) ==="
    )

    for i, t in enumerate(tensors):
        # 1. Decide remap up front --- if the tensor is skipped/failed, don't
        #    even bother dequanting (saves work especially on .scale tensors).
        decision = (
            parse_gguf_name(arch_string, t.name)
            if not args.no_remap
            else RemapDecision(RemapDecision.KIND_MAP, hf_name=t.name, transform="passthrough")
        )
        if decision.kind == RemapDecision.KIND_FAIL:
            print(f"  [{i:>4}] FAIL: {t.name!r} --- {decision.reason}")
            n_failed += 1
            continue
        if decision.kind == RemapDecision.KIND_SKIP:
            print(f"  [{i:>4}] skip: {t.name!r} ({t.tensor_type.name}) --- {decision.reason}")
            n_skipped += 1
            continue

        # 2. Dequant.
        try:
            flat = dequantize_tensor(t)
        except Exception as e:
            print(f"  [{i:>4}] DEQUANT-FAIL: {t.name!r} --- {e}")
            n_failed += 1
            continue
        n_ok += 1

        # 3. Optional bit-exact validation against gguf.quants.
        if args.validate:
            ok, detail = validate_against_reference(t)
            n_validated += 1
            if not ok:
                print(f"  [{i:>4}] VALIDATE-FAIL: {t.name!r} --- {detail}")
                n_failed += 1
                continue

        # 4. Reshape to HF axis order, apply per-tensor transform, cast bf16.
        hf_shape = _gguf_shape_to_hf(t)
        try:
            np_arr = flat.reshape(hf_shape)
        except Exception as e:
            print(f"  [{i:>4}] RESHAPE-FAIL: {t.name!r} -> {hf_shape}: {e}")
            n_failed += 1
            continue

        emitted: list[tuple[str, np.ndarray]] = []
        if decision.transform == "passthrough":
            emitted.append((_retarget(decision.hf_name), np_arr))
        elif decision.transform == "qk_permute":
            # LLAMA only --- never fires on gemma-4 / qwen3 validation targets.
            n_head, n_kv_head = _read_attn_head_counts(reader)
            is_q = decision.hf_name.endswith("q_proj.weight")
            permuted = apply_qk_permute(
                np_arr,
                n_head=n_head,
                n_head_kv=None if is_q else n_kv_head,
            )
            emitted.append((_retarget(decision.hf_name), permuted))
        elif decision.transform == "moe_split_gate_up":
            gate, up = split_fused_gate_up_exps(np_arr)
            # decision.hf_name ends in `<...>.gate_up_proj.weight` --- strip the
            # trailing `gate_up_proj.weight` to get the experts container path,
            # then append the split projection names. Works for both HF stock
            # `mlp.experts` and gemma-4-MoE `experts.switch_glu` namespaces.
            base_prefix = decision.hf_name[: -len("gate_up_proj.weight")].rstrip(".")
            emitted.append((_retarget(f"{base_prefix}.gate_proj.weight"), gate))
            emitted.append((_retarget(f"{base_prefix}.up_proj.weight"), up))
        elif decision.transform == "conv1d_unsqueeze":
            emitted.append((_retarget(decision.hf_name), np_arr[..., None]))
        elif decision.transform == "gate_1d_unsqueeze":
            emitted.append(
                (
                    _retarget(decision.hf_name),
                    np_arr.reshape(1, -1) if np_arr.ndim == 1 else np_arr,
                )
            )
        else:
            print(f"  [{i:>4}] UNKNOWN-TRANSFORM: {decision.transform}")
            n_failed += 1
            continue

        # 5. Write (or just log) each emitted (name, array).
        for emit_name, emit_np in emitted:
            if writer is not None:
                mx_arr = _to_bf16_mx(emit_np)
                writer.add(emit_name, mx_arr)
            n_emitted += 1
            if ".router.proj.weight" in emit_name:
                saw_gemma4_moe_router = True
            if ".router.per_expert_scale" in emit_name:
                saw_gemma4_per_expert_scale = True
            if i < 8 or i % 100 == 0 or args.validate:
                # Spam-protected per-tensor log: first few + every 100th + always under --validate.
                size_gb = emit_np.size * 2 / 1e9
                print(
                    f"  [{i:>4}] {t.tensor_type.name:<6} {t.name!r:55} -> {emit_name!r:65} "
                    f"shape={list(emit_np.shape)} GB={size_gb:.3f}"
                    + ("  [bit-exact]" if args.validate else "")
                )

        del flat, np_arr, emitted

    if writer is not None:
        writer.finalize()
        if args.hf_source and args.source_dir:
            print("WARN: --hf-source and --source-dir both set; preferring --source-dir.")
        if args.source_dir:
            copy_local_assets(Path(args.source_dir), out_dir)
        elif args.hf_source:
            copy_hf_assets(args.hf_source, out_dir)

    if saw_gemma4_moe_router and not saw_gemma4_per_expert_scale and writer is not None:
        print(
            "\nWARN: gemma-4-MoE router detected but no `router.per_expert_scale` "
            "tensors were emitted. Unsloth UD GGUFs ship these as "
            "`blk.{N}.ffn_down_exps.scale`; if your source GGUF is from a different "
            "tool, the trained per-expert routing scales may be unrecoverable. "
            "Load with strict=False or initialize these tensors downstream."
        )

    print("\n=== Summary ===")
    print(f"  dequanted: {n_ok}/{len(tensors) - n_skipped} ({n_skipped} skipped)")
    print(f"  emitted (post-remap): {n_emitted}")
    if args.validate:
        print(
            f"  validated: {n_validated} "
            f"(bit-exact: {n_validated - n_failed}, "
            f"mismatched: {n_failed})"
        )
    if n_failed:
        return 1
    return 0


def _read_attn_head_counts(reader) -> tuple[int, int]:
    """Read n_head and n_kv_head from GGUF metadata (used only for LLAMA Q/K
    permute). Returns (n_head, n_kv_head); falls back to (n_head, n_head) if
    n_kv_head is absent.
    """
    arch = _read_string_field(reader, "general.architecture") or "?"
    f_head = reader.fields.get(f"{arch}.attention.head_count")
    f_kvhd = reader.fields.get(f"{arch}.attention.head_count_kv")
    if f_head is None:
        raise ValueError(f"{arch}.attention.head_count missing --- can't apply Q/K permute")
    n_head = int(f_head.parts[f_head.data[0]][0])
    n_kv_head = int(f_kvhd.parts[f_kvhd.data[0]][0]) if f_kvhd is not None else n_head
    return n_head, n_kv_head


if __name__ == "__main__":
    sys.exit(main())
