"""Load a GGUF file with all 10 K-quant types as raw uint8 wire bytes.

Unlike `mx.load("model.gguf")` — which converts Q4_0/Q4_1/Q8_0 to MLX's
legacy affine format and routes Q5_0/Q5_1/Q2_K..Q6_K through the new
kquant raw path — this loader treats *every* quantized type uniformly,
preserving on-disk wire bytes for direct use with MLX's kquant kernels.

Two loading paths for the same model file:

    # Path A — mx.load (mixed affine + kquant)
    arrays, metadata = mx.load("model.gguf", return_metadata=True)
    # Q4_0/Q4_1/Q8_0 → uint32 + fp16 scales/biases (NAX matmul)
    # Q5_0/Q5_1/Q2_K..Q6_K → uint8 raw (kquant matmul)

    # Path B — load_gguf_kquant (all kquant)
    from mlx_quant_tools.load_gguf_kquant import load_gguf_kquant
    arrays, kquant_meta = load_gguf_kquant("model.gguf")
    # All 10 quantized types → uint8 raw (kquant matmul)

Mixing kquant and affine in one model is safe: nn.QuantizedLinear.mode is
per-layer.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader

# Codec geometry. Source of truth: mlx/primitives.cpp kquant_codec_by_name.
# Each entry: (weights_per_block, bytes_per_block, kquant_type_str).
KQUANT_CODECS: dict[GGMLQuantizationType, tuple[int, int, str]] = {
    GGMLQuantizationType.Q4_0: (32, 18, "q4_0"),
    GGMLQuantizationType.Q4_1: (32, 20, "q4_1"),
    GGMLQuantizationType.Q5_0: (32, 22, "q5_0"),
    GGMLQuantizationType.Q5_1: (32, 24, "q5_1"),
    GGMLQuantizationType.Q8_0: (32, 34, "q8_0"),
    GGMLQuantizationType.Q2_K: (256, 84, "q2_k"),
    GGMLQuantizationType.Q3_K: (256, 110, "q3_k"),
    GGMLQuantizationType.Q4_K: (256, 144, "q4_k"),
    GGMLQuantizationType.Q5_K: (256, 176, "q5_k"),
    GGMLQuantizationType.Q6_K: (256, 210, "q6_k"),
}

# Non-quantized types that pass through with their native dtype.
DIRECT_DTYPES: dict[GGMLQuantizationType, mx.Dtype] = {
    GGMLQuantizationType.F32: mx.float32,
    GGMLQuantizationType.F16: mx.float16,
    GGMLQuantizationType.BF16: mx.bfloat16,
}


def _strip_weight_suffix(name: str) -> str:
    suffix = ".weight"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def load_gguf_kquant(
    path: str,
) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Load a GGUF file, treating all 10 K-quant types as raw uint8.

    Returns:
        arrays: tensor name -> mx.array. Quantized tensors are uint8 with
            shape [..., bytes_per_row]. Each quantized tensor also gets a
            vestigial "<prefix>.scales" placeholder (1-byte uint8, zero) so
            nn.QuantizedLinear's scales-attribute slot is satisfied.
            F32/F16/BF16 tensors keep their native dtype.
        kquant_meta: tensor name -> kquant_type string (e.g.,
            "blk.0.attn_q.weight" -> "q4_k"). Only quantized tensors appear.

    Raises:
        FileNotFoundError: path doesn't exist.
        ValueError: tensor geometry doesn't match the codec table.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GGUF file not found: {path}")

    reader = GGUFReader(path, "r")
    arrays: dict[str, mx.array] = {}
    kquant_meta: dict[str, str] = {}

    for tensor in reader.tensors:
        name = tensor.name
        ttype = tensor.tensor_type

        if ttype in KQUANT_CODECS:
            wpb, bpb, kquant_type = KQUANT_CODECS[ttype]

            # tensor.shape is GGML order (innermost-first); reverse to MLX.
            logical_shape = [int(d) for d in tensor.shape][::-1]
            last_dim = logical_shape[-1]
            if last_dim % wpb != 0:
                raise ValueError(
                    f"Tensor '{name}' last dim {last_dim} is not divisible by "
                    f"weights_per_block {wpb} for codec {kquant_type}"
                )
            bytes_per_row = (last_dim // wpb) * bpb
            packed_shape = list(logical_shape)
            packed_shape[-1] = bytes_per_row

            # tensor.data is already a numpy uint8 array with the wire bytes,
            # shaped (rows, bytes_per_row) for 2D tensors.
            raw = np.ascontiguousarray(tensor.data, dtype=np.uint8)
            expected = 1
            for d in packed_shape:
                expected *= d
            if raw.size != expected:
                raise ValueError(
                    f"Tensor '{name}' ({kquant_type}) byte count mismatch: "
                    f"got {raw.size}, expected {expected}"
                )
            arrays[name] = mx.array(raw.reshape(packed_shape))
            arrays[_strip_weight_suffix(name) + ".scales"] = mx.zeros((1,), dtype=mx.uint8)
            kquant_meta[name] = kquant_type

        elif ttype in DIRECT_DTYPES:
            mlx_dtype = DIRECT_DTYPES[ttype]
            logical_shape = [int(d) for d in tensor.shape][::-1]
            data = np.ascontiguousarray(tensor.data)
            if mlx_dtype == mx.bfloat16:
                # numpy has no native bf16 — gguf returns uint8 raw bytes;
                # view as uint16 then cast to mx.bfloat16 via view-conversion.
                u16 = np.frombuffer(data.tobytes(), dtype=np.uint16).reshape(logical_shape)
                arrays[name] = mx.array(u16).view(mx.bfloat16)
            else:
                arrays[name] = mx.array(data.reshape(logical_shape))

        else:
            print(
                f"WARNING: skipping tensor '{name}' with unsupported "
                f"type {ttype.name} (id={int(ttype)})",
                file=sys.stderr,
            )

    return arrays, kquant_meta
