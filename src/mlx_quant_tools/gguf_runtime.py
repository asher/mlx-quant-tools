"""GGUF K-quant model loading runtime.

Provides the kquant-aware module classes (KQuantEmbedding, KQuantSwitchLinear,
make_kquant_linear), GGUF→HF tensor remap + layout transforms, model
construction, and the end-to-end `load_kquant_model()` entry point.

Factored from the CLI layer so that ``mqt-serve-gguf``, ``mqt-load-kquant``,
and ``mqt-score-kld`` can import the library layer directly.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
from gguf import GGUFReader
from mlx.utils import tree_flatten, tree_map_with_path
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.utils import _get_classes

from mlx_quant_tools.gguf_name_remap import RemapDecision, parse_gguf_name

# Codec geometry — (group_size, bits, bytes_per_block, weights_per_block).
# Source of truth: mlx::core::kquant_codec_by_name in mlx/primitives.cpp.
CODEC_GEOMETRY: dict[str, tuple[int, int, int, int]] = {
    "q8_0": (32, 8, 34, 32),
    "q4_0": (32, 4, 18, 32),
    "q4_1": (32, 4, 20, 32),
    "q5_0": (32, 5, 22, 32),
    "q5_1": (32, 5, 24, 32),
    "q4_k": (256, 4, 144, 256),
    "q5_k": (256, 5, 176, 256),
    "q6_k": (256, 6, 210, 256),
    "q3_k": (256, 3, 110, 256),
    "q2_k": (256, 2, 84, 256),
}


# ---------------------------------------------------------------------------
# kquant-aware module subclasses
# ---------------------------------------------------------------------------


class KQuantEmbedding(nn.Module):
    """Embedding backed by GGUF kquant wire bytes.

    `__call__` gathers per-token wire-byte rows then dequantizes — small
    output sizes only, so it sidesteps the dispatch-overflow bug in
    `mx.dequantize(mode="kquant")` for tensors > INT_MAX elements.

    `as_linear` runs a full kquant matmul for tied lm_head projection
    (gemma-4 / qwen3 etc. tie embed_tokens to lm_head).
    """

    def __init__(self, num_embeddings: int, dims: int, codec: str):
        super().__init__()
        gs, bits, bpb, wpb = CODEC_GEOMETRY[codec]
        self.group_size = gs
        self.bits = bits
        self.kquant_type = codec
        self.num_embeddings = num_embeddings
        self.dims = dims
        bytes_per_row = (dims // wpb) * bpb
        self.weight = mx.zeros((num_embeddings, bytes_per_row), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        self.freeze()

    def __call__(self, x):
        gathered = self["weight"][x]  # [*, bytes_per_row]
        flat = gathered.reshape(-1, gathered.shape[-1])
        deq = mx.dequantize(
            flat,
            self["scales"],
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
        )
        return deq.reshape(*gathered.shape[:-1], self.dims)

    def as_linear(self, x):
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="kquant",
            kquant_type=self.kquant_type,
        )

    def _extra_repr(self):
        return f"{self.num_embeddings}, {self.dims}, kquant_type={self.kquant_type}"


def make_kquant_linear(in_dims: int, out_dims: int, bias: bool, codec: str) -> nn.QuantizedLinear:
    """Construct a kquant-mode QuantizedLinear without invoking the broken
    `mx.quantize(mode="kquant")` path inside `QuantizedLinear.__init__`.
    """
    gs, bits, bpb, wpb = CODEC_GEOMETRY[codec]
    layer = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(layer)
    layer.group_size = gs
    layer.bits = bits
    layer.mode = "kquant"
    layer.kquant_type = codec
    bytes_per_row = (in_dims // wpb) * bpb
    layer.weight = mx.zeros((out_dims, bytes_per_row), dtype=mx.uint8)
    layer.scales = mx.zeros((1,), dtype=mx.uint8)
    layer.biases = None
    if bias:
        layer.bias = mx.zeros((out_dims,))
    layer.freeze()
    return layer


class KQuantSwitchLinear(nn.Module):
    """MoE expert linear layer backed by GGUF kquant wire bytes.

    Counterpart to mlx_lm's QuantizedSwitchLinear (mode="affine"), but for
    kquant codecs. Stores `weight` as uint8 wire bytes shaped
    (n_experts, output_dims, bytes_per_row) and dispatches via
    `mx.gather_qmm(..., transpose=True, mode="kquant", kquant_type=...)`.
    """

    def __init__(
        self, num_experts: int, output_dims: int, input_dims: int, bias: bool, codec: str
    ):
        super().__init__()
        gs, bits, bpb, wpb = CODEC_GEOMETRY[codec]
        self.group_size = gs
        self.bits = bits
        self.mode = "kquant"
        self.kquant_type = codec
        bytes_per_row = (input_dims // wpb) * bpb
        self.weight = mx.zeros((num_experts, output_dims, bytes_per_row), dtype=mx.uint8)
        self.scales = mx.zeros((1,), dtype=mx.uint8)
        self.biases = None
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            kquant_type=self.kquant_type,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def _extra_repr(self):
        n, m, b = self.weight.shape
        return (
            f"num_experts={n}, output_dims={m}, bytes_per_row={b}, kquant_type={self.kquant_type}"
        )


# ---------------------------------------------------------------------------
# mx.load metadata parsing
# ---------------------------------------------------------------------------


def _decode_meta_string(v) -> str:
    if isinstance(v, mx.array):
        return bytes(v).decode("utf-8")
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8")
    return str(v)


def parse_kquant_metadata(meta_entry) -> dict[str, str]:
    """Parse `metadata['__kquant_types__']` into `{tensor_name: codec}`.

    Entries are formatted "<name>:<codec>". Tensor names contain dots but
    never colons, so `rsplit(":", 1)` is safe.
    """
    if meta_entry is None:
        return {}
    if isinstance(meta_entry, list):
        items = [_decode_meta_string(e) for e in meta_entry]
    else:
        items = [_decode_meta_string(meta_entry)]

    out: dict[str, str] = {}
    for s in items:
        if ":" not in s:
            raise ValueError(f"malformed __kquant_types__ entry: {s!r}")
        name, codec = s.rsplit(":", 1)
        out[name] = codec
    return out


def read_arch(metadata: dict) -> str:
    val = metadata.get("general.architecture")
    if val is None:
        raise ValueError("GGUF metadata missing 'general.architecture'")
    return _decode_meta_string(val)


def collapse_legacy_affine(
    arrays: dict[str, mx.array],
    kquant_meta: dict[str, str],
) -> int:
    """Defensive collapse for pre-kquant-fork MLX builds where Q4_0/Q4_1/Q8_0
    came through `gguf_load_quantized` (uint32 weights + fp16 scales + fp16
    biases). On the current MLX fork these codecs route through
    `gguf_load_kquant` instead, so this is a no-op — kept as a guard so the
    script doesn't silently mis-load if run against an older libmlx.

    Mutates `arrays`: each `<prefix>.weight + .scales + .biases` triple
    becomes a single bf16 `<prefix>.weight`. Returns the number collapsed.
    """
    n = 0
    suffix = ".weight"
    for name in list(arrays):
        if not name.endswith(suffix):
            continue
        if name in kquant_meta:
            continue
        prefix = name[: -len(suffix)]
        scales_key = f"{prefix}.scales"
        biases_key = f"{prefix}.biases"
        if biases_key not in arrays or scales_key not in arrays:
            continue
        w = arrays[name]
        s = arrays[scales_key]
        b = arrays[biases_key]
        if w.dtype != mx.uint32 or s.dtype != mx.float16:
            continue
        bits = w.shape[-1] // s.shape[-1] if s.shape[-1] else 0
        if bits not in (4, 8):
            print(
                f"WARNING: skipping affine collapse for {name!r}: "
                f"inferred bits={bits} (shape ratio "
                f"{w.shape[-1]}/{s.shape[-1]})",
                file=sys.stderr,
            )
            continue
        dense = mx.dequantize(w, s, b, group_size=32, bits=bits, mode="affine")
        arrays[name] = dense.astype(mx.bfloat16)
        del arrays[scales_key]
        del arrays[biases_key]
        n += 1
    return n


def _find_split_shards(gguf_path: str) -> list[str]:
    """Detect split GGUF and return all shard paths in order."""
    import re as _re

    m = _re.search(r"-(\d{5})-of-(\d{5})\.gguf$", gguf_path)
    if not m:
        return [gguf_path]
    total = int(m.group(2))
    prefix = gguf_path[: m.start()]
    suffix_fmt = "-{:05d}-of-{:05d}.gguf"
    shards = []
    for i in range(1, total + 1):
        p = prefix + suffix_fmt.format(i, total)
        if os.path.isfile(p):
            shards.append(p)
        else:
            print(f"WARNING: split shard {p} not found", file=sys.stderr)
    return shards if shards else [gguf_path]


def load_gguf_via_mx(gguf_path: str) -> tuple[dict[str, mx.array], dict[str, str], str | None]:
    """Load GGUF tensors via mx.load.

    Returns (arrays, kquant_meta, arch_from_metadata). Handles split GGUFs.
    """
    shards = _find_split_shards(gguf_path)
    arrays: dict[str, mx.array] = {}
    metadata: dict = {}
    for shard in shards:
        shard_arrays, shard_meta = mx.load(shard, return_metadata=True)
        arrays.update(shard_arrays)
        kt_key = "__kquant_types__"
        if kt_key in shard_meta:
            existing = metadata.get(kt_key)
            if existing is None:
                metadata[kt_key] = list(shard_meta[kt_key])
            else:
                metadata[kt_key] = list(existing) + list(shard_meta[kt_key])
            shard_meta = {k: v for k, v in shard_meta.items() if k != kt_key}
        metadata.update(shard_meta)
    if len(shards) > 1:
        print(f"[mx.load] loaded {len(shards)} shards, {len(arrays)} total tensors")

    kquant_meta = parse_kquant_metadata(metadata.get("__kquant_types__"))
    n_collapsed = collapse_legacy_affine(arrays, kquant_meta)
    if n_collapsed:
        print(
            f"[mx.load] collapsed {n_collapsed} legacy-affine (Q4_0/Q4_1/Q8_0) "
            f"tensors to bf16 dense"
        )
    if not kquant_meta:
        print(
            "WARNING: no __kquant_types__ entries — is this actually a "
            "K-quant GGUF? mx.load handles Q4_0/Q4_1/Q8_0 via the legacy "
            "affine path with no annotation.",
            file=sys.stderr,
        )
    arch_val = metadata.get("general.architecture")
    arch = _decode_meta_string(arch_val) if arch_val is not None else None
    return arrays, kquant_meta, arch


# ---------------------------------------------------------------------------
# Tensor-name remap + layout transforms (kquant wire bytes)
# ---------------------------------------------------------------------------


def _split_fused_gate_up_kquant(w: mx.array) -> tuple[mx.array, mx.array]:
    """Split a fused MoE gate-up wire-byte tensor along the byte axis midpoint."""
    if w.ndim != 3:
        raise ValueError(f"expected 3D fused expert tensor, got shape {w.shape}")
    half = w.shape[-2] // 2
    return w[..., :half, :], w[..., half:, :]


def _retarget(name: str, target_prefix: str) -> str:
    if not target_prefix:
        return name
    return f"{target_prefix}.{name}" if not name.startswith(target_prefix) else name


def _qk_permute_wire(w: mx.array, n_head: int) -> mx.array:
    """Invert llama.cpp's `LlamaModel.permute` on Q/K weight rows."""
    n_out = w.shape[0]
    head_dim = n_out // n_head
    return (
        w.reshape(n_head, head_dim // 2, 2, *w.shape[1:])
        .swapaxes(1, 2)
        .reshape(n_out, *w.shape[1:])
    )


def _strip_weight(name: str) -> str:
    return name[: -len(".weight")] if name.endswith(".weight") else name


def remap_arrays(
    arrays: dict[str, mx.array],
    kquant_meta: dict[str, str],
    arch: str,
    *,
    no_remap: bool = False,
    target_prefix: str = "",
    fail_on_unknown: bool = False,
    n_head: int | None = None,
    n_head_kv: int | None = None,
) -> tuple[dict[str, mx.array], dict[str, str], dict[str, int]]:
    """Apply name remap + layout transforms to GGUF arrays.

    Returns `(hf_weights, hf_kquant_meta, stats)` where `hf_kquant_meta` maps
    the post-remap tensor name to its codec string.
    """
    hf_weights: dict[str, mx.array] = {}
    hf_kquant_meta: dict[str, str] = {}
    stats = {
        "mapped": 0,
        "skipped": 0,
        "split": 0,
        "failed": 0,
        "passthrough": 0,
        "qk_permute_applied": 0,
        "qk_permute_skipped": 0,
        "conv1d_unsqueeze": 0,
    }

    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        codec = kquant_meta.get(name)

        if no_remap:
            hf_name = name
            transform = "passthrough"
        else:
            dec = parse_gguf_name(arch, name)
            if dec.kind == RemapDecision.KIND_SKIP:
                stats["skipped"] += 1
                continue
            if dec.kind == RemapDecision.KIND_FAIL:
                if fail_on_unknown:
                    raise RuntimeError(f"unmapped tensor {name!r}: {dec.reason}")
                print(f"WARNING: skipping unmapped tensor {name!r}: {dec.reason}", file=sys.stderr)
                stats["failed"] += 1
                continue
            hf_name = _retarget(dec.hf_name, target_prefix)
            transform = dec.transform

        if transform == "passthrough":
            hf_weights[hf_name] = arr
            if codec is not None:
                hf_weights[_strip_weight(hf_name) + ".scales"] = arrays[
                    _strip_weight(name) + ".scales"
                ]
                hf_kquant_meta[hf_name] = codec
            stats["passthrough"] += 1
            stats["mapped"] += 1

        elif transform == "moe_split_gate_up":
            base = hf_name[: -len("gate_up_proj.weight")].rstrip(".")
            gate_name = f"{base}.gate_proj.weight"
            up_name = f"{base}.up_proj.weight"
            gate, up = _split_fused_gate_up_kquant(arr)
            hf_weights[gate_name] = gate
            hf_weights[up_name] = up
            if codec is not None:
                hf_weights[_strip_weight(gate_name) + ".scales"] = mx.zeros((1,), dtype=mx.uint8)
                hf_weights[_strip_weight(up_name) + ".scales"] = mx.zeros((1,), dtype=mx.uint8)
                hf_kquant_meta[gate_name] = codec
                hf_kquant_meta[up_name] = codec
            stats["split"] += 1
            stats["mapped"] += 2

        elif transform == "qk_permute":
            is_k = hf_name.endswith("k_proj.weight")
            n_heads_for = n_head_kv if (is_k and n_head_kv is not None) else n_head
            if n_heads_for is None:
                print(
                    f"WARNING: qk_permute requested for {hf_name!r} but "
                    f"n_head/n_head_kv not provided; loading without "
                    f"permute (attention will be wrong).",
                    file=sys.stderr,
                )
                hf_weights[hf_name] = arr
                stats["qk_permute_skipped"] += 1
            else:
                hf_weights[hf_name] = _qk_permute_wire(arr, n_heads_for)
                stats["qk_permute_applied"] += 1
            if codec is not None:
                hf_weights[_strip_weight(hf_name) + ".scales"] = arrays[
                    _strip_weight(name) + ".scales"
                ]
                hf_kquant_meta[hf_name] = codec
            stats["mapped"] += 1

        elif transform == "conv1d_unsqueeze":
            hf_weights[hf_name] = arr[..., None]
            stats["conv1d_unsqueeze"] += 1
            stats["mapped"] += 1

        elif transform == "ssm_a_to_a_log":
            out = mx.log(-arr.astype(mx.float32))
            hf_weights[hf_name] = out.reshape(-1) if out.ndim > 1 else out
            stats["mapped"] += 1

        elif transform == "flatten":
            hf_weights[hf_name] = arr.reshape(-1)
            stats["mapped"] += 1

        elif transform == "gate_1d_unsqueeze":
            hf_weights[hf_name] = arr.reshape(1, -1) if arr.ndim == 1 else arr
            stats["mapped"] += 1

        else:
            raise RuntimeError(f"unknown transform {transform!r} for {name!r}")

    return hf_weights, hf_kquant_meta, stats


# ---------------------------------------------------------------------------
# Model construction (bypassing nn.quantize)
# ---------------------------------------------------------------------------


def build_model(config_dict: dict):
    """Instantiate the mlx_lm model class without running nn.quantize()."""
    config = dict(config_dict)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    _UNWRAP_TO_TEXT = {"qwen3_5_moe", "qwen3_5_moe_text"}
    mt = config.get("model_type", "")
    if mt in _UNWRAP_TO_TEXT:
        import importlib

        mod = importlib.import_module("mlx_lm.models.qwen3_5")
        TextModel = mod.TextModel
        TextModelArgs = mod.TextModelArgs
        model_args = TextModelArgs.from_dict(config)
        model = TextModel(model_args)
        print(f"[build] unwrap {mt} → qwen3_5.TextModel (avoid language_model. prefix)")
        return model, config
    Model, ModelArgs = _get_classes(config)
    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)
    return model, config


def install_kquant_modules(model: nn.Module, hf_kquant_meta: dict[str, str]) -> int:
    """Walk leaf modules; replace each Linear/Embedding whose `<path>.weight`
    is in `hf_kquant_meta` with a kquant-mode equivalent. Returns the count
    of replacements made.
    """
    n_replaced = 0

    def _replace(path: str, module):
        nonlocal n_replaced
        weight_key = f"{path}.weight"
        codec = hf_kquant_meta.get(weight_key)
        if codec is None:
            return module
        if isinstance(module, nn.Linear):
            out_dims, in_dims = module.weight.shape
            bias = "bias" in module
            n_replaced += 1
            return make_kquant_linear(in_dims, out_dims, bias, codec)
        if isinstance(module, nn.Embedding):
            num_emb, dims = module.weight.shape
            n_replaced += 1
            return KQuantEmbedding(num_emb, dims, codec)
        if isinstance(module, SwitchLinear):
            n_experts, out_dims, in_dims = module.weight.shape
            bias = "bias" in module
            n_replaced += 1
            return KQuantSwitchLinear(n_experts, out_dims, in_dims, bias, codec)
        return module

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_replace, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return n_replaced


# ---------------------------------------------------------------------------
# GGUF V-head tiling fixup
# ---------------------------------------------------------------------------


def _needs_tiled_v_patch(config: dict) -> bool:
    k = config.get("linear_num_key_heads", 0)
    v = config.get("linear_num_value_heads", 0)
    return k > 0 and v > 0 and k != v


def _patch_gated_delta_tiled_v():
    """Monkey-patch gated_delta to use tiled V-head K mapping.

    GGUF stores V heads in tiled order for ggml broadcast:
        grouped (HF): [G0_v0 G0_v1 G0_v2 G1_v0 G1_v1 G1_v2 ...]
        tiled (GGUF):  [G0_v0 G1_v0 ... GN_v0 G0_v1 G1_v1 ... GN_v1 ...]
    """
    from mlx_lm.models import gated_delta as gd

    def _make_tiled_kernel(has_mask=False, vectorized=False):
        if not mx.metal.is_available():
            return None
        mask_source = "mask[b_idx * T + t]" if has_mask else "true"
        if vectorized:
            g_comment = "// g: [B, T, Hv, Dk]"
            g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
            g_access = "g_[s_idx]"
            g_advance = "g_ += Hv * Dk;"
        else:
            g_comment = "// g: [B, T, Hv]"
            g_setup = "auto g_ = g + b_idx * T * Hv;"
            g_access = "g_[hv_idx]"
            g_advance = "g_ += Hv;"

        source = f"""
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx % Hk;
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }}

            {g_comment}
            {g_setup}
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {{
              if ({mask_source}) {{
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * {g_access};
                  kv_mem += state[i] * k_[s_idx];
                }}
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {{
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }}
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {{
                  y[dv_idx] = static_cast<InT>(out);
                }}
              }} else {{
                y[dv_idx] = static_cast<InT>(0);
              }}
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              {g_advance}
              beta_ += Hv;
            }}
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }}
        """
        inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
        if has_mask:
            inputs.append("mask")
        suffix = ""
        if vectorized:
            suffix += "_vec"
        if has_mask:
            suffix += "_mask"
        return mx.fast.metal_kernel(
            name=f"gated_delta_step_tiled{suffix}",
            input_names=inputs,
            output_names=["y", "state_out"],
            source=source,
        )

    gd._gated_delta_kernel = _make_tiled_kernel(False, False)
    gd._gated_delta_kernel_masked = _make_tiled_kernel(True, False)
    gd._gated_delta_kernel_vec = _make_tiled_kernel(False, True)
    gd._gated_delta_kernel_vec_masked = _make_tiled_kernel(True, True)

    _orig_step = gd._gated_delta_step_ops

    def _tiled_gated_delta_ops(q, k, v, g, beta, state=None, mask=None):
        B, T, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        if state is None:
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if (r := Hv // Hk) > 1:
            q = mx.tile(q, [1, 1, r, 1])
            k = mx.tile(k, [1, 1, r, 1])
        ys = []
        for t in range(T):
            y, state = _orig_step(
                q[:, t],
                k[:, t],
                v[:, t],
                g[:, t],
                beta[:, t],
                state,
                None if mask is None else mask[:, t],
            )
            ys.append(y)
        return mx.stack(ys, axis=1), state

    gd.gated_delta_ops = _tiled_gated_delta_ops
    print("[patch] gated_delta: K→V head mapping set to tiled (GGUF layout)")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_inventory(
    arch: str, kquant_meta: dict[str, str], hf_kquant_meta: dict[str, str], stats: dict[str, int]
) -> None:
    print(f"=== run-gguf-kquant: arch={arch!r} ===")
    print(f"  GGUF tensors with kquant codec : {len(kquant_meta)}")
    print(f"  HF-named kquant tensors        : {len(hf_kquant_meta)}")
    print(f"  remap stats: {stats}")
    by_codec = Counter(hf_kquant_meta.values())
    print("  kquant codec histogram (post-remap):")
    for codec, n in sorted(by_codec.items(), key=lambda x: -x[1]):
        print(f"    {codec:<5} {n}")


# ---------------------------------------------------------------------------
# End-to-end model loader
# ---------------------------------------------------------------------------


def load_kquant_model(
    gguf_path: str,
    *,
    arch: str | None = None,
    target_prefix: str = "",
    no_remap: bool = False,
    fail_on_unknown: bool = False,
    print_inventory_table: bool = True,
):
    """Load a GGUF K-quant file into an mlx_lm model with kquant modules.

    Returns `(model, config, tokenizer)`.
    """
    t0 = time.perf_counter()
    arrays, kquant_meta, arch_meta = load_gguf_via_mx(gguf_path)
    print(
        f"[mx.load] {len(arrays)} arrays, {len(kquant_meta)} kquant "
        f"({time.perf_counter() - t0:.2f}s)"
    )
    arch = arch or arch_meta
    if arch is None:
        raise ValueError("could not determine arch from GGUF metadata; pass --arch")
    print(f"[arch] {arch}")

    reader_for_meta = GGUFReader(gguf_path, "r")

    def _read_first_int(rdr, key: str) -> int | None:
        f = rdr.fields.get(key)
        if f is None:
            return None
        return int(f.parts[f.data[0]][0])

    def _read_first_nonzero_int(rdr, key: str) -> int | None:
        f = rdr.fields.get(key)
        if f is None:
            return None
        for i in f.data:
            v = int(f.parts[i][0])
            if v > 0:
                return v
        return int(f.parts[f.data[0]][0])

    n_head = _read_first_int(reader_for_meta, f"{arch}.attention.head_count")
    n_head_kv = _read_first_nonzero_int(reader_for_meta, f"{arch}.attention.head_count_kv")

    hf_weights, hf_kquant_meta, stats = remap_arrays(
        arrays,
        kquant_meta,
        arch,
        no_remap=no_remap,
        target_prefix=target_prefix,
        fail_on_unknown=fail_on_unknown,
        n_head=n_head,
        n_head_kv=n_head_kv,
    )

    if print_inventory_table:
        print_inventory(arch, kquant_meta, hf_kquant_meta, stats)

    from mlx_quant_tools.gguf_config_synth import synthesize_config

    config_dict = synthesize_config(GGUFReader(gguf_path, "r"))
    model, config = build_model(config_dict)

    if _needs_tiled_v_patch(config):
        _patch_gated_delta_tiled_v()

    if hasattr(model, "sanitize"):
        hf_weights = model.sanitize(hf_weights)
        new_meta: dict[str, str] = {}
        unmatched_meta = set(hf_kquant_meta)
        for new_k in hf_weights:
            if not new_k.endswith(".weight"):
                continue
            for old_k in list(unmatched_meta):
                if new_k == old_k or new_k.endswith("." + old_k):
                    new_meta[new_k] = hf_kquant_meta[old_k]
                    unmatched_meta.discard(old_k)
                    break
        hf_kquant_meta = new_meta

    n_replaced = install_kquant_modules(model, hf_kquant_meta)
    print(f"[install] replaced {n_replaced} leaves with kquant modules")

    model.eval()
    model_params = {p for p, _ in tree_flatten(model.parameters())}
    loadable = {k: v for k, v in hf_weights.items() if k in model_params}
    redundant = sorted(set(hf_weights.keys()) - set(loadable.keys()))
    if redundant:
        print(
            f"[load_weights] dropping {len(redundant)} redundant tensors "
            f"(no model slot): {redundant[:3]}..."
        )

    n_cast = 0
    for k in list(loadable):
        if loadable[k].dtype == mx.float32 and k not in hf_kquant_meta:
            loadable[k] = loadable[k].astype(mx.bfloat16)
            n_cast += 1
    if n_cast:
        print(f"[dtype] cast {n_cast} F32 params (norms etc.) to bf16")

    model.load_weights(list(loadable.items()), strict=False)
    print(f"[load_weights] loaded {len(loadable)} / {len(model_params)} model parameters")

    missing = sorted(model_params - set(loadable.keys()))
    if missing:
        print(
            f"WARNING: {len(missing)} model params not loaded: {missing[:5]}...", file=sys.stderr
        )

    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from mlx_quant_tools.gguf_tokenizer import load_tokenizer_from_gguf

    raw_tokenizer = load_tokenizer_from_gguf(GGUFReader(gguf_path, "r"), arch)
    eos_ids = getattr(raw_tokenizer, "_gguf_eos_token_ids", None)
    tokenizer = TokenizerWrapper(raw_tokenizer, eos_token_ids=eos_ids)

    return model, config, tokenizer
