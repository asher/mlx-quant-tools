"""Convert a GGUF K-quant file into an MLX kquant safetensors checkpoint.

Preserves K-quant wire bytes (no dequant + re-encode round-trip) so the
output is bit-equivalent to the GGUF for any tensor whose codec the MLX
kquant decode path supports (Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K). The
result is loadable via `load-mlx-kquant.py`.

Config and tokenizer are synthesized from GGUF metadata via
`gguf_config_synth.synthesize_config` and `gguf_tokenizer.load_tokenizer_from_gguf`
--- the same GGUF-only path `run-gguf-kquant.py` uses when called without
`--hf-source`. No parallel HF checkpoint is required.

Workflow
--------
1.  Preflight scan: hard-fail on IQ*/Q8_K (no MLX kquant kernel). Warn
    on Q4_0/Q4_1/Q5_0/Q5_1/Q8_1 --- these dequantize to bf16 during the
    load step below, which kills inference perf vs native K-quant.
2.  Load the GGUF into a kquant-native MLX model in memory via
    `load_kquant_model` (gguf_runtime). That helper handles arch
    detection, config synthesis, name remap, sanitize, and the
    `install_kquant_modules` swap. Returns model + synthesized config +
    tokenizer.
3.  Walk the model to read each kquant leaf's `kquant_type` attribute
    -> build the per-tensor codec map.
4.  Save model parameters as sharded safetensors (kquant weights as
    uint8 wire bytes, norms/etc as bf16).
5.  Write config.json (synthesized + `quantization_config.per_tensor`
    injected) and save the tokenizer via `save_pretrained`.

Usage
-----
  mqt-gguf-to-mlx /path/to/qwen3.6-27b-Q4_K_M.gguf \\
      -o ./qwen3.6-27b-Q4_K_M-from-gguf

  mqt-gguf-to-mlx file.gguf --preflight-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import gguf
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_quant_tools.gguf_config_synth import synthesize_config
from mlx_quant_tools.gguf_name_remap import detect_arch
from mlx_quant_tools.gguf_runtime import load_kquant_model

# ---------- codec classification (preflight only) ----------

GT = gguf.GGMLQuantizationType

# Codecs the MLX kquant decode path natively understands. The load
# step routes these into kquant Linears (wire-byte pass-through).
_NATIVE_KQUANT = {GT.Q8_0, GT.Q4_K, GT.Q5_K, GT.Q6_K, GT.Q2_K, GT.Q3_K}

# Float types --- pass through as bf16.
_FLOAT_TYPES = {GT.F32, GT.F16, GT.BF16}

# Legacy linear quants --- dequantized to bf16 during load (slow path).
_LEGACY_DEQUANT = {GT.Q4_0, GT.Q4_1, GT.Q5_0, GT.Q5_1, GT.Q8_1}

# Hard-fail: IQ family + Q8_K accumulator. No MLX kquant kernel exists.
_FAIL_TYPES = {
    GT.IQ1_S,
    GT.IQ1_M,
    GT.IQ2_XXS,
    GT.IQ2_XS,
    GT.IQ2_S,
    GT.IQ3_XXS,
    GT.IQ3_S,
    GT.IQ4_NL,
    GT.IQ4_XS,
    GT.Q8_K,
}


def preflight(reader) -> tuple[bool, list[str]]:
    counts: Counter = Counter()
    fail_examples: dict[str, str] = {}
    legacy_examples: dict[str, str] = {}
    for t in reader.tensors:
        counts[t.tensor_type.name] += 1
        if t.tensor_type in _FAIL_TYPES:
            fail_examples.setdefault(t.tensor_type.name, t.name)
        elif t.tensor_type in _LEGACY_DEQUANT:
            legacy_examples.setdefault(t.tensor_type.name, t.name)

    msgs = ["Tensor type distribution:"]
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        msgs.append(f"  {k:8s}: {v}")

    if fail_examples:
        msgs.append("")
        msgs.append("ERROR: unsupported codec(s) present (no MLX kquant kernel):")
        for name, example in fail_examples.items():
            msgs.append(f"  {name}: e.g. {example} ({counts[name]} tensors)")
        msgs.append(
            "These codecs would need IQ-family kernel support which "
            "MLX kquant does not provide. Aborting."
        )
        return False, msgs

    if legacy_examples:
        msgs.append("")
        msgs.append(
            "WARN: legacy linear quants present --- these will be "
            "dequantized to bf16 in the output (kills inference "
            "perf vs native K-quant):"
        )
        for name, example in legacy_examples.items():
            msgs.append(f"  {name}: e.g. {example} ({counts[name]} tensors)")
        msgs.append(
            "Re-encode the source GGUF with a K-quant codec to keep "
            "these tensors on the fast path. Conversion will proceed."
        )

    return True, msgs


# ---------- per-tensor codec extraction from the loaded model ----------


def _extract_per_tensor_codec(model: nn.Module) -> dict[str, str]:
    """Walk the model and return `{module_path: kquant_type}` for every
    kquant leaf (`QuantizedLinear`, `KQuantEmbedding`,
    `KQuantSwitchLinear`). Mirrors the `per_tensor` shape `load-mlx-kquant.py`
    expects in `config.json/quantization_config`.
    """
    out: dict[str, str] = {}
    for path, module in model.named_modules():
        kq_type = getattr(module, "kquant_type", None)
        if kq_type:
            out[path] = kq_type
    return out


# ---------- sharded safetensors writer ----------


def _save_sharded(
    weights: dict[str, mx.array],
    out_path: Path,
    *,
    shard_bytes: int,
) -> dict[str, str]:
    """Write `weights` as sharded safetensors using the HF naming
    convention. Returns `{tensor_name: shard_filename}`."""
    # Sort by name for deterministic shard contents.
    by_name = sorted(weights.items())
    shards: list[dict[str, mx.array]] = []
    cur: dict[str, mx.array] = {}
    cur_bytes = 0
    for name, arr in by_name:
        sz = arr.nbytes
        if cur and cur_bytes + sz > shard_bytes:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[name] = arr
        cur_bytes += sz
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards, start=1):
        shard_name = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(a.nbytes for a in shard.values())
        mx.save_safetensors(str(out_path / shard_name), shard)
        for k in shard:
            weight_map[k] = shard_name
        print(
            f"[INFO] wrote {shard_name}  ({size / 1e9:.2f} GB, {len(shard)} tensors)",
            file=sys.stderr,
        )
    return weight_map


# ---------- main conversion ----------


def convert(
    gguf_path: Path,
    out_path: Path,
    *,
    preflight_only: bool = False,
    shard_bytes: int = 5_000_000_000,
) -> None:
    print(f"[INFO] reading {gguf_path}", file=sys.stderr)
    reader = gguf.GGUFReader(str(gguf_path))

    ok, msgs = preflight(reader)
    for m in msgs:
        print(m, file=sys.stderr)
    if not ok:
        sys.exit(1)

    if preflight_only:
        return

    # Detect arch up-front for the config + tokenizer write paths.
    arch = detect_arch(reader)
    print(f"[INFO] arch: {arch}", file=sys.stderr)

    # Load the model in-memory via the GGUF-only path. load_kquant_model
    # handles config synthesis, name remap, sanitize, and the
    # install_kquant_modules swap. Returns the fully-loaded model + the
    # PreTrainedTokenizerFast built from GGUF metadata.
    print("[INFO] loading model in-memory via GGUF-only path...", file=sys.stderr)
    model, _config, tokenizer = load_kquant_model(
        str(gguf_path),
        hf_source=None,
        arch=arch,
    )
    mx.eval(model)

    # Read kquant_type off every kquant leaf for the per_tensor map.
    per_tensor = _extract_per_tensor_codec(model)
    print(f"[INFO] per_tensor codec map: {len(per_tensor)} kquant entries", file=sys.stderr)

    # Flatten params (kquant leaves expose `weight` uint8 wire bytes +
    # vestigial `scales` uint8(1); others stay at whatever dtype the
    # load step left them --- bf16 for norms after F32->bf16 cast).
    flat = dict(tree_flatten(model.parameters()))
    print(f"[INFO] total parameters: {len(flat)} tensors", file=sys.stderr)

    out_path.mkdir(parents=True, exist_ok=True)

    weight_map = _save_sharded(flat, out_path, shard_bytes=shard_bytes)

    total_size = sum(p.stat().st_size for p in out_path.glob("*.safetensors"))
    (out_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
    )

    # Synthesize the config fresh --- load_kquant_model strips
    # quantization_config and we want a clean dict to inject into.
    config_dict = synthesize_config(gguf.GGUFReader(str(gguf_path)))
    config_dict["quantization_config"] = {
        "mode": "kquant",
        "source_gguf": str(gguf_path),
        "per_tensor": per_tensor,
    }
    (out_path / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Tokenizer: save_pretrained writes tokenizer.json,
    # tokenizer_config.json, special_tokens_map.json, and chat_template.jinja
    # such that AutoTokenizer.from_pretrained() round-trips it.
    tokenizer.save_pretrained(str(out_path))

    print(f"[INFO] done. wrote {len(flat)} tensors to {out_path}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="mqt-gguf-to-mlx",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("gguf_path", type=Path, help="Path to source GGUF file.")
    p.add_argument(
        "-o",
        "--out-path",
        type=Path,
        help="Output directory. Defaults to ./<gguf-stem>-from-gguf.",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Print tensor-type distribution + warnings, then "
        "exit without loading or writing the model.",
    )
    p.add_argument(
        "--shard-bytes",
        type=int,
        default=5_000_000_000,
        help="Target safetensors shard size in bytes (default 5 GB).",
    )
    args = p.parse_args()

    if not args.gguf_path.exists():
        print(f"[ERROR] {args.gguf_path} not found", file=sys.stderr)
        return 1

    if args.out_path is None:
        args.out_path = Path.cwd() / f"{args.gguf_path.stem}-from-gguf"

    convert(
        gguf_path=args.gguf_path,
        out_path=args.out_path,
        preflight_only=args.preflight_only,
        shard_bytes=args.shard_bytes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
