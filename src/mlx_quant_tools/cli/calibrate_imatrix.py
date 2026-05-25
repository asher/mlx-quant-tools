"""Generate a llama-imatrix-compatible `.dat` file from a bf16 MLX model.

Runs the model forward over a plain-text calibration corpus, capturing
`sum(x^2) / ncall` per input feature on every quantizable Linear. Output
is the legacy llama-imatrix binary `.dat` format, consumable by
`mqt-quantize-kquant --imatrix` (its loader matches HF-style names directly
when no GGUF arch remap is needed).

Tensor names use HF convention (e.g. `model.layers.0.self_attn.q_proj.weight`)
matching what `model_role_classifier.classify_tensors` produces.

Usage
-----
  mqt-calibrate-imatrix mlx-community/Qwen3-0.6B \\
      --corpus /tmp/fineweb-sample.txt --chunks 50 --output imat.dat

  # MoE / instruct: same call. Use --chat-template only when calibrating
  # against a chat-formatted corpus; raw web text is the v0.2 default.

Limitations
-----------
- Hooks `nn.Linear` only --- not SwitchLinear/Embedding. Matches the
  modules `quantize-kquant.py` actually encodes for v0.
- Single-stream forward, batch=1. Smaller is simpler, accuracy is fine
  for imatrix generation since accumulation is order-independent.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map_with_path
from mlx_lm.utils import load

from mlx_quant_tools.model_role_classifier import classify_tensors


class _AccumLinear(nn.Module):
    """Wraps any module that takes activations as the first positional arg
    (nn.Linear, SwitchLinear, ...). Captures the input's per-feature
    sum(x**2) into a shared sink dict keyed by the wrapped module's full
    path + '.weight'.

    *args / **kwargs in __call__: SwitchLinear takes (x, indices, ...) so
    the wrapper must forward whatever extra args its base needs.
    """

    def __init__(self, base, path_with_weight: str, sink: dict):
        super().__init__()
        self.base = base
        self._path_with_weight = path_with_weight
        self._sink = sink

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        flat = x.astype(mx.float32).reshape(-1, x.shape[-1])
        sq = (flat * flat).sum(axis=0)
        ntok = int(flat.shape[0])
        entry = self._sink.get(self._path_with_weight)
        if entry is None:
            self._sink[self._path_with_weight] = {
                "sum_x2": sq,
                "ncall": ntok,
            }
        else:
            entry["sum_x2"] = entry["sum_x2"] + sq
            entry["ncall"] += ntok
        return self.base(x, *args, **kwargs)


def install_hooks(model, sink: dict) -> int:
    """Wrap every classifier-recognized module that has a 2D or 3D
    `.weight` (Linear, SwitchLinear, ...) in _AccumLinear.

    Returns the count of wrappings. Norms/biases/embeddings/scalar params
    are left untouched.
    """
    targets = set(classify_tensors(model).keys())
    n_wrapped = 0

    def _replace(path: str, module):
        nonlocal n_wrapped
        if path not in targets:
            return module
        if not isinstance(module, nn.Linear):
            return module
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim not in (2, 3):
            return module
        n_wrapped += 1
        return _AccumLinear(module, f"{path}.weight", sink)

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_replace, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return n_wrapped


def chunk_corpus(tokenizer, corpus_path: Path, ctx: int, max_chunks: int) -> list[list[int]]:
    """Tokenize the corpus and split into non-overlapping chunks of `ctx`
    tokens. Drops the trailing partial chunk if shorter than ctx//2.
    """
    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[list[int]] = []
    for start in range(0, len(ids), ctx):
        chunk = ids[start : start + ctx]
        if len(chunk) < max(1, ctx // 2):
            break
        chunks.append(chunk)
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
    return chunks


def write_dat(out_path: Path, sink: dict) -> None:
    """Write the legacy llama-imatrix .dat binary format.

    Layout (little-endian int32 sizes):
      int32  n_entries
      for each entry:
        int32  name_len
        bytes  name (utf-8, name_len bytes)
        int32  ncall
        int32  nval
        float32[nval]  avg = sum_x2 / ncall
    """
    items = []
    for name, entry in sink.items():
        ncall = int(entry["ncall"])
        if ncall == 0:
            continue
        avg = entry["sum_x2"] / float(ncall)
        mx.eval(avg)
        items.append((name, ncall, np.asarray(avg, dtype=np.float32).copy()))

    with out_path.open("wb") as f:
        f.write(struct.pack("<i", len(items)))
        for name, ncall, data in items:
            name_b = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<ii", ncall, data.size))
            f.write(data.tobytes())


def main() -> int:
    p = argparse.ArgumentParser(
        prog="mqt-calibrate-imatrix",
        description="Calibrate per-feature activation imatrix from an MLX model.",
    )
    p.add_argument("model", help="HF repo id or local model path")
    p.add_argument("--corpus", required=True, help="Plain-text calibration corpus")
    p.add_argument("--ctx", type=int, default=512, help="Tokens per forward pass (default 512)")
    p.add_argument(
        "--chunks", type=int, default=50, help="Max chunks to process; 0 = all (default 50)"
    )
    p.add_argument("--output", required=True, help="Output .dat path")
    args = p.parse_args()

    print(f"[INFO] loading {args.model}", file=sys.stderr)
    model, tokenizer = load(args.model)

    sink: dict = {}
    n_wrapped = install_hooks(model, sink)
    print(f"[INFO] hooked {n_wrapped} Linear modules", file=sys.stderr)
    if n_wrapped == 0:
        sys.exit("no quantizable Linear modules matched the classifier")

    chunks = chunk_corpus(tokenizer, Path(args.corpus), args.ctx, args.chunks)
    if not chunks:
        sys.exit("corpus produced no chunks (too short?)")
    print(f"[INFO] {len(chunks)} chunks x ctx={args.ctx} tokens", file=sys.stderr)

    for i, ids in enumerate(chunks):
        x = mx.array(ids, dtype=mx.int32).reshape(1, -1)
        logits = model(x)
        mx.eval(logits)
        # Force eval of every accumulator each chunk so memory doesn't
        # snowball with unmaterialized graph nodes.
        mx.eval([v["sum_x2"] for v in sink.values()])
        if (i + 1) % 10 == 0 or i + 1 == len(chunks):
            print(f"[INFO] chunk {i + 1}/{len(chunks)}", file=sys.stderr)

    write_dat(Path(args.output), sink)
    n_entries = sum(1 for v in sink.values() if v["ncall"] > 0)
    print(f"[INFO] wrote {args.output} ({n_entries} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
