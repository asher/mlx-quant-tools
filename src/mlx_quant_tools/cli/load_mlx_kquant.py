"""Load a checkpoint produced by ``mqt-quantize-kquant`` and generate text.

Reads `quantization_config.per_tensor` from config.json, swaps the
corresponding Linear modules for kquant-mode `nn.QuantizedLinear`, then
loads the uint8 wire-byte safetensors. Reuses the kquant module helpers
from `mlx_quant_tools.gguf_runtime`.

Usage
-----
  mqt-load-kquant /path/to/kquant-checkpoint \\
      --prompt "Explain quantization in one paragraph." --max-tokens 64

  mqt-load-kquant /path/to/kquant-checkpoint --no-chat-template \\
      --prompt "Once upon a time" --max-tokens 32

Limitations
-----------
- Loads only quantized Linear modules listed in `per_tensor`. Embeddings,
  MoE switch layers, and tied lm_head from earlier kquant decode paths
  are out of scope for v0 (the encode CLI only quantizes dense Linear).
- Text-only models only. Multimodal sources will fall back to the same
  build_model resolver used by run-gguf-kquant.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import load_tokenizer

from mlx_quant_tools.gguf_runtime import (
    build_model,
    install_kquant_modules,
)


def load_kquant_checkpoint(path: str | Path):
    """Construct a model from the kquant safetensors at `path` and return
    `(model, tokenizer, config)`. Raises if the config lacks the kquant
    quantization marker.
    """
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    qc = config.get("quantization_config") or {}
    if qc.get("mode") != "kquant":
        raise ValueError(
            f"{path}/config.json has quantization_config.mode = "
            f"{qc.get('mode')!r}, not 'kquant'. This loader is for "
            f"quantize-kquant.py output only."
        )
    per_tensor = qc.get("per_tensor") or {}
    if not per_tensor:
        raise ValueError(f"{path}/config.json has empty quantization_config.per_tensor")

    # Build the bf16 model class (empty weights), then swap quantized
    # Linears in place. install_kquant_modules keys on `<path>.weight`.
    model, _ = build_model(config)

    # build_model unwraps wrapper classes (e.g. qwen3_5_moe → TextModel
    # to dodge the `language_model.` prefix). When that happens, the
    # per_tensor paths quantize-kquant.py wrote (which include the
    # wrapper) won't match the loaded model's leaf paths. Detect by
    # probing for any saved key — if the model has no module at the
    # prefixed path, strip the prefix from both per_tensor and weights.
    _WRAPPER_PREFIXES = ("language_model.",)
    sample_path = next(iter(per_tensor))
    model_module_names = {p for p, _ in model.named_modules()}
    strip_prefix: str | None = None
    if sample_path not in model_module_names:
        for prefix in _WRAPPER_PREFIXES:
            if sample_path.startswith(prefix) and sample_path[len(prefix) :] in model_module_names:
                strip_prefix = prefix
                break
    if strip_prefix:
        per_tensor = {k[len(strip_prefix) :]: v for k, v in per_tensor.items()}

    weight_meta = {f"{p}.weight": c for p, c in per_tensor.items()}
    n_replaced = install_kquant_modules(model, weight_meta)
    if n_replaced != len(per_tensor):
        print(
            f"[WARN] swapped {n_replaced} modules but config.json has "
            f"{len(per_tensor)} entries — some paths missed (likely "
            f"non-Linear modules or naming mismatch).",
            file=sys.stderr,
        )

    # Load weights from safetensors. `mx.load` returns {path: array} with
    # the same keys we used during save (e.g. "model.layers.0.self_attn.
    # q_proj.weight"). kquant Linears expect uint8; bf16 modules expect
    # whatever dtype was written.
    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors in {path}")
    weights: dict[str, mx.array] = {}
    for f in weight_files:
        weights.update(mx.load(str(f)))
    if strip_prefix:
        weights = {
            (k[len(strip_prefix) :] if k.startswith(strip_prefix) else k): v
            for k, v in weights.items()
        }
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model)

    tokenizer = load_tokenizer(path)
    return model, tokenizer, config


def main() -> int:
    p = argparse.ArgumentParser(
        description="Load a kquant safetensors checkpoint and generate.",
    )
    p.add_argument("path", help="Path to kquant checkpoint directory.")
    p.add_argument(
        "--prompt", default="Hello, world.", help="User prompt (chat-template-wrapped by default)."
    )
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Send the prompt raw (no chat template wrap).",
    )
    args = p.parse_args()

    print(f"[load] {args.path}", file=sys.stderr)
    model, tokenizer, _ = load_kquant_checkpoint(args.path)
    print("[load] model ready", file=sys.stderr)

    prompt = args.prompt
    if not args.no_chat_template and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    sampler = mlx_lm.sample_utils.make_sampler(temp=args.temp)
    mlx_lm.generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
