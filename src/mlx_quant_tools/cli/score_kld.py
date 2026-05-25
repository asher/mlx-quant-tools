"""KLD scorer for MLX quantized checkpoints.

Tier 1 scope: one teacher, one student, one corpus per invocation.
Top-K-truncated teacher logits cached on disk; KLD computed online in fp32.

Outputs:
  - Markdown report (stdout by default; --md FILE to a file)
  - JSON dump pinned to schema_version=2 (default: <student>/kld-vs-<teacher>.json)

Usage:
  mqt-score-kld <teacher> <student> [options]

Examples:
  mqt-score-kld Qwen/Qwen3-0.6B /tmp/qwen3-0.6B-q4-AP-flat
  mqt-score-kld Qwen/Qwen3.6-27B /path/to/Qwen3.6-27B-UD-MLX-4bit \\
      --num-samples 512 --max-seq-len 2048

Self-test:
  mqt-score-kld --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------- locked constants ----------

# v2 protocol (2026-04-29): defaults match llama.cpp's --kl-divergence so our
# headline KLD reads in the same units as the field's published numbers
# (Unsloth, Bartowski, mradermacher all cite llama.cpp protocol). Three diffs
# previously inflated our numbers vs the field: top-K=128 reconstruction floor
# (~0.29 / ~0.44 nats on Qwen / Gemma anchors per `results/kld-floor-sweep.md`),
# full-sequence vs second-half-only scoring (cold-start bias), and corpus
# choice. n_ctx=2048 partially compensated. The KLD math itself is the same;
# only the data plumbing and defaults change.
#
# Long-context research mode (n_ctx=2048, full-sequence scoring) remains
# available behind --long-context.

SCHEMA_VERSION = 2  # bumped 2026-04-29 — protocol switch
# CACHE_FORMAT_VERSION is intentionally not bumped: v2's only cache change is
# an additive optional `score_window` field on the manifest (None → fallback
# to full-sequence at score time). Cache batch shards are bit-identical, so
# pre-v2 caches remain readable. Bumping the cache key only happens on
# *content* layout changes that would corrupt an old reader.
CACHE_FORMAT_VERSION = 1
DEFAULT_TOP_K = 32_768  # P1.0 result: smallest K with Gemma floor < 0.001 nats
DEFAULT_NUM_SAMPLES = 512
DEFAULT_MAX_SEQ_LEN = 512  # was 2048; matches llama.cpp default n_ctx
DEFAULT_SEED = 123
DEFAULT_DATASET = "Salesforce/wikitext:wikitext-103-raw-v1"  # was wikitext-2-raw-v1
DEFAULT_CACHE_DIR = Path.home() / ".mlx-kld-cache"

# Position buckets are quartiles of the *scored* window. Default protocol
# scores [n_ctx/2, n_ctx); long-context mode scores [0, n_ctx). Buckets are
# computed from the manifest's score_window at scoring time, not pre-baked.
# Kept here as a fallback for any external caller that still hard-codes them.
POSITION_BUCKETS: list[tuple[int, int | None]] = [
    (256, 320),
    (320, 384),
    (384, 448),
    (448, None),
]
LONG_CONTEXT_POSITION_BUCKETS: list[tuple[int, int | None]] = [
    (0, 128),
    (128, 512),
    (512, 1536),
    (1536, None),
]


# ---------- progress logging ----------


def info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


# ---------- multimodal detection / loader dispatch ----------
#
# Mirrors attn-protect-quantize.py's is_multimodal: a config is multimodal iff
# vision_config or audio_config is a non-empty dict. Duplicated rather than
# shared because cross-importing a hyphenated filename is awkward and the
# helper is short. Keep the two definitions in sync.


def is_multimodal(config: dict) -> bool:
    for key in ("vision_config", "audio_config"):
        sub = config.get(key)
        if isinstance(sub, dict) and sub:
            return True
    return False


def _peek_config(path_or_id: str) -> dict:
    """Read config.json without instantiating the model (for early dispatch).

    Resolves HF ids to a local snapshot path via mlx_vlm.utils.get_model_path,
    which works for both text-only and VLM sources.
    """
    from mlx_vlm.utils import get_model_path

    src = get_model_path(path_or_id)
    return json.loads((src / "config.json").read_text())


def _is_kquant(cfg: dict) -> bool:
    qc = cfg.get("quantization_config") or {}
    return qc.get("mode") == "kquant"


def _load_model(path_or_id: str, lazy: bool = True):
    """Dispatch on (peeked) config: kquant → load_mlx_kquant; VLM → mlx_vlm;
    else mlx_lm.

    Returns (model, config). Tokenizer is sourced separately via load_tokenizer
    (works for kquant, VLM, and text-only paths — tokenizer.json is the
    standard HF format in all three cases).

    For VLMs, the model's __call__ returns a LanguageModelOutput object with a
    .logits attribute rather than a raw logits array; callers should unwrap
    via _extract_logits(). kquant checkpoints are always text-only in v0.
    """
    cfg = _peek_config(path_or_id)
    if _is_kquant(cfg):
        from mlx_vlm.utils import get_model_path

        from mlx_quant_tools.cli.load_mlx_kquant import load_kquant_checkpoint

        src = get_model_path(path_or_id)
        # load_kquant_checkpoint always materializes weights (lazy=True is
        # ignored). Returns (model, tokenizer, config); we drop the tokenizer
        # since the caller loads it separately via mlx_lm.utils.load_tokenizer.
        model, _tok, config = load_kquant_checkpoint(src)
        return model, config
    if is_multimodal(cfg):
        from mlx_vlm.utils import fetch_from_hub, get_model_path

        src = get_model_path(path_or_id)
        model, config, _processor = fetch_from_hub(src, lazy=lazy)
        return model, config
    from mlx_lm.utils import load

    model, _tok, config = load(path_or_id, lazy=lazy, return_config=True)
    return model, config


def _extract_logits(out):
    """mlx-lm models return a logits array directly; mlx-vlm wraps it in a
    LanguageModelOutput dataclass. Normalize to the array."""
    return out.logits if hasattr(out, "logits") else out


def _make_fresh_cache(model):
    """Build a fresh per-batch KV cache aligned to the model's LM stack.

    Required for gemma-4 (and likely other VLMs whose __call__ skips RoPE
    bookkeeping when cache is None — verified empirically: without a cache
    on gemma-4-E2B the bf16 teacher emits garbage argmaxes, with a fresh
    one it correctly predicts " Berlin" for the canonical capitals prompt).

    Harmless on text-only mlx-lm models — argmax is unchanged with vs.
    without cache (verified on Qwen3-0.6B). We always pass a cache so the
    two paths share one forward call.
    """
    from mlx_lm.models import cache as cache_mod

    target = model.language_model if hasattr(model, "language_model") else model
    return cache_mod.make_prompt_cache(target, max_kv_size=None)


# ---------- numerical primitives ----------
#
# All KLD math runs in fp32 even when the disk cache is bf16. Two reasons:
#   1. Summing 250k log-prob terms in bf16 loses ~0.5 nats of precision per
#      position — more than the actual signal between two competent quants.
#   2. log1mexp() near zero collapses to garbage in low precision.


def log1mexp_fp32(x):
    """Numerically stable log(1 - exp(x)) for x <= 0, in fp32.

    Two regimes (Mächler 2012): use log(-expm1(x)) when x is close to 0
    (|x| < log 2), else log1p(-exp(x)). Implemented via mx.where so it
    stays a fused MLX op.
    """
    import mlx.core as mx

    cutoff = -math.log(2.0)
    safe_close = mx.minimum(x, mx.array(-1e-30, dtype=mx.float32))
    safe_far = mx.minimum(x, mx.array(cutoff, dtype=mx.float32))
    a = mx.log(-mx.expm1(safe_close))
    b = mx.log1p(-mx.exp(safe_far))
    return mx.where(x > cutoff, a, b)


def kld_from_topk(
    teacher_topk_log_p,  # (B, T, K) fp32, log-softmax of teacher's top-K
    teacher_topk_indices,  # (B, T, K) int32, teacher's top-K vocab ids
    student_logits,  # (B, T, V) any float dtype; cast to fp32 internally
    vocab_size: int,
):
    """Compute per-token KLD(P||Q) where P is the top-K-reconstructed teacher
    and Q is the full student distribution.

    Reconstruction:
        log_p[i in topK] = teacher_topk_log_p
        log_p[i NOT in topK] = log((1 - sum_topK exp(log_p)) / (V - K))
                              = log1mexp(logsumexp(log_p_topK)) - log(V - K)

    Closed-form expansion of KL(P||Q) under this approximation, using the
    student's full log-softmax (computed implicitly without materializing
    the (B,T,V) array — see `log_q_sum_full` below).

    Returns: (kld[B,T] fp32, student_top1[B,T] uint32, student_top5[B,T,5] uint32)
    """
    import mlx.core as mx

    K = teacher_topk_log_p.shape[-1]
    V = vocab_size
    tail_size = V - K
    assert tail_size > 0, "vocab_size must exceed top-K"

    # Cast student logits to fp32 once; reuse for both KLD math and top-1/top-5.
    sl = student_logits.astype(mx.float32)
    lse = mx.logsumexp(sl, axis=-1)  # (B, T)
    sum_logits = mx.sum(sl, axis=-1)  # (B, T)

    # log_softmax(x) = x - lse, so sum over vocab equals sum(x) - V * lse.
    log_q_sum_full = sum_logits - float(V) * lse  # (B, T)
    # Gather student log_q at teacher's top-K positions (avoid full softmax).
    log_q_topk = (
        mx.take_along_axis(sl, teacher_topk_indices, axis=-1) - lse[..., None]
    )  # (B, T, K)
    log_q_sum_topk = mx.sum(log_q_topk, axis=-1)  # (B, T)
    log_q_sum_tail = log_q_sum_full - log_q_sum_topk  # (B, T)

    log_p_topk = teacher_topk_log_p.astype(mx.float32)  # (B, T, K)
    log_p_topk_sum = mx.logsumexp(log_p_topk, axis=-1)  # (B, T)

    # Tail per-element log-prob (uniform residual). Clamp head sum strictly < 0
    # so log1mexp doesn't hit log(0) when teacher mass is fully captured.
    log_p_topk_sum_safe = mx.minimum(log_p_topk_sum, mx.array(-1e-7, dtype=mx.float32))
    log_p_tail = log1mexp_fp32(log_p_topk_sum_safe) - math.log(float(tail_size))  # (B, T)

    # Head term: sum over teacher's top-K of p_i * (log_p_i - log_q_i).
    head_term = mx.sum(mx.exp(log_p_topk) * (log_p_topk - log_q_topk), axis=-1)  # (B, T)

    # Tail term: sum over (V-K) tail positions of p_tail * (log_p_tail - log_q_i)
    #          = exp(log_p_tail) * ((V-K) * log_p_tail - sum_{tail} log_q_i)
    tail_term = mx.exp(log_p_tail) * (float(tail_size) * log_p_tail - log_q_sum_tail)  # (B, T)

    kld = head_term + tail_term

    # Top-1 / top-5 of the student. We only need the index sets; argpartition
    # is O(V) and avoids a full sort.
    student_top1 = mx.argmax(sl, axis=-1)  # (B, T)
    student_top5 = mx.argpartition(sl, kth=-5, axis=-1)[..., -5:]  # (B, T, 5)

    return kld, student_top1, student_top5


# ---------- tokenization / data ----------


def _split_dataset_spec(spec: str) -> tuple[str, str | None]:
    """Split `path:name` into `(path, subset_name)` for HF subset configs.

    Mirrors `attn-protect-quantize.py`'s `_split_calibration_data` so the
    two CLIs accept the same syntax. Lets us pass corpora that require a
    `name=` kwarg to `datasets.load_dataset`, e.g.
    `HuggingFaceFW/fineweb-edu:sample-10BT`. Plain `path` returns
    `(spec, None)`.
    """
    path, sep, name = spec.partition(":")
    return (path, name) if sep and name else (spec, None)


def _render_chat_messages(tokenizer, dataset_name: str, ds) -> list[str]:
    """Apply the tokenizer's chat template to each row's `messages` list.

    For chat-format corpora (allenai/tulu-3-sft-mixture, ultrachat_200k,
    etc.) the per-row text isn't a plain string but a list of role/content
    dicts. Rendering through the tokenizer's chat template produces the
    deployment-shaped string the model would actually see at inference,
    so KLD calibrated on this lands in-distribution for instruct deploys.

    Bails clearly if the tokenizer has no `chat_template` set — most chat
    corpora are useless without one.
    """
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    if not getattr(inner, "chat_template", None):
        sys.exit(
            f"Dataset {dataset_name!r} is chat-format (column 'messages') "
            "but the tokenizer has no chat_template; pick a teacher with a "
            "chat template or use a text-format corpus."
        )
    texts: list[str] = []
    for r in ds:
        msgs = r.get("messages") or []
        if not msgs:
            continue
        try:
            rendered = inner.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            sys.exit(f"apply_chat_template failed on a row from {dataset_name!r}: {e}")
        if rendered.strip():
            texts.append(rendered)
    return texts


def load_calibration_tokens(
    tokenizer,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
):
    """Tokenize the calibration corpus, slice into fixed-length chunks,
    deterministically pick `num_samples` of them.

    Accepts `path:subset` syntax for HF subset configs (e.g.
    `HuggingFaceFW/fineweb-edu:sample-10BT`). Auto-detects chat-format
    corpora (with a `messages` column) and renders them through the
    tokenizer's chat template before tokenizing — so e.g.
    `allenai/tulu-3-sft-mixture` is usable directly without a separate
    text-format conversion.

    Returns mx.array of shape (num_samples, max_seq_len) int32.
    """
    import mlx.core as mx
    import numpy as np
    from datasets import load_dataset

    if dataset_name == "wikitext-2-raw-v1":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [r["text"] for r in ds if r["text"].strip()]
    else:
        path, subset = _split_dataset_spec(dataset_name)
        ds = (
            load_dataset(path, name=subset, split="train")
            if subset
            else load_dataset(path, split="train")
        )
        cols = ds.column_names
        if "text" in cols:
            texts = [r["text"] for r in ds if r["text"].strip()]
        elif "messages" in cols:
            texts = _render_chat_messages(tokenizer, dataset_name, ds)
        else:
            sys.exit(
                f"Dataset {dataset_name!r} has neither a 'text' nor "
                f"'messages' column (found {cols}); --dataset must point "
                "at a text- or chat-format corpus."
            )

    info(f"Tokenizing {dataset_name} ({len(texts)} non-empty rows)…")
    # Concatenate: a single stream of ids, then chunk at max_seq_len. This is
    # the standard wikitext perplexity recipe — no padding, no BOS.
    all_ids: list[int] = []
    chunk_target = num_samples * max_seq_len + max_seq_len  # +slack
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= chunk_target * 2:
            break  # plenty for shuffling

    n_chunks = len(all_ids) // max_seq_len
    if n_chunks < num_samples:
        sys.exit(
            f"Calibration corpus too small: got {n_chunks} chunks of "
            f"length {max_seq_len}, need {num_samples}. Lower --num-samples "
            "or --max-seq-len."
        )

    ids_arr = np.asarray(all_ids[: n_chunks * max_seq_len], dtype=np.int32)
    chunks = ids_arr.reshape(n_chunks, max_seq_len)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_chunks)[:num_samples]
    chunks = chunks[perm]

    return mx.array(chunks)


# ---------- cache I/O ----------


def cache_key(
    teacher_path: str,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    top_k: int,
) -> str:
    payload = "|".join(
        [
            teacher_path,
            dataset_name,
            str(num_samples),
            str(max_seq_len),
            str(seed),
            str(top_k),
            f"v{CACHE_FORMAT_VERSION}",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def tokenizer_hash(tokenizer) -> str:
    """Hash a loaded tokenizer's *behavior*: vocab map + special tokens.

    Hashing the on-disk bytes is too strict — mlx-lm's save() reformats
    tokenizer.json, which produces different bytes for a functionally
    identical tokenizer. We instead hash the (id → token) pairs that drive
    encoding/decoding, so the check passes iff KLD is actually defined.
    """
    h = hashlib.sha256()
    # Inner tokenizer if this is mlx_lm's TokenizerWrapper.
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    vocab_size = getattr(inner, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(inner)
    h.update(f"vocab_size={vocab_size}\n".encode())
    # Walk every id 0..len(tokenizer); convert_ids_to_tokens covers added
    # tokens too. Pin a fixed iteration order.
    try:
        for i in range(len(inner)):
            tok = inner.convert_ids_to_tokens(i)
            h.update(f"{i}\t{tok}\n".encode("utf-8", errors="replace"))
    except Exception as e:
        h.update(f"<walk-failed:{e}>".encode())
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(inner, name, None)
        h.update(f"{name}={v}\n".encode())
    return h.hexdigest()


def cache_is_valid(cache_dir: Path, expected_num_batches: int) -> tuple[bool, str]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False, "no manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False, "manifest corrupt"
    if manifest.get("format_version") != CACHE_FORMAT_VERSION:
        return (
            False,
            "format_version mismatch "
            f"({manifest.get('format_version')}"
            f" != {CACHE_FORMAT_VERSION})",
        )
    declared = manifest.get("num_batches")
    if declared != expected_num_batches:
        return False, f"num_batches mismatch ({declared} != {expected_num_batches})"
    for i in range(declared):
        if not (cache_dir / f"batch-{i:05d}.safetensors").exists():
            return False, f"missing batch-{i:05d}.safetensors"
    return True, "ok"


# ---------- model param/byte counting (for effective_bpw) ----------


def _load_inspector():
    """Import inspect_recipe module for tensor metadata collection."""
    from mlx_quant_tools.cli import inspect_recipe as _inspector_mod

    return _inspector_mod


# Codec geometry — (group_size, bits, bytes_per_block, weights_per_block).
# Mirrors run-gguf-kquant.py:CODEC_GEOMETRY; duplicated here so measure_student
# stays self-contained (no run-gguf-kquant import for the byte-counting path).
_KQUANT_GEOMETRY: dict[str, tuple[int, int, int, int]] = {
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


def _measure_kquant_student(student_dir: Path, config: dict) -> dict:
    """size_bytes + effective_bpw for a kquant checkpoint.

    Affine inspector's bpw math assumes (bits, group_size) from the
    `quantization` config key, which kquant doesn't have — kquant uses
    `quantization_config.per_tensor` (path → codec) instead. Compute logical
    param count per-tensor: wire_bytes // bytes_per_block * weights_per_block
    for kquant entries, prod(shape) for bf16 pass-through tensors.
    """
    inspector = _load_inspector()
    qc = config.get("quantization_config") or {}
    per_tensor = qc.get("per_tensor") or {}
    meta = inspector.collect_tensor_metadata(student_dir)
    total_bytes = 0
    total_params = 0
    for name, info in meta.items():
        nbytes = info["nbytes"]
        total_bytes += nbytes
        base = name.removesuffix(".weight") if name.endswith(".weight") else name
        codec = per_tensor.get(base)
        if codec is not None and codec in _KQUANT_GEOMETRY:
            _gs, _bits, bpb, wpb = _KQUANT_GEOMETRY[codec]
            total_params += (nbytes // bpb) * wpb
        else:
            shape = info.get("shape") or []
            n = 1
            for d in shape:
                n *= int(d)
            total_params += n
    bpw = (total_bytes * 8 / total_params) if total_params else None
    return {
        "size_bytes": total_bytes,
        "effective_bpw": bpw,
        "default_bits": None,
        "default_group_size": None,
    }


def measure_student(student_dir: Path) -> dict:
    """Total on-disk bytes and effective bits-per-weight."""
    inspector = _load_inspector()
    config = json.loads((student_dir / "config.json").read_text())
    if _is_kquant(config):
        return _measure_kquant_student(student_dir, config)
    quant_cfg = config.get("quantization", {}) or {}
    default_bits = quant_cfg.get("bits") if isinstance(quant_cfg, dict) else None
    default_gs = quant_cfg.get("group_size") if isinstance(quant_cfg, dict) else None

    meta = inspector.collect_tensor_metadata(student_dir)
    modules_raw = inspector.group_into_modules(meta)
    modules = [
        inspector.derive_module_recipe(b, parts, quant_cfg, default_bits, default_gs)
        for b, parts in modules_raw.items()
        if "weight" in parts
    ]
    total_params = sum((m["param_count"] or 0) for m in modules)
    total_bytes = sum(m["total_bytes"] for m in modules)
    bpw = (total_bytes * 8 / total_params) if total_params else None
    return {
        "size_bytes": total_bytes,
        "effective_bpw": bpw,
        "default_bits": default_bits,
        "default_group_size": default_gs,
    }


# ---------- recipe sub-object ----------

# Locked recipe key set — mirror of attn-protect-quantize.py's CLI flags.
# Keys can be added (schema-additive) without bumping schema_version: the
# loader treats missing keys in older third-party JSONs as None, matching
# the existing tool="unknown" convention. Removing or renaming a key WOULD
# require a schema bump.
RECIPE_KEYS = (
    "tool",
    "tool_version",
    "bits",
    "group_size",
    "attn_protect_mode",
    "with_dwq",
    "with_mlp_boosts",
    "floor_tied_embed",
    "protect_vlm",
    "quantize_linear_attn",
    "quantize_attn_out",
    "no_attn_floor",
    "no_lm_head_floor",
    "quantize_moe_router",
    # `dwq` is a nested sub-object present iff with_dwq is true; carries the
    # calibration spec (corpus, samples, seed, batch, seq) that produced the
    # refined scales. Pass-through dict; rollup consumers can drill in or
    # ignore. Loader treats absence as None, same as any other optional key.
    "dwq",
    # `dwq_failed` and `dwq_failure_reason` are set iff a DWQ cascade was
    # requested but raised mid-run; the saved weights are AP-only in that
    # case and `with_dwq` is False. Schema-additive — older recipes without
    # these keys are treated as never-failed (the normal path).
    "dwq_failed",
    "dwq_failure_reason",
    # `mlp_boosts` is a nested sub-object present iff with_mlp_boosts is
    # true. Carries the budget, calibration spec, and the resolved
    # per-tensor boost map. Pass-through dict.
    "mlp_boosts",
)

# Pre-rename ablation flag names. Older recipe.json files (written before the
# `no_skip_*` → `quantize_*` flip) get silently migrated at load time so old
# checkpoints stay scoreable without re-running them. Same boolean semantics
# both sides — only the name changed.
_LEGACY_RECIPE_RENAMES = {
    "no_skip_linear_attn": "quantize_linear_attn",
    "no_skip_attn_out": "quantize_attn_out",
    "no_skip_moe_router": "quantize_moe_router",
}


def measure_gguf_student(gguf_path: Path) -> dict:
    """Counterpart to measure_student() for GGUF K-quant students.
    Reports on-disk bytes; effective_bpw is left None because computing it
    from GGUF requires summing per-tensor wire-byte sizes against logical
    parameter counts (out of scope for the KLD report)."""
    return {
        "size_bytes": gguf_path.stat().st_size,
        "effective_bpw": None,
        "default_bits": None,
        "default_group_size": None,
    }


def recipe_for_gguf_student() -> dict:
    """Synthesize a recipe stub for a GGUF K-quant student."""
    out = {k: None for k in RECIPE_KEYS}
    out["tool"] = "kquant-gguf"
    return out


def recipe_for_kquant_student() -> dict:
    """Synthesize a recipe stub for an MLX kquant student.

    The kquant recipe is per-tensor (mixed codecs), so bits/group_size aren't
    meaningful at this granularity — left as None like the GGUF stub. Detail
    lives in `quantize-kquant-report.json` next to the checkpoint.
    """
    out = {k: None for k in RECIPE_KEYS}
    out["tool"] = "kquant-mlx"
    return out


def load_recipe(student_dir: Path, fallback_meta: dict) -> dict:
    """Read attn-protect-recipe.json if present; else synthesize a stub
    with `tool='unknown'` and bits/group_size pulled from the checkpoint.

    Always returns the same key set for diffability."""
    config_path = student_dir / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
        if _is_kquant(cfg):
            return recipe_for_kquant_student()
    recipe_file = student_dir / "attn-protect-recipe.json"
    out = {k: None for k in RECIPE_KEYS}
    if recipe_file.exists():
        loaded = json.loads(recipe_file.read_text())
        for old, new in _LEGACY_RECIPE_RENAMES.items():
            if old in loaded and new not in loaded:
                loaded[new] = loaded[old]
        for k in RECIPE_KEYS:
            if k in loaded:
                out[k] = loaded[k]
        # Always carry these two from the checkpoint as the source of truth
        # in case the recipe file is stale.
        if fallback_meta.get("default_bits") is not None:
            out["bits"] = fallback_meta["default_bits"]
        if fallback_meta.get("default_group_size") is not None:
            out["group_size"] = fallback_meta["default_group_size"]
    else:
        out["tool"] = "unknown"
        out["bits"] = fallback_meta.get("default_bits")
        out["group_size"] = fallback_meta.get("default_group_size")
    return out


# ---------- aggregator ----------


class Aggregator:
    """Streams per-token KLD into running stats + a per-position-bucket view.

    For Tier-1 model sizes (≤27B at 512×2048 = 1M tokens × 4 bytes = 4 MB),
    we keep all per-token KLDs in memory and compute exact quantiles at the
    end. Avoids the t-digest dependency and lets the histogram be exact too.
    """

    def __init__(self):
        self._klds: list = []  # list of np.ndarray fp32 (B, T)
        self._top1: list = []  # list of np.ndarray bool (B, T)
        self._top5: list = []  # list of np.ndarray bool (B, T)
        self._positions: list = []  # list of np.ndarray int32 (B, T)

    def update(self, kld_bt, top1_match_bt, top5_match_bt, position_bt):
        import numpy as np

        self._klds.append(np.asarray(kld_bt, dtype=np.float32))
        self._top1.append(np.asarray(top1_match_bt, dtype=bool))
        self._top5.append(np.asarray(top5_match_bt, dtype=bool))
        self._positions.append(np.asarray(position_bt, dtype=np.int32))

    def finalize(
        self,
        position_buckets: list[tuple[int, int | None]] | None = None,
    ) -> dict:
        import numpy as np

        if not self._klds:
            return {"tokens_scored": 0}
        kld = np.concatenate([a.ravel() for a in self._klds])
        top1 = np.concatenate([a.ravel() for a in self._top1])
        top5 = np.concatenate([a.ravel() for a in self._top5])
        pos = np.concatenate([a.ravel() for a in self._positions])

        # Defensive: clip any negative KLD from fp32 noise on near-identical
        # distributions. KLD is mathematically >= 0.
        kld = np.maximum(kld, 0.0)

        global_stats = {
            "mean": float(kld.mean()),
            "p50": float(np.quantile(kld, 0.5)),
            "p95": float(np.quantile(kld, 0.95)),
            "p99": float(np.quantile(kld, 0.99)),
            "p999": float(np.quantile(kld, 0.999)),
            "max": float(kld.max()),
        }
        agreement = {
            "top1": float(top1.mean()),
            "top5": float(top5.mean()),
        }

        # Per-position buckets — caller passes mode-appropriate buckets
        # (quartiles of the score window); fall back to v2 defaults.
        buckets = position_buckets or POSITION_BUCKETS
        by_position = []
        for start, end in buckets:
            if end is None:
                mask = pos >= start
                label = f"{start}-end"
            else:
                mask = (pos >= start) & (pos < end)
                label = f"{start}-{end}"
            n = int(mask.sum())
            if n == 0:
                by_position.append({"range": label, "tokens": 0})
                continue
            sub = kld[mask]
            by_position.append(
                {
                    "range": label,
                    "tokens": n,
                    "mean_kld": float(sub.mean()),
                    "p95_kld": float(np.quantile(sub, 0.95)),
                    "top1_agreement": float(top1[mask].mean()),
                }
            )

        # Histogram (fixed log-spaced bins; exact counts).
        bins = np.concatenate(
            [
                np.array([0.0]),
                np.logspace(-6, 1, 29),  # 1e-6 .. 1e1, 29 edges → 28 bins
            ]
        )
        counts, edges = np.histogram(kld, bins=bins)
        histogram = {
            "bin_edges": [float(x) for x in edges],
            "counts": [int(c) for c in counts],
        }

        return {
            "tokens_scored": int(kld.size),
            "kld": global_stats,
            "agreement": agreement,
            "by_position": by_position,
            "kld_histogram": histogram,
        }


# ---------- HF revision sniff ----------


def hf_revision_of(path_or_id: str) -> str | None:
    """Return commit SHA if `path_or_id` is a local HF snapshot, else None.

    HF's snapshot_download stores commits at
        ~/.cache/huggingface/hub/models--ORG--NAME/snapshots/<sha>/
    so the local resolved path's parent name is the SHA.
    """
    p = Path(path_or_id)
    if p.exists():
        # If the resolved path looks like an HF snapshot, the parent dir name
        # is the commit hash (40 hex chars).
        if p.parent.name == "snapshots" or len(p.name) == 40:
            return p.name
    return None


# ---------- teacher cache lifecycle (public entry for external tools) ----------


def ensure_teacher_topk_cache(
    *,
    teacher_path: str,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    batch_size: int = 1,
    # Helper-level default is intentionally pinned to 128, NOT DEFAULT_TOP_K.
    # External callers (e.g. attn-protect-quantize.py sensitivity scoring) that
    # don't pass top_k explicitly should keep getting K=128 — sensitivity is a
    # within-loop delta ranking and benefits from cache reuse across the v2
    # protocol switch. The user-facing CLI passes args.top_k = DEFAULT_TOP_K
    # explicitly so absolute publication numbers use the v2 default.
    top_k: int = 128,
    cache_root: Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
    score_window: tuple[int, int] | None = None,
):
    """Ensure a teacher top-K cache exists at ~/.mlx-kld-cache/<key>/ for
    the given (teacher × calibration spec). Returns `(cache_dir, manifest,
    tokenizer)`.

    Tokenizer is returned alongside because external callers (e.g.
    `attn-protect-quantize.py` sensitivity scoring) typically need it for
    parity checks and to confirm the in-memory model's vocab matches what
    the cache was built against.

    Behavior is bit-identical to score-mlx-kld.py main()'s teacher-cache
    section: tokenizer load, hash, vocab probe, calibration tokenize,
    cache validity check, and conditional teacher_pass. Lifted as a single
    callable so external tools that need teacher logits without running
    the full scorer don't have to reproduce the orchestration.
    """
    from mlx_lm.utils import load_tokenizer

    info(f"Loading teacher tokenizer: {teacher_path}")
    teacher_tokenizer = load_tokenizer(teacher_path)
    teacher_tok_hash = tokenizer_hash(teacher_tokenizer)

    tokens = load_calibration_tokens(
        teacher_tokenizer,
        dataset_name,
        num_samples,
        max_seq_len,
        seed,
    )
    info(f"Calibration: {tokens.shape[0]} sequences × {tokens.shape[1]} tokens")

    key = cache_key(teacher_path, dataset_name, num_samples, max_seq_len, seed, top_k)
    cache_dir: Path = cache_root / key
    expected_batches = (tokens.shape[0] + batch_size - 1) // batch_size

    if score_window is None:
        score_window = (0, max_seq_len)
    valid, reason = cache_is_valid(cache_dir, expected_batches)
    if valid and not rebuild:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if manifest.get("tokenizer_hash") not in (None, teacher_tok_hash):
            sys.exit(
                "Cache tokenizer_hash mismatch — cache was built against a "
                f"different tokenizer ({manifest['tokenizer_hash']} != "
                f"{teacher_tok_hash}). Rebuild with rebuild=True."
            )
        info(f"Teacher cache HIT: {cache_dir}")
        # score_window is independent of teacher cache contents — overlay
        # the requested window so callers see the score-time policy in the
        # returned manifest. The manifest on disk is left untouched.
        manifest = dict(manifest)
        manifest["score_window"] = [int(score_window[0]), int(score_window[1])]
        return cache_dir, manifest, teacher_tokenizer

    if valid and rebuild:
        info(f"rebuild=True: rebuilding {cache_dir}")
    else:
        info(f"Teacher cache MISS ({reason})")
    if cache_dir.exists():
        for f in cache_dir.glob("batch-*.safetensors"):
            f.unlink()
        (cache_dir / "manifest.json").unlink(missing_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = teacher_pass(
        teacher_path,
        tokens,
        batch_size,
        top_k,
        cache_dir,
        dataset_name,
        num_samples,
        max_seq_len,
        seed,
        teacher_tok_hash,
        vocab_size=teacher_tokenizer.vocab_size,
        score_window=score_window,
    )
    return cache_dir, manifest, teacher_tokenizer


# ---------- teacher pass ----------


def teacher_pass(
    teacher_path: str,
    tokens,  # mx.array (N, L) int32
    batch_size: int,
    top_k: int,
    cache_dir: Path,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    teacher_tokenizer_hash: str,
    vocab_size: int,
    score_window: tuple[int, int] | None = None,
):
    """Forward the teacher and write top-K cache shards + manifest."""
    import mlx.core as mx
    from tqdm import tqdm

    info(f"Loading teacher: {teacher_path}")
    teacher_model, teacher_config = _load_model(teacher_path, lazy=True)
    teacher_model.eval()

    # Confirm vocab_size matches the (B,T,V) the teacher actually emits.
    # (Some configs have padded vocab vs tokenizer vocab.)
    config_vocab = teacher_config.get("vocab_size")

    cache_dir.mkdir(parents=True, exist_ok=True)
    N, L = tokens.shape
    num_batches = (N + batch_size - 1) // batch_size

    info(f"Teacher pass: {N} sequences × {L} tokens, batch_size={batch_size}, K={top_k}")

    measured_vocab = None
    for i in tqdm(range(num_batches), desc="teacher", file=sys.stderr):
        s = i * batch_size
        e = min(s + batch_size, N)
        batch = tokens[s:e]  # (B, L) int32
        cache = _make_fresh_cache(teacher_model)
        logits = _extract_logits(teacher_model(batch, cache=cache))  # (B, L, V)
        # Cast to fp32 for the top-K extraction so log_softmax is precise.
        lf = logits.astype(mx.float32)
        log_softmax = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
        # argpartition gives unsorted top-K; we sort descending so [..., 0] is
        # always the argmax. That makes top-1/top-5 agreement trivial later.
        idx_part = mx.argpartition(log_softmax, kth=-top_k, axis=-1)[..., -top_k:]
        vals_part = mx.take_along_axis(log_softmax, idx_part, axis=-1)
        order = mx.argsort(-vals_part, axis=-1)
        top_idx = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
        top_log_p = mx.take_along_axis(vals_part, order, axis=-1).astype(mx.bfloat16)
        # Synchronize and free intermediate fp32 tensors before writing.
        mx.eval(top_idx, top_log_p)
        if measured_vocab is None:
            measured_vocab = int(logits.shape[-1])
        attention_mask = mx.ones(batch.shape, dtype=mx.bool_)
        mx.save_safetensors(
            str(cache_dir / f"batch-{i:05d}.safetensors"),
            {
                "top_k_log_softmax": top_log_p,
                "top_k_indices": top_idx,
                "token_ids": batch.astype(mx.int32),
                "attention_mask": attention_mask,
            },
        )

    # Write manifest last so a partial cache won't pass cache_is_valid().
    if score_window is None:
        score_window = (0, max_seq_len)
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "teacher_path": teacher_path,
        "teacher_revision": hf_revision_of(teacher_path),
        "dataset": dataset_name,
        "num_samples": num_samples,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "top_k": top_k,
        "score_window": [int(score_window[0]), int(score_window[1])],
        "vocab_size": measured_vocab,
        "config_vocab_size": config_vocab,
        "tokenizer_hash": teacher_tokenizer_hash,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "logit_dtype": "bfloat16",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    info("Freeing teacher")
    del teacher_model
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and mx.metal.is_available():
        mx.metal.clear_cache()  # mlx-lm <= 0.31 fallback
    return manifest


# ---------- student pass ----------


def score_loaded_student(
    student_model,
    cache_dir: Path,
    manifest: dict,
) -> dict:
    """Forward an already-loaded student model against the cached teacher
    top-K and return the KLD aggregate.

    Public entry point used by both `student_pass` (which loads from disk
    first) and external tools like `attn-protect-quantize.py`'s sensitivity
    scoring loop, which holds the model in memory and swaps individual
    tensors between iterations. Pulling a disk-load before each candidate
    swap would dominate the runtime and defeat the whole point of caching
    the teacher logits.

    The student is responsible for being in eval mode and producing logits
    with vocab matching `manifest["vocab_size"]`.
    """
    import mlx.core as mx
    from tqdm import tqdm

    student_vocab = manifest["vocab_size"]
    num_batches = manifest["num_batches"]
    seq_len = manifest["max_seq_len"]
    # v2: manifest carries score_window = [start, end). v1 caches predate the
    # field; treat them as full-sequence (the v1 default behavior).
    window = manifest.get("score_window")
    if window is None:
        window = [0, seq_len]
    win_start, win_end = int(window[0]), int(window[1])

    agg = Aggregator()
    info(f"Student pass: {num_batches} batches, score window=[{win_start}, {win_end})")
    for i in tqdm(range(num_batches), desc="student", file=sys.stderr):
        shard = mx.load(str(cache_dir / f"batch-{i:05d}.safetensors"))
        token_ids = shard["token_ids"]  # (B, L) int32
        top_log_p = shard["top_k_log_softmax"]  # (B, L, K) bf16
        top_idx = shard["top_k_indices"]  # (B, L, K) int32
        attn_mask = shard["attention_mask"]  # (B, L) bool

        cache = _make_fresh_cache(student_model)
        student_logits = _extract_logits(student_model(token_ids, cache=cache))  # (B, L, V)
        if student_logits.shape[-1] != student_vocab:
            sys.exit(
                f"Student logits vocab ({student_logits.shape[-1]}) "
                f"differs from probed vocab ({student_vocab}); recompute."
            )

        kld_bt, student_top1, student_top5 = kld_from_topk(
            top_log_p,
            top_idx,
            student_logits,
            student_vocab,
        )
        teacher_top1 = top_idx[..., 0]  # (B, L)
        top1_match = student_top1 == teacher_top1
        # Top-5 recall: teacher's argmax appears in student's top-5 set.
        top5_match = mx.any(student_top5 == teacher_top1[..., None], axis=-1)

        # Mask invalid positions before harvesting; intersect with score window.
        m = attn_mask.astype(mx.bool_)
        L = m.shape[1]
        if win_start > 0 or win_end < L:
            pos_idx = mx.arange(L, dtype=mx.int32)
            window_mask = (pos_idx >= win_start) & (pos_idx < win_end)
            m = m & window_mask[None, :]
        # Force eval all per-batch tensors so the next iteration can reclaim memory.
        mx.eval(kld_bt, top1_match, top5_match, m)

        # Pull to host as numpy, masked.
        import numpy as np

        m_np = np.asarray(m)
        positions = np.broadcast_to(np.arange(m_np.shape[1], dtype=np.int32), m_np.shape).copy()
        agg.update(
            np.asarray(kld_bt)[m_np],
            np.asarray(top1_match)[m_np],
            np.asarray(top5_match)[m_np],
            positions[m_np],
        )

    # Quartiles of the actual scored window (caller-agnostic — same logic for
    # default and --long-context modes).
    span = max(win_end - win_start, 4)
    q = span // 4
    buckets: list[tuple[int, int | None]] = [
        (win_start, win_start + q),
        (win_start + q, win_start + 2 * q),
        (win_start + 2 * q, win_start + 3 * q),
        (win_start + 3 * q, None),
    ]
    return agg.finalize(position_buckets=buckets)


def student_pass(
    student_dir: Path,
    cache_dir: Path,
    manifest: dict,
    batch_size: int,
    student_vocab: int,
) -> dict:
    """Forward the student loaded from disk, compute online KLD against
    cached teacher logits.

    Thin wrapper around `score_loaded_student` that handles the disk load.
    `batch_size` and `student_vocab` are kept in the signature for backward
    compatibility with any callers that pre-date the manifest-derived
    refactor; the manifest is now the source of truth for both.
    """
    info(f"Loading student: {student_dir}")
    student_model, _student_config = _load_model(str(student_dir), lazy=True)
    student_model.eval()

    cache_batch_size = manifest["batch_size"]
    if cache_batch_size != batch_size:
        info(
            f"Note: cache was written at batch_size={cache_batch_size}; "
            f"student pass replays at the same batch size for shard alignment."
        )
    if student_vocab != manifest["vocab_size"]:
        info(
            f"Note: caller-supplied student_vocab={student_vocab} differs from "
            f"manifest vocab_size={manifest['vocab_size']}; using manifest value."
        )

    return score_loaded_student(student_model, cache_dir, manifest)


# ---------- output ----------


def render_markdown(report: dict) -> str:
    out: list[str] = []
    p = out.append

    teacher = report["teacher"]
    student = report["student"]
    cal = report["calibration"]
    kld = report["kld"]
    agreement = report["agreement"]

    teacher_label = teacher["path"]
    student_label = Path(student["path"]).name
    p(f"# KLD score — {student_label} vs {teacher_label}\n")

    p("## Calibration\n")
    p(f"- dataset: `{cal['corpus']}`")
    p(f"- num_samples: {cal['num_samples']}")
    p(f"- max_seq_len: {cal['max_seq_len']}")
    p(f"- seed: {cal['seed']}")
    p(f"- top_k cache: {report['cache']['top_k']}")
    p(f"- valid scoring positions: {report['tokens_scored']:,}\n")

    p("## Headline\n")
    p("| Metric | Value |")
    p("|---|---:|")
    p(f"| Mean KLD (nats) | {kld['mean']:.4f} |")
    p(f"| Median KLD | {kld['p50']:.4f} |")
    p(f"| P95 KLD | {kld['p95']:.4f} |")
    p(f"| P99 KLD | {kld['p99']:.4f} |")
    p(f"| P99.9 KLD | {kld['p999']:.4f} |")
    p(f"| Max KLD | {kld['max']:.4f} |")
    p(f"| Top-1 agreement | {agreement['top1'] * 100:.2f}% |")
    p(f"| Top-5 agreement | {agreement['top5'] * 100:.2f}% |\n")

    by_pos = report.get("by_position") or []
    if by_pos:
        p("## By position bucket\n")
        p("| Position range | Tokens scored | Mean KLD | P95 KLD | Top-1 agreement |")
        p("|---|---:|---:|---:|---:|")
        for row in by_pos:
            if row["tokens"] == 0:
                p(f"| {row['range']} | 0 | — | — | — |")
            else:
                p(
                    f"| {row['range']} | {row['tokens']:,} | "
                    f"{row['mean_kld']:.4f} | {row['p95_kld']:.4f} | "
                    f"{row['top1_agreement'] * 100:.2f}% |"
                )
        p("")

    p("## Recipe\n")
    recipe = report["recipe"]
    p("| Field | Value |")
    p("|---|---|")
    for k in RECIPE_KEYS:
        v = recipe.get(k)
        if v is None:
            cell = "—"
        elif isinstance(v, bool):
            cell = "true" if v else "false"
        else:
            cell = str(v)
        p(f"| `{k}` | {cell} |")
    p("")

    p("## Reproducibility\n")
    p(
        f"- teacher: `{teacher['path']}` "
        f"(revision {teacher.get('revision') or 'n/a'}, "
        f"{teacher['precision']})"
    )
    p(f"- student: `{student['path']}`")
    p(
        f"  ({student['size_bytes'] / 1e9:.2f} GB on disk, "
        f"effective_bpw={student['effective_bpw']:.3f})"
        if student.get("effective_bpw")
        else f"  ({student['size_bytes'] / 1e9:.2f} GB on disk)"
    )
    p(f"- cache: `{report['cache']['dir']}/`  ({report['cache']['status']})")
    p(f"- scorer revision: {report['scorer_version']}")
    p(f"- elapsed: {report['elapsed_seconds']:.1f}s ({report['elapsed_phase']})")
    p(f"- timestamp: {report['timestamp']}")
    p("")
    return "\n".join(out)


def build_locked_json(report: dict) -> dict:
    """Distill `report` down to the locked schema_version=1 keys.

    Extras (by_position, kld_histogram, cache info) are kept under non-locked
    keys for downstream tooling; the rollup script only plucks the locked set.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher": report["teacher"],
        "student": report["student"],
        "recipe": report["recipe"],
        "calibration": report["calibration"],
        "kld": report["kld"],
        "agreement": report["agreement"],
        "tokens_scored": report["tokens_scored"],
        "elapsed_seconds": report["elapsed_seconds"],
        "scorer_version": report["scorer_version"],
        "timestamp": report["timestamp"],
        # extras (non-locked, may be expanded over time)
        "by_position": report.get("by_position"),
        "kld_histogram": report.get("kld_histogram"),
        "cache": report.get("cache"),
    }


def validate_locked_schema(payload: dict) -> None:
    """Sanity-check that the locked keys exist + have the right primitive
    types. Called immediately before write so a broken scorer fails loud."""
    required = {
        "schema_version": int,
        "teacher": dict,
        "student": dict,
        "recipe": dict,
        "calibration": dict,
        "kld": dict,
        "agreement": dict,
        "tokens_scored": int,
        "elapsed_seconds": (int, float),
        "scorer_version": str,
        "timestamp": str,
    }
    for k, t in required.items():
        if k not in payload:
            raise AssertionError(f"locked schema: missing key {k!r}")
        if not isinstance(payload[k], t):
            raise AssertionError(f"locked schema: {k!r} should be {t}, got {type(payload[k])}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise AssertionError(f"schema_version must be {SCHEMA_VERSION}")
    for k in ("mean", "p50", "p95", "p99", "p999", "max"):
        if k not in payload["kld"]:
            raise AssertionError(f"locked schema: kld.{k} missing")
    for k in ("top1", "top5"):
        if k not in payload["agreement"]:
            raise AssertionError(f"locked schema: agreement.{k} missing")
    for k in RECIPE_KEYS:
        if k not in payload["recipe"]:
            raise AssertionError(f"locked schema: recipe.{k} missing")
    for k in ("corpus", "num_samples", "max_seq_len", "seed"):
        if k not in payload["calibration"]:
            raise AssertionError(f"locked schema: calibration.{k} missing")


# ---------- K-floor sweep (P1.0 in plans/the-best-dynamic-gguf-binary-flame.md) ----------


def _kld_floor_sweep(
    teacher_path: str,
    *,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    dataset_name: str,
    k_values: list[int],
) -> int:
    """Empirical reconstruction-floor measurement for the top-K KLD approximation.

    The top-K-with-uniform-tail reconstruction in `kld_from_topk` is exact when
    P_teacher's tail is genuinely uniform; on real models it has a residual
    (bias) that depends on tail concentration. We measure that residual by
    running KLD with student==teacher (same forward, same logits) at multiple
    K values from a single forward pass per batch — KLD must be ~0 in the
    limit, so the residual at each K *is* the floor.

    Decision rule: the default K for `score-mlx-kld.py` is the smallest value
    in `k_values` for which the floor is below ~0.001 nats on the larger-vocab
    anchor. See plan P1.0 for the rationale.
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm.utils import load_tokenizer
    from tqdm import tqdm

    info(f"K-floor sweep on {teacher_path}")
    info(f"  dataset={dataset_name}, samples={num_samples}, seq={max_seq_len}, seed={seed}")

    tokenizer = load_tokenizer(teacher_path)
    tokens = load_calibration_tokens(
        tokenizer,
        dataset_name,
        num_samples,
        max_seq_len,
        seed,
    )
    info(f"  loaded {tokens.shape[0]} sequences × {tokens.shape[1]} tokens")

    info("  loading model")
    model, _config = _load_model(teacher_path, lazy=False)
    model.eval()

    floor_sum: dict[int, float] = {K: 0.0 for K in k_values}
    floor_max: dict[int, float] = {K: 0.0 for K in k_values}
    floor_count = 0
    V: int | None = None

    for b in tqdm(range(tokens.shape[0]), desc="floor sweep", file=sys.stderr):
        batch = tokens[b : b + 1]
        cache = _make_fresh_cache(model)
        logits = _extract_logits(model(batch, cache=cache))
        logits_fp32 = logits.astype(mx.float32)

        if V is None:
            V = int(logits_fp32.shape[-1])
            # Filter sweep K values strictly < V (kld_from_topk asserts V-K > 0).
            k_values = [K for K in k_values if 0 < K < V]
            info(f"  V={V:,}, sweep K values = {k_values}")

        log_softmax = logits_fp32 - mx.logsumexp(logits_fp32, axis=-1, keepdims=True)

        for K in k_values:
            idx_part = mx.argpartition(log_softmax, kth=-K, axis=-1)[..., -K:]
            vals = mx.take_along_axis(log_softmax, idx_part, axis=-1)
            order = mx.argsort(-vals, axis=-1)
            idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
            topk_sorted = mx.take_along_axis(vals, order, axis=-1)

            kld, _t1, _t5 = kld_from_topk(topk_sorted, idx_sorted, logits_fp32, V)
            kld_np = np.maximum(np.asarray(kld).ravel(), 0.0)
            floor_sum[K] += float(kld_np.sum())
            floor_max[K] = max(floor_max[K], float(kld_np.max()))

        floor_count += int(batch.shape[1])
        del logits, logits_fp32, log_softmax
        mx.eval(mx.zeros(1))

    print()
    print(f"=== K-floor sweep on {teacher_path} ===")
    print(
        f"V={V:,}, dataset={dataset_name}, samples={num_samples}, seq={max_seq_len}, seed={seed}"
    )
    print(f"{floor_count:,} tokens scored")
    print()
    print(f"  {'K':>10}  {'mean floor (nats)':>20}  {'p100 (nats)':>14}")
    print(f"  {'-' * 10}  {'-' * 20}  {'-' * 14}")
    for K in k_values:
        mean_floor = floor_sum[K] / max(floor_count, 1)
        K_disp = f"{K:,}"
        if V is not None and K == V // 2:
            K_disp = f"{K:,} (V/2)"
        print(f"  {K_disp:>10}  {mean_floor:>20.6f}  {floor_max[K]:>14.6f}")
    print()
    return 0


# ---------- self-test ----------


def _self_test() -> int:
    """Tier-1 unit tests. Returns process exit code."""
    import mlx.core as mx
    import numpy as np

    failures: list[str] = []

    # ---- KLD math: uniform vs uniform == 0 ----
    V = 1024
    K = 16
    # build synthetic teacher = uniform; student logits = constant (also uniform)
    teacher_log = mx.full((1, 1, V), -math.log(V), dtype=mx.float32)
    # top-K: take any K positions (uniform → all equal); their indices are arbitrary.
    idx = mx.arange(K, dtype=mx.int32).reshape(1, 1, K)
    topk = mx.take_along_axis(teacher_log, idx, axis=-1)
    student_logits = mx.zeros((1, 1, V), dtype=mx.float32)  # uniform
    kld, t1, t5 = kld_from_topk(topk, idx, student_logits, V)
    val = float(np.asarray(kld).item())
    if abs(val) > 1e-4:
        failures.append(f"uniform-vs-uniform KLD = {val} (expected ~0)")

    # ---- KLD math: peaked vs uniform > 0 (and large) ----
    np_logits = np.full((V,), -10.0, dtype=np.float32)
    np_logits[0] = 5.0  # very peaked
    teacher_logits = mx.array(np_logits.reshape(1, 1, V))
    teacher_log = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    # build top-K from this teacher
    idx_part = mx.argpartition(teacher_log, kth=-K, axis=-1)[..., -K:]
    vals = mx.take_along_axis(teacher_log, idx_part, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
    topk_sorted = mx.take_along_axis(vals, order, axis=-1)
    student_logits = mx.zeros((1, 1, V), dtype=mx.float32)
    kld, t1, t5 = kld_from_topk(topk_sorted, idx_sorted, student_logits, V)
    val_peaked = float(np.asarray(kld).item())
    if val_peaked < 1.0:
        failures.append(f"peaked-vs-uniform KLD = {val_peaked} (expected > 1)")

    # ---- KLD asymmetry: KL(P||Q) != KL(Q||P) ----
    # P peaked at 0; Q peaked at 1 (but somewhat). Neither is uniform.
    np_p = np.full((V,), -10.0, dtype=np.float32)
    np_p[0] = 5.0
    np_q = np.full((V,), -8.0, dtype=np.float32)
    np_q[1] = 4.0
    p_logits = mx.array(np_p.reshape(1, 1, V))
    q_logits = mx.array(np_q.reshape(1, 1, V))
    p_log = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    q_log = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)

    def topk_of(log_p):
        idx = mx.argpartition(log_p, kth=-K, axis=-1)[..., -K:]
        v = mx.take_along_axis(log_p, idx, axis=-1)
        order = mx.argsort(-v, axis=-1)
        return (
            mx.take_along_axis(idx, order, axis=-1).astype(mx.int32),
            mx.take_along_axis(v, order, axis=-1),
        )

    p_idx, p_topk = topk_of(p_log)
    q_idx, q_topk = topk_of(q_log)
    kl_pq, _, _ = kld_from_topk(p_topk, p_idx, q_logits, V)
    kl_qp, _, _ = kld_from_topk(q_topk, q_idx, p_logits, V)
    a = float(np.asarray(kl_pq).item())
    b = float(np.asarray(kl_qp).item())
    if abs(a - b) < 1e-3:
        failures.append(f"KLD symmetric? KL(P||Q)={a}, KL(Q||P)={b}")

    # ---- top-K reconstruction round-trip ----
    # The uniform-residual approximation is exact when the teacher's tail is
    # uniform. Build that teacher (peak + flat tail), confirm K >= 128 gives
    # essentially zero error, and verify the approximation converges
    # monotonically with K on a realistic non-uniform-tail teacher.
    raw = np.full((V,), -10.0, dtype=np.float32)  # flat tail
    raw[:8] = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)  # head
    teacher_logits = mx.array(raw.reshape(1, 1, V))
    teacher_log_full = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    student_logits = mx.zeros((1, 1, V), dtype=mx.float32)
    student_log_full = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
    exact = float(
        np.asarray(
            mx.sum(mx.exp(teacher_log_full) * (teacher_log_full - student_log_full), axis=-1)
        ).item()
    )
    for K_test in (16, 64, 128):
        idx_part = mx.argpartition(teacher_log_full, kth=-K_test, axis=-1)[..., -K_test:]
        vals = mx.take_along_axis(teacher_log_full, idx_part, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
        topk_sorted = mx.take_along_axis(vals, order, axis=-1)
        approx, _, _ = kld_from_topk(topk_sorted, idx_sorted, student_logits, V)
        approx_v = float(np.asarray(approx).item())
        rel = abs(approx_v - exact) / max(abs(exact), 1e-9)
        # Flat-tail teacher → uniform residual is exact (modulo fp32 rounding).
        if rel > 5e-3:
            failures.append(
                f"top-K reconstruction K={K_test} on flat-tail teacher: "
                f"rel error {rel:.4%} (>0.5%)"
            )
    # Monotonic convergence on a non-flat-tail (non-trivially structured) teacher.
    np.random.seed(0)
    raw2 = np.random.randn(V).astype(np.float32) * 1.5
    raw2[0] = 6.0
    teacher2 = mx.array(raw2.reshape(1, 1, V))
    teacher2_log = teacher2 - mx.logsumexp(teacher2, axis=-1, keepdims=True)
    exact2 = float(
        np.asarray(
            mx.sum(mx.exp(teacher2_log) * (teacher2_log - student_log_full), axis=-1)
        ).item()
    )
    last_err = float("inf")
    for K_test in (16, 64, 256, 512):
        idx_part = mx.argpartition(teacher2_log, kth=-K_test, axis=-1)[..., -K_test:]
        vals = mx.take_along_axis(teacher2_log, idx_part, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
        topk_sorted = mx.take_along_axis(vals, order, axis=-1)
        approx, _, _ = kld_from_topk(topk_sorted, idx_sorted, student_logits, V)
        rel = abs(float(np.asarray(approx).item()) - exact2) / max(abs(exact2), 1e-9)
        if rel > last_err + 1e-6:
            failures.append(
                f"top-K reconstruction non-monotonic at K={K_test}: "
                f"rel {rel:.4%} > prior {last_err:.4%}"
            )
        last_err = rel

    # ---- tokenizer-parity bail: differing hashes produce different strings ----
    h1 = hashlib.sha256(b"a").hexdigest()
    h2 = hashlib.sha256(b"b").hexdigest()
    if h1 == h2:
        failures.append("hashlib parity: identical hashes for distinct inputs")

    # tokenizer_hash should reflect functional differences (vocab divergence
    # or special-token id changes), not on-disk file formatting.
    class _FakeTok:
        def __init__(self, vocab, bos=None, eos=None, pad=None, unk=None):
            self._vocab = vocab
            self.vocab_size = len(vocab)
            self.bos_token_id = bos
            self.eos_token_id = eos
            self.pad_token_id = pad
            self.unk_token_id = unk

        def __len__(self):
            return len(self._vocab)

        def convert_ids_to_tokens(self, i):
            return self._vocab[i]

    a = _FakeTok(["<a>", "<b>", "<c>"], eos=2)
    a_dup = _FakeTok(["<a>", "<b>", "<c>"], eos=2)  # functionally identical
    b = _FakeTok(["<a>", "<b>", "<X>"], eos=2)  # vocab differs
    c = _FakeTok(["<a>", "<b>", "<c>"], eos=1)  # special-token id differs
    if tokenizer_hash(a) != tokenizer_hash(a_dup):
        failures.append("tokenizer_hash unstable across functionally identical inputs")
    if tokenizer_hash(a) == tokenizer_hash(b):
        failures.append("tokenizer_hash collides on differing vocab")
    if tokenizer_hash(a) == tokenizer_hash(c):
        failures.append("tokenizer_hash collides on differing special-token ids")

    # ---- locked-schema validator catches missing keys ----
    try:
        validate_locked_schema({})
    except AssertionError:
        pass
    else:
        failures.append("validate_locked_schema accepted empty payload")

    # ---- recipe loader: schema-additive keys default to None ----
    # An older or third-party recipe JSON that pre-dates a key addition (or
    # comes from a tool that doesn't write the full schema) must still load
    # cleanly, with the absent keys set to None. This is what enables
    # adding new flags without bumping schema_version.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Stub recipe written by a pre-rename attn-protect-quantize using
        # the legacy `no_skip_*` ablation keys and missing protect_vlm. The
        # loader must (a) silently migrate `no_skip_*` → `quantize_*`, and
        # (b) still produce all RECIPE_KEYS in the result with absent ones
        # set to None for the locked-schema validator.
        partial = {
            "tool": "attn-protect-quantize",
            "tool_version": "abc1234",
            "bits": 4,
            "group_size": 64,
            "attn_protect_mode": "bf16",
            "with_dwq": False,
            "floor_tied_embed": False,
            "no_skip_linear_attn": True,
            "no_skip_attn_out": False,
            "no_attn_floor": False,
            "no_lm_head_floor": False,
            # protect_vlm intentionally omitted
        }
        (td_path / "attn-protect-recipe.json").write_text(json.dumps(partial))
        loaded = load_recipe(td_path, fallback_meta={"default_bits": 4, "default_group_size": 64})
        if "protect_vlm" not in loaded:
            failures.append("load_recipe: protect_vlm key missing from result")
        elif loaded["protect_vlm"] is not None:
            failures.append(
                "load_recipe: protect_vlm should be None"
                " when absent in JSON, got"
                f" {loaded['protect_vlm']!r}"
            )
        if loaded.get("quantize_linear_attn") is not True:
            failures.append(
                f"load_recipe: legacy no_skip_linear_attn=True should migrate to "
                f"quantize_linear_attn=True, got {loaded.get('quantize_linear_attn')!r}"
            )
        if loaded.get("quantize_attn_out") is not False:
            failures.append(
                f"load_recipe: legacy no_skip_attn_out=False should migrate to "
                f"quantize_attn_out=False, got {loaded.get('quantize_attn_out')!r}"
            )
        # Also: the third-party "unknown" tool path with neither a recipe
        # file nor protect_vlm should still produce a usable dict.
    with tempfile.TemporaryDirectory() as td:
        loaded = load_recipe(Path(td), fallback_meta={"default_bits": 4, "default_group_size": 64})
        if loaded.get("tool") != "unknown":
            failures.append(
                "load_recipe: missing recipe should set"
                f" tool='unknown', got {loaded.get('tool')!r}"
            )
        for k in RECIPE_KEYS:
            if k not in loaded:
                failures.append(f"load_recipe: third-party stub missing key {k!r}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("All self-tests passed.", file=sys.stderr)
    return 0


# ---------- main ----------


def _scorer_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mqt-score-kld",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "teacher", nargs="?", help="HF id or local path to the (full-precision) teacher."
    )
    p.add_argument("student", nargs="?", type=Path, help="Local path to the quantized student.")
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HF dataset (default: {DEFAULT_DATASET}). Accepts "
        "`path:name` for HF subset configs (e.g. "
        "`HuggingFaceFW/fineweb-edu:sample-10BT`). "
        "Chat-format corpora (with a `messages` column, e.g. "
        "`allenai/tulu-3-sft-mixture`) auto-render through the "
        "teacher's chat template.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of sequences to score (default: %(default)s)",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help=f"Tokens per sequence (default: {DEFAULT_MAX_SEQ_LEN}; or 2048 with --long-context)",
    )
    p.add_argument(
        "--long-context",
        action="store_true",
        help="Research mode: max_seq_len=2048 with full-sequence "
        "scoring (the v1 protocol). Use to study how KLD "
        "varies with context length. Default mode matches "
        "llama.cpp's --kl-divergence (n_ctx=512, second-half "
        "scoring) for cross-tool comparability.",
    )
    p.add_argument(
        "--score-window",
        default=None,
        help="Override the score window as `start:end` "
        "(default: second half under v2, full sequence under "
        "--long-context). Only the positions in [start, end) "
        "contribute to the headline KLD.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-K to cache per teacher position (default: %(default)s)",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampling seed for the dataset slice (default: %(default)s)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Teacher-logits cache root (default: {DEFAULT_CACHE_DIR})",
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force a teacher pass even if a cached one exists.",
    )
    p.add_argument(
        "--allow-tokenizer-mismatch",
        action="store_true",
        help="Bypass the strict tokenizer-hash parity check between "
        "teacher and student. Use ONLY when you have separately "
        "verified that encoding (text → token ids) is identical "
        "between the two tokenizers — e.g. when the only "
        "difference is a non-encoding-affecting field like "
        "pad_token_id or decoder cosmetic config. Example: "
        "scoring an Unsloth UD-MLX-quantized student against "
        "the official upstream teacher when Unsloth changed "
        "pad_token_id (vocab + merges + special-token strings "
        "remain identical, encoding parity holds, KLD is "
        "well-defined). Vocab size mismatch is still fatal.",
    )
    p.add_argument("--md", type=Path, default=None, help="Markdown report path (default: stdout)")
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSON dump path (default: <student>/kld-vs-<teacher>.json)",
    )
    p.add_argument(
        "--teacher-precision",
        default="bfloat16",
        help="Teacher precision label for the report (default: %(default)s)",
    )
    p.add_argument(
        "--self-test", action="store_true", help="Run unit tests and exit. Skips all CLI args."
    )
    p.add_argument(
        "--kld-floor-sweep",
        metavar="MODEL",
        default=None,
        help="Empirical reconstruction-floor measurement: run "
        "teacher==student KLD at multiple K values from a single "
        "forward pass per batch, on the given model. Emits a "
        "floor(K) table. Honors --num-samples, --max-seq-len, "
        "--seed, --dataset. Skips all other positional args.",
    )
    p.add_argument(
        "--floor-sweep-k",
        type=str,
        default="128,512,2048,8192",
        help="Comma-separated K values for --kld-floor-sweep "
        "(default: %(default)s). Values >= V are dropped.",
    )
    p.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="Score a GGUF K-quant student instead of an MLX "
        "checkpoint dir. Mutually exclusive with the "
        "positional `student` arg. Config and tokenizer are "
        "synthesized from GGUF metadata; loaded via "
        "run_gguf_kquant.load_kquant_model.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.self_test:
        return _self_test()

    if args.kld_floor_sweep:
        try:
            k_values = sorted({int(x) for x in args.floor_sweep_k.split(",") if x.strip()})
        except ValueError:
            sys.exit(f"--floor-sweep-k must be comma-separated ints, got: {args.floor_sweep_k!r}")
        if not k_values:
            sys.exit("--floor-sweep-k yielded no values")
        return _kld_floor_sweep(
            args.kld_floor_sweep,
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            dataset_name=args.dataset,
            k_values=k_values,
        )

    # GGUF student path: mutually exclusive with positional `student`.
    # Config and tokenizer are derived from the GGUF itself.
    using_gguf = args.gguf is not None
    if using_gguf:
        if args.student is not None:
            sys.exit("--gguf and the positional `student` arg are mutually exclusive.")
        if not args.gguf.exists():
            sys.exit(f"--gguf path does not exist: {args.gguf}")
        if not args.teacher:
            sys.exit("usage: score-mlx-kld.py <teacher> --gguf <path> [options]")
    else:
        if not args.teacher or not args.student:
            sys.exit("usage: score-mlx-kld.py <teacher> <student> [options]  (or --self-test)")

    if using_gguf:
        student_dir = None
    else:
        student_dir = args.student
        if not student_dir.is_dir():
            sys.exit(f"Student is not a directory: {student_dir}")

    # Resolve protocol-mode defaults: long-context flips to v1-style (n_ctx=2048,
    # full sequence); default mode is llama.cpp-comparable (n_ctx=512, second
    # half). --max-seq-len and --score-window override either default.
    if args.max_seq_len is None:
        max_seq_len = 2048 if args.long_context else DEFAULT_MAX_SEQ_LEN
    else:
        max_seq_len = args.max_seq_len
    if args.score_window is not None:
        try:
            sa, sb = (int(x) for x in args.score_window.split(":"))
        except ValueError:
            sys.exit(f"--score-window must be `start:end`, got: {args.score_window!r}")
        score_window = (sa, sb)
    elif args.long_context:
        score_window = (0, max_seq_len)
    else:
        score_window = (max_seq_len // 2, max_seq_len)
    if not (0 <= score_window[0] < score_window[1] <= max_seq_len):
        sys.exit(
            f"--score-window {score_window} must satisfy "
            f"0 <= start < end <= max_seq_len ({max_seq_len})"
        )
    args.max_seq_len = max_seq_len  # downstream code reads off args
    info(
        f"Protocol: max_seq_len={max_seq_len}, top_k={args.top_k}, "
        f"score_window={list(score_window)}, dataset={args.dataset}"
    )

    from mlx_lm.utils import load_tokenizer

    info(f"Loading teacher tokenizer: {args.teacher}")
    teacher_tokenizer = load_tokenizer(args.teacher)
    if using_gguf:
        # Synthesize tokenizer straight from GGUF metadata — same path
        # load_kquant_model uses internally when hf_source is None.
        from gguf import GGUFReader

        from mlx_quant_tools.gguf_name_remap import detect_arch
        from mlx_quant_tools.gguf_tokenizer import load_tokenizer_from_gguf

        reader = GGUFReader(str(args.gguf), "r")
        gguf_arch = detect_arch(reader)
        info(f"Loading student tokenizer from GGUF: {args.gguf} (arch={gguf_arch})")
        student_tokenizer = load_tokenizer_from_gguf(reader, gguf_arch)
    else:
        info(f"Loading student tokenizer: {student_dir}")
        student_tokenizer = load_tokenizer(str(student_dir))

    teacher_tok_hash = tokenizer_hash(teacher_tokenizer)
    student_tok_hash = tokenizer_hash(student_tokenizer)
    if teacher_tok_hash != student_tok_hash:
        if args.allow_tokenizer_mismatch:
            info(
                "TOKENIZER HASH MISMATCH (bypassed via --allow-tokenizer-mismatch)\n"
                f"  teacher hash : {teacher_tok_hash}\n"
                f"  student hash : {student_tok_hash}\n"
                "  Caller asserts encoding parity; KLD will be computed against "
                "the teacher's tokenization. If encoding actually differs, KLD "
                "numbers are meaningless."
            )
        else:
            sys.exit(
                "TOKENIZER PARITY FAILURE\n"
                f"  teacher hash : {teacher_tok_hash}\n"
                f"  student hash : {student_tok_hash}\n"
                "Teacher and student must share a tokenizer for KLD to be defined.\n"
                "If you have separately verified encoding parity (e.g. only "
                "pad_token_id or decoder cosmetic differs), pass "
                "--allow-tokenizer-mismatch to bypass this check."
            )
    if student_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        sys.exit(
            "Tokenizer vocab_size mismatch: "
            f"teacher={teacher_tokenizer.vocab_size}, student={student_tokenizer.vocab_size}"
        )

    # Tokenize the calibration corpus *once* (must use the shared tokenizer).
    tokens = load_calibration_tokens(
        student_tokenizer,
        args.dataset,
        args.num_samples,
        args.max_seq_len,
        args.seed,
    )
    info(f"Calibration: {tokens.shape[0]} sequences × {tokens.shape[1]} tokens")

    # Cache key folds in everything that affects the teacher logits we'd cache.
    key = cache_key(
        args.teacher,
        args.dataset,
        args.num_samples,
        args.max_seq_len,
        args.seed,
        args.top_k,
    )
    cache_dir: Path = args.cache_dir / key
    expected_batches = (tokens.shape[0] + args.batch_size - 1) // args.batch_size

    elapsed_phase = "cold (teacher + student)"
    teacher_pass_seconds = 0.0
    cache_status = "MISS"

    valid, reason = cache_is_valid(cache_dir, expected_batches)
    if valid and not args.rebuild_cache:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        # Reject if tokenizer hash diverges (cache was built with a different
        # tokenizer — would yield meaningless KLD against this student).
        # Under --allow-tokenizer-mismatch, the cache is keyed to the teacher's
        # hash; we accept it iff it matches the teacher (the calibration-side
        # tokenization the cache was built against).
        valid_hashes = (
            {None, student_tok_hash, teacher_tok_hash}
            if args.allow_tokenizer_mismatch
            else {None, student_tok_hash}
        )
        if manifest.get("tokenizer_hash") not in valid_hashes:
            sys.exit(
                "Cache tokenizer_hash mismatch — cache was built against a "
                f"different tokenizer ({manifest['tokenizer_hash']} != "
                f"{student_tok_hash}). Use --rebuild-cache or --cache-dir."
            )
        cache_status = "HIT"
        elapsed_phase = "warm (student only; cache HIT)"
        info(f"Teacher cache HIT: {cache_dir}")
        # Overlay the run-time score window — the teacher cache contents are
        # invariant to the score window (it's a scoring-policy knob, not a
        # data-content knob), so we don't rebuild for window changes.
        manifest = dict(manifest)
        manifest["score_window"] = [int(score_window[0]), int(score_window[1])]
    else:
        if valid and args.rebuild_cache:
            info(f"--rebuild-cache: rebuilding {cache_dir}")
        else:
            info(f"Teacher cache MISS ({reason})")
        # Wipe stale shards so a smaller new run doesn't read old ones.
        if cache_dir.exists():
            for f in cache_dir.glob("batch-*.safetensors"):
                f.unlink()
            (cache_dir / "manifest.json").unlink(missing_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        manifest = teacher_pass(
            args.teacher,
            tokens,
            args.batch_size,
            args.top_k,
            cache_dir,
            args.dataset,
            args.num_samples,
            args.max_seq_len,
            args.seed,
            student_tok_hash,
            vocab_size=student_tokenizer.vocab_size,
            score_window=score_window,
        )
        teacher_pass_seconds = time.time() - t0

    # Student pass + online KLD
    t1 = time.time()
    if using_gguf:
        from mlx_quant_tools.gguf_runtime import load_kquant_model

        student_model, _student_config, _stu_tok = load_kquant_model(
            str(args.gguf),
        )
        student_model.eval()
        metrics = score_loaded_student(student_model, cache_dir, manifest)
    else:
        metrics = student_pass(
            student_dir,
            cache_dir,
            manifest,
            batch_size=args.batch_size,
            student_vocab=manifest["vocab_size"],
        )
    student_pass_seconds = time.time() - t1
    elapsed_seconds = teacher_pass_seconds + student_pass_seconds

    # Assemble report
    if using_gguf:
        student_meta = measure_gguf_student(args.gguf)
        recipe = recipe_for_gguf_student()
    else:
        student_meta = measure_student(student_dir)
        recipe = load_recipe(student_dir, student_meta)
    report = {
        "teacher": {
            "path": args.teacher,
            "revision": manifest.get("teacher_revision") or hf_revision_of(args.teacher),
            "precision": args.teacher_precision,
        },
        "student": {
            "path": str(args.gguf if using_gguf else student_dir),
            "size_bytes": student_meta["size_bytes"],
            "effective_bpw": student_meta["effective_bpw"],
        },
        "recipe": recipe,
        "calibration": {
            "corpus": args.dataset,
            "num_samples": args.num_samples,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "top_k": args.top_k,
            "score_window": list(score_window),
            "long_context": bool(args.long_context),
        },
        "kld": metrics["kld"],
        "agreement": metrics["agreement"],
        "tokens_scored": metrics["tokens_scored"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_phase": elapsed_phase,
        "scorer_version": _scorer_git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "by_position": metrics.get("by_position"),
        "kld_histogram": metrics.get("kld_histogram"),
        "cache": {
            "dir": str(cache_dir),
            "status": cache_status,
            "top_k": args.top_k,
        },
    }

    md = render_markdown(report)
    if args.md:
        args.md.write_text(md)
        info(f"Markdown report: {args.md}")
    else:
        print(md)

    # Locked-schema JSON
    payload = build_locked_json(report)
    validate_locked_schema(payload)

    json_path = args.json
    if json_path is None:
        teacher_basename = args.teacher.rstrip("/").split("/")[-1].lower()
        if using_gguf:
            stem = args.gguf.stem
            json_path = args.gguf.parent / f"{stem}-kld-vs-{teacher_basename}.json"
        else:
            json_path = student_dir / f"kld-vs-{teacher_basename}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    info(f"JSON dump (schema_version={SCHEMA_VERSION}): {json_path}")
    info(f"Done. Cache={cache_status}; elapsed={elapsed_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
