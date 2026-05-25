"""Build a `PreTrainedTokenizerFast` from GGUF tokenizer metadata.

`load_tokenizer_from_gguf(reader, arch)` returns a fully functional fast
tokenizer (BPE) wrapped in `PreTrainedTokenizerFast`, with chat template,
special tokens, and `add_bos_token` honored — the equivalent of running
`AutoTokenizer.from_pretrained(<hf_source>)` but sourced entirely from
GGUF KV metadata.

Two construction paths, both BPE (verified empirically against the HF
tokenizer.json for each family — the design doc's Unigram-for-gemma4
prescription was wrong; gemma-4 also exports BPE with byte_fallback):

  - SPM-style BPE      (gemma4, gemma3, llama-with-no-pre):
      Replace " " → ▁ in the normalizer; pre-tokenize by splitting on " "
      (merged_with_previous); decoder reverses ▁ → " " and applies byte
      fallback. byte_fallback=True in the BPE model.

  - ByteLevel BPE      (qwen35, qwen3, llama3, gpt2):
      NFC normalizer; pre-tokenize via GPT-4-style regex split + ByteLevel;
      ByteLevel decoder + post-processor. byte_fallback=False.

No transformers monkey-patching, no vendored converters. Built directly
from `tokenizers` primitives.
"""

from __future__ import annotations

import sys

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)
from transformers import PreTrainedTokenizerFast

# GPT-4-style word-split pattern used by Qwen2/3/3.5 and Llama3 BPE
# pre-tokenizers. Matches contractions, letters, digits, punctuation runs,
# and whitespace separately so byte-level encoding can isolate words.
QWEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    r"[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

# pre-tokenizer hint string → algorithm bucket. Hints come from the
# `tokenizer.ggml.pre` GGUF KV (set by llama.cpp's
# convert_hf_to_gguf.py based on the source tokenizer's metadata).
_BYTELEVEL_PRES = {"qwen2", "qwen3", "qwen35", "qwen35moe", "llama-bpe", "llama3"}


# ---------------------------------------------------------------------------
# GGUF metadata helpers
# ---------------------------------------------------------------------------


def _read_string(reader, key: str) -> str | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def _read_int(reader, key: str) -> int | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return int(f.parts[f.data[0]][0])


def _read_str_array(reader, key: str) -> list[str] | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return [bytes(f.parts[i]).decode("utf-8", errors="replace") for i in f.data]


def _read_int_array(reader, key: str) -> list[int] | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return [int(f.parts[i][0]) for i in f.data]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_tokenizer_from_gguf(reader, arch: str) -> PreTrainedTokenizerFast:
    """Build a HF fast tokenizer from GGUF tokenizer metadata."""
    tokens = _read_str_array(reader, "tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError("GGUF missing tokenizer.ggml.tokens")
    raw_merges = _read_str_array(reader, "tokenizer.ggml.merges")
    token_types = _read_int_array(reader, "tokenizer.ggml.token_type")

    model_id = _read_string(reader, "tokenizer.ggml.model") or ""
    pre_id = _read_string(reader, "tokenizer.ggml.pre") or ""

    bos_id = _read_int(reader, "tokenizer.ggml.bos_token_id")
    eos_id = _read_int(reader, "tokenizer.ggml.eos_token_id")
    pad_id = _read_int(reader, "tokenizer.ggml.padding_token_id")
    unk_id = _read_int(reader, "tokenizer.ggml.unknown_token_id")
    # Intentionally NOT reading tokenizer.ggml.add_bos_token /
    # _add_eos_token: those reflect llama.cpp's raw-prompt convention
    # (auto-prepend BOS) while HF tokenizers expect the chat template to
    # handle BOS for instruct models. Leaving the flags at the
    # PreTrainedTokenizerFast default (False) keeps behavior parity with
    # `--hf-source`-loaded tokenizers, including no double-BOS when a
    # chat template is applied.

    chat_template = _read_string(reader, "tokenizer.chat_template")

    style = _classify(model_id, pre_id, has_merges=bool(raw_merges))
    if style == "bytelevel":
        tok = _build_bytelevel_bpe(tokens, raw_merges)
    elif style == "spm":
        unk_str = tokens[unk_id] if unk_id is not None else None
        tok = _build_spm_bpe(tokens, raw_merges, unk_str=unk_str)
    else:
        raise NotImplementedError(
            f"unsupported tokenizer style for arch={arch!r} model={model_id!r} pre={pre_id!r}"
        )

    # Special tokens: registered as AddedToken so the fast tokenizer treats
    # them as atomic during encode and emits them verbatim during decode.
    bos_str = tokens[bos_id] if bos_id is not None else None
    eos_str = tokens[eos_id] if eos_id is not None else None
    pad_str = tokens[pad_id] if pad_id is not None else None

    special_tokens: list[str] = []
    seen: set[str] = set()
    for t in (bos_str, eos_str, pad_str):
        if t is not None and t not in seen:
            special_tokens.append(t)
            seen.add(t)
    # GGUF type=3 = control tokens. Add them as specials so they round-trip
    # through encode/decode without splitting.
    if token_types is not None:
        for idx, ttype in enumerate(token_types):
            if ttype == 3 and tokens[idx] not in seen:
                special_tokens.append(tokens[idx])
                seen.add(tokens[idx])
    if special_tokens:
        tok.add_special_tokens(
            [AddedToken(s, normalized=False, special=True) for s in special_tokens]
        )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=bos_str,
        eos_token=eos_str,
        pad_token=pad_str,
        unk_token=tokens[unk_id] if unk_id is not None else None,
        chat_template=chat_template,
    )

    _self_test_roundtrip(fast)

    extra_eos = _infer_turn_end_eos(fast, eos_id, tokens, token_types)
    all_eos = [eos_id] + extra_eos if eos_id is not None else extra_eos
    fast._gguf_eos_token_ids = all_eos

    eos_str = f"eos={eos_id}"
    if extra_eos:
        extra_strs = [f"{tid}={tokens[tid]!r}" for tid in extra_eos]
        eos_str += f" extra_eos=[{', '.join(extra_strs)}]"

    print(
        f"[tokenizer] built from GGUF: vocab={len(tokens)} "
        f"style={style} model={model_id!r} pre={pre_id!r} "
        f"bos={bos_id} {eos_str} "
        f"chat_template={'yes' if chat_template else 'no'}"
        + (f" ({len(chat_template)} chars)" if chat_template else ""),
        file=sys.stderr,
    )
    return fast


# ---------------------------------------------------------------------------
# Style classification
# ---------------------------------------------------------------------------


def _classify(model_id: str, pre_id: str, *, has_merges: bool) -> str:
    """Pick a construction style from GGUF tokenizer.ggml.{model,pre}.

    GGUF's `model` field is unreliable across families (gemma4 advertises
    "gemma4" but the on-disk HF tokenizer is BPE; "gpt2" is a generic BPE
    marker shared by many byte-level BPEs). Combine with `pre` and the
    presence of merges to pick the right path.
    """
    if not has_merges:
        # No merges → would need Unigram, which we don't have a target for
        # in the supported arch list. Surface as unsupported.
        return "unsupported"
    if pre_id in _BYTELEVEL_PRES or model_id == "gpt2":
        return "bytelevel"
    if model_id in ("gemma4", "gemma3", "gemma3_text", "llama"):
        return "spm"
    # Fallback: if the pre is unspecified but merges exist, treat as SPM
    # BPE. ByteLevel mis-decoding is loud (mojibake), so this default is
    # safer than guessing ByteLevel.
    return "spm"


# ---------------------------------------------------------------------------
# ByteLevel BPE (qwen35, qwen3, llama3, gpt2)
# ---------------------------------------------------------------------------


def _parse_merges(raw_merges: list[str]) -> list[tuple[str, str]]:
    """Split each merge string at its first ASCII space.

    GGUF stores merges as "<token1> <token2>" (a single literal space
    separator). HF tokenizer expects a list of (token1, token2) tuples.
    Tokens themselves never contain literal ASCII space — gemma4 uses ▁,
    qwen35 uses Ġ — so first-space split is unambiguous.
    """
    out: list[tuple[str, str]] = []
    for m in raw_merges:
        sep = m.find(" ")
        if sep < 0:
            raise ValueError(f"malformed merge (no space): {m!r}")
        out.append((m[:sep], m[sep + 1 :]))
    return out


def _build_bytelevel_bpe(tokens: list[str], raw_merges: list[str]) -> Tokenizer:
    vocab = {tok: i for i, tok in enumerate(tokens)}
    merges = _parse_merges(raw_merges)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=False, fuse_unk=False))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


# ---------------------------------------------------------------------------
# SPM-style BPE (gemma4, gemma3)
# ---------------------------------------------------------------------------


def _build_spm_bpe(tokens: list[str], raw_merges: list[str], *, unk_str: str | None) -> Tokenizer:
    vocab = {tok: i for i, tok in enumerate(tokens)}
    merges = _parse_merges(raw_merges)
    tok = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=merges,
            unk_token=unk_str,
            byte_fallback=True,
            fuse_unk=True,
            ignore_merges=False,
        )
    )
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.pre_tokenizer = pre_tokenizers.Split(" ", behavior="merged_with_previous", invert=False)
    tok.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
    )
    return tok


# ---------------------------------------------------------------------------
# Turn-end EOS inference
# ---------------------------------------------------------------------------


def _infer_turn_end_eos(
    fast: PreTrainedTokenizerFast,
    eos_id: int | None,
    tokens: list[str],
    token_types: list[int] | None,
) -> list[int]:
    """Find additional EOS token IDs by detecting the turn-end marker from
    the chat template.

    Renders a test conversation with a sentinel in the assistant slot, then
    checks which control token (GGUF type=3) appears immediately after the
    sentinel.  Returns IDs that differ from the primary eos_id.
    """
    if not fast.chat_template:
        return []
    try:
        rendered = fast.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "GGUF_SENTINEL"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return []

    pos = rendered.rfind("GGUF_SENTINEL")
    if pos < 0:
        return []
    suffix = rendered[pos + len("GGUF_SENTINEL") :]
    if not suffix:
        return []

    extra: list[int] = []
    if token_types is not None:
        for tid, tstr in enumerate(tokens):
            if token_types[tid] == 3 and tid != eos_id and tstr and suffix.startswith(tstr):
                extra.append(tid)
                break
    return extra


# ---------------------------------------------------------------------------
# Round-trip self-test
# ---------------------------------------------------------------------------


def _self_test_roundtrip(fast: PreTrainedTokenizerFast) -> None:
    """Catch gross misconfiguration (wrong normalizer/decoder pairing).

    Fast-fail before the caller burns hours on a model run with a broken
    tokenizer. Doesn't catch all encode-divergence-from-HF cases; the
    caller should also do a fixture-set comparison vs HF when one is
    available.
    """
    samples = [
        "Hello, world!",
        "def f(x): return x*2",
    ]
    for s in samples:
        ids = fast.encode(s, add_special_tokens=False)
        back = fast.decode(ids, skip_special_tokens=False)
        if back != s:
            raise RuntimeError(f"tokenizer round-trip failed: {s!r} -> {ids[:10]} -> {back!r}")
