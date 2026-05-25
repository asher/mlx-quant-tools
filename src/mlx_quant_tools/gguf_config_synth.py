"""Synthesize an HF-equivalent config dict from GGUF metadata.

`synthesize_config(reader)` returns a dict ready to feed into mlx_lm's
`_get_classes(config)` + `ModelArgs.from_dict(config)` — i.e. the same shape
that `load_config()` produces from a real `config.json`. Caller's only job
is to map the GGUF arch string ("gemma4", "qwen35", ...) to mlx_lm's
`model_type` ("gemma4_text", "qwen3_5", ...); that mapping lives here.

Supported arches (have per-arch field synthesis):
  - gemma4  → gemma4_text
  - qwen35  → qwen3_5
  - qwen35moe → qwen3_5_moe   (paired with `_UNWRAP_TO_TEXT` in the caller)

Other archs raise NotImplementedError; the universal fields alone are
generally not enough.

This module is intentionally numpy/mlx-free: only the `gguf` library is
imported. Mirror of the design constraints in `gguf_name_remap.py`.
"""

from __future__ import annotations

import sys
from typing import Any

GGUF_ARCH_TO_MODEL_TYPE = {
    "gemma4": "gemma4_text",
    "gemma3": "gemma3_text",
    "qwen35": "qwen3_5",
    "qwen35moe": "qwen3_5_moe",
    "qwen3": "qwen3",
    "qwen3moe": "qwen3_moe",
    "llama": "llama",
    # llama.cpp 'mistral3' arch covers both Mistral-Small-3.1 and Ministral-3.
    # In mlx_lm both deserialize as model_type='ministral3' (LlamaModel with
    # yarn rope + llama-4-style attention temperature scaling).
    "mistral3": "ministral3",
    "nemotron_h_moe": "nemotron_h",
}

# Arches that synthesize_config() actually produces complete configs for.
# Other archs in the table above are accepted as model_type targets but the
# synthesizer raises NotImplementedError — they need a per-arch extension.
_SUPPORTED = {"gemma4", "qwen3", "qwen35", "qwen35moe", "mistral3", "nemotron_h_moe"}


# ---------------------------------------------------------------------------
# GGUF KV access helpers
# ---------------------------------------------------------------------------
#
# Every scalar GGUF field is stored as `f.parts[f.data[0]]` — a 0-d/1-d
# memmap whose `[0]` element is the value. Strings are byte arrays decoded
# UTF-8. Typed arrays index `f.data` per element. Centralizing this here
# keeps callers one-liners.


def _read_int(reader, key: str) -> int | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return int(f.parts[f.data[0]][0])


def _read_float(reader, key: str) -> float | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return float(f.parts[f.data[0]][0])


def _read_bool(reader, key: str) -> bool | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return bool(f.parts[f.data[0]][0])


def _read_string(reader, key: str) -> str | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def _read_int_array(reader, key: str) -> list[int] | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return [int(f.parts[i][0]) for i in f.data]


def _read_bool_array(reader, key: str) -> list[bool] | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return [bool(f.parts[i][0]) for i in f.data]


def _array_len(reader, key: str) -> int | None:
    f = reader.fields.get(key)
    if f is None:
        return None
    return len(f.data)


# ---------------------------------------------------------------------------
# Tensor-inventory probes (for fields not in KV metadata)
# ---------------------------------------------------------------------------


def _tensor_shapes(reader) -> dict[str, list[int]]:
    """Map tensor name → integer shape list. One-pass walk."""
    return {t.name: [int(x) for x in t.shape] for t in reader.tensors}


def _has_tensor(shapes: dict[str, list[int]], name: str) -> bool:
    return name in shapes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def synthesize_config(reader) -> dict[str, Any]:
    """Build a config dict from GGUF metadata.

    Returns a dict shaped like a HuggingFace `config.json` for the relevant
    text-model class, ready to feed into mlx_lm's `_get_classes(config)` +
    `ModelArgs.from_dict(config)`.
    """
    arch = _read_string(reader, "general.architecture")
    if arch is None:
        raise ValueError("GGUF missing 'general.architecture' KV field")

    model_type = GGUF_ARCH_TO_MODEL_TYPE.get(arch)
    if model_type is None:
        raise ValueError(f"unsupported GGUF arch {arch!r}; extend GGUF_ARCH_TO_MODEL_TYPE")

    if arch not in _SUPPORTED:
        raise NotImplementedError(
            f"config synthesis for arch {arch!r} not implemented "
            f"(GGUF_ARCH_TO_MODEL_TYPE has it, but no per-arch extension exists)"
        )

    shapes = _tensor_shapes(reader)
    config: dict[str, Any] = {"model_type": model_type}

    _add_universal_fields(reader, shapes, config, arch)

    if arch == "gemma4":
        _synth_gemma4(reader, shapes, config)
    elif arch == "qwen3":
        _synth_qwen3(reader, shapes, config)
    elif arch in ("qwen35", "qwen35moe"):
        _synth_qwen35(reader, shapes, config, arch)
    elif arch == "mistral3":
        _synth_mistral3(reader, shapes, config)
    elif arch == "nemotron_h_moe":
        _synth_nemotron_h_moe(reader, shapes, config)

    _print_summary(config, arch)
    return config


# ---------------------------------------------------------------------------
# Universal field extraction
# ---------------------------------------------------------------------------


def _require(value, *, arch: str, gguf_field: str):
    if value is None:
        raise ValueError(f"config synth: missing GGUF field {gguf_field!r} for arch {arch!r}")
    return value


def _first_or_scalar(reader, key: str) -> int | None:
    """Read a field that may be a scalar or a per-layer array; return the
    first element (which should be uniform across layers for the targets
    we support — verified for gemma-4 dense `feed_forward_length` and
    `head_count_kv`)."""
    f = reader.fields.get(key)
    if f is None:
        return None
    return int(f.parts[f.data[0]][0])


def _add_universal_fields(reader, shapes, config: dict, arch: str) -> None:
    config["hidden_size"] = _require(
        _read_int(reader, f"{arch}.embedding_length"),
        arch=arch,
        gguf_field=f"{arch}.embedding_length",
    )
    block_count = _require(
        _read_int(reader, f"{arch}.block_count"), arch=arch, gguf_field=f"{arch}.block_count"
    )
    nextn_layers = _read_int(reader, f"{arch}.nextn_predict_layers") or 0
    config["num_hidden_layers"] = block_count - nextn_layers
    config["num_attention_heads"] = _require(
        _read_int(reader, f"{arch}.attention.head_count"),
        arch=arch,
        gguf_field=f"{arch}.attention.head_count",
    )
    # head_count_kv is per-layer on some MoE configs; first element is uniform.
    config["num_key_value_heads"] = _require(
        _first_or_scalar(reader, f"{arch}.attention.head_count_kv"),
        arch=arch,
        gguf_field=f"{arch}.attention.head_count_kv",
    )

    # feed_forward_length: scalar on most arches, per-layer array on gemma-4.
    intermediate = _first_or_scalar(reader, f"{arch}.feed_forward_length")
    if intermediate is not None:
        config["intermediate_size"] = intermediate

    ctx = _read_int(reader, f"{arch}.context_length")
    if ctx is not None:
        config["max_position_embeddings"] = ctx

    eps = _read_float(reader, f"{arch}.attention.layer_norm_rms_epsilon")
    if eps is not None:
        config["rms_norm_eps"] = eps

    # Default head_dim from key_length; arch-specific code may override
    # (gemma-4 uses different full-attn vs sliding head_dims).
    key_length = _read_int(reader, f"{arch}.attention.key_length")
    if key_length is not None:
        config["head_dim"] = key_length

    # Default rope_theta; arch-specific code may build a richer
    # `rope_parameters` dict on top.
    freq_base = _read_float(reader, f"{arch}.rope.freq_base")
    if freq_base is not None:
        config["rope_theta"] = freq_base

    # tied embeddings: present when the GGUF has no separate output.weight.
    config["tie_word_embeddings"] = not _has_tensor(shapes, "output.weight")

    # vocab size
    vocab_size = _array_len(reader, "tokenizer.ggml.tokens")
    if vocab_size is not None:
        config["vocab_size"] = vocab_size


# ---------------------------------------------------------------------------
# gemma4
# ---------------------------------------------------------------------------


def _synth_gemma4(reader, shapes, config: dict) -> None:
    arch = "gemma4"

    # Asymmetric K/V head dims (full vs SWA).
    full_kdim = _require(
        _read_int(reader, f"{arch}.attention.key_length"),
        arch=arch,
        gguf_field=f"{arch}.attention.key_length",
    )
    swa_kdim = _read_int(reader, f"{arch}.attention.key_length_swa")
    config["global_head_dim"] = full_kdim
    if swa_kdim is not None:
        # mlx_lm gemma4_text uses `head_dim` as the sliding-attention dim
        # and `global_head_dim` for full-attention layers.
        config["head_dim"] = swa_kdim

    config["num_kv_shared_layers"] = _require(
        _read_int(reader, f"{arch}.attention.shared_kv_layers"),
        arch=arch,
        gguf_field=f"{arch}.attention.shared_kv_layers",
    )

    sliding = _read_int(reader, f"{arch}.attention.sliding_window")
    if sliding is not None:
        config["sliding_window"] = sliding

    softcap = _read_float(reader, f"{arch}.final_logit_softcapping")
    if softcap is not None:
        config["final_logit_softcapping"] = softcap

    hspl = _read_int(reader, f"{arch}.embedding_length_per_layer_input")
    if hspl is not None:
        config["hidden_size_per_layer_input"] = hspl

    # layer_types from the per-layer bool array.
    pattern = _read_bool_array(reader, f"{arch}.attention.sliding_window_pattern")
    if pattern is not None:
        config["layer_types"] = ["sliding_attention" if v else "full_attention" for v in pattern]

    # K-eq-V detection: 26B/31B-class models drop the V projection on
    # full-attention layers and reuse K as V (mlx_lm gemma4_text.Attention
    # gates this on `attention_k_eq_v`). Detect by tensor inventory: if any
    # full-attention layer lacks `attn_v.weight`, the model is K-eq-V and
    # the surviving `attn_k.weight` carries n_global_kv_heads × head_dim.
    if pattern is not None:
        full_indices = [i for i, v in enumerate(pattern) if not v]
        if full_indices:
            i = full_indices[0]
            has_v = _has_tensor(shapes, f"blk.{i}.attn_v.weight")
            if not has_v:
                config["attention_k_eq_v"] = True
                k_shape = shapes.get(f"blk.{i}.attn_k.weight")
                if k_shape is None:
                    raise ValueError(
                        f"gemma4 synth: full-attn layer {i} missing both attn_v and attn_k tensors"
                    )
                if k_shape[1] % full_kdim != 0:
                    raise ValueError(
                        f"gemma4 synth: full-attn k_proj cols {k_shape[1]} "
                        f"not divisible by global_head_dim {full_kdim}"
                    )
                config["num_global_key_value_heads"] = k_shape[1] // full_kdim

    # rope_parameters: full-attn + sliding-attn sub-dicts.
    full_freq = _read_float(reader, f"{arch}.rope.freq_base")
    swa_freq = _read_float(reader, f"{arch}.rope.freq_base_swa")
    if full_freq is not None or swa_freq is not None:
        rp: dict[str, Any] = {}
        if full_freq is not None:
            rp["full_attention"] = {
                "partial_rotary_factor": 0.25,
                "rope_theta": full_freq,
                "rope_type": "proportional",
            }
        if swa_freq is not None:
            rp["sliding_attention"] = {
                "partial_rotary_factor": 1.0,
                "rope_theta": swa_freq,
                "rope_type": "default",
            }
        config["rope_parameters"] = rp

    # use_double_wide_mlp: True when kv-shared layers' MLP is 2× the base
    # intermediate size. With num_kv_shared_layers=0, no kv-shared layers
    # exist and the flag is functionally inert — match HF's convention of
    # `False` in that case.
    n_layers = config["num_hidden_layers"]
    n_shared = config["num_kv_shared_layers"]
    if n_shared > 0:
        first_shared = n_layers - n_shared
        base_shape = shapes.get("blk.0.ffn_gate.weight")
        shared_shape = shapes.get(f"blk.{first_shared}.ffn_gate.weight")
        if base_shape is None or shared_shape is None:
            raise ValueError(
                f"gemma4 synth: missing ffn_gate tensors needed to detect "
                f"use_double_wide_mlp (base={base_shape}, "
                f"shared={shared_shape})"
            )
        config["use_double_wide_mlp"] = shared_shape[1] == 2 * base_shape[1]
    else:
        config["use_double_wide_mlp"] = False

    # MoE: enable iff expert_count is present.
    expert_count = _read_int(reader, f"{arch}.expert_count")
    if expert_count is not None:
        config["enable_moe_block"] = True
        config["num_experts"] = expert_count
        config["top_k_experts"] = _require(
            _read_int(reader, f"{arch}.expert_used_count"),
            arch=arch,
            gguf_field=f"{arch}.expert_used_count",
        )
        config["moe_intermediate_size"] = _require(
            _read_int(reader, f"{arch}.expert_feed_forward_length"),
            arch=arch,
            gguf_field=f"{arch}.expert_feed_forward_length",
        )


# ---------------------------------------------------------------------------
# qwen3
# ---------------------------------------------------------------------------


def _synth_qwen3(reader, shapes, config: dict) -> None:
    """Synthesize a qwen3 config. Universal fields cover everything; this
    function handles rope_scaling passthrough so mlx_lm's initialize_rope
    doesn't see unexpected keys."""
    arch = "qwen3"
    rope_scaling_type = _read_string(reader, f"{arch}.rope.scaling.type")
    if rope_scaling_type is not None:
        rp: dict[str, Any] = {"type": rope_scaling_type, "rope_type": rope_scaling_type}
        factor = _read_float(reader, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        config["rope_scaling"] = rp


# ---------------------------------------------------------------------------
# qwen35 / qwen35moe
# ---------------------------------------------------------------------------


def _synth_qwen35(reader, shapes, config: dict, arch: str) -> None:
    config["full_attention_interval"] = _require(
        _read_int(reader, f"{arch}.full_attention_interval"),
        arch=arch,
        gguf_field=f"{arch}.full_attention_interval",
    )

    inner = _require(
        _read_int(reader, f"{arch}.ssm.inner_size"), arch=arch, gguf_field=f"{arch}.ssm.inner_size"
    )
    num_v = _require(
        _read_int(reader, f"{arch}.ssm.time_step_rank"),
        arch=arch,
        gguf_field=f"{arch}.ssm.time_step_rank",
    )
    num_k = _require(
        _read_int(reader, f"{arch}.ssm.group_count"),
        arch=arch,
        gguf_field=f"{arch}.ssm.group_count",
    )
    conv_kernel = _require(
        _read_int(reader, f"{arch}.ssm.conv_kernel"),
        arch=arch,
        gguf_field=f"{arch}.ssm.conv_kernel",
    )

    if inner % num_v != 0:
        raise ValueError(
            f"qwen35 synth: ssm.inner_size={inner} not divisible by time_step_rank={num_v}"
        )
    head_v_dim = inner // num_v

    # Cross-check: ssm_norm.shape == [head_v_dim], ssm_a.shape == [num_v]
    ssm_norm_shape = shapes.get("blk.0.ssm_norm.weight")
    if ssm_norm_shape is not None and ssm_norm_shape[0] != head_v_dim:
        raise ValueError(
            f"qwen35 synth: derived head_v_dim={head_v_dim} but "
            f"blk.0.ssm_norm.weight shape={ssm_norm_shape}"
        )
    ssm_a_shape = shapes.get("blk.0.ssm_a")
    if ssm_a_shape is not None and ssm_a_shape[0] != num_v:
        raise ValueError(
            f"qwen35 synth: derived num_v_heads={num_v} but blk.0.ssm_a shape={ssm_a_shape}"
        )
    # ssm_alpha is Linear(hidden, num_v_heads) — same num_v, not num_k.
    # No tensor directly exposes num_k_heads; the conv1d shape derivation
    # below + the metadata field `<arch>.ssm.group_count` are the sources.

    # head_k_dim is genuinely absent from KV metadata. Derive from
    # ssm_conv1d shape: it's [kernel, inner + 2 * num_k * head_k_dim].
    conv_shape = shapes.get("blk.0.ssm_conv1d.weight")
    if conv_shape is None:
        raise ValueError(
            "qwen35 synth: missing tensor blk.0.ssm_conv1d.weight — "
            "needed to derive linear_key_head_dim"
        )
    conv_in = conv_shape[1]
    if (conv_in - inner) % (2 * num_k) != 0:
        raise ValueError(
            f"qwen35 synth: ssm_conv1d shape={conv_shape} doesn't match "
            f"inner_size={inner} + 2 * num_k_heads={num_k} * head_k_dim"
        )
    head_k_dim = (conv_in - inner) // (2 * num_k)

    config["linear_num_value_heads"] = num_v
    config["linear_num_key_heads"] = num_k
    config["linear_value_head_dim"] = head_v_dim
    config["linear_key_head_dim"] = head_k_dim
    config["linear_conv_kernel_dim"] = conv_kernel

    # GGUF V-indexed tensors are in tiled order (convert_hf_to_gguf reorders).
    config["kv_head_layout"] = "tiled"

    # Build rope_parameters so __post_init__ doesn't override rope_theta
    # with the default (100000). mrope_section comes from the GGUF dimension
    # sections array (last element is padding/zero and gets dropped).
    freq_base = config.get("rope_theta", 10000000.0)
    dim_sections = _read_int_array(reader, f"{arch}.rope.dimension_sections")
    mrope_section = [s for s in (dim_sections or [11, 11, 10, 0]) if s > 0]
    config["rope_parameters"] = {
        "type": "default",
        "rope_theta": freq_base,
        "mrope_section": mrope_section,
        "partial_rotary_factor": 0.25,
    }

    # MoE fields (only on qwen35moe).
    if arch == "qwen35moe":
        config["num_experts"] = _require(
            _read_int(reader, f"{arch}.expert_count"), arch=arch, gguf_field=f"{arch}.expert_count"
        )
        config["num_experts_per_tok"] = _require(
            _read_int(reader, f"{arch}.expert_used_count"),
            arch=arch,
            gguf_field=f"{arch}.expert_used_count",
        )
        config["moe_intermediate_size"] = _require(
            _read_int(reader, f"{arch}.expert_feed_forward_length"),
            arch=arch,
            gguf_field=f"{arch}.expert_feed_forward_length",
        )
        shared_ffn = _read_int(reader, f"{arch}.expert_shared_feed_forward_length")
        if shared_ffn is not None:
            config["shared_expert_intermediate_size"] = shared_ffn


# ---------------------------------------------------------------------------
# mistral3 (Ministral-3 / Mistral-Small-3.1)
# ---------------------------------------------------------------------------


def _synth_mistral3(reader, shapes, config: dict) -> None:
    """Synthesize a ministral3 config from a 'mistral3' GGUF.

    The model is a LlamaModel variant with yarn RoPE scaling and a
    llama-4-style attention temperature scale. mlx_lm's ministral3 packs
    all of that into `rope_parameters` (it serves as both the rope scaling
    config and the bag of llama4 attention-scale knobs).
    """
    arch = "mistral3"

    # head_dim already set from key_length in universal fields.

    # rope_parameters bag. Reads the yarn scaling fields llama.cpp writes
    # for ministral3, plus the llama-4 attention temperature scale fields.
    rp: dict[str, Any] = {"rope_type": "yarn", "type": "yarn"}

    freq_base = _read_float(reader, f"{arch}.rope.freq_base")
    if freq_base is not None:
        rp["rope_theta"] = freq_base

    factor = _read_float(reader, f"{arch}.rope.scaling.factor")
    if factor is not None:
        rp["factor"] = factor

    orig_ctx = _read_int(reader, f"{arch}.rope.scaling.original_context_length")
    if orig_ctx is not None:
        rp["original_max_position_embeddings"] = orig_ctx

    beta_fast = _read_float(reader, f"{arch}.rope.scaling.yarn_beta_fast")
    if beta_fast is not None:
        rp["beta_fast"] = beta_fast
    beta_slow = _read_float(reader, f"{arch}.rope.scaling.yarn_beta_slow")
    if beta_slow is not None:
        rp["beta_slow"] = beta_slow

    # llama.cpp's `add_rope_scaling_yarn_log_mul` is the HF
    # `rope_parameters.mscale_all_dim` value.
    log_mul = _read_float(reader, f"{arch}.rope.scaling.yarn_log_multiplier")
    if log_mul is not None:
        rp["mscale_all_dim"] = log_mul

    # llama.cpp's `add_attn_temperature_scale` is HF
    # `rope_parameters.llama_4_scaling_beta`. ministral3.LanguageModel
    # reads this off rope_parameters when computing the per-position
    # attention scale.
    temp_scale = _read_float(reader, f"{arch}.attention.temperature_scale")
    if temp_scale is not None:
        rp["llama_4_scaling_beta"] = temp_scale
    else:
        # Field is mandatory for ministral3 forward; default to 0.0 (no
        # scaling) so non-ministral3 mistral3 variants still load.
        rp["llama_4_scaling_beta"] = 0.0

    config["rope_parameters"] = rp

    # ministral3.LanguageModel reads
    # rope_parameters["original_max_position_embeddings"] for the attn-scale
    # divisor; if we didn't find it, fall back to max_position_embeddings.
    if "original_max_position_embeddings" not in rp:
        rp["original_max_position_embeddings"] = config.get("max_position_embeddings", 4096)

    # ministral3 is dense, single layer type. (Mistral-Small-3.1 has SWA
    # but the public GGUF conversions don't expose a sliding pattern KV,
    # so default to all full_attention until we see a counter-example.)
    config.setdefault("layer_types", ["full_attention"] * config["num_hidden_layers"])


# ---------------------------------------------------------------------------
# nemotron_h_moe (Nemotron-3-Super, hybrid Mamba2+Attention+MoE)
# ---------------------------------------------------------------------------


def _synth_nemotron_h_moe(reader, shapes, config: dict) -> None:
    arch = "nemotron_h_moe"

    # nemotron_h model uses layer_norm_epsilon (not rms_norm_eps)
    eps = _read_float(reader, f"{arch}.attention.layer_norm_rms_epsilon")
    if eps is not None:
        config["layer_norm_epsilon"] = eps
        config.pop("rms_norm_eps", None)

    # Mamba2 SSM parameters
    ssm_inner = _require(
        _read_int(reader, f"{arch}.ssm.inner_size"), arch=arch, gguf_field=f"{arch}.ssm.inner_size"
    )
    ssm_time_step_rank = _require(
        _read_int(reader, f"{arch}.ssm.time_step_rank"),
        arch=arch,
        gguf_field=f"{arch}.ssm.time_step_rank",
    )
    ssm_state_size = _require(
        _read_int(reader, f"{arch}.ssm.state_size"), arch=arch, gguf_field=f"{arch}.ssm.state_size"
    )
    conv_kernel = _require(
        _read_int(reader, f"{arch}.ssm.conv_kernel"),
        arch=arch,
        gguf_field=f"{arch}.ssm.conv_kernel",
    )
    n_groups = _require(
        _read_int(reader, f"{arch}.ssm.group_count"),
        arch=arch,
        gguf_field=f"{arch}.ssm.group_count",
    )

    config["mamba_num_heads"] = ssm_time_step_rank
    config["mamba_head_dim"] = ssm_inner // ssm_time_step_rank
    config["ssm_state_size"] = ssm_state_size
    config["conv_kernel"] = conv_kernel
    config["n_groups"] = n_groups

    # Bias flags. use_conv_bias=True is architectural for NemotronH
    # (Mamba2 always has conv bias). Shape-based detection fails on
    # split GGUFs where shard 1 has no tensors.
    config["use_conv_bias"] = True
    config["mamba_proj_bias"] = False
    config["attention_bias"] = False
    config["mlp_bias"] = False
    config["use_bias"] = False
    # NemotronH always has a separate lm_head (never tied).
    config["tie_word_embeddings"] = False

    # head_count_kv is per-layer: extract the non-zero value for attention
    # layers (all attention layers share the same GQA group count).
    kv_array = _read_int_array(reader, f"{arch}.attention.head_count_kv")
    if kv_array is not None:
        non_zero = [v for v in kv_array if v > 0]
        if non_zero:
            config["num_key_value_heads"] = non_zero[0]

    # MoE parameters
    n_experts = _read_int(reader, f"{arch}.expert_count")
    if n_experts is not None:
        config["n_routed_experts"] = n_experts
        config["num_experts_per_tok"] = _require(
            _read_int(reader, f"{arch}.expert_used_count"),
            arch=arch,
            gguf_field=f"{arch}.expert_used_count",
        )
        config["moe_intermediate_size"] = _require(
            _read_int(reader, f"{arch}.expert_feed_forward_length"),
            arch=arch,
            gguf_field=f"{arch}.expert_feed_forward_length",
        )

        shared_ffn = _read_int(reader, f"{arch}.expert_shared_feed_forward_length")
        if shared_ffn is not None:
            config["moe_shared_expert_intermediate_size"] = shared_ffn

        shared_count = _read_int(reader, f"{arch}.expert_shared_count")
        if shared_count is not None:
            config["n_shared_experts"] = shared_count

        latent_size = _read_int(reader, f"{arch}.moe_latent_size")
        if latent_size is not None:
            config["moe_latent_size"] = latent_size

        config["n_group"] = _read_int(reader, f"{arch}.expert_group_count") or 1
        config["topk_group"] = _read_int(reader, f"{arch}.expert_group_used_count") or 1

        norm_topk = _read_int(reader, f"{arch}.expert_weights_norm")
        config["norm_topk_prob"] = bool(norm_topk) if norm_topk is not None else True

        scale = _read_float(reader, f"{arch}.expert_weights_scale")
        if scale is None:
            scale_int = _read_int(reader, f"{arch}.expert_weights_scale")
            scale = float(scale_int) if scale_int is not None else 1.0
        config["routed_scaling_factor"] = scale

    # intermediate_size for dense MLP layers (type "-"). Use
    # feed_forward_length if uniform, else default to moe_intermediate_size.
    ff_array = _read_int_array(reader, f"{arch}.feed_forward_length")
    if ff_array is not None:
        non_zero_ff = [v for v in ff_array if v > 0]
        if non_zero_ff:
            config["intermediate_size"] = non_zero_ff[0]
    if "intermediate_size" not in config:
        config["intermediate_size"] = config.get("moe_intermediate_size", 0)

    # Derive hybrid_override_pattern from per-layer metadata arrays.
    # head_count_kv > 0 → attention (*); feed_forward_length > 0 and kv==0
    # → MoE (E) when experts exist, else dense MLP (-); otherwise → Mamba (M).
    if kv_array is not None and ff_array is not None:
        n_layers = len(kv_array)
        pattern = []
        for i in range(n_layers):
            if kv_array[i] > 0:
                pattern.append("*")
            elif ff_array[i] > 0:
                pattern.append("E" if n_experts else "-")
            else:
                pattern.append("M")
        config["hybrid_override_pattern"] = pattern
        config["num_hidden_layers"] = n_layers


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(config: dict, arch: str) -> None:
    bits = [
        f"model_type={config.get('model_type')}",
        f"hidden={config.get('hidden_size')}",
        f"layers={config.get('num_hidden_layers')}",
        f"heads={config.get('num_attention_heads')}",
        f"kv_heads={config.get('num_key_value_heads')}",
        f"vocab={config.get('vocab_size')}",
    ]
    if "num_kv_shared_layers" in config:
        bits.append(f"kv_shared={config['num_kv_shared_layers']}")
    if config.get("enable_moe_block") or config.get("num_experts"):
        bits.append(f"experts={config.get('num_experts')}")
    print("[config] synthesized: " + " ".join(bits), file=sys.stderr)
