# mlx-quant-tools

Quantization tooling for [MLX](https://github.com/ml-explore/mlx) models on Apple Silicon.

Provides an attention-protected mixed-precision quantizer, a KL-divergence scorer for measuring quantization quality, and utilities for working with GGUF K-quant files natively in MLX.

## Requirements

- Python 3.10+
- macOS with Apple Silicon
- MLX with K-quant support (`kquant` branch)

## Installation

```bash
# Core tools (recipe inspection, KLD rollups)
pip install -e .

# GGUF utilities (conversion, dequantization, inference)
pip install -e ".[gguf]"

# Quantization and scoring (requires MLX kquant branches)
pip install -e ".[quant]"

# Everything
pip install -e ".[all]"
```

The `[quant]` extra installs MLX, mlx-lm, and mlx-vlm from the
[kquant](https://github.com/asher/mlx/tree/kquant) branches, which add
native K-quant kernel support to the MLX framework.

## Tools

All tools are installed as `mqt-*` console scripts and accept `--help`.

### Quantization

| Command | Description |
|---------|-------------|
| `mqt-quantize` | Attention-protected mixed-precision quantizer. Applies floor rules that keep attention layers at higher precision while quantizing MLP layers aggressively. Supports text-only and multimodal (VLM) models, DWQ, and per-tensor MLP boosts. |
| `mqt-quantize-kquant` | K-quant format quantizer using MLX's native K-quant codecs (Q2_K through Q6_K). |

### Quality measurement

| Command | Description |
|---------|-------------|
| `mqt-score-kld` | KL-divergence scorer comparing a quantized student model against its teacher. Disk-cached teacher logits, top-K truncation, multimodal-aware. |
| `mqt-make-kld-rollup` | Aggregates per-checkpoint KLD JSON files into sorted rollup tables (Markdown + JSON). |
| `mqt-inspect-recipe` | Extracts per-tensor quantization recipe metadata from any MLX checkpoint. |

### GGUF utilities

| Command | Description |
|---------|-------------|
| `mqt-run-gguf` | Loads a GGUF K-quant file into a kquant-native MLX model and runs text generation or prefill/decode benchmarks. No safetensors round-trip. See [supported architectures](#supported-gguf-architectures). |
| `mqt-serve-gguf` | OpenAI-compatible HTTP server for GGUF K-quant models. Single-model, local-only, with KV prefix caching. See [supported architectures](#supported-gguf-architectures). |
| `mqt-load-kquant` | Loads an MLX K-quant checkpoint and prints the tensor inventory. |
| `mqt-gguf-to-mlx` | **Experimental.** Converts a GGUF file to MLX safetensors format, preserving K-quant wire bytes (no dequant round-trip). Only works with [fully supported architectures](#supported-gguf-architectures). |
| `mqt-dequant-gguf` | Dequantizes a GGUF K-quant file to bf16 safetensors (for use as a teacher model). |
| `mqt-validate-dequant` | Bit-exact validator for `mx.dequantize(mode="kquant")` against the gguf-python numpy reference. |
| `mqt-calibrate-imatrix` | Computes importance matrices for calibration-aware quantization. |

## Usage examples

### Running inference from a GGUF

Generate text directly from a GGUF file with no conversion step:

```bash
mqt-run-gguf /path/to/model-Q4_K_M.gguf \
    --prompt "Explain how transformers work" \
    --max-tokens 500 \
    --temp 0.7
```

### Benchmarking GGUF vs llama.cpp

Run prefill and decode benchmarks on a GGUF file and compare against
llama.cpp's `llama-bench` side by side:

```bash
mqt-run-gguf /path/to/model-Q4_K_L.gguf \
    --bench 128,512,2048,8192 \
    --max-tokens 300 \
    --bench-runs 5 \
    --bench-mode split \
    --bench-no-warmup \
    --vs-llama-bench
```

`--bench` sweeps comma-separated prompt lengths. `--bench-mode split`
separates prefill-only and decode-only timings for apples-to-apples
comparison with llama-bench. `--vs-llama-bench` automatically runs
llama-bench with matching parameters and prints a comparison table.

### Benchmarking GGUF K-quant vs MLX native quant

Compare a GGUF K-quant model against an MLX-native checkpoint (e.g.,
an Unsloth UD-MLX-4bit or an attention-protected recipe) on the same
architecture:

```bash
mqt-run-gguf /path/to/model-Q4_K_M.gguf \
    --bench 128,512,2048,8192 \
    --max-tokens 300 \
    --bench-runs 5 \
    --vs-mlx-source /path/to/model-UD-MLX-4bit
```

`--vs-mlx-source` loads the MLX checkpoint, runs the same benchmark
sweep, and prints a side-by-side comparison. Useful for kernel-isolation
A/Bs (K-quant vs flat affine 4-bit on the same model).

### Calibrating an importance matrix

Compute per-feature activation importance from a calibration corpus.
We recommend a slice of
[HuggingFaceFW/fineweb-edu (sample-10BT)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
as the calibration dataset — it consistently outperforms chat-format and
synthetic corpora across all targets we've measured.

```bash
# Download a calibration slice
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --revision refs/convert/parquet \
    --include "sample/10BT/train/00000.parquet" \
    --local-dir /tmp/fineweb-edu

# Extract plain text (one doc per line)
python -c "
import pandas as pd, pathlib
p = pathlib.Path('/tmp/fineweb-edu/sample/10BT/train/00000.parquet')
df = pd.read_parquet(p, columns=['text'])
pathlib.Path('/tmp/fineweb-cal.txt').write_text('\n'.join(df['text'].tolist()))
"

# Calibrate
mqt-calibrate-imatrix Qwen/Qwen3-0.6B \
    --corpus /tmp/fineweb-cal.txt \
    --ctx 512 \
    --chunks 50 \
    --output imatrix.dat
```

### Scoring KL divergence

Measure quantization quality by comparing a quantized student against
its full-precision teacher. Uses wikitext-103 by default (matching
llama.cpp's `--kl-divergence` protocol for cross-tool comparability):

```bash
# Score an MLX checkpoint
mqt-score-kld Qwen/Qwen3-0.6B /path/to/Qwen3-0.6B-AP4bit

# Score a GGUF directly (no conversion needed)
mqt-score-kld Qwen/Qwen3-0.6B --gguf /path/to/Qwen3-0.6B-Q4_K_M.gguf

# Use a different corpus
mqt-score-kld Qwen/Qwen3-0.6B /path/to/Qwen3-0.6B-AP4bit \
    --dataset "HuggingFaceFW/fineweb-edu:sample-10BT"

# More samples / longer context
mqt-score-kld Qwen/Qwen3-0.6B /path/to/Qwen3-0.6B-AP4bit \
    --num-samples 1024 --max-seq-len 2048 --long-context
```

The scorer writes a Markdown report (stdout) and a JSON dump
(`<student>/kld-vs-<teacher>.json`) pinned to a stable schema.
Use `mqt-make-kld-rollup` to aggregate multiple JSON results into
a sorted comparison table.

### Converting GGUF to MLX safetensors (experimental)

`mqt-gguf-to-mlx` converts a GGUF file into an MLX-loadable safetensors
checkpoint, preserving K-quant wire bytes (no dequant/re-encode
round-trip). The output is bit-equivalent to the GGUF for all supported
K-quant codecs (Q2_K through Q6_K, Q8_0):

```bash
mqt-gguf-to-mlx /path/to/model-Q4_K_M.gguf -o ./model-Q4_K_M

# Preflight check (shows tensor-type distribution without loading)
mqt-gguf-to-mlx /path/to/model-Q4_K_M.gguf --preflight-only
```

Config and tokenizer are synthesized from GGUF metadata — no parallel
HF checkpoint required. This means the tool only works with
[fully supported architectures](#supported-gguf-architectures) that
have both name remap and config synthesis implemented.

**Caveats:**

- The output checkpoint is only loadable by `mqt-load-kquant` or
  `mqt-run-gguf --mlx-source` — it uses the kquant wire format which
  standard `mlx_lm.load` does not understand without the kquant branch.
- Synthesized configs may not capture every model-specific detail
  (e.g. sliding window patterns, tied embedding flags). Always verify
  the output produces coherent generation before relying on it.
- IQ-family codecs (IQ1–IQ4) and Q8_K are not supported and will
  cause the tool to abort at preflight.

For bf16 dequantization (e.g. to create a teacher model from a GGUF),
use `mqt-dequant-gguf` instead.

## Supported GGUF architectures

The GGUF tools (`mqt-run-gguf`, `mqt-serve-gguf`, `mqt-gguf-to-mlx`,
`mqt-dequant-gguf`) require two things per architecture:

1. **Tensor name remap** — translating GGUF tensor names to HF/MLX names
   (handled by `gguf_name_remap`)
2. **Config synthesis** — constructing a model config from GGUF metadata
   (handled by `gguf_config_synth`)

Currently supported architectures (config synthesis + name remap):

| GGUF arch string | Model families |
|---|---|
| `gemma4` | Gemma-4 (E2B, E4B, 12B, 27B, 31B dense) |
| `qwen3` | Qwen3 dense |
| `qwen35` | Qwen3.5 dense (Mamba hybrid) |
| `qwen35moe` | Qwen3.5-MoE / Qwen3.6 (Mamba hybrid MoE) |
| `mistral3` | Mistral-Small-3.1, Ministral-3 |
| `nemotron_h_moe` | Nvidia Nemotron-H / Nemotron-Cascade |

Architectures with name remap only (no config synthesis yet —
need `--mlx-source` for a parallel HF checkpoint to supply config):
`gemma3`, `gemma3n`, `qwen3moe`, `llama`.

Adding a new architecture requires extending `ARCH_ALIAS` and
`CANONICAL_HF` in `gguf_name_remap.py`, plus adding a config builder
to `gguf_config_synth.py`.

## Library modules

The package also exposes importable modules:

- `mlx_quant_tools.model_role_classifier` -- tensor role classification (attention, MLP, embedding, VLM tower)
- `mlx_quant_tools.gguf_runtime` -- GGUF model loading, K-quant module installation, model building
- `mlx_quant_tools.load_gguf_kquant` -- raw uint8 GGUF loader for all 10 K-quant codecs
- `mlx_quant_tools.gguf_name_remap` -- GGUF-to-HF tensor name remapping
- `mlx_quant_tools.gguf_config_synth` -- config synthesis from GGUF metadata
- `mlx_quant_tools.gguf_tokenizer` -- tokenizer construction from GGUF metadata

## License

Apache 2.0
