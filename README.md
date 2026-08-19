# vllm-memory-estimator

Estimates GPU memory requirements for serving Hugging Face models with
[vLLM](https://github.com/vllm-project/vllm). Two modes:

- **Estimate** — pass a `vllm serve` command and get per-GPU memory breakdowns
- **Budget** — given a model and GPU memory, see what context lengths and
  concurrency levels fit

Supports tensor parallelism (TP), pipeline parallelism (PP), data parallelism
(DP), and expert parallelism (EP) for MoE models — including DeepSeek-style
DP+EP (Wide EP) where attention is replicated and experts are distributed.

### Why vLLM is a dependency

This estimator is **intentionally tightly coupled with vLLM** — it imports and
reuses vLLM code rather than reimplementing it. The reasoning:

- **Architecture detection** — vLLM's `ModelConfig` correctly classifies
  hundreds of model architectures (MLA, hybrid Mamba, sliding window, chunked
  local attention) and handles edge cases like TP-aware KV head replication
  (`max(1, kv_heads // tp)`). Reimplementing this logic means falling behind
  every time vLLM adds a new architecture or fixes a classification bug.

- **KV cache sizing** — The estimator constructs vLLM's actual `KVCacheSpec`
  objects (`FullAttentionSpec`, `MLAAttentionSpec`, `SlidingWindowSpec`,
  `MambaSpec`, etc.) and calls their `max_memory_usage_bytes()` methods.
  This guarantees the estimate matches what vLLM would allocate at runtime,
  including block-size alignment and quantization-aware sizing.

- **No GPU required** — Despite importing vLLM, the estimator only uses its
  config and interface layers. It downloads `config.json` from Hugging Face,
  constructs a `ModelConfig` (which runs on CPU), reads safetensors file
  headers for parameter byte counts, and computes estimates — no model
  weights are loaded and no GPU is needed.

The trade-off is that the estimator must be installed alongside a compatible
vLLM version. This is deliberate: an estimate that doesn't match the serving
runtime is worse than no estimate at all.

## Installation

```bash
pip install -e .
```

For development (tests + linting):

```bash
pip install -e ".[dev]"
```

### Dependencies

- `huggingface_hub>=0.20.0` — model config and safetensors header fetching
- `vllm>=0.27.0` — `ModelConfig` for architecture detection and `KVCacheSpec`
  classes for cache estimation (CPU only, no GPU required)

## Usage

### CLI — Memory Estimation

Pass your `vllm serve` command directly — unknown flags like `--host` or
`--port` are silently ignored:

```bash
memory-estimator estimate "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 50"
```

The `estimate` subcommand is the default, so bare usage still works:

```bash
memory-estimator "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 50"
```

Sample output:

```
Model: openai/gpt-oss-120b
----------------------------------
Parameters      :  60.77 GiB (57.73 – 63.81)
Activations     :   0.87 GiB ( 0.43 –  1.74)
KV Cache        :   6.87 GiB ( 6.73 –  7.00)
Workspace       :   0.05 GiB ( 0.02 –  0.15)
----------------------------------
Total (raw)     :  68.55 GiB (64.91 – 72.70)
vLLM overhead   :   8.01 GiB ( 4.00 – 16.01)
----------------------------------
Total (vLLM)    :  76.56 GiB (68.92 – 88.71)
```

Each row shows a nominal estimate plus a (lower – upper) confidence range.

Add `--json` for machine-readable output:

```bash
memory-estimator estimate "vllm serve openai/gpt-oss-120b --max-model-len 2000" --json
```

### CLI — Token Budget Matrix

Compute a matrix showing what fits on your GPU across context lengths and
concurrency levels:

```bash
memory-estimator budget --model meta-llama/Llama-3.1-8B --gpu-memory-gib 80
```

Sample output:

```
Token Budget: meta-llama/Llama-3.1-8B (80.0 GiB per GPU)
════════════════════════════════════════════════════════════════════════════
  Context │   1 seq │   4 seq │   8 seq │  16 seq │  64 seq │ 256 seq │ Max Seqs
──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────
      256 │   16.1  │   16.1  │   16.1  │   16.1  │   16.2  │   16.7  │   4096+
    4,096 │   16.2  │   16.4  │   16.6  │   17.0  │   18.5  │   24.8  │    1258
   32,768 │   17.1  │   19.0  │   20.8  │   24.4  │   35.0  │   77.5  │     296
  131,072 │   19.5  │   27.2  │   34.9  │   50.2  │    ---  │    ---  │      74
════════════════════════════════════════════════════════════════════════════
Values: estimated per-GPU memory (GiB). --- = exceeds 80.0 GiB.
```

With parallelism and quantization:

```bash
memory-estimator budget --model meta-llama/Llama-3.1-70B --gpu-memory-gib 80 --tp 4
memory-estimator budget --model meta-llama/Llama-3.1-70B --gpu-memory-gib 80 --tp 4 -q fp8
```

Custom sweep ranges:

```bash
memory-estimator budget --model meta-llama/Llama-3.1-8B --gpu-memory-gib 80 \
    --seq-lengths 1024,4096,8192,32768 \
    --seq-counts 1,32,128,512
```

Output options:

```bash
memory-estimator budget --model meta-llama/Llama-3.1-8B --gpu-memory-gib 80 --json
memory-estimator budget --model meta-llama/Llama-3.1-8B --gpu-memory-gib 80 --html budget.html
```

### Budget CLI flags

| Flag | Description |
|------|-------------|
| `--model` | HuggingFace model ID (required) |
| `--gpu-memory-gib` | Available GPU memory in GiB (required) |
| `--tp` / `--tensor-parallel-size` | Tensor parallelism degree |
| `--pp` / `--pipeline-parallel-size` | Pipeline parallelism degree |
| `--dp` / `--data-parallel-size` | Data parallelism degree |
| `--enable-expert-parallel` | Enable expert parallelism (MoE) |
| `-q` / `--quantization` | Quantization method |
| `--dtype` | Activation dtype override |
| `--kv-cache-dtype` | KV cache dtype override |
| `--enforce-eager` | Disable CUDA graphs |
| `--block-size` | Paged attention block size |
| `--max-num-batched-tokens` | Tokens per forward pass |
| `--seq-lengths` | Comma-separated context lengths to sweep |
| `--seq-counts` | Comma-separated concurrency levels to sweep |
| `--json` | Output as JSON |
| `--html FILE` | Write HTML report to file |
| `--no-cache` | Force re-fetch model metadata |

### Supported vLLM flags (estimate mode)

The estimator parses these memory-relevant flags from the `vllm serve` command.
Unknown flags (e.g. `--host`, `--port`) are silently ignored.

| vLLM flag | What it affects |
|-----------|----------------|
| `--model` | Model to estimate (or positional first argument) |
| `--max-model-len` | Maximum sequence length (defaults to model's `max_position_embeddings`) |
| `--max-num-seqs` | Concurrent sequences for KV cache sizing (default: 256) |
| `--kv-cache-dtype` | KV cache precision — `fp8` halves cache memory |
| `--dtype` | Activation precision override |
| `--tensor-parallel-size` / `-tp` | Tensor parallelism — shards weights, KV cache, activations |
| `--pipeline-parallel-size` / `-pp` | Pipeline parallelism — splits layers across GPUs |
| `--data-parallel-size` | Data parallelism — full replica per GPU, fewer seqs per rank |
| `--enable-expert-parallel` | Expert parallelism (MoE) — distributes experts, replicates attention |
| `--enforce-eager` | Disable CUDA graphs (reduces overhead) |
| `--block-size` | Paged attention block size |
| `--max-num-batched-tokens` | Tokens per forward pass (controls activation sizing) |
| `--quantization` / `-q` | Quantization method override |
| `--revision` | Model revision / branch |
| `--cpu-offload-gb` | Offload this many GiB of weights to CPU (reduces GPU memory) |
| `--cudagraph-capture-sizes` | Comma-separated CUDA graph batch sizes (e.g. `1,2,4,8`) |

### Python API

The package exposes a clean programmatic API for use in scripts, notebooks, and
web applications:

```python
from memory_estimator import EstimatorInputs, estimate_from_inputs

summary, estimate = estimate_from_inputs(
    EstimatorInputs(
        model_id="meta-llama/Llama-3.1-8B",
        max_seq_len=4096,
        max_active_seqs=256,
        tensor_parallel_size=2,
    )
)

print(estimate.render_table())
print(f"Per GPU: {estimate.total_with_vllm.nominal_gib:.2f} GiB")
print(estimate.as_dict())  # JSON-serializable
```

Token budget matrix:

```python
from memory_estimator import compute_budget

result = compute_budget(
    "meta-llama/Llama-3.1-8B",
    gpu_memory_gib=80.0,
    tensor_parallel_size=4,
    seq_lengths=[1024, 4096, 16384, 65536],
    seq_counts=[1, 32, 128, 512],
)

print(result.render_table())          # terminal table
print(result.as_dict())               # JSON-serializable dict
print(result.render_html())           # self-contained HTML page

# Programmatic lookups
cell = result.cell(4096, 128)         # specific cell
print(cell.total_memory_gib, cell.fits)

max_seqs = result.max_seqs_at(4096)   # max concurrent seqs at this context
```

The `estimate` object exposes each component as a `MemoryComponentEstimate`
with `nominal_gib`, `lower_gib`, and `upper_gib` fields:

- `estimate.parameters` — model weights
- `estimate.activations` — forward pass activation buffers
- `estimate.kv_cache` — key/value cache
- `estimate.workspace` — temporary workspace / scratch memory
- `estimate.vllm_overhead` — CUDA graphs + block tables + worker overhead
- `estimate.total` — sum of first four (raw model memory)
- `estimate.total_with_vllm` — total including vLLM runtime overhead

## How It Works

1. **Command parsing** — The `vllm serve` command string is tokenized with
   `shlex.split` and parsed with argparse using `parse_known_args` to extract
   memory-relevant flags. Unknown flags are ignored.

2. **Configuration loading** — Constructs a vLLM `ModelConfig` from the
   Hugging Face model ID (no GPU required). This provides accurate
   architecture detection (MLA, hybrid Mamba, sliding window), TP-aware KV
   head counts (`max(1, total_kv_heads // tp)`), and quantization config
   parsing — all reusing vLLM's own logic rather than reimplementing it.

3. **Parameter byte counting** — Reads safetensors file headers via
   `huggingface_hub.get_safetensors_metadata` to obtain exact on-disk byte
   counts without downloading weights.

4. **Activation estimation** — Computes peak activation memory as:
   - Hidden state buffer (`tokens × hidden_size × bytes`)
   - Max of FFN intermediate and QKV projection buffers
   - Logits buffer (`tokens × vocab_size × bytes`)
   - MoE expert buffers (when applicable)

5. **KV cache estimation** — Uses vLLM's actual `KVCacheSpec` classes
   (`FullAttentionSpec`, `SlidingWindowSpec`, `MLAAttentionSpec`,
   `SlidingWindowMLASpec`, `ChunkedLocalAttentionSpec`, `MambaSpec`) for
   accurate cache sizing. Supports hybrid models with mixed layer types.

6. **vLLM overhead estimation** — Models three runtime components:
   - **CUDA graphs**: proportional to parameter size and layer count
   - **Block tables**: per-sequence metadata for paged attention
   - **Worker overhead**: fixed ~300 MiB for vLLM worker process state

7. **Parallelism** — Applies per-GPU memory division based on the active
   parallelism strategy:

   | Strategy | Weights | KV Cache | Activations |
   |----------|---------|----------|-------------|
   | **TP** | /TP | /TP (full attention) or unchanged (MLA/Mamba) | /TP |
   | **PP** | /PP | /PP (full attention) or unchanged (MLA/Mamba) | unchanged |
   | **DP** | unchanged | seqs/DP | seqs/DP |
   | **EP** (MoE) | attention replicated, experts /(TP×DP) | /TP | /TP |

   MLA (Multi-Latent Attention) and Mamba/SSM state are per-GPU — each GPU
   holds the full latent or state-space representation, so KV cache is not
   divided by TP for these architectures.

   Vision encoders and multimodal projectors are replicated across TP ranks
   (not sharded), matching vLLM's behavior.

8. **Range construction** — Each component gets a confidence range to account
   for runtime variability (allocator fragmentation, framework overhead, etc.).

## Memory Components

| Component | What it covers |
|-----------|---------------|
| **Parameters** | Model weights as stored on disk (exact byte count from safetensors) |
| **Activations** | Hidden states, FFN intermediates, QKV projections, logits buffer |
| **KV Cache** | Per-layer key/value tensors for all active sequences × sequence length |
| **Workspace** | Temporary scratch buffers (~5% of activation memory) |
| **vLLM Overhead** | CUDA graph captures, block tables, worker process state |

## Project Structure

```
src/memory_estimator/
├── __init__.py          # Public API exports
├── cli.py               # Subcommand-based CLI (estimate + budget)
├── budget.py            # Token budget matrix computation
├── vllm_cmd_parser.py   # vllm serve command string parser
├── estimator.py         # High-level API (EstimatorInputs → MemoryEstimate)
├── buckets.py           # Memory accounting by category
├── reports.py           # MemoryEstimate dataclass and table rendering
├── model_summary.py     # ModelSummary intermediate representation
├── model_shapes.py      # Safetensors-based parameter shape collection
├── quantization.py      # Quantization config parsing
├── kv_cache_specs.py    # KV cache estimation via vLLM spec classes
├── config_utils.py      # Architecture attribute resolution
├── dtype_utils.py       # Dtype normalisation and byte-width helpers
└── vllm_defaults.py     # vLLM-specific constants

tests/
├── conftest.py               # Shared fixtures and CLI options
├── test_buckets.py           # Unit tests for memory bucket calculations
├── test_budget.py            # Unit tests for token budget matrix
├── test_cli.py               # CLI argument parsing tests
├── test_dtype_utils.py       # Unit tests for dtype utilities
├── test_kv_cache_specs.py    # KV cache spec detection and formula tests
├── test_quantization.py      # Unit tests for quantization parsing
├── test_vllm_cmd_parser.py   # vllm serve command parser tests
├── test_validation.py        # Input validation helper tests
├── test_memory_profile.py    # GPU integration test (PyTorch runtime comparison)
└── test_vllm_profile.py      # vLLM integration test (validates against vllm bench)
```

## Testing

### Unit tests (no GPU required)

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_vllm_profile.py --ignore=tests/test_memory_profile.py -v
```

### GPU integration tests

Require a CUDA GPU. Compare estimated vs actual memory usage:

```bash
# PyTorch forward pass comparison
python -m pytest tests/test_memory_profile.py -v -m cuda --profile-report

# vLLM benchmark comparison (requires vLLM installed)
python -m pytest tests/test_vllm_profile.py -v -s --profile-report
```

### Linting

```bash
ruff check .
```

## Supported Models

The estimator works with any Hugging Face model whose `config.json` exposes
standard architecture attributes (`hidden_size`, `num_hidden_layers`,
`num_attention_heads`, etc.) and publishes weights in safetensors format. This
includes most decoder-only architectures:

- LLaMA / Llama 2 / Llama 3 / Llama 4 family
- GPT-2 / GPT-NeoX / OPT
- Mistral / Mixtral (MoE)
- DeepSeek V2 / V3 / V4 / R1 (MLA + MoE, including NVFP4)
- Qwen / Qwen2 / Qwen3 / Qwen3-VL / Qwen3.5 / Qwen3.5-MoE (including MoE)
- Phi / Phi-3
- Falcon / Falcon-H1 (hybrid Mamba)
- Gemma / Gemma-4 (sliding window)
- Jamba (hybrid Mamba + Attention)
- Nemotron-H (hybrid Mamba + Attention)
- Kimi-K2 / Kimi-K3 / MiniMax-M3 (MLA, sliding window MLA)
- Inkling (relative attention)
- MiniCPM-V 4.6 / InternS2 / OpenVLA
- Cohere MoE / EXAONE-4.5 / K-EXAONE-2.0

Quantized checkpoints (GPTQ, AWQ, compressed-tensors, FP8, NVFP4, MXFP4) are handled
automatically when `quantization_config` is present in the model config.

## Validation

The estimator is validated against real vLLM startup logs (231 records, 22
model families) comparing weights, KV cache per-token bytes, and non-weight
overhead. The validation suite lives in a separate private repo:
[vllm-memoryestimator-validation](https://github.com/ashishkamra/vllm-memoryestimator-validation).

Current accuracy: **94.8%** weight estimates in bounds (2.1% mean error),
**<1%** KV per-token error for full attention and MLA models.

## Notes and Limitations

- The estimator runs entirely on CPU — no GPU is needed.
- Parameter memory is the exact on-disk size from safetensors file headers.
  Models without safetensors files are not supported.
- KV cache spec construction (building the `FullAttentionSpec`, `MambaSpec`,
  etc. with the right parameters) is still done by the estimator since
  vLLM's `get_kv_cache_spec()` requires a loaded model with GPU.
- Actual runtime consumption can exceed estimates due to CUDA allocator
  fragmentation, LoRA adapters, speculative decoding, or tensor parallelism
  communication buffers. Leave headroom in production deployments.
- At large model scales (70B+), weights and KV cache dominate memory;
  activation estimates become proportionally less significant.
- Non-weight overhead (activations, CUDA graphs, workspace) has higher
  estimation error (~47%) than weights (~2%) or KV cache (<1%). This is an
  area for improvement.

## License

See [LICENSE](LICENSE) for details.
