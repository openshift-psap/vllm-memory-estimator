"""MCP server exposing vLLM memory estimation as tools for AI agents."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .budget import compute_budget
from .estimator import EstimatorInputs
from .estimator import estimate_from_inputs

mcp = FastMCP(
    "vllm-memory-estimator",
    instructions="Estimate GPU memory requirements for serving models with vLLM",
)


@mcp.tool()
def estimate_memory(
    model_id: str,
    max_seq_len: int | None = None,
    max_active_seqs: int = 256,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    quantization: str | None = None,
    dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    enforce_eager: bool = False,
    max_num_batched_tokens: int | None = None,
    cpu_offload_gb: float = 0.0,
    revision: str | None = None,
    block_size: int | None = None,
    cudagraph_capture_sizes: list[int] | None = None,
) -> dict:
    """Estimate per-GPU memory for serving a HuggingFace model with vLLM.

    Returns a breakdown of parameters, activations, KV cache, workspace, and
    vLLM overhead in GiB, each with nominal/lower/upper bounds.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B")
        max_seq_len: Max sequence length (uses model default if not set)
        max_active_seqs: Max concurrent sequences (default 256)
        tensor_parallel_size: Number of tensor parallel GPUs
        pipeline_parallel_size: Number of pipeline parallel stages
        data_parallel_size: Number of data parallel replicas
        enable_expert_parallel: Enable expert parallelism for MoE models
        quantization: Quantization method (fp8, awq, gptq, nvfp4, etc.)
        dtype: Model dtype (float16, bfloat16)
        kv_cache_dtype: KV cache dtype override
        enforce_eager: Disable CUDA graphs
        max_num_batched_tokens: Max tokens per forward pass
        cpu_offload_gb: Offload this many GiB of weights to CPU
        revision: Model revision/branch on HuggingFace
        block_size: KV cache block size override
        cudagraph_capture_sizes: CUDA graph capture batch sizes
    """
    inputs = EstimatorInputs(
        model_id=model_id,
        max_seq_len=max_seq_len,
        max_active_seqs=max_active_seqs,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        quantization=quantization,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
        max_num_batched_tokens=max_num_batched_tokens,
        cpu_offload_gb=cpu_offload_gb,
        revision=revision,
        block_size=block_size,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    summary, estimate = estimate_from_inputs(inputs)
    return {
        "model_id": summary.model_id,
        "architecture": summary.architecture,
        "parameter_count": summary.parameter_count,
        "max_active_seqs": summary.max_active_seqs,
        "max_seq_len": summary.max_seq_len,
        "tensor_parallel_size": summary.tensor_parallel_size,
        "pipeline_parallel_size": summary.pipeline_parallel_size,
        "data_parallel_size": summary.data_parallel_size,
        "enable_expert_parallel": summary.enable_expert_parallel,
        "total_gpus": summary.total_gpus,
        "quantization": summary.quantization.as_payload(),
        "estimate": estimate.as_dict(),
    }


@mcp.tool()
def budget_matrix(
    model_id: str,
    gpu_memory_gib: float,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    quantization: str | None = None,
    dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    enforce_eager: bool = False,
    block_size: int | None = None,
    revision: str | None = None,
    seq_lengths: list[int] | None = None,
    seq_counts: list[int] | None = None,
    max_num_batched_tokens: int | None = None,
) -> dict:
    """Find what context lengths and concurrency levels fit in GPU memory.

    Sweeps combinations of sequence length and concurrent sequence count,
    returning a matrix showing which configurations fit and their memory usage.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B")
        gpu_memory_gib: Available GPU memory in GiB (e.g. 80 for H100)
        tensor_parallel_size: Number of tensor parallel GPUs
        pipeline_parallel_size: Number of pipeline parallel stages
        data_parallel_size: Number of data parallel replicas
        enable_expert_parallel: Enable expert parallelism for MoE models
        quantization: Quantization method (fp8, awq, gptq, nvfp4, etc.)
        dtype: Model dtype (float16, bfloat16)
        kv_cache_dtype: KV cache dtype override
        enforce_eager: Disable CUDA graphs
        block_size: KV cache block size override
        revision: Model revision/branch on HuggingFace
        seq_lengths: Context lengths to sweep (default: powers of 2 up to model max)
        seq_counts: Concurrent sequence counts to sweep (default: 1..512)
        max_num_batched_tokens: Max tokens per forward pass
    """
    result = compute_budget(
        model_id=model_id,
        gpu_memory_gib=gpu_memory_gib,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        quantization=quantization,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
        block_size=block_size,
        revision=revision,
        seq_lengths=seq_lengths,
        seq_counts=seq_counts,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    return result.as_dict()
