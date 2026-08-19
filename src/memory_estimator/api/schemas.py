"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class EstimateRequest(BaseModel):
    model_id: str = Field(description="HuggingFace model ID")
    max_seq_len: int | None = Field(default=None, description="Max sequence length")
    max_active_seqs: int = Field(default=256, description="Max concurrent sequences")
    revision: str | None = Field(default=None, description="Model revision/branch")
    dtype: str | None = Field(default=None, description="Model dtype (auto, float16, bfloat16)")
    kv_cache_dtype: str | None = Field(default=None, description="KV cache dtype")
    quantization: str | None = Field(
        default=None, description="Quantization method (fp8, awq, gptq, nvfp4, etc.)")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallel size")
    data_parallel_size: int = Field(default=1, ge=1, description="Data parallel size")
    enable_expert_parallel: bool = Field(default=False, description="Enable expert parallelism")
    enforce_eager: bool = Field(default=False, description="Disable CUDA graphs")
    block_size: int | None = Field(default=None, description="KV cache block size")
    max_num_batched_tokens: int | None = Field(
        default=None, description="Max tokens per forward pass")
    cudagraph_capture_sizes: list[int] | None = Field(
        default=None, description="CUDA graph batch sizes")
    cpu_offload_gb: float = Field(default=0.0, ge=0, description="CPU weight offload in GiB")


class BudgetRequest(BaseModel):
    model_id: str = Field(description="HuggingFace model ID")
    gpu_memory_gib: float = Field(gt=0, description="Available GPU memory in GiB")
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    enable_expert_parallel: bool = False
    quantization: str | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    enforce_eager: bool = False
    block_size: int | None = None
    revision: str | None = None
    max_num_batched_tokens: int | None = None
    seq_lengths: list[int] | None = Field(
        default=None, description="Context lengths to sweep")
    seq_counts: list[int] | None = Field(
        default=None, description="Concurrent sequence counts to sweep")


class ComponentEstimate(BaseModel):
    nominal_gib: float
    lower_gib: float
    upper_gib: float


class EstimateResponse(BaseModel):
    model_id: str
    architecture: str
    parameter_count: int
    max_active_seqs: int
    max_seq_len: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    enable_expert_parallel: bool
    total_gpus: int
    quantization: dict
    estimate: dict


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
