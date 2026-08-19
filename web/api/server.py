"""FastAPI backend for the vLLM memory estimator web UI."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory_estimator.budget import compute_budget
from memory_estimator.estimator import EstimatorInputs
from memory_estimator.estimator import estimate_from_inputs

logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Memory Estimator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EstimateRequest(BaseModel):
    model_id: str
    max_seq_len: int | None = None
    max_active_seqs: int = 256
    revision: str | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    quantization: str | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    enforce_eager: bool = False
    block_size: int | None = None
    max_num_batched_tokens: int | None = None
    cudagraph_capture_sizes: list[int] | None = None
    cpu_offload_gb: float = 0.0


class BudgetRequest(BaseModel):
    model_id: str
    gpu_memory_gib: float
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    quantization: str | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    enforce_eager: bool = False
    block_size: int | None = None
    revision: str | None = None
    seq_lengths: list[int] | None = None
    seq_counts: list[int] | None = None
    max_num_batched_tokens: int | None = None


@app.post("/api/estimate")
def api_estimate(req: EstimateRequest) -> dict:
    try:
        inputs = EstimatorInputs(
            model_id=req.model_id,
            max_seq_len=req.max_seq_len,
            max_active_seqs=req.max_active_seqs,
            revision=req.revision,
            dtype=req.dtype,
            kv_cache_dtype=req.kv_cache_dtype,
            quantization=req.quantization,
            tensor_parallel_size=req.tensor_parallel_size,
            pipeline_parallel_size=req.pipeline_parallel_size,
            data_parallel_size=req.data_parallel_size,
            enable_expert_parallel=req.enable_expert_parallel,
            enforce_eager=req.enforce_eager,
            block_size=req.block_size,
            max_num_batched_tokens=req.max_num_batched_tokens,
            cudagraph_capture_sizes=req.cudagraph_capture_sizes,
            cpu_offload_gb=req.cpu_offload_gb,
        )
        summary, estimate = estimate_from_inputs(inputs)
        return {
            "ok": True,
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
    except Exception as exc:
        logger.exception("Estimation failed for %s", req.model_id)
        return {"ok": False, "error": str(exc)}


@app.post("/api/budget")
def api_budget(req: BudgetRequest) -> dict:
    try:
        result = compute_budget(
            model_id=req.model_id,
            gpu_memory_gib=req.gpu_memory_gib,
            tensor_parallel_size=req.tensor_parallel_size,
            pipeline_parallel_size=req.pipeline_parallel_size,
            data_parallel_size=req.data_parallel_size,
            enable_expert_parallel=req.enable_expert_parallel,
            quantization=req.quantization,
            dtype=req.dtype,
            kv_cache_dtype=req.kv_cache_dtype,
            enforce_eager=req.enforce_eager,
            block_size=req.block_size,
            revision=req.revision,
            seq_lengths=req.seq_lengths,
            seq_counts=req.seq_counts,
            max_num_batched_tokens=req.max_num_batched_tokens,
        )
        return {"ok": True, **result.as_dict()}
    except Exception as exc:
        logger.exception("Budget computation failed for %s", req.model_id)
        return {"ok": False, "error": str(exc)}
