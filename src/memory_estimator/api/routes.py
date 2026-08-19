"""API route handlers."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi import HTTPException

from ..budget import compute_budget
from ..estimator import EstimatorInputs
from ..estimator import estimate_from_inputs
from .schemas import BudgetRequest
from .schemas import EstimateRequest
from .schemas import EstimateResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


@router.post("/estimate", response_model=EstimateResponse)
def estimate(req: EstimateRequest) -> EstimateResponse:
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
        summary, est = estimate_from_inputs(inputs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Estimation failed for %s", req.model_id)
        raise HTTPException(status_code=500, detail="Internal server error") from None

    return EstimateResponse(
        model_id=summary.model_id,
        architecture=summary.architecture,
        parameter_count=summary.parameter_count,
        max_active_seqs=summary.max_active_seqs,
        max_seq_len=summary.max_seq_len,
        tensor_parallel_size=summary.tensor_parallel_size,
        pipeline_parallel_size=summary.pipeline_parallel_size,
        data_parallel_size=summary.data_parallel_size,
        enable_expert_parallel=summary.enable_expert_parallel,
        total_gpus=summary.total_gpus,
        quantization=summary.quantization.as_payload(),
        estimate=est.as_dict(),
    )


@router.post("/budget")
def budget(req: BudgetRequest) -> dict:
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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Budget computation failed for %s", req.model_id)
        raise HTTPException(status_code=500, detail="Internal server error") from None

    return result.as_dict()
