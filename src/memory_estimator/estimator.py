"""High-level orchestration for building memory estimates."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from .buckets import build_memory_buckets
from .dtype_utils import normalise_dtype
from .model_shapes import collect_parameter_shapes
from .model_summary import ModelSummary
from .quantization import parse_quantization
from .reports import MemoryEstimate
from .reports import build_estimate
from .validation import validate_positive_int

logger = logging.getLogger(__name__)


@dataclass
class EstimatorInputs:
    model_id: str
    max_seq_len: int | None = None
    max_active_seqs: int = 256
    revision: str | None = None
    enforce_eager: bool = False
    cudagraph_capture_sizes: list[int] | None = None
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: str | None = None
    dtype: str | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    block_size: int | None = None
    quantization: str | None = None
    cpu_offload_gb: float = 0.0
    use_cache: bool = True


def _load_model_config(inputs: EstimatorInputs):
    """Construct a vLLM ModelConfig from EstimatorInputs.

    ModelConfig downloads config.json, resolves architecture attributes
    (head_dim, kv_heads, MLA detection, hybrid detection, etc.), and
    handles quantization config parsing — replacing our custom
    _DictConfig wrapper and config_utils attribute resolvers.
    """
    from vllm.config.model import ModelConfig

    vllm_log = logging.getLogger("vllm")
    prev_level = vllm_log.level
    vllm_log.setLevel(logging.ERROR)
    try:
        kwargs: dict = {
            "model": inputs.model_id,
            "trust_remote_code": True,
        }
        if inputs.revision:
            kwargs["revision"] = inputs.revision
        if inputs.quantization:
            kwargs["quantization"] = inputs.quantization
        if inputs.dtype and inputs.dtype != "auto":
            kwargs["dtype"] = inputs.dtype
        if inputs.max_seq_len is not None:
            kwargs["max_model_len"] = inputs.max_seq_len
        return ModelConfig(**kwargs)
    finally:
        vllm_log.setLevel(prev_level)


def _build_parallel_config(inputs: EstimatorInputs):
    from vllm.config.parallel import ParallelConfig

    return ParallelConfig(
        tensor_parallel_size=inputs.tensor_parallel_size,
        pipeline_parallel_size=inputs.pipeline_parallel_size,
        data_parallel_size=inputs.data_parallel_size,
    )


def _validate_inputs(inputs: EstimatorInputs) -> None:
    if inputs.max_seq_len is not None:
        validate_positive_int(inputs.max_seq_len, name="max_seq_len")
    validate_positive_int(inputs.max_active_seqs, name="max_active_seqs")
    validate_positive_int(inputs.tensor_parallel_size, name="tensor_parallel_size")
    validate_positive_int(inputs.pipeline_parallel_size, name="pipeline_parallel_size")
    validate_positive_int(inputs.data_parallel_size, name="data_parallel_size")
    if inputs.max_num_batched_tokens is not None:
        validate_positive_int(inputs.max_num_batched_tokens, name="max_num_batched_tokens")
    if inputs.cudagraph_capture_sizes is not None:
        if not inputs.cudagraph_capture_sizes:
            raise ValueError("cudagraph_capture_sizes requires at least one positive integer")
        for size in inputs.cudagraph_capture_sizes:
            validate_positive_int(size, name="cudagraph_capture_sizes")
    if inputs.block_size is not None:
        validate_positive_int(inputs.block_size, name="block_size")


def _apply_dtype_overrides(inputs: EstimatorInputs, quant_spec):
    """Apply CLI dtype overrides to the parsed quantization spec."""
    if inputs.kv_cache_dtype and inputs.kv_cache_dtype != "auto":
        quant_spec = dataclasses.replace(
            quant_spec, kv_cache_dtype=normalise_dtype(inputs.kv_cache_dtype)
        )

    if inputs.dtype and inputs.dtype != "auto":
        new_dtype = normalise_dtype(inputs.dtype)
        overrides = {"activation_dtype": new_dtype}
        if not quant_spec.is_quantized:
            overrides["weight_dtype"] = new_dtype
        quant_spec = dataclasses.replace(quant_spec, **overrides)

    return quant_spec


def prepare_summary(inputs: EstimatorInputs) -> ModelSummary:
    _validate_inputs(inputs)
    mc = _load_model_config(inputs)
    pc = _build_parallel_config(inputs)

    seq_len = inputs.max_seq_len if inputs.max_seq_len is not None else mc.max_model_len

    quant_spec = parse_quantization(mc.hf_config, cli_quantization=inputs.quantization)
    quant_spec = _apply_dtype_overrides(inputs, quant_spec)
    parameter_shapes, precomputed_bytes, expert_bytes, non_expert_bytes, \
        replicated_bytes, disk_bytes_per_element = (
            collect_parameter_shapes(
                inputs.model_id, revision=inputs.revision, use_cache=inputs.use_cache,
            )
        )

    # When on-disk dtype is larger than the runtime dtype vLLM will use
    # (e.g. F32 on disk but BF16 at runtime), scale parameter bytes down.
    runtime_bpe = quant_spec.weight_dtype.bits / 8
    if disk_bytes_per_element > runtime_bpe and not quant_spec.is_quantized:
        scale = runtime_bpe / disk_bytes_per_element
        precomputed_bytes = int(precomputed_bytes * scale)
        expert_bytes = int(expert_bytes * scale)
        non_expert_bytes = int(non_expert_bytes * scale)
        replicated_bytes = int(replicated_bytes * scale)

    # When CLI quantization implies a smaller weight dtype than what's on
    # disk, scale parameter bytes to reflect runtime GPU memory rather than
    # on-disk size.
    if inputs.quantization:
        base_spec = parse_quantization(mc.hf_config)
        if quant_spec.weight_dtype.bits < base_spec.weight_dtype.bits:
            scale = quant_spec.weight_dtype.bits / base_spec.weight_dtype.bits
            precomputed_bytes = int(precomputed_bytes * scale)
            expert_bytes = int(expert_bytes * scale)
            non_expert_bytes = int(non_expert_bytes * scale)
            replicated_bytes = int(replicated_bytes * scale)

    return ModelSummary(
        model_id=inputs.model_id,
        model_config=mc,
        parallel_config=pc,
        config=mc.hf_config,
        quantization=quant_spec,
        parameter_shapes=parameter_shapes,
        max_active_seqs=inputs.max_active_seqs,
        max_seq_len=seq_len,
        enforce_eager=inputs.enforce_eager,
        cudagraph_capture_sizes=inputs.cudagraph_capture_sizes,
        max_num_batched_tokens=inputs.max_num_batched_tokens,
        precomputed_parameter_bytes=precomputed_bytes,
        tensor_parallel_size=inputs.tensor_parallel_size,
        pipeline_parallel_size=inputs.pipeline_parallel_size,
        data_parallel_size=inputs.data_parallel_size,
        enable_expert_parallel=inputs.enable_expert_parallel,
        expert_bytes=expert_bytes,
        non_expert_bytes=non_expert_bytes,
        replicated_bytes=replicated_bytes,
        block_size=inputs.block_size,
        cpu_offload_gb=inputs.cpu_offload_gb,
    )


def estimate_memory(summary: ModelSummary) -> MemoryEstimate:
    buckets = build_memory_buckets(
        summary.config,
        summary.precomputed_parameter_bytes,
        summary.max_active_seqs,
        summary.max_seq_len,
        summary.quantization,
        enforce_eager=summary.enforce_eager,
        cudagraph_capture_sizes=summary.cudagraph_capture_sizes,
        max_num_batched_tokens=summary.max_num_batched_tokens,
        tensor_parallel_size=summary.tensor_parallel_size,
        pipeline_parallel_size=summary.pipeline_parallel_size,
        data_parallel_size=summary.data_parallel_size,
        enable_expert_parallel=summary.enable_expert_parallel,
        expert_bytes=summary.expert_bytes,
        non_expert_bytes=summary.non_expert_bytes,
        replicated_bytes=summary.replicated_bytes,
        block_size=summary.block_size,
        model_config=summary.model_config,
        parallel_config=summary.parallel_config,
        cpu_offload_gb=summary.cpu_offload_gb,
    )
    return build_estimate(
        summary.model_id, buckets,
        tensor_parallel_size=summary.tensor_parallel_size,
        pipeline_parallel_size=summary.pipeline_parallel_size,
        data_parallel_size=summary.data_parallel_size,
        enable_expert_parallel=summary.enable_expert_parallel,
    )


def estimate_from_inputs(inputs: EstimatorInputs) -> tuple[ModelSummary, MemoryEstimate]:
    summary = prepare_summary(inputs)
    estimate = estimate_memory(summary)
    return summary, estimate
