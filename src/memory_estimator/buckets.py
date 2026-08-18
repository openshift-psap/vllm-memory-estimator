"""Memory accounting helpers grouped by logical category."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config_utils import hidden_size
from .config_utils import intermediate_size
from .config_utils import num_layers
from .config_utils import resolve_config_attr
from .config_utils import vocab_size
from .dtype_utils import bytes_per_element
from .kv_cache_specs import estimate_kv_cache_bytes_specaware
from .quantization import QuantizationSpec
from .vllm_defaults import ACTIVATION_OVERHEAD_FACTOR
from .vllm_defaults import CUDA_GRAPH_BYTES_PER_CAPTURE
from .vllm_defaults import CUDA_GRAPH_PARAM_FRACTION
from .vllm_defaults import DEFAULT_BLOCK_SIZE
from .vllm_defaults import DEFAULT_MAX_NUM_BATCHED_TOKENS
from .vllm_defaults import WORKER_OVERHEAD_BYTES
from .vllm_defaults import WORKSPACE_FRACTION


@dataclass
class MemoryBuckets:
    parameter_bytes: float
    activation_bytes: float
    kv_cache_bytes: float
    workspace_bytes: float
    cuda_graph_bytes: float = 0.0
    block_table_bytes: float = 0.0
    worker_overhead_bytes: float = 0.0
    cpu_offload_bytes: float = 0.0
    kv_cache_spec_type: str = "full"

    @property
    def vllm_overhead_bytes(self) -> float:
        return self.cuda_graph_bytes + self.block_table_bytes + self.worker_overhead_bytes

    @property
    def total_bytes(self) -> float:
        return (
            self.parameter_bytes
            + self.activation_bytes
            + self.kv_cache_bytes
            + self.workspace_bytes
        )

    @property
    def total_with_vllm_bytes(self) -> float:
        return self.total_bytes + self.vllm_overhead_bytes


def estimate_activation_bytes(
    config,
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    max_num_batched_tokens: int | None = None,
) -> float:
    hidden = hidden_size(config)
    intermediate = intermediate_size(config, hidden)
    vocab = vocab_size(config)
    bytes_per_act = bytes_per_element(quant_spec.activation_dtype)

    if max_num_batched_tokens is None:
        max_num_batched_tokens = max(DEFAULT_MAX_NUM_BATCHED_TOKENS, max_active_seqs)
    tokens = min(max_active_seqs * max_seq_len, max_num_batched_tokens)

    hidden_buf = tokens * hidden * bytes_per_act
    ffn_buf = tokens * intermediate * bytes_per_act
    qkv_buf = tokens * hidden * 3 * bytes_per_act
    logits_buf = tokens * vocab * bytes_per_act if vocab > 0 else 0

    # Logits are computed at the output layer and don't co-exist in memory
    # with intermediate layer activations (hidden/ffn/qkv).
    peak_buffer = max(hidden_buf + max(ffn_buf, qkv_buf), logits_buf)

    moe_experts = resolve_config_attr(config, ("num_local_experts", "num_experts"))
    if moe_experts:
        topk = getattr(config, "num_experts_per_tok", 2)
        expert_hidden = getattr(config, "moe_intermediate_size", intermediate)
        peak_buffer += tokens * expert_hidden * topk * bytes_per_act * 0.5

    return peak_buffer * ACTIVATION_OVERHEAD_FACTOR


def _default_cudagraph_capture_sizes(max_num_seqs: int) -> list[int]:
    sizes = []
    bs = 1
    while bs <= max_num_seqs:
        sizes.append(bs)
        bs *= 2
    return sizes


def estimate_cuda_graph_bytes(
    per_gpu_param_bytes: float, local_layers: int, capture_sizes: list[int]
) -> float:
    per_capture = (per_gpu_param_bytes * CUDA_GRAPH_PARAM_FRACTION
                   + local_layers * CUDA_GRAPH_BYTES_PER_CAPTURE)
    return per_capture * len(capture_sizes)



def build_memory_buckets(
    config,
    parameter_bytes: float,
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    enforce_eager: bool = False,
    cudagraph_capture_sizes: list[int] | None = None,
    max_num_batched_tokens: int | None = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    expert_bytes: float = 0.0,
    non_expert_bytes: float = 0.0,
    replicated_bytes: float = 0.0,
    block_size: int | None = None,
    model_config=None,
    parallel_config=None,
    cpu_offload_gb: float = 0.0,
) -> MemoryBuckets:
    tp = tensor_parallel_size
    pp = pipeline_parallel_size
    dp = data_parallel_size

    # DP: each rank serves a fraction of sequences
    effective_seqs = math.ceil(max_active_seqs / dp) if dp > 1 else max_active_seqs
    effective_batched_tokens = max_num_batched_tokens
    if dp > 1 and max_num_batched_tokens is not None:
        effective_batched_tokens = math.ceil(max_num_batched_tokens / dp)

    effective_block_size = block_size if block_size is not None else DEFAULT_BLOCK_SIZE

    total_params = float(parameter_bytes)
    activations = estimate_activation_bytes(
        config,
        effective_seqs,
        max_seq_len,
        quant_spec,
        max_num_batched_tokens=effective_batched_tokens,
    )

    layers = num_layers(config)

    kv_result = estimate_kv_cache_bytes_specaware(
        config, effective_seqs, max_seq_len, quant_spec,
        block_size=effective_block_size,
        max_num_batched_tokens=effective_batched_tokens,
        model_config=model_config,
        parallel_config=parallel_config,
    )
    kv_cache = kv_result.total_bytes
    kv_spec_type = kv_result.spec_type
    workspace = activations * WORKSPACE_FRACTION

    # --- CPU offload ---
    offload_bytes = cpu_offload_gb * (1024 ** 3)

    # --- Per-GPU parameter bytes ---
    # Replicated tensors (vision encoders, projectors) are loaded on every
    # GPU without sharding, so they are NOT divided by TP.
    shardable_bytes = total_params - replicated_bytes
    if enable_expert_parallel and expert_bytes > 0:
        ep_size = tp * dp
        params = (non_expert_bytes / tp + expert_bytes / ep_size) / pp
    else:
        params = shardable_bytes / (tp * pp)
    params += replicated_bytes
    params = max(0.0, params - offload_bytes)

    # --- Per-GPU KV cache and activations ---
    if not kv_result.per_gpu:
        kv_cache /= (tp * pp)
    activations /= tp       # PP doesn't reduce per-stage activation peak
    workspace /= tp

    # --- vLLM overhead (CUDA graphs use per-GPU params and PP-local layers) ---
    local_layers = math.ceil(layers / pp)
    if enforce_eager:
        cuda_graph = 0.0
    else:
        if cudagraph_capture_sizes is None:
            cudagraph_capture_sizes = _default_cudagraph_capture_sizes(effective_seqs)
        cuda_graph = estimate_cuda_graph_bytes(params, local_layers, cudagraph_capture_sizes)
    blocks_per_seq = math.ceil(max_seq_len / effective_block_size)
    block_table = float(effective_seqs * blocks_per_seq * 4)
    worker = float(WORKER_OVERHEAD_BYTES)

    return MemoryBuckets(params, activations, kv_cache, workspace, cuda_graph, block_table, worker,
                         cpu_offload_bytes=offload_bytes, kv_cache_spec_type=kv_spec_type)
