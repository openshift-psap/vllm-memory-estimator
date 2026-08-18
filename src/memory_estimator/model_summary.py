"""Aggregate representation of model-related metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Sequence

from .model_shapes import ParameterShape
from .model_shapes import count_total_parameters
from .quantization import QuantizationSpec


@dataclass
class ModelSummary:
    """Container holding the information required for memory estimation."""

    model_id: str
    config: Any
    quantization: QuantizationSpec
    parameter_shapes: Sequence[ParameterShape]
    max_active_seqs: int
    max_seq_len: int
    model_config: Any = None
    parallel_config: Any = None
    enforce_eager: bool = False
    cudagraph_capture_sizes: list[int] | None = None
    max_num_batched_tokens: int | None = None
    precomputed_parameter_bytes: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    expert_bytes: int = 0
    non_expert_bytes: int = 0
    replicated_bytes: int = 0
    block_size: int | None = None
    cpu_offload_gb: float = 0.0

    @property
    def parameter_count(self) -> int:
        return count_total_parameters(self.parameter_shapes)

    @property
    def architecture(self) -> str:
        return getattr(self.config, "model_type", "unknown")

    @property
    def total_gpus(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
