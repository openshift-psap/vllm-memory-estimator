"""vLLM-specific constants used in memory estimation.

These values are derived from vLLM's runtime behavior and should be updated
when vLLM changes its memory accounting. Current values are validated against
vLLM v0.8–v0.22.

See also: vllm/worker/worker.py, vllm/engine/arg_utils.py
"""

from __future__ import annotations

# --- CUDA graph capture ---
# Approximate memory overhead per CUDA graph capture, per layer.
CUDA_GRAPH_BYTES_PER_CAPTURE = 2 * 1024 * 1024  # 2 MiB

# Fraction of parameter bytes attributed to CUDA graph state.
CUDA_GRAPH_PARAM_FRACTION = 0.02  # 2%

# --- Worker process ---
# Fixed overhead for the vLLM worker process state (Python objects, NCCL
# buffers, internal bookkeeping).
WORKER_OVERHEAD_BYTES = 300 * 1024 * 1024  # 300 MiB

# --- Paged attention ---
# Default block size for vLLM's paged attention block manager.
DEFAULT_BLOCK_SIZE = 16

# --- Activations ---
# Default max number of batched tokens per forward pass when not specified.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048

# Multiplier applied to activation estimate to account for framework overhead
# (PyTorch autograd buffers, temporary allocations, etc.).
ACTIVATION_OVERHEAD_FACTOR = 1.10  # 10%

# Workspace is estimated as a fraction of activation memory.
WORKSPACE_FRACTION = 0.05  # 5%

# --- Confidence range factors ---
# Each component gets lower/upper multipliers to form a confidence range.
PARAM_RANGE = (0.95, 1.05)
ACTIVATION_RANGE = (0.50, 2.00)
KV_CACHE_RANGE = (0.98, 1.02)
WORKSPACE_RANGE = (0.40, 3.00)
WORKSPACE_FLOOR_GIB = 0.05
VLLM_OVERHEAD_RANGE = (0.50, 2.00)
VLLM_OVERHEAD_FLOOR_GIB = 0.10
