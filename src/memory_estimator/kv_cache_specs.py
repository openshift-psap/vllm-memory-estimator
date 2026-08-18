"""KV cache estimation using vLLM's actual KVCacheSpec classes.

Detects the spec type from HF model config attributes, constructs the
matching vLLM spec object, and delegates byte-counting to its
``page_size_bytes`` / ``max_memory_usage_bytes()`` methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from .config_utils import attention_chunk_size as _cfg_attention_chunk_size
from .config_utils import head_dim
from .config_utils import hidden_size
from .config_utils import intermediate_size
from .config_utils import kv_lora_rank as _cfg_kv_lora_rank
from .config_utils import layers_block_type as _cfg_layers_block_type
from .config_utils import mamba_d_conv as _cfg_mamba_d_conv
from .config_utils import mamba_d_state as _cfg_mamba_d_state
from .config_utils import mamba_expand as _cfg_mamba_expand
from .config_utils import mamba_head_dim as _cfg_mamba_head_dim
from .config_utils import mamba_n_groups as _cfg_mamba_n_groups
from .config_utils import mamba_n_heads as _cfg_mamba_n_heads
from .config_utils import model_type as _cfg_model_type
from .config_utils import no_rope_layers as _cfg_no_rope_layers
from .config_utils import num_attention_heads
from .config_utils import num_layers
from .config_utils import qk_rope_head_dim as _cfg_qk_rope_head_dim
from .config_utils import sliding_window as _cfg_sliding_window
from .dtype_utils import bytes_per_element
from .quantization import QuantizationSpec
from .vllm_defaults import DEFAULT_BLOCK_SIZE
from .vllm_defaults import DEFAULT_MAX_NUM_BATCHED_TOKENS

# Mamba2 models that use Mamba2-style state shapes.
_MAMBA2_MODEL_TYPES = frozenset({
    "mamba2", "bamba", "falcon_h1", "plamo2", "falcon_hybrid",
    "granitemoehybrid", "nemotron_h",
})

# Mamba1 models that use Mamba1-style state shapes.
_MAMBA1_MODEL_TYPES = frozenset({
    "mamba", "jamba", "falcon_mamba",
})


@dataclass
class KVCacheResult:
    """Wraps a KV cache byte estimate with the spec type used."""

    total_bytes: float
    spec_type: str
    layer_groups: list[LayerGroupEstimate] = field(default_factory=list)
    per_gpu: bool = False


@dataclass
class LayerGroupEstimate:
    """Per-group breakdown for hybrid models."""

    spec_type: str
    num_layers: int
    bytes: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_stub(max_model_len: int, max_num_batched_tokens: int) -> SimpleNamespace:
    """Lightweight duck-typed VllmConfig for spec.max_memory_usage_bytes()."""
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
    )


def _kv_torch_dtype(quant_spec: QuantizationSpec) -> torch.dtype:
    """Get torch.dtype for KV cache from the quantization spec."""
    td = quant_spec.kv_cache_dtype.torch_dtype
    if td is not None:
        return td
    return torch.bfloat16


def _quant_scale_bytes(
    layers: int,
    kv_heads: int,
    quant_spec: QuantizationSpec,
) -> float:
    """Extra bytes for per-head quantization scales (K and V)."""
    if quant_spec.kv_cache_dtype.bits <= 8 and quant_spec.kv_cache_scale_dtype:
        scales = layers * kv_heads * 2
        return scales * bytes_per_element(quant_spec.kv_cache_scale_dtype)
    return 0.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _is_mamba_model(config: Any) -> bool:
    """Check if the model is a pure Mamba (non-hybrid) model."""
    mt = _cfg_model_type(config)
    return mt in (_MAMBA1_MODEL_TYPES | _MAMBA2_MODEL_TYPES)


def _is_hybrid_model(config: Any) -> bool:
    """Check if the model has mixed layer types (attention + mamba, etc.)."""
    if _cfg_layers_block_type(config) is not None:
        return True
    if _cfg_no_rope_layers(config) is not None:
        return True
    return False


def _mamba_version(config: Any) -> int:
    """Return 1 for Mamba1 models, 2 for Mamba2 models, 0 if not Mamba."""
    mt = _cfg_model_type(config)
    if mt in _MAMBA1_MODEL_TYPES:
        return 1
    if mt in _MAMBA2_MODEL_TYPES:
        return 2
    return 0


def detect_kv_spec_type(config: Any) -> str:
    """Determine the KV cache spec type from model config attributes."""
    if _is_hybrid_model(config):
        return "hybrid"
    if _is_mamba_model(config):
        return "mamba"
    lora_rank = _cfg_kv_lora_rank(config)
    rope_dim = _cfg_qk_rope_head_dim(config)
    if lora_rank > 0 and rope_dim > 0:
        return "mla"
    window = _cfg_sliding_window(config)
    if window is not None:
        return "sliding_window"
    chunk = _cfg_attention_chunk_size(config)
    if chunk is not None:
        return "chunked_local"
    return "full"


# ---------------------------------------------------------------------------
# Spec construction helpers
# ---------------------------------------------------------------------------

def _full_attention_bytes(
    layers: int,
    max_active_seqs: int,
    kv_heads: int,
    hdim: int,
    quant_spec: QuantizationSpec,
    stub: SimpleNamespace,
    block_size: int,
) -> float:
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_size=hdim,
        dtype=_kv_torch_dtype(quant_spec),
    )
    per_layer = spec.max_memory_usage_bytes(stub)
    total = per_layer * layers * max_active_seqs
    total += _quant_scale_bytes(layers, kv_heads, quant_spec)
    return total


def _sliding_window_bytes(
    layers: int,
    max_active_seqs: int,
    kv_heads: int,
    hdim: int,
    quant_spec: QuantizationSpec,
    stub: SimpleNamespace,
    block_size: int,
    window: int,
) -> float:
    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_size=hdim,
        dtype=_kv_torch_dtype(quant_spec),
        sliding_window=window,
    )
    per_layer = spec.max_memory_usage_bytes(stub)
    total = per_layer * layers * max_active_seqs
    total += _quant_scale_bytes(layers, kv_heads, quant_spec)
    return total


def _mla_bytes(
    layers: int,
    max_active_seqs: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    quant_spec: QuantizationSpec,
    stub: SimpleNamespace,
    block_size: int,
    fp8_ds_mla: bool,
) -> float:
    cache_dtype_str = "fp8_ds_mla" if fp8_ds_mla else None
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=kv_lora_rank + qk_rope_head_dim,
        dtype=_kv_torch_dtype(quant_spec),
        cache_dtype_str=cache_dtype_str,
    )
    per_layer = spec.max_memory_usage_bytes(stub)
    total = per_layer * layers * max_active_seqs
    if not fp8_ds_mla:
        total += _quant_scale_bytes(layers, 1, quant_spec)
    return total


def _chunked_local_bytes(
    layers: int,
    max_active_seqs: int,
    kv_heads: int,
    hdim: int,
    quant_spec: QuantizationSpec,
    stub: SimpleNamespace,
    block_size: int,
    chunk_size: int,
) -> float:
    spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_size=hdim,
        dtype=_kv_torch_dtype(quant_spec),
        attention_chunk_size=chunk_size,
    )
    per_layer = spec.max_memory_usage_bytes(stub)
    total = per_layer * layers * max_active_seqs
    total += _quant_scale_bytes(layers, kv_heads, quant_spec)
    return total


def _sliding_window_mla_bytes(
    layers: int,
    max_active_seqs: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    quant_spec: QuantizationSpec,
    stub: SimpleNamespace,
    block_size: int,
    window: int,
    fp8_ds_mla: bool,
) -> float:
    cache_dtype_str = "fp8_ds_mla" if fp8_ds_mla else None
    spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=kv_lora_rank + qk_rope_head_dim,
        dtype=_kv_torch_dtype(quant_spec),
        sliding_window=window,
        cache_dtype_str=cache_dtype_str,
    )
    per_layer = spec.max_memory_usage_bytes(stub)
    total = per_layer * layers * max_active_seqs
    if not fp8_ds_mla:
        total += _quant_scale_bytes(layers, 1, quant_spec)
    return total


def _build_mamba_spec(
    config: Any,
    block_size: int,
    quant_spec: QuantizationSpec,
) -> MambaSpec:
    """Construct a MambaSpec from model config attributes."""
    hidden = hidden_size(config)
    inter = intermediate_size(config, hidden)
    d_state = _cfg_mamba_d_state(config)
    d_conv = _cfg_mamba_d_conv(config)
    expand = _cfg_mamba_expand(config)
    mamba_inter = int(hidden * expand) if expand > 0 else inter

    td = _kv_torch_dtype(quant_spec)
    mver = _mamba_version(config)

    if mver == 2 or d_state >= 64:
        m_heads = _cfg_mamba_n_heads(config) or 1
        m_hdim = _cfg_mamba_head_dim(config) or d_state
        n_groups = _cfg_mamba_n_groups(config)
        conv_dim = mamba_inter + 2 * n_groups * d_state
        conv_shape = ((d_conv - 1) * conv_dim,)
        temporal_shape = (m_heads * m_hdim * d_state,)
    else:
        conv_shape = ((d_conv - 1) * mamba_inter,)
        temporal_shape = (mamba_inter * d_state,)

    mtype = "mamba2" if (mver == 2 or d_state >= 64) else "mamba1"

    return MambaSpec(
        block_size=block_size,
        shapes=(conv_shape, temporal_shape),
        dtypes=(td, td),
        mamba_type=mtype,
    )


def _mamba_bytes(
    config: Any,
    layers: int,
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    block_size: int,
) -> float:
    spec = _build_mamba_spec(config, block_size, quant_spec)
    stub = _config_stub(max_seq_len, max_seq_len)
    per_layer = spec.max_memory_usage_bytes(stub)
    return per_layer * layers * max_active_seqs


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _compute_hybrid_kv_bytes(
    config: Any,
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    block_size: int,
    batched: int,
    model_config=None,
    parallel_config=None,
) -> KVCacheResult:
    """Compute KV cache for hybrid models with mixed layer types."""
    if model_config is not None:
        hdim = model_config.get_head_size()
        if parallel_config is not None:
            kv_heads = model_config.get_num_kv_heads(parallel_config)
        else:
            kv_heads = model_config.get_total_num_kv_heads()
    else:
        hidden = hidden_size(config)
        n_heads, kv_heads = num_attention_heads(config)
        hdim = head_dim(config, hidden, n_heads)

    block_types = _cfg_layers_block_type(config)
    nope_mask = _cfg_no_rope_layers(config)

    attn_layers = 0
    mamba_layers = 0
    chunked_layers = 0

    if block_types is not None:
        for bt in block_types:
            if bt.lower() in ("mamba", "mamba1", "mamba2"):
                mamba_layers += 1
            else:
                attn_layers += 1
    elif nope_mask is not None:
        chunk = _cfg_attention_chunk_size(config)
        for val in nope_mask:
            if val == 0:
                attn_layers += 1
            else:
                chunked_layers += 1
        if chunk is None:
            attn_layers += chunked_layers
            chunked_layers = 0

    stub = _config_stub(max_seq_len, batched)
    groups: list[LayerGroupEstimate] = []
    total = 0.0

    if attn_layers > 0:
        b = _full_attention_bytes(
            attn_layers, max_active_seqs, kv_heads, hdim, quant_spec, stub, block_size,
        )
        total += b
        groups.append(LayerGroupEstimate("full", attn_layers, b))

    if chunked_layers > 0:
        chunk = _cfg_attention_chunk_size(config)
        assert chunk is not None
        b = _chunked_local_bytes(
            chunked_layers, max_active_seqs, kv_heads, hdim, quant_spec, stub,
            block_size, chunk,
        )
        total += b
        groups.append(LayerGroupEstimate("chunked_local", chunked_layers, b))

    if mamba_layers > 0:
        b = _mamba_bytes(
            config, mamba_layers, max_active_seqs, max_seq_len, quant_spec, block_size,
        )
        total += b
        groups.append(LayerGroupEstimate("mamba", mamba_layers, b))

    tp_applied = model_config is not None and parallel_config is not None
    return KVCacheResult(total_bytes=total, spec_type="hybrid", layer_groups=groups,
                         per_gpu=tp_applied)


def _detect_spec_type_from_model_config(model_config) -> str:
    """Use vLLM's ModelConfig for accurate spec type detection."""
    if model_config.is_hybrid:
        return "hybrid"
    if getattr(model_config, "use_mla", False):
        sw = model_config.get_sliding_window()
        if sw is not None:
            return "sliding_window_mla"
        return "mla"
    if getattr(model_config, "has_inner_state", False):
        return "mamba"
    sw = model_config.get_sliding_window()
    if sw is not None:
        return "sliding_window"
    chunk = _cfg_attention_chunk_size(model_config.hf_text_config)
    if chunk is not None:
        return "chunked_local"
    return "full"


def estimate_kv_cache_bytes_specaware(
    config: Any,
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    block_size: int | None = None,
    max_num_batched_tokens: int | None = None,
    model_config=None,
    parallel_config=None,
) -> KVCacheResult:
    """Dispatch to the right vLLM KV cache spec based on model config.

    When *model_config* (``vllm.config.ModelConfig``) is provided, uses its
    detection logic for accurate hybrid/MLA/Mamba classification and TP-aware
    KV head counts.  Falls back to config-attribute heuristics otherwise.

    MLA and Mamba specs return ``per_gpu=True`` because their state is not
    sharded across TP.
    """
    bs = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
    batched = (max_num_batched_tokens
               if max_num_batched_tokens is not None
               else max(DEFAULT_MAX_NUM_BATCHED_TOKENS, max_active_seqs))

    if model_config is not None:
        spec_type = _detect_spec_type_from_model_config(model_config)
    else:
        spec_type = detect_kv_spec_type(config)

    if spec_type == "hybrid":
        return _compute_hybrid_kv_bytes(
            config, max_active_seqs, max_seq_len, quant_spec, bs, batched,
            model_config=model_config, parallel_config=parallel_config,
        )

    # When model_config + parallel_config are available, kv_heads is already
    # TP-aware (max(1, total // tp)), so all results are per_gpu=True.
    tp_applied = model_config is not None and parallel_config is not None
    if model_config is not None:
        layers_ = model_config.model_arch_config.total_num_hidden_layers
        hdim = model_config.get_head_size()
        if parallel_config is not None:
            kv_heads = model_config.get_num_kv_heads(parallel_config)
        else:
            kv_heads = model_config.get_total_num_kv_heads()
    else:
        hidden = hidden_size(config)
        layers_ = num_layers(config)
        n_heads, kv_heads = num_attention_heads(config)
        hdim = head_dim(config, hidden, n_heads)

    stub = _config_stub(max_seq_len, batched)

    if spec_type == "mamba":
        total = _mamba_bytes(
            config, layers_, max_active_seqs, max_seq_len, quant_spec, bs,
        )
        return KVCacheResult(total_bytes=total, spec_type="mamba", per_gpu=True)

    if spec_type == "mla":
        lora_rank = _cfg_kv_lora_rank(config)
        rope_dim = _cfg_qk_rope_head_dim(config)
        if model_config is not None and lora_rank == 0:
            lora_rank = hdim - (rope_dim or 0)
        use_fp8 = quant_spec.kv_cache_dtype.bits == 8
        total = _mla_bytes(
            layers_, max_active_seqs, lora_rank, rope_dim, quant_spec,
            stub, bs, fp8_ds_mla=use_fp8,
        )
        return KVCacheResult(total_bytes=total, spec_type="mla", per_gpu=True)

    if spec_type == "sliding_window_mla":
        lora_rank = _cfg_kv_lora_rank(config)
        rope_dim = _cfg_qk_rope_head_dim(config)
        if model_config is not None and lora_rank == 0:
            lora_rank = hdim - (rope_dim or 0)
        window = _cfg_sliding_window(config)
        if window is None and model_config is not None:
            window = model_config.get_sliding_window()
        assert window is not None
        use_fp8 = quant_spec.kv_cache_dtype.bits == 8
        total = _sliding_window_mla_bytes(
            layers_, max_active_seqs, lora_rank, rope_dim, quant_spec,
            stub, bs, window, fp8_ds_mla=use_fp8,
        )
        return KVCacheResult(total_bytes=total, spec_type="sliding_window_mla", per_gpu=True)

    if spec_type == "sliding_window":
        window = _cfg_sliding_window(config)
        if window is None and model_config is not None:
            window = model_config.get_sliding_window()
        assert window is not None
        total = _sliding_window_bytes(
            layers_, max_active_seqs, kv_heads, hdim, quant_spec, stub, bs, window,
        )
        return KVCacheResult(total_bytes=total, spec_type="sliding_window", per_gpu=tp_applied)

    if spec_type == "chunked_local":
        chunk = _cfg_attention_chunk_size(config)
        assert chunk is not None
        total = _chunked_local_bytes(
            layers_, max_active_seqs, kv_heads, hdim, quant_spec, stub, bs, chunk,
        )
        return KVCacheResult(total_bytes=total, spec_type="chunked_local", per_gpu=tp_applied)

    # Default: full attention
    total = _full_attention_bytes(
        layers_, max_active_seqs, kv_heads, hdim, quant_spec, stub, bs,
    )
    return KVCacheResult(total_bytes=total, spec_type="full", per_gpu=tp_applied)
