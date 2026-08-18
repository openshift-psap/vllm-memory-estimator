"""Quantization metadata parsing for Hugging Face / vLLM configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping

from .dtype_utils import ScalarType
from .dtype_utils import normalise_dtype


@dataclass
class QuantizationSpec:
    """Normalised quantization information for the estimator."""

    method: str | None
    weight_dtype: ScalarType
    activation_dtype: ScalarType
    kv_cache_dtype: ScalarType
    kv_cache_scale_dtype: ScalarType | None

    @property
    def is_quantized(self) -> bool:
        return self.method is not None

    def as_payload(self) -> dict[str, Any]:
        def _serialise_scalar(scalar: ScalarType | None) -> dict[str, float | str] | None:
            if scalar is None:
                return None
            return {"name": scalar.name, "bits": scalar.bits}

        return {
            "method": self.method or "dense",
            "weight_dtype": _serialise_scalar(self.weight_dtype),
            "activation_dtype": _serialise_scalar(self.activation_dtype),
            "kv_cache_dtype": _serialise_scalar(self.kv_cache_dtype),
            "kv_cache_scale_dtype": _serialise_scalar(self.kv_cache_scale_dtype),
        }


_DEFAULT_ACTIVATION = normalise_dtype("float16")
_DEFAULT_KV = normalise_dtype("float16")


def _first_group(groups: Mapping[str, Any], *keys: str) -> dict[str, Any] | None:
    for group in groups.values():
        if not isinstance(group, Mapping):
            continue
        for key in keys:
            entry = group.get(key)
            if isinstance(entry, Mapping):
                return dict(entry)
    return None


def _dtype_from_group_entry(dtype: Any, bits: Any) -> Any:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        base = dtype.strip().lower()
        if bits is not None:
            try:
                bits_value = int(float(bits))
            except (TypeError, ValueError):
                bits_value = None
            else:
                if base in {"float", "fp"}:
                    return f"float{bits_value}"
                if base in {"int", "uint"}:
                    return f"{base}{bits_value}"
        return dtype
    return dtype


def _canonicalise_quant_dict(quant_dict: dict[str, Any]) -> dict[str, Any]:
    groups = quant_dict.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        return quant_dict

    flattened = dict(quant_dict)

    weight_entry = _first_group(groups, "weights")
    if weight_entry:
        bits = weight_entry.get("num_bits") or weight_entry.get("bits")
        dtype = weight_entry.get("dtype") or weight_entry.get("type")

        if "weight_bits" not in flattened and bits is not None:
            flattened["weight_bits"] = bits
        if "bits" not in flattened and bits is not None:
            flattened["bits"] = bits
        if "weight_dtype" not in flattened and dtype is not None:
            flattened["weight_dtype"] = _dtype_from_group_entry(dtype, bits)

    activation_entry = _first_group(
        groups, "input_activations", "activations", "output_activations"
    )
    if activation_entry:
        act_dtype = activation_entry.get("dtype") or activation_entry.get("type")
        act_bits = activation_entry.get("num_bits") or activation_entry.get("bits")
        if act_dtype is not None and "activation_dtype" not in flattened:
            flattened["activation_dtype"] = _dtype_from_group_entry(act_dtype, act_bits)

    return flattened


def _extract_quant_dict(config) -> dict[str, Any] | None:
    """Return a JSON-like quantization configuration, if present."""

    for key in ("quantization_config", "compression_config"):
        value = getattr(config, key, None)
        if value is not None:
            if hasattr(value, "to_dict"):
                return _canonicalise_quant_dict(value.to_dict())
            if isinstance(value, dict):
                return _canonicalise_quant_dict(dict(value))

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return _extract_quant_dict(text_config)
    return None


def _normalise_activation_dtype(
    config, quant_dict: dict[str, Any] | None
) -> ScalarType:
    if quant_dict:
        candidate = (
            quant_dict.get("activation_dtype")
            or quant_dict.get("act_dtype")
            or quant_dict.get("compute_dtype")
        )
        if candidate:
            return normalise_dtype(candidate)
    for attr in ("dtype", "compute_dtype", "_torch_dtype"):
        value = getattr(config, attr, None)
        if value:
            return normalise_dtype(value)
    dtype = getattr(config, "torch_dtype", None)
    if dtype:
        return normalise_dtype(dtype)
    return _DEFAULT_ACTIVATION


def _normalise_weight_dtype(
    quant_dict: dict[str, Any] | None, activation_dtype: ScalarType
) -> ScalarType:
    if not quant_dict:
        return activation_dtype

    candidate = (
        quant_dict.get("weight_dtype") or quant_dict.get("dtype") or quant_dict.get("w_dtype")
    )
    bits = quant_dict.get("bits") or quant_dict.get("nbits") or quant_dict.get("weight_bits")
    try:
        bits_value = float(bits) if bits is not None else None
    except (TypeError, ValueError):
        bits_value = None
    if candidate:
        return normalise_dtype(candidate)
    elif bits_value is not None:
        return ScalarType(f"quant{bits_value:g}", float(bits_value))
    return activation_dtype


def _kv_cache_dtype(
    config, quant_dict: dict[str, Any] | None
) -> tuple[ScalarType, ScalarType | None]:
    candidate = getattr(config, "kv_cache_dtype", None)
    if candidate:
        return normalise_dtype(candidate), None
    if quant_dict:
        dtype = quant_dict.get("kv_cache_dtype") or quant_dict.get("kv_dtype")
        if dtype:
            scale_dtype = quant_dict.get("kv_cache_scaling_dtype")
            return normalise_dtype(dtype), (normalise_dtype(scale_dtype) if scale_dtype else None)
        kv_section = quant_dict.get("kv_cache")
        if isinstance(kv_section, dict):
            dtype = kv_section.get("dtype")
            scale_dtype = kv_section.get("scale_dtype")
            return (
                normalise_dtype(dtype) if dtype else _DEFAULT_KV,
                normalise_dtype(scale_dtype) if scale_dtype else None,
            )
    return _DEFAULT_KV, None


# Maps CLI -q values to (method, weight_bits) for runtime quantization
# when the model config has no quantization_config.
_CLI_QUANT_DEFAULTS: dict[str, tuple[str, float]] = {
    "fp8": ("fp8", 8.0),
    "fp8_e4m3": ("fp8", 8.0),
    "int8": ("int8", 8.0),
    "int4": ("int4", 4.0),
    "awq": ("awq", 4.0),
    "gptq": ("gptq", 4.0),
    "squeezellm": ("squeezellm", 4.0),
    "bitsandbytes": ("bitsandbytes", 4.0),
    "bnb": ("bitsandbytes", 4.0),
    "marlin": ("marlin", 4.0),
    "nvfp4": ("nvfp4", 4.0),
    "mxfp4": ("mxfp4", 4.0),
    "mxfp8": ("mxfp8", 8.0),
}


def parse_quantization(
    config, cli_quantization: str | None = None,
) -> QuantizationSpec:
    """Collect quantization settings from a Hugging Face config.

    If *cli_quantization* is provided (from ``-q`` / ``--quantization``) and
    the config has no ``quantization_config``, a synthetic spec is created
    from the CLI method name.  When the config already contains quantization
    metadata, the config takes precedence (since it has richer information).
    """

    quant_dict = _extract_quant_dict(config)
    method = None
    if quant_dict:
        method = (
            quant_dict.get("quant_method")
            or quant_dict.get("method")
            or quant_dict.get("quantization_method")
        )
        if isinstance(method, str):
            method = method.lower()

    # Apply CLI override when config has no quantization metadata
    if method is None and cli_quantization:
        cli_key = cli_quantization.strip().lower()
        if cli_key in _CLI_QUANT_DEFAULTS:
            method, bits = _CLI_QUANT_DEFAULTS[cli_key]
            quant_dict = {"quant_method": method, "bits": bits}
        else:
            method = cli_key
            quant_dict = {"quant_method": cli_key}

    activation_dtype = _normalise_activation_dtype(config, quant_dict)
    weight_dtype = _normalise_weight_dtype(quant_dict, activation_dtype)
    kv_dtype, kv_scale_dtype = _kv_cache_dtype(config, quant_dict)

    if method in ("mxfp4", "mxfp8"):
        mx_bits = 4.0 if method == "mxfp4" else 8.0
        weight_dtype = ScalarType(method, mx_bits)

    return QuantizationSpec(
        method=method,
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        kv_cache_dtype=kv_dtype,
        kv_cache_scale_dtype=kv_scale_dtype,
    )
