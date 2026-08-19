"""Stub out CUDA-dependent imports so vLLM's model inspection works on CPU.

vLLM 0.27 eagerly imports the full multimodal stack (torchcodec, etc.)
during ModelConfig construction. These modules load native .so files that
require CUDA. Since the estimator only reads config metadata, we stub
them out before vLLM is imported.
"""
from __future__ import annotations

import importlib.machinery
import importlib.metadata
import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    mod.__file__ = f"<cpu_compat stub: {name}>"
    return mod


_FAKE_VERSIONS = {
    "torchcodec": "0.2.1",
}

_original_metadata_version = importlib.metadata.version


def _patched_metadata_version(name: str) -> str:
    if name in _FAKE_VERSIONS:
        return _FAKE_VERSIONS[name]
    return _original_metadata_version(name)


def patch() -> None:
    importlib.metadata.version = _patched_metadata_version

    stubs = [
        "torchcodec",
        "torchcodec._core",
        "torchcodec._core.ops",
        "torchcodec._core._metadata",
        "torchcodec._internally_replaced_utils",
        "torchcodec.decoders",
        "torchcodec.decoders._core",
        "torchcodec.encoders",
        "torchcodec.samplers",
        "torchcodec.transforms",
    ]
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = _make_stub(name)
