"""Wan-shaped DiT model adapters.

The fake backend is intentionally small and deterministic so structural
compression contracts can be tested without a checkpoint or CUDA.  The
Diffusers backend is imported lazily and fails closed when its optional
dependencies or model layout are unavailable.
"""

from __future__ import annotations

from typing import Any

from .contracts import (
    BlockRef,
    CheckpointArtifact,
    ChannelPruningConfig,
    CompressionResult,
    ModelStructure,
    StructuralChange,
    WanModelAdapter,
)
from .fake import FakeWanDiTAdapter


def build_wan_adapter(backend: str = "fake", **kwargs: Any) -> WanModelAdapter:
    normalized = str(backend).strip().lower()
    if normalized == "fake":
        return FakeWanDiTAdapter(**kwargs)
    if normalized in {"diffusers", "wan", "real"}:
        from .diffusers import DiffusersWanAdapter

        return DiffusersWanAdapter(**kwargs)
    raise ValueError("unsupported_adapter_backend: %s" % backend)


__all__ = [
    "BlockRef",
    "CheckpointArtifact",
    "ChannelPruningConfig",
    "CompressionResult",
    "DiffusersWanAdapter",
    "FakeWanDiTAdapter",
    "ModelStructure",
    "StructuralChange",
    "WanModelAdapter",
    "build_wan_adapter",
]


def __getattr__(name: str) -> Any:
    if name == "DiffusersWanAdapter":
        from .diffusers import DiffusersWanAdapter

        return DiffusersWanAdapter
    raise AttributeError(name)

