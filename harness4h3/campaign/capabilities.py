"""Trusted operator capability snapshots for controller action spaces."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from ..operators.base import OperatorRegistry
from .base import canonical_digest, canonical_json


_EVIDENCE_LEVELS = frozenset({"V0", "V1", "V2", "V3", "V4"})


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value.strip()


@dataclass(frozen=True)
class Capability:
    name: str
    category: str
    backend: str
    schema: Mapping[str, Any]
    evidence_level: str
    available: bool
    reason: str = ""

    def __post_init__(self) -> None:
        _required_string(self.name, "capability.name")
        _required_string(self.category, "capability.category")
        _required_string(self.backend, "capability.backend")
        if self.evidence_level not in _EVIDENCE_LEVELS:
            raise ValueError("unsupported capability evidence level: %s" % self.evidence_level)
        if not isinstance(self.schema, Mapping):
            raise ValueError("capability.schema must be a mapping")
        canonical_json(self.schema)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "backend": self.backend,
            "schema": copy.deepcopy(dict(self.schema)),
            "evidence_level": self.evidence_level,
            "available": bool(self.available),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CapabilitySnapshot:
    capabilities: Tuple[Capability, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        names = [item.name for item in self.capabilities]
        if any(not isinstance(item, Capability) for item in self.capabilities):
            raise ValueError("capabilities must contain Capability values")
        if len(names) != len(set(names)):
            raise ValueError("capability names must be unique")
        payload = {"capabilities": [item.to_dict() for item in self.capabilities]}
        object.__setattr__(self, "digest", canonical_digest(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {"capabilities": [item.to_dict() for item in self.capabilities], "digest": self.digest}

    def available_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.capabilities if item.available)

    def by_name(self, name: str) -> Capability:
        for item in self.capabilities:
            if item.name == name:
                return item
        raise KeyError(name)

    def is_available(self, name: str) -> bool:
        try:
            return self.by_name(name).available
        except KeyError:
            return False


def _category(name: str) -> str:
    if name in {"create_student", "prune_blocks", "prune_heads", "prune_channels"}:
        return "architecture"
    if name in {"quantize"}:
        return "quantization"
    if name in {"distill", "step_distill", "velocity_distill", "progressive_distill", "dmd2", "recovery_finetune"}:
        return "training"
    return "runtime"


def _evidence_level(category: str, status: Mapping[str, Any]) -> str:
    value = status.get("evidence_level")
    if value is not None:
        return str(value)
    return "V2" if category in {"training", "quantization"} else "V1"


class CapabilityRegistry:
    @staticmethod
    def from_operator_registry(
        registry: OperatorRegistry,
        backend_status: Mapping[str, Mapping[str, Any]],
    ) -> CapabilitySnapshot:
        capabilities = []
        for visible in registry.visible():
            name = str(visible["name"])
            status = backend_status.get(name, {})
            if not isinstance(status, Mapping):
                status = {}
            available = bool(status.get("available", False))
            reason = str(status.get("reason", "" if available else "backend_status_missing"))
            backend = str(status.get("backend", "unresolved"))
            capabilities.append(
                Capability(
                    name=name,
                    category=_category(name),
                    backend=backend,
                    schema=copy.deepcopy(dict(visible.get("input_schema") or {})),
                    evidence_level=_evidence_level(_category(name), status),
                    available=available,
                    reason=reason,
                )
            )
        return CapabilitySnapshot(tuple(capabilities))


__all__ = ["Capability", "CapabilityRegistry", "CapabilitySnapshot"]
