"""Typed, immutable target-device contract for Student acceptance."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


_FIELDS = frozenset(
    {
        "id",
        "runtime_backend",
        "max_latency_s",
        "max_memory_gb",
        "max_energy_j",
        "max_thermal_c",
        "max_model_size_gb",
        "supported_precision",
        "supported_quantization",
        "resolution",
        "frames",
        "sampling_steps",
    }
)


def _limits(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % name) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("%s must be finite and positive" % name)
    return parsed


def _choices(name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError("%s must be a non-empty sequence" % name)
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError("%s must contain unique non-empty values" % name)
    return result


def _resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().replace(" ", "").split("x")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
    else:
        raise ValueError("resolution must be [width, height] or WIDTHxHEIGHT")
    if len(parts) != 2:
        raise ValueError("resolution must contain width and height")
    try:
        result = (int(parts[0]), int(parts[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("resolution must contain integer dimensions") from exc
    if any(item <= 0 for item in result):
        raise ValueError("resolution dimensions must be positive")
    return result


@dataclass(frozen=True)
class TargetDeviceProfile:
    id: str
    runtime_backend: str
    max_latency_s: float
    max_memory_gb: float
    max_energy_j: float
    max_thermal_c: float
    max_model_size_gb: float
    supported_precision: tuple[str, ...]
    supported_quantization: tuple[str, ...]
    resolution: tuple[int, int]
    frames: int
    sampling_steps: int

    @property
    def max_edge_memory_gb(self) -> float:
        """Explicit edge-memory name; training memory is a different field."""

        return float(self.max_memory_gb)

    def __post_init__(self) -> None:
        if not str(self.id).strip() or not str(self.runtime_backend).strip():
            raise ValueError("target device id and runtime_backend must be non-empty")
        for name in (
            "max_latency_s",
            "max_memory_gb",
            "max_energy_j",
            "max_thermal_c",
            "max_model_size_gb",
        ):
            object.__setattr__(self, name, _limits(name, getattr(self, name)))
        object.__setattr__(self, "supported_precision", _choices("supported_precision", self.supported_precision))
        object.__setattr__(self, "supported_quantization", _choices("supported_quantization", self.supported_quantization))
        object.__setattr__(self, "resolution", _resolution(self.resolution))
        if int(self.frames) <= 0 or int(self.sampling_steps) <= 0:
            raise ValueError("frames and sampling_steps must be positive")
        object.__setattr__(self, "frames", int(self.frames))
        object.__setattr__(self, "sampling_steps", int(self.sampling_steps))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TargetDeviceProfile":
        if not isinstance(raw, Mapping):
            raise ValueError("target_device must be a mapping")
        unknown = sorted(set(raw) - _FIELDS)
        if unknown:
            raise ValueError("target_device has unknown field(s): %s" % ", ".join(unknown))
        missing = sorted(_FIELDS - set(raw))
        if missing:
            raise ValueError("target_device is missing field(s): %s" % ", ".join(missing))
        return cls(
            id=str(raw["id"]).strip(),
            runtime_backend=str(raw["runtime_backend"]).strip(),
            max_latency_s=_limits("max_latency_s", raw["max_latency_s"]),
            max_memory_gb=_limits("max_memory_gb", raw["max_memory_gb"]),
            max_energy_j=_limits("max_energy_j", raw["max_energy_j"]),
            max_thermal_c=_limits("max_thermal_c", raw["max_thermal_c"]),
            max_model_size_gb=_limits("max_model_size_gb", raw["max_model_size_gb"]),
            supported_precision=_choices("supported_precision", raw["supported_precision"]),
            supported_quantization=_choices("supported_quantization", raw["supported_quantization"]),
            resolution=_resolution(raw["resolution"]),
            frames=int(raw["frames"]),
            sampling_steps=int(raw["sampling_steps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supported_precision"] = list(self.supported_precision)
        value["supported_quantization"] = list(self.supported_quantization)
        value["resolution"] = list(self.resolution)
        return value


__all__ = ["TargetDeviceProfile"]
