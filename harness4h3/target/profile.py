from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


def _optional_non_negative(raw: Mapping[str, Any], key: str) -> Optional[float]:
    value = raw.get(key)
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0:
        raise ValueError("%s must be non-negative" % key)
    return parsed


OBJECTIVE_METRICS = frozenset(
    {"quality_score", "latency_s", "peak_memory_gb", "model_size_gb", "energy_j"}
)


@dataclass(frozen=True)
class ObjectiveSpec:
    """One numeric search objective; hard constraints stay on TargetProfile."""

    name: str
    direction: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in OBJECTIVE_METRICS:
            raise ValueError("unsupported objective metric: %s" % self.name)
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("objective direction must be maximize or minimize")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0:
            raise ValueError("objective weight must be finite and positive")

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


_PRIORITY_OBJECTIVES = {
    "quality": ObjectiveSpec("quality_score", "maximize"),
    "quality_score": ObjectiveSpec("quality_score", "maximize"),
    "latency": ObjectiveSpec("latency_s", "minimize"),
    "latency_s": ObjectiveSpec("latency_s", "minimize"),
    "memory": ObjectiveSpec("peak_memory_gb", "minimize"),
    "peak_memory_gb": ObjectiveSpec("peak_memory_gb", "minimize"),
    "size": ObjectiveSpec("model_size_gb", "minimize"),
    "model_size_gb": ObjectiveSpec("model_size_gb", "minimize"),
    "energy": ObjectiveSpec("energy_j", "minimize"),
    "energy_j": ObjectiveSpec("energy_j", "minimize"),
}


def _objectives_from_priority(priority: Sequence[str]) -> Tuple[ObjectiveSpec, ...]:
    result = []
    names = set()
    for value in priority:
        objective = _PRIORITY_OBJECTIVES.get(str(value).strip().lower())
        if objective is not None and objective.name not in names:
            result.append(objective)
            names.add(objective.name)
    return tuple(result)


@dataclass(frozen=True)
class TargetProfile:
    id: str
    hardware_type: str
    hardware_name: str
    hardware_backend: Optional[str] = None
    max_model_size_gb: Optional[float] = None
    max_peak_memory_gb: Optional[float] = None
    max_latency_s: Optional[float] = None
    max_energy_j: Optional[float] = None
    min_quality_score: Optional[float] = None
    max_quality_drop: Optional[float] = None
    resolution: Optional[str] = None
    duration_s: Optional[float] = None
    fps: Optional[int] = None
    priority: Tuple[str, ...] = ("feasibility", "quality", "latency", "memory")
    objectives: Tuple[ObjectiveSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.hardware_type.strip() or not self.hardware_name.strip():
            raise ValueError("target profile requires id, hardware type, and hardware name")
        numeric = (
            self.max_model_size_gb,
            self.max_peak_memory_gb,
            self.max_latency_s,
            self.max_energy_j,
            self.min_quality_score,
            self.max_quality_drop,
            self.duration_s,
        )
        if any(value is not None and value < 0 for value in numeric):
            raise ValueError("target profile numeric limits must be non-negative")
        if self.min_quality_score is not None and self.min_quality_score > 1:
            raise ValueError("min_quality_score must be between 0 and 1")
        if self.max_quality_drop is not None and self.max_quality_drop > 1:
            raise ValueError("max_quality_drop must be between 0 and 1")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if not self.priority or len(set(self.priority)) != len(self.priority):
            raise ValueError("priority must contain unique objectives")
        if not self.objectives:
            object.__setattr__(self, "objectives", _objectives_from_priority(self.priority))
        if any(not isinstance(item, ObjectiveSpec) for item in self.objectives):
            raise ValueError("objectives must contain ObjectiveSpec values")
        names = [item.name for item in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError("objectives must contain unique metrics")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TargetProfile":
        hardware = dict(raw.get("hardware") or {})
        constraints = dict(raw.get("constraints") or {})
        quality = dict(raw.get("quality") or {})
        video = dict(raw.get("video") or {})
        priority = tuple(str(item) for item in raw.get("priority") or ("feasibility", "quality", "latency", "memory"))
        raw_objectives = raw.get("objectives")
        objectives = []
        if isinstance(raw_objectives, Mapping):
            raw_objectives = [
                dict(value, name=name) if isinstance(value, Mapping) else {"name": name, "direction": value}
                for name, value in raw_objectives.items()
            ]
        if raw_objectives is not None:
            if not isinstance(raw_objectives, (list, tuple)):
                raise ValueError("objectives must be a list or mapping")
            for item in raw_objectives:
                if isinstance(item, ObjectiveSpec):
                    objectives.append(item)
                elif isinstance(item, Mapping):
                    objectives.append(
                        ObjectiveSpec(
                            name=str(item.get("name", "")),
                            direction=str(item.get("direction", "")),
                            weight=float(item.get("weight", 1.0)),
                        )
                    )
                else:
                    raise ValueError("objective entries must be mappings")
        return cls(
            id=str(raw.get("id", "")).strip(),
            hardware_type=str(hardware.get("type", "")).strip(),
            hardware_name=str(hardware.get("name", "")).strip(),
            hardware_backend=str(hardware["backend"]).strip() if hardware.get("backend") else None,
            max_model_size_gb=_optional_non_negative(constraints, "max_model_size_gb"),
            max_peak_memory_gb=_optional_non_negative(constraints, "max_peak_memory_gb"),
            max_latency_s=_optional_non_negative(constraints, "max_latency_s"),
            max_energy_j=_optional_non_negative(constraints, "max_energy_j"),
            min_quality_score=_optional_non_negative(quality, "min_quality_score"),
            max_quality_drop=_optional_non_negative(quality, "max_quality_drop"),
            resolution=str(video["resolution"]) if video.get("resolution") is not None else None,
            duration_s=_optional_non_negative(video, "duration_s"),
            fps=int(video["fps"]) if video.get("fps") is not None else None,
            priority=priority,
            objectives=tuple(objectives),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def objective_score(self, metrics: Mapping[str, Any]) -> float:
        """Return a context-only weighted signed score, never a feasibility decision."""

        total = 0.0
        for objective in self.objectives:
            value = metrics.get(objective.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            signed = float(value) if objective.direction == "maximize" else -float(value)
            total += float(objective.weight) * signed
        return total


def load_target_profile(path: Path) -> TargetProfile:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("target profile must be a YAML mapping")
    return TargetProfile.from_dict(raw)
