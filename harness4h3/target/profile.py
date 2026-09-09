from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml


def _optional_non_negative(raw: Mapping[str, Any], key: str) -> Optional[float]:
    value = raw.get(key)
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0:
        raise ValueError("%s must be non-negative" % key)
    return parsed


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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TargetProfile":
        hardware = dict(raw.get("hardware") or {})
        constraints = dict(raw.get("constraints") or {})
        quality = dict(raw.get("quality") or {})
        video = dict(raw.get("video") or {})
        priority = tuple(str(item) for item in raw.get("priority") or ("feasibility", "quality", "latency", "memory"))
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
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_target_profile(path: Path) -> TargetProfile:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("target profile must be a YAML mapping")
    return TargetProfile.from_dict(raw)
