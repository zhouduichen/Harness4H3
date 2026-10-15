"""Deterministic calibration samples for multi-granularity profiling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..adapters.wan.contracts import stable_digest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class CalibrationSampleSpec:
    """The fixed input identity used by every profiler probe."""

    prompt: str
    seed: int
    latent_shape: tuple[int, ...]
    timesteps: tuple[float, ...]
    sample_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.prompt).strip():
            raise ValueError("calibration prompt must not be empty")
        if not self.latent_shape or any(int(value) <= 0 for value in self.latent_shape):
            raise ValueError("calibration latent_shape must contain positive dimensions")
        if not self.timesteps or any(not math.isfinite(float(value)) for value in self.timesteps):
            raise ValueError("calibration timesteps must contain finite values")
        if not self.sample_id:
            object.__setattr__(self, "sample_id", "sample-" + self.digest()[:12])

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def digest(self) -> str:
        return stable_digest(
            {
                "prompt": self.prompt,
                "seed": int(self.seed),
                "latent_shape": list(self.latent_shape),
                "timesteps": list(self.timesteps),
            }
        )


@dataclass(frozen=True)
class CalibrationSample:
    spec: CalibrationSampleSpec
    payload: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)


class CalibrationProvider(Protocol):
    """Build adapter-compatible payloads without choosing profiler methods."""

    def build(self, spec: CalibrationSampleSpec) -> CalibrationSample:
        ...

    def batch(self, specs: Sequence[CalibrationSampleSpec]) -> tuple[CalibrationSample, ...]:
        ...


class FakeWanCalibrationProvider:
    """Small deterministic CPU provider used by profiling contract tests."""

    def __init__(self, hidden_size: int, *, device: str = "cpu") -> None:
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if str(device) != "cpu":
            raise ValueError("FakeWanCalibrationProvider is CPU-only")
        self.hidden_size = int(hidden_size)
        self.device = "cpu"

    def build(self, spec: CalibrationSampleSpec) -> CalibrationSample:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional test dependency
            raise RuntimeError("fake calibration requires torch") from exc
        generator = torch.Generator(device="cpu").manual_seed(int(spec.seed))
        payload = torch.randn(tuple(int(value) for value in spec.latent_shape), generator=generator, device="cpu")
        return CalibrationSample(
            spec=spec,
            payload=payload,
            provenance={
                "provider": "fake_wan_cpu",
                "device": "cpu",
                "offline_simulation": True,
                "hidden_size": self.hidden_size,
            },
        )

    def batch(self, specs: Sequence[CalibrationSampleSpec]) -> tuple[CalibrationSample, ...]:
        return tuple(self.build(spec) for spec in specs)


@dataclass(frozen=True)
class CalibrationCorpus:
    specs: tuple[CalibrationSampleSpec, ...]
    provider: CalibrationProvider

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("calibration corpus must contain at least one spec")
        ids = [spec.sample_id for spec in self.specs]
        if len(set(ids)) != len(ids):
            raise ValueError("calibration sample IDs must be unique")

    def samples(self) -> tuple[CalibrationSample, ...]:
        return tuple(self.provider.build(spec) for spec in self.specs)

    def digest(self) -> str:
        return stable_digest([spec.to_dict() for spec in self.specs])

    def to_dict(self) -> dict[str, Any]:
        return {
            "specs": [spec.to_dict() for spec in self.specs],
            "digest": self.digest(),
            "provider": type(self.provider).__name__,
            "provenance": {
                "offline_simulation": all(
                    bool(item.provenance.get("offline_simulation", False))
                    for item in self.samples()
                ),
            },
        }

