"""Typed sigma grids for Wan Flow-Match training and inference."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class WanSigmaSchedule:
    """A Wan model-evaluation grid, including its terminal zero sigma."""

    mode: str
    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...]
    num_train_timesteps: int
    shift: float
    source_nfe: int | None = None

    def __post_init__(self) -> None:
        if not self.mode:
            raise ValueError("Wan schedule mode must be non-empty")
        if len(self.sigmas) < 2:
            raise ValueError("Wan schedule needs at least one model evaluation")
        if len(self.timesteps) != len(self.sigmas) - 1:
            raise ValueError("Wan schedule timesteps must match non-terminal sigmas")
        if self.sigmas[-1] != 0.0:
            raise ValueError("Wan schedule must end at sigma 0")
        if any(left <= right for left, right in zip(self.sigmas, self.sigmas[1:])):
            raise ValueError("Wan schedule sigmas must be strictly decreasing")
        expected = tuple(float(sigma) * float(self.num_train_timesteps) for sigma in self.sigmas[:-1])
        if any(not math.isclose(actual, wanted, rel_tol=1e-5, abs_tol=1e-4) for actual, wanted in zip(self.timesteps, expected)):
            raise ValueError("Wan schedule timesteps must equal sigma*num_train_timesteps")

    @property
    def nfe(self) -> int:
        return len(self.sigmas) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sigmas": list(self.sigmas),
            "timesteps": list(self.timesteps),
            "nfe": self.nfe,
            "num_train_timesteps": self.num_train_timesteps,
            "shift": self.shift,
            "source_nfe": self.source_nfe,
        }


def _scheduler_config_value(scheduler: Any, name: str, default: Any) -> Any:
    config = getattr(scheduler, "config", None)
    return getattr(config, name, default)


def _schedule_from_sigmas(
    *,
    mode: str,
    sigmas: tuple[float, ...],
    num_train_timesteps: int,
    shift: float,
    source_nfe: int | None = None,
    timesteps: tuple[float, ...] | None = None,
) -> WanSigmaSchedule:
    model_timesteps = timesteps or tuple(float(sigma) * float(num_train_timesteps) for sigma in sigmas[:-1])
    return WanSigmaSchedule(
        mode=mode,
        sigmas=tuple(float(sigma) for sigma in sigmas),
        timesteps=tuple(float(value) for value in model_timesteps),
        num_train_timesteps=int(num_train_timesteps),
        shift=float(shift),
        source_nfe=source_nfe,
    )


def native_wan_schedule(scheduler: Any, nfe: int) -> WanSigmaSchedule:
    """Read the checkpoint scheduler's native grid without mutating it."""

    count = int(nfe)
    if count <= 0:
        raise ValueError("Wan schedule nfe must be positive")
    probe = deepcopy(scheduler)
    probe.set_timesteps(count, device="cpu")
    sigmas = tuple(float(value) for value in probe.sigmas.detach().cpu().tolist())
    expected = tuple(float(value) for value in probe.timesteps.detach().cpu().tolist())
    schedule = _schedule_from_sigmas(
        mode="native",
        sigmas=sigmas,
        num_train_timesteps=int(_scheduler_config_value(probe, "num_train_timesteps", 1000)),
        shift=float(_scheduler_config_value(probe, "shift", 1.0)),
        timesteps=expected,
    )
    return schedule


def coarsen_wan_schedule(source: WanSigmaSchedule, factor: int) -> WanSigmaSchedule:
    """Select exact binary-aligned endpoints from a source Wan grid."""

    divisor = int(factor)
    if divisor <= 1:
        raise ValueError("Wan schedule coarsening factor must be greater than 1")
    if source.nfe % divisor:
        raise ValueError("Wan schedule coarsening factor must divide source nfe")
    selected = source.sigmas[::divisor]
    if selected[-1] != 0.0:
        selected = selected + (0.0,)
    return _schedule_from_sigmas(
        mode="coarsened_from_source",
        sigmas=selected,
        num_train_timesteps=source.num_train_timesteps,
        shift=source.shift,
        source_nfe=source.nfe,
    )


def training_uniform_wan_schedule(nfe: int, num_train_timesteps: int = 1000) -> WanSigmaSchedule:
    """Return the unshifted uniform grid used only as an adapter fallback."""

    count = int(nfe)
    if count <= 0:
        raise ValueError("Wan schedule nfe must be positive")
    sigmas = tuple(1.0 - (index / float(count)) for index in range(count + 1))
    return _schedule_from_sigmas(
        mode="training_uniform",
        sigmas=sigmas,
        num_train_timesteps=int(num_train_timesteps),
        shift=1.0,
    )
