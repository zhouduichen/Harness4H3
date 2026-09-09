from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional

from ..h3.state import ModelState


@dataclass(frozen=True)
class CostEstimate:
    wall_time_s: float = 0.0
    gpu_hours: float = 0.0
    controller_calls: int = 0

    def __post_init__(self) -> None:
        if self.wall_time_s < 0 or self.gpu_hours < 0 or self.controller_calls < 0:
            raise ValueError("cost estimates must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CostEstimate":
        return cls(
            wall_time_s=float(raw.get("wall_time_s", 0.0)),
            gpu_hours=float(raw.get("gpu_hours", 0.0)),
            controller_calls=int(raw.get("controller_calls", 0)),
        )


@dataclass(frozen=True)
class BudgetState:
    max_iterations: int
    max_failed_experiments: int
    max_wall_time_s: Optional[float] = None
    max_gpu_hours: Optional[float] = None
    max_controller_calls: Optional[int] = None
    used_iterations: int = 0
    used_failures: int = 0
    used_wall_time_s: float = 0.0
    used_gpu_hours: float = 0.0
    used_controller_calls: int = 0

    def __post_init__(self) -> None:
        if self.max_iterations <= 0 or self.max_failed_experiments < 0:
            raise ValueError("budget iteration limit must be positive and failure limit non-negative")
        numeric = (
            self.max_wall_time_s,
            self.max_gpu_hours,
            self.max_controller_calls,
            self.used_iterations,
            self.used_failures,
            self.used_wall_time_s,
            self.used_gpu_hours,
            self.used_controller_calls,
        )
        if any(value is not None and value < 0 for value in numeric):
            raise ValueError("budget values must be non-negative")

    def can_afford(self, cost: CostEstimate, controller_calls: int = 0) -> bool:
        if self.used_iterations + 1 > self.max_iterations:
            return False
        if self.max_wall_time_s is not None and self.used_wall_time_s + cost.wall_time_s > self.max_wall_time_s:
            return False
        if self.max_gpu_hours is not None and self.used_gpu_hours + cost.gpu_hours > self.max_gpu_hours:
            return False
        calls = self.used_controller_calls + cost.controller_calls + controller_calls
        if self.max_controller_calls is not None and calls > self.max_controller_calls:
            return False
        return True

    def consume(self, cost: CostEstimate, failed: bool = False, controller_calls: int = 0) -> "BudgetState":
        return replace(
            self,
            used_iterations=self.used_iterations + 1,
            used_failures=self.used_failures + (1 if failed else 0),
            used_wall_time_s=self.used_wall_time_s + cost.wall_time_s,
            used_gpu_hours=self.used_gpu_hours + cost.gpu_hours,
            used_controller_calls=self.used_controller_calls + cost.controller_calls + controller_calls,
        )

    def stop_reason(self) -> Optional[str]:
        if self.used_failures >= self.max_failed_experiments:
            return "max_failures"
        if self.used_iterations >= self.max_iterations:
            return "max_iterations"
        if self.max_wall_time_s is not None and self.used_wall_time_s >= self.max_wall_time_s:
            return "budget_exhausted"
        if self.max_gpu_hours is not None and self.used_gpu_hours >= self.max_gpu_hours:
            return "budget_exhausted"
        if self.max_controller_calls is not None and self.used_controller_calls >= self.max_controller_calls:
            return "budget_exhausted"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BudgetState":
        return cls(**{key: value for key, value in raw.items() if key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    parent_model_id: str
    diagnosis: str
    objective: str
    hypothesis: str
    operator: str
    operator_args: Mapping[str, Any]
    expected_effects: Mapping[str, Any]
    risks: List[str]
    required_budget: Mapping[str, Any]
    acceptance: Mapping[str, Any]
    stop_conditions: Mapping[str, Any]
    rationale: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentPlan":
        required = tuple(cls.__dataclass_fields__)
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError("experiment plan missing fields: %s" % ", ".join(missing))
        return cls(
            experiment_id=str(raw["experiment_id"]),
            parent_model_id=str(raw["parent_model_id"]),
            diagnosis=str(raw["diagnosis"]),
            objective=str(raw["objective"]),
            hypothesis=str(raw["hypothesis"]),
            operator=str(raw["operator"]),
            operator_args=copy.deepcopy(dict(raw["operator_args"])),
            expected_effects=copy.deepcopy(dict(raw["expected_effects"])),
            risks=[str(item) for item in raw["risks"]],
            required_budget=copy.deepcopy(dict(raw["required_budget"])),
            acceptance=copy.deepcopy(dict(raw["acceptance"])),
            stop_conditions=copy.deepcopy(dict(raw["stop_conditions"])),
            rationale=str(raw["rationale"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareMetrics:
    latency_s: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    model_size_gb: Optional[float] = None
    energy_j: Optional[float] = None
    throughput: Optional[float] = None
    thermal: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class EvaluationResult:
    quality_score: float
    quality_metrics: Mapping[str, Any]
    hardware: HardwareMetrics
    feasible: bool
    violations: List[str] = field(default_factory=list)
    critical_regression: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluationResult":
        hardware = raw.get("hardware") or {}
        return cls(
            quality_score=float(raw["quality_score"]),
            quality_metrics=dict(raw.get("quality_metrics") or {}),
            hardware=HardwareMetrics(**dict(hardware)),
            feasible=bool(raw.get("feasible", False)),
            violations=[str(item) for item in raw.get("violations", [])],
            critical_regression=bool(raw.get("critical_regression", False)),
        )


@dataclass(frozen=True)
class OperatorResult:
    status: str
    output_state: Optional[ModelState]
    cost: CostEstimate = field(default_factory=CostEstimate)
    artifacts: List[str] = field(default_factory=list)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
