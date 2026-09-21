from __future__ import annotations

import copy
import hashlib
import json
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
        if self.used_failures > 0 and self.used_failures >= self.max_failed_experiments:
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
    consumed_observation_ids: List[str] = field(default_factory=list)
    diagnosis_evidence: List[str] = field(default_factory=list)
    resource_request: Mapping[str, Any] = field(default_factory=dict)
    parent_system_id: Optional[str] = None
    repeat_for_statistics: bool = False
    # A primary planning call may carry the bounded policy for the next round.
    # Prefetch callers leave this unset; the campaign decides whether a policy
    # is allowed to become active after all normal plan gates pass.
    round_policy: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentPlan":
        if not isinstance(raw, Mapping):
            raise ValueError("experiment plan must be a mapping")
        # The three audit fields were added after the original plan format.
        # Providers receive a schema that requires them, while persisted or
        # mocked legacy plans remain readable and are rejected later by the
        # remote campaign's evidence/resource safety checks when applicable.
        required = (
            "experiment_id",
            "parent_model_id",
            "diagnosis",
            "objective",
            "hypothesis",
            "operator",
            "operator_args",
            "expected_effects",
            "risks",
            "required_budget",
            "acceptance",
            "stop_conditions",
            "rationale",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError("experiment plan missing fields: %s" % ", ".join(missing))
        mapping_fields = ("operator_args", "expected_effects", "required_budget", "acceptance", "stop_conditions")
        if any(not isinstance(raw[name], Mapping) for name in mapping_fields):
            raise ValueError("experiment plan structured fields must be mappings")
        if not isinstance(raw["risks"], (list, tuple)):
            raise ValueError("experiment plan risks must be a list")
        round_policy = raw.get("round_policy")
        if round_policy is not None and not isinstance(round_policy, Mapping):
            raise ValueError("experiment plan round_policy must be an object")
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
            consumed_observation_ids=[str(item) for item in raw.get("consumed_observation_ids", [])],
            diagnosis_evidence=[str(item) for item in raw.get("diagnosis_evidence", [])],
            resource_request=copy.deepcopy(dict(raw.get("resource_request") or {})),
            parent_system_id=str(raw["parent_system_id"]) if raw.get("parent_system_id") else None,
            repeat_for_statistics=bool(raw.get("repeat_for_statistics", False)),
            round_policy=copy.deepcopy(dict(round_policy)) if round_policy is not None else None,
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
class EvaluationRecord:
    """Canonical quality, hardware, validity, and provenance evidence."""

    quality_score: float
    quality_metrics: Mapping[str, Any]
    hardware: HardwareMetrics = field(default_factory=HardwareMetrics)
    feasible: bool = False
    violations: List[str] = field(default_factory=list)
    critical_regression: bool = False
    failure_type: Optional[str] = None
    model_id: Optional[str] = None
    system_id: Optional[str] = None
    device_id: Optional[str] = None
    task_split: Optional[str] = None
    validity: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    search_score: Optional[float] = None
    evaluation_id: str = ""

    def __post_init__(self) -> None:
        if self.evaluation_id:
            return
        identity = {
            "model_id": self.model_id,
            "system_id": self.system_id,
            "device_id": self.device_id,
            "task_split": self.task_split,
            "evaluator": self.provenance.get("evaluator") if isinstance(self.provenance, Mapping) else None,
            "benchmark_recipe": self.provenance.get("benchmark_recipe") if isinstance(self.provenance, Mapping) else None,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "evaluation_id", "E-" + digest)

    @property
    def score(self) -> float:
        """Compatibility alias for the legacy subprocess evaluator."""

        return self.quality_score

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Compatibility alias for the legacy subprocess evaluator."""

        return self.quality_metrics

    @property
    def quality(self) -> float:
        return self.quality_score

    @property
    def latency(self) -> Optional[float]:
        return self.hardware.latency_s

    @property
    def peak_memory(self) -> Optional[float]:
        return self.hardware.peak_memory_gb

    @property
    def energy(self) -> Optional[float]:
        return self.hardware.energy_j

    @property
    def model_size(self) -> Optional[float]:
        return self.hardware.model_size_gb

    @property
    def evidence(self) -> Mapping[str, Any]:
        return {"validity": dict(self.validity), "provenance": dict(self.provenance)}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluationRecord":
        if "quality_score" not in raw and "score" in raw:
            return cls(
                quality_score=float(raw["score"]),
                quality_metrics=dict(raw.get("metrics") or {}),
                hardware=HardwareMetrics(),
                feasible=bool(raw.get("feasible", False)),
                critical_regression=bool(raw.get("critical_regression", False)),
                failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
            )
        hardware = raw.get("hardware") or {}
        return cls(
            quality_score=float(raw["quality_score"]),
            quality_metrics=dict(raw.get("quality_metrics") or {}),
            hardware=HardwareMetrics(**dict(hardware)),
            feasible=bool(raw.get("feasible", False)),
            violations=[str(item) for item in raw.get("violations", [])],
            critical_regression=bool(raw.get("critical_regression", False)),
            failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
            model_id=str(raw["model_id"]) if raw.get("model_id") else None,
            system_id=str(raw["system_id"]) if raw.get("system_id") else None,
            device_id=str(raw["device_id"]) if raw.get("device_id") else None,
            task_split=str(raw["task_split"]) if raw.get("task_split") else None,
            validity=dict(raw.get("validity") or {}),
            provenance=dict(raw.get("provenance") or {}),
            search_score=float(raw["search_score"]) if raw.get("search_score") is not None else None,
            evaluation_id=str(raw.get("evaluation_id") or ""),
        )


# Compatibility import used by the current controller/archive modules.
EvaluationResult = EvaluationRecord


@dataclass(frozen=True)
class OperatorResult:
    status: str
    output_state: Optional[ModelState]
    cost: CostEstimate = field(default_factory=CostEstimate)
    artifacts: List[str] = field(default_factory=list)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    message: str = ""
    output_system: Optional[Any] = None

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
