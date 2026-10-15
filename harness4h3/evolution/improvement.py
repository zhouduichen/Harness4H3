"""Recursive algorithm-improvement requests and controlled validation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from ..adapters.wan.contracts import stable_digest
from .candidate import AlgorithmCandidate
from ..sandbox.runner import AlgorithmSandbox, SandboxResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class AlgorithmImprovementRequest:
    parent: AlgorithmCandidate
    parent_id: Optional[str]
    profile: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    failures: tuple[str, ...] = ()
    search_space: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_algorithm_id": self.parent.algorithm_id,
            "parent_source_sha256": self.parent.source_sha256,
            "parent_source": self.parent.source,
            "parent_id": self.parent_id,
            "profile": _jsonable(self.profile),
            "history": _jsonable(self.history),
            "failures": list(self.failures),
            "search_space": _jsonable(self.search_space),
        }


@dataclass(frozen=True)
class AlgorithmValidationRecord:
    algorithm_id: str
    parent_algorithm_id: str
    parent_source_sha256: str
    candidate_source_sha256: str
    valid: bool
    evidence_level: str
    controlled_conditions_digest: str
    parent_plan: Optional[Mapping[str, Any]] = None
    candidate_plan: Optional[Mapping[str, Any]] = None
    parent_metrics: Mapping[str, Any] = field(default_factory=dict)
    candidate_metrics: Mapping[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.algorithm_id.startswith("A") or not self.parent_algorithm_id.startswith("A"):
            raise ValueError("algorithm validation requires algorithm IDs")
        for name in ("parent_source_sha256", "candidate_source_sha256", "controlled_conditions_digest"):
            if len(str(getattr(self, name))) != 64:
                raise ValueError("%s must be a SHA-256 digest" % name)
        if self.evidence_level not in {"sandbox_only", "cpu_fake", "real_measured"}:
            raise ValueError("unsupported algorithm validation evidence level")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


PlanEvaluator = Callable[[SandboxResult], Mapping[str, Any]]


class AlgorithmValidator:
    """Compare two source versions under the same controlled conditions."""

    def validate(
        self,
        parent: AlgorithmCandidate,
        child: AlgorithmCandidate,
        sandbox: AlgorithmSandbox,
        profile: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        evaluate_plan: PlanEvaluator,
        *,
        budget: Optional[Mapping[str, Any]] = None,
    ) -> AlgorithmValidationRecord:
        if child.parent_algorithm_id != parent.algorithm_id:
            return self._failed(parent, child, profile, history, "parent_algorithm_mismatch", budget=budget)
        if child.parent_source_sha256 != parent.source_sha256:
            return self._failed(parent, child, profile, history, "parent_source_hash_mismatch", budget=budget)
        parent_result = sandbox.run(parent.source, profile, history)
        child_result = sandbox.run(child.source, profile, history)
        if parent_result.status != "success" or parent_result.plan is None:
            return self._failed(parent, child, profile, history, "parent_algorithm_execution_failed", budget=budget)
        if child_result.status != "success" or child_result.plan is None:
            return self._failed(parent, child, profile, history, "candidate_algorithm_execution_failed", budget=budget)
        try:
            parent_metrics = dict(evaluate_plan(parent_result))
            child_metrics = dict(evaluate_plan(child_result))
        except (RuntimeError, TypeError, ValueError) as exc:
            return self._failed(parent, child, profile, history, "algorithm_comparison_failed:%s" % exc, budget=budget)

        reasons: list[str] = []
        if parent_metrics.get("valid") is not True or child_metrics.get("valid") is not True:
            reasons.append("comparison_evidence_invalid")
        parent_score = _score(parent_metrics)
        child_score = _score(child_metrics)
        if parent_score is None or child_score is None:
            reasons.append("comparison_score_missing")
        elif child_score < parent_score:
            reasons.append("candidate_score_regressed")
        elif child_score == parent_score and not _lower_cost(child_metrics, parent_metrics):
            reasons.append("candidate_has_no_measured_improvement")
        evidence_level = _evidence_level(parent_metrics, child_metrics)
        return AlgorithmValidationRecord(
            algorithm_id=child.algorithm_id,
            parent_algorithm_id=parent.algorithm_id,
            parent_source_sha256=parent.source_sha256,
            candidate_source_sha256=child.source_sha256,
            valid=not reasons,
            evidence_level=evidence_level,
            controlled_conditions_digest=_conditions_digest(profile, history, budget),
            parent_plan=parent_result.plan.to_dict(),
            candidate_plan=child_result.plan.to_dict(),
            parent_metrics=parent_metrics,
            candidate_metrics=child_metrics,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _failed(
        parent: AlgorithmCandidate,
        child: AlgorithmCandidate,
        profile: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        reason: str,
        *,
        budget: Optional[Mapping[str, Any]],
    ) -> AlgorithmValidationRecord:
        return AlgorithmValidationRecord(
            algorithm_id=child.algorithm_id,
            parent_algorithm_id=parent.algorithm_id,
            parent_source_sha256=parent.source_sha256,
            candidate_source_sha256=child.source_sha256,
            valid=False,
            evidence_level="sandbox_only",
            controlled_conditions_digest=_conditions_digest(profile, history, budget),
            reasons=(reason,),
        )


def _score(metrics: Mapping[str, Any]) -> Optional[float]:
    for key in ("search_score", "quality_score"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _lower_cost(candidate: Mapping[str, Any], parent: Mapping[str, Any]) -> bool:
    for key in ("model_size_gb", "latency_s", "cost_units"):
        left = candidate.get(key)
        right = parent.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and float(left) < float(right):
            return True
    return False


def _evidence_level(parent: Mapping[str, Any], child: Mapping[str, Any]) -> str:
    if bool(parent.get("offline_simulation")) or bool(child.get("offline_simulation")):
        return "cpu_fake"
    if str(parent.get("evaluation_status", "")).startswith("measured") and str(child.get("evaluation_status", "")).startswith("measured"):
        return "real_measured"
    return "sandbox_only"


def _conditions_digest(
    profile: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    budget: Optional[Mapping[str, Any]],
) -> str:
    history_ids = [
        str(item.get("experience_id", item.get("round_id")))
        for item in history
        if isinstance(item, Mapping) and (item.get("experience_id") or item.get("round_id"))
    ][:32]
    return stable_digest(
        {
            "model": dict(profile.get("model") or {}),
            "profile_seed": profile.get("provenance", {}).get("seed") if isinstance(profile.get("provenance"), Mapping) else None,
            "history_ids": history_ids,
            "budget": dict(budget or {}),
        }
    )


__all__ = [
    "AlgorithmImprovementRequest",
    "AlgorithmValidationRecord",
    "AlgorithmValidator",
]
