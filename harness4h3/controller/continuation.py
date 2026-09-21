from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping, Optional

from ..controller.schemas import EvaluationRecord
from ..target.profile import TargetProfile


class ContinuationStatus(str, Enum):
    REJECT = "reject"
    EXPLORATORY_KEEP = "exploratory_keep"
    PARETO_KEEP = "pareto_keep"
    FINAL_ACCEPT = "final_accept"


@dataclass(frozen=True)
class ContinuationDecision:
    status: ContinuationStatus
    advance: bool
    reasons: List[str]

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "status": self.status.value,
            "decision_status": self.status.value,
            "advance": self.advance,
            "keep": self.advance,
            "reasons": list(self.reasons),
        }


class ContinuationPolicy:
    """Fixed post-evaluation policy; Controller acceptance is only audit input."""

    def __init__(self, exploration_enabled: bool = False):
        self.exploration_enabled = bool(exploration_enabled)

    @staticmethod
    def _quality_floor(
        target: TargetProfile,
        baseline_quality: Optional[float],
        plan_acceptance: Optional[Mapping[str, Any]],
    ) -> Optional[float]:
        floors = []
        if target.min_quality_score is not None:
            floors.append(float(target.min_quality_score))
        acceptance = plan_acceptance or {}
        if acceptance.get("min_quality_score") is not None:
            floors.append(float(acceptance["min_quality_score"]))
        drop = acceptance.get("max_quality_drop")
        if drop is not None and baseline_quality is not None:
            floors.append(float(baseline_quality) - float(drop))
        if target.max_quality_drop is not None and baseline_quality is not None:
            floors.append(float(baseline_quality) - float(target.max_quality_drop))
        return max(floors) if floors else None

    @staticmethod
    def _metrics(evaluation: EvaluationRecord) -> Mapping[str, Any]:
        return {
            "quality_score": evaluation.quality_score,
            "latency_s": evaluation.hardware.latency_s,
            "peak_memory_gb": evaluation.hardware.peak_memory_gb,
            "model_size_gb": evaluation.hardware.model_size_gb,
            "energy_j": evaluation.hardware.energy_j,
        }

    def decide(
        self,
        evaluation: EvaluationRecord,
        on_front: bool,
        target: TargetProfile,
        plan_acceptance: Optional[Mapping[str, Any]] = None,
        baseline_quality: Optional[float] = None,
        parent_metrics: Optional[Mapping[str, Any]] = None,
    ) -> ContinuationDecision:
        reasons: List[str] = []
        if not math.isfinite(float(evaluation.quality_score)):
            return ContinuationDecision(ContinuationStatus.REJECT, False, ["quality_score_not_finite"])
        if evaluation.failure_type:
            return ContinuationDecision(ContinuationStatus.REJECT, False, ["evaluation_failure:%s" % evaluation.failure_type])
        if any(value is False for value in evaluation.validity.values()):
            return ContinuationDecision(ContinuationStatus.REJECT, False, ["invalid_evidence"])
        if evaluation.critical_regression:
            return ContinuationDecision(ContinuationStatus.REJECT, False, ["critical_quality_regression"])

        floor = self._quality_floor(target, baseline_quality, plan_acceptance)
        if floor is not None and evaluation.quality_score < floor:
            return ContinuationDecision(ContinuationStatus.REJECT, False, ["quality_floor"])
        if not evaluation.feasible:
            # The legacy offline protocol may deliberately walk through an
            # infeasible fake point to test search mechanics.  This escape
            # hatch is provenance-bound: real worker/benchmark evidence never
            # receives it, so a real hard-constraint violation is always REJECT.
            if evaluation.provenance.get("offline_simulation") is True and on_front:
                return ContinuationDecision(ContinuationStatus.PARETO_KEEP, True, ["offline_search_point"])
            return ContinuationDecision(
                ContinuationStatus.REJECT,
                False,
                ["hard_constraint:%s" % violation for violation in evaluation.violations]
                or ["hard_constraints_not_satisfied"],
            )
        if evaluation.feasible:
            return ContinuationDecision(ContinuationStatus.FINAL_ACCEPT, True, ["hard_constraints_satisfied"])
        if on_front:
            return ContinuationDecision(ContinuationStatus.PARETO_KEEP, True, ["non_dominated_search_point"])

        if self.exploration_enabled and parent_metrics is not None:
            current_score = (
                evaluation.search_score
                if evaluation.search_score is not None
                else target.objective_score(self._metrics(evaluation))
            )
            parent_score = target.objective_score(parent_metrics)
            if current_score > parent_score:
                return ContinuationDecision(
                    ContinuationStatus.EXPLORATORY_KEEP,
                    True,
                    ["objective_score_improved", "explicit_exploration_enabled"],
                )
            reasons.append("objective_score_not_improved")
        else:
            reasons.append("not_on_pareto_front")
        return ContinuationDecision(ContinuationStatus.REJECT, False, reasons)


__all__ = ["ContinuationDecision", "ContinuationPolicy", "ContinuationStatus"]
