from __future__ import annotations

from harness4h3.controller.continuation import ContinuationPolicy, ContinuationStatus
from harness4h3.controller.schemas import EvaluationRecord, HardwareMetrics
from harness4h3.target.profile import TargetProfile


def target():
    return TargetProfile("gpu", "gpu", "L40", max_latency_s=30, max_quality_drop=0.05)


def result(quality=0.90, latency=60, feasible=False, critical=False, validity=None):
    return EvaluationRecord(
        quality_score=quality,
        quality_metrics={"quality_score": quality},
        hardware=HardwareMetrics(latency_s=latency, peak_memory_gb=8),
        feasible=feasible,
        critical_regression=critical,
        validity=validity or {},
    )


def test_policy_rejects_critical_regression_even_when_front_member():
    decision = ContinuationPolicy().decide(result(quality=0.5, critical=True), True, target(), baseline_quality=0.9)
    assert decision.status is ContinuationStatus.REJECT
    assert decision.advance is False


def test_policy_final_accept_is_evaluator_authority():
    decision = ContinuationPolicy().decide(result(latency=20, feasible=True), False, target(), plan_acceptance={"max_quality_drop": 1.0})
    assert decision.status is ContinuationStatus.FINAL_ACCEPT
    assert decision.advance is True


def test_policy_rejects_non_feasible_pareto_search_point():
    decision = ContinuationPolicy().decide(
        result(latency=60, feasible=False), True, target(), plan_acceptance={"max_quality_drop": 1.0}, baseline_quality=0.9
    )
    assert decision.status is ContinuationStatus.REJECT
    assert decision.advance is False
    assert decision.reasons == ["hard_constraints_not_satisfied"]


def test_policy_rejects_invalid_evidence():
    decision = ContinuationPolicy().decide(result(validity={"generation_valid": False}), True, target())
    assert decision.status is ContinuationStatus.REJECT
    assert decision.reasons == ["invalid_evidence"]
