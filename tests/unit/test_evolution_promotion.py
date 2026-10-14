from __future__ import annotations

from harness4h3.evolution.evaluator import PromotionPolicy
from harness4h3.evolution.policy import SearchPolicy


def _evidence(*, quality=0.99, latency=90.0, memory=9.0, size=8.0, valid=True, baseline_quality=1.0):
    return {
        "quality_score": quality,
        "baseline_quality_score": baseline_quality,
        "valid": valid,
        # The evaluator's self-reported feasible flag must not be trusted.
        "feasible": True,
        "hardware_metrics": {"latency_s": latency, "peak_memory_gb": memory, "model_size_gb": size},
        "baseline_hardware_metrics": {"latency_s": 100.0, "peak_memory_gb": 10.0, "model_size_gb": 10.0},
    }


def test_promotion_requires_real_efficiency_gain_not_feasible_flag():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(latency=100.0, memory=10.0, size=10.0),
        child_integrity=True,
    )
    assert decision["accepted"] is False
    assert "no_substantial_efficiency_improvement" in decision["reasons"]


def test_missing_evidence_never_promotes():
    decision = PromotionPolicy().decide(parent={}, evaluation={"feasible": True}, child_integrity=True)
    assert decision["accepted"] is False
    assert "quality_data_missing" in decision["reasons"]
    assert "hardware_data_missing" in decision["reasons"]


def test_quality_regression_never_promotes_even_with_hardware_gain():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(quality=0.97),
        child_integrity=True,
    )
    assert decision["accepted"] is False
    assert "quality_floor_failed" in decision["reasons"]


def test_pareto_dominated_candidate_is_rejected():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(quality=0.99, latency=90.0, memory=9.0, size=9.0),
        child_integrity=True,
        pareto_front=(
            {
                "quality_score": 0.99,
                "hardware_metrics": {"latency_s": 80.0, "peak_memory_gb": 8.0, "model_size_gb": 8.0},
            },
        ),
    )
    assert decision["accepted"] is False
    assert "pareto_dominated" in decision["reasons"]


def test_real_improvement_can_promote():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(),
        child_integrity=True,
    )
    assert decision["accepted"] is True
    assert decision["improvements"]["latency_s"] == 0.1


def test_efficiency_gain_without_required_model_size_gain_does_not_promote():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(latency=90.0, memory=7.0, size=9.5),
        child_integrity=True,
    )
    assert decision["accepted"] is True
    assert decision["objective_passes"]["latency_s"] is True
    assert decision["objective_passes"]["model_size_gb"] is False


def test_partial_hardware_evidence_does_not_promote():
    evidence = _evidence()
    del evidence["hardware_metrics"]["peak_memory_gb"]
    decision = PromotionPolicy().decide(parent={}, evaluation=evidence, child_integrity=True)
    assert decision["accepted"] is True
    assert "peak_memory_gb" in decision["missing_metrics"]


def test_latency_only_gain_can_promote():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(latency=94.0, memory=10.0, size=9.5),
        child_integrity=True,
    )
    assert decision["accepted"] is True
    assert decision["objective_passes"]["latency_s"] is True


def test_memory_only_gain_can_promote():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(latency=100.0, memory=9.0, size=9.5),
        child_integrity=True,
    )
    assert decision["accepted"] is True
    assert decision["objective_passes"]["peak_memory_gb"] is True


def test_model_size_only_gain_can_promote():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(latency=100.0, memory=10.0, size=8.9),
        child_integrity=True,
    )
    assert decision["accepted"] is True
    assert decision["objective_passes"]["model_size_gb"] is True


def test_zero_peak_memory_is_missing_not_zero_usage():
    decision = PromotionPolicy().decide(
        parent={},
        evaluation=_evidence(latency=100.0, memory=0.0, size=9.5),
        child_integrity=True,
    )
    assert decision["accepted"] is False
    assert "peak_memory_gb" in decision["missing_metrics"]


def test_quality_submetric_floor_is_checked_when_parent_data_exists():
    evidence = _evidence()
    evidence["quality_metrics"] = {"semantic": 0.90, "temporal": 0.99, "motion": 0.99}
    evidence["baseline_quality_metrics"] = {"semantic": 1.0, "temporal": 1.0, "motion": 1.0}
    decision = PromotionPolicy().decide(parent={}, evaluation=evidence, child_integrity=True)
    assert decision["accepted"] is False
    assert "quality_metric_floor_failed:semantic" in decision["reasons"]


def test_non_target_hardware_regression_is_a_hard_failure():
    evidence = _evidence()
    evidence["hardware_metrics"]["energy_j"] = 11.0
    evidence["baseline_hardware_metrics"]["energy_j"] = 10.0
    decision = PromotionPolicy().decide(parent={}, evaluation=evidence, child_integrity=True)
    assert decision["accepted"] is False
    assert "hardware_regression:energy_j" in decision["reasons"]


def test_separate_latency_and_memory_thresholds_are_configurable():
    policy = PromotionPolicy.from_mapping(
        {"min_latency_improvement": 0.20, "min_memory_improvement": 0.05, "min_model_size_improvement": 0.10}
    )
    decision = policy.decide(parent={}, evaluation=_evidence(latency=80.0, memory=9.0, size=9.5), child_integrity=True)
    assert decision["accepted"] is True
    assert decision["min_latency_improvement"] == 0.20
    assert decision["min_memory_improvement"] == 0.05


def test_promotion_uses_active_parent_before_initial_baseline():
    decision = PromotionPolicy().decide(
        parent={
            "quality_score": 0.90,
            "hardware_metrics": {"latency_s": 80.0, "peak_memory_gb": 8.0, "model_size_gb": 8.0},
        },
        evaluation={
            # This is the immutable initial baseline and is intentionally
            # better than the active Parent after an earlier promotion.
            "baseline_quality_score": 1.0,
            "baseline_hardware_metrics": {"latency_s": 100.0, "peak_memory_gb": 10.0, "model_size_gb": 10.0},
            "quality_score": 0.87,
            "valid": True,
            "hardware_metrics": {"latency_s": 70.0, "peak_memory_gb": 7.0, "model_size_gb": 7.0},
        },
        child_integrity=True,
    )
    assert decision["accepted"] is False
    assert "quality_floor_failed" in decision["reasons"]


def test_search_policy_records_high_cost_low_gain_feedback():
    policy = SearchPolicy()
    policy.update(feasible=False, improved=False, failure_type="model_size_improvement_below_threshold")
    assert policy.costly_failure_rounds == 1
    assert policy.exploit_ratio == 0.6


def test_search_policy_records_metric_specific_hardware_regression():
    policy = SearchPolicy()
    policy.update(feasible=False, improved=False, failure_type="hardware_regression:latency_s")
    assert policy.costly_failure_rounds == 1
    assert policy.exploit_ratio == 0.6
