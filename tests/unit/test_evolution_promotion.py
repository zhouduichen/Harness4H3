from __future__ import annotations

from harness4h3.evolution.evaluator import PromotionPolicy


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
    assert decision["accepted"] is False
    assert "model_size_improvement_below_threshold" in decision["reasons"]


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
