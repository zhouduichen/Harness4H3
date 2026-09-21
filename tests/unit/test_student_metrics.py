from __future__ import annotations

import math

from harness4h3.student.metrics import (
    MetricVerifierBank,
    normalize_reward,
    pareto_decision,
    teacher_relative_metrics,
)


def test_metric_bank_extracts_continuous_quality_latency_memory_and_size():
    result = MetricVerifierBank().evaluate(
        {"score": 0.814},
        {
            "latency_s": 0.0378,
            "peak_memory_gb": 6.2,
            "checkpoint_bytes": int(4.7 * 1024**3),
        },
    )

    assert result.reward is not None
    assert result.missing == ("energy",)
    assert result.invalid == ()
    assert math.isclose(result.metrics["latency"].value or 0, 37.8, rel_tol=1e-6)
    assert math.isclose(result.metrics["memory"].normalized or 0, 6.2 / 8.0, rel_tol=1e-6)
    assert math.isclose(result.metrics["size"].value or 0, 4.7, rel_tol=1e-6)
    assert set(result.reward_terms) == {"quality", "latency", "memory", "size"}


def test_metric_bank_fails_reward_closed_when_required_metric_is_missing():
    result = MetricVerifierBank().evaluate({"score": 0.8}, {"latency_s": 0.04, "peak_memory_gb": 6.0})

    assert result.reward is None
    assert "size" in result.missing


def test_metric_bank_rejects_invalid_quality():
    result = MetricVerifierBank().evaluate({"score": 1.2}, {"latency_s": 0.04, "peak_memory_gb": 6.0, "model_size_gb": 4.0})

    assert result.reward is None
    assert result.invalid == ("quality",)


def test_teacher_relative_metrics_and_bounded_reward():
    teacher = {"quality": 0.80, "latency": 10000.0, "memory": 12.0, "size": 2.0}
    student = {"quality": 0.72, "latency": 5000.0, "memory": 10.0, "size": 2.0}

    relative = teacher_relative_metrics(student, teacher)

    assert math.isclose(relative.quality_ratio, 0.90, rel_tol=1e-6)
    reward = normalize_reward(relative, incumbent=teacher)
    assert reward > 0.0
    assert -1.0 <= relative.efficiency_deltas["latency"] <= 1.0


def test_pareto_gate_accepts_material_efficiency_gain_at_quality_floor():
    decision = pareto_decision(
        candidate={"quality_ratio": 0.90, "latency": 5000.0, "memory": 10.0, "size": 2.0},
        incumbent={"quality_ratio": 0.95, "latency": 10000.0, "memory": 12.0, "size": 2.0},
        quality_floor_ratio=0.90,
        min_reward_delta=0.02,
        material_efficiency_gain=0.05,
        max_regression=0.02,
    )

    assert decision.promotable is True
    assert decision.reason == "pareto_improvement"
    assert "latency" in decision.improved_metrics


def test_pareto_gate_rejects_quality_floor_and_no_improvement():
    below_floor = pareto_decision(
        candidate={"quality_ratio": 0.899, "latency": 5000.0, "memory": 10.0, "size": 2.0},
        incumbent={"quality_ratio": 0.95, "latency": 10000.0, "memory": 12.0, "size": 2.0},
        quality_floor_ratio=0.90,
    )
    unchanged = pareto_decision(
        candidate={"quality_ratio": 0.90, "latency": 10000.0, "memory": 12.0, "size": 2.0},
        incumbent={"quality_ratio": 0.90, "latency": 10000.0, "memory": 12.0, "size": 2.0},
        quality_floor_ratio=0.90,
    )

    assert below_floor.promotable is False
    assert below_floor.reason == "quality_floor_failed"
    assert unchanged.promotable is False
    assert unchanged.reason == "pareto_rejected"
