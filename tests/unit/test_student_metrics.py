from __future__ import annotations

import math

from harness4h3.student.metrics import MetricVerifierBank


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
