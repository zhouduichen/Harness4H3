import pytest


def test_reward_normalizes_against_parent_and_requires_energy():
    from harness4h3.remote.reward import RewardWeights, compute_reward

    result = compute_reward(
        quality=0.85,
        hardware={"latency_s": 18.0, "peak_memory_gb": 36.0, "energy_j": 90.0},
        baseline={"latency_s": 30.0, "peak_memory_gb": 42.0, "energy_j": 120.0},
        weights=RewardWeights(1.0, 0.2, 0.2, 0.2),
    )
    assert result.terms == {"Q": 0.85, "L": 0.6, "M": 36.0 / 42.0, "E": 0.75}
    assert result.reward == pytest.approx(0.85 - 0.2 * 0.6 - 0.2 * (36.0 / 42.0) - 0.2 * 0.75)
    assert compute_reward(
        0.85,
        {"latency_s": 18, "peak_memory_gb": 36},
        {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120},
        RewardWeights(1, 1, 1, 1),
    ).reward is None


def test_critical_quality_failure_rejects_even_when_latency_improves():
    from harness4h3.remote.decision import AcceptanceInput, decide

    result = decide(
        AcceptanceInput(
            training_metrics={
                "optimizer_steps": 1,
                "gradient_norm": 1.0,
                "changed_trainable_tensors": 4,
                "unchanged_frozen_tensors": 531,
                "child_reloaded": True,
                "parent_sha256_before": "a" * 64,
                "parent_sha256_after": "a" * 64,
                "parent_sha256": "a" * 64,
                "child_sha256": "b" * 64,
            },
            benchmark_summary={
                "quality_score": 0.4,
                "hardware": {"latency_s": 1, "peak_memory_gb": 1, "energy_j": 1},
                "hard_gates": {"generation_valid": True, "decode_success": True, "no_critical_temporal_collapse": False},
            },
            parent_summary={"quality_score": 0.9, "hardware": {"latency_s": 10, "peak_memory_gb": 10, "energy_j": 10}},
            target={"max_quality_drop": 0.05},
            efficiency_thresholds={"latency_s": 0.15},
        )
    )
    assert result.accepted is False
    assert "quality_gate" in result.violations


def test_research_grade_acceptance_requires_all_gates_and_metrics():
    from harness4h3.remote.decision import AcceptanceInput, decide
    from harness4h3.remote.reward import RewardWeights

    result = decide(
        AcceptanceInput(
            training_metrics={
                "optimizer_steps": 1,
                "gradient_norm": 1.0,
                "changed_trainable_tensors": 1,
                "unchanged_frozen_tensors": 1,
                "child_reloaded": True,
                "parent_sha256_before": "a" * 64,
                "parent_sha256_after": "a" * 64,
                "child_sha256": "b" * 64,
            },
            benchmark_summary={
                "quality_score": 0.9,
                "hardware": {"latency_s": 9, "peak_memory_gb": 9, "energy_j": 90},
                "hard_gates": {"generation_valid": True, "decode_success": True, "no_critical_temporal_collapse": True},
            },
            parent_summary={"quality_score": 0.9, "hardware": {"latency_s": 10, "peak_memory_gb": 10, "energy_j": 100}},
            target={"max_quality_drop": 0.05},
            efficiency_thresholds={"latency_s": 0.05},
            reward_weights=RewardWeights(1, 0.2, 0.2, 0.2),
            research_grade=True,
        )
    )
    assert result.status == "accepted"
    assert result.pareto_eligible is True

