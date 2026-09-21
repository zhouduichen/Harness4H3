from __future__ import annotations

from harness4h3.campaign.gates import AcceptanceGate, MetricEvidence, pareto_dominates
from harness4h3.campaign.proposals import CandidateEnvelope


def candidate():
    return CandidateEnvelope(
        candidate_id="C0001",
        parent_candidate_id="M0000",
        generation=1,
        experiment_id="exp-C0001",
        proposal_digest="sha256:C0001",
        mutation_fields=("training.method",),
        architecture={"family": "video_latent_dit"},
        training_recipe={"method": "velocity_distill"},
        deployment_recipe={"precision": "bf16"},
        provenance={"source": "test"},
        predicted_metric_delta={"quality": 0.01},
    )


def evidence(name, value, hard=False, source="target-runtime"):
    return MetricEvidence(name, "test-v1", "fixture", value, 1.0, source, "mobile", hard)


def all_required_evidence():
    return {
        "quality": evidence("quality", 0.9, hard=True),
        "latency_s": evidence("latency_s", 1.0, hard=True),
        "peak_memory_gb": evidence("peak_memory_gb", 4.0, hard=True),
        "model_size_gb": evidence("model_size_gb", 3.0, hard=True),
    }


def test_hard_constraint_failure_beats_better_scalar_reward():
    decision = AcceptanceGate().evaluate(
        candidate=candidate(),
        evidence={
            "quality": evidence("quality", 0.99, hard=True),
            "latency": evidence("latency_s", 8.0, hard=True),
        },
        hard_constraints={"max_latency_s": 3.0},
        objectives={"quality": "maximize", "latency_s": "minimize"},
        min_rounds_met=True,
    )
    assert decision.feasible is False
    assert decision.promotable is False
    assert decision.target_satisfied is False
    assert "max_latency_s" in decision.violations


def test_promotable_candidate_does_not_imply_target_satisfied():
    decision = AcceptanceGate().evaluate(
        candidate=candidate(),
        evidence=all_required_evidence(),
        hard_constraints={},
        objectives={"quality": "maximize"},
        min_rounds_met=False,
    )
    assert decision.promotable is True
    assert decision.target_satisfied is False


def test_pareto_dominance_respects_minimize_and_maximize_directions():
    better = {"quality": evidence("quality", 0.9), "latency_s": evidence("latency_s", 1.0)}
    worse = {"quality": evidence("quality", 0.8), "latency_s": evidence("latency_s", 2.0)}
    assert pareto_dominates(better, worse, {"quality": "maximize", "latency_s": "minimize"})
    assert not pareto_dominates(worse, better, {"quality": "maximize", "latency_s": "minimize"})
