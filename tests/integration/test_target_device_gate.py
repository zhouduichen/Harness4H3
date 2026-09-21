from __future__ import annotations

from pathlib import Path

from harness4h3.campaign.gates import AcceptanceGate, MetricEvidence
from harness4h3.campaign.proposals import CandidateEnvelope
from harness4h3.student.edge import FakeTargetDeviceRunner, TargetDeviceEvaluator


def _candidate() -> CandidateEnvelope:
    return CandidateEnvelope(
        candidate_id="c1",
        parent_candidate_id="M0000",
        generation=1,
        experiment_id="e1",
        proposal_digest="p1",
        mutation_fields=("training.method",),
        architecture={"family": "video_latent_dit"},
        training_recipe={"method": "dmd2"},
        deployment_recipe={"precision": "bf16"},
        provenance={},
        predicted_metric_delta={},
    )


def test_server_evidence_is_only_promotable():
    evidence = {
        "quality": MetricEvidence("quality", "v1", "server-quality", 0.8, True, "server", "server", True),
    }
    decision = AcceptanceGate().evaluate(_candidate(), evidence, hard_constraints={}, objectives={}, min_rounds_met=True)
    assert decision.promotable is True
    assert decision.target_satisfied is False


def test_complete_target_device_evidence_can_satisfy_gate(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"student-checkpoint")
    edge = TargetDeviceEvaluator(FakeTargetDeviceRunner(), target_device_id="fake-edge-v1").evaluate(
        checkpoint, {"proposal_digest": "p1"}, tmp_path / "round"
    )
    evidence = {item.metric_name: item.to_metric_evidence() for item in edge}
    decision = AcceptanceGate().evaluate(
        _candidate(),
        evidence,
        hard_constraints={"max_latency_s": 0.02},
        objectives={},
        min_rounds_met=True,
    )
    assert decision.target_satisfied is True
