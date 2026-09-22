from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from harness4h3.campaign.gates import AcceptanceGate, MetricEvidence
from harness4h3.campaign.proposals import CandidateEnvelope
from harness4h3.student.edge import FakeTargetDeviceRunner, TargetDeviceEvaluator
from harness4h3.student.target import TargetDeviceProfile


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
    profile = TargetDeviceProfile(
        id="fake-edge-v1",
        runtime_backend="fake-runtime",
        max_latency_s=0.02,
        max_memory_gb=1.0,
        max_energy_j=2.0,
        max_thermal_c=60.0,
        max_model_size_gb=1.0,
        supported_precision=("bf16",),
        supported_quantization=("none",),
        resolution=(512, 512),
        frames=5,
        sampling_steps=1,
    )
    decision = AcceptanceGate().evaluate(
        _candidate(),
        evidence,
        hard_constraints={"max_latency_s": 0.02},
        objectives={},
        min_rounds_met=True,
        target_device_profile=profile,
    )
    assert decision.target_satisfied is True


def test_target_device_profile_failure_is_not_target_satisfied(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"student-checkpoint")
    edge = TargetDeviceEvaluator(FakeTargetDeviceRunner(), target_device_id="fake-edge-v1").evaluate(
        checkpoint, {"proposal_digest": "p1"}, tmp_path / "round"
    )
    profile = TargetDeviceProfile(
        id="fake-edge-v1",
        runtime_backend="fake-runtime",
        max_latency_s=0.02,
        max_memory_gb=1.0,
        max_energy_j=1.0,
        max_thermal_c=60.0,
        max_model_size_gb=0.1,
        supported_precision=("bf16",),
        supported_quantization=("none",),
        resolution=(512, 512),
        frames=5,
        sampling_steps=1,
    )
    decision = AcceptanceGate().evaluate(
        _candidate(),
        {item.metric_name: item.to_metric_evidence() for item in edge},
        hard_constraints={},
        objectives={},
        min_rounds_met=True,
        target_device_profile=profile,
    )
    assert decision.target_satisfied is False
    assert decision.promotable is False
    assert "target_edge_model_size" in decision.violations


def test_edge_complete_marker_cannot_bypass_missing_edge_metric(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"student-checkpoint")
    edge = TargetDeviceEvaluator(FakeTargetDeviceRunner(), target_device_id="fake-edge-v1").evaluate(
        checkpoint, {"proposal_digest": "p1"}, tmp_path / "round"
    )
    evidence = {item.metric_name: item.to_metric_evidence() for item in edge}
    evidence.pop("edge_exported")
    evidence["edge_evidence_complete"] = MetricEvidence(
        "edge_evidence_complete", "v1", "edge-complete", 1.0, True, "edge-runtime", "fake-edge-v1"
    )
    profile = TargetDeviceProfile(
        id="fake-edge-v1",
        runtime_backend="fake-runtime",
        max_latency_s=0.02,
        max_memory_gb=1.0,
        max_energy_j=2.0,
        max_thermal_c=60.0,
        max_model_size_gb=1.0,
        supported_precision=("bf16",),
        supported_quantization=("none",),
        resolution=(512, 512),
        frames=5,
        sampling_steps=1,
    )
    decision = AcceptanceGate().evaluate(
        _candidate(),
        evidence,
        hard_constraints={},
        objectives={},
        min_rounds_met=True,
        target_device_profile=profile,
    )
    assert decision.promotable is True
    assert decision.target_satisfied is False


def test_missing_target_profile_metadata_cannot_satisfy_gate(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"student-checkpoint")
    edge = TargetDeviceEvaluator(FakeTargetDeviceRunner(), target_device_id="fake-edge-v1").evaluate(
        checkpoint, {"proposal_digest": "p1"}, tmp_path / "round"
    )
    evidence = {
        item.metric_name: replace(item.to_metric_evidence(), metadata={})
        for item in edge
    }
    profile = TargetDeviceProfile(
        id="fake-edge-v1",
        runtime_backend="fake-runtime",
        max_latency_s=0.02,
        max_memory_gb=1.0,
        max_energy_j=2.0,
        max_thermal_c=60.0,
        max_model_size_gb=1.0,
        supported_precision=("bf16",),
        supported_quantization=("none",),
        resolution=(512, 512),
        frames=5,
        sampling_steps=1,
    )
    decision = AcceptanceGate().evaluate(
        _candidate(),
        evidence,
        hard_constraints={},
        objectives={},
        min_rounds_met=True,
        target_device_profile=profile,
    )
    assert decision.promotable is False
    assert decision.target_satisfied is False
