from __future__ import annotations

from pathlib import Path

import pytest

from harness4h3.student.edge import EDGE_METRICS, EdgeEvidence, FakeTargetDeviceRunner, TargetDeviceEvaluator, validate_edge_evidence


class _IncompleteTargetRunner(FakeTargetDeviceRunner):
    def benchmark(self, deployment_id, compiled, output_dir, proposal):
        measurements = dict(super().benchmark(deployment_id, compiled, output_dir, proposal))
        measurements.pop("runtime_backend")
        return measurements


def test_target_device_evaluator_runs_full_artifact_path(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"student-checkpoint")
    evidence = TargetDeviceEvaluator(
        FakeTargetDeviceRunner(),
        target_device_id="fake-edge-v1",
    ).evaluate(checkpoint, {"proposal_digest": "p"}, tmp_path / "round")
    assert tuple(item.metric_name for item in evidence) == EDGE_METRICS
    assert all(item.target_device_id == "fake-edge-v1" for item in evidence)
    assert len({item.artifact_sha256 for item in evidence}) == 1
    assert (tmp_path / "round" / "target-device" / "benchmark" / "edge-evidence.json").is_file()


def test_edge_evidence_rejects_server_or_partial_measurements(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"student-checkpoint")
    evidence = TargetDeviceEvaluator(
        FakeTargetDeviceRunner(), target_device_id="fake-edge-v1"
    ).evaluate(checkpoint, {"proposal_digest": "p"}, tmp_path / "round")
    with pytest.raises(ValueError, match="device identity"):
        validate_edge_evidence(
            tuple(
                EdgeEvidence(
                    item.metric_name,
                    item.value,
                    item.artifact_sha256,
                    "server",
                    item.measurement_reference,
                )
                for item in evidence
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        validate_edge_evidence(evidence[:-1])


def test_target_device_evaluator_rejects_missing_profile_measurements(tmp_path: Path):
    checkpoint = tmp_path / "student.safetensors"
    checkpoint.write_bytes(b"student-checkpoint")
    with pytest.raises(ValueError, match="runtime_backend"):
        TargetDeviceEvaluator(
            _IncompleteTargetRunner(), target_device_id="fake-edge-v1"
        ).evaluate(checkpoint, {"proposal_digest": "p"}, tmp_path / "round")
