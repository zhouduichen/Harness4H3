from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from harness4h3.campaign.base import ActorIdentity
from harness4h3.campaign.capabilities import Capability, CapabilitySnapshot
from harness4h3.campaign.reviews import ReviewPipeline
from harness4h3.student.campaign import StudentCampaign
from harness4h3.student.edge import FakeTargetDeviceRunner, TargetDeviceEvaluator
from harness4h3.student.evaluator import StudentEvaluation
from harness4h3.student.proposal import StudentTarget
from harness4h3.student.target import TargetDeviceProfile
from harness4h3.student.worker import TrainingResult
from tests.unit.campaign_fixtures import make_base
from tests.unit.test_student_proposal import valid_payload


class BatchProvider:
    provider_name = "scripted-controller"
    model_name = "controller-v1"

    def __init__(self):
        self.calls = 0

    def propose_batch(self, context):
        self.calls += 1
        values = []
        for index, (hidden_size, depth) in enumerate(((2048, 24), (1920, 24), (1536, 36)), 1):
            payload = valid_payload(hidden_size=hidden_size, depth=depth)
            payload["proposal_id"] = "student_%04d" % (self.calls * 10 + index)
            values.append(payload)
        return values


class FakeManifest:
    graph_status = "compiled"
    parameter_count = 1_200_000_000
    manifest_digest = "sha256:manifest"

    def __init__(self, proposal_digest):
        self.proposal_digest = proposal_digest

    def to_dict(self):
        return {
            "graph_status": self.graph_status,
            "parameter_count": self.parameter_count,
            "manifest_digest": self.manifest_digest,
            "proposal_digest": self.proposal_digest,
        }


class FakeCompiler:
    target = StudentTarget()

    def compile(self, proposal, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return FakeManifest(proposal.digest)


class FakeWorker:
    def run(self, manifest, round_dir):
        checkpoint = Path(round_dir) / "student.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"child")
        return TrainingResult(
            status="success",
            proposal_digest=manifest.proposal_digest,
            compiler_digest=manifest.manifest_digest,
            parent_sha256="0" * 64,
            child_sha256="1" * 64,
            child_checkpoint=str(checkpoint),
            optimizer_steps=1,
            initial_loss=1.0,
            final_loss=0.5,
            gradient_norm=1.0,
            wall_time_s=0.1,
            peak_memory_gb=4.0,
            changed_parameter_count=1,
            offline_simulation=True,
        )


class RestartAwareWorker(FakeWorker):
    def __init__(self):
        self.parent_hashes = []
        self.calls = 0

    def run(
        self,
        manifest,
        round_dir,
        *,
        parent_checkpoint=None,
        parent_candidate_id=None,
        parent_checkpoint_sha256=None,
        fidelity="F1",
    ):
        del parent_candidate_id, fidelity
        self.calls += 1
        self.parent_hashes.append(parent_checkpoint_sha256)
        checkpoint = Path(round_dir) / "student.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(("child-%d" % self.calls).encode("utf-8"))
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return TrainingResult(
            status="success",
            proposal_digest=manifest.proposal_digest,
            compiler_digest=manifest.manifest_digest,
            parent_sha256=parent_checkpoint_sha256 or "0" * 64,
            child_sha256=checkpoint_sha256,
            child_checkpoint=str(checkpoint),
            full_precision_checkpoint=str(checkpoint),
            full_precision_sha256=checkpoint_sha256,
            optimizer_steps=1,
            initial_loss=1.0,
            final_loss=0.5,
            gradient_norm=1.0,
            wall_time_s=0.1,
            peak_memory_gb=4.0,
            changed_parameter_count=1,
            offline_simulation=True,
            fidelity="F1",
        )


class RecoveringWorker(FakeWorker):
    def __init__(self):
        self.calls = 0

    def run(self, manifest, round_dir):
        self.calls += 1
        if self.calls <= 3:
            return TrainingResult(
                status="failed",
                proposal_digest=manifest.proposal_digest,
                compiler_digest=manifest.manifest_digest,
                parent_sha256="0" * 64,
                child_sha256=None,
                child_checkpoint=None,
                optimizer_steps=0,
                initial_loss=None,
                final_loss=None,
                gradient_norm=None,
                wall_time_s=0.1,
                peak_memory_gb=0.0,
                changed_parameter_count=0,
                offline_simulation=True,
                failure_code="worker_oom",
                message="scripted transient OOM",
            )
        return super().run(manifest, round_dir)


class FakeEvaluator:
    def evaluate(self, checkpoint, round_dir):
        return StudentEvaluation(
            True,
            True,
            None,
            "evaluation_ok",
            str(checkpoint),
            quality_score=0.8,
            hardware={"latency_s": 1.0},
        )


class Advocate:
    identity = ActorIdentity("advocate", "reviewer", "1")

    def review(self, request: Mapping[str, Any]):
        return {
            "bottleneck": "quality-latency tradeoff",
            "changed_fields": ["architecture.hidden_size", "training.method"],
            "expected_metric_delta": {"quality": 0.01},
            "supporting_evidence_ids": list(request["evidence_ids"]),
            "falsification_experiment": "fixed verifier bank evaluation",
            "resource_assumptions": {"device": "edge"},
        }


class Critical:
    identity = ActorIdentity("critic", "model-b", "1")

    def review(self, request: Mapping[str, Any]):
        return {
            "objections": [],
            "objection_categories": [],
            "missing_evidence_ids": [],
            "proxy_gaming_risks": [],
            "target_device_risks": [],
            "required_revisions": [],
            "hard_objection": False,
        }


class Modifier:
    identity = ActorIdentity("modifier", "reviewer", "1")

    def review(self, request):
        raise AssertionError("no revision is expected")


def test_control_plane_runs_batch_review_gate_and_persists_lineage(tmp_path):
    snapshot = CapabilitySnapshot((
        Capability("distill", "training", "scripted", {}, "V2", True, ""),
        Capability("dmd2", "training", "scripted", {}, "V2", True, ""),
    ))
    base = make_base(
        target_profile={"id": "edge", "quality": {"min_quality_score": 0.7}, "objectives": {"quality": "maximize"}},
        capability_snapshot=snapshot.to_dict(),
    )
    pipeline = ReviewPipeline(Advocate(), Critical(), Modifier(), base, max_rounds=1)
    result = StudentCampaign(
        BatchProvider(),
        FakeCompiler(),
        FakeWorker(),
        FakeEvaluator(),
        output_root=tmp_path,
        campaign_base=base,
        capability_snapshot=snapshot,
        review_pipeline=pipeline,
        min_rounds_before_success=1,
    ).run(max_rounds=1)

    assert result.promotable is True
    assert result.target_satisfied is False
    assert result.status == "PROMOTABLE"
    assert len(result.candidate_decisions) == 3
    assert (tmp_path / "experience.jsonl").is_file()
    archive = [json.loads(line) for line in (tmp_path / "archive.jsonl").read_text().splitlines()]
    assert {item["archive_kind"] for item in archive} == {"pareto", "novelty"}
    events = [json.loads(line) for line in (tmp_path / "decision-trace.jsonl").read_text().splitlines()]
    assert events[0]["event_type"] == "campaign.created"
    assert any(item["event_type"] == "parent.selected" for item in events)
    assert all(item["base_digest"] == base.digest for item in events)


def test_control_plane_recovery_keeps_parent_until_verified_child(tmp_path):
    snapshot = CapabilitySnapshot((
        Capability("distill", "training", "scripted", {}, "V2", True, ""),
        Capability("dmd2", "training", "scripted", {}, "V2", True, ""),
    ))
    base = make_base(
        target_profile={"id": "edge", "quality": {"min_quality_score": 0.7}},
        capability_snapshot=snapshot.to_dict(),
    )
    result = StudentCampaign(
        BatchProvider(),
        FakeCompiler(),
        RecoveringWorker(),
        FakeEvaluator(),
        output_root=tmp_path,
        campaign_base=base,
        capability_snapshot=snapshot,
        review_pipeline=ReviewPipeline(Advocate(), Critical(), Modifier(), base, max_rounds=1),
        max_failures=2,
    ).run(max_rounds=2)

    assert result.status == "PROMOTABLE"
    assert result.target_satisfied is False
    assert any(item.get("failure_code") == "worker_oom" for item in result.candidate_decisions)
    events = [json.loads(line) for line in (tmp_path / "decision-trace.jsonl").read_text().splitlines()]
    selected = [item for item in events if item["event_type"] == "parent.selected"]
    assert selected[-1]["payload"]["generation"] == 1
    assert selected[-1]["parent_candidate_id"] == "M0000"
    assert any(item["event_type"] == "campaign.replanned" and item["payload"].get("failure", {}).get("failure_code") == "worker_oom" for item in events)
    archive = [json.loads(line) for line in (tmp_path / "archive.jsonl").read_text().splitlines()]
    assert any(item["archive_kind"] == "failure" for item in archive)
    assert any(item["archive_kind"] == "pareto" for item in archive)


def test_campaign_restart_restores_and_verifies_selected_parent_checkpoint(tmp_path):
    snapshot = CapabilitySnapshot((
        Capability("distill", "training", "scripted", {}, "V2", True, ""),
        Capability("dmd2", "training", "scripted", {}, "V2", True, ""),
    ))
    base = make_base(
        target_profile={"id": "edge", "quality": {"min_quality_score": 0.7}},
        capability_snapshot=snapshot.to_dict(),
    )
    worker = RestartAwareWorker()

    first = StudentCampaign(
        BatchProvider(), FakeCompiler(), worker, FakeEvaluator(), output_root=tmp_path,
        campaign_base=base, capability_snapshot=snapshot,
        review_pipeline=ReviewPipeline(Advocate(), Critical(), Modifier(), base, max_rounds=1),
        fidelity_schedule=("F1",),
    ).run(max_rounds=1)
    assert first.status == "PROMOTABLE"
    selected = [
        json.loads(line)
        for line in (tmp_path / "decision-trace.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "parent.selected"
    ][-1]
    expected_parent_sha256 = selected["payload"]["inheritance_checkpoint_sha256"]
    assert expected_parent_sha256

    second = StudentCampaign(
        BatchProvider(), FakeCompiler(), worker, FakeEvaluator(), output_root=tmp_path,
        campaign_base=base, capability_snapshot=snapshot,
        review_pipeline=ReviewPipeline(Advocate(), Critical(), Modifier(), base, max_rounds=1),
        fidelity_schedule=("F1",),
    ).run(max_rounds=2)
    assert second.status == "PROMOTABLE"
    assert expected_parent_sha256 in worker.parent_hashes


def _edge_profile() -> TargetDeviceProfile:
    return TargetDeviceProfile(
        id="edge",
        runtime_backend="fake-runtime",
        max_latency_s=0.02,
        max_memory_gb=1.0,
        max_energy_j=2.0,
        max_thermal_c=60.0,
        max_model_size_gb=1.0,
        supported_precision=("bf16",),
        supported_quantization=("none", "int8"),
        resolution=(512, 512),
        frames=5,
        sampling_steps=1,
    )


def test_campaign_adapter_campaign_gate_merges_complete_edge_evidence(tmp_path):
    snapshot = CapabilitySnapshot((
        Capability("distill", "training", "scripted", {}, "V2", True, ""),
        Capability("dmd2", "training", "scripted", {}, "V2", True, ""),
    ))
    base = make_base(
        target_profile={
            "id": "edge",
            "quality": {"min_quality_score": 0.7},
            "objectives": {"quality": "maximize", "latency_s": "minimize"},
        },
        capability_snapshot=snapshot.to_dict(),
    )
    target = _edge_profile()
    result = StudentCampaign(
        BatchProvider(),
        FakeCompiler(),
        FakeWorker(),
        FakeEvaluator(),
        output_root=tmp_path,
        campaign_base=base,
        capability_snapshot=snapshot,
        review_pipeline=ReviewPipeline(Advocate(), Critical(), Modifier(), base, max_rounds=1),
        target_device_evaluator=TargetDeviceEvaluator(FakeTargetDeviceRunner(), target_device_id="edge", profile=target),
        target_device_profile=target,
        fidelity_schedule=("F1",),
        min_rounds_before_success=1,
    ).run(max_rounds=1)

    assert result.status == "TARGET_SATISFIED"
    assert result.target_satisfied is True
    assert all(item["target_satisfied"] is True for item in result.candidate_decisions if item.get("candidate_id"))
    events = [json.loads(line) for line in (tmp_path / "decision-trace.jsonl").read_text().splitlines()]
    assert any(item["event_type"] == "edge.completed" for item in events)
    gate_events = [item for item in events if item["event_type"] == "gate.decided"]
    assert gate_events and all(item["payload"]["decision"]["target_satisfied"] for item in gate_events)
    assert gate_events[0]["payload"]["decision"]["objective_values"]["latency"] == 0.0125


class EdgeSelectionWorker(FakeWorker):
    def run(self, manifest, round_dir, **kwargs):
        checkpoint = Path(round_dir) / "student.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        marker = b"edge-a" if manifest.proposal["architecture"]["hidden_size"] == 2048 else b"edge-b"
        checkpoint.write_bytes(marker)
        return TrainingResult(
            status="success",
            proposal_digest=manifest.proposal_digest,
            compiler_digest=manifest.manifest_digest,
            parent_sha256="0" * 64,
            child_sha256="1" * 64,
            child_checkpoint=str(checkpoint),
            optimizer_steps=1,
            initial_loss=1.0,
            final_loss=0.5,
            gradient_norm=1.0,
            wall_time_s=0.1,
            peak_memory_gb=4.0,
            changed_parameter_count=1,
            offline_simulation=True,
        )


class EdgeSelectionCompiler(FakeCompiler):
    def compile(self, proposal, output_dir):
        manifest = super().compile(proposal, output_dir)
        manifest.proposal = proposal.to_dict()
        return manifest


class EdgeSelectionEvaluator(FakeEvaluator):
    def evaluate(self, checkpoint, round_dir):
        server_latency = 0.1 if Path(checkpoint).read_bytes() == b"edge-a" else 0.2
        return StudentEvaluation(
            True,
            True,
            None,
            "evaluation_ok",
            str(checkpoint),
            quality_score=0.8,
            hardware={"latency_s": server_latency},
        )


class EdgeSelectionRunner(FakeTargetDeviceRunner):
    def benchmark(self, deployment_id, compiled, output_dir, proposal):
        result = dict(super().benchmark(deployment_id, compiled, output_dir, proposal))
        if compiled.read_bytes() == b"edge-a":
            # Candidate A is faster on the server but slower on the target;
            # its lower edge memory keeps both candidates Pareto-feasible.
            result.update({"latency_s": 0.018, "memory_gb": 0.4})
        else:
            result.update({"latency_s": 0.010, "memory_gb": 0.9})
        return result


def test_parent_selection_uses_edge_latency_after_edge_evidence(tmp_path):
    snapshot = CapabilitySnapshot((
        Capability("distill", "training", "scripted", {}, "V2", True, ""),
        Capability("dmd2", "training", "scripted", {}, "V2", True, ""),
    ))
    base = make_base(
        target_profile={
            "id": "edge",
            "quality": {"min_quality_score": 0.7},
            "objectives": {
                "quality": "maximize", "latency_s": "minimize", "peak_memory_gb": "minimize",
                "energy_j": "minimize", "model_size_gb": "minimize",
            },
        },
        capability_snapshot=snapshot.to_dict(),
    )
    target = _edge_profile()
    result = StudentCampaign(
        BatchProvider(),
        EdgeSelectionCompiler(),
        EdgeSelectionWorker(),
        EdgeSelectionEvaluator(),
        output_root=tmp_path,
        campaign_base=base,
        capability_snapshot=snapshot,
        review_pipeline=ReviewPipeline(Advocate(), Critical(), Modifier(), base, max_rounds=1),
        target_device_evaluator=TargetDeviceEvaluator(EdgeSelectionRunner(), target_device_id="edge", profile=target),
        target_device_profile=target,
        fidelity_schedule=("F1",),
        min_rounds_before_success=1,
    ).run(max_rounds=1)

    assert result.status == "TARGET_SATISFIED"
    events = [json.loads(line) for line in (tmp_path / "decision-trace.jsonl").read_text().splitlines()]
    selected = [item for item in events if item["event_type"] == "parent.selected"][-1]
    assert selected["candidate_id"] in {"student_0012", "student_0013"}
    assert selected["payload"]["candidate_id"] in {"student_0012", "student_0013"}
