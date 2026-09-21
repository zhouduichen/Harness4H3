from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from harness4h3.campaign.base import ActorIdentity
from harness4h3.campaign.capabilities import Capability, CapabilitySnapshot
from harness4h3.campaign.reviews import ReviewPipeline
from harness4h3.student.campaign import StudentCampaign
from harness4h3.student.evaluator import StudentEvaluation
from harness4h3.student.proposal import StudentTarget
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
    assert result.target_satisfied is True
    assert result.status == "target_satisfied"
    assert len(result.candidate_decisions) == 3
    assert (tmp_path / "experience.jsonl").is_file()
    archive = [json.loads(line) for line in (tmp_path / "archive.jsonl").read_text().splitlines()]
    assert {item["archive_kind"] for item in archive} == {"pareto"}
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

    assert result.status == "target_satisfied"
    assert result.target_satisfied is True
    assert any(item.get("failure_code") == "worker_oom" for item in result.candidate_decisions)
    events = [json.loads(line) for line in (tmp_path / "decision-trace.jsonl").read_text().splitlines()]
    selected = [item for item in events if item["event_type"] == "parent.selected"]
    assert selected[-1]["payload"]["generation"] == 1
    assert selected[-1]["parent_candidate_id"] == "M0000"
    assert any(item["event_type"] == "campaign.replanned" and item["payload"].get("failure", {}).get("failure_code") == "worker_oom" for item in events)
    archive = [json.loads(line) for line in (tmp_path / "archive.jsonl").read_text().splitlines()]
    assert any(item["archive_kind"] == "failure" for item in archive)
    assert any(item["archive_kind"] == "pareto" for item in archive)
