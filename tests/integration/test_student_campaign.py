from __future__ import annotations

import json
from pathlib import Path

import torch

from harness4h3.student.campaign import StudentCampaign
from harness4h3.student.compiler import StudentCompiler
from harness4h3.student.evaluator import StudentEvaluation
from harness4h3.student.proposal import StudentProposal
from harness4h3.student.worker import TrainingResult
from tests.unit.test_student_proposal import valid_payload


class SequenceStudentProvider:
    provider_name = "test"
    model_name = "sequence"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if len(self.contexts) == 2:
            assert any(item["failure_code"] == "video_decode_failed" for item in context["failures"])
        return self.payloads.pop(0)


class ScriptedWorker:
    def __init__(self):
        self.calls = 0

    def run(self, manifest, round_dir):
        self.calls += 1
        child = Path(round_dir) / "student.safetensors"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_bytes(b"changed-child-%d" % self.calls)
        return TrainingResult(
            status="success",
            proposal_digest=manifest.proposal_digest,
            compiler_digest=manifest.manifest_digest,
            parent_sha256="0" * 64,
            child_sha256="1" * 64,
            child_checkpoint=str(child),
            optimizer_steps=1,
            initial_loss=1.0,
            final_loss=0.5,
            gradient_norm=1.0,
            wall_time_s=0.1,
            peak_memory_gb=1.0,
            changed_parameter_count=1,
            offline_simulation=True,
        )


class ScriptedEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, checkpoint, round_dir):
        self.calls += 1
        if self.calls == 1:
            return StudentEvaluation(False, False, "video_decode_failed", "bad video", str(checkpoint))
        return StudentEvaluation(True, True, None, "evaluation_ok", str(checkpoint), quality_score=0.8)


def test_two_round_campaign_passes_failure_to_revised_proposal(tmp_path):
    provider = SequenceStudentProvider([valid_payload(depth=24), valid_payload(depth=32)])
    worker = ScriptedWorker()
    evaluator = ScriptedEvaluator()
    result = StudentCampaign(
        provider,
        StudentCompiler(),
        worker,
        evaluator,
        output_root=tmp_path,
        max_failures=3,
    ).run(max_rounds=2)
    assert result.rounds_completed == 2
    assert result.status == "PROMOTABLE"
    assert worker.calls == 2

    events = [json.loads(line) for line in (tmp_path / "campaign-events.jsonl").read_text().splitlines()]
    assert events[0]["failure_code"] == "video_decode_failed"
    assert events[1]["status"] == "success"


def test_duplicate_proposal_is_rejected_without_worker_launch(tmp_path):
    provider = SequenceStudentProvider([valid_payload(depth=24), valid_payload(depth=24)])
    worker = ScriptedWorker()
    evaluator = ScriptedEvaluator()
    result = StudentCampaign(
        provider,
        StudentCompiler(),
        worker,
        evaluator,
        output_root=tmp_path,
        max_failures=3,
    ).run(max_rounds=2)
    assert worker.calls == 1
    assert result.failure_code == "duplicate_proposal"
