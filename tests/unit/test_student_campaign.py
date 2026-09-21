from __future__ import annotations

import json
from pathlib import Path
import urllib.error

from harness4h3.student.campaign import OpenAICompatibleStudentProposalProvider, StudentCampaign, student_proposal_json_schema
from harness4h3.student.compiler import StudentCompiler
from harness4h3.student.evaluator import StudentEvaluation
from harness4h3.student.proposal import StudentTarget
from harness4h3.student.worker import TrainingResult
from tests.unit.test_student_proposal import valid_payload


class SequenceProvider:
    provider_name = "test"
    model_name = "sequence"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return self.payloads.pop(0)


class StrictWorker:
    def run(self, manifest, round_dir):
        checkpoint = Path(round_dir) / "student.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"changed")
        return TrainingResult(
            status="success",
            proposal_digest=manifest.proposal_digest,
            compiler_digest=manifest.manifest_digest,
            parent_sha256="a" * 64,
            child_sha256="b" * 64,
            child_checkpoint=str(checkpoint),
            optimizer_steps=256,
            initial_loss=1.0,
            final_loss=0.5,
            gradient_norm=1.0,
            wall_time_s=1.0,
            peak_memory_gb=10.0,
            changed_parameter_count=1,
            offline_simulation=False,
        )


class StrictEvaluator:
    def __init__(self, metrics):
        self.metrics = list(metrics)

    def evaluate(self, checkpoint, round_dir):
        return StudentEvaluation(
            valid=True,
            promotable=True,
            failure_code=None,
            message="evaluation_ok",
            video_path=str(Path(round_dir) / "student.mp4"),
            quality_score=self.metrics[0]["quality"],
            metric_evidence={"optimization_metrics": self.metrics.pop(0)},
        )


def test_student_schema_is_strict_and_contains_target_limits():
    schema = student_proposal_json_schema(StudentTarget())
    assert schema["additionalProperties"] is False
    assert schema["properties"]["architecture"]["properties"]["latent_channels"]["const"] == 24
    assert schema["properties"]["architecture"]["properties"]["hidden_size"]["minimum"] == 1024


def test_campaign_records_invalid_proposal_for_next_call(tmp_path):
    invalid = valid_payload()
    invalid["architecture"]["hidden_size"] = 1000
    valid = valid_payload(depth=32)
    provider = SequenceProvider([invalid, valid])

    class NoWorker:
        def run(self, manifest, round_dir):
            raise AssertionError("invalid proposal must not launch worker")

    result = StudentCampaign(
        provider,
        StudentCompiler(),
        NoWorker(),
        object(),
        target=StudentTarget(),
        output_root=tmp_path,
        max_failures=2,
    ).run(max_rounds=1)
    assert result.status == "failed"
    event = json.loads((tmp_path / "campaign-events.jsonl").read_text().splitlines()[0])
    assert event["failure_code"] == "proposal_invalid"


def test_openai_compatible_provider_parses_vllm_choices(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(valid_payload())}}]}).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    provider = OpenAICompatibleStudentProposalProvider(
        "qwen3.5-controller", StudentTarget(), base_url="http://127.0.0.1:8000/v1"
    )
    result = provider.propose({"round": 1})
    assert result["proposal_id"] == "student_0001"
    assert provider.endpoint.endswith("/v1/chat/completions")


def test_openai_compatible_provider_waits_through_server_restart(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(valid_payload(depth=36))}}]}).encode("utf-8")

    calls = {"count": 0}

    def urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.URLError("connection refused")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("harness4h3.student.campaign.time.sleep", lambda _: None)
    provider = OpenAICompatibleStudentProposalProvider(
        "qwen3.5-controller", StudentTarget(), base_url="http://127.0.0.1:8000/v1", timeout_s=5
    )
    result = provider.propose({"round": 2})
    assert result["proposal_id"] == "student_0001"
    assert calls["count"] == 2


def _strict_payload(index: int, *, hidden_size: int = 2048, depth: int = 24):
    payload = valid_payload(hidden_size=hidden_size, depth=depth)
    payload["proposal_id"] = "student_%04d" % index
    return payload


def test_strict_campaign_rejects_valid_but_dominated_candidates(tmp_path):
    provider = SequenceProvider([_strict_payload(1), _strict_payload(2, hidden_size=1920), _strict_payload(3, hidden_size=2048)])
    evaluator = StrictEvaluator(
        [
            {"quality": 0.90, "latency": 10000.0, "memory": 12.0, "size": 2.0},
            {"quality": 0.90, "latency": 10000.0, "memory": 12.0, "size": 2.0},
            {"quality": 0.90, "latency": 10000.0, "memory": 12.0, "size": 2.0},
        ]
    )
    result = StudentCampaign(
        provider,
        StudentCompiler(),
        StrictWorker(),
        evaluator,
        target=StudentTarget(),
        output_root=tmp_path,
        retention_handler=lambda *args: None,
        teacher_baseline={"optimization_metrics": {"quality": 1.0}},
        quality_policy={"no_improvement_patience": 1},
    ).run(max_rounds=3)

    assert result.status == "no_pareto_improvement"
    assert any(item.failure_code == "pareto_rejected" for item in result.rounds[1:])


def test_strict_campaign_accepts_efficiency_frontier_and_persists_context(tmp_path):
    provider = SequenceProvider([_strict_payload(1), _strict_payload(2, hidden_size=1920), _strict_payload(3, hidden_size=2048), _strict_payload(4, hidden_size=1920)])
    evaluator = StrictEvaluator(
        [
            {"quality": 0.90, "latency": 10000.0, "memory": 12.0, "size": 2.0},
            {"quality": 0.90, "latency": 9000.0, "memory": 12.0, "size": 2.0},
            {"quality": 0.90, "latency": 9000.0, "memory": 12.0, "size": 2.0},
            {"quality": 0.90, "latency": 9000.0, "memory": 12.0, "size": 2.0},
        ]
    )
    result = StudentCampaign(
        provider,
        StudentCompiler(),
        StrictWorker(),
        evaluator,
        target=StudentTarget(),
        output_root=tmp_path,
        retention_handler=lambda *args: None,
        teacher_baseline={"optimization_metrics": {"quality": 1.0}},
        quality_policy={"no_improvement_patience": 2},
    ).run(max_rounds=4)

    assert result.status == "success"
    resume = json.loads((tmp_path / "resume.json").read_text())
    assert resume["improvement_count"] == 1
    assert resume["incumbent"]["latency"] == 9000.0
    assert provider.contexts[1]["optimization"]["incumbent"]["latency"] == 10000.0
    assert provider.contexts[3]["failures"][-1]["type"] == "optimization_rejected"
