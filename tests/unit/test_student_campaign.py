from __future__ import annotations

import json

from harness4h3.student.campaign import OpenAICompatibleStudentProposalProvider, StudentCampaign, student_proposal_json_schema
from harness4h3.student.compiler import StudentCompiler
from harness4h3.student.proposal import StudentTarget
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
