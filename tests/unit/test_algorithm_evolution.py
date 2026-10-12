from __future__ import annotations

import hashlib
import json

from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution.agent import build_algorithm_provider
from harness4h3.evolution.policy import SearchPolicy
from harness4h3.evolution.skill_hub import SkillHub
from harness4h3.memory.experience import ExperienceRecord


def _experience(tmp_id: str, architecture: str, operator: str, reward: float) -> ExperienceRecord:
    source = "memory://%s" % tmp_id
    return ExperienceRecord(
        experience_id=tmp_id,
        source_uri=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_kind="compression_round",
        experiment_id=tmp_id,
        parent_model_id="M0000",
        child_model_id="M0001",
        operator=operator,
        operator_args={},
        training={"architecture": architecture, "operator": operator},
        evaluation={"quality_score": 0.8},
        decision={"status": "accepted"},
        reward=reward,
        status="accepted",
        provenance={"architecture": architecture},
        created_at=tmp_id,
    )


def test_skill_hub_filters_by_architecture_and_operator_family(tmp_path):
    hub = SkillHub(tmp_path / "experience.jsonl", tmp_path / "genes.jsonl")
    hub.record(_experience("wan-1", "wan", "prune_blocks", reward=0.4))
    hub.record(_experience("h3-1", "h3", "prune_blocks", reward=0.9))
    matches = hub.search(architecture="wan", operator="prune_blocks", limit=5)
    assert len(matches) == 1
    assert matches[0]["architecture"] == "wan"


def test_search_policy_uses_seeded_explore_ratio():
    policy = SearchPolicy(exploit_ratio=0.7, seed=3)
    choices = [policy.choose(has_best=True) for _ in range(10)]
    assert all(item in {"exploit", "explore"} for item in choices)
    assert policy.to_dict()["exploit_ratio"] == 0.7


def test_search_policy_resume_preserves_next_choice():
    running = SearchPolicy(exploit_ratio=0.7, seed=11)
    running.choose(has_best=True)
    running.choose(has_best=True)
    restored = SearchPolicy.from_dict(running.to_dict())
    assert running.choose(has_best=True) == restored.choose(has_best=True)


def test_deterministic_agent_generates_source_from_low_sensitivity_block():
    candidate = AlgorithmAgent().generate(
        "A0001",
        parent_id="M0000",
        profile={"block_sensitivity": [{"block_id": "block.0", "importance_score": 0.5}, {"block_id": "block.1", "importance_score": 0.1}]},
        history=[],
    )
    assert candidate.parent_id == "M0000"
    assert "block.1" in candidate.source


def test_provider_agent_rejects_parent_override():
    agent = AlgorithmAgent(provider=lambda payload: {"source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n", "parent_id": "M9999"})
    try:
        agent.generate("A0001", parent_id="M0000", profile={}, history=[])
    except ValueError as exc:
        assert "cannot change parent_id" in str(exc)
    else:
        raise AssertionError("provider parent override was accepted")


def test_vllm_algorithm_provider_parses_structured_json_without_network(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"source": "def propose(profile, history):\\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\\n", "hypothesis": "test", "parent_id": "M0000"})}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    provider = build_algorithm_provider("vllm", "test-model", endpoint="http://127.0.0.1:8000/v1/chat/completions")
    result = provider({"parent_id": "M0000", "profile": {}, "history": [], "skills": []})
    assert result["parent_id"] == "M0000"
    assert "def propose" in result["source"]


def test_vllm_provider_binds_missing_parent_to_request_context(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n", "hypothesis": "test", "parent_id": None})}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    provider = build_algorithm_provider("vllm", "test-model", endpoint="http://127.0.0.1:8000/v1/chat/completions")
    result = provider({"parent_id": "M0007", "profile": {}, "history": [], "skills": []})
    assert result["parent_id"] == "M0007"
