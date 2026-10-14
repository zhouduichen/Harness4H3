from __future__ import annotations

import hashlib
import json

from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution.agent import build_algorithm_provider
from harness4h3.evolution.policy import SearchPolicy
from harness4h3.evolution.skill_hub import SkillHub
from harness4h3.memory.design_gene import DesignGene
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


def test_skill_hub_retrieves_design_genes_without_dropping_status(tmp_path):
    hub = SkillHub(tmp_path / "experience.jsonl", tmp_path / "genes.jsonl")
    hub.record(
        DesignGene(
            gene_id="G-R0001",
            status="rejected",
            state={"architecture": "wan"},
            bottleneck={"kind": "latency"},
            intervention={"operator": "compose"},
            controlled_conditions={"manifest": "dev"},
            benefit={"latency_s": None},
            remaining_limitation={"reason": "quality_floor_failed"},
            risks_lessons=["reduce aggressiveness after quality failure"],
            evidence={"round_id": "R0001"},
            confidence="partial",
            created_at="2026-10-14T00:00:00+00:00",
        )
    )

    matches = hub.search(architecture="wan")

    assert len(matches) == 1
    assert matches[0]["memory_kind"] == "design_gene"
    assert matches[0]["status"] == "rejected"
    assert matches[0]["training"]["design_strategy"] == "compose"


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


def test_memory_feedback_changes_deterministic_selection_logic():
    profile = {
        "block_sensitivity": [
            {"block_id": "block.0", "importance_score": 0.2},
            {"block_id": "block.1", "importance_score": 0.1},
        ]
    }
    agent = AlgorithmAgent()
    without_memory = agent.generate("A0001", parent_id="M0000", profile=profile, history=[])
    with_memory = agent.generate(
        "A0002",
        parent_id="M0000",
        profile=profile,
        history=[{"decision": {"operator_args": {"block_ids": ["block.1"]}}}],
    )
    assert "block.1" in without_memory.source
    assert "block.0" in with_memory.source
    assert without_memory.source != with_memory.source


def test_selection_history_excludes_blocks_used_inside_composite_plans():
    profile = {
        "block_sensitivity": [
            {"block_id": "block.0", "importance_score": 0.1},
            {"block_id": "block.1", "importance_score": 0.2},
            {"block_id": "block.2", "importance_score": 0.3},
        ]
    }
    candidate = AlgorithmAgent().generate(
        "A0002",
        parent_id="M0000",
        profile=profile,
        history=[
            {
                "decision": {
                    "operator_args": {
                        "steps": [
                            {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
                            {"operator": "prune_ffn_channels", "args": {"block_ids": ["block.1"]}},
                        ]
                    }
                }
            }
        ],
    )

    assert "block.0" not in candidate.source
    assert "block.1" not in candidate.source
    assert "block.2" in candidate.source


def test_memory_feedback_tightens_ffn_exploration_after_size_miss():
    agent = AlgorithmAgent(explore_ffn_retention_ratio=0.5)
    profile = {"model": {"ffn_width": 100}, "layers": [{"block_id": "block.0"}]}
    history = [{
        "evaluation": {
            "baseline_hardware_metrics": {"model_size_gb": 10.0},
            "hardware_metrics": {"model_size_gb": 9.5},
        },
        "decision": {"status": "rejected"},
    }]
    candidate = agent.generate("A0001", parent_id="M0000", profile=profile, history=history, search_choice="explore")
    assert candidate.metadata["retention_ratio"] == 0.375
    assert "int(width * 0.375000)" in candidate.source


def test_exploration_agent_generates_profile_guided_composite_plan(tmp_path):
    profile = {
        "model": {"ffn_width": 16},
        "layers": [
            {"block_id": "block.0", "latency_ms": 1.0, "estimated_flops": 10.0},
            {"block_id": "block.1", "latency_ms": 5.0, "estimated_flops": 50.0},
            {"block_id": "block.2", "latency_ms": 3.0, "estimated_flops": 30.0},
        ],
        "block_sensitivity": [
            {"block_id": "block.0", "importance_score": 0.1, "available": True},
            {"block_id": "block.1", "importance_score": 0.2, "available": True},
            {"block_id": "block.2", "importance_score": 0.3, "available": True},
        ],
        "channel_sensitivity": [
            {
                "block_id": "block.1",
                "available": True,
                "channels": [
                    {"channel": 0, "importance_score": 0.1, "available": True},
                    {"channel": 1, "importance_score": 0.9, "available": True},
                    {"channel": 2, "importance_score": 0.8, "available": True},
                ],
            }
        ],
    }

    candidate = AlgorithmAgent(explore_ffn_retention_ratio=0.5).generate(
        "A0001", parent_id="M0000", profile=profile, history=[], search_choice="explore"
    )
    result = __import__("harness4h3.sandbox", fromlist=["AlgorithmSandbox"]).AlgorithmSandbox(tmp_path).run(
        candidate.source, profile, []
    )

    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.operator == "compose"
    assert len(result.plan.args["steps"]) == 2
    assert result.plan.args["steps"][0]["operator"] == "prune_blocks"
    assert result.plan.args["steps"][1]["operator"] == "prune_ffn_channels"


def test_quality_failure_disables_aggressive_composition(tmp_path):
    profile = {
        "model": {"ffn_width": 16},
        "layers": [{"block_id": "block.0", "latency_ms": 2.0, "estimated_flops": 20.0}, {"block_id": "block.1", "latency_ms": 3.0, "estimated_flops": 30.0}],
        "block_sensitivity": [{"block_id": "block.0", "importance_score": 0.1, "available": True}, {"block_id": "block.1", "importance_score": 0.2, "available": True}],
        "channel_sensitivity": [{"block_id": "block.1", "available": True, "channels": [{"channel": 0, "importance_score": 0.1, "available": True}]}],
    }
    history = [{"decision": {"promotion": {"reasons": ["quality_floor_failed"]}}}]

    candidate = AlgorithmAgent().generate(
        "A0001", parent_id="M0000", profile=profile, history=history, search_choice="explore"
    )
    result = __import__("harness4h3.sandbox", fromlist=["AlgorithmSandbox"]).AlgorithmSandbox(tmp_path).run(
        candidate.source, profile, history
    )

    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.operator == "prune_blocks"


def test_provider_prompt_documents_composite_plan(monkeypatch):
    payloads = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({
                "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
                "hypothesis": "test",
                "parent_id": "M0000",
            })}}]}).encode()

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data.decode())['messages'][1]['content'])
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = build_algorithm_provider("vllm", "test-model", endpoint="http://127.0.0.1:8000/v1/chat/completions")
    provider({"parent_id": "M0000", "profile": {}, "history": [], "skills": []})

    assert "compose" in payloads[0]
    assert "steps" in payloads[0]
    assert "top-level keys" in payloads[0]
    assert "top-level `{'steps': [...]}` object is invalid" in payloads[0]
    assert "one flat" in payloads[0]
    assert "Never emit nested lists" in payloads[0]


def test_provider_agent_receives_compact_pareto_front():
    payloads = []

    def provider(payload):
        payloads.append(payload)
        return {
            "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
            "hypothesis": "use the measured front",
            "parent_id": "M0000",
        }

    AlgorithmAgent(provider=provider).generate(
        "A0001",
        parent_id="M0000",
        profile={},
        history=[],
        pareto_front=[
            {
                "candidate_id": "M0000",
                "quality_score": 0.9,
                "hardware_metrics": {"latency_s": 2.0, "peak_memory_gb": 4.0, "model_size_gb": 8.0, "ignored": "drop"},
                "ignored": "drop",
            }
        ],
    )
    assert payloads[0]["pareto_front"] == [
        {
            "candidate_id": "M0000",
            "quality_score": 0.9,
            "hardware_metrics": {"latency_s": 2.0, "peak_memory_gb": 4.0, "model_size_gb": 8.0},
        }
    ]


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
