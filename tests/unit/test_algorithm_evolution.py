from __future__ import annotations

import hashlib
import json

from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution.agent import build_algorithm_provider
from harness4h3.evolution.agent import _compact_context_items
from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.evolution.loop import EvolutionLoop
from harness4h3.evolution.policy import SearchPolicy
from harness4h3.evolution.skill_hub import SkillHub
from harness4h3.adapters.wan import FakeWanDiTAdapter
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


def test_measured_block_feedback_survives_compaction_and_selects_feasible_algorithm_parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", algorithm_improvement=True)

    for index, (quality, feasible) in enumerate(((0.974, False), (0.976, False), (0.981, True)), start=1):
        source = (
            "def propose(profile, history):\n"
            "    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.%d']}}\n" % index
        )
        loop.algorithm_store.create(AlgorithmCandidate.from_source("A%04d" % index, source))
        round_dir = loop.rounds_root / ("R%04d" % index)
        round_dir.mkdir()
        trace = {
            "algorithm_id": "A%04d" % index,
            "measured_search": {
                "parent_checkpoint_sha256": parent.sha256,
                "quality_floor_ratio": 0.98,
                "candidate_results": [{
                    "plan": {"operator": "prune_blocks", "args": {"block_ids": ["block.%d" % index]}},
                    "valid": True,
                    "quality_score": quality,
                    "quality_floor_ratio": 0.98,
                    "quality_feasible": feasible,
                    "search_score": quality + 0.003,
                    "efficiency_reduction_ratio": 0.03,
                }],
            },
        }
        (round_dir / "algorithm-program-trace.json").write_text(json.dumps(trace))

    history = loop._algorithm_search_feedback_history(limit=2)
    compact = _compact_context_items(history)
    assert len(compact) == 2
    assert compact[0]["memory_kind"] == "algorithm_search_feedback"
    assert compact[0]["parent_checkpoint_sha256"] == parent.sha256
    assert compact[0]["measured_candidates"][0]["plan"]["args"]["block_ids"] == ["block.3"]
    assert compact[0]["measured_candidates"][0]["quality_score"] == 0.981

    selection = loop._algorithm_parent_selection()
    assert selection["effective"].algorithm_id == "A0003"
    assert selection["selection_reason"] == "validated_then_measured_block_search_evidence_then_latest_fallback"


def test_duplicate_algorithm_source_gets_one_feedback_driven_retry(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent_artifact = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent_artifact, output_root=tmp_path / "run", algorithm_improvement=True)
    source_a1 = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n"
    source_a2 = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}\n"
    source_a3 = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}}\n"
    parent = AlgorithmCandidate.from_source("A0001", source_a1)
    duplicate = AlgorithmCandidate.from_source(
        "A0002",
        source_a2,
        parent_id="M0000",
        parent_algorithm_id="A0001",
        parent_source_sha256=parent.source_sha256,
    )
    loop.algorithm_store.create(parent)
    loop.algorithm_store.create_child(duplicate, parent.source_sha256)

    class RetryAgent:
        def __init__(self):
            self.calls = []

        def improve(self, algorithm_id, **kwargs):
            self.calls.append(kwargs)
            source = source_a2 if len(self.calls) == 1 else source_a3
            selected_parent = kwargs["parent"]
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=kwargs["parent_id"],
                parent_algorithm_id=selected_parent.algorithm_id,
                parent_source_sha256=selected_parent.source_sha256,
            )

    retry_agent = RetryAgent()
    loop.agent = retry_agent
    candidate, selected_parent = loop._improve_unique_algorithm(
        algorithm_id="A0003",
        parent=parent,
        parent_id="M0000",
        profile={},
        history=[{"memory_kind": "algorithm_search_feedback", "experience_id": "R0001"}],
        failures=("algorithm_search_failed",),
        skills=[],
        search_space={},
        protocol_mode="standard",
        round_id="R0002",
    )

    assert candidate.source == source_a3
    assert candidate.parent_algorithm_id == "A0002"
    assert selected_parent.algorithm_id == "A0002"
    assert retry_agent.calls[1]["failures"] == ("algorithm_search_failed", "duplicate_algorithm_source")
    assert retry_agent.calls[1]["history"][0]["memory_kind"] == "algorithm_duplicate_feedback"
    duplicate_attempt = candidate.metadata["duplicate_source_attempts"][0]
    assert duplicate_attempt["candidate_source_sha256"] == AlgorithmCandidate.from_source("A0002", source_a2).source_sha256
    compact_retry = _compact_context_items([retry_agent.calls[1]["history"][0]])[0]
    assert compact_retry["duplicate_algorithm_id"] == "A0002"
    assert compact_retry["duplicate_algorithm_source_sha256"] == AlgorithmCandidate.from_source("A0002", source_a2).source_sha256


def test_duplicate_algorithm_source_reuses_selected_parent_after_retry(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent_artifact = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent_artifact, output_root=tmp_path / "run", algorithm_improvement=True)
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n"
    selected_parent = AlgorithmCandidate.from_source("A0001", source)
    loop.algorithm_store.create(selected_parent)

    class DuplicateAgent:
        def __init__(self):
            self.calls = []

        def improve(self, algorithm_id, **kwargs):
            self.calls.append(kwargs)
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=kwargs["parent_id"],
                parent_algorithm_id=kwargs["parent"].algorithm_id,
                parent_source_sha256=kwargs["parent"].source_sha256,
            )

    agent = DuplicateAgent()
    loop.agent = agent
    algorithm, algorithm_parent = loop._improve_unique_algorithm(
        algorithm_id="A0002",
        parent=selected_parent,
        parent_id="M0000",
        profile={},
        history=[{"memory_kind": "algorithm_search_feedback", "round_id": "R0001"}],
        failures=("algorithm_search_failed",),
        skills=[],
        search_space={},
        protocol_mode="standard",
        round_id="R0002",
    )

    assert len(agent.calls) == 2
    assert algorithm.algorithm_id == "A0001"
    assert algorithm_parent.algorithm_id == "A0001"
    assert algorithm.metadata["reused_existing_algorithm_after_duplicate"] is True
    duplicate_attempts = algorithm.metadata["duplicate_source_attempts"]
    assert len(duplicate_attempts) == 2
    assert all(item["duplicate_algorithm_id"] == "A0001" for item in duplicate_attempts)
    assert all(item["candidate_source_sha256"] == selected_parent.source_sha256 for item in duplicate_attempts)
    assert [item.algorithm_id for item in loop.algorithm_store.read()] == ["A0001"]


def test_memory_feedback_keeps_size_miss_exploration_block_only():
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
    assert candidate.metadata["operator"] == "prune_blocks"
    assert "prune_ffn_channels" not in candidate.source


def test_exploration_agent_generates_profile_guided_block_plan(tmp_path):
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
    assert result.plan.operator == "prune_blocks"
    assert result.plan.args["block_ids"] == ["block.1"]


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


def test_provider_prompt_documents_multigranular_search(monkeypatch):
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

    assert "per-Block/per-timestep and per-Head sensitivity" in payloads[0]
    assert "`search_space['operators']` is the source" in payloads[0]
    assert "Avoid Wan structural Head Pruning" in payloads[0]
    assert "search(profile, history, state, search_space)" in payloads[0]


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


def test_provider_agent_receives_all_active_search_space_schemas():
    payloads = []

    def provider(payload):
        payloads.append(payload)
        return {
            "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
            "hypothesis": "remove a profile-ranked Block",
            "parent_id": "M0000",
        }

    AlgorithmAgent(provider=provider).generate(
        "A0001",
        parent_id="M0000",
        profile={},
        history=[],
        search_space={
            "version_id": "S0001",
            "operators": [
                {"name": "prune_blocks", "argument_schema": {"block_ids": "list[str]"}, "execution_domain": "structural"},
                {"name": "kv_token_select", "argument_schema": {"keep_ratio": "float", "score": "str"}, "execution_domain": "runtime"},
            ],
        },
    )
    assert payloads[0]["search_space"]["version_id"] == "S0001"
    assert [item["name"] for item in payloads[0]["search_space"]["operators"]] == ["prune_blocks", "kv_token_select"]
    assert "implementation_source" not in json.dumps(payloads[0]["search_space"])


def test_provider_prompt_exposes_active_multigranular_search_space(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({
                "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
                "hypothesis": "remove a profile-ranked Block",
                "parent_id": "M0000",
            })}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: (requests.append(json.loads(request.data.decode())) or Response()))
    provider = build_algorithm_provider("vllm", "test-model", endpoint="http://127.0.0.1:8000/v1/chat/completions")
    provider({
        "parent_id": "M0000",
        "profile": {},
        "history": [],
        "skills": [],
        "search_space": {"operators": [
            {"name": "prune_blocks", "argument_schema": {"block_ids": "list[str]"}},
        ]},
    })
    prompt = requests[0]["messages"][1]["content"]
    assert "search_space['operators']" in prompt
    assert "active operator registry" in prompt
    assert "`search_space['operators']` is the source" in prompt
    assert "registered argument schema" in prompt


def test_provider_agent_compacts_full_memory_artifacts():
    payloads = []

    def provider(payload):
        payloads.append(payload)
        return {
            "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
            "hypothesis": "use compact evidence",
            "parent_id": "M0000",
        }

    large_memory_item = {
        "experience_id": "R0001",
        "memory_kind": "experience",
        "architecture": "wan",
        "operator": "prune_ffn_channels",
        "status": "rejected",
        "training": {"hypothesis": "x" * 1000, "design": {"plan": {"steps": ["x"]}}},
        "evaluation": {
            "quality_score": 0.6,
            "quality_metrics": {"semantic": 0.5, "temporal": 0.9, "motion": 0.3},
            "hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 2.0, "model_size_gb": 3.0},
            "cases": [{"video_path": "/tmp/video.mp4", "quality": {"semantic": 0.5}}] * 16,
        },
        "decision": {"promotion": {"accepted": False, "reasons": ["quality_floor_failed"]}},
    }
    AlgorithmAgent(provider=provider).generate(
        "A0002",
        parent_id="M0000",
        profile={},
        history=[large_memory_item],
        skills=[large_memory_item],
    )

    payload = payloads[0]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "cases" not in serialized
    assert "video_path" not in serialized
    assert len(serialized) < 10000
    assert payload["history"][0]["evaluation"]["hardware_metrics"]["model_size_gb"] == 3.0


def test_provider_agent_deduplicates_history_and_skills():
    payloads = []

    def provider(payload):
        payloads.append(payload)
        return {
            "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
            "hypothesis": "deduplicate context",
            "parent_id": "M0000",
        }

    item = {"experience_id": "R0001", "memory_kind": "experience", "status": "rejected"}
    AlgorithmAgent(provider=provider).generate(
        "A0002", parent_id="M0000", profile={}, history=[item], skills=[item]
    )

    assert payloads[0]["history"]
    assert payloads[0]["skills"] == []


def test_provider_agent_rejects_parent_override():
    agent = AlgorithmAgent(provider=lambda payload: {"source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n", "parent_id": "M9999"})
    try:
        agent.generate("A0001", parent_id="M0000", profile={}, history=[])
    except ValueError as exc:
        assert "cannot change parent_id" in str(exc)
    else:
        raise AssertionError("provider parent override was accepted")


def test_vllm_algorithm_provider_parses_structured_json_without_network(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({"source": "def propose(profile, history):\\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\\n", "hypothesis": "test", "parent_id": "M0000"})}}]}).encode()

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = build_algorithm_provider("vllm", "test-model", endpoint="http://127.0.0.1:8000/v1/chat/completions")
    result = provider({"parent_id": "M0000", "profile": {}, "history": [], "skills": []})
    assert result["parent_id"] == "M0000"
    assert "def propose" in result["source"]
    assert requests[0]["max_tokens"] == 1536


def test_vllm_algorithm_improvement_reserves_room_for_complete_source(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            content = {
                "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
                "hypothesis": "retain deterministic Block-only search",
                "parent_id": "M0000",
                "change_summary": "remove unsupported import",
                "change_rationale": "use the sandbox allowlist",
            }
            return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = build_algorithm_provider("vllm", "test-model", endpoint="http://127.0.0.1:8000/v1/chat/completions")
    result = provider({
        "task": "improve_existing_algorithm_source",
        "protocol_mode": "standard",
        "parent_id": "M0000",
        "parent_source": "def propose(profile, history):\n    return {}\n",
        "profile": {},
        "history": [],
        "skills": [],
        "search_space": {"operators": [{"name": "prune_blocks"}]},
        "failure_reasons": ["forbidden_import: random"],
    })

    assert result["parent_id"] == "M0000"
    assert requests[0]["max_tokens"] == 3072
    prompt = requests[0]["messages"][1]["content"]
    assert "never import or use random" in prompt
    assert "one dictionary, never a list" in prompt
    assert "under 7000 characters" in prompt
    assert "candidate-generation" in prompt


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
