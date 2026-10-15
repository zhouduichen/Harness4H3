from __future__ import annotations

import pytest
from dataclasses import replace

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution.algorithm_evaluation import AlgorithmModelEvaluator
from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.evolution.loop import EvolutionLoop
from harness4h3.evolution.improvement import AlgorithmValidator
from harness4h3.evolution.store import AlgorithmStore
from harness4h3.evolution.taxonomy import classify_algorithm_change
from harness4h3.operators.wan import build_wan_compression_registry
from harness4h3.sandbox import AlgorithmSandbox
from harness4h3.sandbox.runner import CompressionPlan, SandboxResult


def fake_profile():
    return {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 16},
        "layers": [
            {"block_id": "block.0", "estimated_flops": 10.0, "latency_ms": 1.0},
            {"block_id": "block.1", "estimated_flops": 40.0, "latency_ms": 4.0},
            {"block_id": "block.2", "estimated_flops": 20.0, "latency_ms": 2.0},
            {"block_id": "block.3", "estimated_flops": 30.0, "latency_ms": 3.0},
        ],
        "block_sensitivity": [
            {"block_id": "block.%d" % index, "importance_score": 0.1 + index * 0.1, "available": True}
            for index in range(4)
        ],
        "channel_sensitivity": [
            {
                "block_id": "block.0",
                "width": 16,
                "available": True,
                "channels": [{"channel": index, "importance_score": float(index + 1), "available": True} for index in range(16)],
            }
        ],
    }


def parent_source():
    return "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n"


def quality_failure():
    return {"experience_id": "R0001", "status": "rejected", "decision": {"promotion": {"reasons": ["quality_floor_failed"]}}}


def test_deterministic_improvement_changes_parent_source_and_records_reason(tmp_path):
    parent = AlgorithmCandidate.from_source("A0001", parent_source(), hypothesis="baseline")
    child = AlgorithmAgent().improve(
        "A0002",
        parent=parent,
        parent_id="M0000",
        profile=fake_profile(),
        history=[quality_failure()],
        failures=["quality_floor_failed"],
        search_space={"version_id": "S0001"},
    )
    assert child.parent_algorithm_id == parent.algorithm_id
    assert child.parent_source_sha256 == parent.source_sha256
    assert child.metadata["parent_source_sha256"] == parent.source_sha256
    assert child.source != parent.source
    assert child.change_rationale
    result = AlgorithmSandbox(tmp_path / "sandbox").run(child.source, fake_profile(), [])
    assert result.status == "success"


def test_algorithm_store_rejects_wrong_parent_hash(tmp_path):
    store = AlgorithmStore(tmp_path)
    parent = store.create(AlgorithmCandidate.from_source("A0001", parent_source()))
    child = AlgorithmAgent().improve(
        "A0002", parent=parent, parent_id="M0000", profile=fake_profile(), history=[], failures=["efficiency_plateau"]
    )
    with pytest.raises(ValueError, match="parent_source_hash_mismatch"):
        store.create_child(child, "0" * 64)


def test_algorithm_store_persists_auditable_source_diff(tmp_path):
    store = AlgorithmStore(tmp_path)
    parent = store.create(AlgorithmCandidate.from_source("A0001", parent_source()))
    child = AlgorithmAgent().improve(
        "A0002", parent=parent, parent_id="M0000", profile=fake_profile(), history=[], failures=["efficiency_plateau"]
    )
    stored = store.create_child(child, parent.source_sha256)
    assert stored.source_diff
    assert "--- A0001.py" in stored.source_diff
    assert "+++ A0002.py" in stored.source_diff
    assert (tmp_path / "candidates" / "A0002.diff").read_text() == stored.source_diff
    assert store.get("A0002").source_diff == stored.source_diff


def test_algorithm_validator_compares_parent_and_child_under_same_conditions(tmp_path):
    parent = AlgorithmCandidate.from_source("A0001", parent_source())
    child = AlgorithmAgent().improve(
        "A0002", parent=parent, parent_id="M0000", profile=fake_profile(), history=[], failures=["efficiency_plateau"]
    )
    sandbox = AlgorithmSandbox(tmp_path / "sandbox")

    def evaluate(result):
        return {
            "valid": True,
            "offline_simulation": True,
            "search_score": 0.8 if result.plan.operator == "prune_blocks" else 0.9,
            "model_size_gb": 1.0 if result.plan.operator == "prune_blocks" else 0.7,
        }

    record = AlgorithmValidator().validate(parent, child, sandbox, fake_profile(), [], evaluate, budget={"steps": 1})
    assert record.valid is True
    assert record.evidence_level == "cpu_fake"
    assert record.parent_source_sha256 == parent.source_sha256
    assert record.candidate_plan["operator"] == "prune_ffn_channels"


def test_algorithm_validator_compares_multiple_fixed_profiles(tmp_path):
    parent = AlgorithmCandidate.from_source("A0001", parent_source())
    child = AlgorithmAgent().improve(
        "A0002", parent=parent, parent_id="M0000", profile=fake_profile(), history=[], failures=["efficiency_plateau"]
    )
    alternate = dict(fake_profile())
    alternate["provenance"] = {"seed": 8, "validation_scenario": "alternate_seed"}
    sandbox = AlgorithmSandbox(tmp_path / "sandbox")

    def evaluate(result):
        return {
            "valid": True,
            "offline_simulation": True,
            "search_score": 0.8 if result.plan.operator == "prune_blocks" else 0.9,
            "model_size_gb": 1.0 if result.plan.operator == "prune_blocks" else 0.7,
        }

    record = AlgorithmValidator().validate(
        parent,
        child,
        sandbox,
        fake_profile(),
        [],
        evaluate,
        budget={"steps": 1},
        profiles=(fake_profile(), alternate),
    )
    assert record.valid is True
    assert len(record.scenario_results) == 2
    assert record.parent_metrics["scenario_count"] == 2
    assert record.candidate_metrics["search_score"] == 0.9


def test_provider_improvement_receives_complete_parent_source():
    captured = {}
    parent_text = "def propose(profile, history):\n" + "    # line %04d\\n" * 700
    parent_text = parent_text % tuple(range(700))
    parent = AlgorithmCandidate.from_source("A0001", parent_text)

    def provider(payload):
        captured.update(payload)
        return {
            "source": parent.source + "    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
            "parent_id": payload["parent_id"],
            "hypothesis": "complete source was revised",
        }

    child = AlgorithmAgent(provider=provider).improve(
        "A0002", parent=parent, parent_id="M0000", profile=fake_profile(), history=[], failures=["efficiency_plateau"]
    )
    assert captured["parent_source"] == parent.source
    assert captured["parent_source_line_count"] == len(parent.source.splitlines())
    assert child.source_diff


def test_agent_can_propose_untrusted_search_space_extension():
    proposal = AlgorithmAgent().propose_search_space_change(
        search_space={"version_id": "S0001", "operators": [{"name": "prune_blocks"}]},
        history=[{"round_id": "R0001", "status": "rejected"}],
        failures=["channel_importance_plateau"],
    )
    assert proposal["spec"]["name"] == "prune_ffn_by_weight_threshold"
    assert "not_real_wan_validated" in proposal["spec"]["limitations"]


def test_provider_can_propose_a_generated_search_space_rule():
    source = "def propose(profile, history):\n    return {'operator': 'prune_ffn_channels', 'args': {'block_ids': [], 'retained_indices': [0]}}\n"

    def provider(payload):
        assert payload["task"] == "propose_search_space_extension"
        return {
            "spec": {
                "name": "agent_generated_rule",
                "version": "1",
                "argument_schema": {"block_ids": "list[str]"},
                "granularity": "channel",
                "trusted_factory_key": "generated_channel_rule",
                "implementation_source": source,
            },
            "rationale": "fixed operators cannot express this ranking rule",
            "evidence_ids": ["R0001"],
        }

    result = AlgorithmAgent(provider=provider).propose_search_space_change(
        search_space={"version_id": "S0001", "operators": [{"name": "prune_blocks"}]},
        history=[],
        failures=["channel_importance_plateau"],
    )
    assert result["spec"]["trusted_factory_key"] == "generated_channel_rule"
    assert result["spec"]["implementation_source"] == source


def test_model_level_algorithm_measurement_depends_on_structure_not_operator_label(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16, seed=7)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    profile = fake_profile()
    evaluator = AlgorithmModelEvaluator(adapter, build_wan_compression_registry(), parent, profile, tmp_path / "measure")
    block_plan = CompressionPlan("prune_blocks", {"block_ids": ["block.0"]})
    channel_plan = CompressionPlan("prune_ffn_channels", {"block_ids": ["block.0", "block.1", "block.2", "block.3"], "retained_indices": list(range(8))})
    block_result = evaluator.evaluate(SandboxResult("success", "a" * 64, block_plan))
    same_result = evaluator.evaluate(SandboxResult("success", "b" * 64, block_plan))
    channel_result = evaluator.evaluate(SandboxResult("success", "c" * 64, channel_plan))

    assert block_result["valid"] is True
    assert block_result["evidence_level"] == "cpu_fake_quality_proxy"
    assert block_result["parameter_reduction_ratio"] > 0.0
    assert block_result["compute_metrics"]["available"] is True
    assert block_result["output_metrics"]["cosine_similarity"] <= 1.0
    assert block_result["hardware_metrics"]["available"] is False
    assert block_result["evaluation_cost"]["input_seed"] == 7
    assert block_result["output_metrics"]["mse"] == same_result["output_metrics"]["mse"]
    assert block_result["parameter_count"] == same_result["parameter_count"]
    assert channel_result["parameter_count"] != block_result["parameter_count"]
    assert channel_result["plan_digest"] != block_result["plan_digest"]


def test_change_taxonomy_does_not_call_a_new_rule_a_new_structural_space():
    active = {"operators": [{"name": "prune_ffn_channels"}, {"name": "compose"}]}
    parent_plan = {"operator": "prune_ffn_channels", "args": {"block_ids": ["block.0"], "retained_indices": [0, 1]}}
    child_plan = {"operator": "prune_attention_heads", "args": {"heads": [0]}}
    result = classify_algorithm_change("def propose(): pass", "def propose(): return 1", parent_plan, child_plan, active)
    assert result["change_class"] == "compression_space_extension"
    assert result["reachable_by_existing_closure"] is False
    rule_result = classify_algorithm_change("def propose(): pass", "def propose(): return 1", parent_plan, {"operator": "prune_ffn_channels", "args": {"block_ids": ["block.0"], "retained_indices": [0]}}, active)
    assert rule_result["change_class"] == "algorithm_rule_evolution"
    assert rule_result["reachable_by_existing_closure"] is True


def test_algorithm_parent_selection_falls_back_from_latest_rejected_branch(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    first = loop.algorithm_store.create(AlgorithmCandidate.from_source("A0001", parent_source(), parent_id="M0000"))
    second = AlgorithmCandidate.from_source(
        "A0002",
        "def propose(profile, history):\n    return {'operator': 'prune_ffn_channels', 'args': {'block_ids': ['block.0'], 'retained_indices': [0]}}\n",
        parent_id="M0000",
        parent_algorithm_id=first.algorithm_id,
        parent_source_sha256=first.source_sha256,
        validation_status="validated",
    )
    loop.algorithm_store.create_child(second, first.source_sha256)
    loop.algorithm_store.update(replace(second, status="rejected"))
    third = AlgorithmCandidate.from_source(
        "A0003",
        "def propose(profile, history):\n    return {'operator': 'compose', 'args': {'steps': []}}\n",
        parent_id="M0000",
        parent_algorithm_id=second.algorithm_id,
        parent_source_sha256=second.source_sha256,
        validation_status="rejected",
    )
    loop.algorithm_store.create_child(third, second.source_sha256)
    loop.algorithm_store.update(replace(third, status="rejected"))

    selection = loop._algorithm_parent_selection()
    assert selection["latest"].algorithm_id == "A0003"
    assert selection["effective"].algorithm_id == "A0002"
    assert selection["historical_best"].algorithm_id == "A0002"
    assert any(item.algorithm_id == "A0003" for item in selection["rejected"])
