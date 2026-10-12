from __future__ import annotations

import json

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution import EvolutionLoop


def test_evolution_loop_runs_fake_round_and_persists_evidence(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    outcome = loop.run_round()
    assert outcome.status == "promoted"
    assert outcome.child_model_id == "M0001"
    round_dir = tmp_path / "run" / "rounds" / outcome.round_id
    assert (round_dir / "decision.json").exists()
    round_state = json.loads((round_dir / "round.json").read_text(encoding="utf-8"))
    assert round_state["status"] == "promoted"
    assert round_state["child_model_id"] == "M0001"
    records = list(loop.experience_store.read())
    assert records
    assert records[0].operator == "prune_blocks"
    assert records[0].operator_args["block_ids"]
    assert (tmp_path / "run" / "search-policy.json").exists()
    assert loop.model_store.active_id == "M0001"


def test_evolution_loop_keeps_failed_algorithm_as_experience(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")

    def provider(payload):
        return {"source": "import os\ndef propose(profile, history):\n    return {}\n", "parent_id": payload["parent_id"]}

    from harness4h3.evolution.agent import AlgorithmAgent

    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", agent=AlgorithmAgent(provider=provider))
    outcome = loop.run_round()
    assert outcome.status == "failed"
    round_state = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "round.json").read_text(encoding="utf-8"))
    assert round_state["status"] == "failed"
    assert round_state["failure_type"] == "sandbox_policy"
    record = list(loop.experience_store.read())[0]
    assert record.status == "failed"
    assert record.decision["failure_type"] == "sandbox_policy"
    assert record.decision["search_policy"]["exploit_ratio"] == 0.6


def test_evolution_loop_restarts_from_persisted_parent_and_search_policy(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    first = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    assert first.run_round().status == "promoted"

    restarted = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    outcome = restarted.run_round()
    assert outcome.round_id == "R0002"
    assert outcome.parent_model_id == "M0001"
    # The restarted loop compares the new candidate with the active measured
    # Parent, not only with the immutable initial baseline.  The structural
    # fake evaluator does not provide a further 10% size improvement, so the
    # candidate is retained as evidence without replacing M0001.
    assert outcome.status == "rejected"
    round_state = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "round.json").read_text(encoding="utf-8"))
    assert round_state["status"] == "rejected"
    assert restarted.model_store.active_id == "M0001"
    assert restarted.search_policy.rounds == 2
    assert (tmp_path / "run" / "initial-baseline.json").is_file()


def test_duplicate_algorithm_is_recorded_without_overwriting_active_parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n"

    def provider(payload):
        return {"source": source, "hypothesis": "same logic", "parent_id": payload["parent_id"]}

    from harness4h3.evolution.agent import AlgorithmAgent

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        agent=AlgorithmAgent(provider=provider),
    )
    assert loop.run_round().status == "promoted"
    duplicate = loop.run_round()
    assert duplicate.status == "failed"
    assert duplicate.decision["failure_type"] == "duplicate_algorithm"
    assert loop.model_store.active_id == "M0001"
    assert loop.search_policy.duplicate_rounds == 1


def test_algorithm_lineage_advances_after_rejected_model(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    calls = {"count": 0}

    def evaluator(child, profile):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "quality_score": 0.5,
                "baseline_quality_score": 1.0,
                "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
                "hardware_metrics": {"latency_s": 0.5, "peak_memory_gb": 0.5, "model_size_gb": 0.5},
                "valid": True,
            }
        return {
            "quality_score": 1.0,
            "baseline_quality_score": 1.0,
            "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
            "hardware_metrics": {"latency_s": 0.5, "peak_memory_gb": 0.5, "model_size_gb": 0.5},
            "valid": True,
        }

    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", evaluator=evaluator)
    assert loop.run_round().status == "rejected"
    assert loop.run_round().status == "promoted"
    assert loop.algorithm_store.get("A0002").parent_algorithm_id == "A0001"


def test_incomplete_round_reuses_persisted_compression_after_process_crash(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")

    def crash_evaluator(child, profile):
        raise KeyboardInterrupt("simulated process crash after compression")

    first = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", evaluator=crash_evaluator)
    try:
        first.run_round()
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("simulated crash should propagate")

    round_dir = tmp_path / "run" / "rounds" / "R0001"
    assert (round_dir / "compression.json").is_file()
    assert not (round_dir / "decision.json").exists()

    calls = {"count": 0}

    def recovered_evaluator(child, profile):
        calls["count"] += 1
        return {
            "quality_score": 0.99,
            "baseline_quality_score": 0.99,
            "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
            "hardware_metrics": {"latency_s": 0.5, "peak_memory_gb": 0.5, "model_size_gb": 0.5},
            "valid": True,
        }

    restarted = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", evaluator=recovered_evaluator)
    outcome = restarted.run_round()

    assert outcome.round_id == "R0001"
    assert calls["count"] == 1
    assert not (tmp_path / "run" / "rounds" / "R0002").exists()
    assert len(list(restarted.algorithm_store.read())) == 1
