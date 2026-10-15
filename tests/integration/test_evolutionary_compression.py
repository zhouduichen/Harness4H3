from __future__ import annotations

import json

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution import EvolutionLoop
from harness4h3.evolution.policy import SearchPolicy


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
    # Parent.  The new protocol permits promotion when any one of the
    # independent efficiency objectives clears its configured threshold.
    assert outcome.status == "promoted"
    round_state = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "round.json").read_text(encoding="utf-8"))
    assert round_state["status"] == "promoted"
    assert restarted.model_store.active_id == "M0002"
    assert restarted.search_policy.rounds == 2
    assert (tmp_path / "run" / "initial-baseline.json").is_file()


def test_deterministic_baseline_avoids_duplicate_selection_across_three_rounds(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=8, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        memory_enabled=False,
        adapt_search_policy=False,
        search_policy=SearchPolicy(exploit_ratio=1.0, seed=0),
    )

    outcomes = [loop.run_round() for _ in range(3)]

    assert all(outcome.status in {"promoted", "rejected"} for outcome in outcomes)
    assert loop.search_policy.duplicate_rounds == 0
    selected = []
    for outcome in outcomes:
        decision = json.loads(
            (tmp_path / "run" / "rounds" / outcome.round_id / "decision.json").read_text()
        )
        selected.append(decision["operator_args"]["block_ids"][0])
    assert len(set(selected)) == 3


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


def test_design_feedback_changes_next_algorithm_and_persists_gene(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    calls = {"count": 0}

    def evaluator(child, profile):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "quality_score": 0.50,
                "baseline_quality_score": 1.0,
                "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
                "hardware_metrics": {"latency_s": 0.40, "peak_memory_gb": 0.40, "model_size_gb": 0.40},
                "valid": True,
            }
        return {
            "quality_score": 1.0,
            "baseline_quality_score": 1.0,
            "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
            "hardware_metrics": {"latency_s": 0.90, "peak_memory_gb": 0.90, "model_size_gb": 0.90},
            "valid": True,
        }

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        evaluator=evaluator,
        search_policy=SearchPolicy(exploit_ratio=0.0, seed=0),
    )

    first = loop.run_round()
    second = loop.run_round()

    assert first.status == "rejected"
    assert second.status == "promoted"
    assert loop.model_store.active_id == second.child_model_id
    first_design = json.loads((tmp_path / "run" / "rounds" / first.round_id / "design.json").read_text())
    second_design = json.loads((tmp_path / "run" / "rounds" / second.round_id / "design.json").read_text())
    assert first_design["fingerprint"] != second_design["fingerprint"]
    assert first_design["strategy"] == "composed"
    assert second_design["strategy"] == "prune_blocks"
    genes = (tmp_path / "run" / "design-genes.jsonl").read_text().splitlines()
    assert len(genes) == 2
    assert "quality_floor_failed" in genes[0]


def test_failed_composite_design_is_retained_as_negative_gene(tmp_path):
    from harness4h3.evolution.agent import AlgorithmAgent

    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    source = """def propose(profile, history):
    return {'operator': 'compose', 'args': {'steps': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
    ]}}
"""

    def provider(payload):
        return {"source": source, "hypothesis": "repeat a block to test failure memory", "parent_id": payload["parent_id"]}

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        agent=AlgorithmAgent(provider=provider),
    )

    outcome = loop.run_round()

    assert outcome.status == "failed"
    assert outcome.decision["failure_type"] == "invalid_block_selection"
    assert (tmp_path / "run" / "rounds" / outcome.round_id / "design.json").is_file()
    gene_lines = (tmp_path / "run" / "design-genes.jsonl").read_text().splitlines()
    assert len(gene_lines) == 1
    assert '"status": "failed"' in gene_lines[0]


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


def test_evo_compress_features_form_restartable_cpu_fake_pipeline(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        advanced_profiling=True,
        algorithm_improvement=True,
        search_space_evolution=True,
        candidate_screening=True,
    )

    first = loop.run_round()
    second = loop.run_round()

    assert first.status in {"promoted", "rejected"}
    assert second.status in {"promoted", "rejected"}
    first_dir = tmp_path / "run" / "rounds" / first.round_id
    second_dir = tmp_path / "run" / "rounds" / second.round_id
    first_profile = json.loads((first_dir / "profile.json").read_text())
    assert first_profile["calibration"]["provider"] == "FakeWanCalibrationProvider"
    assert first_profile["evidence"]["advanced"] is True
    search_space = json.loads((first_dir / "search-space.json").read_text())
    assert search_space["version_id"] == "S0002"
    assert "prune_ffn_by_weight_threshold" in {item["name"] for item in search_space["operators"]}
    screening = json.loads((second_dir / "screening" / "screening.json").read_text())
    assert [item["stage"] for item in screening["results"]] == ["static", "diagnostic", "full"]
    validation = json.loads((second_dir / "algorithm-validation.json").read_text())
    assert validation["parent_algorithm_id"] == "A0001"
    assert validation["evidence_level"] == "cpu_fake"
    algorithm = json.loads((second_dir / "algorithm.json").read_text())
    assert algorithm["parent_algorithm_id"] == "A0001"
    assert algorithm["parent_source_sha256"] == validation["parent_source_sha256"]
    assert algorithm["validation_status"] in {"validated", "rejected"}
    genes = [json.loads(line) for line in (tmp_path / "run" / "design-genes.jsonl").read_text().splitlines()]
    assert any(item["gene_id"].startswith("G-SS-") for item in genes)
    assert any(item["gene_id"] == "G-" + first.round_id for item in genes)
    assert any(item["gene_id"] == "G-" + second.round_id for item in genes)
