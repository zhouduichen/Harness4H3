from __future__ import annotations

import json

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution import EvolutionLoop
from harness4h3.operators.wan import OperatorSpec, SearchSpaceManager


def test_tica_loop_keeps_human_baseline_and_records_fair_recursive_comparison(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        tica_prune=True,
        algorithm_improvement=True,
    )

    first = loop.run_round()
    second = loop.run_round()
    first_dir = tmp_path / "run" / "rounds" / first.round_id
    second_dir = tmp_path / "run" / "rounds" / second.round_id

    first_algorithm = json.loads((first_dir / "algorithm.json").read_text())
    assert first_algorithm["metadata"]["origin"] == "human_baseline"
    assert first_algorithm["metadata"]["not_agent_discovered"] is True
    assert first_algorithm["source_diff"] == ""

    search = json.loads((first_dir / "tica-search.json").read_text())
    assert len(search["candidates"]) == 4
    assert {item["kind"] for item in search["candidates"]} == {
        "independent_block",
        "tica_block",
        "interaction_block",
        "tica_block_ffn",
    }
    interaction = next(item for item in search["candidates"] if item["kind"] == "interaction_block")
    assert interaction["interaction_metrics"]["available"] is True
    assert interaction["interaction_metrics"]["interaction_delta"] is not None
    assert interaction["full_metrics"]["structural_change"]["kind"] == "tica_prune"
    first_evaluation = json.loads((first_dir / "evaluation.json").read_text())
    assert first_evaluation["evaluation_split"] == "validation"
    assert first_evaluation["measurement_manifest_digest"] == loop.tica_splits.validation.digest()
    assert first_evaluation["measurement_manifest_digest"] != loop.tica_splits.calibration.digest()

    validation_path = second_dir / "algorithm-validation.json"
    assert validation_path.exists()
    validation = json.loads(validation_path.read_text())
    assert validation["parent_source_sha256"] == first_algorithm["source_sha256"]
    assert validation["candidate_source_sha256"] != validation["parent_source_sha256"]
    assert validation["calibration_digest"]
    assert validation["search_budget"]["max_full_evals"] == 4
    assert validation["evidence_level"] == "cpu_fake_quality_proxy"
    assert all(item["parent_metrics"]["algorithm_search"]["complete"] for item in validation["scenario_results"])
    assert all(item["candidate_metrics"]["algorithm_search"]["complete"] for item in validation["scenario_results"])
    assert all(len(item["parent_metrics"]["algorithm_search"]["candidates"]) > 1 for item in validation["scenario_results"])
    assert all(len(item["candidate_metrics"]["algorithm_search"]["candidates"]) > 1 for item in validation["scenario_results"])
    assert all(item["parent_metrics"]["search_budget"] == item["candidate_metrics"]["search_budget"] for item in validation["scenario_results"])
    child_algorithm = json.loads((second_dir / "algorithm.json").read_text())
    assert child_algorithm["parent_algorithm_id"] == first_algorithm["algorithm_id"]
    assert child_algorithm["source_diff"]
    assert child_algorithm["metadata"]["algorithm_improvement"] is True


def test_tica_loop_performs_three_recursive_source_generations(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        tica_prune=True,
        algorithm_improvement=True,
    )

    rounds = [loop.run_round() for _ in range(3)]
    algorithms = [
        json.loads((tmp_path / "run" / "rounds" / item.round_id / "algorithm.json").read_text())
        for item in rounds
    ]
    assert [item["algorithm_id"] for item in algorithms] == ["A0001", "A0002", "A0003"]
    assert len({item["source_sha256"] for item in algorithms}) == 3
    assert algorithms[1]["parent_algorithm_id"] == algorithms[0]["algorithm_id"]
    assert algorithms[2]["parent_algorithm_id"] in {algorithms[0]["algorithm_id"], algorithms[1]["algorithm_id"]}
    assert all(item["source_diff"] for item in algorithms[1:])
    assert all(
        (tmp_path / "run" / "rounds" / item.round_id / "algorithm-program-trace.json").exists()
        for item in rounds
    )


def test_active_timestep_gate_enters_algorithm_candidates_and_executes(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    manager = SearchSpaceManager(tmp_path / "space")
    suggestion = AlgorithmAgent().propose_search_space_change(
        search_space=manager.active.to_dict(),
        history=[],
        failures=["timestep_latency_plateau"],
    )
    proposal = manager.propose(
        OperatorSpec.from_dict(suggestion["spec"]),
        suggestion["rationale"],
        suggestion["evidence_ids"],
    )
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "space-validation")
    manager.register(validated)

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        tica_prune=True,
        algorithm_improvement=True,
        search_space_evolution=True,
        search_space_manager=manager,
    )
    loop.run_round()
    second = loop.run_round()
    search = json.loads((tmp_path / "run" / "rounds" / second.round_id / "tica-search.json").read_text())
    gate = next(item for item in search["candidates"] if item["kind"] == "timestep_block_gate")
    assert gate["full_metrics"]["valid"] is True
    assert gate["full_metrics"]["structural_change"]["runtime_operator"] == "timestep_block_gate"
    assert gate["full_metrics"]["structural_change"]["parameter_reduction_ratio"] == 0.0
    assert "timestep_block_gate" in search["algorithm_trace"][0]["search_space_operator_names"]
