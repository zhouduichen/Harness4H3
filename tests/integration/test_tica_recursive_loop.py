from __future__ import annotations

import json

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution import EvolutionLoop


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

    validation_path = second_dir / "algorithm-validation.json"
    assert validation_path.exists()
    validation = json.loads(validation_path.read_text())
    assert validation["parent_source_sha256"] == first_algorithm["source_sha256"]
    assert validation["candidate_source_sha256"] != validation["parent_source_sha256"]
    assert validation["calibration_digest"]
    assert validation["search_budget"]["max_full_evals"] == 4
    assert validation["evidence_level"] == "cpu_fake_quality_proxy"
    child_algorithm = json.loads((second_dir / "algorithm.json").read_text())
    assert child_algorithm["parent_algorithm_id"] == first_algorithm["algorithm_id"]
    assert child_algorithm["source_diff"]
    assert child_algorithm["metadata"]["algorithm_improvement"] is True
