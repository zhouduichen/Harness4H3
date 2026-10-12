from __future__ import annotations

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution import EvolutionLoop


def test_evolution_loop_runs_fake_round_and_persists_evidence(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    outcome = loop.run_round()
    assert outcome.status == "promoted"
    assert outcome.child_model_id == "M0001"
    assert (tmp_path / "run" / "rounds" / outcome.round_id / "decision.json").exists()
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
    record = list(loop.experience_store.read())[0]
    assert record.status == "failed"
    assert record.decision["failure_type"] == "sandbox_policy"
    assert record.decision["search_policy"]["exploit_ratio"] == 0.6
