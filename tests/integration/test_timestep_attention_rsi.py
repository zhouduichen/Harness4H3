from __future__ import annotations

import json

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.loop import EvolutionLoop
from harness4h3.operators.wan import OperatorSpec, SearchSpaceManager


def test_activated_timestep_attention_operator_is_evaluated_inside_search_budget(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    manager = SearchSpaceManager(tmp_path / "search-space")
    proposal = manager.propose(
        OperatorSpec(
            "timestep_attention_policy",
            "1",
            {"rules": "list[object]", "policy": "str"},
            "timestep_attention",
            ("per_sample_timestep", "reduced_kv_length", "self_attention_only"),
            ("runtime_only",),
            "timestep_attention_policy",
            execution_domain="runtime",
            implementation_ref="harness4h3.operators.wan.compression.TimestepAttentionPolicyOperator",
        ),
        "block-only timestep gating leaves attention work unchanged",
    )
    proposal = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert proposal.status == "validated", proposal.rejection_reason
    manager.register(proposal)

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        search_space_manager=manager,
        search_space_evolution=True,
        advanced_profiling=True,
        algorithm_improvement=True,
        candidate_screening=True,
        memory_enabled=True,
    )
    outcome = loop.run_round()
    trace = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "algorithm-program-trace.json").read_text())
    results = trace["measured_search"]["candidate_results"]
    candidate = next(item for item in results if item["plan"]["operator"] == "timestep_attention_policy")
    assert candidate["valid"] is True
    assert candidate["failure_reason"] is None
    assert candidate["runtime_execution"]["available"] is True
    assert candidate["runtime_execution"]["hardware_latency_available"] is False
    assert candidate["runtime_execution"]["hardware_latency_reason"]
    assert "timestep_attention" in candidate["runtime_execution"]["policies"]
    assert candidate["runtime_execution"]["measurement_domain"] == "post_forward_runtime_trace"
    assert candidate["runtime_execution"]["derived"]["timestep_attention_reduction_ratio"] > 0.0
    assert len(results) <= 8
