from __future__ import annotations

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.operators.wan import build_wan_compression_registry


def test_tica_operator_executes_global_ffn_recipe_and_reloads(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = build_wan_compression_registry(include_tica=True).execute(
        "tica_prune",
        adapter,
        parent,
        {
            "block_ids": ["block.0"],
            "retained_indices": list(range(8)),
            "mode": "block_plus_ffn",
            "time_weight_formula": "early_log",
            "risk_limit": 0.25,
            "activation_weight": 0.5,
            "interaction_weight": 0.5,
            "beam_width": 2,
            "max_fast_evals": 8,
            "max_full_evals": 4,
            "max_interaction_evals": 2,
        },
        tmp_path / "child",
    )
    assert result.status == "success"
    reloaded = adapter.reload_model(result.child_checkpoint)
    structure = adapter.inspect(reloaded)
    assert structure.block_count == 3
    assert structure.ffn_width == 8
    assert result.structural_change["mode"] == "block_plus_ffn"


def test_tica_operator_rejects_partial_global_alignment(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = build_wan_compression_registry(include_tica=True).execute(
        "tica_prune",
        adapter,
        parent,
        {
            "block_ids": ["block.0"],
            "retained_indices": [0, 1],
            "per_layer_retained_indices": {"block.0": [0], "block.1": [0, 1]},
            "mode": "block_plus_ffn",
        },
        tmp_path / "child",
    )
    assert result.status == "failed"
    assert result.failure_type == "global_ffn_alignment_required"


def test_tica_operator_can_run_block_only_without_ffn_mutation(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = build_wan_compression_registry(include_tica=True).execute(
        "tica_prune",
        adapter,
        parent,
        {"block_ids": ["block.1"], "mode": "block"},
        tmp_path / "child",
    )
    assert result.status == "success"
    structure = adapter.inspect(adapter.reload_model(result.child_checkpoint))
    assert structure.block_count == 3
    assert structure.ffn_width == 16
