from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.operators.wan import build_wan_compression_registry


def _parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    return adapter, parent


def test_registry_executes_bounded_composite_plan_step_by_step(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry().execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
                {"operator": "prune_ffn_channels", "args": {"block_ids": ["block.1"], "retained_indices": list(range(8))}},
            ]
        },
        tmp_path / "composite",
    )

    assert result.status == "success"
    assert result.child_checkpoint is not None
    assert (tmp_path / "composite" / "step-00").is_dir()
    assert (tmp_path / "composite" / "step-01").is_dir()
    assert len(result.structural_change["steps"]) == 2
    child = adapter.reload_model(result.child_checkpoint)
    assert len(child.blocks) == 3
    assert child.blocks[0].up_proj.out_features == 8


def test_composite_failure_does_not_return_partial_child(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry().execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
                {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
            ]
        },
        tmp_path / "composite-failure",
    )

    assert result.status == "failed"
    assert result.child_checkpoint is None
    assert result.failure_type == "invalid_block_selection"
    assert result.structural_change["completed_steps"] == 1
    assert (tmp_path / "composite-failure" / "step-00").is_dir()


def test_registry_keeps_legacy_operator_names():
    assert build_wan_compression_registry().names() == (
        "prune_blocks",
        "prune_ffn_channels",
        "compose",
    )
