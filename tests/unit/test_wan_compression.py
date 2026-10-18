from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.operators.wan import build_wan_compression_registry
from harness4h3.sandbox import AlgorithmSandbox


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


def test_external_runtime_operators_round_trip_and_remain_distinct(tmp_path):
    adapter, parent = _parent(tmp_path)
    registry = build_wan_compression_registry(include_multigranular=True)
    assert "sparse_runtime" in registry.names()
    assert "svoo_attention" in registry.names()
    assert "magcache" in registry.names()
    result = registry.execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": "svoo_attention", "args": {"method": "svoo", "model_key": "wan21-t2v-1.3b", "config": {"upstream_extra": 3}}},
                {"operator": "magcache", "args": {"method": "magcache", "config": {"threshold": 0.12}}},
            ]
        },
        tmp_path / "external-composite",
    )

    assert result.status == "success"
    assert [item["operator"] for item in result.structural_change["steps"]] == ["svoo_attention", "magcache"]
    assert all(item["structural_change"]["parameter_reduction_ratio"] == 0.0 for item in result.structural_change["steps"])
    child = adapter.reload_model(result.child_checkpoint)
    assert [item["operator"] for item in child._harness4h3_external_runtime_policies] == ["svoo_attention", "magcache"]
    assert child._harness4h3_external_runtime_policies[0]["args"]["config"]["upstream_extra"] == 3


def test_profile_guided_exploration_remains_block_only(tmp_path):
    adapter, parent = _parent(tmp_path)
    profile = {
        "model": {"ffn_width": 16},
        "layers": [
            {"block_id": "block.0", "latency_ms": 1.0, "estimated_flops": 10.0},
            {"block_id": "block.1", "latency_ms": 5.0, "estimated_flops": 50.0},
            {"block_id": "block.2", "latency_ms": 3.0, "estimated_flops": 30.0},
            {"block_id": "block.3", "latency_ms": 2.0, "estimated_flops": 20.0},
        ],
        "block_sensitivity": [
            {"block_id": "block.0", "importance_score": 0.1, "available": True},
            {"block_id": "block.1", "importance_score": 0.2, "available": True},
            {"block_id": "block.2", "importance_score": 0.3, "available": True},
            {"block_id": "block.3", "importance_score": 0.4, "available": True},
        ],
        "channel_sensitivity": [{"block_id": "block.1", "available": True}],
    }
    candidate = AlgorithmAgent(explore_ffn_retention_ratio=0.5).generate(
        "A0001", parent_id="M0000", profile=profile, history=[], search_choice="explore"
    )
    sandbox = AlgorithmSandbox(tmp_path / "sandbox")
    sandbox_result = sandbox.run(candidate.source, profile, [])
    assert sandbox_result.status == "success"
    assert sandbox_result.plan is not None
    result = build_wan_compression_registry().execute(
        "prune_blocks", adapter, parent, sandbox_result.plan.args, tmp_path / "generated-block-only"
    )
    assert result.status == "success"
    assert result.child_checkpoint is not None
    child = adapter.reload_model(result.child_checkpoint)
    assert len(child.blocks) == 3
    assert all(block.up_proj.out_features == 16 for block in child.blocks)


def test_composite_normalizes_all_remaining_blocks_for_reindexed_adapter(tmp_path):
    class ReindexedFakeAdapter(FakeWanDiTAdapter):
        channel_pruning_requires_all_blocks = True

        def remove_blocks(self, model, block_ids):
            change = super().remove_blocks(model, block_ids)
            model.block_ids = ["block.%d" % index for index in range(len(model.blocks))]
            return change

    adapter = ReindexedFakeAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
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
        tmp_path / "reindexed-composite",
    )
    assert result.status == "success"
    assert result.structural_change["steps"][1]["args"]["block_ids"] == ["block.0", "block.1", "block.2"]
