from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter, ModelVariant
from harness4h3.operators.wan import build_wan_compression_registry


def _parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    return adapter, parent


def test_runtime_only_variant_manifest_preserves_order_and_zero_weight_claim(tmp_path):
    adapter, parent = _parent(tmp_path)
    plan = {
        "operator": "compose",
        "args": {
            "steps": [
                {"operator": "prune_tokens", "args": {"keep_ratio": 0.75}},
                {"operator": "sparse_runtime", "args": {"keep_ratio": 0.5, "attention": "attn1"}},
                {"operator": "sampling_step_skip", "args": {"target_steps": 8}},
            ]
        },
    }
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "compose", adapter, parent, plan["args"], tmp_path / "runtime-child"
    )
    assert result.status == "success"

    variant = ModelVariant.from_compression(
        "M0001",
        parent,
        result,
        plan,
        algorithm_id="A0001",
        search_space_version="S0001",
        source_revision="test-source",
        runtime_revision="test-runtime",
    )
    assert variant.runtime_only is True
    assert variant.structural_checkpoint is None
    assert variant.structural_checkpoint_sha256 is None
    assert variant.base_checkpoint_sha256 == parent.sha256
    assert [item["operator"] for item in variant.ordered_operations] == [
        "prune_tokens",
        "sparse_runtime",
        "sampling_step_skip",
    ]
    assert set(variant.runtime_policy) == {"prune_tokens", "sparse_runtime"}
    assert set(variant.sampler_policy) == {"sampling_step_skip"}

    path = variant.write_manifest(tmp_path / "manifests" / "variant_manifest.json")
    restored = ModelVariant.read_manifest(path)
    assert restored.to_dict() == variant.to_dict()
    assert restored.variant_digest != parent.sha256


def test_static_variant_records_structural_checkpoint(tmp_path):
    adapter, parent = _parent(tmp_path)
    plan = {
        "operator": "compose",
        "args": {
            "steps": [
                {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
                {"operator": "sampling_step_skip", "args": {"target_steps": 8}},
            ]
        },
    }
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "compose", adapter, parent, plan["args"], tmp_path / "static-child"
    )
    assert result.status == "success"
    variant = ModelVariant.from_compression("M0002", parent, result, plan)
    assert variant.runtime_only is False
    assert variant.structural_checkpoint == result.child_checkpoint.path
    assert variant.structural_checkpoint_sha256 == result.child_checkpoint.sha256


def test_head_sparse_hybrid_executes_in_declared_order(tmp_path):
    adapter, parent = _parent(tmp_path)
    plan = {
        "operator": "compose",
        "args": {
            "steps": [
                {"operator": "prune_attention_heads", "args": {"head_indices": [0]}},
                {"operator": "sparse_runtime", "args": {"keep_ratio": 0.5, "attention": "attn1"}},
            ]
        },
    }
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "compose", adapter, parent, plan["args"], tmp_path / "head-sparse"
    )
    assert result.status == "success", result.message
    assert [item["operator"] for item in result.structural_change["steps"]] == [
        "prune_attention_heads",
        "sparse_runtime",
    ]
    child = adapter.reload_model(result.child_checkpoint)
    assert child.blocks[0].attn1.heads == 3
