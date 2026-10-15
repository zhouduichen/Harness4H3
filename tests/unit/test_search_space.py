from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.operators.wan import (
    OperatorSpec,
    SearchSpaceManager,
    ThresholdFFNChannelOperator,
    build_wan_compression_registry,
)


def _parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    return adapter, parent


def _threshold_spec():
    return OperatorSpec(
        name="prune_ffn_by_weight_threshold",
        version="1",
        argument_schema={"block_ids": "list[str]", "threshold": "float"},
        granularity="channel",
        constraints=("wan_ffn_width_aligned", "threshold_must_change_width"),
        limitations=("weight_proxy_only", "not_real_wan_validated"),
        trusted_factory_key="prune_ffn_by_weight_threshold",
    )


def test_unvalidated_operator_is_described_but_not_executable(tmp_path):
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(_threshold_spec(), "weight proxy plateau", ["R0001"])
    registry = build_wan_compression_registry()
    assert proposal.status == "proposed"
    assert proposal.spec.name not in registry.names()
    with pytest.raises(ValueError, match="unsupported_compression_operator"):
        registry.get(proposal.spec.name)


def test_fake_threshold_operator_validates_registers_and_reloads(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    proposal = manager.propose(_threshold_spec(), "weight proxy plateau", ["R0001"])
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert validated.status == "validated"
    assert validated.validation["offline_simulation"] is True
    version = manager.register(validated)
    assert version.status == "validated"
    assert proposal.spec.name in version.operator_names()

    registry = manager.registry_for(version, build_wan_compression_registry())
    assert proposal.spec.name in registry.names()
    result = registry.execute(
        proposal.spec.name,
        adapter,
        parent,
        {"block_ids": ["block.0"], "threshold": float(validated.validation["threshold"])},
        tmp_path / "child",
    )
    assert result.status == "success"
    assert result.child_checkpoint is not None
    child = adapter.reload_model(result.child_checkpoint)
    assert child.blocks[0].up_proj.out_features < 16
    assert all(block.up_proj.out_features == 16 for block in child.blocks[1:])
    assert result.structural_change["operator"] == "prune_ffn_by_weight_threshold"
    assert result.structural_change["block_ids"] == ["block.0"]


def test_rejected_proposal_cannot_register_or_mutate_registry(tmp_path):
    adapter, parent = _parent(tmp_path)
    manager = SearchSpaceManager(tmp_path / "space")
    invalid = OperatorSpec(
        name="unknown_operator",
        version="1",
        argument_schema={"x": "int"},
        granularity="unknown",
        trusted_factory_key="not_trusted",
    )
    proposal = manager.propose(invalid, "invalid factory", [])
    rejected = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "bad")
    assert rejected.status == "rejected"
    assert rejected.rejection_reason
    with pytest.raises(ValueError, match="only_validated"):
        manager.register(rejected)
    assert "unknown_operator" not in manager.registry_for().names()


def test_threshold_operator_rejects_non_structural_threshold(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = ThresholdFFNChannelOperator().execute(
        adapter, parent, {"block_ids": ["block.0"], "threshold": 0.0000001}, tmp_path / "unchanged"
    )
    assert result.status == "failed"
    assert result.child_checkpoint is None
