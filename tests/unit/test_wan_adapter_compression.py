from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.operators.wan import BlockPruningOperator, FFNChannelPruningOperator


def test_fake_adapter_discovers_blocks_and_ffn_width():
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline", device="cpu")
    structure = adapter.inspect(model)
    assert structure.block_count == 4
    assert structure.ffn_width == 16
    assert structure.parameter_count == sum(int(item.numel()) for item in model.blocks.parameters()) + sum(int(item.numel()) for item in model.output.parameters())
    assert tuple(item.block_id for item in adapter.get_blocks(model)) == ("block.0", "block.1", "block.2", "block.3")


def test_fake_adapter_save_reload_preserves_forward_shape(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    model = adapter.load_model("fake://baseline")
    artifact = adapter.save_model(model, tmp_path / "child")
    reloaded = adapter.reload_model(artifact)
    assert tuple(reloaded(torch.zeros(2, 8)).shape) == (2, 8)


def test_fake_adapter_rejects_parent_or_child_bytes_changed_after_save(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    artifact = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "checkpoint")
    model_path = tmp_path / "checkpoint" / "model.pt"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        adapter.reload_model(artifact)


def test_block_pruning_updates_structure_and_parent_digest(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = BlockPruningOperator().execute(adapter, parent, {"block_ids": ["block.1"]}, tmp_path / "child")
    assert result.status == "success"
    assert result.structural_change["retained_blocks"] == 3
    assert result.parent_checkpoint_sha256 == parent.sha256
    assert result.child_checkpoint is not None
    assert adapter.inspect(adapter.reload_model(result.child_checkpoint)).block_count == 3


def test_channel_pruning_updates_paired_projections(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = FFNChannelPruningOperator().execute(
        adapter,
        parent,
        {"block_ids": ["block.0"], "retained_indices": list(range(8))},
        tmp_path / "child",
    )
    assert result.status == "success"
    child = adapter.reload_model(result.child_checkpoint)
    assert child.blocks[0].up_proj.out_features == 8
    assert child.blocks[0].down_proj.in_features == 8
    assert child.blocks[1].up_proj.out_features == 16


def test_invalid_structural_mutations_fail_closed(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=2, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    removed_all = BlockPruningOperator().execute(adapter, parent, {"block_ids": ["block.0", "block.1"]}, tmp_path / "all")
    empty_channels = FFNChannelPruningOperator().execute(adapter, parent, {"block_ids": ["block.0"], "retained_indices": []}, tmp_path / "empty")
    assert removed_all.failure_type == "invalid_block_selection"
    assert empty_channels.failure_type == "invalid_channel_selection"
