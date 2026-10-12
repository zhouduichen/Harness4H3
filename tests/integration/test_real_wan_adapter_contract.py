from __future__ import annotations

import os

import pytest

from harness4h3.adapters.wan import build_wan_adapter
from harness4h3.operators.wan import BlockPruningOperator


def test_real_wan_adapter_contract_when_checkpoint_is_declared(tmp_path):
    checkpoint = os.environ.get("HARNESS4H3_WAN_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set HARNESS4H3_WAN_CHECKPOINT for the real Wan GPU smoke")
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    adapter = build_wan_adapter("diffusers", local_files_only=True)
    model = adapter.load_model(checkpoint, device=os.environ.get("HARNESS4H3_WAN_DEVICE", "cuda"), dtype="bfloat16")
    parent = adapter.save_model(model, tmp_path / "parent")
    assert parent.structure.block_count >= 2
    block_ids = parent.structure.metadata.get("block_ids", ["block.0"])
    result = BlockPruningOperator().execute(adapter, parent, {"block_ids": [block_ids[0]]}, tmp_path / "child")
    assert result.status == "success"
    assert result.child_checkpoint is not None
    reloaded = adapter.reload_model(result.child_checkpoint)
    assert adapter.inspect(reloaded).block_count == parent.structure.block_count - 1
