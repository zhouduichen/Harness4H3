from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.tica_calibration import TICACalibrationManifest
from harness4h3.operators.wan import TimestepBlockGatingOperator


def test_timestep_gate_changes_selected_denoising_steps_and_reloads(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent_model = adapter.load_model("fake://baseline")
    parent = adapter.save_model(parent_model, tmp_path / "parent")
    result = TimestepBlockGatingOperator().execute(
        adapter,
        parent,
        {
            "rules": [{"block_id": "block.0", "min_timestep": 0.0, "max_timestep": 0.3}],
            "policy": "skip_when_in_range",
        },
        tmp_path / "child",
    )
    assert result.status == "success"
    assert result.child_checkpoint is not None
    assert result.structural_change["parameter_reduction_ratio"] == 0.0
    assert result.structural_change["dynamic_compute"] is True

    child = adapter.reload_model(result.child_checkpoint)
    manifest = TICACalibrationManifest.from_fake(hidden_size=8, sample_count=1)
    sample = adapter.tica_calibration_inputs(child, manifest)[0]
    high = adapter.forward_denoising(child, sample, 0.9)
    low = adapter.forward_denoising(child, sample, 0.1)
    assert tuple(high.shape) == tuple(low.shape)
    assert child.runtime_gate_stats["block.0"]["skipped"] == 1
    assert child.runtime_gate_stats["block.0"]["executed"] == 1
    assert child.runtime_gate_rules == result.structural_change["rules"]


def test_timestep_gate_does_not_change_parameter_count(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = TimestepBlockGatingOperator().execute(
        adapter,
        parent,
        {
            "rules": [{"block_id": "block.1", "min_timestep": 0.2, "max_timestep": 0.4}],
            "policy": "skip_when_in_range",
        },
        tmp_path / "child",
    )
    assert result.status == "success"
    assert result.child_checkpoint is not None
    child = adapter.reload_model(result.child_checkpoint)
    assert child.parameters()
    assert adapter.inspect(child).parameter_count == adapter.inspect(adapter.reload_model(parent)).parameter_count


def test_timestep_gate_rejects_unknown_block_and_invalid_interval(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    unknown = TimestepBlockGatingOperator().execute(
        adapter,
        parent,
        {"rules": [{"block_id": "block.99", "min_timestep": 0.0, "max_timestep": 0.3}], "policy": "skip_when_in_range"},
        tmp_path / "unknown",
    )
    invalid = TimestepBlockGatingOperator().execute(
        adapter,
        parent,
        {"rules": [{"block_id": "block.0", "min_timestep": 0.8, "max_timestep": 0.2}], "policy": "skip_when_in_range"},
        tmp_path / "invalid",
    )
    assert unknown.status == "failed"
    assert invalid.status == "failed"

