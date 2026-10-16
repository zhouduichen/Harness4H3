from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.operators.wan import OperatorSpec, SearchSpaceManager, TimestepAttentionPolicyOperator
from harness4h3.sandbox.runner import CompressionPlan


def _plan() -> dict[str, object]:
    return {
        "rules": [
            {"min_timestep": 0.0, "max_timestep": 0.3, "keep_ratio": 0.5, "block_ids": ["block.0"], "score": "l2_norm"},
            {"min_timestep": 0.7, "max_timestep": 1.0, "keep_ratio": 0.75, "block_ids": ["block.0"], "score": "l2_norm"},
        ],
        "policy": "select_kv_tokens",
    }


def test_timestep_attention_policy_changes_kv_work_per_sample_and_reloads(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    result = TimestepAttentionPolicyOperator().execute(adapter, parent, _plan(), tmp_path / "child")

    assert result.status == "success"
    assert result.child_checkpoint is not None
    assert result.structural_change["parameter_reduction_ratio"] == 0.0
    assert result.structural_change["per_sample_timestep"] is True

    child = adapter.reload_model(result.child_checkpoint)
    hidden = torch.randn(2, 8, 8)
    with torch.no_grad():
        output = child(hidden, timestep=torch.tensor([100, 900]))
    assert tuple(output.shape) == tuple(hidden.shape[:-1]) + (8,)
    stats = child.runtime_timestep_attention_stats["block.0"]
    assert stats["calls"] == 2
    assert stats["full_kv_tokens"] == 16
    assert stats["selected_kv_tokens"] == 10
    assert set(stats["by_rule"]) == {"0.0:0.3", "0.7:1.0"}


def test_timestep_attention_policy_is_strict_and_sandbox_visible():
    plan = CompressionPlan.from_mapping({"operator": "timestep_attention_policy", "args": _plan()})
    assert plan.operator == "timestep_attention_policy"
    with pytest.raises(ValueError, match="overlapping"):
        TimestepAttentionPolicyOperator  # keep the operator import covered by the contract test
        from harness4h3.operators.wan.runtime_gating import validate_timestep_attention_plan

        validate_timestep_attention_plan(
            {
                "rules": [
                    {"min_timestep": 0.0, "max_timestep": 0.6, "keep_ratio": 0.5},
                    {"min_timestep": 0.5, "max_timestep": 1.0, "keep_ratio": 0.75},
                ],
                "policy": "select_kv_tokens",
            },
            ["block.0"],
        )


def test_timestep_attention_policy_can_be_validated_and_registered_as_evolved_space(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    manager = SearchSpaceManager(tmp_path / "space")
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
    validated = manager.validate(proposal, adapter, parent, output_dir=tmp_path / "validation")
    assert validated.status == "validated", validated.rejection_reason
    version = manager.register(validated)
    assert "timestep_attention_policy" in version.operator_names()
