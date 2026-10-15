from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.operators.wan import build_wan_compression_registry
from harness4h3.profiler import profile_model


def _parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    return adapter, parent


@pytest.mark.parametrize(
    ("operator", "plan"),
    (
        ("prune_layers", {"block_ids": ["block.0"]}),
        ("prune_attention_heads", {"head_indices": [0]}),
        ("prune_heads", {"head_indices": [0]}),
        ("prune_tokens", {"keep_ratio": 0.5}),
        ("sparse_runtime", {"keep_ratio": 0.5, "attention": "attn1"}),
        ("sampling_step_skip", {"target_steps": 4}),
    ),
)
def test_multigranular_operator_executes_and_reloads(tmp_path, operator, plan):
    adapter, parent = _parent(tmp_path)
    registry = build_wan_compression_registry(include_multigranular=True)
    result = registry.execute(operator, adapter, parent, plan, tmp_path / operator)

    assert result.status == "success", result.message
    assert result.child_checkpoint is not None
    if operator not in {"prune_layers", "prune_attention_heads", "prune_heads"}:
        assert result.structural_change["child_reload_verified"] is True
    child = adapter.reload_model(result.child_checkpoint)
    assert tuple(child(torch.randn(2, 5, 8)).shape) == (2, 5, 8)


def test_profile_contains_all_search_granularities(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=12)
    model = adapter.load_model("fake://baseline")
    report = profile_model(adapter, model, measure_latency=False)

    assert report.head_sensitivity
    assert report.attention_sensitivity
    assert report.token_sensitivity
    assert report.sampling_step_sensitivity["available"] is True
    payload = report.to_dict()
    assert {"head_sensitivity", "attention_sensitivity", "token_sensitivity", "sampling_step_sensitivity"} <= set(payload)


def test_sampling_schedule_is_executed_by_adapter(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "sampling_step_skip", adapter, parent, {"target_steps": 4, "strategy": "uniform"}, tmp_path / "steps"
    )
    child = adapter.reload_model(result.child_checkpoint)
    selected = adapter.sampling_steps(child, tuple(range(16, 0, -1)))
    assert len(selected) == 4
    assert selected[0] == 16
    assert selected[-1] == 1


def test_runtime_operation_order_survives_fake_checkpoint_reload(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=12)
    model = adapter.load_model("fake://baseline")
    adapter.apply_sampling_step_policy(model, {"target_steps": 8, "strategy": "uniform"})
    adapter.apply_sparse_runtime(model, {"keep_ratio": 0.5, "attention": "attn1"})
    adapter.apply_token_sparsity(model, {"keep_ratio": 0.75, "block_ids": ["block.1"]})
    artifact = adapter.save_model(model, tmp_path / "ordered-runtime")

    reloaded = adapter.reload_model(artifact)
    assert [item["operator"] for item in reloaded._harness4h3_runtime_operations] == [
        "sampling_step_skip",
        "sparse_runtime",
        "prune_tokens",
    ]
    assert reloaded._harness4h3_sampling_step_policy["target_steps"] == 8
    assert reloaded._harness4h3_sparse_runtime_policy["keep_ratio"] == 0.5
    assert reloaded._harness4h3_token_sparsity_policy["block_ids"] == ["block.1"]


def test_diffusers_wan_attention_contract_is_structurally_executable():
    diffusers = pytest.importorskip("diffusers")
    from types import SimpleNamespace
    from torch import nn
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock
    from harness4h3.adapters.wan.diffusers import DiffusersWanAdapter

    class TinyWan(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([WanTransformerBlock(dim=8, ffn_dim=16, num_heads=4) for _ in range(2)])
            self.config = SimpleNamespace(hidden_size=8)

        def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=True, **kwargs):
            temb = torch.zeros(hidden_states.shape[0], 6, 8, device=hidden_states.device)
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, temb, None)
            return hidden_states

    model = TinyWan()
    adapter = DiffusersWanAdapter()
    before = sum(int(item.numel()) for item in model.parameters())
    change = adapter.prune_attention_heads(model, [0], ("attn1", "attn2"))
    assert change.details["parameter_reduction_ratio"] > 0.0
    assert sum(int(item.numel()) for item in model.parameters()) < before
    adapter.apply_token_sparsity(model, {"keep_ratio": 0.5})
    adapter.apply_sparse_runtime(model, {"keep_ratio": 0.5, "attention": "attn1"})
    assert [item["operator"] for item in model.config.harness4h3_runtime_operations] == [
        "prune_tokens",
        "sparse_runtime",
    ]
    output = model(torch.randn(1, 5, 8), torch.tensor([500]), torch.randn(1, 4, 8))
    assert tuple(output.shape) == (1, 5, 8)
