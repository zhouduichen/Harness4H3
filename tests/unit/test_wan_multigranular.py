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
        ("token_merge", {"keep_ratio": 0.5, "mode": "adjacent"}),
        ("kv_token_select", {"keep_ratio": 0.5, "score": "l2_norm"}),
        ("sparse_runtime", {"keep_ratio": 0.5, "attention": "attn1"}),
        ("attention_window", {"window_size": 3, "attention": "attn1"}),
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


def test_attention_window_reduces_fake_attention_context(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "attention_window", adapter, parent, {"window_size": 2, "attention": "attn1"}, tmp_path / "window"
    )
    assert result.status == "success", result.message
    child = adapter.reload_model(result.child_checkpoint)
    hidden = torch.randn(1, 7, 8)
    with torch.no_grad():
        child.blocks[0].attn1(hidden)
    stats = child.blocks[0].attn1._harness4h3_window_stats
    assert stats["query_tokens"] == 7
    assert stats["selected_kv_tokens"] < 7 * 7
    assert stats["windows"] == 4


def test_token_merge_reduces_fake_attention_and_unmerges_after_reload(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "token_merge", adapter, parent, {"keep_ratio": 0.5, "mode": "adjacent"}, tmp_path / "merge"
    )
    assert result.status == "success", result.message
    child = adapter.reload_model(result.child_checkpoint)
    hidden = torch.randn(2, 7, 8)
    with torch.no_grad():
        output = child.blocks[0].attn1(hidden)
    assert tuple(output.shape) == (2, 7, 8)
    stats = child.blocks[0].attn1._harness4h3_token_merge_stats
    assert stats["input_tokens"] == 7
    assert stats["merged_tokens"] == 4
    assert len(stats["anchors"]) == 4
    assert child._harness4h3_token_merge_policy["keep_ratio"] == 0.5


def test_token_merge_ratio_one_is_dense_equivalent():
    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=12)
    plain = adapter.load_model("fake://baseline")
    merged = adapter.load_model("fake://baseline")
    adapter.apply_token_merge(merged, {"keep_ratio": 1.0, "mode": "adjacent"})
    hidden = torch.randn(2, 7, 8)
    with torch.no_grad():
        expected = plain.blocks[0].attn1(hidden)
        actual = merged.blocks[0].attn1(hidden)
    assert torch.equal(expected, actual)


def test_kv_token_selection_keeps_full_queries_and_reduces_kv_after_reload(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "kv_token_select", adapter, parent, {"keep_ratio": 0.5, "score": "l2_norm"}, tmp_path / "kv-select"
    )
    assert result.status == "success", result.message
    child = adapter.reload_model(result.child_checkpoint)
    hidden = torch.randn(2, 7, 8)
    with torch.no_grad():
        output = child.blocks[0].attn1(hidden)
    assert tuple(output.shape) == (2, 7, 8)
    stats = child.blocks[0].attn1._harness4h3_kv_token_selection_stats
    assert stats["query_tokens"] == 7
    assert stats["full_kv_tokens"] == 7
    assert stats["selected_kv_tokens"] == 4

    targeted = adapter.load_model("fake://baseline")
    adapter.apply_kv_token_selection(targeted, {"keep_ratio": 0.5, "block_ids": ["block.1"], "score": "l2_norm"})
    with torch.no_grad():
        targeted.blocks[0].attn1(torch.randn(1, 7, 8))
        targeted.blocks[1].attn1(torch.randn(1, 7, 8))
    assert not hasattr(targeted.blocks[0].attn1, "_harness4h3_kv_token_selection_stats")
    assert targeted.blocks[1].attn1._harness4h3_kv_token_selection_stats["selected_kv_tokens"] == 4


def test_kv_selection_and_sparse_runtime_compose_into_one_reduced_kv_path(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": "kv_token_select", "args": {"keep_ratio": 0.75, "score": "l2_norm"}},
                {"operator": "sparse_runtime", "args": {"keep_ratio": 0.5, "attention": "attn1"}},
            ]
        },
        tmp_path / "kv-sparse",
    )
    assert result.status == "success", result.message
    child = adapter.reload_model(result.child_checkpoint)
    with torch.no_grad():
        output = child.blocks[0].attn1(torch.randn(1, 8, 8))
    assert tuple(output.shape) == (1, 8, 8)
    assert child.blocks[0].attn1._harness4h3_sparse_stats["selected_tokens"] < 8


def test_head_pruning_and_sparse_runtime_compose_after_reload(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": "prune_attention_heads", "args": {"head_indices": [0], "attention_names": ["attn1", "attn2"]}},
                {"operator": "sparse_runtime", "args": {"keep_ratio": 0.5, "attention": "attn1"}},
            ]
        },
        tmp_path / "head-sparse",
    )
    assert result.status == "success", result.message
    child = adapter.reload_model(result.child_checkpoint)
    assert child.blocks[0].attn1.heads == 3
    with torch.no_grad():
        output = child.blocks[0].attn1(torch.randn(1, 8, 8))
    assert tuple(output.shape) == (1, 8, 8)
    assert child.blocks[0].attn1._harness4h3_sparse_stats["selected_tokens"] < 8


def test_layer_pruning_and_sampling_schedule_compose_in_order(tmp_path):
    adapter, parent = _parent(tmp_path)
    result = build_wan_compression_registry(include_multigranular=True).execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": "prune_layers", "args": {"block_ids": ["block.0"]}},
                {"operator": "sampling_step_skip", "args": {"target_steps": 4, "strategy": "uniform"}},
            ]
        },
        tmp_path / "layer-sampling",
    )
    assert result.status == "success", result.message
    child = adapter.reload_model(result.child_checkpoint)
    assert len(child.blocks) == 3
    assert len(adapter.sampling_steps(child, tuple(range(16, 0, -1)))) == 4
    assert [item["operator"] for item in child._harness4h3_runtime_operations] == ["sampling_step_skip"]


def test_attention_window_rejects_cross_attention_and_sparse_composition(tmp_path):
    adapter, parent = _parent(tmp_path)
    model = adapter.load_model("fake://baseline")
    with pytest.raises(ValueError, match="self-attention only"):
        adapter.apply_attention_window(model, {"window_size": 2, "attention": "all"})

    result = build_wan_compression_registry(include_multigranular=True).execute(
        "compose",
        adapter,
        parent,
        {
            "steps": [
                {"operator": "sparse_runtime", "args": {"keep_ratio": 0.5, "attention": "attn1"}},
                {"operator": "attention_window", "args": {"window_size": 2, "attention": "attn1"}},
            ]
        },
        tmp_path / "incompatible",
    )
    assert result.status == "failed"
    assert "incompatible_runtime_attention_composition" in result.message


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
    for block in model.blocks:
        block.attn1.fuse_projections()
        block.attn2.fuse_projections()
    before = sum(int(item.numel()) for item in model.parameters())
    change = adapter.prune_attention_heads(model, [0], ("attn1", "attn2"))
    assert change.details["parameter_reduction_ratio"] > 0.0
    assert sum(int(item.numel()) for item in model.parameters()) < before
    adapter.apply_token_sparsity(model, {"keep_ratio": 0.5})
    adapter.apply_sparse_runtime(model, {"keep_ratio": 0.5, "attention": "attn1"})
    rotary = (torch.ones(1, 5, 1, 2), torch.zeros(1, 5, 1, 2))
    with torch.no_grad():
        sparse_output = model.blocks[0].attn1(torch.randn(1, 5, 8), None, None, rotary)
    assert tuple(sparse_output.shape) == (1, 5, 8)
    assert model._harness4h3_sparse_runtime_stats["selected_tokens"] < model._harness4h3_sparse_runtime_stats["full_tokens"]
    assert [item["operator"] for item in model.config.harness4h3_runtime_operations] == [
        "prune_tokens",
        "sparse_runtime",
    ]
    output = model(torch.randn(1, 5, 8), torch.tensor([500]), torch.randn(1, 4, 8))
    assert tuple(output.shape) == (1, 5, 8)

    merge_model = TinyWan()
    adapter.apply_token_merge(merge_model, {"keep_ratio": 0.5, "mode": "adjacent"})
    with torch.no_grad():
        merge_output = merge_model.blocks[0].attn1(torch.randn(1, 5, 8), None, None, rotary)
    assert tuple(merge_output.shape) == (1, 5, 8)
    assert merge_model._harness4h3_token_merge_stats["merged_tokens"] == 2

    kv_model = TinyWan()
    adapter.apply_kv_token_selection(kv_model, {"keep_ratio": 0.5, "score": "l2_norm"})
    with torch.no_grad():
        kv_output = kv_model.blocks[0].attn1(torch.randn(1, 5, 8), None, None, rotary)
    assert tuple(kv_output.shape) == (1, 5, 8)
    assert kv_model._harness4h3_kv_token_selection_stats["selected_kv_tokens"] == 2

    window_model = TinyWan()
    adapter.apply_attention_window(window_model, {"window_size": 2, "attention": "attn1"})
    with torch.no_grad():
        window_output = window_model.blocks[0].attn1(torch.randn(1, 5, 8), None, None, rotary)
    assert tuple(window_output.shape) == (1, 5, 8)
    assert window_model._harness4h3_attention_window_stats["selected_kv_tokens"] == 5
    adapter.apply_timestep_block_gating(
        model,
        [{"block_id": "block.0", "min_timestep": 0.0, "max_timestep": 0.3}],
    )
    mixed = model(torch.randn(2, 5, 8), torch.tensor([100, 900]), torch.randn(2, 4, 8))
    assert tuple(mixed.shape) == (2, 5, 8)
    assert model._harness4h3_gate_stats["block.0"] == {"executed": 1, "skipped": 1}
    from diffusers import FlowMatchEulerDiscreteScheduler

    model.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    adapter.apply_sampling_step_policy(model, {"target_steps": 4, "strategy": "uniform"})
    model.scheduler.set_timesteps(16, device="cpu")
    assert len(model.scheduler.timesteps) == 4
    assert model._harness4h3_sampling_step_stats["effective_steps"] == 4
