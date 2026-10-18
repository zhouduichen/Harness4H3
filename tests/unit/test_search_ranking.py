import math

from harness4h3.evolution.search_ranking import (
    budgeted_candidate_order,
    candidate_family,
    efficiency_reduction_ratio,
    measured_candidate_key,
    measured_pareto_key,
    measured_search_score,
    pareto_front_indices,
)


def test_budgeted_order_interleaves_families_and_keeps_agent_order_within_family():
    plans = [
        {"operator": "prune_blocks", "args": {"block_ids": ["block.0"]}},
        {"operator": "prune_blocks", "args": {"block_ids": ["block.1"]}},
        {"operator": "head_gating", "args": {"attention": "attn1", "keep_ratio": 0.75}},
        {"operator": "prune_tokens", "args": {"keep_ratio": 0.75}},
        {"operator": "attention_window", "args": {"window_size": 8, "attention": "attn1"}},
        {"operator": "sampling_step_skip", "args": {"target_steps": 8}},
        {"operator": "compose", "args": {"steps": []}},
        {"operator": "kv_token_select", "args": {"keep_ratio": 0.75}},
    ]
    specs = {
        "prune_blocks": {"granularity": "block"},
        "head_gating": {"granularity": "head_runtime"},
        "prune_tokens": {"granularity": "token"},
        "attention_window": {"granularity": "attention"},
        "sampling_step_skip": {"granularity": "sampling_step"},
        "compose": {"granularity": "composite"},
        "kv_token_select": {"granularity": "token"},
    }

    ordered = budgeted_candidate_order(plans, list(range(len(plans))), specs, 6)

    assert ordered == (0, 2, 3, 4, 5, 6)
    assert candidate_family(plans[ordered[1]], specs) == "head"
    assert 1 not in ordered


def test_runtime_compute_reduction_contributes_to_search_score():
    metrics = {
        "valid": True,
        "quality_score": 0.8,
        "parameter_reduction_ratio": 0.0,
        "compute_metrics": {"reduction_ratio": 0.1},
        "runtime_execution": {
            "derived": {
                "head_gating_reduction_ratio": 0.25,
                "attention_window_reduction_ratio": 0.2,
            }
        },
    }

    assert efficiency_reduction_ratio(metrics) == 0.25
    assert math.isclose(measured_search_score(metrics), 0.825)


def test_invalid_measurements_cannot_win_selection_by_tie_break():
    invalid = {
        "valid": False,
        "search_score": None,
        "quality_score": None,
        "efficiency_reduction_ratio": 0.0,
        "rank_position": 0,
    }
    valid_later = {
        "valid": True,
        "search_score": 0.7,
        "quality_score": 0.65,
        "efficiency_reduction_ratio": 0.5,
        "rank_position": 4,
    }

    assert measured_candidate_key(invalid) is None
    assert measured_candidate_key(valid_later) == (0.7, 0.65, 0.5, -4.0)


def test_nan_metrics_cannot_rank():
    assert measured_candidate_key({
        "valid": True,
        "search_score": float("nan"),
        "quality_score": 0.9,
        "efficiency_reduction_ratio": 0.2,
        "rank_position": 0,
    }) is None


def test_small_budget_pays_down_family_coverage_without_forcing_winner():
    plans = [{'operator': name} for name in ['prune_blocks', 'prune_ffn_channels', 'head_gating', 'kv_token_select', 'sampling_step_skip', 'timestep_block_gate']]
    visited = set()
    counts = {}
    for _ in range(3):
        chosen = budgeted_candidate_order(plans, list(range(len(plans))), {}, 2, family_counts=counts)
        assert len(chosen) == 2
        visited.update(chosen)
        for index in chosen:
            family = candidate_family(plans[index], {})
            counts[family] = counts.get(family, 0) + 1
    assert visited == set(range(len(plans)))


def test_real_gpu_family_utility_guides_remaining_budget_after_probe_coverage():
    plans = [
        {"operator": "svoo_attention"},
        {"operator": "magcache"},
        {"operator": "prune_blocks"},
        {"operator": "svoo_attention", "args": {"config": {"x": 2}}},
        {"operator": "magcache", "args": {"config": {"x": 2}}},
    ]
    result = budgeted_candidate_order(
        plans,
        list(range(len(plans))),
        {},
        4,
        family_counts={"attention": 1, "cache": 1, "structure": 1},
        family_utilities={"cache": 0.01, "attention": 0.2},
    )
    assert result == (0, 3, 1, 4)


def test_pareto_front_uses_quality_efficiency_and_risk_and_excludes_invalid():
    rows = [
        {"valid": True, "quality_score": 0.95, "efficiency_reduction_ratio": 0.2, "risk_value": 0.05},
        {"valid": True, "quality_score": 0.90, "efficiency_reduction_ratio": 0.4, "risk_value": 0.08},
        {"valid": True, "quality_score": 0.90, "efficiency_reduction_ratio": 0.1, "risk_value": 0.10},
        {"valid": False, "quality_score": 1.0, "efficiency_reduction_ratio": 1.0, "risk_value": 0.0},
    ]

    assert pareto_front_indices(rows) == (0, 1)
    assert measured_pareto_key(rows[0]) is not None


def test_legacy_invalid_cache_is_not_reusable():
    from harness4h3.evolution.search_ranking import reusable_measured_trace
    assert not reusable_measured_trace({'selected_plan': {'operator': 'prune_blocks', 'args': {}}, 'measured_search': {'complete': True, 'candidate_results': [{'valid': False}]}})
