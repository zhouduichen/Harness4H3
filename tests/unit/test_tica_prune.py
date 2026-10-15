from __future__ import annotations

import pytest

from harness4h3.evolution.tica_prune import (
    TICAPruner,
    TICARecipe,
    build_human_tica_source,
    collect_tica_profile,
    tica_score,
)
from harness4h3.sandbox import AlgorithmSandbox


def fake_tica_profile(block_bias: float = 0.1):
    return {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 8},
        "layers": [{"block_id": "block.%d" % index} for index in range(4)],
        "tica_block_sensitivity": [
            {
                "block_id": "block.%d" % index,
                "per_timestep": {
                    "high_noise": (block_bias if index == 0 else 0.2 + 0.1 * index),
                    "middle": (block_bias if index == 0 else 0.3 + 0.1 * index),
                    "low_noise": (block_bias if index == 0 else 0.4 + 0.1 * index),
                },
            }
            for index in range(4)
        ],
        "tica_channel_sensitivity": [
            {
                "block_id": "block.%d" % index,
                "channels": [
                    {"channel": channel, "contribution": float(channel + index)}
                    for channel in range(8)
                ],
            }
            for index in range(4)
        ],
    }


def test_time_weighted_risk_changes_when_profile_formula_changes():
    profile = fake_tica_profile()
    uniform = tica_score(profile, time_weight_formula="uniform")
    early = tica_score(profile, time_weight_formula="early_log")
    assert uniform.block_risk != early.block_risk
    assert uniform.time_weights != early.time_weights


def test_global_ffn_recipe_rejects_layer_specific_widths():
    with pytest.raises(ValueError, match="global FFN"):
        TICARecipe.from_mapping({"per_layer_retained_indices": {"block.0": [0]}})


def test_human_source_is_profile_dependent_and_auditable(tmp_path):
    source = build_human_tica_source("A0001")
    sandbox = AlgorithmSandbox(
        tmp_path / "sandbox",
        operator_specs=[
            {
                "name": "tica_prune",
                "argument_schema": {
                    "block_ids": "list[str]",
                    "retained_indices": "list[int]",
                    "mode": "string",
                    "time_weight_formula": "string",
                    "risk_limit": "number",
                    "fast_risk_limit": "number",
                    "activation_weight": "number",
                    "interaction_weight": "number",
                    "beam_width": "number",
                    "max_fast_evals": "number",
                    "max_full_evals": "number",
                    "max_interaction_evals": "number",
                    "strategy": "string",
                    "max_search_depth": "number",
                },
            }
        ],
    )
    low = sandbox.run(source, fake_tica_profile(block_bias=0.1), [])
    high = sandbox.run(source, fake_tica_profile(block_bias=0.9), [])
    assert low.status == high.status == "success"
    assert low.plan.to_dict() != high.plan.to_dict()
    assert low.plan.args["time_weight_formula"] == "early_log"


def test_recipe_budget_is_explicit_and_bounded():
    recipe = TICARecipe.from_mapping(
        {
            "baseline_name": "tica_block_ffn",
            "beam_width": 2,
            "max_fast_evals": 3,
            "max_full_evals": 2,
            "max_interaction_evals": 1,
        }
    )
    assert recipe.to_dict()["max_full_evals"] == 2
    assert recipe.budget()["max_interaction_evals"] == 1


def test_interaction_delta_is_derived_from_measured_errors():
    result = TICAPruner().measure_interaction(
        [{"valid": True, "quality_error": 0.10, "risk_metric": "normalized_mse", "risk_value": 0.10, "risk_scale": "normalized_mse", "risk_is_approximate": False}, {"valid": True, "quality_error": 0.20, "risk_metric": "normalized_mse", "risk_value": 0.20, "risk_scale": "normalized_mse", "risk_is_approximate": False}],
        {"valid": True, "quality_error": 0.45, "risk_metric": "normalized_mse", "risk_value": 0.45, "risk_scale": "normalized_mse", "risk_is_approximate": False},
    )
    assert result["available"] is True
    assert result["interaction_delta"] == pytest.approx(0.15)


def test_search_obeys_full_and_interaction_budgets():
    from harness4h3.evolution.tica_calibration import TICACalibrationManifest

    calls = []

    def measure(plan, full, stage):
        calls.append((full, stage, dict(plan)))
        return {"valid": True, "quality_error": 0.1, "risk_metric": "normalized_mse" if full else "local_relative_output_error", "risk_value": 0.1, "risk_scale": "normalized_mse" if full else "profile_relative_error", "risk_is_approximate": not full, "parameter_reduction_ratio": 0.2}

    result = TICAPruner().search(
        adapter=None,
        parent=None,
        manifest=TICACalibrationManifest.from_fake(hidden_size=8),
        profile=fake_tica_profile(),
        recipe=TICARecipe(max_full_evals=4, max_interaction_evals=1),
        evaluate_candidate=measure,
    )
    assert result.budget["full_evals"] == 4
    assert result.budget["interaction_evals"] == 1
    assert len(result.candidates) == 4


def test_collect_tica_profile_uses_real_fake_execution_for_all_timesteps():
    from harness4h3.adapters.wan import FakeWanDiTAdapter
    from harness4h3.evolution.tica_calibration import TICACalibrationManifest

    adapter = FakeWanDiTAdapter(block_count=3, hidden_size=8, ffn_width=8)
    model = adapter.load_model("fake://baseline")
    manifest = TICACalibrationManifest.from_fake(hidden_size=8, sample_count=2, timesteps=(0.9, 0.5, 0.1))
    profile = collect_tica_profile(adapter, model, manifest)
    assert profile["tica"]["calibration_digest"] == manifest.digest()
    assert len(profile["tica_block_sensitivity"]) == 3
    assert set(profile["tica_block_sensitivity"][0]["per_timestep"]) == {"0.9", "0.5", "0.1"}
    assert all(item["available"] is True for item in profile["tica_channel_sensitivity"])
