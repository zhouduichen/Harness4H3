from __future__ import annotations

import hashlib

import pytest

from harness4h3.evolution.tica_calibration import TICACalibrationManifest, TICAEvaluationSplits
from harness4h3.evolution.tica_prune import TICAPruner, TICARecipe, build_human_tica_source, recipe_plan, tica_operator_spec
from harness4h3.sandbox import AlgorithmSandbox


def _profile() -> dict:
    return {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 8},
        "layers": [{"block_id": "block.%d" % index} for index in range(4)],
        "tica": {"timesteps": ["0.9", "0.5", "0.1"]},
        "tica_block_sensitivity": [
            {"block_id": "block.0", "per_timestep": {"0.9": 0.01, "0.5": 0.40, "0.1": 0.50}},
            {"block_id": "block.1", "per_timestep": {"0.9": 0.30, "0.5": 0.30, "0.1": 0.30}},
            {"block_id": "block.2", "per_timestep": {"0.9": 0.30, "0.5": 0.30, "0.1": 0.30}},
            {"block_id": "block.3", "per_timestep": {"0.9": 0.40, "0.5": 0.40, "0.1": 0.40}},
        ],
        "tica_channel_sensitivity": [
            {
                "block_id": "block.%d" % index,
                "available": True,
                "channels": [{"channel": channel, "contribution": float(channel + 1 + index)} for channel in range(8)],
            }
            for index in range(4)
        ],
    }


def _manifest() -> TICACalibrationManifest:
    return TICACalibrationManifest.from_fake(hidden_size=8, sample_count=1)


def _measure(plan, full, stage):
    count = len(plan.get("block_ids", []))
    error = 0.04 if count <= 1 else 0.08 + 0.02 * count
    return {
        "valid": True,
        "quality_error": error,
        "quality_score": 1.0 - error,
        "parameter_reduction_ratio": 0.10 + 0.05 * count,
    }


def test_greedy_and_beam_have_different_state_trajectories():
    common = dict(include_baseline_suite=False, max_full_evals=20, max_fast_evals=30, max_interaction_evals=20, max_search_depth=3)
    greedy = TICAPruner().search(None, None, _manifest(), _profile(), TICARecipe(strategy="greedy", **common), evaluate_candidate=_measure)
    beam = TICAPruner().search(None, None, _manifest(), _profile(), TICARecipe(strategy="greedy_beam", beam_width=2, **common), evaluate_candidate=_measure)
    assert len(beam.search_states) > len(greedy.search_states)
    assert {item["depth"] for item in beam.search_states} >= {1, 2}


def test_selected_plan_carries_search_rule_and_is_on_pareto_front():
    recipe = TICARecipe(
        baseline_name="tica_block",
        include_baseline_suite=False,
        strategy="greedy",
        max_search_depth=2,
        risk_limit=0.8,
        max_full_evals=4,
        max_fast_evals=8,
    )
    result = TICAPruner().search(None, None, _manifest(), _profile(), recipe, evaluate_candidate=_measure)
    assert result.selected_candidate_id is not None
    selected = next(item for item in result.candidates if item.candidate_id == result.selected_candidate_id)
    assert selected.plan["strategy"] == "greedy"
    assert selected.plan["max_search_depth"] == 2
    assert result.selected_candidate_id in {item["candidate_id"] for item in result.pareto_front}


def test_agent_seed_plan_changes_the_search_trajectory():
    recipe = TICARecipe(
        baseline_name="tica_block",
        include_baseline_suite=False,
        risk_limit=0.8,
        max_full_evals=2,
        max_fast_evals=4,
    )
    first = TICAPruner().search(
        None,
        None,
        _manifest(),
        _profile(),
        recipe,
        evaluate_candidate=_measure,
        seed_plan={"block_ids": ["block.0"], "retained_indices": [0, 1, 2, 3, 4, 5]},
    )
    second = TICAPruner().search(
        None,
        None,
        _manifest(),
        _profile(),
        recipe,
        evaluate_candidate=_measure,
        seed_plan={"block_ids": ["block.3"], "retained_indices": [2, 3, 4, 5, 6, 7]},
    )
    assert first.candidates[0].plan["block_ids"] != second.candidates[0].plan["block_ids"]
    assert first.selected_plan_digest != second.selected_plan_digest


def test_risk_time_and_activation_rules_change_search_decisions():
    profile = _profile()
    loose = TICAPruner().search(
        None,
        None,
        _manifest(),
        profile,
        TICARecipe(include_baseline_suite=False, risk_limit=0.8, activation_weight=0.0, max_full_evals=4, max_fast_evals=8),
        evaluate_candidate=lambda plan, full, stage: {"quality_error": 0.10, "quality_score": 0.90, "parameter_reduction_ratio": 0.2},
    )
    strict = TICAPruner().search(
        None,
        None,
        _manifest(),
        profile,
        TICARecipe(include_baseline_suite=False, risk_limit=0.05, activation_weight=1.0, max_full_evals=4, max_fast_evals=8),
        evaluate_candidate=lambda plan, full, stage: {"quality_error": 0.10, "quality_score": 0.90, "parameter_reduction_ratio": 0.2},
    )
    assert loose.selected_plan_digest != strict.selected_plan_digest or loose.selected_candidate_id != strict.selected_candidate_id
    early = recipe_plan(profile, TICARecipe(time_weight_formula="early_log"))
    late = recipe_plan(profile, TICARecipe(time_weight_formula="late_log"))
    assert early["block_ids"] != late["block_ids"] or early["retained_indices"] != late["retained_indices"]


def test_interaction_full_measurements_share_the_full_budget_and_cache():
    calls = []

    def measure(plan, full, stage):
        calls.append((full, stage, tuple(plan.get("block_ids", [])), plan.get("mode")))
        return {"valid": True, "quality_error": 0.10, "quality_score": 0.90, "parameter_reduction_ratio": 0.2}

    result = TICAPruner().search(
        None,
        None,
        _manifest(),
        _profile(),
        TICARecipe(max_full_evals=3, max_interaction_evals=2),
        evaluate_candidate=measure,
    )
    assert result.budget["full_evals"] <= 3
    assert sum(1 for full, _stage, _blocks, _mode in calls if full) <= 3
    assert any(item["stage"] == "interaction" for item in result.budget_audit)
    assert result.budget["cache_hits"] >= 1


def test_search_refuses_test_split_and_missing_ffn_evidence():
    profile = _profile()
    profile["evaluation_split"] = "test"
    with pytest.raises(ValueError, match="Test split"):
        TICAPruner().search(None, None, _manifest(), profile, TICARecipe(), evaluate_candidate=_measure)

    missing = _profile()
    missing["tica_channel_sensitivity"] = [{"block_id": "block.0", "available": False, "channels": []}]
    result = TICAPruner().search(
        None,
        None,
        _manifest(),
        missing,
        TICARecipe(include_baseline_suite=False, max_full_evals=4),
        evaluate_candidate=_measure,
    )
    assert result.selected_candidate_id is None
    assert all(item["status"] in {"baseline", "risk_unknown", "risk_rejected", "screened"} for item in result.search_states)

    missing_blocks = _profile()
    missing_blocks["tica_block_sensitivity"] = [
        {**item, "available": False} for item in missing_blocks["tica_block_sensitivity"]
    ]
    result = TICAPruner().search(
        None,
        None,
        _manifest(),
        missing_blocks,
        TICARecipe(include_baseline_suite=False, max_full_evals=4),
        evaluate_candidate=_measure,
    )
    assert result.selected_candidate_id is None


def test_evaluation_splits_reject_reused_cached_source_hash():
    digest = "a" * 64
    def manifest(sample_id, prefix):
        return TICACalibrationManifest.from_dict(
            {
                "provider_name": "cached_real_wan",
                "specs": [
                    {
                        "sample_id": sample_id,
                        "prompt": prefix,
                        "seed": 7,
                        "latent_shape": [1, 1, 1, 1, 1],
                        "timesteps": [0.5],
                        "latent_path": "/tmp/%s-latent.pt" % prefix,
                        "encoder_hidden_states_path": "/tmp/%s-hidden.pt" % prefix,
                        "latent_sha256": digest,
                        "encoder_hidden_states_sha256": "b" * 64,
                    }
                ],
            }
        )

    calibration = manifest("calibration-0", "shared")
    validation = manifest("validation-0", "shared")
    test = TICACalibrationManifest.from_fake(hidden_size=1, sample_prefix="test")
    with pytest.raises(ValueError, match="source is reused"):
        TICAEvaluationSplits(calibration=calibration, validation=validation, test=test)


def test_evaluation_splits_reject_reused_sample_ids():
    calibration = TICACalibrationManifest.from_fake(hidden_size=8, sample_count=1)
    validation = TICACalibrationManifest.from_fake(hidden_size=8, sample_count=1)
    with pytest.raises(ValueError, match="split leakage"):
        TICAEvaluationSplits(calibration=calibration, validation=validation, test=calibration)


def test_real_manifest_verifies_cached_tensor_source_hashes(tmp_path):
    latent = tmp_path / "latent.bin"
    hidden = tmp_path / "hidden.bin"
    latent.write_bytes(b"latent-v1")
    hidden.write_bytes(b"hidden-v1")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = TICACalibrationManifest.from_dict(
        {
            "provider_name": "cached_real_wan",
            "specs": [
                {
                    "sample_id": "real-0",
                    "prompt": "a real sample",
                    "seed": 7,
                    "latent_shape": [2, 8],
                    "timesteps": [0.9, 0.5, 0.1],
                    "latent_path": str(latent),
                    "encoder_hidden_states_path": str(hidden),
                    "latent_sha256": digest(latent),
                    "encoder_hidden_states_sha256": digest(hidden),
                }
            ],
        }
    )
    manifest.verify_cached_sources()
    latent.write_bytes(b"latent-v2")
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest.verify_cached_sources()


def test_human_tica_source_matches_full_rule_schema(tmp_path):
    sandbox = AlgorithmSandbox(
        tmp_path / "sandbox",
        operator_specs=[tica_operator_spec()],
    )
    result = sandbox.run(build_human_tica_source("human-tica-v2"), _profile(), [])
    assert result.status == "success"
    assert result.plan.args["strategy"] == "greedy_beam"
    assert result.plan.args["max_search_depth"] == 3
