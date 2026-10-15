from __future__ import annotations

from harness4h3.evolution.tica_calibration import TICACalibrationManifest
from harness4h3.evolution.tica_prune import TICAPruner, TICARecipe
from harness4h3.sandbox import AlgorithmSandbox


def _profile():
    return {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 8},
        "layers": [{"block_id": "block.%d" % index} for index in range(4)],
        "evaluation_split": "calibration",
    }


def _manifest():
    return TICACalibrationManifest.from_fake(hidden_size=8, sample_count=1)


def _measure(plan, full, stage):
    args = plan.get("args", plan)
    count = len(args.get("block_ids", []))
    error = 0.04 + 0.01 * count
    return {
        "valid": True,
        "quality_error": error,
        "risk_metric": "normalized_mse" if full else "local_relative_output_error",
        "risk_value": error,
        "risk_scale": "normalized_mse" if full else "profile_relative_error",
        "risk_is_approximate": not full,
        "quality_score": 1.0 - error,
        "parameter_reduction_ratio": 0.10 + 0.05 * count,
    }


def _run(source, root):
    return TICAPruner().search(
        None,
        None,
        _manifest(),
        _profile(),
        TICARecipe(
            baseline_name="tica_block",
            include_baseline_suite=False,
            max_search_depth=1,
            max_fast_evals=8,
            max_full_evals=4,
        ),
        evaluate_candidate=_measure,
        algorithm_source=source,
        algorithm_history=[{"experience_id": "R0001", "failure_reason": "quality_regression"}],
        search_space={
            "version_id": "S0001",
            "operators": [
                {"name": "prune_blocks", "argument_schema": {"block_ids": "list[str]"}},
                {"name": "prune_ffn_channels", "argument_schema": {"block_ids": "list[str]", "retained_indices": "list[int]"}},
            ],
        },
        sandbox=AlgorithmSandbox(root),
    )


def test_two_algorithm_programs_change_candidate_membership_and_order(tmp_path):
    source_a = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}
def search(profile, history, state, search_space):
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
    ], 'rank': [0, 1], 'stop': True}
"""
    source_b = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}
def search(profile, history, state, search_space):
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
    ], 'rank': [1, 0], 'stop': True}
"""
    first = _run(source_a, tmp_path / "a")
    second = _run(source_b, tmp_path / "b")
    assert first.algorithm_trace[0]["source_sha256"] != second.algorithm_trace[0]["source_sha256"]
    assert first.algorithm_trace[0]["plan_digests"] != second.algorithm_trace[0]["plan_digests"]
    assert first.algorithm_trace[0]["rank"] == [0, 1]
    assert second.algorithm_trace[0]["rank"] == [1, 0]
    assert first.candidates[0].plan["args"]["block_ids"] == ["block.0"]
    assert second.candidates[0].plan["args"]["block_ids"] == ["block.2"]


def test_search_program_receives_current_search_space_and_history(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}
def search(profile, history, state, search_space):
    assert history[0]['failure_reason'] == 'quality_regression'
    assert search_space['version_id'] == 'S0001'
    assert len(search_space['operators']) == 2
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.3']}},
    ], 'rank': [0], 'stop': True}
"""
    result = _run(source, tmp_path / "space")
    assert result.algorithm_trace[0]["search_space_operator_names"] == ["prune_blocks", "prune_ffn_channels"]
    assert result.candidates[0].plan["args"]["block_ids"] == ["block.3"]

