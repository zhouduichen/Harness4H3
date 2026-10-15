from __future__ import annotations

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.algorithm_evaluation import AlgorithmModelEvaluator
from harness4h3.evolution.tica_calibration import TICACalibrationManifest
from harness4h3.operators.wan import build_wan_compression_registry
from harness4h3.sandbox.runner import CompressionPlan, SandboxResult


def test_tica_evaluator_measures_each_fixed_timestep_and_keeps_hardware_unavailable(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    manifest = TICACalibrationManifest.from_fake(hidden_size=8, sample_count=2, timesteps=(0.9, 0.5, 0.1))
    evaluator = AlgorithmModelEvaluator(
        adapter,
        build_wan_compression_registry(include_tica=True),
        parent,
        {"model": parent.structure.to_dict()},
        tmp_path / "measure",
        calibration_manifest=manifest,
    )
    plan = CompressionPlan(
        "tica_prune",
        {
            "block_ids": ["block.0"],
            "retained_indices": list(range(8)),
            "mode": "block_plus_ffn",
            "time_weight_formula": "early_log",
            "risk_limit": 0.25,
            "activation_weight": 0.5,
            "interaction_weight": 0.5,
            "beam_width": 2,
            "max_fast_evals": 8,
            "max_full_evals": 4,
            "max_interaction_evals": 2,
        },
    )
    result = evaluator.evaluate(SandboxResult("success", "a" * 64, plan))
    assert result["valid"] is True
    assert result["calibration_digest"] == manifest.digest()
    assert len(result["output_metrics"]["timestep_records"]) == 6
    assert result["hardware_metrics"]["available"] is False
    assert result["search_budget"]["max_full_evals"] == 4
