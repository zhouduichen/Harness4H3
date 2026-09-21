from __future__ import annotations

import pytest

from harness4h3.controller.schemas import EvaluationRecord, HardwareMetrics
from harness4h3.evaluator.composite import CompositeEvaluator
from harness4h3.evaluator.evaluator import legacy_evaluator_result, validate_result
from harness4h3.evaluator.constraints import ConstraintEvaluator
from harness4h3.evaluator.hardware import FakeHardwareEvaluator
from harness4h3.evaluator.quality import FakeQualityEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.target.profile import ObjectiveSpec, TargetProfile


def test_target_parses_explicit_objectives_and_scores_signed_metrics():
    target = TargetProfile.from_dict(
        {
            "id": "gpu",
            "hardware": {"type": "gpu", "name": "L40"},
            "objectives": [
                {"name": "quality_score", "direction": "maximize", "weight": 2},
                {"name": "latency_s", "direction": "minimize", "weight": 1},
            ],
        }
    )
    assert target.objectives[0] == ObjectiveSpec("quality_score", "maximize", 2)
    assert target.objective_score({"quality_score": 0.9, "latency_s": 20}) == pytest.approx(-18.2)


def test_target_translates_legacy_priority_when_objectives_are_omitted():
    target = TargetProfile("gpu", "gpu", "L40", priority=("feasibility", "quality", "memory"))
    assert [(item.name, item.direction) for item in target.objectives] == [
        ("quality_score", "maximize"),
        ("peak_memory_gb", "minimize"),
    ]


def test_legacy_evaluator_factory_returns_canonical_record():
    record = legacy_evaluator_result(0.8, {"quality": 0.8})
    assert isinstance(record, EvaluationRecord)
    assert record.score == pytest.approx(0.8)
    assert record.metrics["quality"] == pytest.approx(0.8)


def test_legacy_evaluator_payload_is_read_as_canonical_record():
    record = validate_result({"score": 0.8, "metrics": {"quality": 0.8}})
    assert record == legacy_evaluator_result(0.8, {"quality": 0.8})


def test_composite_evaluator_overlays_system_runtime_without_changing_model():
    state = ModelState.fake_baseline()
    model = ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")
    system = SystemCandidate.from_model_candidate(
        "S0000", model, runtime_state={"runtime_recipe": [{"kind": "vae_tiling"}]}, status="baseline"
    )
    result = CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator()).evaluate(
        state, TargetProfile("gpu", "gpu", "L40"), 0.90, system=system, device_id="L40-0", task_split="dev"
    )
    assert result.model_id == "M0000"
    assert result.system_id == "S0000"
    assert result.device_id == "L40-0"
    assert result.task_split == "dev"
    assert result.hardware == HardwareMetrics(
        latency_s=60.0, peak_memory_gb=12.0, model_size_gb=7.0, energy_j=120.0, throughput=1 / 60.0
    )


def test_composite_evaluator_rejects_mismatched_pair():
    state = ModelState.fake_baseline("M0000")
    model = ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")
    other_state = ModelState.fake_baseline("M0000").derive("M0001")
    other = ModelCandidate("M0001", "M0000", 1, other_state.checkpoint_path, other_state, "exp", "candidate")
    system = SystemCandidate.from_model_candidate("S0000", other)
    evaluator = CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator())
    with pytest.raises(ValueError, match="references"):
        evaluator.evaluate(state, TargetProfile("gpu", "gpu", "L40"), 0.90, system=system)
