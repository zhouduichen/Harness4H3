from __future__ import annotations

import pytest

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.controller.schemas import ExperimentPlan
from harness4h3.evaluator.composite import CompositeEvaluator
from harness4h3.evaluator.constraints import ConstraintEvaluator
from harness4h3.evaluator.hardware import FakeHardwareEvaluator
from harness4h3.evaluator.quality import FakeQualityEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import ExecutionContext, OperatorValidationError
from harness4h3.operators.fake import FakeOperatorBackend, build_fake_registry
from harness4h3.target.profile import TargetProfile


def target():
    return TargetProfile("mobile", "mobile", "fake", max_peak_memory_gb=6, max_latency_s=30, max_quality_drop=0.05)


def root():
    state = ModelState.fake_baseline()
    return ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")


def test_fake_quantize_and_step_distill_produce_new_state_without_mutating_parent(tmp_path):
    backend = FakeOperatorBackend()
    registry = build_fake_registry(backend)
    parent = root()
    quantized = registry.execute("quantize", parent, {"bits": 4}, target(), ExecutionContext(tmp_path, "M0001"))
    assert quantized.ok
    assert quantized.output_state.measured_metrics["peak_memory_gb"] == pytest.approx(7.8)
    assert parent.state.measured_metrics["peak_memory_gb"] == 12.0
    distilled_parent = ModelCandidate("M0001", "M0000", 1, "fake://M0001", quantized.output_state, "exp_0001", "candidate")
    distilled = registry.execute("step_distill", distilled_parent, {"target_steps": 8}, target(), ExecutionContext(tmp_path, "M0002"))
    assert distilled.output_state.measured_metrics["latency_s"] == pytest.approx(26.4)
    assert distilled.output_state.measured_metrics["peak_memory_gb"] == pytest.approx(5.46)


def test_operator_registry_rejects_unknown_operator_and_arguments():
    registry = build_fake_registry(FakeOperatorBackend())
    with pytest.raises(OperatorValidationError, match="not registered"):
        registry.validate("arbitrary_shell", root().state, {}, target())
    with pytest.raises(OperatorValidationError, match="unsupported argument"):
        registry.validate("quantize", root().state, {"command": "rm anything"}, target())


def test_fake_operator_backend_returns_stable_oom_failure(tmp_path):
    backend = FakeOperatorBackend({"quantize": ["training_oom"]})
    result = build_fake_registry(backend).execute(
        "quantize", root(), {"bits": 4}, target(), ExecutionContext(tmp_path, "M0001")
    )
    assert result.ok is False
    assert result.failure_type == "training_oom"
    assert result.output_state is None


def test_composite_evaluator_enforces_quality_and_hardware_constraints():
    evaluator = CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator())
    baseline = root().state
    result = evaluator.evaluate(baseline, target(), baseline_quality=0.90)
    assert result.feasible is False
    assert "max_peak_memory_gb" in result.violations
    assert "max_latency_s" in result.violations
    regressed = baseline.derive("M0001", measured_metrics={**baseline.measured_metrics, "quality_score": 0.70})
    result = evaluator.evaluate(regressed, target(), baseline_quality=0.90)
    assert result.critical_regression is True
