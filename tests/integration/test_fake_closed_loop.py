from __future__ import annotations

from dataclasses import replace

import pytest

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.model_store import ModelStore
from harness4h3.archive.pareto import ParetoArchive
from harness4h3.controller.loop import OptimizationLoop
from harness4h3.controller.provider import RuleBasedMockController
from harness4h3.controller.schemas import BudgetState, ExperimentPlan, OperatorResult
from harness4h3.evaluator.composite import CompositeEvaluator
from harness4h3.evaluator.constraints import ConstraintEvaluator
from harness4h3.evaluator.hardware import FakeHardwareEvaluator
from harness4h3.evaluator.quality import FakeQualityEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.memory.experiment_store import ExperimentStore
from harness4h3.operators.fake import FakeOperatorBackend, build_fake_registry
from harness4h3.target.profile import TargetProfile


def mobile_target(**changes):
    target = TargetProfile(
        "mobile_h3_v1",
        "mobile",
        "fake_device",
        max_peak_memory_gb=6,
        max_latency_s=30,
        max_quality_drop=0.05,
    )
    return replace(target, **changes)


def baseline():
    state = ModelState.fake_baseline()
    return ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")


def make_loop(tmp_path, controller=None, backend=None):
    root = tmp_path / "session"
    return OptimizationLoop(
        controller=controller or RuleBasedMockController(),
        operators=build_fake_registry(backend or FakeOperatorBackend()),
        evaluator=CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator()),
        models=ModelStore(root / "models"),
        pareto=ParetoArchive(root / "pareto"),
        experiments=ExperimentStore(root / "experiments.jsonl"),
        run_root=root / "runs",
        checkpoint_path=root / "session.json",
        max_repeated_failures=2,
    )


def budget(**changes):
    return replace(BudgetState(5, 4, max_controller_calls=5, max_gpu_hours=1), **changes)


class StaticController:
    provider_name = "test"
    model_name = "static"

    def __init__(self, payload):
        self.payload = payload

    def plan(self, context):
        return self.payload(context) if callable(self.payload) else self.payload


def valid_plan(context, operator="quantize", args=None):
    return ExperimentPlan(
        "exp_%04d" % (context.budget_state.used_iterations + 1),
        context.current_model_state.model_id,
        "constraint violation",
        "reduce resource use",
        "the registered operator should improve the blocking metric",
        operator,
        args or {"bits": 4},
        {"latency_s": "decrease"},
        ["quality regression"],
        {"wall_time_s": 0.2},
        {"max_quality_drop": 0.05},
        {"critical_regression": True},
        "bounded offline experiment",
    )


def test_success_loop_quantizes_then_step_distills_and_reaches_target(tmp_path):
    loop = make_loop(tmp_path)
    result = loop.run("session-1", mobile_target(), budget(), baseline())
    assert result.status == "target_satisfied"
    assert result.current_model_id == "M0002"
    assert [item.id for item in loop.models.lineage()] == ["M0000", "M0001", "M0002"]
    assert [item.plan["operator"] for item in loop.experiments.read()] == ["quantize", "step_distill"]
    assert loop.models.active_id == "M0002"
    assert "UPDATE_PARETO" in result.transitions


def test_illegal_operator_never_executes(tmp_path):
    controller = StaticController(lambda context: valid_plan(context, "arbitrary_shell", {}))
    loop = make_loop(tmp_path, controller)
    result = loop.run("illegal", mobile_target(), budget(), baseline())
    assert result.status == "no_valid_plan"
    assert len(loop.models.lineage()) == 1
    assert all(item.failure_type == "operator_invalid" for item in loop.experiments.read())


def test_invalid_llm_plan_never_executes(tmp_path):
    loop = make_loop(tmp_path, StaticController({"operator": "quantize"}))
    result = loop.run("invalid", mobile_target(), budget(), baseline())
    assert result.status == "no_valid_plan"
    assert len(loop.models.lineage()) == 1
    assert all(item.failure_type == "schema_invalid" for item in loop.experiments.read())


def test_operator_failure_is_recorded_then_recovered(tmp_path):
    loop = make_loop(tmp_path, backend=FakeOperatorBackend({"quantize": ["operator_failure"]}))
    result = loop.run("operator-failure", mobile_target(), budget(), baseline())
    records = list(loop.experiments.read())
    assert result.status == "target_satisfied"
    assert records[0].failure_type == "operator_failure"
    assert records[0].child_model_id is None


def test_training_oom_is_stable_and_stops_after_repetition(tmp_path):
    failures = FakeOperatorBackend({"quantize": ["training_oom", "training_oom", "training_oom"]})
    loop = make_loop(tmp_path, backend=failures)
    result = loop.run("oom", mobile_target(), budget(), baseline())
    assert result.status == "no_valid_plan"
    assert [item.failure_type for item in loop.experiments.read()] == ["training_oom", "training_oom"]


def test_quality_critical_regression_archives_child_but_keeps_parent_active(tmp_path):
    loop = make_loop(tmp_path)
    strict = mobile_target(max_quality_drop=0.001)
    result = loop.run("quality", strict, budget(), baseline())
    assert result.status == "critical_failure"
    assert loop.models.active_id == "M0000"
    assert [item.id for item in loop.models.lineage()] == ["M0000", "M0001"]
    assert list(loop.experiments.read())[-1].decision["keep"] is False


def test_budget_stop_is_bounded_without_while_true(tmp_path):
    loop = make_loop(tmp_path)
    result = loop.run("budget", mobile_target(), budget(max_iterations=1), baseline())
    assert result.status == "max_iterations"
    assert result.current_model_id == "M0001"
    assert len(list(loop.experiments.read())) == 1


def test_crash_recovery_resumes_from_last_atomic_checkpoint(tmp_path):
    loop = make_loop(tmp_path)

    def crash_after_first(state):
        if state.budget.used_iterations == 1:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        loop.run("recover", mobile_target(), budget(), baseline(), checkpoint_hook=crash_after_first)
    resumed = make_loop(tmp_path).run("recover", mobile_target(), budget(), baseline())
    assert resumed.status == "target_satisfied"
    assert resumed.current_model_id == "M0002"
    assert len(list(loop.experiments.read())) == 2


def test_initial_target_reached_stops_without_controller_call(tmp_path):
    state = ModelState.fake_baseline()
    state = replace(
        state,
        measured_metrics={**state.measured_metrics, "latency_s": 20.0, "peak_memory_gb": 5.0},
    )
    candidate = ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")
    loop = make_loop(tmp_path)
    result = loop.run("already-done", mobile_target(), budget(), candidate)
    assert result.status == "target_satisfied"
    assert result.budget.used_controller_calls == 0
    assert list(loop.experiments.read()) == []
