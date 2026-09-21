from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from research.experiments.a0_model_evolution import A0RuleBasedController
from research.experiments.a1_real_evolution import A1BootstrapController, TieredRealBenchmarkEvaluator, _hardware_metrics
from harness4h3.benchmark.h3 import BenchmarkSummary, BenchmarkTaskResult
from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.controller.context import ControllerContext
from harness4h3.controller.schemas import BudgetState, EvaluationResult, ExperimentPlan, HardwareMetrics
from harness4h3.h3.state import ModelState
from harness4h3.operators.model_evolution import build_model_evolution_registry
from harness4h3.target.profile import TargetProfile


def target():
    return TargetProfile("rtx", "gpu", "rtx5080", max_peak_memory_gb=16, max_latency_s=300, max_quality_drop=0.05)


def summary(model_id: str, tasks: int) -> BenchmarkSummary:
    run = BenchmarkTaskResult(
        task_id="task",
        status="success",
        prompt_id="p1",
        artifacts=("result.mp4",),
        wall_time_s=1.0,
        quality_score=0.99,
        quality_metrics={"decodable": 1.0},
        operator_execution_success=True,
        artifact_generation_success=True,
        semantic_generation_valid=True,
    )
    return BenchmarkSummary(
        model_id=model_id,
        task_count=tasks,
        quality_score=0.99,
        quality_metrics={"real": True},
        hardware=HardwareMetrics(latency_s=1.0, peak_memory_gb=4.0, model_size_gb=2.0),
        feasible=True,
        violations=(),
        runs=tuple(run for _ in range(tasks)),
        hard_gates={"target_feasible": True},
        accepted=True,
    )


class StubBenchmark:
    def __init__(self):
        self.calls = []

    def run(self, state, tasks, **kwargs):
        self.calls.append((state.model_id, len(tasks), kwargs["target"].id))
        return replace(
            summary(state.model_id, len(tasks)),
            system_id=getattr(kwargs.get("system"), "id", None),
            device_id=kwargs.get("device_id"),
            task_split=kwargs.get("task_split"),
            benchmark_recipe=dict(kwargs.get("benchmark_recipe") or {"recipe": "fixed"}),
            evaluator_version="test-evaluator-v1",
        )


def test_real_evaluator_progresses_dev_to_heldout_per_child():
    benchmark = StubBenchmark()
    evaluator = TieredRealBenchmarkEvaluator(
        benchmark,
        {1: [object()], 2: [object(), object()], 3: [object(), object(), object()]},
        target(),
        0.991,
        HardwareMetrics(latency_s=90.0, peak_memory_gb=16.3, model_size_gb=12.5),
    )
    state = ModelState.fake_baseline("M0001")
    first = evaluator.evaluate(state, target(), 0.991)
    second = evaluator.evaluate(state, target(), 0.991)
    third = evaluator.evaluate(state, target(), 0.991)
    assert first.feasible and second.feasible and third.feasible
    assert [call[1] for call in benchmark.calls] == [1, 2, 3]
    assert [first.quality_metrics["fidelity_tier"], second.quality_metrics["fidelity_tier"], third.quality_metrics["fidelity_tier"]] == [1, 2, 3]
    assert all(item.quality_metrics["real_benchmark"] for item in (first, second, third))


def test_real_evaluator_emits_pair_keyed_canonical_record():
    benchmark = StubBenchmark()
    evaluator = TieredRealBenchmarkEvaluator(
        benchmark,
        {1: [object()]},
        target(),
        0.991,
        HardwareMetrics(latency_s=90.0, peak_memory_gb=16.3, model_size_gb=12.5),
    )
    state = ModelState.fake_baseline("M0000")
    model = ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")
    system = SystemCandidate.from_model_candidate("S0000", model, status="baseline")
    record = evaluator.evaluate(
        state,
        target(),
        0.991,
        system=system,
        device_id="L40-0",
        task_split="dev",
        benchmark_recipe={"name": "a1-t0", "steps": 32},
    )
    assert record.model_id == "M0000"
    assert record.system_id == "S0000"
    assert record.device_id == "L40-0"
    assert record.task_split == "dev"
    assert record.provenance["offline_simulation"] is False
    assert record.provenance["benchmark_recipe"] == {"name": "a1-t0", "steps": 32}
    assert record.quality_metrics["benchmark_summary"]["evaluator_version"] == "test-evaluator-v1"


def test_a1_bootstrap_forces_full_fidelity_without_changing_plan_schema():
    state = ModelState.fake_baseline("M0000")
    context = ControllerContext(
        target(),
        state,
        BudgetState(2, 2, max_gpu_hours=8, max_controller_calls=2),
        tuple(build_model_evolution_registry().visible()),
    )
    plan = A1BootstrapController(A0RuleBasedController()).plan(context)
    assert plan.operator == "create_student"
    assert plan.required_budget["tier"] == 3
    assert "tier" in plan.required_budget
    assert set(plan.to_dict()) == set(ExperimentPlan.__dataclass_fields__)
    assert plan.parent_model_id == "M0000"


def test_baseline_hardware_requires_all_real_values():
    metrics = _hardware_metrics({"model_size_gb": 1.0, "latency_s": 2.0, "peak_memory_gb": 3.0})
    assert metrics.model_size_gb == 1.0


def test_a1_second_plan_uses_first_child_and_prior_evidence(tmp_path):
    class FeasibleEvaluator:
        def evaluate(self, state, target, baseline_quality):
            return EvaluationResult(
                quality_score=0.9,
                quality_metrics={"real_benchmark": True, "offline_simulation": False},
                hardware=HardwareMetrics(latency_s=30.0, peak_memory_gb=4.0, model_size_gb=2.0),
                feasible=True,
            )

    from research.experiments.a0_model_evolution import A0Budget, run_campaign

    result = run_campaign(
        target(),
        A1BootstrapController(A0RuleBasedController()),
        build_model_evolution_registry(),
        FeasibleEvaluator(),
        tmp_path / "campaign",
        budget=A0Budget(max_gpu_hours=8.0, max_experiments=2, max_failed_experiments=2),
        offline_simulation=False,
    )
    assert [item["operator"] for item in result.report["full_autonomous_experiment_sequence"]] == ["create_student", "distill"]
    assert [item["model_parent_id"] for item in result.report["full_autonomous_experiment_sequence"]] == ["M0000", "M0001"]
    assert result.report["full_autonomous_experiment_sequence"][0]["fidelity_tier"] == 3
    assert result.report["full_autonomous_experiment_sequence"][1]["fidelity_tier"] == 3


def test_real_benchmark_failure_is_not_feasible():
    class FailedBenchmark:
        def run(self, state, tasks, **kwargs):
            failed = summary(state.model_id, 0)
            return BenchmarkSummary(
                model_id=failed.model_id,
                task_count=0,
                quality_score=None,
                quality_metrics={},
                hardware=failed.hardware,
                feasible=False,
                violations=("comfyui_unreachable",),
                runs=(),
                hard_gates={"target_feasible": False},
                accepted=False,
            )

    evaluator = TieredRealBenchmarkEvaluator(
        FailedBenchmark(),
        {1: [object()]},
        target(),
        0.991,
        HardwareMetrics(latency_s=90.0, peak_memory_gb=16.3, model_size_gb=12.5),
    )
    result = evaluator.evaluate(ModelState.fake_baseline("M0001"), target(), 0.991)
    assert result.feasible is False
    assert result.critical_regression is True
    assert "comfyui_unreachable" in result.violations
