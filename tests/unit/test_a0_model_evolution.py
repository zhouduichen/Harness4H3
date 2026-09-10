from __future__ import annotations

from dataclasses import replace

from experiments.a0_model_evolution import (
    A0Budget,
    A0RuleBasedController,
    build_a0_report,
    run_campaign,
)
from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.controller.schemas import CostEstimate, ExperimentPlan, OperatorResult
from harness4h3.evaluator.composite import CompositeEvaluator
from harness4h3.evaluator.constraints import ConstraintEvaluator
from harness4h3.evaluator.hardware import FakeHardwareEvaluator
from harness4h3.evaluator.quality import FakeQualityEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.operators.fake import FakeOperatorBackend
from harness4h3.operators.model_evolution import ModelEvolutionBackend, build_model_evolution_registry
from harness4h3.target.profile import TargetProfile


def target(**changes):
    return replace(
        TargetProfile("mobile", "mobile", "fake", max_model_size_gb=4, max_peak_memory_gb=6, max_latency_s=30, max_quality_drop=0.05),
        **changes,
    )


def evaluator():
    return CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator())


def test_a0_creates_model_lineage_and_records_fidelity(tmp_path):
    result = run_campaign(
        target(),
        A0RuleBasedController(),
        build_model_evolution_registry(),
        evaluator(),
        tmp_path,
        budget=A0Budget(max_gpu_hours=3, max_experiments=4, max_failed_experiments=2),
    )
    assert result.status == "completed"
    assert result.report["offline_simulation"] is True
    assert result.report["total_experiments"] == 4
    assert len(result.report["model_lineage"]) >= 2
    assert all(item["harness_version"] == "Harness4H3-v1.0" for item in result.report["full_autonomous_experiment_sequence"])
    assert any(item["fidelity_tier"] in {1, 2, 3} for item in result.report["full_autonomous_experiment_sequence"])


def test_rejected_candidate_never_becomes_parent(tmp_path):
    strict = target(max_quality_drop=0.0001)
    result = run_campaign(
        strict,
        A0RuleBasedController(),
        build_model_evolution_registry(),
        evaluator(),
        tmp_path,
        budget=A0Budget(max_gpu_hours=2, max_experiments=2, max_failed_experiments=2),
    )
    records = result.report["full_autonomous_experiment_sequence"]
    rejected = [item for item in records if item["outcome"] == "rejected_candidate"]
    assert rejected
    assert all(item["model_parent_id"] == "M0000" for item in rejected)
    assert result.report["repeated_failure_avoidance"]["rejected_parent_reuse"] is False


def test_execution_failure_consumes_failure_budget(tmp_path):
    result = run_campaign(
        target(),
        A0RuleBasedController(),
        build_model_evolution_registry(ModelEvolutionBackend({"create_student": ["training_oom", "training_oom"]})),
        evaluator(),
        tmp_path,
        budget=A0Budget(max_gpu_hours=2, max_experiments=5, max_failed_experiments=1),
    )
    assert result.payload["failed_experiment_count"] == 1
    assert result.payload["termination_reason"] == "failure_budget_exhausted"


class DuplicateController:
    provider_name = "test"
    model_name = "duplicate"

    def plan(self, context):
        return ExperimentPlan(
            "exp_%04d" % (context.budget_state.used_iterations + 1),
            context.current_model_state.model_id,
            "memory",
            "reduce memory",
            "try create_student",
            "create_student",
            {"width_ratio": 0.5, "block_ratio": 0.5},
            {"peak_memory_gb": "decrease"},
            ["quality regression"],
            {"wall_time_s": 1.0, "gpu_hours": 0.25, "controller_calls": 0, "tier": 1},
            {"max_quality_drop": 0.05},
            {"critical_regression": True},
            "duplicate test",
        )


def test_duplicate_fingerprint_is_blocked_without_execution(tmp_path):
    result = run_campaign(
        target(),
        DuplicateController(),
        build_model_evolution_registry(),
        evaluator(),
        tmp_path,
        budget=A0Budget(max_gpu_hours=2, max_experiments=2, max_failed_experiments=2),
    )
    assert result.report["repeated_failure_avoidance"]["duplicate_fingerprints_blocked"] == 1
    assert result.report["failed_experiment_count"] == 1
    assert result.report["full_autonomous_experiment_sequence"][1]["operator_executed"] is False


def test_research_report_preserves_sequence_and_counts():
    report = build_a0_report(
        {
            "campaign_id": "A0",
            "harness": {"version": "Harness4H3-v1.0"},
            "target_profile_id": "rtx",
            "status": "completed",
            "target_satisfied": True,
            "iterations": [{"outcome": "accepted_candidate"}],
            "system_candidates": [],
            "budget": {"used_gpu_hours": 1.0},
            "pareto_front": [],
            "gpu_hours": 1.0,
            "gpu_hours_available": False,
        },
        "start",
        "end",
    )
    assert report["target_satisfied"] is True
    assert report["total_experiments"] == 1
    assert report["human_intervention_count"] == 0
