from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from research.experiments.a0_model_evolution import (
    A0Budget,
    A0RuleBasedController,
    apply_checkpoint_retention,
    build_a0_report,
    run_campaign,
)
from harness4h3.controller.schemas import CostEstimate, ExperimentPlan, OperatorResult
from harness4h3.evaluator.composite import CompositeEvaluator
from harness4h3.evaluator.constraints import ConstraintEvaluator
from harness4h3.evaluator.hardware import FakeHardwareEvaluator
from harness4h3.evaluator.quality import FakeQualityEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import OperatorRegistry
from harness4h3.operators.model_evolution import ModelEvolutionBackend, ModelEvolutionOperator, build_model_evolution_registry
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


def test_rejected_local_child_is_deleted_and_evidence_sidecar_is_retained(tmp_path):
    output_root = tmp_path / "campaign"
    child = output_root / "runs" / "exp_0001" / "artifacts" / "M0001.safetensors"
    evidence = child.with_suffix(child.suffix + ".evidence.json")
    child.parent.mkdir(parents=True)
    child.write_bytes(b"child")
    evidence.write_text('{"child_sha256": "kept"}\n', encoding="utf-8")

    retention = apply_checkpoint_retention(
        output_root,
        "rejected_candidate",
        child,
        output_root / "baseline" / "M0000.safetensors",
        "M0001",
    )

    assert retention["policy"] == "v1"
    assert retention["retained"] is False
    assert retention["deleted"] is True
    assert retention["delete_error"] is None
    assert not child.exists()
    assert evidence.exists()


def test_accepted_child_is_retained(tmp_path):
    output_root = tmp_path / "campaign"
    child = output_root / "runs" / "exp_0001" / "artifacts" / "M0001.safetensors"
    child.parent.mkdir(parents=True)
    child.write_bytes(b"accepted")

    retention = apply_checkpoint_retention(
        output_root,
        "accepted_candidate",
        child,
        output_root / "baseline" / "M0000.safetensors",
        "M0001",
    )

    assert retention["retained"] is True
    assert retention["deleted"] is False
    assert retention["delete_error"] is None
    assert child.read_bytes() == b"accepted"


def test_m0000_is_never_deleted(tmp_path):
    output_root = tmp_path / "campaign"
    baseline = output_root / "baseline" / "M0000.safetensors"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"immutable baseline")

    retention = apply_checkpoint_retention(
        output_root,
        "rejected_candidate",
        baseline,
        None,
        "M0000",
    )

    assert retention["retained"] is True
    assert retention["deleted"] is False
    assert baseline.read_bytes() == b"immutable baseline"


def test_outside_output_root_is_refused(tmp_path):
    output_root = tmp_path / "campaign"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"must remain")

    retention = apply_checkpoint_retention(
        output_root,
        "rejected_candidate",
        outside,
        None,
        "M0001",
    )

    assert retention["retained"] is True
    assert retention["deleted"] is False
    assert retention["delete_error"]
    assert outside.read_bytes() == b"must remain"


def test_symlink_escape_is_refused(tmp_path):
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"must remain")
    link = output_root / "runs" / "exp_0001" / "artifacts" / "M0001.safetensors"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    retention = apply_checkpoint_retention(
        output_root,
        "rejected_candidate",
        link,
        None,
        "M0001",
    )

    assert retention["deleted"] is False
    assert retention["delete_error"]
    assert link.is_symlink()
    assert outside.read_bytes() == b"must remain"


def test_missing_checkpoint_is_a_non_fatal_noop(tmp_path):
    missing = tmp_path / "campaign" / "runs" / "exp_0001" / "artifacts" / "missing.pt"

    retention = apply_checkpoint_retention(
        tmp_path / "campaign",
        "rejected_candidate",
        missing,
        None,
        "M0001",
    )

    assert retention["retained"] is False
    assert retention["deleted"] is True
    assert retention["delete_error"] is None


def test_failed_experiment_removes_temporary_weights_but_keeps_evidence(tmp_path):
    output_root = tmp_path / "campaign"
    experiment_dir = output_root / "runs" / "exp_0001"
    artifacts = experiment_dir / "artifacts"
    artifacts.mkdir(parents=True)
    child = artifacts / "trainer-child.pt"
    optimizer = artifacts / "optimizer_state.pt"
    evidence = artifacts / "reject.evidence.json"
    child.write_bytes(b"child")
    optimizer.write_bytes(b"optimizer")
    evidence.write_text('{"failure": "kept"}\n', encoding="utf-8")

    retention = apply_checkpoint_retention(
        output_root,
        "failed_experiment",
        None,
        tmp_path / "immutable-parent.safetensors",
        "M0001",
        experiment_dir=experiment_dir,
    )

    assert retention["deleted"] is True
    assert retention["delete_error"] is None
    assert not child.exists()
    assert not optimizer.exists()
    assert evidence.exists()


class LocalRejectedBackend:
    def execute(self, name, parent, args, runtime):
        child = Path(runtime.experiment_dir) / "artifacts" / (runtime.child_model_id + ".safetensors")
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_bytes(b"local rejected child")
        metrics = dict(parent.state.measured_metrics)
        metrics["quality_score"] = 0.5
        state = parent.state.derive(runtime.child_model_id, checkpoint_path=str(child), measured_metrics=metrics)
        return OperatorResult("success", state, CostEstimate(wall_time_s=0.01), metrics={"child_sha256": "retained-in-metadata"})


def test_campaign_retention_preserves_metadata_and_trajectory_after_reject(tmp_path):
    registry = OperatorRegistry()
    registry.register(
        ModelEvolutionOperator(
            "create_student",
            "local child test",
            LocalRejectedBackend(),
            {"width_ratio": (float,), "block_ratio": (float,)},
            CostEstimate(wall_time_s=0.01),
        )
    )
    result = run_campaign(
        target(max_quality_drop=0.0001),
        A0RuleBasedController(),
        registry,
        evaluator(),
        tmp_path / "campaign",
        budget=A0Budget(max_gpu_hours=1.0, max_experiments=1, max_failed_experiments=1),
    )

    record = result.report["full_autonomous_experiment_sequence"][0]
    child_path = Path(record["execution_result"]["output_state"]["checkpoint_path"])
    assert record["outcome"] == "rejected_candidate"
    assert record["checkpoint_retention"]["deleted"] is True
    assert not child_path.exists()
    assert (tmp_path / "campaign" / "models" / "candidates" / "M0001.json").exists()
    assert (tmp_path / "campaign" / "campaign.json").exists()
    assert (tmp_path / "campaign" / "report.json").exists()
    trajectory_path = tmp_path / "campaign" / "trajectories.jsonl"
    assert trajectory_path.exists()
    trajectory = trajectory_path.read_text(encoding="utf-8")
    assert "retained-in-metadata" in trajectory
    assert '"checkpoint_retention"' in trajectory
