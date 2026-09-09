from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.controller.schemas import BudgetState, CostEstimate, ExperimentPlan
from harness4h3.h3.state import ModelState
from harness4h3.memory.experiment_store import ExperimentRecord, ExperimentStore
from harness4h3.target.profile import TargetProfile


def test_target_profile_loads_nested_session_goal():
    target = TargetProfile.from_dict(
        {
            "id": "mobile_h3_v1",
            "hardware": {"type": "mobile", "name": "target_device"},
            "constraints": {"max_peak_memory_gb": 6, "max_latency_s": 30},
            "quality": {"max_quality_drop": 0.05},
            "video": {"resolution": "480p", "duration_s": 5, "fps": 16},
            "priority": ["feasibility", "quality", "latency", "memory"],
        }
    )
    assert target.hardware_type == "mobile"
    assert target.max_peak_memory_gb == 6
    assert target.priority == ("feasibility", "quality", "latency", "memory")
    with pytest.raises(FrozenInstanceError):
        target.id = "changed"  # type: ignore[misc]


def test_budget_rejects_an_experiment_that_exceeds_remaining_gpu_hours():
    budget = BudgetState(max_iterations=3, max_failed_experiments=2, max_gpu_hours=1.0)
    assert budget.can_afford(CostEstimate(gpu_hours=1.1)) is False
    consumed = budget.consume(CostEstimate(wall_time_s=10, gpu_hours=0.25), failed=True, controller_calls=1)
    assert consumed.used_iterations == 1
    assert consumed.used_failures == 1
    assert consumed.used_gpu_hours == 0.25
    assert consumed.used_controller_calls == 1


def test_experiment_plan_requires_structured_fields_and_never_a_shell_command():
    plan = ExperimentPlan.from_dict(
        {
            "experiment_id": "exp_0001",
            "parent_model_id": "M0000",
            "diagnosis": "Peak memory exceeds target.",
            "objective": "Reduce peak memory.",
            "hypothesis": "Four-bit quantization will reduce memory with bounded quality loss.",
            "operator": "quantize",
            "operator_args": {"bits": 4},
            "expected_effects": {"peak_memory_gb": "decrease"},
            "risks": ["quality regression"],
            "required_budget": {"wall_time_s": 1},
            "acceptance": {"max_quality_drop": 0.05},
            "stop_conditions": {"critical_regression": True},
            "rationale": "Memory is the first blocking constraint.",
        }
    )
    assert plan.operator == "quantize"
    assert "command" not in plan.to_dict()


def test_experiment_plan_rejects_unstructured_risks():
    with pytest.raises(ValueError, match="risks must be a list"):
        ExperimentPlan.from_dict(
            {
                "experiment_id": "exp_0001",
                "parent_model_id": "M0000",
                "diagnosis": "memory",
                "objective": "reduce memory",
                "hypothesis": "quantization helps",
                "operator": "quantize",
                "operator_args": {},
                "expected_effects": {"memory": "down"},
                "risks": "quality",
                "required_budget": {},
                "acceptance": {"max_quality_drop": 0.05},
                "stop_conditions": {},
                "rationale": "bounded",
            }
        )


def test_model_candidate_uses_model_namespace_and_embeds_model_state():
    state = ModelState.fake_baseline()
    candidate = ModelCandidate(
        id="M0000",
        parent_id=None,
        generation=0,
        checkpoint_path=state.checkpoint_path,
        state=state,
        created_by_experiment_id=None,
        status="baseline",
    )
    assert candidate.state.model_id == "M0000"
    with pytest.raises(ValueError, match="model candidate id"):
        ModelCandidate("H0000", None, 0, state.checkpoint_path, state, None, "baseline")


def test_experiment_records_are_append_only_and_redact_secrets(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.jsonl")
    record = ExperimentRecord(
        "exp_0001",
        "session",
        "target",
        {"provider": "fake", "api_token": "secret"},
        "M0000",
        None,
        "digest",
        {},
        {},
        {"status": "failed"},
        [],
        {},
        None,
        "failed",
        {"keep": False},
        {},
        "now",
    )
    store.append(record)
    store.append(record)
    records = list(store.read())
    assert len(records) == 2
    assert records[0].controller["api_token"] == "[REDACTED]"
