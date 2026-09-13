from __future__ import annotations

import pytest

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import ExecutionContext, OperatorValidationError
from harness4h3.operators.model_evolution import (
    MODEL_OPERATORS,
    ModelEvolutionBackend,
    build_external_model_evolution_registry,
    build_model_evolution_registry,
)
from harness4h3.target.profile import TargetProfile


def target():
    return TargetProfile("mobile", "mobile", "fake", max_peak_memory_gb=6, max_latency_s=30, max_quality_drop=0.05)


def parent():
    state = ModelState.fake_baseline("M0000")
    return ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")


def test_registry_exposes_all_model_level_operators():
    registry = build_model_evolution_registry()
    assert registry.names() == MODEL_OPERATORS
    assert {item["name"] for item in registry.visible()} == set(MODEL_OPERATORS)


def test_create_student_changes_architecture_without_mutating_parent(tmp_path):
    original = parent()
    result = build_model_evolution_registry().execute(
        "create_student", original, {"width_ratio": 0.5, "block_ratio": 0.5}, target(), ExecutionContext(tmp_path, "M0001")
    )
    assert result.ok
    assert result.output_state.model_id == "M0001"
    assert result.output_state.parent_model_id == "M0000"
    assert result.output_state.parameter_count < original.state.parameter_count
    assert original.state.parameter_count == 13_000_000_000
    assert result.output_state.provenance["offline_simulation"] is True


@pytest.mark.parametrize(
    "name,args",
    [
        ("prune_blocks", {"ratio": 0.1}),
        ("prune_heads", {"ratio": 0.1}),
        ("prune_channels", {"ratio": 0.1}),
        ("distill", {"dataset_fraction": 0.1, "training_steps": 100}),
        ("recovery_finetune", {"training_steps": 100}),
        ("dmd2", {"training_steps": 100}),
        ("step_distill", {"target_steps": 8}),
        ("quantize", {"bits": 4}),
    ],
)
def test_each_operator_returns_a_child_and_cost(tmp_path, name, args):
    result = build_model_evolution_registry().execute(name, parent(), args, target(), ExecutionContext(tmp_path, "M0001"))
    assert result.ok
    assert result.output_state.model_id == "M0001"
    assert result.cost.wall_time_s > 0
    assert result.metrics["offline_simulation"] is True


def test_operator_rejects_invalid_ratio_and_lower_precision(tmp_path):
    registry = build_model_evolution_registry()
    with pytest.raises(OperatorValidationError, match="ratios"):
        registry.validate("create_student", parent().state, {"width_ratio": 1.2, "block_ratio": 0.5}, target())
    quantized = parent().state.derive("M0001", quantization={"bits": 4})
    with pytest.raises(OperatorValidationError, match="already quantized"):
        registry.validate("quantize", quantized, {"bits": 4}, target())


def test_backend_failure_is_execution_level(tmp_path):
    result = build_model_evolution_registry(ModelEvolutionBackend({"distill": ["training_oom"]})).execute(
        "distill", parent(), {"dataset_fraction": 0.1, "training_steps": 100}, target(), ExecutionContext(tmp_path, "M0001")
    )
    assert not result.ok
    assert result.failure_type == "training_oom"


def test_external_registry_uses_the_safe_worker_contract():
    registry = build_external_model_evolution_registry(("worker",))
    assert registry.names() == MODEL_OPERATORS
    assert all(item["input_schema"] for item in registry.visible())
