from __future__ import annotations

import pytest

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.controller.schemas import CostEstimate
from harness4h3.h3.state import ModelState
from harness4h3.operators.base import ExecutionContext, OperatorValidationError
from harness4h3.operators.fake import build_fake_registry
from harness4h3.operators.runtime_memory import RuntimeMemoryOperator, build_runtime_registry
from harness4h3.target.profile import TargetProfile


def _parent() -> ModelCandidate:
    state = ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "model_id": "M0001",
            "parent_model_id": "M0000",
            "checkpoint_path": "D:\\models\\nvfp4.safetensors",
            "architecture_name": "MiniMax-H3",
            "quantization": {"bits": 4, "scheme": "nvfp4"},
            "runtime_state": {"metrics_stale": False},
        }
    )
    return ModelCandidate("M0001", "M0000", 1, state.checkpoint_path, state, "exp_0001", "candidate")


def _target() -> TargetProfile:
    return TargetProfile("rtx", "gpu", "local", max_peak_memory_gb=16.0, max_quality_drop=0.05)


def test_runtime_registry_exposes_single_intervention_operators():
    assert build_runtime_registry().names() == (
        "inspect",
        "quantize",
        "step_distill",
        "rollback",
        "runtime_offload",
        "vae_tiling",
        "inference_chunking",
        "component_lifecycle_optimize",
        "vae_decode_offload",
        "cache_release",
    )


@pytest.mark.parametrize(
    ("name", "args", "kind"),
    [
        ("runtime_offload", {"mode": "aggressive"}, "runtime_offload"),
        ("vae_tiling", {"tile_size": 256, "overlap": 32}, "vae_tiling"),
        ("inference_chunking", {"chunk_size": 4}, "inference_chunking"),
        (
            "component_lifecycle_optimize",
            {
                "unload_text_encoder_after_encode": True,
                "offload_vae_until_decode": True,
                "free_cache_before_decode": True,
            },
            "component_lifecycle_optimize",
        ),
        ("vae_decode_offload", {"mode": "cpu"}, "vae_decode_offload"),
        ("cache_release", {"stage": "before_decode"}, "cache_release"),
    ],
)
def test_runtime_operator_clones_parent_and_records_policy(tmp_path, name, args, kind):
    registry = build_runtime_registry()
    parent = _parent()
    parent_system = SystemCandidate.from_model_candidate("S0000", parent, status="baseline")
    result = registry.execute(
        name,
        parent,
        args,
        _target(),
        ExecutionContext(tmp_path, "M0002", child_system_id="S0001", parent_system=parent_system),
    )
    assert result.ok
    assert result.output_state is None
    assert result.output_system.id == "S0001"
    assert result.output_system.model_ref == parent.id
    assert result.output_system.runtime_state["runtime_policy"]["kind"] == kind
    assert result.output_system.runtime_state["runtime_policy"]["args"] == args
    assert result.output_system.runtime_state["runtime_recipe"][-1]["kind"] == kind
    assert parent.state.runtime_state == {"metrics_stale": False}


def test_runtime_operator_requires_explicit_system_context(tmp_path):
    result = build_runtime_registry().execute(
        "vae_tiling",
        _parent(),
        {"tile_size": 256, "overlap": 32},
        _target(),
        ExecutionContext(tmp_path, "M0002"),
    )
    assert not result.ok
    assert result.failure_type == "runtime_system_context_required"


def test_runtime_operator_emits_system_child_without_model_child(tmp_path):
    parent_model = _parent()
    parent_system = SystemCandidate.from_model_candidate("S0000", parent_model, status="baseline")
    result = build_runtime_registry().execute(
        "vae_tiling",
        parent_model,
        {"tile_size": 256, "overlap": 32},
        _target(),
        ExecutionContext(
            tmp_path,
            "M0001",
            child_system_id="S0001",
            parent_system=parent_system,
        ),
    )
    assert result.output_state is None
    assert result.output_system is not None
    assert result.output_system.id == "S0001"
    assert result.output_system.model_ref == parent_model.id
    assert result.output_system.runtime_state["runtime_policy"]["kind"] == "vae_tiling"


def test_component_lifecycle_operator_materializes_explicit_runtime_state(tmp_path):
    args = {
        "unload_text_encoder_after_encode": True,
        "offload_vae_until_decode": True,
        "free_cache_before_decode": True,
    }
    result = build_runtime_registry().execute(
        "component_lifecycle_optimize",
        _parent(),
        args,
        _target(),
        ExecutionContext(
            tmp_path,
            "M0002",
            child_system_id="S0001",
            parent_system=SystemCandidate.from_model_candidate("S0000", _parent(), status="baseline"),
        ),
    )
    lifecycle = result.output_system.runtime_state["component_lifecycle"]
    assert lifecycle["text_encoder_loaded"] is False
    assert lifecycle["vae_loaded"] is False
    assert lifecycle["cache_state"] == "release_before_decode"
    assert lifecycle["unload_points"] == ["after_encode"]


def test_runtime_operator_rejects_invalid_tiling_without_execution():
    operator = RuntimeMemoryOperator("vae_tiling", "tile", "vae_tiling", {"tile_size": (int,), "overlap": (int,)}, CostEstimate(0.1))
    with pytest.raises(OperatorValidationError, match="tile_size"):
        operator.validate(_parent().state, {"tile_size": 16, "overlap": 32}, _target())


def test_default_fake_registry_remains_phase_i_compatible():
    assert "runtime_offload" not in build_fake_registry().names()
