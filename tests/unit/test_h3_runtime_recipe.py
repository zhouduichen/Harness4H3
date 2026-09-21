import json
from pathlib import Path

from harness4h3.benchmark.h3 import H3BenchmarkRunner
from harness4h3.config import Target, WorkflowConfig
from harness4h3.h3.state import ModelState
from harness4h3.harness.state import Task
from harness4h3.operators.model_evolution import build_model_evolution_registry
from harness4h3.target.profile import TargetProfile


def _runner():
    workflow_path = Path("examples/remote_linux_h3_workflow_api.json").resolve()
    workflow_config = WorkflowConfig(
        workflow_path,
        Target("139", "prompt"),
        Target("137", "noise_seed"),
        {"steps": Target("132", "steps"), "cfg": Target("138", "cfg")},
    )
    return H3BenchmarkRunner(
        None,
        None,
        json.loads(workflow_path.read_text(encoding="utf-8")),
        workflow_config,
        Path("/tmp/harness4h3-h3-runtime-test"),
    )


def test_step_distill_can_carry_optional_lpl_tdtm_recipe_without_breaking_legacy_args():
    registry = build_model_evolution_registry()
    state = ModelState.fake_baseline().derive("M0001", sampling_steps=32)
    target = TargetProfile("test", "cuda", "h3")

    registry.validate("step_distill", state, {"target_steps": 16}, target)
    registry.validate(
        "step_distill",
        state,
        {
            "target_steps": 16,
            "lpl_target_steps": 8,
            "tdtm_merge_steps": 4,
            "tdtm_similarity_threshold": 0.99,
        },
        target,
    )


def test_h3_runtime_recipe_rewrites_only_model_sampling_links():
    state = ModelState.fake_baseline().derive(
        "M0001",
        sampling_steps=16,
        runtime_state={
            "h3_optimizations": {
                "lpl": {"target_steps": 8},
                "tdtm": {"merge_steps": 4, "similarity_threshold": 0.99},
            }
        },
    )

    workflow = _runner()._workflow(state, Task("t1", "test", "heldout"))

    assert workflow["132"]["class_type"] == "H3LPLScheduler"
    assert workflow["132"]["inputs"]["target_steps"] == 8
    assert workflow["141"]["class_type"] == "H3OptimizationConfig"
    assert workflow["132"]["inputs"]["model"] == ["141", 0]
    assert workflow["138"]["inputs"]["model"] == ["141", 0]
