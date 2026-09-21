from dataclasses import replace

from harness4h3.controller.schemas import ExperimentPlan
from research.experiments.remote_h3_closed_loop import RemoteCampaign


def _plan(operator, *, experiment_id="exp_0012", resource_request=None):
    return ExperimentPlan(
        experiment_id=experiment_id,
        parent_model_id="M0007",
        parent_system_id="S0007",
        diagnosis="bounded test diagnosis",
        objective="improve throughput",
        hypothesis="the intervention improves the measured tradeoff",
        operator=operator,
        operator_args={"target_steps": 16} if operator in {"distill", "step_distill"} else {},
        expected_effects={"quality_score": 0.01},
        risks=["quality regression"],
        required_budget={"max_steps": 1},
        acceptance={"min_quality_score": 0.8},
        stop_conditions={"critical_regression": True},
        rationale="test candidate",
        consumed_observation_ids=["obs-1"],
        diagnosis_evidence=["obs-1"],
        resource_request=resource_request or {},
    )


def _gpu_request(**overrides):
    request = {
        "gpu_count": 2,
        "min_gpu_count": 2,
        "max_gpu_count": 4,
        "elastic": True,
        "distributed": True,
        "exclusive": False,
        "evaluation_workers": 1,
        "on_unavailable": "wait",
    }
    request.update(overrides)
    return request


def test_reuses_same_batch_gpu_candidate_and_rebases_experiment_id():
    base_training_calls = 8
    primary = _plan(
        "prune_blocks",
        resource_request={
            "gpu_count": 0,
            "min_gpu_count": 0,
            "max_gpu_count": 0,
            "elastic": False,
            "distributed": False,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "replan",
        },
    )
    alternate = _plan("distill", resource_request=_gpu_request())
    handle = {
        "prefetch_state_key": "primary",
        # Prefetch stores the successor cursor, not the current evaluation
        # cursor.  The overlap path must therefore consume base + 1.
        "training_calls": base_training_calls + 1,
        "result": {
            "plan": primary,
            "parallel_candidates": [primary, alternate],
        },
    }

    reused = RemoteCampaign._parallel_candidate_from_prefetch_handle(
        handle,
        primary,
        base_training_calls + 1,
    )

    assert reused is not None
    assert reused.experiment_id == "exp_0013"
    assert reused.parent_model_id == primary.parent_model_id
    assert reused.parent_system_id == primary.parent_system_id
    assert reused.operator == "distill"


def test_reuses_same_batch_gpu_candidate_and_rebases_stale_system_id():
    primary = _plan(
        "prune_blocks",
        resource_request={
            "gpu_count": 0,
            "min_gpu_count": 0,
            "max_gpu_count": 0,
            "elastic": False,
            "distributed": False,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "replan",
        },
    )
    alternate = _plan("distill", resource_request=_gpu_request())
    alternate = replace(alternate, parent_system_id="S0000")
    handle = {
        "prefetch_state_key": "primary",
        "training_calls": 9,
        "result": {
            "plan": primary,
            "parallel_candidates": [primary, alternate],
        },
    }

    reused = RemoteCampaign._parallel_candidate_from_prefetch_handle(handle, primary, 9)

    assert reused is not None
    assert reused.parent_system_id == "S0007"


def test_does_not_reuse_non_distributed_or_stale_prefetch_candidate():
    primary = _plan("quantize")
    alternate = _plan("step_distill", resource_request=_gpu_request(elastic=False))
    handle = {
        "prefetch_state_key": "primary",
        "training_calls": 9,
        "result": {"plan": primary, "parallel_candidates": [alternate]},
    }

    assert RemoteCampaign._parallel_candidate_from_prefetch_handle(handle, primary, 8) is None
    assert RemoteCampaign._parallel_candidate_from_prefetch_handle(handle, primary, 9) is None
