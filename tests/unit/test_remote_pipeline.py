import pytest

from harness4h3.remote.lane_packer import pack_worker_request
from harness4h3.remote.pipeline import evaluation_gpu_count, pack_evaluation_overlap, pack_overlap_resources


def test_pack_worker_request_caps_four_gpu_plan_to_three_for_controller():
    effective = pack_worker_request(
        {
            "gpu_count": 4,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": True,
            "evaluation_workers": 0,
            "on_unavailable": "wait",
        },
        total_gpu_count=4,
        controller_overlap_gpus=1,
    )
    assert effective["gpu_count"] == 3
    assert effective["min_gpu_count"] == 2
    assert effective["max_gpu_count"] == 3
    assert effective["elastic"] is True
    assert effective["exclusive"] is False


def test_pack_worker_request_rejects_controller_cap_below_distributed_minimum():
    with pytest.raises(ValueError, match="controller overlap leaves fewer"):
        pack_worker_request(
            {
                "gpu_count": 3,
                "min_gpu_count": 3,
                "max_gpu_count": 3,
                "elastic": False,
                "distributed": True,
                "exclusive": True,
                "evaluation_workers": 0,
                "on_unavailable": "wait",
            },
            total_gpu_count=4,
            controller_overlap_gpus=2,
        )


def test_three_training_gpus_leave_one_for_controller_or_evaluation():
    result = pack_overlap_resources(
        4,
        reserved=(0,),
        minimum_training_gpus=2,
        maximum_training_gpus=4,
    )
    assert result.training_gpus == (1, 2, 3)
    assert result.free_gpus == ()
    assert result.mode == "evaluation_plus_training"


def test_full_training_requires_plan_before_worker_lease():
    result = pack_overlap_resources(
        4,
        reserved=(),
        minimum_training_gpus=4,
        maximum_training_gpus=4,
    )
    assert result.training_gpus == (0, 1, 2, 3)
    assert result.plan_must_be_ready_before_training is True


def test_insufficient_resources_are_reported_without_partial_training():
    result = pack_overlap_resources(
        4,
        reserved=(0, 1, 2),
        minimum_training_gpus=2,
        maximum_training_gpus=4,
    )
    assert result.training_gpus == ()
    assert result.free_gpus == (3,)
    assert result.mode == "waiting"


def test_evaluation_worker_count_never_exceeds_tasks_or_cards():
    assert evaluation_gpu_count(task_count=1, configured_workers=4, free_gpu_count=4) == 1
    assert evaluation_gpu_count(task_count=4, configured_workers=4, free_gpu_count=3) == 3


def test_evaluation_overlap_leaves_two_training_cards():
    result = pack_evaluation_overlap(4, evaluator_gpus=(0,), controller_gpus=(1,), minimum_training_gpus=2)
    assert result.training_gpus == (2, 3)
    assert result.disjoint is True
    assert result.mode == "evaluation_controller_training"


def test_overlap_waits_when_evaluator_and_controller_leave_too_few_cards():
    result = pack_evaluation_overlap(4, evaluator_gpus=(0, 1), controller_gpus=(2,), minimum_training_gpus=2)
    assert result.mode == "waiting"
    assert result.training_gpus == ()


def test_evaluation_overlap_rejects_shared_gpu_sets():
    with pytest.raises(ValueError, match="disjoint"):
        pack_evaluation_overlap(4, evaluator_gpus=(0,), controller_gpus=(0,), minimum_training_gpus=2)
