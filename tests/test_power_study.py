from experiments.power_study import bootstrap_ci, build_tasks, is_retryable_interruption
from harness4h3.memory.trajectory import Trajectory


def test_power_study_has_requested_prompt_and_seed_counts():
    sanity, dev, heldout = build_tasks()
    assert len(sanity) == 3
    assert len(dev) == 30 * 3
    assert len(heldout) == 60 * 3
    assert len({task.id for task in sanity + dev + heldout}) == 273


def test_bootstrap_ci_is_deterministic_and_positive():
    first = bootstrap_ci([0.01, 0.02, 0.03], samples=1000)
    second = bootstrap_ci([0.01, 0.02, 0.03], samples=1000)
    assert first == second
    assert first[0] > 0


def test_resource_guard_interruption_is_retryable():
    item = Trajectory(
        task_id="dev-001-s42",
        split="dev",
        harness_version="H0",
        inputs={},
        steps=[{"action": "failure", "message": "execution_interrupted"}],
        final_result=None,
        evaluation={},
        score=0.0,
        failure_type="backend_execution",
        critical_regression=True,
        cost={},
    )
    assert is_retryable_interruption(item)
