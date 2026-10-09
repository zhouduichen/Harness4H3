import pytest
from diffusers import FlowMatchEulerDiscreteScheduler

from harness4h3.student.wan_schedule import (
    coarsen_wan_schedule,
    native_wan_schedule,
    training_uniform_wan_schedule,
)


def test_native_schedule_matches_checkpoint_scheduler_and_exposes_terminal_zero():
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    actual = native_wan_schedule(scheduler, 16)
    scheduler.set_timesteps(16, device="cpu")
    assert actual.sigmas == tuple(float(value) for value in scheduler.sigmas.tolist())
    assert actual.timesteps == tuple(float(value) for value in scheduler.timesteps.tolist())
    assert actual.sigmas[-1] == 0.0
    assert len(actual.timesteps) == 16


def test_coarsened_native_32_schedule_is_exactly_binary_aligned():
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    source = native_wan_schedule(scheduler, 32)
    target = coarsen_wan_schedule(source, 2)
    assert target.sigmas == source.sigmas[::2]
    assert target.nfe == 16
    assert target.timesteps == source.timesteps[::2]


def test_training_uniform_schedule_is_not_h3_shifted():
    schedule = training_uniform_wan_schedule(4)
    assert schedule.sigmas == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert schedule.timesteps == (1000.0, 750.0, 500.0, 250.0)


def test_schedule_rejects_odd_coarsening():
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    source = native_wan_schedule(scheduler, 15)
    with pytest.raises(ValueError, match="factor"):
        coarsen_wan_schedule(source, 2)
