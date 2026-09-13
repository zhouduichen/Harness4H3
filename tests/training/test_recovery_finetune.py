import copy

import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.recovery_finetune import RecoveryConfig, RecoveryFineTune
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.engine.state import TrainingFailure
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from h3_training.tiny.model import TinyH3Model


def clone_named_parameters(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def changed_names(before, after):
    return {name for name in before if not torch.equal(before[name], after[name])}


def recovery_fixture(scope="heads"):
    torch.manual_seed(4)
    student = TinyH3Model()
    teacher = copy.deepcopy(student)
    method = RecoveryFineTune(
        student,
        TinyH3Adapter(),
        RecoveryConfig(learning_rate=3e-3, trainable_scope=scope, drift_weight=0.1),
        teacher,
    )
    raw = [[SyntheticH3Dataset(1, 55)[0]]]
    return method, raw


def test_recovery_updates_only_heads_and_decreases_loss():
    method, batches = recovery_fixture()
    before = clone_named_parameters(method.student_model)
    result = TrainerEngine(TrainerConfig(seed=8, max_gradient_norm=10.0)).run(method, batches, max_steps=12)
    after = clone_named_parameters(method.student_model)
    assert result.final_loss < result.initial_loss
    assert changed_names(before, after)
    assert changed_names(before, after) <= method.trainable_parameter_names


def test_teacher_is_frozen_and_unchanged():
    method, batches = recovery_fixture()
    before = clone_named_parameters(method.teacher_model)
    TrainerEngine(TrainerConfig(seed=8)).run(method, batches, max_steps=2)
    after = clone_named_parameters(method.teacher_model)
    assert all(not parameter.requires_grad for parameter in method.teacher_model.parameters())
    assert not changed_names(before, after)


def test_empty_freeze_policy_fails_before_forward():
    method, batches = recovery_fixture("does-not-exist")
    with pytest.raises(TrainingFailure, match="no_trainable_parameters"):
        TrainerEngine().run(method, batches, max_steps=1)


def test_recovery_resume_matches_uninterrupted(tmp_path):
    full_method, batches = recovery_fixture()
    full_engine = TrainerEngine(TrainerConfig(seed=19))
    full_result = full_engine.run(full_method, batches, max_steps=6)

    resumed_method, resumed_batches = recovery_fixture()
    first_engine = TrainerEngine(TrainerConfig(seed=19))
    first_result = first_engine.run(resumed_method, resumed_batches, max_steps=3)
    checkpoint = first_engine.save_checkpoint(resumed_method, first_result.loop_state, tmp_path / "recovery.pt")
    second_engine = TrainerEngine(TrainerConfig(seed=19))
    resumed_result = second_engine.run(resumed_method, resumed_batches, max_steps=6, resume_from=checkpoint)

    full = clone_named_parameters(full_method.student_model)
    resumed = clone_named_parameters(resumed_method.student_model)
    assert all(torch.equal(full[name], resumed[name]) for name in full)
    assert resumed_result.loop_state == full_result.loop_state
