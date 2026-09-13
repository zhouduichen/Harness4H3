import copy

import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.progressive_distillation import (
    DistillationStage,
    ProgressiveDistillation,
    ProgressiveDistillationConfig,
    plan_binary_stages,
)
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from h3_training.tiny.model import TinyH3Model


def clone(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def equal(left, right):
    return all(torch.equal(left[name], right[name]) for name in left)


def fixture():
    torch.manual_seed(7)
    teacher = TinyH3Model()
    student = copy.deepcopy(teacher)
    config = ProgressiveDistillationConfig(DistillationStage(4, 2), learning_rate=2e-3)
    method = ProgressiveDistillation(student, teacher, TinyH3Adapter(), config)
    return method, [[SyntheticH3Dataset(1, 91)[0]]]


def test_binary_stage_requires_two_to_one_nfe():
    with pytest.raises(ValueError, match="twice"):
        DistillationStage(teacher_nfe=12, student_nfe=8)


def test_plan_binary_stages():
    assert plan_binary_stages(16, 4) == (DistillationStage(16, 8), DistillationStage(8, 4))


def test_progressive_distillation_updates_student_not_teacher():
    method, batches = fixture()
    teacher_before = clone(method.teacher_model)
    student_before = clone(method.student_model)
    result = TrainerEngine(TrainerConfig(seed=13)).run(method, batches, max_steps=4)
    assert equal(clone(method.teacher_model), teacher_before)
    assert not equal(clone(method.student_model), student_before)
    assert result.optimizer_steps == {"student": 4}


def test_progressive_resume_matches_uninterrupted(tmp_path):
    full_method, batches = fixture()
    full_engine = TrainerEngine(TrainerConfig(seed=13))
    full_result = full_engine.run(full_method, batches, max_steps=6)

    resumed_method, resumed_batches = fixture()
    first_engine = TrainerEngine(TrainerConfig(seed=13))
    first = first_engine.run(resumed_method, resumed_batches, max_steps=3)
    path = first_engine.save_checkpoint(resumed_method, first.loop_state, tmp_path / "distill.pt")
    resumed_engine = TrainerEngine(TrainerConfig(seed=13))
    resumed = resumed_engine.run(resumed_method, resumed_batches, max_steps=6, resume_from=path)
    assert equal(clone(full_method.student_model), clone(resumed_method.student_model))
    assert resumed.loop_state == full_result.loop_state
