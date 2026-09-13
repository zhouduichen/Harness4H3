import copy

import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.dmd2 import DMD2, DMD2Config
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from h3_training.tiny.model import TinyH3Model


def clone(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def equal(left, right):
    return all(torch.equal(left[name], right[name]) for name in left)


def fixture(**changes):
    torch.manual_seed(23)
    teacher = TinyH3Model()
    student = copy.deepcopy(teacher)
    critic = copy.deepcopy(teacher)
    config = DMD2Config(**changes)
    method = DMD2(student, teacher, critic, TinyH3Adapter(), config)
    return method, [[SyntheticH3Dataset(1, 100)[0]]]


def test_dmd2_updates_critic_every_step_and_student_on_interval():
    method, batches = fixture(generator_update_interval=2)
    result = TrainerEngine(TrainerConfig(seed=31, max_gradient_norm=10.0)).run(method, batches, max_steps=4)
    assert result.optimizer_steps["critic"] == 4
    assert result.optimizer_steps["student"] == 2
    assert method.ema.num_updates == 2
    assert method.critic_updates == 4


def test_dmd2_loss_is_finite_and_gradients_nonzero():
    method, raw = fixture(generator_update_interval=2)
    method.prepare()
    batch = method.prepare_batch(raw[0], torch.Generator().manual_seed(7))
    output = method.training_step(batch, iteration=2)
    output.losses["total_loss"].backward()
    assert torch.isfinite(output.losses["total_loss"])
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in method.student_model.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in method.critic_model.parameters())


def test_dmd2_optional_real_latent_losses_are_exercised():
    method, raw = fixture(
        generator_update_interval=1,
        data_mode="real_latent",
        regression_weight=0.2,
        adversarial_weight=0.1,
    )
    method.prepare()
    batch = method.prepare_batch(raw[0], torch.Generator().manual_seed(8))
    output = method.training_step(batch, iteration=1)
    assert output.losses["regression_loss"].item() > 0
    assert output.losses["adversarial_loss"].item() > 0


def test_dmd2_resume_restores_roles_optimizers_ema_and_sampler(tmp_path):
    full_method, batches = fixture(generator_update_interval=2)
    full_engine = TrainerEngine(TrainerConfig(seed=41, max_gradient_norm=10.0))
    full_result = full_engine.run(full_method, batches, max_steps=4)

    resumed_method, resumed_batches = fixture(generator_update_interval=2)
    first_engine = TrainerEngine(TrainerConfig(seed=41, max_gradient_norm=10.0))
    first = first_engine.run(resumed_method, resumed_batches, max_steps=2)
    checkpoint = first_engine.save_checkpoint(resumed_method, first.loop_state, tmp_path / "dmd2.pt")
    resumed_engine = TrainerEngine(TrainerConfig(seed=41, max_gradient_norm=10.0))
    resumed_result = resumed_engine.run(resumed_method, resumed_batches, max_steps=4, resume_from=checkpoint)
    assert equal(clone(full_method.student_model), clone(resumed_method.student_model))
    assert equal(clone(full_method.critic_model), clone(resumed_method.critic_model))
    assert full_method.ema.num_updates == resumed_method.ema.num_updates
    assert all(torch.equal(full_method.ema.shadow[n], resumed_method.ema.shadow[n]) for n in full_method.ema.shadow)
    assert full_method.sampler.samples_drawn == resumed_method.sampler.samples_drawn
    assert resumed_result.loop_state == full_result.loop_state
