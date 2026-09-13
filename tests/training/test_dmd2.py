import copy

import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.dmd2 import DMD2, DMD2Config, TimestepNoiseSampler
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.data.schema import Conditioning, ModalLatents, PreparedBatch
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
    student_before = clone(method.student_model)
    critic_before = clone(method.critic_model)
    teacher_before = clone(method.teacher_model)
    result = TrainerEngine(TrainerConfig(seed=31, max_gradient_norm=10.0)).run(method, batches, max_steps=4)
    assert result.optimizer_steps["critic"] == 4
    assert result.optimizer_steps["student"] == 2
    assert method.ema.num_updates == 2
    assert method.critic_updates == 4
    assert method.fake_score_updates == 2
    assert not equal(student_before, clone(method.student_model))
    assert not equal(critic_before, clone(method.critic_model))
    assert equal(teacher_before, clone(method.teacher_model))
    assert all(not parameter.requires_grad for parameter in method.teacher_model.parameters())


def test_dmd2_loss_is_finite_and_gradients_nonzero():
    method, raw = fixture(generator_update_interval=2)
    method.prepare()
    batch = method.prepare_batch(raw[0], torch.Generator().manual_seed(7))
    output = method.training_step(batch, iteration=2)
    output.losses["total_loss"].backward()
    assert torch.isfinite(output.losses["total_loss"])
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in method.student_model.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in method.critic_model.parameters())


def test_dmd2_text_only_mode_needs_no_real_latent():
    method, _ = fixture(generator_update_interval=1, data_mode="text_only")
    config = method.student_model.config
    generator = torch.Generator().manual_seed(88)
    batch = PreparedBatch(
        conditioning=Conditioning(torch.randn(1, config.condition_dim, generator=generator)),
        noise=ModalLatents(
            video=torch.randn(1, config.video_tokens, config.latent_dim, generator=generator),
            audio=torch.randn(1, config.audio_tokens, config.latent_dim, generator=generator),
        ),
    )
    result = TrainerEngine(TrainerConfig(seed=2, max_gradient_norm=10.0)).run(method, [batch], max_steps=1)
    assert result.optimizer_steps == {"critic": 1, "student": 1}


def test_dmd2_real_latent_mode_rejects_noise_only_batch():
    method, _ = fixture(generator_update_interval=1, data_mode="real_latent")
    config = method.student_model.config
    generator = torch.Generator().manual_seed(89)
    batch = PreparedBatch(
        conditioning=Conditioning(torch.randn(1, config.condition_dim, generator=generator)),
        noise=ModalLatents(video=torch.randn(1, config.video_tokens, config.latent_dim, generator=generator)),
    )
    with pytest.raises(RuntimeError, match="invalid_training_config.*real_latent"):
        TrainerEngine(TrainerConfig(seed=2)).run(method, [batch], max_steps=1)


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


def test_dmd2_critic_phase_only_builds_critic_gradients():
    method, raw = fixture(generator_update_interval=2)
    method.prepare()
    batch = method.prepare_batch(raw[0], torch.Generator().manual_seed(9))
    output = method.training_step(batch, iteration=1)
    output.losses["critic_loss"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in method.critic_model.parameters())
    assert all(p.grad is None for p in method.student_model.parameters())


def test_dmd2_student_adversarial_loss_freezes_critic_parameters():
    method, raw = fixture(generator_update_interval=1, adversarial_weight=0.2)
    method.prepare()
    batch = method.prepare_batch(raw[0], torch.Generator().manual_seed(10))
    output = method.training_step(batch, iteration=1)
    output.losses["adversarial_loss"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in method.student_model.parameters())
    assert all(p.grad is None for p in method.critic_model.parameters())
    assert all(p.requires_grad for p in method.critic_model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dmd2_sampler_preserves_cuda_device_and_latent_dtype():
    latents = ModalLatents(
        video=torch.randn(1, 4, 8, device="cuda", dtype=torch.float16),
        audio=torch.randn(1, 3, 8, device="cuda", dtype=torch.float16),
    )
    sampler = TimestepNoiseSampler(7)
    noise = sampler.noise_like(latents)
    timestep = sampler.timestep(1, latents)
    assert noise.video.device.type == "cuda"
    assert noise.video.dtype == latents.video.dtype
    assert timestep.video.device.type == "cuda"
    assert timestep.video.dtype == latents.video.dtype


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
