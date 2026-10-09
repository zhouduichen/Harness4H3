import torch

from h3_training.data.schema import ModalInterval

from harness4h3.student.wan_training import (
    WanTrainingSpec,
    flow_match_batch,
    resample_video_frames,
    wan_velocity_loss,
)


class FakeScheduler:
    class Config:
        num_train_timesteps = 1000

    config = Config()


def test_wan_training_spec_uses_native_five_second_contract():
    spec = WanTrainingSpec.from_mapping({"model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"})
    assert spec.num_frames == 81
    assert spec.native_fps == 16.0
    assert 5.0 < spec.duration_s < 5.1


def test_flow_match_batch_target_is_noise_minus_clean():
    clean = torch.zeros(1, 16, 21, 32, 32)
    noise = torch.ones_like(clean)
    noisy, timestep, target, sigma = flow_match_batch(
        clean,
        scheduler=FakeScheduler(),
        generator=torch.Generator(device="cpu").manual_seed(1),
        device=torch.device("cpu"),
        noise=noise,
        sigma=torch.tensor([0.25]),
    )
    assert torch.allclose(noisy, torch.full_like(clean, 0.25))
    assert torch.allclose(target, torch.ones_like(clean))
    assert torch.equal(timestep, torch.tensor([250.0]))
    assert torch.equal(sigma, torch.tensor([0.25]))


def test_wan_velocity_loss_has_clean_anchor_term():
    clean = torch.zeros(1, 1, 1, 1, 1)
    noisy = torch.full_like(clean, 0.25)
    prediction = torch.zeros_like(clean)
    target = torch.ones_like(clean)
    total, standard, distill = wan_velocity_loss(
        prediction, target, noisy, clean, torch.tensor([0.25]), standard_weight=1.0, distill_weight=0.5
    )
    assert standard.item() == 1.0
    assert distill.item() == 0.0625
    assert total.item() == 1.03125


def test_wan_velocity_loss_is_zero_for_exact_flow_prediction():
    clean = torch.full((1, 1, 1, 1, 1), 2.0)
    noise = torch.full_like(clean, 1.0)
    noisy = torch.full_like(clean, 1.5)
    prediction = noise - clean
    total, standard, distill = wan_velocity_loss(
        prediction,
        prediction,
        noisy,
        clean,
        torch.tensor([0.5]),
        standard_weight=1.0,
        distill_weight=0.5,
    )
    assert total.item() == 0.0
    assert standard.item() == 0.0
    assert distill.item() == 0.0


def test_resample_video_frames_to_wan_contract():
    frames = torch.zeros(120, 256, 256, 3)
    result = resample_video_frames(frames, num_frames=81, height=256, width=256)
    assert tuple(result.shape) == (81, 256, 256, 3)
