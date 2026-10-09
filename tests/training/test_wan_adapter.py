import pytest
import torch

from h3_training.adapters.wan import WanEndpointTeacher, WanTrainingAdapter
from h3_training.algorithms.base import ModelRole
from h3_training.data.schema import ModalInterval, ModalLatents, ModalPrediction, ModalTimesteps
from h3_training.engine.state import TrainingFailure
from harness4h3.student.wan_schedule import WanSigmaSchedule


class FakeScheduler:
    class Config:
        num_train_timesteps = 1000

    config = Config()


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_head = torch.nn.Linear(1, 1)

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        scale = self.output_head.weight.reshape(1, 1, 1, 1, 1)
        result = hidden_states * scale + encoder_hidden_states.mean() * 0.0 + timestep.reshape(-1, 1, 1, 1, 1) * 0.0
        return (result,)


class CFGTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.calls.append(float(encoder_hidden_states.mean()))
        return (torch.ones_like(hidden_states) * encoder_hidden_states.mean(),)


class TimestepTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_head = torch.nn.Linear(1, 1)
        self.timesteps = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.timesteps.append(float(timestep.reshape(-1)[0]))
        return (torch.zeros_like(hidden_states),)


def make_adapter():
    transformer = FakeTransformer()
    return WanTrainingAdapter(
        transformer=transformer,
        scheduler=FakeScheduler(),
        encode_frames=lambda frames: torch.as_tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous(),
        encode_prompt=lambda prompt: torch.ones(1, 2, 4),
    )


def test_wan_adapter_flow_match_and_clean_equation():
    adapter = make_adapter()
    clean = ModalLatents(video=torch.zeros(1, 4, 2, 2, 2))
    noise = ModalLatents(video=torch.ones_like(clean.video))
    noisy = adapter.add_noise(clean, noise, ModalTimesteps(video=torch.tensor([0.25])))
    assert torch.allclose(noisy.video, torch.full_like(clean.video, 0.25))
    prediction = ModalPrediction(video=torch.ones_like(clean.video))
    recovered = adapter.prediction_to_clean(noisy, prediction, ModalTimesteps(video=torch.tensor([0.25])))
    assert torch.allclose(recovered.video, torch.zeros_like(clean.video))


def test_wan_adapter_schedule_is_binary_aligned():
    adapter = make_adapter()
    teacher = adapter.schedule(8).video_sigmas
    student = adapter.schedule(4).video_sigmas
    assert teacher is not None and student is not None
    assert all(abs(student[index] - teacher[2 * index]) < 1e-8 for index in range(5))


def test_wan_adapter_prepares_rgb_boundary_and_native_conditioning():
    adapter = make_adapter()
    batch = adapter.prepare_batch(
        {"sample_id": "s0", "prompt": "red train", "frames": torch.zeros(2, 2, 2, 3)},
        torch.Generator().manual_seed(7),
    )
    assert batch.latents is not None and batch.latents.video.shape == (1, 3, 2, 2, 2)
    assert batch.conditioning.text.shape[0] == 1
    assert batch.sample_ids == ("s0",)


def test_wan_adapter_forward_uses_native_transformer_role():
    adapter = make_adapter()
    batch = adapter.prepare_batch(
        {"sample_id": "s0", "prompt": "red train", "frames": torch.zeros(2, 2, 2, 3)},
        torch.Generator().manual_seed(7),
    )
    role = ModelRole("student", adapter.transformer, adapter, True)
    prediction = adapter.predict(
        role,
        batch.latents,
        ModalTimesteps(video=torch.tensor([0.5])),
        batch.conditioning,
    )
    assert prediction.video.shape == batch.latents.video.shape


def test_wan_adapter_matches_runtime_cfg_formula():
    transformer = CFGTransformer()
    adapter = WanTrainingAdapter(
        transformer=transformer,
        scheduler=FakeScheduler(),
        encode_frames=lambda frames: torch.as_tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous(),
        encode_prompt=lambda prompt: (torch.full((1, 2, 4), 3.0), torch.full((1, 2, 4), 1.0)),
        guidance_scale=5.0,
    )
    batch = adapter.prepare_batch(
        {"sample_id": "s0", "prompt": "red train", "frames": torch.zeros(2, 2, 2, 3)},
        torch.Generator().manual_seed(7),
    )
    prediction = adapter.predict(
        ModelRole("student", transformer, adapter, True),
        batch.latents,
        ModalTimesteps(video=torch.tensor([0.5])),
        batch.conditioning,
    )
    assert transformer.calls == [3.0, 1.0]
    assert torch.allclose(prediction.video, torch.full_like(prediction.video, 11.0))


def test_wan_adapter_passes_native_quantized_timestep_to_transformer():
    transformer = TimestepTransformer()
    schedule = WanSigmaSchedule(
        mode="native",
        sigmas=(0.9999989867210386, 0.0),
        timesteps=(999.0,),
        num_train_timesteps=1000,
        shift=3.0,
    )
    adapter = WanTrainingAdapter(
        transformer=transformer,
        scheduler=FakeScheduler(),
        encode_frames=lambda frames: torch.as_tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous(),
        encode_prompt=lambda prompt: torch.ones(1, 2, 4),
        schedule_provider=lambda _: schedule,
    )
    adapter.schedule(1)
    batch = adapter.prepare_batch(
        {"sample_id": "s0", "prompt": "red train", "frames": torch.zeros(2, 2, 2, 3)},
        torch.Generator().manual_seed(7),
    )
    adapter.predict(
        ModelRole("student", transformer, adapter, True),
        batch.latents,
        ModalTimesteps(video=torch.tensor([0.9999989867210386])),
        batch.conditioning,
    )
    assert transformer.timesteps == [999.0]


def test_wan_adapter_endpoint_teacher_is_role_type_not_role_name():
    adapter = make_adapter()
    batch = adapter.prepare_batch(
        {"sample_id": "s0", "prompt": "red train", "frames": torch.zeros(2, 2, 2, 3)},
        torch.Generator().manual_seed(7),
    )
    endpoint = ModelRole("critic", WanEndpointTeacher(), adapter, False)
    prediction = adapter.predict(
        endpoint,
        batch.latents,
        ModalTimesteps(video=torch.tensor([0.5])),
        batch.conditioning,
    )
    assert torch.allclose(prediction.video, batch.latents.video / 0.5)


def test_wan_adapter_scheduler_step_moves_down_sigma_with_noise_minus_clean_velocity():
    adapter = make_adapter()
    latent = ModalLatents(video=torch.full((1, 1, 1, 1, 1), 0.5))
    prediction = ModalPrediction(video=torch.ones_like(latent.video))
    updated = adapter.scheduler_step(
        ModelRole("student", adapter.transformer, adapter, True),
        latent,
        prediction,
        ModalInterval(video=(0.75, 0.25)),
    )
    assert torch.equal(updated.video, torch.zeros_like(latent.video))


def test_wan_endpoint_teacher_matches_noise_minus_clean_for_nonzero_clean():
    adapter = make_adapter()
    clean = torch.full((1, 1, 1, 1, 1), 2.0)
    batch = adapter.prepare_batch(
        {"sample_id": "s0", "prompt": "test", "wan_latent": clean, "frames": torch.zeros(1, 1, 1, 3)},
        torch.Generator().manual_seed(7),
    )
    noisy = ModalLatents(video=torch.full_like(clean, 1.5))
    endpoint = ModelRole("critic", WanEndpointTeacher(), adapter, False)
    prediction = adapter.predict(endpoint, noisy, ModalTimesteps(video=torch.tensor([0.5])), batch.conditioning)
    assert torch.equal(prediction.video, torch.full_like(clean, -1.0))


def test_wan_prediction_to_clean_is_exact_for_endpoint_velocity():
    adapter = make_adapter()
    noisy = ModalLatents(video=torch.full((1, 1, 1, 1, 1), 1.5))
    prediction = ModalPrediction(video=torch.full_like(noisy.video, -1.0))
    clean = adapter.prediction_to_clean(noisy, prediction, ModalTimesteps(video=torch.tensor([0.5])))
    assert torch.equal(clean.video, torch.full_like(noisy.video, 2.0))


def test_wan_adapter_scope_rejects_empty_selection():
    adapter = make_adapter()
    role = ModelRole("student", adapter.transformer, adapter, True)
    with pytest.raises(TrainingFailure, match="no_trainable"):
        tuple(adapter.resolve_trainable_parameters(role, "missing.*"))
