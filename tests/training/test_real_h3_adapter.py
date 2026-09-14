from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from h3_training.adapters.real_h3 import RealMiniMaxH3Adapter, TRAINABLE_TENSOR_NAMES
from h3_training.algorithms.base import ModelRole
from h3_training.engine.state import TrainingFailure


class FakePackedLayout:
    def __init__(self, *values):
        self.values = values


class FakeH3Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.final_layer = nn.Module()
        self.final_layer.video_out = nn.Linear(1, 1, bias=True)
        self.final_layer.audio_out = nn.Linear(1, 1, bias=True)

    def forward(self, values, timestep, context, transformer_options=None, minimax_payload=None):
        video, audio = values
        video_scale = self.final_layer.video_out.weight.reshape(1, 1, 1, 1, 1)
        audio_scale = self.final_layer.audio_out.weight.reshape(1, 1, 1, 1)
        return video * video_scale, audio * audio_scale


def fake_api():
    def time_shift_sigma(sigma, _video_shift, audio_shift):
        return audio_shift * sigma / (1.0 + (audio_shift - 1.0) * sigma)

    return {
        "ops": SimpleNamespace(disable_weight_init=None),
        "MiniMaxH3Model": FakeH3Model,
        "PackedLayout": FakePackedLayout,
        "time_shift_sigma": time_shift_sigma,
    }


def checkpoint(tmp_path: Path) -> Path:
    model = FakeH3Model()
    path = tmp_path / "M0000.safetensors"
    save_file(
        {name: value.detach() for name, value in model.state_dict().items()},
        str(path),
        metadata={"config": json.dumps({"transformer": {"hidden_size": 1}})},
    )
    return path


def adapter(tmp_path: Path) -> RealMiniMaxH3Adapter:
    value = RealMiniMaxH3Adapter(tmp_path, device="cpu", dtype=torch.float32)
    value._api = fake_api()
    return value


def raw_cache():
    generator = torch.Generator(device="cpu").manual_seed(7)
    return {
        "id": "sample-0",
        "video": torch.randn((320, 96), generator=generator),
        "audio": torch.randn((74, 32), generator=generator),
        "prompt": torch.randn((8, 5120), generator=generator),
    }


def test_real_adapter_loads_prepares_forwards_and_reloads(tmp_path):
    real = adapter(tmp_path)
    role = real.load_role(checkpoint(tmp_path), trainable=True)
    batch = real.prepare_batch(raw_cache(), torch.Generator(device="cpu").manual_seed(11))

    assert batch.metadata["real_h3"] is True
    assert tuple(batch.latents.video.shape) == (1, 24, 5, 16, 16)
    assert tuple(batch.latents.audio.shape) == (1, 32, 2, 37)
    noisy = real.add_noise(batch.latents, batch.noise, batch.timesteps)
    prediction = real.predict(role, noisy, batch.timesteps, batch.conditioning)
    assert prediction.video.shape == noisy.video.shape
    assert prediction.audio.shape == noisy.audio.shape
    assert torch.isfinite(prediction.video).all()
    assert torch.isfinite(prediction.audio).all()
    assert set(real.resolve_trainable_parameters(role, "heads")) == set(TRAINABLE_TENSOR_NAMES)

    child = tmp_path / "M0001.safetensors"
    evidence = real.save_role(role, child)
    reloaded = real.reload_role(child)
    assert evidence["offline_simulation"] is False
    assert set(reloaded.model.state_dict()) == set(role.model.state_dict())


def test_real_adapter_uses_h3_schedule_and_velocity_reconstruction(tmp_path):
    real = adapter(tmp_path)
    schedule = real.schedule(4)
    assert schedule.nfe == 4
    assert schedule.video_sigmas[-1] == 0.0
    assert schedule.audio_sigmas[-1] == 0.0

    clean = torch.ones((1, 1, 2, 2, 2))
    noise = torch.zeros_like(clean)
    timestep = torch.full((1,), 0.25)
    from h3_training.data.schema import ModalLatents, ModalPrediction, ModalTimesteps

    reconstructed = real.prediction_to_clean(
        ModalLatents(video=clean),
        ModalPrediction(video=torch.ones_like(clean)),
        ModalTimesteps(video=timestep),
    )
    assert torch.allclose(reconstructed.video, torch.ones_like(clean) * 1.25)


def test_real_adapter_fails_closed_when_comfyui_is_unavailable(tmp_path):
    real = RealMiniMaxH3Adapter(tmp_path, device="cpu", dtype=torch.float32)
    with pytest.raises(TrainingFailure) as error:
        real.load_role(checkpoint(tmp_path))
    assert error.value.code == "h3_adapter_unavailable"
