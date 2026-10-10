from types import SimpleNamespace

import pytest
import torch

from h3_training.data.schema import Conditioning, ModalInterval, ModalLatents, ModalPrediction, ModalSchedule, ModalTimesteps
from h3_training.wan_trajectory_builder import build_wan_trajectory_record, replay_teacher_from_record


class FakeNativeAdapter:
    def __init__(self):
        self.index = 0
        self.prediction_coordinates = []

    def schedule(self, nfe):
        return ModalSchedule(video_sigmas=tuple(float(nfe - i) / nfe for i in range(nfe)) + (0.0,))

    def native_schedule_metadata(self, schedule):
        sigmas = schedule.video_sigmas
        assert sigmas is not None
        return {"class": "fake.Native", "timesteps": [1000.0 * value for value in sigmas[:-1]], "nfe": schedule.nfe}

    def begin_trajectory(self, role, schedule, start_index, device):
        self.index = int(start_index)

    def capture_native_scheduler_state(self, role):
        return {"index": self.index, "history": torch.tensor([float(self.index)])}

    def add_noise(self, clean, noise, timestep):
        assert clean.video is not None and noise.video is not None and timestep.video is not None
        sigma = timestep.video.reshape(-1, 1, 1, 1, 1)
        return ModalLatents(video=(1.0 - sigma) * clean.video + sigma * noise.video)

    def predict(self, role, noisy, timestep, conditioning):
        assert noisy.video is not None
        assert timestep.video is not None
        self.prediction_coordinates.extend(float(value) for value in timestep.video.reshape(-1))
        return ModalPrediction(video=torch.ones_like(noisy.video) * 0.125)

    def scheduler_step(self, role, latent, prediction, interval):
        assert latent.video is not None and prediction.video is not None and interval.video is not None
        self.index += 1
        current, next_value = interval.video
        return ModalLatents(video=latent.video + (next_value - current) * prediction.video)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, *args, **kwargs):
        return args[0]


def components(tmp_path):
    adapter = FakeNativeAdapter()
    role = SimpleNamespace(model=FakeModel())
    clean = ModalLatents(video=torch.full((1, 1, 1, 2, 2), 0.5))
    conditioning = Conditioning(text=torch.ones(1, 2, 3), metadata={"prompt": "test"})
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"native-h3-cache")
    teacher = tmp_path / "teacher.safetensors"
    teacher.write_bytes(b"teacher")
    return adapter, role, clean, conditioning, cache, teacher


def test_builder_repeats_noise_and_boundaries(tmp_path):
    adapter, role, clean, conditioning, cache, teacher = components(tmp_path)
    first = build_wan_trajectory_record(
        adapter=adapter, teacher_role=role, clean=clean, conditioning=conditioning,
        cache_item=cache, teacher_checkpoint=teacher, output_dir=tmp_path / "a", seed=20261008,
        source_nfe=4, target_nfe=2,
    )
    second = build_wan_trajectory_record(
        adapter=adapter, teacher_role=role, clean=clean, conditioning=conditioning,
        cache_item=cache, teacher_checkpoint=teacher, output_dir=tmp_path / "b", seed=20261008,
        source_nfe=4, target_nfe=2,
    )
    assert first.metadata.noise_digest == second.metadata.noise_digest
    assert first.metadata.tensor_digest == second.metadata.tensor_digest
    assert first.noise.dtype == torch.float32
    assert len(first.teacher_states) == 5
    assert len(first.teacher_predictions) == 4
    assert len(first.teacher_scheduler_snapshots) == 5


def test_builder_replay_matches_record(tmp_path):
    adapter, role, clean, conditioning, cache, teacher = components(tmp_path)
    trajectory = build_wan_trajectory_record(
        adapter=adapter, teacher_role=role, clean=clean, conditioning=conditioning,
        cache_item=cache, teacher_checkpoint=teacher, output_dir=tmp_path / "record", seed=20261008,
        source_nfe=4, target_nfe=2,
    )
    replay = replay_teacher_from_record(
        adapter=adapter, teacher_role=role, conditioning=conditioning, trajectory=trajectory
    )
    assert all(torch.equal(left, right) for left, right in zip(replay, trajectory.teacher_states))


def test_builder_passes_sigma_coordinates_to_adapter(tmp_path):
    adapter, role, clean, conditioning, cache, teacher = components(tmp_path)
    build_wan_trajectory_record(
        adapter=adapter, teacher_role=role, clean=clean, conditioning=conditioning,
        cache_item=cache, teacher_checkpoint=teacher, output_dir=None, seed=20261008,
        source_nfe=4, target_nfe=2,
    )
    assert adapter.prediction_coordinates
    assert all(0.0 <= value <= 1.0 for value in adapter.prediction_coordinates)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA RNG parity")
def test_builder_noise_matches_wan_device_rng(tmp_path):
    adapter, role, clean, conditioning, cache, teacher = components(tmp_path)
    clean = ModalLatents(video=clean.video.cuda())
    trajectory = build_wan_trajectory_record(
        adapter=adapter, teacher_role=role, clean=clean, conditioning=conditioning,
        cache_item=cache, teacher_checkpoint=teacher, output_dir=None, seed=20261008,
        source_nfe=4, target_nfe=2,
    )
    expected = torch.randn(
        clean.video.shape,
        generator=torch.Generator(device="cuda").manual_seed(20261008),
        device="cuda",
        dtype=torch.float32,
    ).cpu()
    assert torch.equal(trajectory.noise, expected)
