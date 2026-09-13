import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.base import ModelRole
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.tiny.model import TinyH3Model


def test_tiny_h3_joint_forward_has_real_gradients():
    model = TinyH3Model()
    adapter = TinyH3Adapter()
    batch = adapter.prepare_batch([SyntheticH3Dataset(1, 3)[0]], torch.Generator().manual_seed(5))
    output = model(batch.latents, batch.conditioning, batch.timesteps)
    loss = output.video.square().mean() + output.audio.square().mean()
    loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in model.parameters())


def test_h3_style_schedule_has_separate_audio_video_sigmas():
    schedule = TinyH3Adapter().schedule(4)
    assert len(schedule.video_sigmas) == 5
    assert len(schedule.audio_sigmas) == 5
    assert schedule.video_sigmas[1] != schedule.audio_sigmas[1]
    assert schedule.video_sigmas[-1] == schedule.audio_sigmas[-1] == 0.0


def test_synthetic_dataset_is_index_deterministic():
    dataset = SyntheticH3Dataset(2, 17)
    assert torch.equal(dataset[1].latents.video, dataset[1].latents.video)
    assert not torch.equal(dataset[0].latents.video, dataset[1].latents.video)
