"""Deterministic synthetic data for the TinyH3 reference trainer."""

from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset

from h3_training.data.schema import ModalLatents, TrainingSample
from h3_training.tiny.model import TinyH3Config


class SyntheticH3Dataset(Dataset):
    def __init__(self, size: int = 32, base_seed: int = 0, config: TinyH3Config = TinyH3Config()) -> None:
        if size <= 0:
            raise ValueError("dataset size must be positive")
        self.size = size
        self.base_seed = base_seed
        self.config = config

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> TrainingSample:
        if not 0 <= index < self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.base_seed + index)
        condition = torch.randn(1, self.config.condition_dim, generator=generator)
        video_base = torch.randn(1, self.config.video_tokens, self.config.latent_dim, generator=generator)
        audio_base = torch.randn(1, self.config.audio_tokens, self.config.latent_dim, generator=generator)
        signal = condition.mean(dim=-1, keepdim=True).unsqueeze(1)
        return TrainingSample(
            sample_id=f"synthetic-{index:06d}",
            prompt=f"synthetic prompt {index}",
            seed=self.base_seed + index,
            text_embedding=condition,
            latents=ModalLatents(video=0.25 * video_base + signal, audio=0.25 * audio_base + signal),
        )


def deterministic_batches(dataset: SyntheticH3Dataset, batch_size: int = 2) -> Iterator[Sequence[TrainingSample]]:
    for start in range(0, len(dataset), batch_size):
        yield [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
