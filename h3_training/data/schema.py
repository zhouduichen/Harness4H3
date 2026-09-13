"""Modality-aware tensor schema for H3-style training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor


def _require_modality(video: Optional[Any], audio: Optional[Any]) -> None:
    if video is None and audio is None:
        raise ValueError("at least one modality is required")


def _validate_tensor(name: str, value: Optional[Tensor]) -> None:
    if value is None:
        return
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ModalLatents:
    video: Optional[Tensor] = None
    audio: Optional[Tensor] = None

    def __post_init__(self) -> None:
        _require_modality(self.video, self.audio)
        _validate_tensor("video", self.video)
        _validate_tensor("audio", self.audio)
        sizes = {value.shape[0] for value in (self.video, self.audio) if value is not None}
        if len(sizes) > 1:
            raise ValueError("modalities must have matching batch sizes")

    def map(self, function):
        return type(self)(
            video=None if self.video is None else function(self.video),
            audio=None if self.audio is None else function(self.audio),
        )


@dataclass(frozen=True)
class ModalPrediction(ModalLatents):
    pass


@dataclass(frozen=True)
class ModalTimesteps:
    video: Optional[Tensor] = None
    audio: Optional[Tensor] = None

    def __post_init__(self) -> None:
        _require_modality(self.video, self.audio)
        _validate_tensor("video timestep", self.video)
        _validate_tensor("audio timestep", self.audio)
        sizes = {value.shape[0] for value in (self.video, self.audio) if value is not None and value.ndim}
        if len(sizes) > 1:
            raise ValueError("modality timesteps must have matching batch sizes")


@dataclass(frozen=True)
class ModalInterval:
    video: Optional[Tuple[float, float]] = None
    audio: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        _require_modality(self.video, self.audio)
        for name, interval in (("video", self.video), ("audio", self.audio)):
            if interval is None:
                continue
            start, end = interval
            if not (1.0 >= start > end >= 0.0):
                raise ValueError(f"{name} interval must decrease within [0, 1]")


@dataclass(frozen=True)
class ModalSchedule:
    video_sigmas: Optional[Tuple[float, ...]] = None
    audio_sigmas: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        _require_modality(self.video_sigmas, self.audio_sigmas)
        lengths = set()
        for name, values in (("video", self.video_sigmas), ("audio", self.audio_sigmas)):
            if values is None:
                continue
            if len(values) < 2:
                raise ValueError(f"{name} schedule must contain at least one model evaluation")
            if values[-1] != 0.0:
                raise ValueError(f"{name} schedule must end at sigma 0")
            if any(not (1.0 >= value >= 0.0) for value in values):
                raise ValueError(f"{name} schedule must stay within [0, 1]")
            if any(left <= right for left, right in zip(values, values[1:])):
                raise ValueError(f"{name} schedule must be strictly decreasing")
            lengths.add(len(values))
        if len(lengths) > 1:
            raise ValueError("modality schedules must have equal NFE")

    @property
    def nfe(self) -> int:
        values = self.video_sigmas or self.audio_sigmas
        assert values is not None
        return len(values) - 1


@dataclass(frozen=True)
class Conditioning:
    text: Tensor
    negative_text: Optional[Tensor] = None

    def __post_init__(self) -> None:
        _validate_tensor("text conditioning", self.text)
        _validate_tensor("negative text conditioning", self.negative_text)


@dataclass(frozen=True)
class MediaReferences:
    video: Optional[Path] = None
    audio: Optional[Path] = None


@dataclass(frozen=True)
class TeacherSignals:
    prediction: Optional[ModalPrediction] = None
    trajectory: Tuple[ModalLatents, ...] = ()


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    prompt: str
    seed: int
    media: Optional[MediaReferences] = None
    text_embedding: Optional[Tensor] = None
    latents: Optional[ModalLatents] = None
    noise: Optional[ModalLatents] = None
    timesteps: Optional[ModalTimesteps] = None
    teacher_signals: Optional[TeacherSignals] = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        _validate_tensor("text embedding", self.text_embedding)


@dataclass(frozen=True)
class PreparedBatch:
    conditioning: Conditioning
    latents: Optional[ModalLatents] = None
    noise: Optional[ModalLatents] = None
    timesteps: Optional[ModalTimesteps] = None
    sample_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch_sizes = []
        if self.conditioning.text.ndim:
            batch_sizes.append(self.conditioning.text.shape[0])
        for modal in (self.latents, self.noise):
            if modal is not None:
                value = modal.video if modal.video is not None else modal.audio
                assert value is not None
                batch_sizes.append(value.shape[0])
        if len(set(batch_sizes)) > 1:
            raise ValueError("prepared batch fields must have matching batch sizes")
        if self.sample_ids and batch_sizes and len(self.sample_ids) != batch_sizes[0]:
            raise ValueError("sample_ids must match the prepared batch size")
