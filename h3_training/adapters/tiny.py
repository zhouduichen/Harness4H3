"""Rectified-flow adapter for TinyH3."""

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch

from h3_training.adapters.base import DenoisingModelAdapter
from h3_training.algorithms.base import ModelRole
from h3_training.data.schema import (
    Conditioning,
    ModalInterval,
    ModalLatents,
    ModalPrediction,
    ModalSchedule,
    ModalTimesteps,
    PreparedBatch,
    TrainingSample,
)
from h3_training.tiny.factory import load_tiny_checkpoint, tiny_checkpoint_payload


def shifted_sigma(base_sigma: float, shift: float) -> float:
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


class TinyH3Adapter(DenoisingModelAdapter):
    video_shift = 12.0
    audio_shift = 3.0

    def prepare_batch(self, raw: Any, generator: torch.Generator) -> PreparedBatch:
        if isinstance(raw, PreparedBatch):
            return raw
        samples = [raw] if isinstance(raw, TrainingSample) else list(raw)
        if not samples or any(sample.latents is None or sample.text_embedding is None for sample in samples):
            raise ValueError("TinyH3 cached-latent batches require embeddings and latents")
        video = [sample.latents.video for sample in samples]
        audio = [sample.latents.audio for sample in samples]
        latents = ModalLatents(
            video=torch.cat(video) if all(value is not None for value in video) else None,
            audio=torch.cat(audio) if all(value is not None for value in audio) else None,
        )

        def random_tensor(shape, reference, uniform: bool = False):
            # Keep the algorithm generator on CPU for reproducible checkpoints,
            # then transfer the sampled value to the actual model tensor.
            generator_device = getattr(generator, "device", torch.device("cpu"))
            sampler = torch.rand if uniform else torch.randn
            value = sampler(shape, generator=generator, device=generator_device, dtype=torch.float32)
            return value.to(device=reference.device, dtype=reference.dtype)

        noise = ModalLatents(
            video=random_tensor(latents.video.shape, latents.video) if latents.video is not None else None,
            audio=random_tensor(latents.audio.shape, latents.audio) if latents.audio is not None else None,
        )
        batch_size = len(samples)
        reference = latents.video if latents.video is not None else latents.audio
        assert reference is not None
        timestep = random_tensor((batch_size,), reference, uniform=True)
        return PreparedBatch(
            conditioning=Conditioning(torch.cat([sample.text_embedding for sample in samples])),
            latents=latents,
            noise=noise,
            timesteps=ModalTimesteps(
                video=timestep if latents.video is not None else None,
                audio=timestep if latents.audio is not None else None,
            ),
            sample_ids=tuple(sample.sample_id for sample in samples),
        )

    @staticmethod
    def _binary(left: ModalLatents, right: ModalLatents, function) -> ModalLatents:
        return ModalLatents(
            video=function(left.video, right.video) if left.video is not None and right.video is not None else None,
            audio=function(left.audio, right.audio) if left.audio is not None and right.audio is not None else None,
        )

    def add_noise(self, clean: ModalLatents, noise: ModalLatents, timestep: ModalTimesteps) -> ModalLatents:
        def mix(clean_value, noise_value, time):
            time = time.to(device=clean_value.device, dtype=clean_value.dtype)
            noise_value = noise_value.to(device=clean_value.device, dtype=clean_value.dtype)
            while time.ndim < clean_value.ndim:
                time = time.unsqueeze(-1)
            return time * clean_value + (1.0 - time) * noise_value
        return ModalLatents(
            video=mix(clean.video, noise.video, timestep.video) if clean.video is not None else None,
            audio=mix(clean.audio, noise.audio, timestep.audio) if clean.audio is not None else None,
        )

    def predict(self, role: ModelRole, noisy: ModalLatents, timestep: ModalTimesteps, conditioning: Conditioning) -> ModalPrediction:
        return role.model(noisy, conditioning, timestep)

    def prediction_to_clean(self, noisy: ModalLatents, prediction: ModalPrediction, timestep: ModalTimesteps) -> ModalLatents:
        def clean(value, velocity, time):
            time = time.to(device=value.device, dtype=value.dtype)
            velocity = velocity.to(device=value.device, dtype=value.dtype)
            while time.ndim < value.ndim:
                time = time.unsqueeze(-1)
            return value + (1.0 - time) * velocity
        return ModalLatents(
            video=clean(noisy.video, prediction.video, timestep.video) if noisy.video is not None else None,
            audio=clean(noisy.audio, prediction.audio, timestep.audio) if noisy.audio is not None else None,
        )

    def scheduler_step(self, role: ModelRole, latent: ModalLatents, prediction: ModalPrediction, interval: ModalInterval) -> ModalLatents:
        return ModalLatents(
            video=latent.video + (interval.video[0] - interval.video[1]) * prediction.video if latent.video is not None else None,
            audio=latent.audio + (interval.audio[0] - interval.audio[1]) * prediction.audio if latent.audio is not None else None,
        )

    def schedule(self, num_model_evaluations: int) -> ModalSchedule:
        if num_model_evaluations <= 0:
            raise ValueError("number of model evaluations must be positive")
        base = [1.0 - index / num_model_evaluations for index in range(num_model_evaluations + 1)]
        return ModalSchedule(
            video_sigmas=tuple(shifted_sigma(value, self.video_shift) for value in base),
            audio_sigmas=tuple(shifted_sigma(value, self.audio_shift) for value in base),
        )

    def save_role(self, role: ModelRole, path: Path) -> Mapping[str, Any]:
        metadata = dict(role.scheduler_state)
        payload = tiny_checkpoint_payload(
            role.model,
            str(metadata.get("model_id", "child")),
            metadata.get("parent_id"),
            int(metadata.get("sampling_nfe", 4)),
            dict(metadata.get("provenance", {})),
        )
        torch.save(payload, Path(path))
        return {"architecture_name": "TinyH3", "sampling_nfe": payload["sampling_nfe"]}

    def reload_role(self, path: Path) -> ModelRole:
        model, metadata = load_tiny_checkpoint(path)
        return ModelRole("student", model, self, False, metadata)

    def resolve_trainable_parameters(self, role: ModelRole, policy: str) -> Iterable[str]:
        names = [name for name, _ in role.model.named_parameters()]
        if policy == "all":
            return names
        if policy == "heads":
            return [name for name in names if name.startswith(("video_head.", "audio_head."))]
        prefixes = tuple(item.strip() for item in policy.split(",") if item.strip())
        return [name for name in names if name.startswith(prefixes)]
