"""Small DMD2-style multi-role reference trainer.

This validates role ownership, score-difference gradients, alternating
optimizers, EMA, and resume. It is not an H3 production recipe.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from h3_training.adapters.base import DenoisingModelAdapter
from h3_training.algorithms.base import ModelRole, StepOutput, TrainingMethod
from h3_training.data.schema import Conditioning, ModalLatents, ModalPrediction, ModalTimesteps, PreparedBatch
from h3_training.engine.state import TrainingFailure


@dataclass(frozen=True)
class DMD2Config:
    student_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    generator_update_interval: int = 2
    ema_decay: float = 0.999
    video_weight: float = 1.0
    audio_weight: float = 1.0
    regression_weight: float = 0.0
    adversarial_weight: float = 0.0
    data_mode: str = "text_only"
    sampler_seed: int = 1234

    def __post_init__(self) -> None:
        if self.student_learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ValueError("DMD2 learning rates must be positive")
        if self.generator_update_interval <= 0:
            raise ValueError("generator update interval must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        if self.data_mode not in {"text_only", "real_latent"}:
            raise ValueError("DMD2 data_mode must be text_only or real_latent")
        if min(self.video_weight, self.audio_weight, self.regression_weight, self.adversarial_weight) < 0:
            raise ValueError("DMD2 loss weights must be non-negative")


class TimestepNoiseSampler:
    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.samples_drawn = 0

    def timestep(self, batch_size: int, modalities: ModalLatents) -> ModalTimesteps:
        values = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.05
        self.samples_drawn += batch_size
        return ModalTimesteps(
            video=values if modalities.video is not None else None,
            audio=values.clone() if modalities.audio is not None else None,
        )

    def noise_like(self, latents: ModalLatents) -> ModalLatents:
        self.samples_drawn += sum(value.shape[0] for value in (latents.video, latents.audio) if value is not None)
        return ModalLatents(
            video=torch.randn(latents.video.shape, generator=self.generator, dtype=latents.video.dtype) if latents.video is not None else None,
            audio=torch.randn(latents.audio.shape, generator=self.generator, dtype=latents.audio.dtype) if latents.audio is not None else None,
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"rng_state": self.generator.get_state(), "samples_drawn": self.samples_drawn}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.generator.set_state(state["rng_state"])
        self.samples_drawn = int(state["samples_drawn"])


class StudentEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        self.num_updates = 0

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, value in model.state_dict().items():
                source = value.detach().cpu()
                if self.shadow[name].is_floating_point():
                    self.shadow[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
                else:
                    self.shadow[name].copy_(source)
        self.num_updates += 1

    def state_dict(self) -> Mapping[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow, "num_updates": self.num_updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise TrainingFailure("resume_mismatch", "EMA decay does not match")
        self.shadow = {name: value.clone() for name, value in state["shadow"].items()}
        self.num_updates = int(state["num_updates"])


class DMD2(TrainingMethod):
    algorithm_name = "dmd2"

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        critic_model: nn.Module,
        adapter: DenoisingModelAdapter,
        config: DMD2Config = DMD2Config(),
        scheduler_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.critic_model = critic_model
        self.adapter = adapter
        self.config = config
        self.student = ModelRole("student", student_model, adapter, True, dict(scheduler_state or {}))
        self.teacher = ModelRole("teacher", teacher_model, adapter, False, {})
        self.critic = ModelRole("critic", critic_model, adapter, True, {})
        self.student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.student_learning_rate)
        self.critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=config.critic_learning_rate)
        self.sampler = TimestepNoiseSampler(config.sampler_seed)
        self.ema = StudentEMA(student_model, config.ema_decay)
        self.critic_updates = 0
        self.student_updates = 0

    def prepare(self) -> None:
        for parameter in self.student_model.parameters():
            parameter.requires_grad_(True)
        for parameter in self.critic_model.parameters():
            parameter.requires_grad_(True)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    def prepare_batch(self, raw, generator: torch.Generator) -> PreparedBatch:
        if isinstance(raw, PreparedBatch):
            return raw
        return self.adapter.prepare_batch(raw, generator)

    @staticmethod
    def _map(latents: ModalLatents, function) -> ModalLatents:
        return ModalLatents(
            video=function(latents.video) if latents.video is not None else None,
            audio=function(latents.audio) if latents.audio is not None else None,
        )

    @staticmethod
    def _mse(actual: ModalLatents, expected: ModalLatents, video_weight: float, audio_weight: float):
        terms = []
        if actual.video is not None:
            terms.append(video_weight * torch.mean((actual.video - expected.video) ** 2))
        if actual.audio is not None:
            terms.append(audio_weight * torch.mean((actual.audio - expected.audio) ** 2))
        return sum(terms)

    @staticmethod
    def _velocity_target(clean: ModalLatents, noise: ModalLatents) -> ModalPrediction:
        return ModalPrediction(
            video=clean.video - noise.video if clean.video is not None else None,
            audio=clean.audio - noise.audio if clean.audio is not None else None,
        )

    def _generator_sample(self, batch: PreparedBatch, with_grad: bool) -> ModalLatents:
        source = batch.noise
        if source is None:
            if batch.latents is None:
                raise TrainingFailure("invalid_training_config", "DMD2 requires a latent shape or initial noise")
            source = self.sampler.noise_like(batch.latents)
        sample = source.video if source.video is not None else source.audio
        timesteps = ModalTimesteps(
            video=torch.zeros(sample.shape[0]) if source.video is not None else None,
            audio=torch.zeros(sample.shape[0]) if source.audio is not None else None,
        )
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            prediction = self.adapter.predict(self.student, source, timesteps, batch.conditioning)
            return self.adapter.prediction_to_clean(source, prediction, timesteps)

    def training_step(self, batch: PreparedBatch, iteration: int) -> StepOutput:
        update_student = iteration % self.config.generator_update_interval == 0
        generated = self._generator_sample(batch, update_student)
        detached_generated = self._map(generated, lambda value: value.detach())
        critic_noise = self.sampler.noise_like(detached_generated)
        sample = detached_generated.video if detached_generated.video is not None else detached_generated.audio
        timestep = self.sampler.timestep(sample.shape[0], detached_generated)
        critic_noisy = self.adapter.add_noise(detached_generated, critic_noise, timestep)
        critic_prediction = self.adapter.predict(self.critic, critic_noisy, timestep, batch.conditioning)
        critic_target = self._velocity_target(detached_generated, critic_noise)
        critic_loss = self._mse(
            critic_prediction, critic_target, self.config.video_weight, self.config.audio_weight
        )
        student_loss = critic_loss.new_zeros(())
        dm_loss = critic_loss.new_zeros(())
        regression_loss = critic_loss.new_zeros(())
        adversarial_loss = critic_loss.new_zeros(())
        if update_student:
            score_noise = self.sampler.noise_like(detached_generated)
            score_timestep = self.sampler.timestep(sample.shape[0], detached_generated)
            score_noisy = self.adapter.add_noise(detached_generated, score_noise, score_timestep)
            with torch.no_grad():
                teacher_prediction = self.adapter.predict(
                    self.teacher, score_noisy, score_timestep, batch.conditioning
                )
                fake_prediction = self.adapter.predict(
                    self.critic, score_noisy, score_timestep, batch.conditioning
                )
                teacher_clean = self.adapter.prediction_to_clean(score_noisy, teacher_prediction, score_timestep)
                fake_clean = self.adapter.prediction_to_clean(score_noisy, fake_prediction, score_timestep)

            def pseudo(generated_value, teacher_value, fake_value):
                difference = teacher_value - fake_value
                scale = difference.abs().mean().clamp_min(1e-6)
                direction = (difference / scale).clamp(-10.0, 10.0)
                return generated_value.detach() + direction.detach()

            pseudo_target = ModalLatents(
                video=pseudo(generated.video, teacher_clean.video, fake_clean.video) if generated.video is not None else None,
                audio=pseudo(generated.audio, teacher_clean.audio, fake_clean.audio) if generated.audio is not None else None,
            )
            dm_loss = self._mse(generated, pseudo_target, self.config.video_weight, self.config.audio_weight)
            if self.config.regression_weight:
                if batch.latents is None:
                    raise TrainingFailure("invalid_training_config", "regression anchor requires real latents")
                regression_loss = self._mse(
                    generated, batch.latents, self.config.video_weight, self.config.audio_weight
                )
            if self.config.adversarial_weight:
                adversarial_prediction = self.adapter.predict(
                    self.critic, generated, ModalTimesteps(
                        video=torch.ones(sample.shape[0]) if generated.video is not None else None,
                        audio=torch.ones(sample.shape[0]) if generated.audio is not None else None,
                    ), batch.conditioning
                )
                zeros = self._map(generated, torch.zeros_like)
                adversarial_loss = self._mse(
                    adversarial_prediction, zeros, self.config.video_weight, self.config.audio_weight
                )
            student_loss = (
                dm_loss
                + self.config.regression_weight * regression_loss
                + self.config.adversarial_weight * adversarial_loss
            )
        total = critic_loss + student_loss
        return StepOutput(
            losses={
                "total_loss": total,
                "critic_loss": critic_loss,
                "distribution_matching_loss": dm_loss,
                "regression_loss": regression_loss,
                "adversarial_loss": adversarial_loss,
            },
            metrics={"student_update": float(update_student)},
        )

    def optimizers(self, iteration: int):
        result = {"critic": self.critic_optimizer}
        if iteration % self.config.generator_update_interval == 0:
            result["student"] = self.student_optimizer
        return result

    def optimizer_map(self):
        return {"critic": self.critic_optimizer, "student": self.student_optimizer}

    def grad_clip_targets(self, iteration: int):
        result = {"critic": self.critic_model}
        if iteration % self.config.generator_update_interval == 0:
            result["student"] = self.student_model
        return result

    def on_optimizers_stepped(self, names):
        if "critic" in names:
            self.critic_updates += 1
        if "student" in names:
            self.student_updates += 1
            self.ema.update(self.student_model)

    def algorithm_state(self):
        return {
            "sampler": self.sampler.state_dict(),
            "ema": self.ema.state_dict(),
            "critic_updates": self.critic_updates,
            "student_updates": self.student_updates,
        }

    def load_algorithm_state(self, state):
        self.sampler.load_state_dict(state["sampler"])
        self.ema.load_state_dict(state["ema"])
        self.critic_updates = int(state["critic_updates"])
        self.student_updates = int(state["student_updates"])
