"""Supervised rectified-flow recovery with an optional teacher drift anchor."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from h3_training.adapters.base import DenoisingModelAdapter
from h3_training.algorithms.base import ModelRole, StepOutput, TrainingMethod
from h3_training.data.schema import ModalPrediction, PreparedBatch
from h3_training.engine.state import TrainingFailure


@dataclass(frozen=True)
class RecoveryConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    trainable_scope: str = "heads"
    sft_weight: float = 1.0
    drift_weight: float = 0.0
    video_weight: float = 1.0
    audio_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are invalid")
        if self.sft_weight < 0 or self.drift_weight < 0 or self.sft_weight + self.drift_weight <= 0:
            raise ValueError("at least one recovery loss weight must be positive")
        if self.video_weight < 0 or self.audio_weight < 0 or self.video_weight + self.audio_weight <= 0:
            raise ValueError("at least one modality weight must be positive")


class RecoveryFineTune(TrainingMethod):
    algorithm_name = "recovery_finetune"

    def __init__(
        self,
        student_model: nn.Module,
        adapter: DenoisingModelAdapter,
        config: RecoveryConfig = RecoveryConfig(),
        teacher_model: Optional[nn.Module] = None,
        scheduler_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.adapter = adapter
        self.config = config
        self.student = ModelRole("student", self.student_model, adapter, True, dict(scheduler_state or {}))
        self.teacher = (
            ModelRole("teacher", self.teacher_model, adapter, False, {}) if self.teacher_model is not None else None
        )
        self._optimizer = None
        self.trainable_parameter_names = frozenset()
        self.frozen_parameter_names = frozenset()

    def prepare(self) -> None:
        names = frozenset(self.adapter.resolve_trainable_parameters(self.student, self.config.trainable_scope))
        all_names = frozenset(name for name, _ in self.student_model.named_parameters())
        if not names or not names.issubset(all_names):
            raise TrainingFailure("no_trainable_parameters", self.config.trainable_scope)
        self.trainable_parameter_names = names
        self.frozen_parameter_names = all_names - names
        parameters = []
        for name, parameter in self.student_model.named_parameters():
            parameter.requires_grad_(name in names)
            if name in names:
                parameters.append(parameter)
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )

    def prepare_batch(self, raw, generator: torch.Generator) -> PreparedBatch:
        return self.adapter.prepare_batch(raw, generator)

    @staticmethod
    def _mse(actual, expected, video_weight: float, audio_weight: float):
        terms = []
        if actual.video is not None:
            if video_weight <= 0:
                raise TrainingFailure("invalid_training_config", "present video modality has zero weight")
            terms.append(video_weight * torch.mean((actual.video - expected.video) ** 2))
        if actual.audio is not None:
            if audio_weight <= 0:
                raise TrainingFailure("invalid_training_config", "present audio modality has zero weight")
            terms.append(audio_weight * torch.mean((actual.audio - expected.audio) ** 2))
        return sum(terms)

    def training_step(self, batch: PreparedBatch, iteration: int) -> StepOutput:
        if batch.latents is None or batch.noise is None or batch.timesteps is None:
            raise TrainingFailure("invalid_training_config", "recovery requires latent, noise, and timestep")
        noisy = self.adapter.add_noise(batch.latents, batch.noise, batch.timesteps)
        prediction = self.adapter.predict(self.student, noisy, batch.timesteps, batch.conditioning)
        target = ModalPrediction(
            video=batch.latents.video - batch.noise.video if batch.latents.video is not None else None,
            audio=batch.latents.audio - batch.noise.audio if batch.latents.audio is not None else None,
        )
        flow_loss = self._mse(prediction, target, self.config.video_weight, self.config.audio_weight)
        drift_loss = flow_loss.new_zeros(())
        if self.config.drift_weight:
            if self.teacher is None:
                raise TrainingFailure("invalid_training_config", "drift loss requires a teacher")
            with torch.no_grad():
                teacher_prediction = self.adapter.predict(
                    self.teacher, noisy, batch.timesteps, batch.conditioning
                )
            drift_loss = self._mse(
                prediction, teacher_prediction, self.config.video_weight, self.config.audio_weight
            )
        total = self.config.sft_weight * flow_loss + self.config.drift_weight * drift_loss
        return StepOutput(
            losses={"total_loss": total, "flow_loss": flow_loss, "drift_loss": drift_loss},
            metrics={"iteration": float(iteration)},
        )

    def optimizers(self, iteration: int):
        if self._optimizer is None:
            raise TrainingFailure("invalid_training_config", "recovery method was not prepared")
        return {"student": self._optimizer}

    def optimizer_map(self):
        return self.optimizers(1)

    def grad_clip_targets(self, iteration: int):
        return {"student": self.student_model}
