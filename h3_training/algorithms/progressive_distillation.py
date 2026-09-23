"""Binary progressive distillation over adapter-owned modality schedules."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import nn

from h3_training.adapters.base import DenoisingModelAdapter
from h3_training.algorithms.base import ModelRole, StepOutput, TrainingMethod
from h3_training.data.schema import ModalInterval, ModalLatents, ModalTimesteps, PreparedBatch
from h3_training.engine.state import TrainingFailure


@dataclass(frozen=True)
class DistillationStage:
    teacher_nfe: int
    student_nfe: int

    def __post_init__(self) -> None:
        if self.student_nfe <= 0 or self.teacher_nfe != 2 * self.student_nfe:
            raise ValueError("teacher NFE must be exactly twice student NFE")


def plan_binary_stages(source_nfe: int, target_nfe: int) -> Tuple[DistillationStage, ...]:
    if source_nfe <= 0 or target_nfe <= 0 or target_nfe > source_nfe:
        raise ValueError("invalid distillation NFE range")
    stages = []
    current = source_nfe
    while current > target_nfe:
        if current % 2:
            raise ValueError("binary distillation requires an even teacher NFE")
        following = current // 2
        if following < target_nfe:
            raise ValueError("target NFE is not reachable by binary stages")
        stages.append(DistillationStage(current, following))
        current = following
    return tuple(stages)


@dataclass(frozen=True)
class ProgressiveDistillationConfig:
    stage: DistillationStage
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    trainable_scope: str = "all"
    video_weight: float = 1.0
    audio_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are invalid")
        if self.video_weight < 0 or self.audio_weight < 0 or self.video_weight + self.audio_weight <= 0:
            raise ValueError("at least one modality weight must be positive")


class ProgressiveDistillation(TrainingMethod):
    algorithm_name = "progressive_distillation"

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        adapter: DenoisingModelAdapter,
        config: ProgressiveDistillationConfig,
        scheduler_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.adapter = adapter
        self.config = config
        self.student = ModelRole("student", student_model, adapter, True, dict(scheduler_state or {}))
        self.teacher = ModelRole("teacher", teacher_model, adapter, False, {})
        self._optimizer = None
        self.trainable_parameter_names = frozenset()
        self.frozen_parameter_names = frozenset()
        self._student_schedule = None
        self._teacher_schedule = None

    def prepare(self) -> None:
        self._student_schedule = self.adapter.schedule(self.config.stage.student_nfe)
        self._teacher_schedule = self.adapter.schedule(self.config.stage.teacher_nfe)
        for modality in ("video_sigmas", "audio_sigmas"):
            student = getattr(self._student_schedule, modality)
            teacher = getattr(self._teacher_schedule, modality)
            if student is None or teacher is None:
                continue
            if any(abs(student[index] - teacher[2 * index]) > 1e-8 for index in range(len(student))):
                raise TrainingFailure("invalid_training_config", f"misaligned {modality} schedule")
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
    def _times(batch_size: int, video_sigma, audio_sigma, device) -> ModalTimesteps:
        # A video-only Student can still be trained against the multimodal H3
        # Teacher.  When no separate audio schedule is declared, reuse the
        # video sigma so the Teacher request carries a valid audio timestep;
        # the configured audio loss weight still controls whether audio is
        # optimized or merely transported.
        effective_audio_sigma = audio_sigma if audio_sigma is not None else video_sigma
        return ModalTimesteps(
            video=torch.full((batch_size,), 1.0 - video_sigma, device=device) if video_sigma is not None else None,
            audio=torch.full((batch_size,), 1.0 - effective_audio_sigma, device=device)
            if effective_audio_sigma is not None
            else None,
        )

    @staticmethod
    def _interval(video, audio, index: int, stride: int = 1) -> ModalInterval:
        return ModalInterval(
            video=(video[index], video[index + stride]) if video is not None else None,
            audio=(audio[index], audio[index + stride]) if audio is not None else None,
        )

    def training_step(self, batch: PreparedBatch, iteration: int) -> StepOutput:
        if batch.latents is None or batch.noise is None:
            raise TrainingFailure("invalid_training_config", "progressive distillation requires clean latent and noise")
        student_schedule = self._student_schedule
        teacher_schedule = self._teacher_schedule
        if student_schedule is None or teacher_schedule is None:
            raise TrainingFailure("invalid_training_config", "method was not prepared")
        outer = (iteration - 1) % self.config.stage.student_nfe
        teacher_index = 2 * outer
        sample = batch.latents.video if batch.latents.video is not None else batch.latents.audio
        batch_size = sample.shape[0]
        device = sample.device
        start_timestep = self._times(
            batch_size,
            student_schedule.video_sigmas[outer] if student_schedule.video_sigmas else None,
            student_schedule.audio_sigmas[outer] if student_schedule.audio_sigmas else None,
            device,
        )
        start = self.adapter.add_noise(batch.latents, batch.noise, start_timestep)
        with torch.no_grad():
            first_time = self._times(
                batch_size,
                teacher_schedule.video_sigmas[teacher_index] if teacher_schedule.video_sigmas else None,
                teacher_schedule.audio_sigmas[teacher_index] if teacher_schedule.audio_sigmas else None,
                device,
            )
            first_prediction = self.adapter.predict(self.teacher, start, first_time, batch.conditioning)
            middle = self.adapter.scheduler_step(
                self.teacher,
                start,
                first_prediction,
                self._interval(teacher_schedule.video_sigmas, teacher_schedule.audio_sigmas, teacher_index),
            )
            second_time = self._times(
                batch_size,
                teacher_schedule.video_sigmas[teacher_index + 1] if teacher_schedule.video_sigmas else None,
                teacher_schedule.audio_sigmas[teacher_index + 1] if teacher_schedule.audio_sigmas else None,
                device,
            )
            second_prediction = self.adapter.predict(self.teacher, middle, second_time, batch.conditioning)
            target = self.adapter.scheduler_step(
                self.teacher,
                middle,
                second_prediction,
                self._interval(teacher_schedule.video_sigmas, teacher_schedule.audio_sigmas, teacher_index + 1),
            )
        student_prediction = self.adapter.predict(self.student, start, start_timestep, batch.conditioning)
        endpoint = self.adapter.scheduler_step(
            self.student,
            start,
            student_prediction,
            self._interval(student_schedule.video_sigmas, student_schedule.audio_sigmas, outer),
        )
        losses = []
        modality_losses = {}
        for name, actual, expected, weight in (
            ("video", endpoint.video, target.video, self.config.video_weight),
            ("audio", endpoint.audio, target.audio, self.config.audio_weight),
        ):
            if actual is None:
                if weight != 0:
                    raise TrainingFailure(
                        "invalid_training_config", f"absent {name} modality must have zero weight"
                    )
                continue
            if weight <= 0:
                raise TrainingFailure("invalid_training_config", f"present {name} modality has zero weight")
            value = torch.mean((actual - expected) ** 2)
            modality_losses[f"{name}_loss"] = value
            losses.append(weight * value)
        total = sum(losses)
        return StepOutput(
            losses={"total_loss": total, **modality_losses},
            metrics={"interval_index": float(outer)},
            detached={"teacher_endpoint": target},
        )

    def optimizers(self, iteration: int):
        if self._optimizer is None:
            raise TrainingFailure("invalid_training_config", "method was not prepared")
        return {"student": self._optimizer}

    def optimizer_map(self):
        return self.optimizers(1)

    def grad_clip_targets(self, iteration: int):
        return {"student": self.student_model}
