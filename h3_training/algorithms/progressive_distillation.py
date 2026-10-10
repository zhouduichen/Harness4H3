"""Binary progressive distillation over adapter-owned modality schedules."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import nn

from h3_training.adapters.base import DenoisingModelAdapter
from h3_training.algorithms.base import ModelRole, StepOutput, TrainingMethod
from h3_training.data.schema import (
    ModalInterval,
    ModalLatents,
    ModalPrediction,
    ModalTimesteps,
    PreparedBatch,
)
from h3_training.engine.state import TrainingFailure


class _FP32StateAdamW(torch.optim.Optimizer):
    """AdamW with FP32 moments while keeping Student parameters in BF16."""

    def __init__(self, parameters, *, lr: float, weight_decay: float = 0.0):
        if lr <= 0 or weight_decay < 0:
            raise ValueError("optimizer values are invalid")
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": float(weight_decay),
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError("_FP32StateAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["step"] += 1
                step = int(state["step"])
                gradient = gradient.float()
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                update = exp_avg / (exp_avg_sq.sqrt() / bias_correction2**0.5 + eps)
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr / bias_correction1)
        return loss


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
    endpoint_weight: float = 1.0
    velocity_weight: float = 1.0
    clean_anchor_weight: float = 1.0
    rollout_steps: int = 0
    rollout_every: int = 1
    rollout_endpoint_weight: float = 0.0
    rollout_clean_weight: float = 0.0
    fp32_optimizer_state: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are invalid")
        if (
            self.video_weight < 0
            or self.audio_weight < 0
            or self.endpoint_weight < 0
            or self.video_weight + self.audio_weight <= 0
            or self.velocity_weight < 0
            or self.clean_anchor_weight < 0
        ):
            raise ValueError("at least one modality weight must be positive")
        if self.rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative")
        if self.rollout_every <= 0:
            raise ValueError("rollout_every must be positive")
        if self.rollout_endpoint_weight < 0 or self.rollout_clean_weight < 0:
            raise ValueError("rollout weights must be non-negative")
        if self.rollout_steps == 0 and (
            self.rollout_endpoint_weight != 0 or self.rollout_clean_weight != 0
        ):
            raise ValueError("disabled rollout must have zero rollout weights")
        if self.rollout_steps > 0 and (
            self.rollout_endpoint_weight + self.rollout_clean_weight <= 0
        ):
            raise ValueError("active rollout requires a positive rollout weight")
        if (
            self.endpoint_weight
            + self.velocity_weight
            + self.clean_anchor_weight
            + self.rollout_endpoint_weight
            + self.rollout_clean_weight
            <= 0
        ):
            raise ValueError("at least one distillation loss weight must be positive")


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
            optimizer_type = _FP32StateAdamW if self.config.fp32_optimizer_state else torch.optim.AdamW
            self._optimizer = optimizer_type(
                parameters, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )

    def prepare_batch(self, raw, generator: torch.Generator) -> PreparedBatch:
        return self.adapter.prepare_batch(raw, generator)

    @staticmethod
    def _mse(actual: ModalLatents, expected: ModalLatents, video_weight: float, audio_weight: float):
        terms = []
        if actual.video is not None:
            if expected.video is None:
                raise TrainingFailure("invalid_training_config", "video velocity target is missing")
            # Student and real H3 tensors are normally BF16.  Accumulating
            # the reduction in BF16 quantizes the loss (and its gradient
            # scale) too aggressively for a billion-parameter Student.
            terms.append(
                video_weight
                * torch.mean((actual.video.float() - expected.video.float()) ** 2)
            )
        if actual.audio is not None:
            if expected.audio is None:
                raise TrainingFailure("invalid_training_config", "audio velocity target is missing")
            terms.append(
                audio_weight
                * torch.mean((actual.audio.float() - expected.audio.float()) ** 2)
            )
        return sum(terms)

    @staticmethod
    def _times(batch_size: int, video_sigma, audio_sigma, device) -> ModalTimesteps:
        # A video-only Student can still be trained against the multimodal H3
        # Teacher.  When no separate audio schedule is declared, reuse the
        # video sigma so the Teacher request carries a valid audio timestep;
        # the configured audio loss weight still controls whether audio is
        # optimized or merely transported.
        effective_audio_sigma = audio_sigma if audio_sigma is not None else video_sigma
        # Adapter schedules are H3 sigmas: 1 is noise and 0 is clean.  Do
        # not convert them to the legacy TinyH3 time convention here; the
        # real Teacher and Student production adapter consume sigma directly.
        return ModalTimesteps(
            video=torch.full((batch_size,), video_sigma, device=device) if video_sigma is not None else None,
            audio=torch.full((batch_size,), effective_audio_sigma, device=device)
            if effective_audio_sigma is not None
            else None,
        )

    @staticmethod
    def _interval(video, audio, index: int, stride: int = 1) -> ModalInterval:
        return ModalInterval(
            video=(video[index], video[index + stride]) if video is not None else None,
            audio=(audio[index], audio[index + stride]) if audio is not None else None,
        )

    def _integrate_rollout(
        self,
        role: ModelRole,
        latent: ModalLatents,
        conditioning,
        schedule,
        start_index: int,
        steps: int,
    ) -> Tuple[ModalLatents, Optional[ModalPrediction]]:
        """Integrate one role continuously over adjacent schedule intervals."""

        state = latent
        last_prediction: Optional[ModalPrediction] = None
        sample = state.video if state.video is not None else state.audio
        assert sample is not None
        self.adapter.begin_trajectory(role, schedule, int(start_index), sample.device)
        for offset in range(int(steps)):
            index = int(start_index) + offset
            sample = state.video if state.video is not None else state.audio
            assert sample is not None
            timestep = self._times(
                sample.shape[0],
                schedule.video_sigmas[index] if schedule.video_sigmas else None,
                schedule.audio_sigmas[index] if schedule.audio_sigmas else None,
                sample.device,
            )
            last_prediction = self.adapter.predict(role, state, timestep, conditioning)
            state = self.adapter.scheduler_step(
                role,
                state,
                last_prediction,
                self._interval(schedule.video_sigmas, schedule.audio_sigmas, index),
            )
        return state, last_prediction

    def _rollout_losses(
        self,
        batch: PreparedBatch,
        start: ModalLatents,
        outer: int,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Match a continuous Student rollout to a continuous Teacher rollout."""

        sample = start.video if start.video is not None else start.audio
        assert sample is not None
        zero = sample.new_zeros(())
        if self.config.rollout_steps <= 0 or ((iteration - 1) % self.config.rollout_every) != 0:
            return zero, zero, 0
        if self._student_schedule is None or self._teacher_schedule is None:
            raise TrainingFailure("invalid_training_config", "rollout schedules are unavailable")
        remaining = self.config.stage.student_nfe - outer
        steps = min(self.config.rollout_steps, remaining)
        if steps <= 0:
            raise TrainingFailure("invalid_training_config", "rollout has no remaining student interval")

        # The Teacher starts from exactly the same noisy state as the Student.
        # Only the Teacher trajectory is detached; the Student trajectory must
        # retain its graph so rollout errors train the Student parameters.
        with torch.no_grad():
            teacher_endpoint, _ = self._integrate_rollout(
                self.teacher,
                start,
                batch.conditioning,
                self._teacher_schedule,
                2 * outer,
                2 * steps,
            )
        student_endpoint, _ = self._integrate_rollout(
            self.student,
            start,
            batch.conditioning,
            self._student_schedule,
            outer,
            steps,
        )
        endpoint_loss = self._mse(
            student_endpoint,
            teacher_endpoint,
            self.config.video_weight,
            self.config.audio_weight,
        )

        endpoint_index = outer + steps
        endpoint_sample = student_endpoint.video if student_endpoint.video is not None else student_endpoint.audio
        assert endpoint_sample is not None
        endpoint_time = self._times(
            endpoint_sample.shape[0],
            self._student_schedule.video_sigmas[endpoint_index]
            if self._student_schedule.video_sigmas
            else None,
            self._student_schedule.audio_sigmas[endpoint_index]
            if self._student_schedule.audio_sigmas
            else None,
            endpoint_sample.device,
        )
        if endpoint_index == self.config.stage.student_nfe:
            clean_hat = student_endpoint
        else:
            endpoint_prediction = self.adapter.predict(
                self.student,
                student_endpoint,
                endpoint_time,
                batch.conditioning,
            )
            clean_hat = self.adapter.prediction_to_clean(
                student_endpoint,
                endpoint_prediction,
                endpoint_time,
            )
        clean_loss = self._mse(
            clean_hat,
            batch.latents,
            self.config.video_weight,
            self.config.audio_weight,
        )
        return endpoint_loss, clean_loss, steps

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
        native_trajectory = bool(getattr(self.adapter, "native_trajectory_training", False))
        if native_trajectory and self.config.rollout_steps > 0:
            raise TrainingFailure(
                "invalid_training_config",
                "native multistep trajectory training does not support rollout_steps yet",
            )
        if native_trajectory:
            # UniPC is history-dependent.  Start both roles at the native
            # noise endpoint and replay the prefix so the multistep history
            # at `outer` matches the real Wan inference loop.  Prefixes are
            # deliberately no-grad: only the current Student prediction is
            # optimized, while prior state/history remains an exact detached
            # solver context without retaining 1B-model activations.
            initial_time = self._times(
                batch_size,
                student_schedule.video_sigmas[0] if student_schedule.video_sigmas else None,
                student_schedule.audio_sigmas[0] if student_schedule.audio_sigmas else None,
                device,
            )
            trajectory_start = self.adapter.add_noise(batch.latents, batch.noise, initial_time)
            with torch.no_grad():
                self.adapter.begin_trajectory(self.teacher, teacher_schedule, 0, device)
                teacher_state = trajectory_start
                teacher_velocity_target = None
                for index in range(teacher_index + 2):
                    teacher_sample = teacher_state.video if teacher_state.video is not None else teacher_state.audio
                    assert teacher_sample is not None
                    teacher_time = self._times(
                        teacher_sample.shape[0],
                        teacher_schedule.video_sigmas[index] if teacher_schedule.video_sigmas else None,
                        teacher_schedule.audio_sigmas[index] if teacher_schedule.audio_sigmas else None,
                        device,
                    )
                    teacher_prediction = self.adapter.predict(
                        self.teacher, teacher_state, teacher_time, batch.conditioning
                    )
                    if index == teacher_index:
                        teacher_velocity_target = teacher_prediction
                    teacher_state = self.adapter.scheduler_step(
                        self.teacher,
                        teacher_state,
                        teacher_prediction,
                        self._interval(teacher_schedule.video_sigmas, teacher_schedule.audio_sigmas, index),
                    )
                if teacher_velocity_target is None:
                    raise TrainingFailure("invalid_training_config", "native Teacher trajectory missed current interval")
                target = teacher_state
            self.adapter.begin_trajectory(self.student, student_schedule, 0, device)
            student_input = trajectory_start
            with torch.no_grad():
                for index in range(outer):
                    student_sample = student_input.video if student_input.video is not None else student_input.audio
                    assert student_sample is not None
                    prefix_time = self._times(
                        student_sample.shape[0],
                        student_schedule.video_sigmas[index] if student_schedule.video_sigmas else None,
                        student_schedule.audio_sigmas[index] if student_schedule.audio_sigmas else None,
                        device,
                    )
                    prefix_prediction = self.adapter.predict(
                        self.student, student_input, prefix_time, batch.conditioning
                    )
                    student_input = self.adapter.scheduler_step(
                        self.student,
                        student_input,
                        prefix_prediction,
                        self._interval(student_schedule.video_sigmas, student_schedule.audio_sigmas, index),
                    )
            student_timestep = self._times(
                batch_size,
                student_schedule.video_sigmas[outer] if student_schedule.video_sigmas else None,
                student_schedule.audio_sigmas[outer] if student_schedule.audio_sigmas else None,
                device,
            )
        else:
            student_timestep = self._times(
                batch_size,
                student_schedule.video_sigmas[outer] if student_schedule.video_sigmas else None,
                student_schedule.audio_sigmas[outer] if student_schedule.audio_sigmas else None,
                device,
            )
            student_input = self.adapter.add_noise(batch.latents, batch.noise, student_timestep)
            with torch.no_grad():
                self.adapter.begin_trajectory(self.teacher, teacher_schedule, teacher_index, device)
                first_time = self._times(
                    batch_size,
                    teacher_schedule.video_sigmas[teacher_index] if teacher_schedule.video_sigmas else None,
                    teacher_schedule.audio_sigmas[teacher_index] if teacher_schedule.audio_sigmas else None,
                    device,
                )
                first_prediction = self.adapter.predict(self.teacher, student_input, first_time, batch.conditioning)
                teacher_velocity_target = first_prediction
                middle = self.adapter.scheduler_step(
                    self.teacher,
                    student_input,
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
            self.adapter.begin_trajectory(self.student, student_schedule, outer, device)
        student_prediction = self.adapter.predict(self.student, student_input, student_timestep, batch.conditioning)
        endpoint = self.adapter.scheduler_step(
            self.student,
            student_input,
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
            value = torch.mean((actual.float() - expected.float()) ** 2)
            modality_losses[f"{name}_loss"] = value
            losses.append(weight * value)
        velocity_loss = self._mse(
            student_prediction,
            ModalLatents(video=teacher_velocity_target.video, audio=teacher_velocity_target.audio),
            self.config.video_weight,
            self.config.audio_weight,
        )
        modality_losses["velocity_loss"] = velocity_loss
        clean_anchor = self.adapter.prediction_to_clean(student_input, student_prediction, student_timestep)
        clean_anchor_loss = self._mse(
            clean_anchor,
            ModalLatents(video=batch.latents.video, audio=batch.latents.audio),
            self.config.video_weight,
            self.config.audio_weight,
        )
        modality_losses["clean_anchor_loss"] = clean_anchor_loss
        endpoint_loss = sum(losses)
        # Stable names make the four learnability signals auditable without
        # changing the historical per-modality loss keys.
        modality_losses["endpoint_loss"] = endpoint_loss
        modality_losses["flow_loss"] = velocity_loss
        modality_losses["clean_loss"] = clean_anchor_loss
        rollout_endpoint_loss, rollout_clean_loss, effective_rollout_steps = self._rollout_losses(
            batch,
            student_input,
            outer,
            iteration,
        )
        total = (
            self.config.endpoint_weight * sum(losses)
            + self.config.velocity_weight * velocity_loss
            + self.config.clean_anchor_weight * clean_anchor_loss
            + self.config.rollout_endpoint_weight * rollout_endpoint_loss
            + self.config.rollout_clean_weight * rollout_clean_loss
        )
        return StepOutput(
            losses={
                "total_loss": total,
                **modality_losses,
                "rollout_endpoint_loss": rollout_endpoint_loss,
                "rollout_clean_loss": rollout_clean_loss,
            },
            metrics={
                "interval_index": float(outer),
                "rollout_steps": float(effective_rollout_steps),
                "rollout_index": float(outer) if effective_rollout_steps else -1.0,
            },
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
