"""Deterministic optimizer-iteration training engine."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from h3_training.algorithms.base import StepOutput, TrainingMethod
from h3_training.data.schema import PreparedBatch
from .checkpoint import canonical_digest, load_training_checkpoint, save_training_checkpoint
from .evidence import ChildEvidence, ParentEvidence, save_verified_child
from .optimizer import finite_total_loss, gradient_norm
from .state import LoopState, TrainingFailure, TrainingRunResult


@dataclass(frozen=True)
class TrainerConfig:
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    seed: int = 0
    parent_sha256: str = ""
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps <= 0 or self.max_gradient_norm <= 0:
            raise ValueError("gradient accumulation and clipping values must be positive")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


class _BatchCursor:
    """Cycle a re-iterable loader without materializing the whole epoch."""

    def __init__(self, source: Iterable[Any]) -> None:
        self.source = source
        self.iterator = iter(source)
        self.one_shot = self.iterator is source

    def _restart(self) -> None:
        if self.one_shot:
            raise TrainingFailure(
                "invalid_training_config",
                "dataloader must be re-iterable when training needs more than one pass",
            )
        self.iterator = iter(self.source)

    def next(self) -> Any:
        try:
            return next(self.iterator)
        except StopIteration:
            self._restart()
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise TrainingFailure("invalid_training_config", "dataloader must not be empty") from exc

    def skip(self, count: int) -> None:
        if count < 0:
            raise TrainingFailure("resume_mismatch", "sampler position must not be negative")
        if count and self.one_shot:
            raise TrainingFailure(
                "resume_mismatch",
                "exact resume requires a re-iterable dataloader",
            )
        for _ in range(count):
            self.next()


class TrainerEngine:
    def __init__(self, config: TrainerConfig = TrainerConfig()) -> None:
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def backward(self, losses: Mapping[str, torch.Tensor], accumulation_steps: int) -> None:
        if accumulation_steps <= 0:
            raise TrainingFailure("invalid_training_config", "accumulation_steps must be positive")
        if "total_loss" not in losses:
            raise TrainingFailure("invalid_training_config", "losses must contain total_loss")
        for name, loss in losses.items():
            try:
                finite_total_loss(loss)
            except TrainingFailure as exc:
                raise TrainingFailure(exc.code, f"{name}: {exc.detail}") from exc
        try:
            (losses["total_loss"] / accumulation_steps).backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise TrainingFailure("training_oom", str(exc)) from exc
            raise

    def optimizer_step(self, method: TrainingMethod, iteration: int) -> Mapping[str, float]:
        scheduled = dict(method.optimizers(iteration))
        if not scheduled:
            raise TrainingFailure("invalid_training_config", "iteration has no scheduled optimizer")
        norm = gradient_norm(method.grad_clip_targets(iteration).values(), self.config.max_gradient_norm)
        for optimizer in scheduled.values():
            try:
                optimizer.step()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise TrainingFailure("training_oom", str(exc)) from exc
                raise
        for name, scheduler in method.schedulers().items():
            if name in scheduled:
                scheduler.step()
        method.on_optimizers_stepped(tuple(scheduled))
        return {"gradient_norm": norm, **{f"optimizer_step/{name}": 1.0 for name in scheduled}}

    @staticmethod
    def _move_batch(batch: PreparedBatch, device: torch.device, dtype: Optional[torch.dtype]) -> PreparedBatch:
        def move(value):
            if value is None:
                return None
            target_dtype = dtype if dtype is not None and value.is_floating_point() else value.dtype
            return value.to(device=device, dtype=target_dtype)

        def move_modal(value):
            if value is None:
                return None
            return type(value)(video=move(value.video), audio=move(value.audio))

        conditioning = type(batch.conditioning)(
            text=move(batch.conditioning.text),
            negative_text=move(batch.conditioning.negative_text),
        )
        return PreparedBatch(
            conditioning=conditioning,
            latents=move_modal(batch.latents),
            noise=move_modal(batch.noise),
            timesteps=(
                type(batch.timesteps)(video=move(batch.timesteps.video), audio=move(batch.timesteps.audio))
                if batch.timesteps is not None
                else None
            ),
            sample_ids=batch.sample_ids,
            metadata=batch.metadata,
        )


    def run(
        self,
        method: TrainingMethod,
        dataloader: Iterable[Any],
        max_steps: int,
        resume_from: Optional[Path] = None,
    ) -> TrainingRunResult:
        if max_steps <= 0:
            raise TrainingFailure("invalid_training_config", "max_steps must be positive")
        try:
            device = torch.device(self.config.device)
        except (TypeError, RuntimeError) as exc:
            raise TrainingFailure("invalid_training_config", f"invalid device: {self.config.device}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise TrainingFailure("device_unavailable", f"CUDA is not available for device {self.config.device}")
        method.to(device)
        method.prepare()
        state = LoopState()
        if resume_from is not None:
            state = load_training_checkpoint(
                resume_from, method, self.config.parent_sha256, self.config.digest, self.generator
            )
        if max_steps < state.global_step:
            raise TrainingFailure("resume_mismatch", "max_steps precedes checkpoint step")
        cursor = _BatchCursor(dataloader)
        cursor.skip(state.sampler_position)
        parameter_dtype = next(
            (parameter.dtype for parameter in method.parameters() if parameter.is_floating_point()), None
        )
        initial_loss = None
        final_loss = float("nan")
        maximum_gradient_norm = 0.0
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(state.global_step + 1, max_steps + 1):
            scheduled = dict(method.optimizers(iteration))
            if not scheduled:
                raise TrainingFailure("invalid_training_config", "iteration has no scheduled optimizer")
            for optimizer in scheduled.values():
                optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for accumulation_index in range(self.config.gradient_accumulation_steps):
                raw = cursor.next()
                state.sampler_position += 1
                batch = method.prepare_batch(raw, self.generator) if hasattr(method, "prepare_batch") else raw
                if not isinstance(batch, PreparedBatch):
                    raise TrainingFailure("invalid_training_config", "training method did not prepare a batch")
                batch = self._move_batch(batch, device, parameter_dtype)
                try:
                    output = method.training_step(batch, iteration)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise TrainingFailure("training_oom", str(exc)) from exc
                    raise
                loss = output.losses["total_loss"]
                if initial_loss is None:
                    initial_loss = float(loss.detach())
                accumulated += float(loss.detach())
                self.backward(output.losses, self.config.gradient_accumulation_steps)
                state.microbatches_consumed += 1
                state.accumulation_position = accumulation_index + 1
            step_metrics = self.optimizer_step(method, iteration)
            norm = step_metrics["gradient_norm"]
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            for name in scheduled:
                state.optimizer_steps[name] = state.optimizer_steps.get(name, 0) + 1
            state.global_step = iteration
            state.accumulation_position = 0
            final_loss = accumulated / self.config.gradient_accumulation_steps
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        return TrainingRunResult(
            loop_state=state,
            initial_loss=float(initial_loss if initial_loss is not None else final_loss),
            final_loss=final_loss,
            max_gradient_norm=maximum_gradient_norm,
            optimizer_steps=dict(state.optimizer_steps),
            wall_time_s=time.perf_counter() - started,
            peak_memory_bytes=peak,
        )

    def save_checkpoint(self, method: TrainingMethod, loop_state: LoopState, path: Path) -> Path:
        return save_training_checkpoint(
            path, method, loop_state, self.config.parent_sha256, self.config.digest, self.generator
        )

    def save_child(
        self,
        method: TrainingMethod,
        parent: ParentEvidence,
        path: Path,
    ) -> ChildEvidence:
        role = getattr(method, "student", None)
        if role is None:
            raise TrainingFailure("invalid_training_config", "method has no student role")
        return save_verified_child(role, parent, path)
