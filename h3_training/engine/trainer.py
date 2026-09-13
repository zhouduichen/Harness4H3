"""Deterministic optimizer-iteration training engine."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import torch

from h3_training.algorithms.base import StepOutput, TrainingMethod
from h3_training.data.schema import PreparedBatch
from .checkpoint import canonical_digest, load_training_checkpoint, save_training_checkpoint
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


class TrainerEngine:
    def __init__(self, config: TrainerConfig = TrainerConfig()) -> None:
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def run(
        self,
        method: TrainingMethod,
        dataloader: Iterable[Any],
        max_steps: int,
        resume_from: Optional[Path] = None,
    ) -> TrainingRunResult:
        if max_steps <= 0:
            raise TrainingFailure("invalid_training_config", "max_steps must be positive")
        batches = list(dataloader)
        if not batches:
            raise TrainingFailure("invalid_training_config", "dataloader must not be empty")
        method.prepare()
        state = LoopState()
        if resume_from is not None:
            state = load_training_checkpoint(
                resume_from, method, self.config.parent_sha256, self.config.digest, self.generator
            )
        if max_steps < state.global_step:
            raise TrainingFailure("resume_mismatch", "max_steps precedes checkpoint step")
        initial_loss = None
        final_loss = float("nan")
        maximum_gradient_norm = 0.0
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for iteration in range(state.global_step + 1, max_steps + 1):
            scheduled = dict(method.optimizers(iteration))
            if not scheduled:
                raise TrainingFailure("invalid_training_config", "iteration has no scheduled optimizer")
            for optimizer in scheduled.values():
                optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for accumulation_index in range(self.config.gradient_accumulation_steps):
                raw = batches[state.sampler_position % len(batches)]
                state.sampler_position += 1
                batch = method.prepare_batch(raw, self.generator) if hasattr(method, "prepare_batch") else raw
                if not isinstance(batch, PreparedBatch):
                    raise TrainingFailure("invalid_training_config", "training method did not prepare a batch")
                try:
                    output = method.training_step(batch, iteration)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise TrainingFailure("training_oom", str(exc)) from exc
                    raise
                finite_total_loss(output.losses["total_loss"])
                loss = output.losses["total_loss"]
                if initial_loss is None:
                    initial_loss = float(loss.detach())
                accumulated += float(loss.detach())
                try:
                    (loss / self.config.gradient_accumulation_steps).backward()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise TrainingFailure("training_oom", str(exc)) from exc
                    raise
                state.microbatches_consumed += 1
                state.accumulation_position = accumulation_index + 1
            norm = gradient_norm(method.grad_clip_targets(iteration).values(), self.config.max_gradient_norm)
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            for name, optimizer in scheduled.items():
                try:
                    optimizer.step()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise TrainingFailure("training_oom", str(exc)) from exc
                    raise
                state.optimizer_steps[name] = state.optimizer_steps.get(name, 0) + 1
            for scheduler in method.schedulers().values():
                scheduler.step()
            method.on_optimizers_stepped(tuple(scheduled))
            state.global_step = iteration
            state.accumulation_position = 0
            final_loss = accumulated / self.config.gradient_accumulation_steps
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
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
