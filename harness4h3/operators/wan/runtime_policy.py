"""Validated runtime policies shared by Fake and Diffusers Wan adapters."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


def _ratio(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("%s must be numeric" % name)
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise ValueError("%s must be in (0, 1]" % name)
    return result


@dataclass(frozen=True)
class TokenSparsityPolicy:
    keep_ratio: float
    block_ids: tuple[str, ...] = ()
    score: str = "l2_norm"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TokenSparsityPolicy":
        block_ids = raw.get("block_ids", [])
        if not isinstance(block_ids, (list, tuple)) or not all(isinstance(item, str) for item in block_ids):
            raise ValueError("token_sparsity.block_ids must be a list of strings")
        score = str(raw.get("score", "l2_norm"))
        if score not in {"l2_norm", "mean_abs"}:
            raise ValueError("unsupported token sparsity score")
        return cls(_ratio(raw.get("keep_ratio"), "token_sparsity.keep_ratio"), tuple(block_ids), score)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "block_ids": list(self.block_ids)}


@dataclass(frozen=True)
class SparseRuntimePolicy:
    keep_ratio: float
    attention: str = "attn1"
    score: str = "l2_norm"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SparseRuntimePolicy":
        attention = str(raw.get("attention", "attn1"))
        if attention not in {"attn1", "self", "all"}:
            raise ValueError("unsupported sparse attention target")
        score = str(raw.get("score", "l2_norm"))
        if score not in {"l2_norm", "mean_abs"}:
            raise ValueError("unsupported sparse runtime score")
        return cls(_ratio(raw.get("keep_ratio"), "sparse_runtime.keep_ratio"), attention, score)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionWindowPolicy:
    """Local Self-Attention policy with a bounded per-query K/V window."""

    window_size: int
    attention: str = "attn1"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AttentionWindowPolicy":
        value = raw.get("window_size")
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= int(value) <= 4096:
            raise ValueError("attention_window.window_size must be an integer in [1, 4096]")
        attention = str(raw.get("attention", "attn1"))
        if attention not in {"attn1", "self", "all"}:
            raise ValueError("unsupported attention window target")
        return cls(int(value), attention)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplingStepPolicy:
    target_steps: int
    strategy: str = "uniform"
    min_steps: int = 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SamplingStepPolicy":
        value = raw.get("target_steps")
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= int(value) <= 256:
            raise ValueError("sampling_step_skip.target_steps must be an integer in [1, 256]")
        strategy = str(raw.get("strategy", "uniform"))
        if strategy not in {"uniform", "leading", "trailing"}:
            raise ValueError("unsupported sampling step strategy")
        minimum = raw.get("min_steps", 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or not 1 <= int(minimum) <= int(value):
            raise ValueError("sampling_step_skip.min_steps must be in [1, target_steps]")
        return cls(int(value), strategy, int(minimum))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_sampling_steps(timesteps: Sequence[Any], policy: Mapping[str, Any]) -> tuple[Any, ...]:
    """Select an actual denoising schedule, preserving endpoints when possible."""

    parsed = SamplingStepPolicy.from_mapping(policy)
    values = tuple(timesteps)
    if not values:
        return ()
    count = min(len(values), parsed.target_steps)
    count = max(parsed.min_steps, count)
    if count >= len(values):
        return values
    if parsed.strategy == "leading":
        indices = tuple(range(count))
    elif parsed.strategy == "trailing":
        indices = tuple(range(len(values) - count, len(values)))
    else:
        if count == 1:
            indices = (0,)
        else:
            indices = tuple(round(index * (len(values) - 1) / (count - 1)) for index in range(count))
    return tuple(values[index] for index in indices)
