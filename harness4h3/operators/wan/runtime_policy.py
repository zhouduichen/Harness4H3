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
    grid_shape: tuple[int, int, int] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TokenSparsityPolicy":
        block_ids = raw.get("block_ids", [])
        if not isinstance(block_ids, (list, tuple)) or not all(isinstance(item, str) for item in block_ids):
            raise ValueError("token_sparsity.block_ids must be a list of strings")
        score = str(raw.get("score", "l2_norm"))
        if score not in {"l2_norm", "mean_abs"}:
            raise ValueError("unsupported token sparsity score")
        grid_shape = raw.get("grid_shape", ())
        if grid_shape in (None, []):
            grid_shape = ()
        if not isinstance(grid_shape, (list, tuple)) or len(grid_shape) not in {0, 3} or any(
            isinstance(item, bool) or not isinstance(item, int) or int(item) <= 0 for item in grid_shape
        ):
            raise ValueError("token_sparsity.grid_shape must be three positive integers")
        return cls(_ratio(raw.get("keep_ratio"), "token_sparsity.keep_ratio"), tuple(block_ids), score, tuple(int(item) for item in grid_shape))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "block_ids": list(self.block_ids), "grid_shape": list(self.grid_shape)}


@dataclass(frozen=True)
class TokenMergePolicy:
    """Self-attention token merging policy with an explicit unmerge map."""

    keep_ratio: float
    block_ids: tuple[str, ...] = ()
    mode: str = "adjacent"
    grid_shape: tuple[int, int, int] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TokenMergePolicy":
        block_ids = raw.get("block_ids", [])
        if not isinstance(block_ids, (list, tuple)) or not all(isinstance(item, str) for item in block_ids):
            raise ValueError("token_merge.block_ids must be a list of strings")
        mode = str(raw.get("mode", "adjacent"))
        if mode not in {"adjacent", "spatiotemporal"}:
            raise ValueError("unsupported token merge mode")
        grid_shape = raw.get("grid_shape", ())
        if grid_shape in (None, []):
            grid_shape = ()
        if not isinstance(grid_shape, (list, tuple)) or len(grid_shape) not in {0, 3} or any(isinstance(item, bool) or not isinstance(item, int) or int(item) <= 0 for item in grid_shape):
            raise ValueError("token_merge.grid_shape must be three positive integers")
        if mode == "spatiotemporal" and len(grid_shape) != 3:
            raise ValueError("spatiotemporal token merge requires grid_shape")
        return cls(_ratio(raw.get("keep_ratio"), "token_merge.keep_ratio"), tuple(block_ids), mode, tuple(int(item) for item in grid_shape))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "block_ids": list(self.block_ids), "grid_shape": list(self.grid_shape)}


@dataclass(frozen=True)
class KVTokenSelectionPolicy:
    """Keep full queries while selecting a smaller Self-Attention K/V set."""

    keep_ratio: float
    block_ids: tuple[str, ...] = ()
    score: str = "l2_norm"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "KVTokenSelectionPolicy":
        block_ids = raw.get("block_ids", [])
        if not isinstance(block_ids, (list, tuple)) or not all(isinstance(item, str) for item in block_ids):
            raise ValueError("kv_token_select.block_ids must be a list of strings")
        score = str(raw.get("score", "l2_norm"))
        if score not in {"l2_norm", "mean_abs"}:
            raise ValueError("unsupported KV token selection score")
        return cls(_ratio(raw.get("keep_ratio"), "kv_token_select.keep_ratio"), tuple(block_ids), score)

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
class HeadGatingRule:
    """One timestep-conditioned retained-head rule.

    ``head_indices`` are retained indices, rather than removed indices.  This
    makes the runtime executor explicit about the work it is allowed to
    perform and avoids the common ``compute-all-then-zero`` fake gate.
    """

    min_timestep: float
    max_timestep: float
    keep_ratio: float | None = None
    head_indices: tuple[int, ...] = ()
    score: str = "weight_l2"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HeadGatingRule":
        if not isinstance(raw, Mapping):
            raise ValueError("head_gating_rule_schema_invalid")
        required = {"min_timestep", "max_timestep"}
        if not required <= set(raw):
            raise ValueError("head_gating_rule_schema_invalid")
        try:
            minimum = float(raw["min_timestep"])
            maximum = float(raw["max_timestep"])
        except (TypeError, ValueError) as exc:
            raise ValueError("head_gating_bounds_invalid") from exc
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError("head_gating_bounds_invalid")
        raw_indices = raw.get("head_indices", [])
        if not isinstance(raw_indices, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or int(item) < 0 for item in raw_indices
        ):
            raise ValueError("head_gating_head_indices_invalid")
        indices = tuple(dict.fromkeys(int(item) for item in raw_indices))
        ratio = raw.get("keep_ratio")
        parsed_ratio = None if ratio is None else _ratio(ratio, "head_gating.keep_ratio")
        if not indices and parsed_ratio is None:
            raise ValueError("head_gating_requires_indices_or_ratio")
        score = str(raw.get("score", "weight_l2"))
        if score not in {"weight_l2", "activation_l2", "activation_abs", "mean_abs", "l2_norm"}:
            raise ValueError("unsupported head gating score")
        return cls(minimum, maximum, parsed_ratio, indices, score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_timestep": float(self.min_timestep),
            "max_timestep": float(self.max_timestep),
            "keep_ratio": None if self.keep_ratio is None else float(self.keep_ratio),
            "head_indices": list(self.head_indices),
            "score": str(self.score),
        }


@dataclass(frozen=True)
class HeadGatingPolicy:
    """Dynamic head selection policy for real self-attention execution."""

    attention: str = "attn1"
    block_ids: tuple[str, ...] = ()
    keep_ratio: float | None = None
    head_indices: tuple[int, ...] = ()
    score: str = "weight_l2"
    rules: tuple[HeadGatingRule, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HeadGatingPolicy":
        if not isinstance(raw, Mapping):
            raise ValueError("head_gating_policy_schema_invalid")
        attention = str(raw.get("attention", "attn1"))
        if attention not in {"attn1", "self"}:
            raise ValueError("head_gating supports self-attention only")
        block_ids = raw.get("block_ids", [])
        if not isinstance(block_ids, (list, tuple)) or not all(isinstance(item, str) and item for item in block_ids):
            raise ValueError("head_gating.block_ids must be a list of strings")
        raw_indices = raw.get("head_indices", [])
        if not isinstance(raw_indices, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or int(item) < 0 for item in raw_indices
        ):
            raise ValueError("head_gating.head_indices must be a list of non-negative integers")
        indices = tuple(dict.fromkeys(int(item) for item in raw_indices))
        ratio = raw.get("keep_ratio")
        parsed_ratio = None if ratio is None else _ratio(ratio, "head_gating.keep_ratio")
        raw_rules = raw.get("rules", [])
        if not isinstance(raw_rules, (list, tuple)):
            raise ValueError("head_gating.rules must be a list")
        rules = tuple(HeadGatingRule.from_mapping(item) for item in raw_rules)
        if not indices and parsed_ratio is None and not rules:
            raise ValueError("head_gating_requires_indices_ratio_or_rules")
        score = str(raw.get("score", "weight_l2"))
        if score not in {"weight_l2", "activation_l2", "activation_abs", "mean_abs", "l2_norm"}:
            raise ValueError("unsupported head gating score")
        for left_index, left in enumerate(rules):
            for right in rules[left_index + 1 :]:
                if max(left.min_timestep, right.min_timestep) <= min(left.max_timestep, right.max_timestep):
                    raise ValueError("head_gating_overlapping_rules")
        return cls(attention, tuple(dict.fromkeys(str(item) for item in block_ids)), parsed_ratio, indices, score, rules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention": str(self.attention),
            "block_ids": list(self.block_ids),
            "keep_ratio": None if self.keep_ratio is None else float(self.keep_ratio),
            "head_indices": list(self.head_indices),
            "score": str(self.score),
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass(frozen=True)
class AttentionWindowPolicy:
    """Local Self-Attention policy with a bounded per-query K/V window."""

    window_size: int
    attention: str = "attn1"
    grid_shape: tuple[int, int, int] = ()
    window_radius: tuple[int, int, int] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AttentionWindowPolicy":
        value = raw.get("window_size")
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= int(value) <= 4096:
            raise ValueError("attention_window.window_size must be an integer in [1, 4096]")
        attention = str(raw.get("attention", "attn1"))
        if attention not in {"attn1", "self"}:
            raise ValueError("attention window supports self-attention only")
        grid_shape = raw.get("grid_shape", ())
        window_radius = raw.get("window_radius", ())
        if grid_shape in (None, []):
            grid_shape = ()
        if window_radius in (None, []):
            window_radius = ()
        for name, shape in (("grid_shape", grid_shape), ("window_radius", window_radius)):
            minimum = 1 if name == "grid_shape" else 0
            if not isinstance(shape, (list, tuple)) or len(shape) not in {0, 3} or any(isinstance(item, bool) or not isinstance(item, int) or int(item) < minimum for item in shape):
                raise ValueError("attention_window.%s must contain three positive integers" % name)
        if bool(grid_shape) != bool(window_radius):
            raise ValueError("3-D attention window requires grid_shape and window_radius together")
        return cls(int(value), attention, tuple(int(item) for item in grid_shape), tuple(int(item) for item in window_radius))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "grid_shape": list(self.grid_shape), "window_radius": list(self.window_radius)}


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
