from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ...adapters.wan.contracts import CheckpointArtifact, ChannelPruningConfig, CompressionResult, WanModelAdapter


class CompressionOperator(Protocol):
    name: str

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        ...


def _failed(parent: CheckpointArtifact, failure_type: str, message: str, started: float) -> CompressionResult:
    return CompressionResult("failed", parent.sha256, None, failure_type=failure_type, message=message, wall_time_s=time.monotonic() - started)


class BlockPruningOperator:
    name = "prune_blocks"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        try:
            raw = plan.get("block_ids")
            if not isinstance(raw, (list, tuple)) or not all(isinstance(item, str) for item in raw):
                return _failed(parent, "invalid_block_selection", "block_ids must be a list of strings", started)
            model = adapter.reload_model(parent)
            change = adapter.remove_blocks(model, tuple(raw))
            child = adapter.save_model(model, Path(output_dir))
            if child.sha256 == parent.sha256:
                return _failed(parent, "checkpoint_not_changed", "block pruning did not change checkpoint", started)
            return CompressionResult("success", parent.sha256, child, {"kind": change.kind, **dict(change.details)}, wall_time_s=time.monotonic() - started)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return _failed(parent, _failure_type(exc, "block_pruning_failed"), str(exc), started)


class FFNChannelPruningOperator:
    name = "prune_ffn_channels"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        try:
            block_ids = plan.get("block_ids")
            retained = plan.get("retained_indices")
            if not isinstance(block_ids, (list, tuple)) or not all(isinstance(item, str) for item in block_ids):
                return _failed(parent, "invalid_channel_selection", "block_ids must be a list of strings", started)
            if not isinstance(retained, (list, tuple)) or not all(isinstance(item, int) and not isinstance(item, bool) for item in retained):
                return _failed(parent, "invalid_channel_selection", "retained_indices must be a list of integers", started)
            if not retained:
                return _failed(parent, "invalid_channel_selection", "retained_indices must not be empty", started)
            config = ChannelPruningConfig(tuple(block_ids), tuple(retained), str(plan.get("projection_name", "ffn")))
            model = adapter.reload_model(parent)
            change = adapter.prune_channels(model, config)
            child = adapter.save_model(model, Path(output_dir))
            if child.sha256 == parent.sha256:
                return _failed(parent, "checkpoint_not_changed", "channel pruning did not change checkpoint", started)
            return CompressionResult("success", parent.sha256, child, {"kind": change.kind, **dict(change.details)}, wall_time_s=time.monotonic() - started)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return _failed(parent, _failure_type(exc, "channel_pruning_failed"), str(exc), started)


class CompositeCompressionOperator:
    """Execute a short, ordered program of registered base operators."""

    name = "compose"
    _BASE_OPERATORS = {
        "prune_blocks": BlockPruningOperator,
        "prune_ffn_channels": FFNChannelPruningOperator,
    }

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        steps = plan.get("steps")
        if not isinstance(steps, (list, tuple)) or not 2 <= len(steps) <= 3:
            return _failed(parent, "invalid_composite_plan", "compose requires 2 to 3 steps", started)
        current = parent
        changes: list[Mapping[str, Any]] = []
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping) or set(raw_step) != {"operator", "args"}:
                return _failed(parent, "invalid_composite_plan", "each compose step requires operator and args", started)
            operator_name = raw_step.get("operator")
            args = raw_step.get("args")
            operator_type = self._BASE_OPERATORS.get(operator_name)
            if operator_type is None or not isinstance(args, Mapping):
                return _failed(parent, "invalid_composite_plan", "compose steps must use base operators", started)
            result = operator_type().execute(adapter, current, args, Path(output_dir) / ("step-%02d" % index))
            if result.status != "success" or result.child_checkpoint is None:
                return CompressionResult(
                    "failed",
                    parent.sha256,
                    None,
                    {
                        "kind": "composite",
                        "completed_steps": len(changes),
                        "steps": changes,
                        "failed_step": index,
                    },
                    failure_type=result.failure_type or "composite_step_failed",
                    message=result.message,
                    wall_time_s=time.monotonic() - started,
                )
            changes.append(
                {
                    "operator": operator_name,
                    "args": dict(args),
                    "structural_change": dict(result.structural_change),
                    "child_checkpoint_sha256": result.child_checkpoint.sha256,
                }
            )
            current = result.child_checkpoint
        return CompressionResult(
            "success",
            parent.sha256,
            current,
            {"kind": "composite", "steps": changes},
            wall_time_s=time.monotonic() - started,
        )


def _failure_type(exc: BaseException, default: str) -> str:
    message = str(exc)
    for name in ("invalid_block_selection", "invalid_channel_selection", "unsupported_model_layout", "checkpoint_reload_failed"):
        if name in message:
            return name
    return default


class WanCompressionRegistry:
    def __init__(self, operators: Sequence[CompressionOperator] = ()) -> None:
        self._operators = {operator.name: operator for operator in operators}

    def names(self) -> tuple[str, ...]:
        return tuple(self._operators)

    def get(self, name: str) -> CompressionOperator:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise ValueError("unsupported_compression_operator: %s" % name) from exc

    def execute(self, name: str, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        return self.get(name).execute(adapter, parent, plan, output_dir)


def build_wan_compression_registry() -> WanCompressionRegistry:
    return WanCompressionRegistry((BlockPruningOperator(), FFNChannelPruningOperator(), CompositeCompressionOperator()))
