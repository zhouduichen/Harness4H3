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
    return WanCompressionRegistry((BlockPruningOperator(), FFNChannelPruningOperator()))
