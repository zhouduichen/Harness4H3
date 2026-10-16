from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ...adapters.wan.contracts import CheckpointArtifact, ChannelPruningConfig, CompressionResult, WanModelAdapter
from .runtime_gating import validate_timestep_attention_plan, validate_timestep_gate_plan
from .runtime_policy import AttentionWindowPolicy, KVTokenSelectionPolicy, SamplingStepPolicy, SparseRuntimePolicy, TokenMergePolicy, TokenSparsityPolicy


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


class LayerPruningOperator(BlockPruningOperator):
    """Layer-granularity spelling of the block operator.

    Wan's transformer layers are exposed as blocks by both supported adapters;
    keeping a distinct name lets the Agent search at the layer level without
    duplicating structural mutation logic.
    """

    name = "prune_layers"


class AttentionHeadPruningOperator:
    """Structurally remove attention heads from every selected Wan module."""

    name = "prune_attention_heads"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        try:
            raw = plan.get("head_indices")
            if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw):
                return _failed(parent, "invalid_attention_head_selection", "head_indices must be a non-empty list of integers", started)
            attention_names = plan.get("attention_names", ["attn1", "attn2"])
            if not isinstance(attention_names, (list, tuple)) or not attention_names or not all(isinstance(item, str) for item in attention_names):
                return _failed(parent, "invalid_attention_selection", "attention_names must be a list of strings", started)
            model = adapter.reload_model(parent)
            change = adapter.prune_attention_heads(model, tuple(raw), tuple(attention_names))
            child = adapter.save_model(model, Path(output_dir))
            if child.sha256 == parent.sha256:
                return _failed(parent, "checkpoint_not_changed", "attention head pruning did not change checkpoint", started)
            return CompressionResult("success", parent.sha256, child, {"kind": change.kind, **dict(change.details), "granularity": "attention_head"}, wall_time_s=time.monotonic() - started)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return _failed(parent, _failure_type(exc, "attention_head_pruning_failed"), str(exc), started)


class HeadPruningOperator(AttentionHeadPruningOperator):
    """Short alias kept for Agent-authored head-level plans."""

    name = "prune_heads"


def _execute_runtime_policy(
    adapter: WanModelAdapter,
    parent: CheckpointArtifact,
    plan: Mapping[str, Any],
    output_dir: Path,
    *,
    policy_name: str,
    parser: Callable[[Mapping[str, Any]], Any],
    adapter_method: str,
    failure_type: str,
) -> CompressionResult:
    started = time.monotonic()
    try:
        policy = parser(plan)
        model = adapter.reload_model(parent)
        method = getattr(adapter, adapter_method, None)
        if not callable(method):
            return _failed(parent, "unsupported_model_layout", "%s is not implemented by this Wan adapter" % adapter_method, started)
        change = method(model, policy.to_dict())
        child = adapter.save_model(model, Path(output_dir))
        if child.sha256 == parent.sha256:
            return _failed(parent, "checkpoint_not_changed", "%s did not change runtime policy" % policy_name, started)
        reloaded = adapter.reload_model(child)
        restored = getattr(reloaded, "_harness4h3_%s_policy" % policy_name, None)
        if restored is None:
            transformer = getattr(reloaded, "transformer", reloaded)
            restored = getattr(getattr(transformer, "config", None), "harness4h3_%s" % policy_name, None)
        if restored is None:
            return _failed(parent, "%s_reload_failed" % policy_name, "%s policy was not restored" % policy_name, started)
        return CompressionResult(
            "success",
            parent.sha256,
            child,
            {
                "kind": "runtime_%s" % policy_name,
                "runtime_operator": policy_name,
                "policy": policy.to_dict(),
                "parameter_reduction_ratio": 0.0,
                "dynamic_compute": True,
                "child_reload_verified": True,
                **dict(change.details),
            },
            wall_time_s=time.monotonic() - started,
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _failed(parent, _failure_type(exc, failure_type), str(exc), started)


class TokenPruningOperator:
    """Dynamic token sparsity at transformer-layer input, with reloadable policy."""

    name = "prune_tokens"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        return _execute_runtime_policy(adapter, parent, plan, output_dir, policy_name="token_sparsity", parser=TokenSparsityPolicy.from_mapping, adapter_method="apply_token_sparsity", failure_type="token_pruning_failed")


class TokenMergeOperator:
    """Merge Self-Attention tokens and restore the original residual shape."""

    name = "token_merge"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        return _execute_runtime_policy(adapter, parent, plan, output_dir, policy_name="token_merge", parser=TokenMergePolicy.from_mapping, adapter_method="apply_token_merge", failure_type="token_merge_failed")


class KVTokenSelectionOperator:
    """Keep full queries while reducing the Self-Attention K/V sequence."""

    name = "kv_token_select"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        return _execute_runtime_policy(adapter, parent, plan, output_dir, policy_name="kv_token_selection", parser=KVTokenSelectionPolicy.from_mapping, adapter_method="apply_kv_token_selection", failure_type="kv_token_selection_failed")


class SparseRuntimeOperator:
    """Dynamic sparse self-attention policy with top-k key selection."""

    name = "sparse_runtime"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        return _execute_runtime_policy(adapter, parent, plan, output_dir, policy_name="sparse_runtime", parser=SparseRuntimePolicy.from_mapping, adapter_method="apply_sparse_runtime", failure_type="sparse_runtime_failed")


class AttentionWindowOperator:
    """Dynamic local Self-Attention with an actual per-query K/V window."""

    name = "attention_window"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        return _execute_runtime_policy(adapter, parent, plan, output_dir, policy_name="attention_window", parser=AttentionWindowPolicy.from_mapping, adapter_method="apply_attention_window", failure_type="attention_window_failed")


class SamplingStepSkipOperator:
    """Persist and execute a reduced denoising schedule policy."""

    name = "sampling_step_skip"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        return _execute_runtime_policy(adapter, parent, plan, output_dir, policy_name="sampling_step", parser=SamplingStepPolicy.from_mapping, adapter_method="apply_sampling_step_policy", failure_type="sampling_step_policy_failed")


class TimestepBlockGatingOperator:
    """Skip selected Wan blocks for selected denoising timestep intervals."""

    name = "timestep_block_gate"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        try:
            model = adapter.reload_model(parent)
            block_ids = [item.block_id for item in adapter.get_blocks(model)]
            rules = validate_timestep_gate_plan(plan, block_ids)
            change = adapter.apply_timestep_block_gating(model, [rule.to_dict() for rule in rules])
            child = adapter.save_model(model, Path(output_dir))
            if child.sha256 == parent.sha256:
                return _failed(parent, "checkpoint_not_changed", "runtime gate did not change checkpoint configuration", started)
            reloaded = adapter.reload_model(child)
            restored = getattr(reloaded, "runtime_gate_rules", None)
            transformer = getattr(reloaded, "transformer", reloaded)
            if restored is None:
                restored = getattr(getattr(transformer, "config", None), "harness4h3_timestep_block_gating", None)
            if not restored:
                return _failed(parent, "runtime_gate_reload_failed", "runtime gate rules were not restored", started)
            structural_change = {
                "kind": "runtime_timestep_gating",
                "runtime_operator": self.name,
                "rules": [rule.to_dict() for rule in rules],
                "parameter_reduction_ratio": 0.0,
                "dynamic_compute": True,
                "child_reload_verified": True,
                **dict(change.details),
            }
            return CompressionResult("success", parent.sha256, child, structural_change, wall_time_s=time.monotonic() - started)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return _failed(parent, _failure_type(exc, "timestep_gate_failed"), str(exc), started)


class TimestepAttentionPolicyOperator:
    """Select the actual self-attention K/V budget from each sample's timestep."""

    name = "timestep_attention_policy"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        try:
            model = adapter.reload_model(parent)
            block_ids = [item.block_id for item in adapter.get_blocks(model)]
            rules = validate_timestep_attention_plan(plan, block_ids)
            change = adapter.apply_timestep_attention_policy(model, {"rules": [rule.to_dict() for rule in rules], "policy": "select_kv_tokens"})
            child = adapter.save_model(model, Path(output_dir))
            if child.sha256 == parent.sha256:
                return _failed(parent, "checkpoint_not_changed", "timestep attention policy did not change checkpoint configuration", started)
            reloaded = adapter.reload_model(child)
            restored = getattr(reloaded, "timestep_attention_rules", None)
            transformer = getattr(reloaded, "transformer", reloaded)
            if restored is None:
                restored = getattr(getattr(transformer, "config", None), "harness4h3_timestep_attention_policy", None)
            if not restored:
                return _failed(parent, "timestep_attention_reload_failed", "timestep attention rules were not restored", started)
            return CompressionResult(
                "success",
                parent.sha256,
                child,
                {
                    "kind": "runtime_timestep_attention_policy",
                    "runtime_operator": self.name,
                    "rules": [rule.to_dict() for rule in rules],
                    "policy": "select_kv_tokens",
                    "parameter_reduction_ratio": 0.0,
                    "dynamic_compute": True,
                    "per_sample_timestep": True,
                    "child_reload_verified": True,
                },
                wall_time_s=time.monotonic() - started,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return _failed(parent, _failure_type(exc, "timestep_attention_policy_failed"), str(exc), started)


class TICAPruningOperator:
    """Execute a TICA recipe through existing trusted structural mutations."""

    name = "tica_prune"

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        try:
            mode = str(plan.get("mode", "block_plus_ffn"))
            if mode not in {"block", "ffn", "block_plus_ffn"}:
                return _failed(parent, "invalid_tica_mode", "unsupported TICA execution mode", started)
            if "per_layer_retained_indices" in plan:
                return _failed(parent, "global_ffn_alignment_required", "TICA requires one global FFN retained-index set", started)
            raw_block_ids = plan.get("block_ids", [])
            if not isinstance(raw_block_ids, (list, tuple)) or not all(isinstance(item, str) for item in raw_block_ids):
                return _failed(parent, "invalid_tica_plan", "TICA block_ids must be a list of strings", started)
            retained = plan.get("retained_indices")
            if mode in {"ffn", "block_plus_ffn"} and (not isinstance(retained, (list, tuple)) or not all(isinstance(item, int) and not isinstance(item, bool) for item in retained)):
                return _failed(parent, "invalid_tica_plan", "TICA retained_indices must be a list of integers", started)
            model = adapter.reload_model(parent)
            changes: list[Mapping[str, Any]] = []
            if mode in {"block", "block_plus_ffn"}:
                if not raw_block_ids:
                    return _failed(parent, "invalid_block_selection", "TICA block mode requires block_ids", started)
                change = adapter.remove_blocks(model, tuple(raw_block_ids))
                changes.append({"kind": change.kind, **dict(change.details)})
            if mode in {"ffn", "block_plus_ffn"}:
                remaining = tuple(item.block_id for item in adapter.get_blocks(model))
                if not remaining:
                    return _failed(parent, "invalid_channel_selection", "TICA cannot prune FFN after removing all blocks", started)
                if getattr(adapter, "channel_pruning_requires_all_blocks", False) and set(remaining) != {item.block_id for item in adapter.get_blocks(model)}:
                    return _failed(parent, "global_ffn_alignment_required", "TICA FFN mutation does not cover all remaining blocks", started)
                config = ChannelPruningConfig(remaining, tuple(retained), "ffn")
                change = adapter.prune_channels(model, config)
                changes.append({"kind": change.kind, **dict(change.details)})
            child = adapter.save_model(model, Path(output_dir))
            if child.sha256 == parent.sha256:
                return _failed(parent, "checkpoint_not_changed", "TICA recipe did not change checkpoint", started)
            return CompressionResult(
                "success",
                parent.sha256,
                child,
                {
                    "kind": "tica_prune",
                    "mode": mode,
                    "block_ids": list(raw_block_ids),
                    "retained_indices": list(retained or []),
                    "steps": changes,
                    "recipe": {key: plan[key] for key in ("time_weight_formula", "risk_limit", "activation_weight", "interaction_weight", "beam_width", "max_fast_evals", "max_full_evals", "max_interaction_evals", "strategy", "max_search_depth") if key in plan},
                },
                wall_time_s=time.monotonic() - started,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return _failed(parent, _failure_type(exc, "tica_pruning_failed"), str(exc), started)


class CompositeCompressionOperator:
    """Execute a short, ordered program of registered base operators."""

    name = "compose"
    _BASE_OPERATORS = {
        "prune_blocks": BlockPruningOperator,
        "prune_layers": LayerPruningOperator,
        "prune_ffn_channels": FFNChannelPruningOperator,
        "prune_attention_heads": AttentionHeadPruningOperator,
        "prune_heads": HeadPruningOperator,
        "prune_tokens": TokenPruningOperator,
        "token_merge": TokenMergeOperator,
        "kv_token_select": KVTokenSelectionOperator,
        "sparse_runtime": SparseRuntimeOperator,
        "attention_window": AttentionWindowOperator,
        "sampling_step_skip": SamplingStepSkipOperator,
        "timestep_block_gate": TimestepBlockGatingOperator,
        "timestep_attention_policy": TimestepAttentionPolicyOperator,
    }

    def __init__(self, operator_resolver: Optional[Callable[[str], CompressionOperator]] = None) -> None:
        self._operator_resolver = operator_resolver

    def _resolve(self, name: str) -> CompressionOperator:
        if self._operator_resolver is not None:
            return self._operator_resolver(name)
        operator_type = self._BASE_OPERATORS.get(name)
        if operator_type is None:
            raise ValueError("unsupported_composite_operator: %s" % name)
        return operator_type()

    def execute(self, adapter: WanModelAdapter, parent: CheckpointArtifact, plan: Mapping[str, Any], output_dir: Path) -> CompressionResult:
        started = time.monotonic()
        steps = plan.get("steps")
        if not isinstance(steps, (list, tuple)) or not 2 <= len(steps) <= 6:
            return _failed(parent, "invalid_composite_plan", "compose requires 2 to 6 steps", started)
        current = parent
        changes: list[Mapping[str, Any]] = []
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping) or set(raw_step) != {"operator", "args"}:
                return _failed(parent, "invalid_composite_plan", "each compose step requires operator and args", started)
            operator_name = raw_step.get("operator")
            args = raw_step.get("args")
            if not isinstance(args, Mapping):
                return _failed(parent, "invalid_composite_plan", "compose steps must use base operators", started)
            if operator_name == "prune_ffn_channels" and getattr(adapter, "channel_pruning_requires_all_blocks", False):
                # The Diffusers Wan adapter uses positional block IDs after a
                # structural removal and requires channel widths to remain
                # aligned across all remaining blocks.  Keep the algorithm's
                # intent but bind this adapter-specific invariant at the
                # trusted operator boundary.
                args = {
                    **dict(args),
                    "block_ids": ["block.%d" % current_index for current_index in range(int(current.structure.block_count))],
                }
            try:
                operator = self._resolve(str(operator_name))
            except ValueError as exc:
                return _failed(parent, "invalid_composite_plan", str(exc), started)
            result = operator.execute(adapter, current, args, Path(output_dir) / ("step-%02d" % index))
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

    def for_search_space(self, version: Any, factories: Mapping[str, Callable[..., CompressionOperator]]) -> "WanCompressionRegistry":
        """Return a registry view containing exactly the active version."""

        view = WanCompressionRegistry()
        for spec in getattr(version, "operators", ()):
            if str(getattr(spec, "name", "")) in self._operators:
                view._operators[spec.name] = self._operators[spec.name]
                continue
            factory_key = str(getattr(spec, "implementation_ref", "") or getattr(spec, "trusted_factory_key", ""))
            if not factory_key:
                raise ValueError("unavailable_trusted_operator_factory: %s" % getattr(spec, "name", ""))
            factory = factories.get(factory_key)
            if factory is None:
                raise ValueError("unavailable_trusted_operator_factory: %s" % factory_key)
            try:
                operator = factory(spec)
            except TypeError:
                # Existing trusted factories predate version-aware factories.
                # Keep their zero-argument contract while allowing a validated
                # generated rule to receive its immutable OperatorSpec.
                operator = factory()
            if operator.name != spec.name:
                raise ValueError("trusted_operator_factory_name_mismatch")
            view._operators[operator.name] = operator
        if isinstance(view._operators.get("compose"), CompositeCompressionOperator):
            view._operators["compose"] = CompositeCompressionOperator(lambda name: view.get(name))
        return view


def build_wan_compression_registry(*, include_tica: bool = False, include_multigranular: bool = False) -> WanCompressionRegistry:
    operators: list[CompressionOperator] = [BlockPruningOperator(), FFNChannelPruningOperator(), CompositeCompressionOperator()]
    if include_multigranular:
        operators[2] = CompositeCompressionOperator()
        operators.extend(
            [
                LayerPruningOperator(),
                AttentionHeadPruningOperator(),
                HeadPruningOperator(),
                TokenPruningOperator(),
                TokenMergeOperator(),
                KVTokenSelectionOperator(),
                SparseRuntimeOperator(),
                AttentionWindowOperator(),
                SamplingStepSkipOperator(),
                TimestepBlockGatingOperator(),
                TimestepAttentionPolicyOperator(),
            ]
        )
    if include_tica:
        operators.append(TICAPruningOperator())
    return WanCompressionRegistry(tuple(operators))
