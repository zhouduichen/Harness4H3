"""Reproducible model-level measurements for algorithm comparison."""

from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact, WanModelAdapter, stable_digest
from ..operators.wan import WanCompressionRegistry
from ..sandbox.runner import SandboxResult
from .tica_prune import normalized_time_weights


def _first_tensor(value: Any) -> Any:
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    elif hasattr(value, "detach") and hasattr(value, "shape"):
        return value
    return None


def _tensor_digest(value: Any) -> str:
    tensor = _first_tensor(value)
    if tensor is None:
        return stable_digest({"type": type(value).__name__, "value": repr(value)})
    try:
        import torch

        buffer = io.BytesIO()
        torch.save(tensor.detach().cpu().contiguous(), buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    except (OSError, RuntimeError, TypeError, ValueError):
        return stable_digest({"shape": list(getattr(tensor, "shape", ())), "dtype": str(getattr(tensor, "dtype", ""))})


def _output_metrics(parent_output: Any, child_output: Any) -> Optional[dict[str, Any]]:
    parent_tensor = _first_tensor(parent_output)
    child_tensor = _first_tensor(child_output)
    if parent_tensor is None or child_tensor is None:
        return None
    if tuple(parent_tensor.shape) != tuple(child_tensor.shape):
        return None
    try:
        import torch

        parent_flat = parent_tensor.detach().float().reshape(-1)
        child_flat = child_tensor.detach().float().reshape(-1)
        delta = child_flat - parent_flat
        mse = float(torch.mean(delta.square()).item())
        scale = float(torch.mean(parent_flat.square()).item())
        normalized_mse = mse / max(scale, 1.0e-12)
        cosine = float(torch.nn.functional.cosine_similarity(parent_flat.unsqueeze(0), child_flat.unsqueeze(0)).item())
        if not all(math.isfinite(value) for value in (mse, normalized_mse, cosine)):
            return None
        return {
            "mse": mse,
            "normalized_mse": normalized_mse,
            "cosine_similarity": cosine,
            "parent_output_digest": _tensor_digest(parent_output),
            "child_output_digest": _tensor_digest(child_output),
            "shape": list(parent_tensor.shape),
        }
    except (RuntimeError, TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    """Convert runtime counters to a stable JSON-compatible snapshot."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sum_counter(value: Any, key: str) -> int:
    """Sum a named counter through per-block runtime telemetry."""

    if isinstance(value, Mapping):
        own = value.get(key, 0)
        total = int(own) if isinstance(own, (int, float)) and not isinstance(own, bool) else 0
        return total + sum(_sum_counter(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_sum_counter(item, key) for item in value)
    return 0


def _runtime_execution_evidence(model: Any, structure: Any = None) -> Mapping[str, Any]:
    """Collect actual runtime-policy counters after a probe forward.

    These counters are execution evidence, not GPU performance claims.  The
    Fake and Diffusers adapters intentionally expose the same policy families
    with slightly different storage names, so this boundary normalizes both
    without making the algorithm evaluator depend on either implementation.
    """

    root = getattr(model, "transformer", model)
    aliases = {
        "timestep_gate": ("runtime_gate_stats", "_harness4h3_gate_stats"),
        "timestep_attention": ("runtime_timestep_attention_stats", "_harness4h3_timestep_attention_stats"),
        "token_sparsity": ("runtime_token_stats", "_harness4h3_token_sparsity_stats"),
        "token_merge": ("_harness4h3_token_merge_stats",),
        "kv_token_selection": ("_harness4h3_kv_token_selection_stats",),
        "sparse_runtime": ("_harness4h3_sparse_runtime_stats",),
        "attention_window": ("_harness4h3_attention_window_stats",),
        "head_gating": ("_harness4h3_head_gating_stats",),
        "sampling": ("_harness4h3_sampling_step_stats",),
    }
    stats: dict[str, Any] = {}
    for name, candidates in aliases.items():
        for candidate in candidates:
            value = getattr(root, candidate, None)
            if value is None and root is not model:
                value = getattr(model, candidate, None)
            if isinstance(value, Mapping):
                stats[name] = _json_safe(value)
                break

    # Fake Wan keeps some attention counters on each self-attention module,
    # while the Diffusers adapter aggregates them on the transformer.  Use the
    # module-level form only when no aggregate exists so counters are not
    # double-counted.
    attention_aliases = {
        "token_merge": ("_harness4h3_token_merge_stats",),
        "kv_token_selection": ("_harness4h3_kv_token_selection_stats",),
        "sparse_runtime": ("_harness4h3_sparse_stats",),
        "attention_window": ("_harness4h3_window_stats",),
        "head_gating": ("_harness4h3_head_gating_stats",),
    }
    blocks = getattr(root, "blocks", None)
    if blocks is None:
        blocks = getattr(root, "transformer_blocks", ())
    for name, candidates in attention_aliases.items():
        if name in stats:
            continue
        per_block: dict[str, Any] = {}
        for index, block in enumerate(tuple(blocks or ())):
            attention = getattr(block, "attn1", None)
            if attention is None:
                continue
            for candidate in candidates:
                value = getattr(attention, candidate, None)
                if isinstance(value, Mapping):
                    per_block["block.%d" % index] = _json_safe(value)
                    break
        if per_block:
            stats[name] = {"by_block": per_block}

    policies: dict[str, Any] = {}
    for name, attribute in {
        "runtime_operations": "_harness4h3_runtime_operations",
        "token_sparsity": "_harness4h3_token_sparsity_policy",
        "token_merge": "_harness4h3_token_merge_policy",
        "kv_token_selection": "_harness4h3_kv_token_selection_policy",
        "sparse_runtime": "_harness4h3_sparse_runtime_policy",
        "attention_window": "_harness4h3_attention_window_policy",
        "head_gating": "_harness4h3_head_gating_policy",
        "sampling": "_harness4h3_sampling_step_policy",
        "timestep_attention": "_harness4h3_timestep_attention_policy",
    }.items():
        value = getattr(root, attribute, None)
        if isinstance(value, (Mapping, list, tuple)) and value:
            policies[name] = _json_safe(value)
    fake_timestep_rules = getattr(root, "timestep_attention_rules", None)
    if fake_timestep_rules and "timestep_attention" not in policies:
        policies["timestep_attention"] = _json_safe(fake_timestep_rules)
    if structure is not None:
        metadata = getattr(structure, "metadata", {})
        if isinstance(metadata, Mapping):
            for name in (
                "runtime_operations",
                "timestep_attention_policy",
                "token_sparsity_policy",
                "token_merge_policy",
                "kv_token_selection_policy",
                "sparse_runtime_policy",
                "attention_window_policy",
                "head_gating_policy",
                "sampling_step_policy",
            ):
                value = metadata.get(name)
                if value and name not in policies:
                    policies[name] = _json_safe(value)

    derived: dict[str, Any] = {}
    for name, full_key, selected_key in (
        ("kv_token_selection", "full_kv_tokens", "selected_kv_tokens"),
        ("sparse_runtime", "full_tokens", "selected_tokens"),
        ("attention_window", "query_tokens", "selected_kv_tokens"),
        ("timestep_attention", "full_kv_tokens", "selected_kv_tokens"),
    ):
        if name not in stats:
            continue
        full = _sum_counter(stats[name], full_key)
        selected = _sum_counter(stats[name], selected_key)
        if full > 0:
            derived[name + "_selected_ratio"] = float(selected) / float(full)
            derived[name + "_reduction_ratio"] = 1.0 - float(selected) / float(full)
    if "head_gating" in stats:
        full = _sum_counter(stats["head_gating"], "full_heads")
        selected = _sum_counter(stats["head_gating"], "selected_heads")
        if full > 0:
            derived["head_gating_selected_ratio"] = float(selected) / float(full)
            derived["head_gating_reduction_ratio"] = 1.0 - float(selected) / float(full)
    if "token_sparsity" in stats:
        selected = _sum_counter(stats["token_sparsity"], "selected") + _sum_counter(stats["token_sparsity"], "selected_tokens")
        skipped = _sum_counter(stats["token_sparsity"], "skipped") + _sum_counter(stats["token_sparsity"], "skipped_tokens")
        if selected + skipped > 0:
            derived["token_sparsity_selected_ratio"] = float(selected) / float(selected + skipped)
            derived["token_sparsity_reduction_ratio"] = float(skipped) / float(selected + skipped)
    if "timestep_gate" in stats:
        executed = _sum_counter(stats["timestep_gate"], "executed")
        skipped = _sum_counter(stats["timestep_gate"], "skipped")
        if executed + skipped > 0:
            derived["timestep_gate_skip_ratio"] = float(skipped) / float(executed + skipped)
    if "sampling" in stats:
        requested = _sum_counter(stats["sampling"], "requested_steps")
        effective = _sum_counter(stats["sampling"], "effective_steps")
        if requested > 0:
            derived["sampling_step_reduction_ratio"] = max(0.0, 1.0 - float(effective) / float(requested))

    return {
        "available": bool(stats or policies),
        "measurement_domain": "post_forward_runtime_trace",
        "stats": stats,
        "policies": policies,
        "derived": derived,
        "hardware_latency_available": False,
        "hardware_latency_reason": "algorithm probe records execution counters; GPU timing requires the independent Wan evaluator",
    }


def _logical_bytes(path: str | Path) -> int:
    """Count logical checkpoint bytes once per inode.

    Wan children share VAE/text-encoder files with hard links or symlinks;
    counting directory entries would turn unchanged components into fake
    compression gains.
    """

    root = Path(path)
    if not root.exists() and not root.is_symlink():
        return 0
    items = [root] if root.is_file() or root.is_symlink() else [item for item in root.rglob("*") if item.is_file() or item.is_symlink()]
    seen: set[tuple[int, int]] = set()
    total = 0
    for item in items:
        stat = item.stat()
        identity = (int(stat.st_dev), int(stat.st_ino))
        if identity in seen:
            continue
        seen.add(identity)
        total += int(stat.st_size)
    return total


def _checkpoint_bytes(path: str) -> int:
    return _logical_bytes(path)


@dataclass
class AlgorithmModelEvaluator:
    """Execute one Sandbox plan against one fixed parent/probe condition."""

    adapter: WanModelAdapter
    registry: WanCompressionRegistry
    parent: CheckpointArtifact
    profile: Mapping[str, Any]
    output_root: Path
    seed: int = 7
    calibration_manifest: Any = None
    search_budget: Mapping[str, int] = field(default_factory=lambda: {"max_fast_evals": 8, "max_full_evals": 4, "max_interaction_evals": 2})

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._parent_model = self.adapter.reload_model(self.parent)
        self._sample = self.adapter.profile_sample(self._parent_model, int(self.seed))
        self._parent_output = self.adapter.forward_profile(self._parent_model, self._sample)
        self._parent_flops = self._estimate_flops(self._parent_model)
        self._parent_bytes = _checkpoint_bytes(self.parent.path)
        self._parent_transformer_bytes = _logical_bytes(Path(self.parent.path) / "transformer")
        self._parent_fixed_component_bytes = {
            name: _logical_bytes(Path(self.parent.path) / name)
            for name in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae")
        }
        self._invocations = 0
        self._tica_samples: tuple[Any, ...] = ()
        self._parent_tica_outputs: dict[tuple[str, float], Any] = {}
        if self.calibration_manifest is not None:
            self._tica_samples = tuple(self.calibration_manifest.samples(self._parent_model, self.adapter))
            forward_denoising = getattr(self.adapter, "forward_denoising", None)
            if not callable(forward_denoising):
                raise ValueError("TICA calibration requires adapter.forward_denoising")
            for sample in self._tica_samples:
                for timestep in sample.spec.timesteps:
                    self._parent_tica_outputs[(str(sample.spec.sample_id), float(timestep))] = forward_denoising(
                        self._parent_model, sample, float(timestep)
                    )

    @property
    def input_digest(self) -> str:
        return _tensor_digest(self._sample)

    def evaluate(self, result: SandboxResult) -> Mapping[str, Any]:
        started = time.perf_counter()
        self._invocations += 1
        if result.plan is None:
            return self._invalid("algorithm_plan_missing", started, result)
        plan_digest = stable_digest(result.plan.to_dict())
        output_dir = self.output_root / ("plan-" + plan_digest)
        try:
            compression = self.registry.execute(
                result.plan.operator,
                self.adapter,
                self.parent,
                result.plan.args,
                output_dir,
            )
            if compression.status != "success" or compression.child_checkpoint is None:
                return self._invalid(compression.failure_type or "compression_failed", started, result, compression=compression)
            child_model = self.adapter.reload_model(compression.child_checkpoint)
            child_structure = self.adapter.inspect(child_model)
            parent_structure = self.adapter.inspect(self._parent_model)
            self._assert_finite(child_model)
            child_output = self.adapter.forward_profile(child_model, self._sample)
            runtime_probe_error = None
            runtime_metadata = child_structure.metadata if isinstance(child_structure.metadata, Mapping) else {}
            has_runtime_operation = bool(runtime_metadata.get("runtime_operations"))
            runtime_probe = getattr(self.adapter, "runtime_profile", None)
            if callable(runtime_probe) and has_runtime_operation:
                try:
                    runtime_probe(child_model, seed=int(self.seed))
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    runtime_probe_error = str(exc)
            sampling_steps = getattr(self.adapter, "sampling_steps", None)
            if callable(sampling_steps) and has_runtime_operation:
                try:
                    sampling_steps(child_model, tuple(range(16, 0, -1)))
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    runtime_probe_error = runtime_probe_error or str(exc)
            child_structure = self.adapter.inspect(child_model)
            runtime_execution = _runtime_execution_evidence(child_model, child_structure)
            if runtime_probe_error:
                runtime_execution = {**dict(runtime_execution), "probe_error": runtime_probe_error}
            if self.calibration_manifest is not None:
                output = self._measure_tica_outputs(child_model, result.plan)
            else:
                output = _output_metrics(self._parent_output, child_output)
            if output is None:
                return self._invalid("probe_output_unavailable_or_shape_mismatch", started, result, compression=compression)
            parent_parameters = int(parent_structure.parameter_count)
            child_parameters = int(child_structure.parameter_count)
            parameter_reduction = 1.0 - (child_parameters / max(parent_parameters, 1))
            child_flops = self._estimate_flops(child_model)
            compute = {
                "available": self._parent_flops is not None and child_flops is not None,
                "method": "adapter_structural_estimate" if child_flops is not None else "unavailable",
                "parent_estimated_flops": self._parent_flops,
                "child_estimated_flops": child_flops,
                "reduction_ratio": (1.0 - child_flops / max(self._parent_flops, 1)) if self._parent_flops and child_flops is not None else None,
            }
            quality = max(-1.0, min(1.0, float(output["quality_score"]))) if "quality_score" in output else max(-1.0, min(1.0, float(output["cosine_similarity"]) - float(output["normalized_mse"])))
            checkpoint_bytes = _checkpoint_bytes(compression.child_checkpoint.path)
            transformer_bytes = _logical_bytes(Path(compression.child_checkpoint.path) / "transformer")
            fixed_component_bytes = {
                name: _logical_bytes(Path(compression.child_checkpoint.path) / name)
                for name in self._parent_fixed_component_bytes
            }
            reduction_score = max(0.0, min(1.0, float(parameter_reduction)))
            return {
                "valid": True,
                "offline_simulation": bool(self.parent.metadata.get("offline_simulation", False)),
                "evidence_level": "cpu_fake_quality_proxy" if bool(self.parent.metadata.get("offline_simulation", False)) else "real_wan_structure",
                "evaluation_status": "cpu_fake_algorithm_measurement" if bool(self.parent.metadata.get("offline_simulation", False)) else "real_wan_structure_measurement",
                "parent_checkpoint_sha256": self.parent.sha256,
                "child_checkpoint_sha256": compression.child_checkpoint.sha256,
                "plan_digest": plan_digest,
                "operator": result.plan.operator,
                "structural_change": dict(compression.structural_change),
                "runtime_execution": runtime_execution,
                "parent_structure": parent_structure.to_dict(),
                "child_structure": child_structure.to_dict(),
                "parameter_count": child_parameters,
                "parent_parameter_count": parent_parameters,
                "parameter_reduction_ratio": parameter_reduction,
                "compute_metrics": compute,
                "estimated_flops": child_flops,
                "output_metrics": output,
                "quality_error": float(output.get("quality_error", output.get("normalized_mse", 0.0))),
                # TICAPruner consumes the declared risk contract at the
                # measurement boundary.  Keep the same evidence at the
                # top-level as well as inside output_metrics so a complete
                # model execution cannot be mistaken for missing risk.
                "risk_metric": output.get("risk_metric", "normalized_mse"),
                "risk_value": output.get("risk_value", output.get("normalized_mse")),
                "risk_scale": output.get("risk_scale", "normalized_mse"),
                "risk_is_approximate": output.get("risk_is_approximate") is True,
                "quality_score": quality,
                "search_score": quality + 0.10 * reduction_score,
                "checkpoint_size_bytes": checkpoint_bytes,
                "parent_checkpoint_size_bytes": self._parent_bytes,
                "checkpoint_size_reduction_ratio": 1.0 - checkpoint_bytes / max(self._parent_bytes, 1),
                "pipeline_model_size_bytes": checkpoint_bytes,
                "transformer_model_size_bytes": transformer_bytes,
                "parent_transformer_model_size_bytes": self._parent_transformer_bytes,
                "fixed_component_size_bytes": fixed_component_bytes,
                "parent_fixed_component_size_bytes": dict(self._parent_fixed_component_bytes),
                "model_size_scope": "pipeline_logical_unique_files",
                "transformer_model_size_scope": "transformer_logical_unique_files",
                "parameter_count_scope": "transformer_only",
                "hardware_measurement_scope": "not_measured_on_cpu_fake",
                "hardware_metrics": {
                    "latency_s": None,
                    "peak_memory_gb": None,
                    "model_size_gb": float(checkpoint_bytes) / float(1024**3),
                    "pipeline_model_size_gb": float(checkpoint_bytes) / float(1024**3),
                    "transformer_model_size_gb": float(transformer_bytes) / float(1024**3) if transformer_bytes else None,
                    "fixed_component_size_gb": {
                        name: float(value) / float(1024**3) if value else None
                        for name, value in fixed_component_bytes.items()
                    },
                    "model_size_scope": "pipeline_logical_unique_files",
                    "transformer_model_size_scope": "transformer_logical_unique_files",
                    "parameter_count_scope": "transformer_only",
                    "available": False,
                    "unavailable_reason": "CPU Fake algorithm evaluator does not measure GPU hardware",
                },
                "invocations": {"compression": 1, "forward": 2, "total": 3},
                "evaluation_cost": {
                    "elapsed_s": time.perf_counter() - started,
                    "compression_wall_time_s": float(compression.wall_time_s),
                    "measurement_invocations": self._invocations,
                    "input_seed": int(self.seed),
                    "input_digest": self.input_digest,
                },
                "calibration_digest": self.calibration_manifest.digest() if self.calibration_manifest is not None else None,
                "measurement_manifest_digest": self.calibration_manifest.digest() if self.calibration_manifest is not None else None,
                "evaluation_split": str(self.profile.get("evaluation_split", "calibration")),
                "calibration_input_digests": [sample.digest() for sample in self._tica_samples],
                "search_budget": dict(self.search_budget),
                "plan_time_weight_formula": result.plan.args.get("time_weight_formula") if self.calibration_manifest is not None else None,
            }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._invalid("algorithm_measurement_failed:%s" % exc, started, result)

    def _invalid(self, reason: str, started: float, result: SandboxResult, *, compression: Any = None) -> dict[str, Any]:
        return {
            "valid": False,
            "offline_simulation": bool(self.parent.metadata.get("offline_simulation", False)),
            "evidence_level": "cpu_fake_structure" if bool(self.parent.metadata.get("offline_simulation", False)) else "real_wan_structure",
            "evaluation_status": "cpu_fake_algorithm_measurement_failed" if bool(self.parent.metadata.get("offline_simulation", False)) else "real_wan_structure_measurement_failed",
            "parent_checkpoint_sha256": self.parent.sha256,
            "plan_digest": stable_digest(result.plan.to_dict()) if result.plan is not None else None,
            "operator": result.plan.operator if result.plan is not None else None,
            "failure_reason": str(reason),
            "structural_change": dict(getattr(compression, "structural_change", {}) or {}),
            "hardware_metrics": {"latency_s": None, "peak_memory_gb": None, "available": False},
            "invocations": {"compression": 1 if compression is not None else 0, "forward": 0, "total": self._invocations},
            "evaluation_cost": {"elapsed_s": time.perf_counter() - started, "input_seed": int(self.seed), "input_digest": self.input_digest},
            "evaluation_split": str(self.profile.get("evaluation_split", "calibration")),
        }

    @staticmethod
    def _assert_finite(model: Any) -> None:
        for parameter in model.parameters():
            if not bool(parameter.detach().isfinite().all()):
                raise ValueError("model_integrity_failed: non-finite parameter")

    def _estimate_flops(self, model: Any) -> Optional[int]:
        estimator = getattr(self.adapter, "estimate_flops", None)
        if not callable(estimator):
            return None
        value = estimator(model)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None

    def _measure_tica_outputs(self, child_model: Any, plan: Any) -> Optional[dict[str, Any]]:
        forward_denoising = getattr(self.adapter, "forward_denoising", None)
        if not callable(forward_denoising) or not self._tica_samples:
            return None
        records: list[dict[str, Any]] = []
        errors: list[float] = []
        output_pairs: list[tuple[Any, Any]] = []
        timesteps: list[str] = []
        for sample in self._tica_samples:
            for timestep in sample.spec.timesteps:
                parent_output = self._parent_tica_outputs.get((str(sample.spec.sample_id), float(timestep)))
                child_output = forward_denoising(child_model, sample, float(timestep))
                metrics = _output_metrics(parent_output, child_output)
                if metrics is None:
                    return None
                relative_error = float(metrics["normalized_mse"])
                errors.append(relative_error)
                output_pairs.append((parent_output, child_output))
                timesteps.append(str(timestep))
                records.append(
                    {
                        "sample_id": str(sample.spec.sample_id),
                        "timestep": float(timestep),
                        "relative_error": relative_error,
                        "mse": metrics["mse"],
                        "normalized_mse": metrics["normalized_mse"],
                        "cosine_similarity": metrics["cosine_similarity"],
                        "parent_output_digest": metrics["parent_output_digest"],
                        "child_output_digest": metrics["child_output_digest"],
                    }
                )
        if not errors:
            return None
        unique_timesteps = tuple(dict.fromkeys(timesteps))
        weights = normalized_time_weights(unique_timesteps, "uniform")
        sample_count = max(len(self._tica_samples), 1)
        weighted_error = sum((weights[name] / sample_count) * error for name, error in zip(timesteps, errors))
        mse = sum(float(item["mse"]) for item in records) / len(records)
        normalized_mse = sum(float(item["normalized_mse"]) for item in records) / len(records)
        cosine = sum(float(item["cosine_similarity"]) for item in records) / len(records)
        return {
            "mse": mse,
            "normalized_mse": normalized_mse,
            "cosine_similarity": cosine,
            "quality_error": weighted_error,
            "risk_metric": "normalized_mse",
            "risk_value": weighted_error,
            "risk_scale": "normalized_mse",
            "risk_is_approximate": False,
            "quality_score": max(-1.0, min(1.0, 1.0 - weighted_error)),
            "timestep_records": records,
            "timestep_weights": weights,
            "evaluation_condition": "fixed_cached_tica_inputs_uniform_quality_aggregation",
            "output_digest": stable_digest([(str(item[0]), str(item[1])) for item in output_pairs]),
        }


__all__ = ["AlgorithmModelEvaluator"]
