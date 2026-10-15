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
                "parent_structure": parent_structure.to_dict(),
                "child_structure": child_structure.to_dict(),
                "parameter_count": child_parameters,
                "parent_parameter_count": parent_parameters,
                "parameter_reduction_ratio": parameter_reduction,
                "compute_metrics": compute,
                "estimated_flops": child_flops,
                "output_metrics": output,
                "quality_error": float(output.get("quality_error", output.get("normalized_mse", 0.0))),
                "quality_score": quality,
                "search_score": quality + 0.10 * reduction_score,
                "checkpoint_size_bytes": checkpoint_bytes,
                "parent_checkpoint_size_bytes": self._parent_bytes,
                "checkpoint_size_reduction_ratio": 1.0 - checkpoint_bytes / max(self._parent_bytes, 1),
                "pipeline_model_size_bytes": checkpoint_bytes,
                "transformer_model_size_bytes": transformer_bytes,
                "parent_transformer_model_size_bytes": self._parent_transformer_bytes,
                "model_size_scope": "pipeline_logical_unique_files",
                "transformer_model_size_scope": "transformer_logical_unique_files",
                "parameter_count_scope": "transformer_only",
                "hardware_metrics": {
                    "latency_s": None,
                    "peak_memory_gb": None,
                    "model_size_gb": float(checkpoint_bytes) / float(1024**3),
                    "pipeline_model_size_gb": float(checkpoint_bytes) / float(1024**3),
                    "transformer_model_size_gb": float(transformer_bytes) / float(1024**3) if transformer_bytes else None,
                    "model_size_scope": "pipeline_logical_unique_files",
                    "transformer_model_size_scope": "transformer_logical_unique_files",
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
            "quality_score": max(-1.0, min(1.0, 1.0 - weighted_error)),
            "timestep_records": records,
            "timestep_weights": weights,
            "evaluation_condition": "fixed_cached_tica_inputs_uniform_quality_aggregation",
            "output_digest": stable_digest([(str(item[0]), str(item[1])) for item in output_pairs]),
        }


__all__ = ["AlgorithmModelEvaluator"]
