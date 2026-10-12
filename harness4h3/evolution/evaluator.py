"""Measured Wan video evaluation for evolutionary compression rounds."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact, stable_digest


class WanEvaluationError(RuntimeError):
    """A real evaluation could not produce trusted evidence."""


EVALUATOR_VERSION = "wan-video-evaluator-v5"
LATENCY_MEASUREMENT_METHOD = "denoising+vae_decode_with_case_cached_prompt_embeddings"

DEFAULT_VISUAL_QUALITY_GATE = {
    "enabled": True,
    "version": "visual-structure-v1",
    # These are conservative floors for the 256x256/81-frame Wan contract.
    # They reject blank/near-solid videos without importing the stricter
    # 5-second campaign thresholds into this evaluator.
    "min_spatial_gradient": 0.05,
    "min_laplacian_std": 0.50,
    "min_block_variance": 1.0,
    "min_frame_difference": 0.001,
    "max_low_texture_frame_ratio": 0.75,
}


@dataclass(frozen=True)
class PromotionPolicy:
    """Independent promotion gate for measured candidate evidence."""

    quality_floor_ratio: float = 0.98
    min_efficiency_improvement: float = 0.05
    min_model_size_improvement: float = 0.10
    max_metric_regression_ratio: float = 0.05

    def __post_init__(self) -> None:
        for name in ("quality_floor_ratio", "min_efficiency_improvement", "min_model_size_improvement", "max_metric_regression_ratio"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in [0, 1]" % name)

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]] = None) -> "PromotionPolicy":
        value = dict(raw or {})
        return cls(
            quality_floor_ratio=float(value.get("quality_floor_ratio", value.get("min_quality_ratio", 0.98))),
            min_efficiency_improvement=float(value.get("min_efficiency_improvement", 0.05)),
            min_model_size_improvement=float(value.get("min_model_size_improvement", 0.10)),
            max_metric_regression_ratio=float(value.get("max_metric_regression_ratio", value.get("max_hardware_regression", 0.05))),
        )

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) and value >= 0.0 else None

    @classmethod
    def _metric(cls, mapping: Mapping[str, Any], names: tuple[str, ...]) -> Optional[float]:
        for name in names:
            value = cls._number(mapping.get(name))
            if value is not None:
                return value
        return None

    @classmethod
    def _hardware(cls, mapping: Mapping[str, Any]) -> Mapping[str, Any]:
        value = mapping.get("hardware_metrics")
        return value if isinstance(value, Mapping) else mapping

    @classmethod
    def _dominates(cls, left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        compared = False
        strict = False
        left_quality = cls._number(left.get("quality_score"))
        right_quality = cls._number(right.get("quality_score"))
        if left_quality is None or right_quality is None or left_quality < right_quality:
            return False
        strict = left_quality > right_quality
        for names in (("latency_s", "inference_wall_time_s_mean"), ("peak_memory_gb", "peak_memory_gb_max"), ("model_size_gb",)):
            left_value = cls._metric(cls._hardware(left), names)
            right_value = cls._metric(cls._hardware(right), names)
            if left_value is None or right_value is None:
                continue
            compared = True
            if left_value > right_value:
                return False
            strict = strict or left_value < right_value
        return compared and strict

    def decide(
        self,
        *,
        parent: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        child_integrity: bool,
        pareto_front: tuple[Mapping[str, Any], ...] = (),
    ) -> Mapping[str, Any]:
        """Return an auditable decision; never trust evaluator ``feasible``."""

        reasons: list[str] = []
        if not child_integrity:
            reasons.append("child_integrity_failed")
        if evaluation.get("valid") is not True:
            reasons.append("independent_evaluation_invalid_or_missing")
        candidate_quality = self._number(evaluation.get("quality_score"))
        # Once a candidate has been promoted, its measured metrics become the
        # next Parent.  The evaluator still keeps the immutable initial
        # baseline for drift reporting, but promotion must compare against the
        # active Parent first.
        parent_quality = self._number(parent.get("quality_score"))
        if parent_quality is None:
            parent_quality = self._number(evaluation.get("baseline_quality_score"))
        if candidate_quality is None or parent_quality is None:
            reasons.append("quality_data_missing")
        elif candidate_quality < parent_quality * self.quality_floor_ratio:
            reasons.append("quality_floor_failed")

        metric_specs = (
            (("latency_s", "inference_wall_time_s_mean"), self.min_efficiency_improvement),
            (("peak_memory_gb", "peak_memory_gb_max"), self.min_efficiency_improvement),
            (("model_size_gb",), self.min_model_size_improvement),
        )
        candidate_hw = self._hardware(evaluation)
        parent_hw = self._hardware(parent)
        has_parent_measurement = any(
            self._metric(parent_hw, names) is not None
            for names, _minimum in metric_specs
        )
        baseline_hw = parent_hw if has_parent_measurement else evaluation.get("baseline_hardware_metrics")
        if not isinstance(baseline_hw, Mapping):
            baseline_hw = parent_hw
        improvements: dict[str, float] = {}
        comparable = 0
        for names, minimum in metric_specs:
            before = self._metric(baseline_hw, names)
            after = self._metric(candidate_hw, names)
            if before is None or after is None or before <= 0.0:
                continue
            comparable += 1
            ratio = (before - after) / before
            improvements[names[0]] = ratio
            if ratio < -self.max_metric_regression_ratio:
                reasons.append("hardware_regression:%s" % names[0])
        if comparable == 0:
            reasons.append("hardware_data_missing")
        efficiency_improved = any(
            improvements.get(names[0], -1.0) >= self.min_efficiency_improvement
            for names, _minimum in metric_specs[:2]
        )
        if comparable > 0 and not efficiency_improved:
            reasons.append("no_substantial_efficiency_improvement")
        model_size_improvement = improvements.get("model_size_gb")
        if model_size_improvement is None:
            reasons.append("model_size_data_missing")
        elif model_size_improvement < self.min_model_size_improvement:
            reasons.append("model_size_improvement_below_threshold")

        # A candidate must not be dominated by the current independent front.
        if pareto_front and any(self._dominates(front, evaluation) for front in pareto_front):
            reasons.append("pareto_dominated")

        accepted = not reasons
        return {
            "accepted": accepted,
            "reasons": reasons,
            "quality_floor_ratio": self.quality_floor_ratio,
            "min_efficiency_improvement": self.min_efficiency_improvement,
            "min_model_size_improvement": self.min_model_size_improvement,
            "max_metric_regression_ratio": self.max_metric_regression_ratio,
            "improvements": improvements,
            "independent_evaluation": True,
        }


@dataclass(frozen=True)
class WanVideoEvaluationConfig:
    wan_config: Path
    evaluation_manifest: Path
    clip_model_path: Path
    output_root: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    max_cases: int = 1
    seed_count: int = 1
    sample_frames: int = 4
    max_quality_drop: float = 0.05
    latency_warmup: int = 1
    latency_repeats: int = 3
    visual_quality_gate: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_VISUAL_QUALITY_GATE))
    quality_device: str = "cpu"
    source_revision: str = "unknown"

    def __post_init__(self) -> None:
        if not str(self.wan_config).strip() or not str(self.evaluation_manifest).strip():
            raise ValueError("real Wan evaluation requires wan_config and evaluation_manifest")
        if not str(self.clip_model_path).strip():
            raise ValueError("real Wan evaluation requires clip_model_path")
        if self.max_cases <= 0 or self.seed_count <= 0 or self.sample_frames < 2 or self.latency_warmup < 0 or self.latency_repeats <= 0:
            raise ValueError("real Wan evaluation budgets must be positive")
        if not 0.0 <= float(self.max_quality_drop) <= 1.0:
            raise ValueError("max_quality_drop must be in [0, 1]")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], output_root: Path) -> "WanVideoEvaluationConfig":
        required = {"wan_config", "evaluation_manifest", "clip_model_path"}
        missing = sorted(name for name in required if not str(raw.get(name, "")).strip())
        if missing:
            raise ValueError("real Wan evaluator missing field(s): %s" % ", ".join(missing))
        return cls(
            wan_config=Path(str(raw["wan_config"])).expanduser(),
            evaluation_manifest=Path(str(raw["evaluation_manifest"])).expanduser(),
            clip_model_path=Path(str(raw["clip_model_path"])).expanduser(),
            output_root=Path(output_root),
            device=str(raw.get("device", "cuda:0")),
            dtype=str(raw.get("dtype", "bfloat16")),
            local_files_only=bool(raw.get("local_files_only", True)),
            max_cases=int(raw.get("max_cases", 1)),
            seed_count=int(raw.get("seed_count", 1)),
            sample_frames=int(raw.get("sample_frames", 4)),
            max_quality_drop=float(raw.get("max_quality_drop", 0.05)),
            latency_warmup=int(raw.get("latency_warmup", 1)),
            latency_repeats=int(raw.get("latency_repeats", 3)),
            visual_quality_gate=dict(raw.get("visual_quality_gate") or DEFAULT_VISUAL_QUALITY_GATE),
            quality_device=str(raw.get("quality_device", raw.get("device", "cpu"))),
            source_revision=str(
                raw.get("source_revision")
                or os.environ.get("HARNESS4H3_SOURCE_REVISION", "unknown")
            ).strip() or "unknown",
        )


class WanVideoEvaluator:
    """Generate fixed prompt/seed cases and score them with CLIP-temporal."""

    def __init__(
        self,
        config: WanVideoEvaluationConfig,
        baseline: CheckpointArtifact,
        *,
        quality_backend: Optional[Any] = None,
        pipeline_loader: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.config = config
        self.baseline = baseline
        self._quality_backend = quality_backend
        self._pipeline_loader = pipeline_loader
        self._cache: dict[str, Mapping[str, Any]] = {}
        self.manifest_sha256 = hashlib.sha256(Path(config.evaluation_manifest).read_bytes()).hexdigest()
        self._manifest = self._load_manifest(config.evaluation_manifest)
        if len(self._manifest) < int(config.max_cases):
            raise WanEvaluationError(
                "evaluation_manifest_insufficient_cases: requested %d, found %d"
                % (int(config.max_cases), len(self._manifest))
            )
        for item in self._manifest[: int(config.max_cases)]:
            seeds = item.get("seeds")
            if not isinstance(seeds, list) or len(seeds) < int(config.seed_count):
                raise WanEvaluationError(
                    "evaluation_manifest_insufficient_seeds: case %s requires %d"
                    % (str(item.get("case_id", "case")), int(config.seed_count))
                )
        self._wan_spec = self._load_wan_spec(config.wan_config)
        self.recipe_digest = stable_digest(
            {
                "manifest": self._manifest,
                "manifest_sha256": self.manifest_sha256,
                "wan_spec": self._wan_spec,
                "evaluator_version": EVALUATOR_VERSION,
                "device": config.device,
                "dtype": config.dtype,
                "max_cases": config.max_cases,
                "seed_count": config.seed_count,
                "sample_frames": config.sample_frames,
                "latency_warmup": config.latency_warmup,
                "latency_repeats": config.latency_repeats,
                "latency_measurement_method": LATENCY_MEASUREMENT_METHOD,
                "visual_quality_gate": dict(config.visual_quality_gate),
                "quality_device": config.quality_device,
                "quality_backend": type(quality_backend).__name__ if quality_backend is not None else "ClipTemporalQualityBackend",
                "clip_model_path": str(config.clip_model_path),
                "source_revision": config.source_revision,
            }
        )

    @staticmethod
    def _load_manifest(path: Path) -> tuple[Mapping[str, Any], ...]:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WanEvaluationError("evaluation_manifest_read_failed: %s" % exc) from exc
        cases = raw.get("cases") if isinstance(raw, Mapping) else None
        if not isinstance(cases, list) or not cases:
            raise WanEvaluationError("evaluation_manifest_invalid: cases are required")
        return tuple(dict(item) for item in cases[:])

    @staticmethod
    def _load_wan_spec(path: Path) -> Mapping[str, Any]:
        try:
            import yaml

            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise WanEvaluationError("wan_config_read_failed: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise WanEvaluationError("wan_config_invalid: expected mapping")
        spec = raw.get("wan_student", raw)
        if not isinstance(spec, Mapping):
            raise WanEvaluationError("wan_config_invalid: wan_student must be a mapping")
        required = ("width", "height", "num_frames", "native_fps", "num_inference_steps", "guidance_scale")
        if any(name not in spec for name in required):
            raise WanEvaluationError("wan_config_invalid: missing generation contract")
        return dict(spec)

    def __call__(self, child: CheckpointArtifact, profile: Mapping[str, Any]) -> Mapping[str, Any]:
        del profile
        try:
            baseline = self._evaluate_artifact(self.baseline)
            candidate = self._evaluate_artifact(child)
        except WanEvaluationError as exc:
            return {
                "evaluation_status": "failed",
                "failure_type": "evaluation_failed",
                "failure_message": str(exc),
                "feasible": False,
                "critical_regression": True,
                "offline_simulation": False,
                "search_score": 0.0,
                "evaluator_recipe_sha256": self.recipe_digest,
                "evaluator_version": EVALUATOR_VERSION,
                "evaluation_manifest_sha256": self.manifest_sha256,
                "source_revision": self.config.source_revision,
            }
        baseline_quality = float(baseline["quality_score"])
        candidate_quality = float(candidate["quality_score"])
        quality_regression = candidate_quality < baseline_quality - float(self.config.max_quality_drop)
        candidate_hardware = dict(candidate.get("hardware_metrics") or {})
        baseline_hardware = dict(baseline.get("hardware_metrics") or {})
        candidate_latency = float(candidate_hardware.get("inference_wall_time_s_mean", float("inf")))
        baseline_latency = float(baseline_hardware.get("inference_wall_time_s_mean", float("inf")))
        latency_ratio = candidate_latency / baseline_latency if baseline_latency > 0 and math.isfinite(baseline_latency) else 1.0
        baseline_valid = bool(baseline.get("valid", False))
        candidate_valid = bool(candidate.get("valid", False))
        feasible = baseline_valid and candidate_valid and not quality_regression
        search_score = candidate_quality + max(0.0, 1.0 - latency_ratio) * 0.05
        return {
            **dict(candidate),
            "baseline_quality_score": baseline_quality,
            "baseline_hardware_metrics": dict(baseline.get("hardware_metrics") or {}),
            "baseline_evaluation_valid": baseline_valid,
            "candidate_evaluation_valid": candidate_valid,
            "quality_delta": candidate_quality - baseline_quality,
            "latency_ratio": latency_ratio,
            "feasible": feasible,
            "critical_regression": quality_regression or not baseline_valid or not candidate_valid,
            "offline_simulation": False,
            "search_score": search_score,
            "evaluator_recipe_sha256": self.recipe_digest,
            "source_revision": self.config.source_revision,
        }

    def _evaluate_artifact(self, artifact: CheckpointArtifact) -> Mapping[str, Any]:
        cached = self._cache.get(artifact.sha256)
        if cached is not None:
            return cached
        result_dir = self.config.output_root / "evaluations" / artifact.sha256 / self.recipe_digest
        result_path = result_dir / "evaluation.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(result, Mapping) and result.get("status") == "success":
                    self._cache[artifact.sha256] = dict(result)
                    return dict(result)
            except (OSError, json.JSONDecodeError):
                pass
        result_dir.mkdir(parents=True, exist_ok=True)
        progress_path = result_dir / "progress.json"
        completed_cases = self._load_progress(progress_path, artifact.sha256, self.recipe_digest)
        completed_keys: set[tuple[str, int]] = set()
        for item in completed_cases:
            try:
                completed_keys.add((str(item["case_id"]), int(item["seed"])))
            except (KeyError, TypeError, ValueError):
                continue
        resumed_case_count = len(completed_cases)
        pipeline = None
        try:
            pipeline, torch = self._load_pipeline(artifact.path)
            quality = self._get_quality_backend()
            cases = list(completed_cases)
            for raw_case in self._manifest[: self.config.max_cases]:
                case_id = str(raw_case.get("case_id", "case"))
                caption = str(raw_case.get("caption", "")).strip()
                seeds = tuple(int(seed) for seed in raw_case.get("seeds", ()))[: self.config.seed_count]
                if not caption or not seeds:
                    raise WanEvaluationError("evaluation_manifest_invalid: case requires caption and seeds")
                for seed in seeds:
                    if (case_id, seed) in completed_keys:
                        continue
                    case_dir = result_dir / "videos" / case_id / ("seed_%d" % seed)
                    video_path = case_dir / "wan.mp4"
                    prompt_embeddings = self._prepare_prompt_embeddings(pipeline, torch, caption)
                    self._reset_peak_memory(torch)
                    for _ in range(int(self.config.latency_warmup)):
                        self._generate_case(pipeline, torch, caption, seed, prompt_embeddings)
                    latency_samples: list[float] = []
                    frames = None
                    for _ in range(int(self.config.latency_repeats)):
                        started = time.perf_counter()
                        current = self._generate_case(pipeline, torch, caption, seed, prompt_embeddings)
                        latency_samples.append(time.perf_counter() - started)
                        if frames is None:
                            frames = current
                    inference_wall = float(statistics.median(latency_samples))
                    if frames is None:
                        raise WanEvaluationError("video_generation_empty")
                    self._write_video(frames, video_path, float(self._wan_spec["native_fps"]))
                    validity = self._validate_video(video_path, frames)
                    evidence = quality.evaluate(video_path, caption)
                    quality_record = evidence.to_dict()
                    visual_gate = quality_record.get("visual_quality_gate")
                    visual_gate_passed = not isinstance(visual_gate, Mapping) or visual_gate.get("passed", True) is not False
                    validity = {**dict(validity), "visual_quality_gate_passed": visual_gate_passed}
                    cases.append({
                        "case_id": case_id,
                        "seed": seed,
                        "caption": caption,
                        "video_path": str(video_path),
                        "generation_wall_time_s": inference_wall,
                        "latency_samples_s": latency_samples,
                        "latency_warmup": int(self.config.latency_warmup),
                        "latency_measurement_method": LATENCY_MEASUREMENT_METHOD if prompt_embeddings is not None else "full_generate",
                        "prompt_embeddings_cached": prompt_embeddings is not None,
                        "peak_memory_gb": self._peak_memory_gb(torch),
                        "validity": validity,
                        "quality": quality_record,
                    })
                    completed_keys.add((case_id, seed))
                    _atomic_json(
                        progress_path,
                        {
                            "status": "in_progress",
                            "checkpoint_sha256": artifact.sha256,
                            "evaluator_recipe_sha256": self.recipe_digest,
                            "evaluator_version": EVALUATOR_VERSION,
                            "evaluation_manifest_sha256": self.manifest_sha256,
                            "source_revision": self.config.source_revision,
                            "cases": cases,
                        },
                    )
            if not cases:
                raise WanEvaluationError("evaluation_manifest_invalid: no cases selected")
            scores = [float(item["quality"]["aggregate"]) for item in cases]
            result = {
                "status": "success",
                "checkpoint_sha256": artifact.sha256,
                "evaluator_version": EVALUATOR_VERSION,
                "evaluation_manifest_sha256": self.manifest_sha256,
                "evaluation_status": "measured_video",
                "valid": all(bool(item["validity"]["valid"]) and bool(item["validity"].get("visual_quality_gate_passed", True)) for item in cases),
                "promotable": all(bool(item["validity"]["valid"]) and bool(item["validity"].get("visual_quality_gate_passed", True)) for item in cases),
                "offline_simulation": False,
                "quality_score": sum(scores) / len(scores),
                "quality_metrics": {
                    "semantic": sum(float(item["quality"]["semantic"]) for item in cases) / len(cases),
                    "temporal": sum(float(item["quality"]["temporal"]) for item in cases) / len(cases),
                    "motion": sum(float(item["quality"]["motion"]) for item in cases) / len(cases),
                    "score_type": "clip_temporal",
                },
                "hardware_metrics": {
                    "inference_wall_time_s_mean": sum(float(item["generation_wall_time_s"]) for item in cases) / len(cases),
                    "inference_wall_time_s_median": float(statistics.median(float(item["generation_wall_time_s"]) for item in cases)),
                    "inference_wall_time_s_total": sum(float(item["generation_wall_time_s"]) for item in cases),
                    "gpu_runtime_cost_s": sum(sum(float(value) for value in item["latency_samples_s"]) for item in cases),
                    "latency_measurement_method": LATENCY_MEASUREMENT_METHOD,
                    "prompt_encoding_included": False,
                    "peak_memory_gb_max": max(float(item["peak_memory_gb"]) for item in cases),
                    "model_size_gb": self._checkpoint_size_gb(artifact.path),
                    "parameter_count": int(artifact.structure.parameter_count),
                    "case_count": len(cases),
                },
                "resumed_case_count": resumed_case_count,
                "source_revision": self.config.source_revision,
                "cases": cases,
            }
            _atomic_json(result_path, result)
            self._cache[artifact.sha256] = result
            return result
        except WanEvaluationError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WanEvaluationError("wan_evaluation_failed: %s" % exc) from exc
        finally:
            if pipeline is not None:
                del pipeline
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (AttributeError, RuntimeError, UnboundLocalError):
                pass

    @staticmethod
    def _load_progress(path: Path, checkpoint_sha256: str, recipe_digest: str) -> list[Mapping[str, Any]]:
        """Load only case records belonging to this artifact and recipe."""

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, Mapping) or raw.get("checkpoint_sha256") != checkpoint_sha256:
            return []
        if raw.get("evaluator_recipe_sha256") != recipe_digest or not isinstance(raw.get("cases"), list):
            return []
        records: list[Mapping[str, Any]] = []
        for item in raw["cases"]:
            if not isinstance(item, Mapping):
                continue
            if not item.get("case_id") or item.get("seed") is None:
                continue
            video_path = item.get("video_path")
            quality = item.get("quality")
            validity = item.get("validity")
            if not isinstance(video_path, str) or not Path(video_path).is_file():
                continue
            if not isinstance(quality, Mapping) or not isinstance(validity, Mapping):
                continue
            records.append(dict(item))
        return records

    def _get_quality_backend(self) -> Any:
        if self._quality_backend is None:
            from ..student.quality import ClipTemporalQualityBackend

            self._quality_backend = ClipTemporalQualityBackend(
                self.config.clip_model_path,
                device=self.config.quality_device,
                sample_frames=self.config.sample_frames,
                visual_quality_gate=self.config.visual_quality_gate,
            )
        return self._quality_backend

    def _cuda_device(self, torch: Any) -> Any:
        return torch.device(self.config.device)

    def _reset_peak_memory(self, torch: Any) -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self._cuda_device(torch))
        except (AttributeError, RuntimeError):
            pass

    def _peak_memory_gb(self, torch: Any) -> float:
        try:
            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated(self._cuda_device(torch))) / float(1024**3)
        except (AttributeError, RuntimeError):
            pass
        return 0.0

    @staticmethod
    def _checkpoint_size_gb(path: str) -> float:
        root = Path(path)
        try:
            if root.is_file():
                total = root.stat().st_size
            else:
                total = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        except OSError as exc:
            raise WanEvaluationError("checkpoint_size_failed: %s" % exc) from exc
        return float(total) / float(1024**3)

    def _load_pipeline(self, checkpoint: str) -> tuple[Any, Any]:
        if self._pipeline_loader is not None:
            loaded = self._pipeline_loader(checkpoint)
            return loaded
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise WanEvaluationError("diffusers_or_torch_missing: %s" % exc) from exc
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                checkpoint,
                local_files_only=self.config.local_files_only,
                torch_dtype=getattr(torch, self.config.dtype),
            )
            pipeline.transformer.to(self.config.device).eval()
            execution_device = torch.device(self.config.device)
            cls = type(
                "WanEvolutionCudaExecution",
                (pipeline.__class__,),
                {"_execution_device": property(lambda self: self._evolution_execution_device)},
            )
            pipeline.__class__ = cls
            pipeline._evolution_execution_device = execution_device
            if hasattr(pipeline, "set_progress_bar_config"):
                pipeline.set_progress_bar_config(disable=True)
            return pipeline, torch
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WanEvaluationError("wan_pipeline_load_failed: %s" % exc) from exc

    @staticmethod
    def _prepare_prompt_embeddings(pipeline: Any, torch: Any, caption: str) -> Optional[tuple[Any, Any]]:
        """Encode fixed text once per case; it is not a compressed component."""

        getter = getattr(pipeline, "_get_t5_prompt_embeds", None)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if not callable(getter) or text_encoder is None:
            return None
        text_dtype = getattr(text_encoder, "dtype", getattr(torch, "bfloat16"))
        return (
            getter(caption, device=torch.device("cpu"), dtype=text_dtype),
            getter("", device=torch.device("cpu"), dtype=text_dtype),
        )

    def _generate_case(
        self,
        pipeline: Any,
        torch: Any,
        caption: str,
        seed: int,
        prompt_embeddings: Optional[tuple[Any, Any]],
    ) -> Any:
        if prompt_embeddings is None:
            return self._generate(pipeline, torch, caption, seed)
        return self._generate(
            pipeline,
            torch,
            caption,
            seed,
            prompt_embeds=prompt_embeddings[0],
            negative_prompt_embeds=prompt_embeddings[1],
        )

    def _generate(
        self,
        pipeline: Any,
        torch: Any,
        caption: str,
        seed: int,
        *,
        prompt_embeds: Any = None,
        negative_prompt_embeds: Any = None,
    ) -> Any:
        device = torch.device(self.config.device)
        if prompt_embeds is None or negative_prompt_embeds is None:
            text_dtype = getattr(pipeline.text_encoder, "dtype", getattr(torch, self.config.dtype))
            prompt_embeds = pipeline._get_t5_prompt_embeds(caption, device=torch.device("cpu"), dtype=text_dtype)
            negative_prompt_embeds = pipeline._get_t5_prompt_embeds("", device=torch.device("cpu"), dtype=text_dtype)
        generator = torch.Generator(device=str(device)).manual_seed(int(seed))
        with torch.inference_mode():
            output = pipeline(
                prompt=None,
                prompt_embeds=prompt_embeds.to(device),
                negative_prompt_embeds=negative_prompt_embeds.to(device),
                height=int(self._wan_spec["height"]),
                width=int(self._wan_spec["width"]),
                num_frames=int(self._wan_spec["num_frames"]),
                num_inference_steps=int(self._wan_spec["num_inference_steps"]),
                guidance_scale=float(self._wan_spec["guidance_scale"]),
                generator=generator,
                output_type="latent",
            )
            latents = output.frames.to("cpu", dtype=pipeline.vae.dtype)
            decoded = pipeline.vae.decode(latents, return_dict=False)[0]
            return pipeline.video_processor.postprocess_video(decoded, output_type="np")

    @staticmethod
    def _write_video(frames: Any, path: Path, fps: float) -> None:
        import cv2
        import numpy as np

        value = np.asarray(frames)
        if value.ndim == 5:
            value = value[0]
        if value.ndim != 4 or value.shape[-1] != 3:
            raise WanEvaluationError("video_output_invalid: expected [T,H,W,3]")
        if float(value.min()) < 0.0:
            value = (value + 1.0) * 127.5
        elif float(value.max()) <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0.0, 255.0).astype("uint8")
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (value.shape[2], value.shape[1]))
        if not writer.isOpened():
            raise WanEvaluationError("video_writer_failed: %s" % path)
        try:
            for frame in value:
                writer.write(frame[..., ::-1])
        finally:
            writer.release()
        if not path.is_file() or path.stat().st_size <= 0:
            raise WanEvaluationError("video_writer_failed: empty output")

    @staticmethod
    def _validate_video(path: Path, frames: Any) -> Mapping[str, Any]:
        import numpy as np

        value = np.asarray(frames)
        if value.ndim == 5:
            value = value[0]
        valid = bool(value.ndim == 4 and value.shape[-1] == 3 and value.shape[0] > 0 and np.isfinite(value).all())
        return {
            "valid": valid,
            "frames": int(value.shape[0]) if value.ndim >= 1 else 0,
            "height": int(value.shape[1]) if value.ndim >= 3 else 0,
            "width": int(value.shape[2]) if value.ndim >= 3 else 0,
            "finite_pixels": bool(np.isfinite(value).all()) if value.size else False,
            "video_path": str(path),
        }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["WanEvaluationError", "WanVideoEvaluationConfig", "WanVideoEvaluator"]
