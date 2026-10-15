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
from typing import Any, Callable, ClassVar, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact, stable_digest


class WanEvaluationError(RuntimeError):
    """A real evaluation could not produce trusted evidence."""


EVALUATOR_VERSION = "wan-video-evaluator-v7"
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
    # Kept as a compatibility knob for existing configs.  New configs can
    # override latency and memory independently below.
    min_efficiency_improvement: float = 0.05
    min_latency_improvement: Optional[float] = None
    min_memory_improvement: Optional[float] = None
    min_model_size_improvement: float = 0.10
    max_metric_regression_ratio: float = 0.05

    _NON_TARGET_MINIMIZE_METRICS: ClassVar[tuple[str, ...]] = ("energy_j", "energy_j_mean", "gpu_runtime_cost_s")

    def __post_init__(self) -> None:
        for name in (
            "quality_floor_ratio",
            "min_efficiency_improvement",
            "min_model_size_improvement",
            "max_metric_regression_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in [0, 1]" % name)
        for name in ("min_latency_improvement", "min_memory_improvement"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError("%s must be in [0, 1]" % name)

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]] = None) -> "PromotionPolicy":
        value = dict(raw or {})
        legacy_efficiency = float(value.get("min_efficiency_improvement", 0.05))
        return cls(
            quality_floor_ratio=float(value.get("quality_floor_ratio", value.get("min_quality_ratio", 0.98))),
            min_efficiency_improvement=legacy_efficiency,
            min_latency_improvement=float(value.get("min_latency_improvement", legacy_efficiency)),
            min_memory_improvement=float(value.get("min_memory_improvement", legacy_efficiency)),
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
    def _positive_metric(cls, mapping: Mapping[str, Any], names: tuple[str, ...]) -> Optional[float]:
        value = cls._metric(mapping, names)
        return value if value is not None and value > 0.0 else None

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
        for names in (
            ("latency_s", "inference_wall_time_s_mean"),
            ("peak_memory_gb", "peak_memory_gb_max"),
            ("model_size_gb", "pipeline_model_size_gb"),
        ):
            left_value = cls._positive_metric(cls._hardware(left), names)
            right_value = cls._positive_metric(cls._hardware(right), names)
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
        missing_metrics: list[str] = []
        objective_failures: list[str] = []
        quality_metrics_checked: list[str] = []
        quality_metrics_unavailable: list[str] = []
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

        candidate_quality_metrics = evaluation.get("quality_metrics")
        if not isinstance(candidate_quality_metrics, Mapping):
            candidate_quality_metrics = {}
        parent_quality_metrics = parent.get("quality_metrics")
        if not isinstance(parent_quality_metrics, Mapping):
            parent_quality_metrics = evaluation.get("baseline_quality_metrics")
        if not isinstance(parent_quality_metrics, Mapping):
            parent_quality_metrics = {}
        for metric_name in ("semantic", "temporal", "motion"):
            candidate_value = self._number(candidate_quality_metrics.get(metric_name))
            parent_value = self._number(parent_quality_metrics.get(metric_name))
            if candidate_value is None or parent_value is None:
                if candidate_value is not None or parent_value is not None:
                    quality_metrics_unavailable.append(metric_name)
                continue
            quality_metrics_checked.append(metric_name)
            if candidate_value < parent_value * self.quality_floor_ratio:
                reasons.append("quality_metric_floor_failed:%s" % metric_name)

        latency_threshold = float(self.min_latency_improvement if self.min_latency_improvement is not None else self.min_efficiency_improvement)
        memory_threshold = float(self.min_memory_improvement if self.min_memory_improvement is not None else self.min_efficiency_improvement)
        metric_specs = (
            (("latency_s", "inference_wall_time_s_mean"), latency_threshold),
            (("peak_memory_gb", "peak_memory_gb_max"), memory_threshold),
            (("model_size_gb", "pipeline_model_size_gb"), self.min_model_size_improvement),
        )
        candidate_hw = self._hardware(evaluation)
        parent_hw = self._hardware(parent)
        has_parent_measurement = any(
            self._positive_metric(parent_hw, names) is not None
            for names, _minimum in metric_specs
        )
        baseline_hw = parent_hw if has_parent_measurement else evaluation.get("baseline_hardware_metrics")
        if not isinstance(baseline_hw, Mapping):
            baseline_hw = parent_hw
        improvements: dict[str, float] = {}
        comparable = 0
        objective_passes: dict[str, bool] = {}
        for names, minimum in metric_specs:
            canonical_name = names[0]
            before = self._positive_metric(baseline_hw, names)
            after = self._positive_metric(candidate_hw, names)
            if before is None or after is None:
                missing_metrics.append(canonical_name)
                objective_passes[canonical_name] = False
                continue
            comparable += 1
            ratio = (before - after) / before
            improvements[canonical_name] = ratio
            objective_passes[canonical_name] = ratio >= minimum
            if ratio < minimum:
                objective_failures.append(canonical_name)
            if ratio < -self.max_metric_regression_ratio:
                reasons.append("hardware_regression:%s" % canonical_name)
        if comparable == 0:
            reasons.append("hardware_data_missing")
        if comparable > 0 and not any(objective_passes.values()):
            reasons.append("no_substantial_efficiency_improvement")
            reasons.extend("objective_below_threshold:%s" % name for name in objective_failures)
        if not any(objective_passes.values()):
            for metric_name in missing_metrics:
                reasons.append("hardware_data_missing:%s" % metric_name)

        # Optional metrics such as energy are minimized too.  They are hard
        # regression checks when both sides provide a real positive value, but
        # they never become an improvement objective by accident.
        for metric_name in self._NON_TARGET_MINIMIZE_METRICS:
            before = self._positive_metric(baseline_hw, (metric_name,))
            after = self._positive_metric(candidate_hw, (metric_name,))
            if before is not None and after is not None and (after - before) / before > self.max_metric_regression_ratio:
                reasons.append("hardware_regression:%s" % metric_name)

        # A candidate must not be dominated by the current independent front.
        if pareto_front and any(self._dominates(front, evaluation) for front in pareto_front):
            reasons.append("pareto_dominated")

        accepted = not reasons
        return {
            "accepted": accepted,
            "reasons": reasons,
            "quality_floor_ratio": self.quality_floor_ratio,
            "min_efficiency_improvement": self.min_efficiency_improvement,
            "min_latency_improvement": latency_threshold,
            "min_memory_improvement": memory_threshold,
            "min_model_size_improvement": self.min_model_size_improvement,
            "max_metric_regression_ratio": self.max_metric_regression_ratio,
            "improvements": improvements,
            "objective_passes": objective_passes,
            "objective_failures": objective_failures,
            "missing_metrics": sorted(set(missing_metrics)),
            "quality_metrics_checked": quality_metrics_checked,
            "quality_metrics_unavailable": quality_metrics_unavailable,
            "independent_evaluation": True,
        }


@dataclass(frozen=True)
class WanVideoEvaluationConfig:
    wan_config: Path
    evaluation_manifest: Path
    clip_model_path: Path
    output_root: Path
    device: str = "cuda:0"
    pipeline_device: Optional[str] = None
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
    evaluation_split: str = "unspecified"

    def __post_init__(self) -> None:
        if not str(self.wan_config).strip() or not str(self.evaluation_manifest).strip():
            raise ValueError("real Wan evaluation requires wan_config and evaluation_manifest")
        if not str(self.clip_model_path).strip():
            raise ValueError("real Wan evaluation requires clip_model_path")
        if self.max_cases <= 0 or self.seed_count <= 0 or self.sample_frames < 2 or self.latency_warmup < 0 or self.latency_repeats <= 0:
            raise ValueError("real Wan evaluation budgets must be positive")
        if not 0.0 <= float(self.max_quality_drop) <= 1.0:
            raise ValueError("max_quality_drop must be in [0, 1]")
        if self.evaluation_split not in {"unspecified", "calibration", "validation", "test"}:
            raise ValueError("unsupported real Wan evaluation split")

    @property
    def resolved_pipeline_device(self) -> str:
        """Device for the complete Wan pipeline, defaulting to generation device."""

        return str(self.pipeline_device or self.device)

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
            pipeline_device=(str(raw["pipeline_device"]) if raw.get("pipeline_device") is not None else None),
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
            evaluation_split=str(raw.get("evaluation_split", "unspecified")).strip().lower() or "unspecified",
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
                "pipeline_device": config.resolved_pipeline_device,
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
                "generation_device": self.config.device,
                "pipeline_device": self.config.resolved_pipeline_device,
                "dtype": self.config.dtype,
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
            "baseline_quality_metrics": dict(baseline.get("quality_metrics") or {}),
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

    def evaluate_final(self, artifact: CheckpointArtifact) -> Mapping[str, Any]:
        """Evaluate one frozen checkpoint on Test without adaptive side effects.

        This method intentionally does not compare against or update the active
        model.  ``FinalEvaluationRunner`` owns the immutable request envelope;
        this evaluator only measures the requested artifact under the Test
        recipe and returns a result that is explicitly non-promotable.
        """

        if str(self.config.evaluation_split).lower() != "test":
            raise WanEvaluationError("final_test_evaluation_requires_test_split")
        try:
            measured = dict(self._evaluate_artifact(artifact))
        except WanEvaluationError as exc:
            return {
                "valid": False,
                "evaluation_status": "final_test_failed",
                "failure_type": "evaluation_failed",
                "failure_message": str(exc),
                "offline_simulation": False,
                "promotable": False,
                "adaptive_state_mutated": False,
                "evaluation_manifest_sha256": self.manifest_sha256,
                "evaluator_recipe_sha256": self.recipe_digest,
                "evaluator_version": EVALUATOR_VERSION,
                "source_revision": self.config.source_revision,
            }
        return {
            **measured,
            "evaluation_status": "final_test_measured",
            "evaluation_split": "test",
            "promotable": False,
            "adaptive_state_mutated": False,
            "evaluation_manifest_sha256": self.manifest_sha256,
            "evaluator_recipe_sha256": self.recipe_digest,
            "evaluator_version": EVALUATOR_VERSION,
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
                # Prompt text is fixed across seeds for one manifest case.  It
                # is not a compressed component, so encode it once and reuse
                # the tensors for every latency sample and seed.
                prompt_embeddings = self._prepare_prompt_embeddings(
                    pipeline,
                    torch,
                    caption,
                    device=self.config.resolved_pipeline_device,
                )
                for seed in seeds:
                    if (case_id, seed) in completed_keys:
                        continue
                    case_dir = result_dir / "videos" / case_id / ("seed_%d" % seed)
                    video_path = case_dir / "wan.mp4"
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
                            "generation_device": self.config.device,
                            "pipeline_device": self.config.resolved_pipeline_device,
                            "dtype": self.config.dtype,
                            "cases": cases,
                        },
                    )
            if not cases:
                raise WanEvaluationError("evaluation_manifest_invalid: no cases selected")
            scores = [float(item["quality"]["aggregate"]) for item in cases]
            peak_memory_values = [item.get("peak_memory_gb") for item in cases]
            peak_memory_available = all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0
                for value in peak_memory_values
            )
            pipeline_model_size_gb = self._checkpoint_size_gb(artifact.path)
            transformer_model_size_gb = self._component_size_gb(Path(artifact.path) / "transformer")
            fixed_component_sizes_gb = {
                name: self._component_size_gb(Path(artifact.path) / name)
                for name in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae")
            }
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
                    "generation_device": self.config.device,
                    "pipeline_device": self.config.resolved_pipeline_device,
                    "dtype": self.config.dtype,
                    "input_shape": {
                        "height": int(self._wan_spec["height"]),
                        "width": int(self._wan_spec["width"]),
                        "num_frames": int(self._wan_spec["num_frames"]),
                    },
                    "peak_memory_gb_max": max(float(value) for value in peak_memory_values) if peak_memory_available else None,
                    "peak_memory_available": peak_memory_available,
                    "peak_memory_device": self.config.device,
                    "model_size_gb": pipeline_model_size_gb,
                    "pipeline_model_size_gb": pipeline_model_size_gb,
                    "transformer_model_size_gb": transformer_model_size_gb,
                    "fixed_component_size_gb": fixed_component_sizes_gb,
                    "model_size_scope": "pipeline_logical_unique_files",
                    "parameter_count": int(artifact.structure.parameter_count),
                    "transformer_parameter_count": int(artifact.structure.parameter_count),
                    "parameter_count_scope": "transformer",
                    "hardware_measurement_scope": "generation_device_only",
                    "case_count": len(cases),
                },
                "resumed_case_count": resumed_case_count,
                "source_revision": self.config.source_revision,
                "generation_device": self.config.device,
                "pipeline_device": self.config.resolved_pipeline_device,
                "dtype": self.config.dtype,
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

    @staticmethod
    def _device_type(device: Any) -> str:
        """Read torch.device.type while keeping lightweight test doubles valid."""

        value = getattr(device, "type", None)
        if value is None:
            value = str(device).split(":", 1)[0]
        return str(value).strip().lower()

    def _reset_peak_memory(self, torch: Any) -> None:
        try:
            device = self._cuda_device(torch)
            if torch.cuda.is_available() and self._device_type(device) == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        except (AttributeError, RuntimeError):
            pass

    def _peak_memory_gb(self, torch: Any) -> Optional[float]:
        try:
            device = self._cuda_device(torch)
            if self._device_type(device) == "cuda" and torch.cuda.is_available():
                value = float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
                if math.isfinite(value) and value > 0.0:
                    return value
        except (AttributeError, RuntimeError):
            pass
        return None

    @staticmethod
    def _checkpoint_size_gb(path: str) -> float:
        root = Path(path)
        try:
            if not root.exists():
                raise FileNotFoundError(str(root))
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
        except OSError as exc:
            raise WanEvaluationError("checkpoint_size_failed: %s" % exc) from exc
        return float(total) / float(1024**3)

    @classmethod
    def _component_size_gb(cls, path: Path) -> Optional[float]:
        if not path.exists() and not path.is_symlink():
            return None
        return cls._checkpoint_size_gb(str(path))

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
            pipeline = self._move_pipeline_to_device(pipeline, self.config.resolved_pipeline_device)
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
    def _move_pipeline_to_device(pipeline: Any, device: str) -> Any:
        """Place all inference components together when the pipeline supports it."""

        mover = getattr(pipeline, "to", None)
        if callable(mover):
            moved = mover(device)
            if moved is not None:
                pipeline = moved
        else:
            for component_name in ("transformer", "vae", "text_encoder", "text_encoder_2"):
                component = getattr(pipeline, component_name, None)
                component_mover = getattr(component, "to", None)
                if callable(component_mover):
                    component_mover(device)
        transformer = getattr(pipeline, "transformer", None)
        evaluator = getattr(transformer, "eval", None)
        if callable(evaluator):
            evaluator()
        return pipeline

    @staticmethod
    def _prepare_prompt_embeddings(
        pipeline: Any,
        torch: Any,
        caption: str,
        *,
        device: str,
    ) -> Optional[tuple[Any, Any]]:
        """Encode fixed text once per case; it is not a compressed component."""

        getter = getattr(pipeline, "_get_t5_prompt_embeds", None)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if not callable(getter) or text_encoder is None:
            return None
        text_dtype = getattr(text_encoder, "dtype", getattr(torch, "bfloat16"))
        return (
            getter(caption, device=torch.device(device), dtype=text_dtype),
            getter("", device=torch.device(device), dtype=text_dtype),
        )

    @staticmethod
    def _module_device(module: Any, torch: Any, fallback: str) -> Any:
        try:
            return next(module.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device(fallback)

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
            prompt_embeds = pipeline._get_t5_prompt_embeds(
                caption,
                device=torch.device(self.config.resolved_pipeline_device),
                dtype=text_dtype,
            )
            negative_prompt_embeds = pipeline._get_t5_prompt_embeds(
                "",
                device=torch.device(self.config.resolved_pipeline_device),
                dtype=text_dtype,
            )
        generator = torch.Generator(device=str(device)).manual_seed(int(seed))
        requested_steps = int(self._wan_spec["num_inference_steps"])
        transformer = getattr(pipeline, "transformer", None)
        sampling_policy = getattr(getattr(transformer, "config", None), "harness4h3_sampling_step", None)
        if isinstance(sampling_policy, Mapping):
            requested_steps = min(requested_steps, int(sampling_policy.get("target_steps", requested_steps)))
        with torch.inference_mode():
            output = pipeline(
                prompt=None,
                prompt_embeds=prompt_embeds.to(device),
                negative_prompt_embeds=negative_prompt_embeds.to(device),
                height=int(self._wan_spec["height"]),
                width=int(self._wan_spec["width"]),
                num_frames=int(self._wan_spec["num_frames"]),
                num_inference_steps=requested_steps,
                guidance_scale=float(self._wan_spec["guidance_scale"]),
                generator=generator,
                output_type="latent",
            )
            vae_device = self._module_device(
                pipeline.vae,
                torch,
                self.config.resolved_pipeline_device,
            )
            latents = output.frames.to(vae_device, dtype=pipeline.vae.dtype)
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
