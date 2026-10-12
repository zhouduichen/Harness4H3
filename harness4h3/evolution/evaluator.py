"""Measured Wan video evaluation for evolutionary compression rounds."""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact


class WanEvaluationError(RuntimeError):
    """A real evaluation could not produce trusted evidence."""


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

    def __post_init__(self) -> None:
        if not str(self.wan_config).strip() or not str(self.evaluation_manifest).strip():
            raise ValueError("real Wan evaluation requires wan_config and evaluation_manifest")
        if not str(self.clip_model_path).strip():
            raise ValueError("real Wan evaluation requires clip_model_path")
        if self.max_cases <= 0 or self.seed_count <= 0 or self.sample_frames < 2:
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
        self._manifest = self._load_manifest(config.evaluation_manifest)
        self._wan_spec = self._load_wan_spec(config.wan_config)

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
            }
        baseline_quality = float(baseline["quality_score"])
        candidate_quality = float(candidate["quality_score"])
        quality_regression = candidate_quality < baseline_quality - float(self.config.max_quality_drop)
        candidate_hardware = dict(candidate.get("hardware_metrics") or {})
        baseline_hardware = dict(baseline.get("hardware_metrics") or {})
        candidate_latency = float(candidate_hardware.get("inference_wall_time_s_mean", float("inf")))
        baseline_latency = float(baseline_hardware.get("inference_wall_time_s_mean", float("inf")))
        latency_ratio = candidate_latency / baseline_latency if baseline_latency > 0 and math.isfinite(baseline_latency) else 1.0
        feasible = bool(candidate.get("valid", False)) and not quality_regression
        search_score = candidate_quality + max(0.0, 1.0 - latency_ratio) * 0.05
        return {
            **dict(candidate),
            "baseline_quality_score": baseline_quality,
            "quality_delta": candidate_quality - baseline_quality,
            "latency_ratio": latency_ratio,
            "feasible": feasible,
            "critical_regression": quality_regression or not bool(candidate.get("valid", False)),
            "offline_simulation": False,
            "search_score": search_score,
        }

    def _evaluate_artifact(self, artifact: CheckpointArtifact) -> Mapping[str, Any]:
        cached = self._cache.get(artifact.sha256)
        if cached is not None:
            return cached
        result_dir = self.config.output_root / "evaluations" / artifact.sha256
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
        pipeline = None
        try:
            pipeline, torch = self._load_pipeline(artifact.path)
            quality = self._get_quality_backend()
            cases = []
            for raw_case in self._manifest[: self.config.max_cases]:
                case_id = str(raw_case.get("case_id", "case"))
                caption = str(raw_case.get("caption", "")).strip()
                seeds = tuple(int(seed) for seed in raw_case.get("seeds", ()))[: self.config.seed_count]
                if not caption or not seeds:
                    raise WanEvaluationError("evaluation_manifest_invalid: case requires caption and seeds")
                for seed in seeds:
                    case_dir = result_dir / "videos" / case_id / ("seed_%d" % seed)
                    video_path = case_dir / "wan.mp4"
                    started = time.perf_counter()
                    frames = self._generate(pipeline, torch, caption, seed)
                    inference_wall = time.perf_counter() - started
                    self._write_video(frames, video_path, float(self._wan_spec["native_fps"]))
                    validity = self._validate_video(video_path, frames)
                    evidence = quality.evaluate(video_path, caption)
                    cases.append(
                        {
                            "case_id": case_id,
                            "seed": seed,
                            "caption": caption,
                            "video_path": str(video_path),
                            "generation_wall_time_s": inference_wall,
                            "validity": validity,
                            "quality": evidence.to_dict(),
                        }
                    )
            if not cases:
                raise WanEvaluationError("evaluation_manifest_invalid: no cases selected")
            scores = [float(item["quality"]["aggregate"]) for item in cases]
            result = {
                "status": "success",
                "valid": all(bool(item["validity"]["valid"]) for item in cases),
                "promotable": all(bool(item["validity"]["valid"]) for item in cases),
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
                    "inference_wall_time_s_total": sum(float(item["generation_wall_time_s"]) for item in cases),
                    "case_count": len(cases),
                },
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

    def _get_quality_backend(self) -> Any:
        if self._quality_backend is None:
            from ..student.quality import ClipTemporalQualityBackend

            self._quality_backend = ClipTemporalQualityBackend(
                self.config.clip_model_path,
                device="cpu",
                sample_frames=self.config.sample_frames,
            )
        return self._quality_backend

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

    def _generate(self, pipeline: Any, torch: Any, caption: str, seed: int) -> Any:
        device = torch.device(self.config.device)
        text_dtype = getattr(pipeline.text_encoder, "dtype", getattr(torch, self.config.dtype))
        prompt_embeds = pipeline._get_t5_prompt_embeds(caption, device=torch.device("cpu"), dtype=text_dtype)
        negative_embeds = pipeline._get_t5_prompt_embeds("", device=torch.device("cpu"), dtype=text_dtype)
        generator = torch.Generator(device=str(device)).manual_seed(int(seed))
        with torch.inference_mode():
            output = pipeline(
                prompt=None,
                prompt_embeds=prompt_embeds.to(device),
                negative_prompt_embeds=negative_embeds.to(device),
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
