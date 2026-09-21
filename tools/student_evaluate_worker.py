#!/usr/bin/env python3
"""Fixed Student checkpoint→latent sampling→H3 VAE→video evaluator."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness4h3.student.evaluator import StudentEvaluation, StudentEvaluator
from harness4h3.student.evaluation_manifest import EvaluationManifest
from harness4h3.student.gpu import GPUResourceUnavailable, release_controller_handoff, select_free_cuda_device
from harness4h3.student.inference import StudentGenerationError, generate_video
from harness4h3.student.metrics import MetricVerifierBank
from harness4h3.student.proposal import StudentProposal, StudentTarget
from harness4h3.student.quality import ClipTemporalQualityBackend, QualityBackendUnavailable


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _failure(path: Path, code: str, message: str) -> StudentEvaluation:
    return StudentEvaluation(
        valid=False,
        promotable=False,
        failure_code=code,
        message=message,
        video_path=str(path),
        validity={},
        quality_metrics={},
        hardware={},
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1))
    return ordered[index]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate and evaluate one Student video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--vae-name", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wait-for-gpu-s", type=int, default=600)
    parser.add_argument("--min-free-memory-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--black-frame-ratio-threshold", type=float, default=0.0)
    parser.add_argument("--energy-j", type=float, default=None, help="optional measured generation energy; never inferred")
    parser.add_argument("--evaluation-manifest", default="")
    parser.add_argument("--clip-model-path", default="")
    parser.add_argument("--quality-device", default="cpu", help="device for semantic/temporal quality; CPU avoids competing with H3 VAE memory")
    parser.add_argument("--quality-backend", choices=("clip_temporal", "structural_proxy"), default="structural_proxy")
    parser.add_argument("--controller-hold-file", default="")
    parser.add_argument("--controller-release-file", default="")
    parser.add_argument("--controller-worker-lease-file", default="")
    parser.add_argument("--release-controller-handoff", action="store_true")
    args = parser.parse_args(argv)
    output_dir = Path(args.output).resolve()
    result_path = Path(args.result).resolve()
    video_path = output_dir / "student-generation.mp4"
    started = time.perf_counter()
    try:
        manifest = json.loads((output_dir / "compile_manifest.json").read_text(encoding="utf-8"))
        proposal = StudentProposal.from_dict(manifest["proposal"])
        target = StudentTarget(**dict(manifest["target"]))
        selected_device = (
            select_free_cuda_device(args.min_free_memory_gb, args.wait_for_gpu_s)
            if args.device == "auto"
            else args.device
        )
        training = {}
        training_path = output_dir / "training-result.json"
        if training_path.is_file():
            raw_training = json.loads(training_path.read_text(encoding="utf-8"))
            if isinstance(raw_training, Mapping):
                training = dict(raw_training)
        evaluator = StudentEvaluator(black_frame_ratio_threshold=args.black_frame_ratio_threshold)
        manifest = EvaluationManifest.from_path(Path(args.evaluation_manifest)) if args.evaluation_manifest else None
        if args.quality_backend == "clip_temporal" and manifest is None:
            raise QualityBackendUnavailable("clip_temporal evaluation requires --evaluation-manifest")
        quality_backend = ClipTemporalQualityBackend(args.clip_model_path, device=args.quality_device) if args.quality_backend == "clip_temporal" else None
        cases = []
        if manifest is not None:
            for case in manifest.cases:
                for seed in case.seeds:
                    cases.append((case.case_id, Path(case.cache_path), case.caption, int(seed)))
        else:
            cases.append(("default", None, "", int(args.seed)))
        case_results = []
        case_validity = []
        for case_id, cache_path, caption, seed in cases:
            case_path = output_dir / "student-generation" / ("%s-%d.mp4" % (case_id, seed))
            generation = generate_video(
                proposal,
                Path(args.checkpoint),
                target,
                Path(args.comfyui_root),
                Path(args.cache_dir),
                args.vae_name,
                case_path,
                device=selected_device,
                cache_path=cache_path,
                seed=seed,
            )
            hardware_case = {
                **generation,
                "generation_wall_time_s": float(generation["latency_s"]),
                "training_peak_memory_gb": training.get("peak_memory_gb"),
                "quantization": training.get("quantization"),
                "model_size_gb": float(generation["checkpoint_bytes"]) / float(1024**3),
                "energy_j": args.energy_j,
            }
            preliminary = evaluator.evaluate(case_path, hardware=hardware_case)
            if not preliminary.valid:
                raise StudentGenerationError(preliminary.failure_code or "video_invalid", preliminary.message)
            if quality_backend is not None:
                quality_evidence = quality_backend.evaluate(case_path, caption)
                quality = {
                    "score": quality_evidence.aggregate,
                    "score_type": quality_evidence.backend,
                    "semantic": quality_evidence.semantic,
                    "temporal": quality_evidence.temporal,
                    "motion": quality_evidence.motion,
                }
            else:
                sample_mean = float(preliminary.validity.get("sample_mean", 0.0))
                black_ratio = float(preliminary.validity.get("black_frame_ratio", 1.0))
                structural_score = max(0.0, min(1.0, 0.6 * (1.0 - black_ratio) + 0.4 * min(sample_mean / 128.0, 1.0)))
                quality = {
                    "score": structural_score,
                    "score_type": "structural_proxy",
                    "sample_mean": sample_mean,
                    "black_frame_ratio": black_ratio,
                }
            case_results.append({"case_id": case_id, "seed": seed, "quality": quality, "hardware": hardware_case, "video_path": str(case_path.resolve())})
            case_validity.append({"case_id": case_id, "seed": seed, **dict(preliminary.validity)})
        qualities = [float(item["quality"]["score"]) for item in case_results]
        latencies = [float(item["hardware"]["latency_s"]) for item in case_results]
        peak_memory = max(float(item["hardware"].get("peak_memory_gb", 0.0)) for item in case_results)
        representative_hardware = dict(case_results[0]["hardware"])
        hardware = {
            **representative_hardware,
            "generation_wall_time_s": float(sum(latencies)),
            "latency_s": float(statistics.median(latencies)),
            "latency_p95_s": _p95(latencies),
            "latency_ms": float(statistics.median(latencies)) * 1000.0,
            "peak_memory_gb": peak_memory,
            "training_peak_memory_gb": training.get("peak_memory_gb"),
            "quantization": training.get("quantization"),
            "model_size_gb": float(representative_hardware["checkpoint_bytes"]) / float(1024**3),
            "energy_j": args.energy_j,
            "case_count": len(case_results),
            "evaluation_manifest_digest": manifest.digest if manifest is not None else None,
        }
        aggregate_score = float(statistics.mean(qualities))
        quality = {
            "score": aggregate_score,
            "score_type": case_results[0]["quality"].get("score_type", "structural_proxy"),
            "case_count": len(case_results),
            "cases": case_results,
            "semantic": float(statistics.mean([float(item["quality"].get("semantic", item["quality"]["score"])) for item in case_results])),
            "temporal": float(statistics.mean([float(item["quality"].get("temporal", 0.0)) for item in case_results])),
            "motion": float(statistics.mean([float(item["quality"].get("motion", 0.0)) for item in case_results])),
        }
        metric_evidence = MetricVerifierBank().evaluate(quality, hardware)
        metric_payload = metric_evidence.to_dict()
        metric_payload["optimization_metrics"] = {
            "quality": aggregate_score,
            "latency": hardware["latency_ms"],
            "memory": hardware["peak_memory_gb"],
            "size": hardware["model_size_gb"],
            **({"energy": float(args.energy_j)} if args.energy_j is not None else {}),
        }
        evaluation = StudentEvaluation(
            valid=True,
            promotable=metric_evidence.reward is not None,
            failure_code=None if metric_evidence.reward is not None else "metric_evidence_missing",
            message="evaluation_ok" if metric_evidence.reward is not None else "required continuous metric evidence is missing or invalid",
            video_path=str(case_results[0]["video_path"]),
            validity={"case_count": len(case_validity), "all_valid": True, "cases": case_validity},
            quality_score=aggregate_score,
            quality_metrics=quality,
            hardware=hardware,
            metric_evidence=metric_payload,
            reward=metric_evidence.reward,
            reward_terms=metric_evidence.reward_terms,
        )
    except GPUResourceUnavailable as exc:
        evaluation = _failure(video_path, "resource_unavailable", str(exc))
    except StudentGenerationError as exc:
        evaluation = _failure(video_path, exc.code, exc.message)
    except QualityBackendUnavailable as exc:
        evaluation = _failure(video_path, "quality_evaluator_unavailable", str(exc))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        evaluation = _failure(video_path, "student_evaluator_failed", str(exc))
    _write(result_path, evaluation.to_dict())
    if args.release_controller_handoff:
        release_controller_handoff(
            args.controller_hold_file or None,
            args.controller_release_file or None,
            args.controller_worker_lease_file or None,
        )
    return 0 if evaluation.promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
