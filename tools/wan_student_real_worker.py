#!/usr/bin/env python3
"""Run the real pretrained Wan W0 baseline on an experiment host.

This worker consumes the immutable H3 evaluation manifest for captions and
seeds, but it does not consume H3 latent tensors. The Wan checkpoint remains
the actual Student root and every output records ``pretrained=true`` and
``lineage_root=W0000``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.evaluation_manifest import EvaluationManifest
from harness4h3.student.evaluator import StudentEvaluator
from harness4h3.student.quality import ClipTemporalQualityBackend, QualityBackendUnavailable
from harness4h3.student.wan import WanStudentError, generate_wan_clip, load_pretrained_wan, load_wan_spec, write_wan_video


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_size_bytes(path: str | None) -> int | None:
    if not path:
        return None
    source = Path(path)
    if source.is_file():
        return int(source.stat().st_size)
    if source.is_dir():
        return int(sum(item.stat().st_size for item in source.rglob("*") if item.is_file()))
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-config", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--phase", default="W0_pretrained_baseline")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--quality-backend", choices=("clip_temporal", "none"), default="clip_temporal")
    parser.add_argument("--clip-model-path", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--target-frames", type=int, default=720)
    parser.add_argument("--target-duration-s", type=float, default=30.0)
    parser.add_argument("--target-window-count", type=int, default=7)
    parser.add_argument("--target-overlap-latent-frames", type=int, default=6)
    parser.add_argument("--target-window-stride-latent-frames", type=int, default=30)
    args = parser.parse_args(argv)
    result_path = Path(args.result).resolve()
    output_root = Path(args.output).resolve()
    try:
        if args.max_cases <= 0 or args.seed_count <= 0:
            raise ValueError("max-cases and seed-count must be positive")
        if args.phase != "W0_pretrained_baseline":
            raise ValueError("only W0_pretrained_baseline is implemented")
        if args.device == "auto":
            raise ValueError("device must be fixed for the real remote experiment")
        if args.target_fps <= 0 or args.target_frames <= 0 or args.target_duration_s <= 0:
            raise ValueError("target video contract values must be positive")
        spec = load_wan_spec(Path(args.wan_config))
        manifest = EvaluationManifest.from_path(Path(args.evaluation_manifest))
        pipeline, device = load_pretrained_wan(
            spec,
            device=args.device,
            local_files_only=bool(args.local_files_only),
        )
        quality_backend = None
        if args.quality_backend == "clip_temporal":
            if not args.clip_model_path:
                raise ValueError("--clip-model-path is required for clip_temporal")
            # The real host may run the H3 Teacher/Student on GPUs 0-3. Keep
            # quality evaluation on CPU so W0 cannot silently steal another
            # rank's GPU after Wan generation completes.
            quality_backend = ClipTemporalQualityBackend(args.clip_model_path, device="cpu", sample_frames=4)
        evaluator = StudentEvaluator(max_frames=max(4096, int(spec.num_frames)))
        cases = []
        latencies = []
        model_size = _model_size_bytes(spec.pretrained_path)
        for case in manifest.cases[: int(args.max_cases)]:
            for seed in case.seeds[: int(args.seed_count)]:
                case_dir = output_root / case.case_id / ("seed_%d" % int(seed))
                video_path = case_dir / "wan.mp4"
                started = time.perf_counter()
                frames, generation = generate_wan_clip(
                    pipeline,
                    spec,
                    case.caption,
                    seed=int(seed),
                    device=device,
                )
                write_wan_video(frames, video_path, spec.native_fps)
                latency = time.perf_counter() - started
                latencies.append(latency)
                hardware = {
                    **generation,
                    "device": str(device),
                    "latency_s": float(latency),
                    "generation_latency_s": float(latency),
                    "written_fps": float(spec.native_fps),
                    "offline_simulation": False,
                    "model_size_gb": float(model_size) / float(1024**3) if model_size is not None else None,
                }
                preliminary = evaluator.evaluate(video_path, hardware=hardware)
                if not preliminary.valid:
                    raise WanStudentError(preliminary.failure_code or "video_invalid", preliminary.message)
                quality = {"score": None, "score_type": "none"}
                if quality_backend is not None:
                    evidence = quality_backend.evaluate(video_path, case.caption)
                    quality = {
                        "score": evidence.aggregate,
                        "score_type": evidence.backend,
                        "semantic": evidence.semantic,
                        "semantic_mean": evidence.semantic_mean,
                        "semantic_p10": evidence.semantic_p10,
                        "temporal": evidence.temporal,
                        "motion": evidence.motion,
                    }
                cases.append({
                    "case_id": case.case_id,
                    "seed": int(seed),
                    "caption": case.caption,
                    "video_path": str(video_path),
                    "generation": generation,
                    "hardware": hardware,
                    "validity": dict(preliminary.validity),
                    "quality": quality,
                })
        scores = [float(item["quality"]["score"]) for item in cases if item["quality"]["score"] is not None]
        result = {
            "status": "success",
            "pretrained": True,
            "offline_simulation": False,
            "lineage_root": spec.lineage_root,
            "model_id": spec.model_id,
            "pretrained_path": spec.pretrained_path,
            "evaluation_manifest": str(Path(args.evaluation_manifest).resolve()),
            "evaluation_manifest_digest": manifest.digest,
            "native_fps": spec.native_fps,
            "target_output_fps": spec.output_fps,
            "phase": args.phase,
            "target_contract": {
                "fps": args.target_fps,
                "frames": args.target_frames,
                "duration_s": args.target_duration_s,
                "window_count": args.target_window_count,
                "overlap_latent_frames": args.target_overlap_latent_frames,
                "window_stride_latent_frames": args.target_window_stride_latent_frames,
                "continuation_verified": False,
            },
            "case_count": len(cases),
            "latency_s_median": statistics.median(latencies) if latencies else None,
            "quality_score": statistics.mean(scores) if scores else None,
            "quality_backend": args.quality_backend,
            "cases": cases,
        }
        _write(result_path, result)
        return 0
    except (WanStudentError, QualityBackendUnavailable, OSError, TypeError, ValueError, RuntimeError) as exc:
        _write(
            result_path,
            {
                "status": "failed",
                "pretrained": True,
                "error_code": getattr(exc, "code", "wan_real_worker_failed"),
                "message": str(exc),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
