#!/usr/bin/env python3
"""Trusted server-side H3 baseline calibration for Student optimization."""

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
from harness4h3.student.gpu import select_free_cuda_device
from harness4h3.student.inference import decode_video_latent, load_h3_cache_item, write_video
from harness4h3.student.quality import ClipTemporalQualityBackend, QualityBackendUnavailable
from harness4h3.student.proposal import StudentTarget


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate H3 teacher quality on a fixed evaluation manifest")
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--vae-name", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--quality-device", default="cpu", help="device for semantic/temporal quality; CPU avoids competing with H3 VAE memory")
    parser.add_argument("--latent-channels", type=int, default=24)
    parser.add_argument("--latent-frames", type=int, default=5)
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=16)
    parser.add_argument("--condition-dim", type=int, default=5120)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wait-for-gpu-s", type=int, default=600)
    args = parser.parse_args(argv)
    output_dir = Path(args.output).resolve()
    result_path = Path(args.result).resolve()
    try:
        import torch

        manifest = EvaluationManifest.from_path(Path(args.evaluation_manifest))
        device_name = select_free_cuda_device(8.0, args.wait_for_gpu_s) if args.device == "auto" else args.device
        device = torch.device(device_name)
        quality_backend = ClipTemporalQualityBackend(args.clip_model_path, device=args.quality_device)
        target = StudentTarget(
            latent_channels=args.latent_channels,
            latent_frames=args.latent_frames,
            latent_height=args.latent_height,
            latent_width=args.latent_width,
            condition_dim=args.condition_dim,
        )
        cases = []
        latencies = []
        for case in manifest.cases:
            cache_item = load_h3_cache_item(Path(args.cache_dir), target, device, torch.float32, cache_path=Path(case.cache_path))
            for seed in case.seeds:
                started = time.perf_counter()
                latent = torch.as_tensor(cache_item["latent"], device=device)
                frames = decode_video_latent(Path(args.comfyui_root), args.vae_name, latent)
                video_path = output_dir / "teacher-baseline" / ("%s-%d.mp4" % (case.case_id, seed))
                write_video(frames, video_path)
                latency = time.perf_counter() - started
                evidence = quality_backend.evaluate(video_path, case.caption)
                latencies.append(latency)
                cases.append(
                    {
                        "case_id": case.case_id,
                        "seed": int(seed),
                        "video_path": str(video_path),
                        "quality": evidence.to_dict(),
                        "latency_s": latency,
                        "frame_count": int(frames.shape[0]),
                        "resolution": [int(frames.shape[2]), int(frames.shape[1])],
                    }
                )
                del frames, latent
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        quality = float(statistics.mean(float(item["quality"]["aggregate"]) for item in cases))
        hardware = {
            "latency_s": float(statistics.median(latencies)),
            "latency_p95_s": _p95(latencies),
            "latency_ms": float(statistics.median(latencies)) * 1000.0,
            "case_count": len(cases),
            "device": device_name,
            "evaluation_manifest_digest": manifest.digest,
        }
        payload = {
            "status": "success",
            "manifest_digest": manifest.digest,
            "quality_backend": "clip_temporal",
            "quality": {"score": quality, "score_type": "clip_temporal", "cases": cases},
            "hardware": hardware,
            "optimization_metrics": {
                "quality": quality,
                "latency": hardware["latency_ms"],
                "memory": 0.0,
                "size": 0.0,
            },
        }
        _write(result_path, payload)
        _write(output_dir / "teacher-baseline.json", payload)
        return 0
    except QualityBackendUnavailable as exc:
        _write(result_path, {"status": "failed", "failure_code": "quality_evaluator_unavailable", "message": str(exc)})
        return 1
    except Exception as exc:
        _write(result_path, {"status": "failed", "failure_code": "teacher_baseline_failed", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
