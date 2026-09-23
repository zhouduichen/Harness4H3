#!/usr/bin/env python3
"""Trusted server-side H3 baseline calibration for Student optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness4h3.student.evaluation_manifest import EvaluationManifest
from harness4h3.student.gpu import (
    acquire_controller_handoff,
    publish_worker_gpu_lease,
    release_controller_handoff,
    release_worker_gpu_lease,
    select_free_cuda_device,
    select_teacher_gpu_devices,
)
from harness4h3.student.inference import (
    decode_video_latent,
    load_h3_cache_item,
    sample_h3_latent_with_predictor,
    write_video,
)
from harness4h3.student.teacher_service import TeacherService, TeacherServiceHandle
from harness4h3.student.quality import ClipTemporalQualityBackend, QualityBackendUnavailable
from harness4h3.student.proposal import StudentTarget
from harness4h3.campaign.base import sha256_path


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate H3 teacher quality on a fixed evaluation manifest")
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--teacher", default="", help="authentic MiniMax-H3 teacher checkpoint")
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
    parser.add_argument("--decode-device", default="", help="explicit CUDA device for VAE/quality decoding after Teacher teardown")
    parser.add_argument("--wait-for-gpu-s", type=int, default=600)
    parser.add_argument("--teacher-world-size", type=int, default=3)
    parser.add_argument("--teacher-rank-min-free-memory-gb", type=float, default=20.0)
    parser.add_argument("--teacher-devices", default="", help="comma-separated cuda:N values; otherwise select three GPUs")
    parser.add_argument("--controller-hold-file", default="")
    parser.add_argument("--controller-release-file", default="")
    parser.add_argument("--controller-worker-lease-file", default="")
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument(
        "--teacher-service-batch-size",
        type=int,
        default=2,
        help="close/restart the FSDP Teacher after this many manifest samples to bound allocator growth",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--baseline-kind",
        choices=("h3_teacher_generation_baseline", "reference_reconstruction_baseline"),
        default="h3_teacher_generation_baseline",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output).resolve()
    result_path = Path(args.result).resolve()
    teacher_service: TeacherServiceHandle | None = None
    handoff_acquired = False
    try:
        import torch

        manifest = EvaluationManifest.from_path(Path(args.evaluation_manifest))
        evaluation_manifest_digest = "sha256:" + hashlib.sha256(Path(args.evaluation_manifest).read_bytes()).hexdigest()
        if args.baseline_kind == "h3_teacher_generation_baseline" and not args.teacher:
            raise ValueError("--teacher is required for h3_teacher_generation_baseline")
        teacher_checkpoint_sha256 = sha256_path(Path(args.teacher)) if args.teacher else None
        clip_model_hash = sha256_path(Path(args.clip_model_path))
        if args.baseline_kind == "h3_teacher_generation_baseline" and args.teacher_world_size != TeacherService.REQUIRED_WORLD_SIZE:
            raise ValueError("h3_teacher_generation_baseline requires teacher-world-size=3")
        if args.baseline_kind == "h3_teacher_generation_baseline":
            acquire_controller_handoff(
                args.controller_hold_file or None,
                args.controller_release_file or None,
                args.controller_worker_lease_file or None,
            )
            handoff_acquired = True
            if args.teacher_devices:
                teacher_devices = tuple(item.strip() for item in args.teacher_devices.split(",") if item.strip())
            elif args.device == "auto":
                teacher_devices = select_teacher_gpu_devices(
                    args.teacher_world_size,
                    args.teacher_rank_min_free_memory_gb,
                    args.wait_for_gpu_s,
                )
            else:
                raise ValueError("explicit H3 baseline device requires --teacher-devices with three cuda:N values")
        else:
            if args.teacher_devices:
                teacher_devices = tuple(item.strip() for item in args.teacher_devices.split(",") if item.strip())
            else:
                teacher_devices = (select_free_cuda_device(8.0, args.wait_for_gpu_s) if args.device == "auto" else args.device,)
        if args.baseline_kind == "h3_teacher_generation_baseline" and len(teacher_devices) != 3:
            raise ValueError("h3_teacher_generation_baseline requires three distinct Teacher GPUs")
        if len(set(teacher_devices)) != len(teacher_devices) or any(not item.startswith("cuda:") for item in teacher_devices):
            raise ValueError("Teacher devices must be distinct explicit cuda:N values")
        if handoff_acquired:
            publish_worker_gpu_lease(args.controller_worker_lease_file or None, teacher_devices)
        device_name = teacher_devices[0]
        device = torch.device(device_name)
        quality_backend = ClipTemporalQualityBackend(args.clip_model_path, device=args.quality_device)
        target = StudentTarget(
            latent_channels=args.latent_channels,
            latent_frames=args.latent_frames,
            latent_height=args.latent_height,
            latent_width=args.latent_width,
            condition_dim=args.condition_dim,
        )
        sampled_cases = []
        cases = []
        latencies = []
        peak_memory_gb = 0.0
        model_load_time_s = 0.0
        teacher_forward_count = 0
        teacher_peak_memory_gb = 0.0
        teacher_ranks_used: set[int] = set()
        teacher_sharded = False
        teacher_service_restarts = 0

        def close_teacher_service() -> None:
            nonlocal teacher_forward_count, teacher_peak_memory_gb
            nonlocal teacher_sharded, teacher_service, teacher_service_restarts
            if teacher_service is None:
                return
            teacher_forward_count += int(teacher_service.forward_count)
            teacher_peak_memory_gb = max(teacher_peak_memory_gb, float(teacher_service.teacher_peak_memory_gb))
            teacher_ranks_used.update(int(rank) for rank in teacher_service.ranks_used)
            teacher_sharded = teacher_sharded or bool(teacher_service.sharded)
            teacher_service.close()
            teacher_service = None
            teacher_service_restarts += 1
            torch.cuda.empty_cache()

        def start_teacher_service() -> None:
            nonlocal model_load_time_s, teacher_service
            started = time.perf_counter()
            teacher_service = TeacherService(
                Path(args.teacher),
                Path(args.comfyui_root),
                teacher_devices,
                dtype=torch.bfloat16,
            ).start()
            model_load_time_s += time.perf_counter() - started

        latent_dir = output_dir / "teacher-baseline-latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        if args.baseline_kind == "h3_teacher_generation_baseline":
            start_teacher_service()
        total_samples = sum(len(case.seeds) for case in manifest.cases)
        if args.teacher_service_batch_size <= 0:
            raise ValueError("teacher-service-batch-size must be positive")
        sample_index = 0
        for case in manifest.cases:
            cache_key = str(Path(case.cache_path).resolve())
            for seed in case.seeds:
                latent_path = latent_dir / ("%s-%d.pt" % (case.case_id, seed))
                cache_item = load_h3_cache_item(Path(args.cache_dir), target, device, torch.bfloat16, cache_path=Path(case.cache_path))
                if args.baseline_kind == "h3_teacher_generation_baseline" and cache_item.get("audio_latent") is None:
                    raise ValueError("H3 teacher baseline requires audio latent in cache item %s" % cache_key)
                if args.baseline_kind == "h3_teacher_generation_baseline":
                    latent, sampling_latency = sample_h3_latent_with_predictor(
                        teacher_service,
                        cache_item["prompt"],
                        tuple(cache_item["latent"].shape),
                        tuple(cache_item["audio_latent"].shape),
                        device,
                        sampling_steps=args.sampling_steps,
                        seed=int(seed),
                        dtype=torch.bfloat16,
                    )
                else:
                    # This is intentionally a separate upper-reference: it
                    # reconstructs cached latents and never claims to be an
                    # executable H3 generation baseline.
                    latent = cache_item["latent"].to(device=device, dtype=torch.bfloat16)
                    sampling_latency = 0.0
                torch.save({"latent": latent.detach().to(device="cpu").contiguous()}, latent_path)
                sampled_cases.append(
                    {
                        "case_id": case.case_id,
                        "caption": case.caption,
                        "seed": int(seed),
                        "latent_path": str(latent_path),
                        "sampling_latency_s": float(sampling_latency),
                    }
                )
                del cache_item, latent
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                sample_index += 1
                if (
                    args.baseline_kind == "h3_teacher_generation_baseline"
                    and sample_index < total_samples
                    and sample_index % args.teacher_service_batch_size == 0
                ):
                    close_teacher_service()
                    start_teacher_service()

        close_teacher_service()
        teacher_ranks_used = sorted(teacher_ranks_used)
        peak_memory_gb = max(peak_memory_gb, teacher_peak_memory_gb)
        if handoff_acquired:
            # Keep the outer controller hold during decode/CLIP, but release
            # the Teacher's lease and all FSDP ranks before loading the VAE.
            release_worker_gpu_lease(args.controller_worker_lease_file or None)

        # Do not decode on teacher_devices[0].  FSDP teardown and VAE loading
        # can otherwise race on the same allocator even after close() returns.
        # Re-select only after the Teacher lease is released so the decode has
        # an independently measured GPU-capacity boundary.
        if args.decode_device:
            decode_device = torch.device(args.decode_device)
            if decode_device.type != "cuda":
                raise ValueError("decode-device must be an explicit cuda:N device")
        else:
            decode_device = device
        if not args.decode_device and args.device == "auto":
            decode_device = torch.device(select_free_cuda_device(8.0, args.wait_for_gpu_s))

        for sampled in sampled_cases:
            video_path = output_dir / "teacher-baseline" / ("%s-%d.mp4" % (sampled["case_id"], sampled["seed"]))
            payload = torch.load(Path(sampled["latent_path"]), map_location="cpu", weights_only=False)
            latent = payload["latent"].to(device=decode_device, dtype=torch.bfloat16)
            decode_started = time.perf_counter()
            frames = decode_video_latent(Path(args.comfyui_root), args.vae_name, latent)
            write_video(frames, video_path)
            decode_latency = time.perf_counter() - decode_started
            sampling_latency = float(sampled["sampling_latency_s"])
            latency = float(sampling_latency + decode_latency)
            if decode_device.type == "cuda":
                peak_memory_gb = max(peak_memory_gb, float(torch.cuda.max_memory_allocated(decode_device)) / float(1024 ** 3))
            quality_started = time.perf_counter()
            evidence = quality_backend.evaluate(video_path, sampled["caption"]).to_dict()
            quality_latency = time.perf_counter() - quality_started
            latencies.append(latency)
            cases.append(
                {
                    "case_id": sampled["case_id"],
                    "seed": int(sampled["seed"]),
                    "video_path": str(video_path),
                    "quality": evidence,
                    "latency_s": latency,
                    "sampling_latency_s": float(sampling_latency),
                    "sampling_latency": float(sampling_latency),
                    "decode_latency_s": float(decode_latency),
                    "decode_latency": float(decode_latency),
                    "quality_latency_s": float(quality_latency),
                    "quality_latency": float(quality_latency),
                    "model_load_time_s": float(model_load_time_s),
                    "model_load_time": float(model_load_time_s),
                    "peak_memory": float(peak_memory_gb),
                    "generation_latency_s": latency,
                    "frame_count": int(frames.shape[0]),
                    "resolution": [int(frames.shape[2]), int(frames.shape[1])],
                }
            )
            del frames, latent, payload
            if decode_device.type == "cuda":
                torch.cuda.empty_cache()
        if handoff_acquired:
            release_controller_handoff(
                args.controller_hold_file or None,
                args.controller_release_file or None,
                args.controller_worker_lease_file or None,
            )
        quality = float(statistics.mean(float(item["quality"]["aggregate"]) for item in cases))
        hardware = {
            "latency_s": float(statistics.median(latencies)),
            "latency_p95_s": _p95(latencies),
            "latency_ms": float(statistics.median(latencies)) * 1000.0,
            "peak_memory_gb": peak_memory_gb,
            "model_load_time_s": float(model_load_time_s),
            "model_load_time": float(model_load_time_s),
            "sampling_latency_s": float(statistics.median([item["sampling_latency_s"] for item in cases])),
            "sampling_latency": float(statistics.median([item["sampling_latency_s"] for item in cases])),
            "decode_latency_s": float(statistics.median([item["decode_latency_s"] for item in cases])),
            "decode_latency": float(statistics.median([item["decode_latency_s"] for item in cases])),
            "quality_latency_s": float(statistics.median([item["quality_latency_s"] for item in cases])),
            "quality_latency": float(statistics.median([item["quality_latency_s"] for item in cases])),
            "peak_memory": float(peak_memory_gb),
            "model_size_gb": (
                float(Path(args.teacher).stat().st_size) / float(1024 ** 3)
                if args.teacher
                else None
            ),
            "case_count": len(cases),
            "device": str(decode_device),
            "evaluation_manifest_digest": manifest.digest,
        }
        payload = {
            "status": "success",
            "kind": args.baseline_kind,
            "manifest_digest": manifest.digest,
            "evaluation_manifest_digest": evaluation_manifest_digest,
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "clip_model_hash": clip_model_hash,
            "teacher_checkpoint": str(Path(args.teacher).resolve()) if args.teacher else None,
            "sampling_steps": int(args.sampling_steps),
            "seed_policy": "evaluation_manifest seeds",
            "generation_latency_boundary": (
                "teacher_sampling_plus_vae_decode_excluding_quality"
                if args.baseline_kind == "h3_teacher_generation_baseline"
                else "vae_decode_only_excluding_quality"
            ),
            "used_for": (
                "student_campaign_optimization_baseline"
                if args.baseline_kind == "h3_teacher_generation_baseline"
                else "reconstruction_quality_upper_reference"
            ),
            "quality_backend": "clip_temporal",
            "quality": {"score": quality, "score_type": "clip_temporal", "cases": cases},
            "hardware": hardware,
            "teacher_service": {
                "sharded": teacher_sharded,
                "world_size": len(teacher_devices),
                "devices": list(teacher_devices),
                "ranks_used": teacher_ranks_used,
                "forward_count": teacher_forward_count,
                "peak_memory_gb": teacher_peak_memory_gb,
                "service_restarts": teacher_service_restarts,
                "service_batch_size": int(args.teacher_service_batch_size),
            },
            "optimization_metrics": {
                "quality": quality,
                "latency": hardware["latency_ms"],
                "memory": hardware["peak_memory_gb"],
                "size": hardware["model_size_gb"],
            },
        }
        _write(result_path, payload)
        _write(output_dir / "teacher-baseline.json", payload)
        return 0
    except QualityBackendUnavailable as exc:
        if teacher_service is not None:
            teacher_service.close()
        if handoff_acquired:
            release_controller_handoff(
                args.controller_hold_file or None,
                args.controller_release_file or None,
                args.controller_worker_lease_file or None,
            )
        _write(result_path, {"status": "failed", "failure_code": "quality_evaluator_unavailable", "message": str(exc)})
        return 1
    except Exception as exc:
        if teacher_service is not None:
            teacher_service.close()
        if handoff_acquired:
            release_controller_handoff(
                args.controller_hold_file or None,
                args.controller_release_file or None,
                args.controller_worker_lease_file or None,
            )
        _write(result_path, {"status": "failed", "failure_code": "teacher_baseline_failed", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
