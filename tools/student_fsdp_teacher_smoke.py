#!/usr/bin/env python3
"""Run one real collective FSDP MiniMax-H3 Teacher forward and record evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from harness4h3.campaign.base import sha256_path
from harness4h3.student.inference import load_h3_cache_item
from harness4h3.student.proposal import StudentTarget
from harness4h3.student.teacher_service import TeacherService
from h3_training.data.schema import Conditioning, ModalLatents, ModalTimesteps


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _target_from_latent_options(latent_height: int | None, latent_width: int | None) -> StudentTarget:
    target_kwargs = {}
    if latent_height is not None:
        target_kwargs["latent_height"] = latent_height
    if latent_width is not None:
        target_kwargs["latent_width"] = latent_width
    return StudentTarget(**target_kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the collective online H3 Teacher")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--comfyui-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", required=True, help="exactly three distinct cuda:N values")
    parser.add_argument("--student-device", default="", help="optional Student GPU used for overlap assertion")
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=600.0,
        help="maximum time allowed for all three real Teacher ranks to load and become ready",
    )
    parser.add_argument(
        "--distributed-timeout-s",
        type=float,
        default=600.0,
        help="distributed collective timeout, including slow first-time checkpoint loading",
    )
    parser.add_argument(
        "--latent-height",
        type=int,
        default=None,
        help="H3 cache latent height; required when it differs from the default StudentTarget",
    )
    parser.add_argument(
        "--latent-width",
        type=int,
        default=None,
        help="H3 cache latent width; required when it differs from the default StudentTarget",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)
    result_path = Path(args.output).resolve()
    service = None
    started = time.perf_counter()
    try:
        devices = tuple(item.strip() for item in str(args.devices).split(",") if item.strip())
        if len(devices) != 3 or len(set(devices)) != 3 or any(not item.startswith("cuda:") for item in devices):
            raise ValueError("--devices must contain exactly three distinct cuda:N values")
        if args.student_device and args.student_device in devices:
            raise ValueError("Student GPU overlaps a Teacher GPU")
        checkpoint = Path(args.checkpoint).resolve()
        target = _target_from_latent_options(args.latent_height, args.latent_width)
        input_device = torch.device(devices[0])
        cache = load_h3_cache_item(Path(args.cache_dir), target, torch.device("cpu"), torch.bfloat16)
        if cache.get("audio_latent") is None:
            raise ValueError("H3 cache item has no audio latent")
        noisy = ModalLatents(
            video=cache["latent"].to(device=input_device, dtype=torch.bfloat16),
            audio=cache["audio_latent"].to(device=input_device, dtype=torch.bfloat16),
        )
        conditioning = Conditioning(cache["prompt"].to(device=input_device, dtype=torch.bfloat16))
        timestep = ModalTimesteps(
            video=torch.full((1,), 0.5, device=input_device, dtype=torch.float32),
            audio=torch.full((1,), 0.75, device=input_device, dtype=torch.float32),
        )
        service = TeacherService(
            checkpoint,
            Path(args.comfyui_root),
            devices,
            start_timeout_s=float(args.startup_timeout_s),
            distributed_timeout_s=float(args.distributed_timeout_s),
        ).start()
        forward_started = time.perf_counter()
        prediction = service.predict(noisy, timestep, conditioning)
        forward_latency_s = time.perf_counter() - forward_started
        if prediction.video is None or prediction.audio is None:
            raise RuntimeError("collective Teacher returned incomplete prediction")
        if not torch.isfinite(prediction.video).all() or not torch.isfinite(prediction.audio).all():
            raise RuntimeError("collective Teacher returned non-finite prediction")
        payload = {
            "status": "success",
            "kind": "fsdp_h3_online_teacher_smoke",
            "checkpoint": str(checkpoint),
            "teacher_checkpoint_sha256": sha256_path(checkpoint),
            "teacher_service": {
                "sharded": service.sharded,
                "world_size": service.world_size,
                "devices": list(service.devices),
                "ranks_used": sorted(service.ranks_used),
                "rank_forward_counts": dict(service.rank_forward_counts),
                "forward_count": service.forward_count,
                "peak_memory_gb": service.teacher_peak_memory_gb,
            },
            "forward_latency_s": forward_latency_s,
            "prediction_shapes": {
                "video": list(prediction.video.shape),
                "audio": list(prediction.audio.shape),
            },
            "gpu_allocation": {
                "teacher_devices": list(devices),
                "student_device": args.student_device or None,
            },
            "wall_time_s": time.perf_counter() - started,
            "evidence_sha256": _sha256(checkpoint),
        }
        if service is not None:
            service.close()
        _write(result_path, payload)
        return 0
    except Exception as exc:
        if service is not None:
            service.close()
        _write(
            result_path,
            {
                "status": "failed",
                "kind": "fsdp_h3_online_teacher_smoke",
                "failure_code": "fsdp_teacher_smoke_failed",
                "message": str(exc),
                "wall_time_s": time.perf_counter() - started,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
