#!/usr/bin/env python3
"""Fixed remote entrypoint for one declarative H3→Student training round."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

try:
    from harness4h3.student.compiler import CompileManifest
    from harness4h3.student.gpu import (
        GPUResourceUnavailable,
        acquire_controller_handoff,
        publish_worker_gpu_lease,
        release_controller_handoff,
        select_free_cuda_devices,
    )
    from harness4h3.student.proposal import StudentTarget
    from harness4h3.student.worker import RealH3TeacherBackend, StudentTrainWorker, TrainingResult
except ModuleNotFoundError:  # direct invocation from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness4h3.student.compiler import CompileManifest
    from harness4h3.student.gpu import (
        GPUResourceUnavailable,
        acquire_controller_handoff,
        publish_worker_gpu_lease,
        release_controller_handoff,
        select_free_cuda_devices,
    )
    from harness4h3.student.proposal import StudentTarget
    from harness4h3.student.worker import RealH3TeacherBackend, StudentTrainWorker, TrainingResult


def _write_result(path: Path, result: TrainingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stage_teacher_checkpoint(source: Path, stage_dir: Path) -> Path:
    """Use an existing fast staged copy, or create one atomically once."""

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError("teacher checkpoint does not exist: %s" % source)
    stage_dir.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    candidates = sorted(stage_dir.glob(source.stem + "-*.safetensors"))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size == source_stat.st_size:
            return candidate
    target = stage_dir / ("%s-%d-%d.safetensors" % (source.stem, source_stat.st_size, source_stat.st_mtime_ns))
    if not target.is_file() or target.stat().st_size != source_stat.st_size:
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            partial.unlink()
        shutil.copyfile(source, partial)
        if partial.stat().st_size != source_stat.st_size:
            raise IOError("staged teacher size mismatch: %s" % partial)
        partial.replace(target)
    return target


def _run_teacher_target_worker(args, teacher_devices: tuple[str, ...], output_dir: Path) -> Path:
    target_dir = output_dir / "teacher-targets"
    result_path = target_dir / "teacher-result.json"
    target_dir.mkdir(parents=True, exist_ok=True)
    visible = ",".join(device.split(":", 1)[1] for device in teacher_devices)
    teacher_checkpoint = _stage_teacher_checkpoint(Path(args.teacher), Path(args.teacher_stage_dir))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.teacher_world_size),
        "--master_port",
        str(_free_port()),
        "tools/h3_teacher_target_worker.py",
        "--checkpoint",
        str(teacher_checkpoint),
        "--comfyui-root",
        args.comfyui_root,
        "--cache-dir",
        args.cache_dir,
        "--output",
        str(target_dir),
        "--result",
        str(result_path),
        "--world-size",
        str(args.teacher_world_size),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = visible
    print("starting distributed H3 teacher target worker checkpoint=%s gpus=%s" % (teacher_checkpoint, visible), flush=True)
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        check=False,
        timeout=max(3600, int(args.wait_for_gpu_s) + 1800),
    )
    if completed.returncode != 0 or not (target_dir / "00000000.pt").is_file():
        detail = "teacher target worker exited with code %d" % completed.returncode
        if result_path.is_file():
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                detail = "%s: %s" % (detail, raw.get("message", raw))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        raise RuntimeError(detail)
    return teacher_checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one fixed H3-to-Student training round")
    parser.add_argument("--manifest", required=True, help="trusted compile_manifest.json")
    parser.add_argument("--teacher", required=True, help="real H3 teacher checkpoint")
    parser.add_argument("--output", required=True, help="isolated child output directory")
    parser.add_argument("--result", required=True, help="machine-readable result JSON")
    parser.add_argument("--comfyui-root", required=True, help="ComfyUI checkout containing MiniMax-H3")
    parser.add_argument("--cache-dir", required=True, help="trusted H3 latent cache directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wait-for-gpu-s", type=int, default=1800)
    parser.add_argument("--min-free-memory-gb", type=float, default=44.3, help="minimum free memory for the H3 teacher GPU")
    parser.add_argument("--student-min-free-memory-gb", type=float, default=20.0, help="minimum free memory for the Student GPU")
    parser.add_argument("--teacher-world-size", type=int, default=3)
    parser.add_argument("--teacher-rank-min-free-memory-gb", type=float, default=20.0)
    parser.add_argument(
        "--teacher-stage-dir",
        default="/tmp/harness4h3-remote-h3-controller-20260914/staged-checkpoints/teacher",
    )
    parser.add_argument("--controller-hold-file", default="")
    parser.add_argument("--controller-release-file", default="")
    parser.add_argument("--controller-worker-lease-file", default="")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    if args.teacher_world_size < 2 or args.teacher_world_size > 3:
        parser.error("--teacher-world-size must be between 2 and 3")
    manifest_path = Path(args.manifest).resolve()
    result_path = Path(args.result).resolve()
    handoff_acquired = False
    handoff_kept_for_evaluation = False
    try:
        manifest = CompileManifest.from_path(manifest_path)
        target = StudentTarget(**manifest.target)
        if args.device == "auto":
            acquire_controller_handoff(
                args.controller_hold_file or None,
                args.controller_release_file or None,
                args.controller_worker_lease_file or None,
            )
            handoff_acquired = True
            selected_devices = select_free_cuda_devices(
                (args.teacher_rank_min_free_memory_gb,) * args.teacher_world_size,
                args.wait_for_gpu_s,
            )
            teacher_devices = selected_devices[: args.teacher_world_size]
            selected_teacher_device = teacher_devices[0]
            selected_student_device = selected_devices[-1]
            publish_worker_gpu_lease(
                args.controller_worker_lease_file or None,
                selected_devices,
            )
            staged_teacher_checkpoint = _run_teacher_target_worker(
                args, teacher_devices, Path(args.output).resolve()
            )
        else:
            raise RuntimeError("production Student worker requires --device auto for distributed H3 teacher targets")
        backend = RealH3TeacherBackend(
            Path(args.comfyui_root),
            Path(args.cache_dir),
            teacher_targets_dir=Path(args.output).resolve() / "teacher-targets",
        )
        result = StudentTrainWorker(backend, target=target).run(
            manifest,
            staged_teacher_checkpoint,
            Path(args.output),
            max_steps=args.max_steps,
            device=selected_student_device,
            teacher_device=selected_teacher_device,
            student_device=selected_student_device,
        )
        if result.status == "success" and handoff_acquired:
            # Keep the Controller stopped until the independent evaluator
            # releases the exact handoff marker.
            release_controller_handoff(
                None,
                args.controller_release_file or None,
                args.controller_worker_lease_file or None,
            )
            handoff_kept_for_evaluation = bool(args.controller_hold_file)
    except GPUResourceUnavailable as exc:
        result = TrainingResult(
            status="failed",
            proposal_digest="",
            compiler_digest="",
            parent_sha256=None,
            child_sha256=None,
            child_checkpoint=None,
            optimizer_steps=0,
            initial_loss=None,
            final_loss=None,
            gradient_norm=None,
            wall_time_s=0.0,
            peak_memory_gb=0.0,
            changed_parameter_count=0,
            offline_simulation=False,
            failure_code="resource_unavailable",
            message=str(exc),
        )
    except Exception as exc:
        result = TrainingResult(
            status="failed",
            proposal_digest="",
            compiler_digest="",
            parent_sha256=None,
            child_sha256=None,
            child_checkpoint=None,
            optimizer_steps=0,
            initial_loss=None,
            final_loss=None,
            gradient_norm=None,
            wall_time_s=0.0,
            peak_memory_gb=0.0,
            changed_parameter_count=0,
            offline_simulation=False,
            failure_code="worker_startup",
            message=str(exc),
        )
    finally:
        if handoff_acquired and not handoff_kept_for_evaluation:
            release_controller_handoff(
                args.controller_hold_file or None,
                args.controller_release_file or None,
                args.controller_worker_lease_file or None,
            )
    _write_result(result_path, result)
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
