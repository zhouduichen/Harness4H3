#!/usr/bin/env python3
"""Fixed remote entrypoint for one declarative H3→Student training round."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from harness4h3.student.compiler import CompileManifest
    from harness4h3.student.proposal import StudentTarget
    from harness4h3.student.worker import RealH3TeacherBackend, StudentTrainWorker, TrainingResult
except ModuleNotFoundError:  # direct invocation from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness4h3.student.compiler import CompileManifest
    from harness4h3.student.proposal import StudentTarget
    from harness4h3.student.worker import RealH3TeacherBackend, StudentTrainWorker, TrainingResult


def _write_result(path: Path, result: TrainingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class WorkerResourceUnavailable(RuntimeError):
    """Raised when the remote worker cannot acquire a safe GPU lease."""


def _select_gpu(min_free_memory_gb: float, wait_s: int) -> str:
    deadline = time.monotonic() + max(0, int(wait_s))
    minimum_mib = int(float(min_free_memory_gb) * 1024)
    last_status = "nvidia-smi returned no GPU with enough free memory"
    while True:
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_status = "unable to query nvidia-smi: %s" % exc
        else:
            candidates = []
            for line in probe.stdout.splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 2:
                    continue
                try:
                    index, free_mib = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
                if free_mib >= minimum_mib:
                    candidates.append((free_mib, index))
            if candidates:
                _, index = max(candidates)
                return "cuda:%d" % index
            last_status = "no GPU has at least %.1f GiB free" % float(min_free_memory_gb)
        if time.monotonic() >= deadline:
            raise WorkerResourceUnavailable("GPU lease timeout after %ss: %s" % (int(wait_s), last_status))
        time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))


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
    parser.add_argument("--min-free-memory-gb", type=float, default=43.0)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    result_path = Path(args.result).resolve()
    try:
        manifest = CompileManifest.from_path(manifest_path)
        backend = RealH3TeacherBackend(Path(args.comfyui_root), Path(args.cache_dir))
        target = StudentTarget(**manifest.target)
        selected_device = (
            _select_gpu(args.min_free_memory_gb, args.wait_for_gpu_s)
            if args.device == "auto"
            else args.device
        )
        result = StudentTrainWorker(backend, target=target).run(
            manifest,
            Path(args.teacher),
            Path(args.output),
            max_steps=args.max_steps,
            device=selected_device,
        )
    except WorkerResourceUnavailable as exc:
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
    _write_result(result_path, result)
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
