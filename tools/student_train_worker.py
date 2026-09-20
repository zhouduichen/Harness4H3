#!/usr/bin/env python3
"""Fixed remote entrypoint for one declarative H3→Student training round."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from harness4h3.student.compiler import CompileManifest
    from harness4h3.student.worker import RealH3TeacherBackend, StudentTrainWorker, TrainingResult
except ModuleNotFoundError:  # direct invocation from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness4h3.student.compiler import CompileManifest
    from harness4h3.student.worker import RealH3TeacherBackend, StudentTrainWorker, TrainingResult


def _write_result(path: Path, result: TrainingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one fixed H3-to-Student training round")
    parser.add_argument("--manifest", required=True, help="trusted compile_manifest.json")
    parser.add_argument("--teacher", required=True, help="real H3 teacher checkpoint")
    parser.add_argument("--output", required=True, help="isolated child output directory")
    parser.add_argument("--result", required=True, help="machine-readable result JSON")
    parser.add_argument("--comfyui-root", required=True, help="ComfyUI checkout containing MiniMax-H3")
    parser.add_argument("--cache-dir", required=True, help="trusted H3 latent cache directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    result_path = Path(args.result).resolve()
    try:
        manifest = CompileManifest.from_path(manifest_path)
        backend = RealH3TeacherBackend(Path(args.comfyui_root), Path(args.cache_dir))
        result = StudentTrainWorker(backend).run(
            manifest,
            Path(args.teacher),
            Path(args.output),
            max_steps=args.max_steps,
            device=args.device,
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
