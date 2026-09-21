#!/usr/bin/env python3
"""Stage one immutable H3 checkpoint into the local worker cache.

This helper intentionally uses only the standard library so the campaign can
prefetch a checkpoint before the CUDA worker starts.  The training worker uses
the same lock and filename scheme; a prefetch and a worker therefore share one
copy instead of racing or duplicating a multi-gigabyte read.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path


LOCK_NAME = ".h3-checkpoint-stage.lock"


def _target_for(source: Path, stage_dir: Path) -> Path:
    stat = source.stat()
    return stage_dir / (
        "%s-%d-%d.safetensors"
        % (source.stem, int(stat.st_size), int(stat.st_mtime_ns))
    )


def stage_checkpoint(source: Path, stage_dir: Path) -> dict[str, object]:
    source = Path(source).resolve()
    stage_dir = Path(stage_dir).resolve()
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise FileNotFoundError("checkpoint is not a safetensors file: %s" % source)
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = _target_for(source, stage_dir)
    lock_path = stage_dir / LOCK_NAME
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if target.is_file() and target.stat().st_size == source.stat().st_size:
                action = "cached"
            else:
                partial = target.with_name(target.name + ".part")
                if partial.exists():
                    partial.unlink()
                shutil.copyfile(source, partial)
                if partial.stat().st_size != source.stat().st_size:
                    raise IOError("staged checkpoint size mismatch: %s" % partial)
                with partial.open("r+b") as handle:
                    os.fsync(handle.fileno())
                partial.replace(target)
                action = "copied"
            # The stage directory is a one-entry read-through cache.  It is
            # not a second model archive and must not grow with every round.
            for candidate in stage_dir.glob("*.safetensors"):
                if candidate != target:
                    candidate.unlink()
            for candidate in stage_dir.glob("*.safetensors.part"):
                if candidate != target:
                    candidate.unlink()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "status": "ready",
        "action": action,
        "source": str(source),
        "target": str(target),
        "size_bytes": int(source.stat().st_size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stage-dir", required=True)
    args = parser.parse_args()
    try:
        result = stage_checkpoint(Path(args.source), Path(args.stage_dir))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)[:2000]}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
