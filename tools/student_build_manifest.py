#!/usr/bin/env python3
"""Create the immutable fixed evaluation manifest on the server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from harness4h3.student.evaluation_manifest import build_manifest
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness4h3.student.evaluation_manifest import build_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a fixed H3 evaluation manifest")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260920, 20260921])
    args = parser.parse_args(argv)
    manifest = build_manifest(
        Path(args.cache_dir),
        Path(args.output),
        case_count=args.case_count,
        seeds=tuple(args.seeds),
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
