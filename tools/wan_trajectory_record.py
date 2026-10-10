#!/usr/bin/env python3
"""Generate a native Wan trajectory record from an injected component factory.

The real Wan worker already owns model/cache loading.  This small CLI keeps
record generation reusable without duplicating that loader: --factory points
to module:function returning the keyword arguments accepted by
build_wan_trajectory_record.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from h3_training.wan_trajectory_builder import build_wan_trajectory_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic native Wan Teacher trajectory record")
    parser.add_argument("--cache-item", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-nfe", type=int, default=32)
    parser.add_argument("--target-nfe", type=int, default=16)
    parser.add_argument(
        "--factory",
        required=True,
        help="module:function returning adapter, teacher_role, clean, and conditioning",
    )
    return parser


def _load_factory(specification: str):
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--factory must be formatted as module:function")
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise TypeError("trajectory factory is not callable: %s" % specification)
    return function


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values: dict[str, Any] = dict(_load_factory(args.factory)(args))
    values.update(
        {
            "cache_item": args.cache_item,
            "teacher_checkpoint": args.teacher,
            "output_dir": args.output,
            "seed": args.seed,
            "source_nfe": args.source_nfe,
            "target_nfe": args.target_nfe,
        }
    )
    trajectory = build_wan_trajectory_record(**values)
    print(trajectory.metadata.tensor_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
