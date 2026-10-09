#!/usr/bin/env python3
"""Evaluate one Wan checkpoint on the authenticated native 5s paper manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.evaluation_manifest import CacheContract, EvaluationManifest, validate_h3_cache_item
from tools.wan_vllm_campaign import evaluate_checkpoint


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("wan_eval"), Mapping):
        raise ValueError("evaluation config requires a wan_eval mapping")
    return dict(raw["wan_eval"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = _load(args.config)
    manifest = EvaluationManifest.from_path(Path(str(config["evaluation_manifest"])))
    if manifest.cache_contract != CacheContract.native_5s():
        raise ValueError("native evaluation requires the native_5s cache contract")
    for case in manifest.cases:
        validate_h3_cache_item(Path(case.cache_path), manifest.cache_contract)
    result = evaluate_checkpoint(config, args.checkpoint.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
