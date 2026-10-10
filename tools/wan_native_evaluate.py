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
from h3_training.wan_native_gate import evaluate_native_promotion
from tools.wan_vllm_campaign import evaluate_checkpoint


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("wan_eval"), Mapping):
        raise ValueError("evaluation config requires a wan_eval mapping")
    return dict(raw["wan_eval"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=False, type=Path)
    parser.add_argument("--manifest", required=False, type=Path)
    parser.add_argument("--checkpoint", required=False, type=Path)
    parser.add_argument("--output", required=False, type=Path)
    parser.add_argument("--stage", choices=("F1", "F2", "F3"), default=None)
    parser.add_argument("--base-wan32-quality", type=float, default=None)
    parser.add_argument("--frozen-r2-quality", type=float, default=None)
    parser.add_argument("--failure-memory", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        manifest_path = args.manifest
        if manifest_path is None and args.config is not None:
            manifest_path = Path(str(_load(args.config)["evaluation_manifest"]))
        if manifest_path is None:
            parser.error("--validate-only requires --manifest or --config")
        manifest = EvaluationManifest.from_path(manifest_path)
        if manifest.cache_contract != CacheContract.native_5s():
            raise ValueError("native evaluation requires the native_5s cache contract")
        print(json.dumps({"status": "validated", "manifest_digest": manifest.digest, "case_count": len(manifest.cases)}, sort_keys=True))
        return 0
    if args.config is None or args.checkpoint is None or args.output is None:
        parser.error("evaluation requires --config, --checkpoint, and --output")
    config = _load(args.config)
    if args.manifest is not None:
        config["evaluation_manifest"] = str(args.manifest.resolve())
    manifest = EvaluationManifest.from_path(Path(str(config["evaluation_manifest"])))
    if manifest.cache_contract != CacheContract.native_5s():
        raise ValueError("native evaluation requires the native_5s cache contract")
    for case in manifest.cases:
        validate_h3_cache_item(Path(case.cache_path), manifest.cache_contract)
    config["adaptive_visual_retry_enabled"] = False
    result = evaluate_checkpoint(config, args.checkpoint.resolve(), args.output.resolve())
    stage = str(args.stage or config.get("stage", "F3"))
    record_path = config.get("trajectory_record")
    record_valid = bool(config.get("trajectory_record_valid", False))
    if record_path:
        record_valid = record_valid or Path(str(record_path)).joinpath("trajectory-record.json").is_file()
    required_artifacts = []
    configured_artifacts = config.get("required_artifacts", [])
    if isinstance(configured_artifacts, list):
        for item in configured_artifacts:
            if isinstance(item, Mapping):
                artifact_path = Path(str(item.get("path", ""))).resolve()
                required_artifacts.append(
                    {
                        "case_id": item.get("case_id", ""),
                        "seed": item.get("seed", 0),
                        "path": str(artifact_path),
                        "present": artifact_path.is_file(),
                        "reason": item.get("reason", "required_artifact_missing"),
                    }
                )
            else:
                artifact_path = Path(str(item)).resolve()
                required_artifacts.append(
                    {
                        "path": str(artifact_path),
                        "present": artifact_path.is_file(),
                        "reason": "required_artifact_missing",
                    }
                )
    gate = evaluate_native_promotion(
        result,
        stage=stage,
        base_wan32_quality=float(
            args.base_wan32_quality
            if args.base_wan32_quality is not None
            else config.get("base_wan32_quality", 0.6633350338532843)
        ),
        frozen_r2_quality=float(
            args.frozen_r2_quality
            if args.frozen_r2_quality is not None
            else config.get("frozen_r2_quality", 0.6552734404647101)
        ),
        record_valid=record_valid,
        scheduler_parity_ok=bool(config.get("scheduler_parity_ok", False)),
        expected_nfe=int(config.get("expected_nfe", 192)),
        failure_memory_path=args.failure_memory or args.output.resolve() / "failure-memory.jsonl",
        required_artifacts=required_artifacts,
    )
    result = {**dict(result), "native_promotion": gate, "promotion_allowed": gate["promotion_allowed"]}
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "native-promotion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "success" and gate["promotion_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
