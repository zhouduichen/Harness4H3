#!/usr/bin/env python3
"""Run and record controlled one-hot Wan loss ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml


_VARIANT_ORDER = ("endpoint", "flow", "clean", "rollout")


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("base"), Mapping) or not isinstance(raw.get("variants"), Mapping):
        raise ValueError("ablation config requires base and variants mappings")
    return dict(raw)


def _one_hot(weights: Mapping[str, Any]) -> bool:
    names = ("endpoint", "flow", "clean", "rollout_endpoint", "rollout_clean")
    active = [name for name in names if float(weights.get(name, 0.0)) > 0.0]
    return len(active) == 1 and float(weights.get(active[0], 0.0)) == 1.0


def run(config_path: Path, output_root: Path, *, worker_python: str, worker_path: Path, dry_run: bool = False) -> dict[str, Any]:
    raw = _load(config_path)
    base = dict(raw["base"])
    variants = raw["variants"]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "config": str(Path(config_path).resolve()),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "variants": [],
    }
    _write_json(output_root / "ablation-manifest.json", manifest)
    for name in _VARIANT_ORDER:
        if name not in variants:
            continue
        override = variants[name]
        if not isinstance(override, Mapping):
            raise ValueError("variant %s must be a mapping" % name)
        selected = dict(base)
        selected.update(dict(override))
        selected["ablation_name"] = name
        expected_weights = {
            "endpoint": float(selected.get("endpoint_weight", 1.0)),
            "flow": float(selected.get("velocity_weight", 1.0)),
            "clean": float(selected.get("clean_anchor_weight", 0.5)),
            "rollout_endpoint": float(selected.get("rollout_endpoint_weight", 0.0)),
            "rollout_clean": float(selected.get("rollout_clean_weight", 0.0)),
        }
        if not _one_hot(expected_weights):
            raise ValueError("variant %s is not one-hot: %s" % (name, expected_weights))
        variant_dir = output_root / name
        result_path = variant_dir / "result.json"
        config_path_for_worker = variant_dir / "config.yaml"
        log_path = variant_dir / "worker.log"
        variant_dir.mkdir(parents=True, exist_ok=True)
        config_path_for_worker.write_text(
            yaml.safe_dump({"schema_version": 1, "wan_training": selected}, sort_keys=False),
            encoding="utf-8",
        )
        command = [
            str(worker_python),
            str(worker_path),
            "--config", str(config_path_for_worker),
            "--output", str(variant_dir),
            "--result", str(result_path),
            "--parent-checkpoint", str(selected["parent_checkpoint"]),
        ]
        entry: dict[str, Any] = {
            "name": name,
            "weights": expected_weights,
            "config": str(config_path_for_worker),
            "result": str(result_path),
            "log": str(log_path),
            "command": command,
            "config_sha256": hashlib.sha256(config_path_for_worker.read_bytes()).hexdigest(),
        }
        if dry_run:
            entry["status"] = "dry_run"
            manifest["variants"].append(entry)
            continue
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        entry["wall_time_s"] = time.perf_counter() - started
        entry["returncode"] = int(completed.returncode)
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            entry["status"] = str(result.get("status", "missing_status"))
            entry["student_checkpoint_sha256"] = result.get("student_checkpoint_sha256")
            entry["optimizer_steps"] = result.get("optimizer_steps")
            entry["loss"] = result.get("loss")
            entry["loss_trace"] = result.get("loss_trace", [])
            entry["reported_loss_weights"] = result.get("loss_weights")
            if entry["status"] == "success" and result.get("loss_weights") != expected_weights:
                entry["status"] = "failed_loss_weight_mismatch"
        else:
            entry["status"] = "failed_missing_result"
        manifest["variants"].append(entry)
        manifest["status"] = "running"
        _write_json(output_root / "ablation-manifest.json", manifest)
        if not dry_run and entry.get("status") != "success":
            current_index = _VARIANT_ORDER.index(name)
            for remaining in _VARIANT_ORDER[current_index + 1 :]:
                if remaining not in variants:
                    continue
                manifest["variants"].append(
                    {
                        "name": remaining,
                        "status": "skipped_after_failure",
                        "reason": "previous_loss_ablation_failed",
                    }
                )
            _write_json(output_root / "ablation-manifest.json", manifest)
            break
    manifest["status"] = "dry_run" if dry_run else (
        "passed" if all(item.get("status") == "success" for item in manifest["variants"]) else "partial_failed"
    )
    manifest["variant_count"] = len(manifest["variants"])
    manifest["manifest_digest"] = _digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
    _write_json(output_root / "ablation-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("wan_student_train_worker.py"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        args.config,
        args.output_root,
        worker_python=args.worker_python,
        worker_path=args.worker,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"passed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
