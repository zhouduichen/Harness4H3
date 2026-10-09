#!/usr/bin/env python3
"""Run a fixed-recipe, fail-closed native Wan F1 -> F2 -> F3 campaign."""

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


_STAGES = ("F1", "F2", "F3")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("base"), Mapping) or not isinstance(raw.get("stages"), Mapping):
        raise ValueError("fidelity config requires base and stages mappings")
    return dict(raw)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "student_checkpoint": result.get("student_checkpoint"),
        "student_checkpoint_sha256": result.get("student_checkpoint_sha256"),
        "parent_checkpoint": result.get("parent_checkpoint"),
        "parent_checkpoint_sha256": result.get("parent_checkpoint_sha256"),
        "teacher_checkpoint_sha256": result.get("teacher_checkpoint_sha256"),
        "teacher_forward_count": result.get("teacher_forward_count"),
        "optimizer_steps": result.get("optimizer_steps"),
        "loss": result.get("loss"),
        "loss_weights": result.get("loss_weights"),
        "wall_time_s": result.get("wall_time_s"),
        "offline_simulation": result.get("offline_simulation"),
        "wan_schedule_digest": result.get("wan_schedule_digest"),
    }


def run(
    config_path: Path,
    output_root: Path,
    *,
    worker_python: str,
    worker_path: Path,
    evaluator_path: Path,
    evaluation_config: Path,
    skip_evaluation: bool = False,
) -> dict[str, Any]:
    raw = _load(config_path)
    base = dict(raw["base"])
    stages = raw["stages"]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "config": str(Path(config_path).resolve()),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "evaluation_config": str(Path(evaluation_config).resolve()),
        "controller_search": False,
        "stages": [],
    }
    _write_json(output_root / "fidelity-manifest.json", manifest)
    parent = Path(str(base["parent_checkpoint"])).resolve()
    if not parent.is_file():
        raise FileNotFoundError("initial parent checkpoint does not exist: %s" % parent)
    parent_sha = _sha256_file(parent)
    for stage_name in _STAGES:
        if stage_name not in stages:
            raise ValueError("missing fidelity stage %s" % stage_name)
        override = stages[stage_name]
        if not isinstance(override, Mapping):
            raise ValueError("stage %s must be a mapping" % stage_name)
        selected = dict(base)
        selected.update(dict(override))
        selected["parent_checkpoint"] = str(parent)
        selected["fidelity_stage"] = stage_name
        stage_dir = output_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        worker_config = stage_dir / "config.yaml"
        worker_config.write_text(
            yaml.safe_dump({"schema_version": 1, "wan_training": selected}, sort_keys=False),
            encoding="utf-8",
        )
        result_path = stage_dir / "training-result.json"
        log_path = stage_dir / "worker.log"
        command = [
            str(worker_python), str(worker_path),
            "--config", str(worker_config),
            "--output", str(stage_dir / "training"),
            "--result", str(result_path),
            "--parent-checkpoint", str(parent),
        ]
        record: dict[str, Any] = {
            "stage": stage_name,
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": parent_sha,
            "train_steps": int(selected["train_steps"]),
            "teacher_target_groups": int(selected["teacher_target_groups"]),
            "worker_config": str(worker_config),
            "worker_log": str(log_path),
            "command": command,
            "status": "running",
        }
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        record["worker_wall_time_s"] = time.perf_counter() - started
        record["worker_returncode"] = int(completed.returncode)
        if not result_path.is_file():
            record["status"] = "failed_missing_training_result"
            manifest["stages"].append(record)
            _write_json(output_root / "fidelity-manifest.json", manifest)
            break
        training = json.loads(result_path.read_text(encoding="utf-8"))
        record["training_result"] = str(result_path)
        record["training"] = _summary(training)
        if training.get("status") != "success" or training.get("offline_simulation") is not False:
            record["status"] = "failed_training_contract"
            manifest["stages"].append(record)
            _write_json(output_root / "fidelity-manifest.json", manifest)
            break
        checkpoint = Path(str(training["student_checkpoint"])).resolve()
        if not checkpoint.is_file():
            record["status"] = "failed_missing_checkpoint"
            manifest["stages"].append(record)
            _write_json(output_root / "fidelity-manifest.json", manifest)
            break
        checkpoint_sha = _sha256_file(checkpoint)
        if checkpoint_sha != str(training.get("student_checkpoint_sha256")):
            record["status"] = "failed_checkpoint_digest"
            manifest["stages"].append(record)
            _write_json(output_root / "fidelity-manifest.json", manifest)
            break
        if not skip_evaluation:
            eval_dir = stage_dir / "evaluation"
            eval_log = stage_dir / "evaluation.log"
            eval_command = [
                str(worker_python), str(evaluator_path),
                "--config", str(evaluation_config),
                "--checkpoint", str(checkpoint),
                "--output", str(eval_dir),
            ]
            eval_started = time.perf_counter()
            with eval_log.open("w", encoding="utf-8") as log:
                eval_completed = subprocess.run(eval_command, stdout=log, stderr=subprocess.STDOUT, check=False)
            record["evaluation_wall_time_s"] = time.perf_counter() - eval_started
            record["evaluation_returncode"] = int(eval_completed.returncode)
            eval_result_path = eval_dir / "evaluation-result.json"
            record["evaluation_result"] = str(eval_result_path)
            if not eval_result_path.is_file():
                record["status"] = "failed_missing_evaluation_result"
                manifest["stages"].append(record)
                _write_json(output_root / "fidelity-manifest.json", manifest)
                break
            evaluation = json.loads(eval_result_path.read_text(encoding="utf-8"))
            record["evaluation"] = {
                "status": evaluation.get("status"),
                "valid": evaluation.get("valid"),
                "promotable": evaluation.get("promotable"),
                "case_count": evaluation.get("case_count"),
                "quality_score": evaluation.get("quality_score"),
                "quality_metrics": evaluation.get("quality_metrics"),
                "hardware_metrics": evaluation.get("hardware_metrics"),
                "student_checkpoint_sha256": evaluation.get("student_checkpoint_sha256"),
                "failure_code": evaluation.get("failure_code"),
            }
            if evaluation.get("status") != "success" or not evaluation.get("promotable", False):
                record["status"] = "failed_promotion_gate"
                manifest["stages"].append(record)
                _write_json(output_root / "fidelity-manifest.json", manifest)
                break
        record["status"] = "passed"
        manifest["stages"].append(record)
        _write_json(output_root / "fidelity-manifest.json", manifest)
        parent = checkpoint
        parent_sha = checkpoint_sha
    passed = len(manifest["stages"]) == len(_STAGES) and all(item.get("status") == "passed" for item in manifest["stages"])
    manifest["status"] = "passed" if passed else "partial_failed"
    manifest["completed_stages"] = [item["stage"] for item in manifest["stages"] if item.get("status") == "passed"]
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps({key: value for key, value in manifest.items() if key != "manifest_digest"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(output_root / "fidelity-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("wan_student_train_worker.py"))
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).with_name("wan_native_evaluate.py"))
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        args.config,
        args.output_root,
        worker_python=args.worker_python,
        worker_path=args.worker,
        evaluator_path=args.evaluator,
        evaluation_config=args.evaluation_config,
        skip_evaluation=args.skip_evaluation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
