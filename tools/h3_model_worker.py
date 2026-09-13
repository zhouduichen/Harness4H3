#!/usr/bin/env python3
"""Safe adapter between Harness4H3 and a real H3 training worker.

The Harness invokes this file with ``--request`` and ``--result``. The actual
trainer command, optional teacher-signal command, and optional ComfyUI model
directory are loaded from a fixed operator config, never from the Controller
request. All child checkpoints are staged under the experiment artifacts
directory before the machine-readable result is returned to the Harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


MAX_JSON_BYTES = 8 * 1024 * 1024


TRAINING_FAILURE_TYPES = frozenset(
    {
        "unsupported_training_operator",
        "invalid_training_config",
        "no_trainable_parameters",
        "nonfinite_loss",
        "zero_gradient",
        "training_oom",
        "checkpoint_corrupt",
        "resume_mismatch",
        "cache_corrupt",
        "parent_modified",
        "unchanged_child",
        "frozen_tensor_changed",
        "child_reload_failed",
        "device_unavailable",
    }
)


TRAINING_OPERATORS = frozenset({"recovery_finetune", "step_distill", "dmd2"})
REQUIRED_TRAINING_METRICS = frozenset(
    {
        "initial_loss",
        "final_loss",
        "gradient_norm",
        "optimizer_steps",
        "trainable_parameter_count",
        "parent_sha256",
        "parent_sha256_before",
        "parent_sha256_after",
        "child_sha256",
        "changed_trainable_tensors",
        "unchanged_frozen_tensors",
        "child_reloaded",
    }
)


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON file exceeds 8 MiB: %s" % path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON object required: %s" % path)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_config(path_value: str, cwd: Path) -> Path:
    requested = Path(path_value)
    candidates = [requested] if requested.is_absolute() else [cwd / requested, Path(__file__).resolve().parents[1] / requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("worker config not found: %s" % path_value)


def _argv(value: Any, name: str) -> Sequence[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("%s must be a non-empty argv sequence" % name)
    command = tuple(value)
    if any(not isinstance(item, str) or not item or "\x00" in item for item in command):
        raise ValueError("%s must contain only non-empty strings" % name)
    return command


def _result_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    resolved = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(cwd.resolve())
    except ValueError:
        raise ValueError("result/request path must stay inside experiment directory")
    return resolved


def _cache_manifest(request: Mapping[str, Any], config: Mapping[str, Any], cwd: Path) -> Mapping[str, Any]:
    command_value = config.get("teacher_signal_command")
    cache_value = config.get("teacher_cache_dir")
    if not command_value and not cache_value:
        return {"enabled": False, "hit": False}
    command = _argv(command_value, "teacher_signal_command") if command_value else None
    cache_root = Path(str(cache_value or (cwd / "teacher-cache")))
    if not cache_root.is_absolute():
        cache_root = cwd / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    key_payload = {
        "parent_state": request.get("parent", {}).get("state", {}),
        "operator": request.get("operator"),
        "operator_args": request.get("operator_args", {}),
    }
    encoded = json.dumps(key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    manifest_path = (cache_root / ("teacher-signals-" + key + ".json")).resolve()
    if manifest_path.is_file():
        return {"enabled": True, "hit": True, "path": str(manifest_path), "key": key}
    if command is None:
        return {"enabled": True, "hit": False, "path": str(manifest_path), "key": key, "generated": False}

    signal_request = cwd / "teacher_signal_request.json"
    _write_json(signal_request, {**dict(request), "teacher_cache_path": str(manifest_path)})
    signal_command = tuple(command) + ("--request", signal_request.name, "--output", str(manifest_path))
    timeout_s = float(config.get("teacher_signal_timeout_s", 1800.0))
    if timeout_s <= 0:
        raise ValueError("teacher_signal_timeout_s must be positive")
    try:
        completed = subprocess.run(
            signal_command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("teacher signal command exceeded %.3f seconds" % timeout_s)
    (cwd / "teacher_signal.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cwd / "teacher_signal.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not manifest_path.is_file():
        raise RuntimeError("teacher signal command failed or did not create cache manifest")
    return {"enabled": True, "hit": False, "path": str(manifest_path), "key": key, "generated": True}


def _stage_child(raw_state: Mapping[str, Any], request: Mapping[str, Any], artifacts_dir: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    child_id = str(request["child_model_id"])
    state = dict(raw_state)
    architecture = str(state.get("architecture_name", ""))
    if "h3" not in architecture.lower():
        raise ValueError("trainer child architecture is not recognized as H3")
    source_value = state.get("checkpoint_path")
    if not source_value or "://" in str(source_value):
        raise ValueError("trainer must return a local child checkpoint path")
    source = Path(str(source_value)).resolve()
    parent_value = request.get("parent", {}).get("checkpoint_path")
    if parent_value and "://" not in str(parent_value) and source == Path(str(parent_value)).resolve():
        raise ValueError("trainer returned the immutable parent checkpoint")
    if not source.is_file():
        raise ValueError("trainer child checkpoint does not exist: %s" % source)
    suffix = source.suffix.lower() if source.suffix.lower() in {".safetensors", ".gguf", ".ckpt", ".pt"} else ".safetensors"
    staged = (artifacts_dir / (child_id + suffix)).resolve()
    shutil.copy2(source, staged)
    staged_evidence = None
    source_evidence = source.with_suffix(source.suffix + ".evidence.json")
    if source_evidence.is_file():
        manifest = dict(_read_json(source_evidence))
        if manifest.get("child_sha256") != _sha256_file(source):
            raise ValueError("trainer evidence manifest does not match child checkpoint")
        staged_evidence = staged.with_suffix(staged.suffix + ".evidence.json")
        manifest["path"] = str(staged)
        manifest["manifest_path"] = str(staged_evidence)
        _write_json(staged_evidence, manifest)
    deployment: Dict[str, Any] = {"deployed": False}
    deploy_value = config.get("deploy_model_dir")
    if deploy_value:
        deploy_dir = Path(str(deploy_value)).resolve()
        deploy_dir.mkdir(parents=True, exist_ok=True)
        deployed = deploy_dir / staged.name
        if parent_value and "://" not in str(parent_value) and deployed == Path(str(parent_value)).resolve():
            raise ValueError("deployment path would overwrite parent checkpoint")
        shutil.copy2(staged, deployed)
        deployment = {"deployed": True, "path": str(deployed)}
    state["model_id"] = child_id
    state["parent_model_id"] = str(request["parent"]["id"])
    state["checkpoint_path"] = str(staged)
    provenance = dict(state.get("provenance") or {})
    provenance.update({"real_worker": True, "offline_simulation": False, "deployment": deployment})
    state["provenance"] = provenance
    return {
        "state": state,
        "deployment": deployment,
        "evidence_manifest": str(staged_evidence) if staged_evidence else None,
    }


def _failure(
    result_path: Path,
    failure_type: str,
    message: str,
    wall_time_s: float = 0.0,
    metrics: Optional[Mapping[str, Any]] = None,
) -> int:
    failure_metrics = dict(metrics or {})
    failure_metrics.update({"real_worker": True, "offline_simulation": False})
    _write_json(
        result_path,
        {
            "status": "failed",
            "failure_type": failure_type,
            "message": message,
            "cost": {"wall_time_s": float(wall_time_s), "gpu_hours": 0.0, "controller_calls": 0},
            "metrics": failure_metrics,
        },
    )
    return 1


def _propagated_training_failure(payload: Any) -> Optional[Tuple[str, str, Mapping[str, Any]]]:
    if not isinstance(payload, Mapping) or payload.get("status") != "failed":
        return None
    failure_type = payload.get("failure_type")
    if not isinstance(failure_type, str) or failure_type not in TRAINING_FAILURE_TYPES:
        return None
    message = str(payload.get("message") or failure_type)
    metrics = payload.get("metrics")
    return failure_type, message, dict(metrics) if isinstance(metrics, Mapping) else {}


def _validate_training_evidence(
    operator: str,
    trainer_result: Mapping[str, Any],
    parent_sha256: Optional[str],
) -> Mapping[str, Any]:
    if operator not in TRAINING_OPERATORS:
        return {}
    metrics = trainer_result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("training result must contain measured metrics")
    missing = sorted(REQUIRED_TRAINING_METRICS - set(metrics))
    if missing:
        raise ValueError("missing measured training metric(s): %s" % ", ".join(missing))
    for name in ("initial_loss", "final_loss", "gradient_norm"):
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("training metric %s must be finite" % name)
    if float(metrics["gradient_norm"]) <= 0:
        raise ValueError("training metric gradient_norm must be positive")
    for name in ("optimizer_steps", "trainable_parameter_count", "changed_trainable_tensors"):
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("training metric %s must be a positive integer" % name)
    frozen_count = metrics["unchanged_frozen_tensors"]
    if isinstance(frozen_count, bool) or not isinstance(frozen_count, int) or frozen_count < 0:
        raise ValueError("training metric unchanged_frozen_tensors must be a non-negative integer")
    if metrics["child_reloaded"] is not True:
        raise ValueError("training result did not prove child reload")
    for name in ("parent_sha256", "child_sha256"):
        value = metrics[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("training metric %s must be a SHA-256 digest" % name)
    if metrics["parent_sha256"] == metrics["child_sha256"]:
        raise ValueError("training result reports identical parent and child hashes")
    if metrics["parent_sha256_before"] != metrics["parent_sha256"] or metrics["parent_sha256_after"] != metrics["parent_sha256"]:
        raise ValueError("training result parent before/after hashes do not agree")
    if parent_sha256 is not None and metrics["parent_sha256"] != parent_sha256:
        raise ValueError("training result parent hash does not match the immutable parent")
    return dict(metrics)


def run(request_path: Path, result_path: Path, config_path: Path) -> int:
    started = time.monotonic()
    cwd = Path.cwd().resolve()
    try:
        request = _read_json(request_path)
        config = _read_json(config_path)
        operator = str(request["operator"])
        parent = request["parent"]
        if not isinstance(parent, Mapping) or not parent.get("id") or not request.get("child_model_id"):
            raise ValueError("request requires parent and child_model_id")
        parent_value = parent.get("checkpoint_path")
        parent_path = None
        parent_sha256 = None
        if parent_value and "://" not in str(parent_value):
            parent_path = Path(str(parent_value)).resolve()
            if parent_path.is_file():
                parent_sha256 = _sha256_file(parent_path)
        teacher_cache = _cache_manifest(request, config, cwd)
        artifacts_dir = Path(str(request["artifacts_dir"])).resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        trainer_request = {
            **dict(request),
            "real_worker": True,
            "offline_simulation": False,
            "teacher_cache": teacher_cache,
            "artifacts_dir": str(artifacts_dir),
        }
        trainer_request_path = cwd / "trainer_request.json"
        trainer_result_path = cwd / "trainer_result.json"
        _write_json(trainer_request_path, trainer_request)
        trainer_command = tuple(_argv(config.get("trainer_command"), "trainer_command")) + (
            "--request",
            trainer_request_path.name,
            "--result",
            trainer_result_path.name,
        )
        timeout_s = float(config.get("trainer_timeout_s", 7200.0))
        if timeout_s <= 0:
            raise ValueError("trainer_timeout_s must be positive")
        try:
            completed = subprocess.run(
                trainer_command,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            return _failure(result_path, "training_timeout", "trainer exceeded %.3f seconds" % timeout_s, elapsed)
        (cwd / "trainer.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (cwd / "trainer.stderr.log").write_text(completed.stderr, encoding="utf-8")
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            if trainer_result_path.is_file():
                try:
                    trainer_failure = _propagated_training_failure(_read_json(trainer_result_path))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    trainer_failure = None
                if trainer_failure is not None:
                    failure_type, message, metrics = trainer_failure
                    return _failure(result_path, failure_type, message, elapsed, metrics)
            return _failure(result_path, "training_process", "trainer exited with status %d" % completed.returncode, elapsed)
        if not trainer_result_path.is_file():
            return _failure(result_path, "missing_trainer_result", "trainer did not create trainer_result.json", elapsed)
        trainer_result = _read_json(trainer_result_path)
        trainer_failure = _propagated_training_failure(trainer_result)
        if trainer_failure is not None:
            failure_type, message, metrics = trainer_failure
            return _failure(result_path, failure_type, message, elapsed, metrics)
        if trainer_result.get("status") != "success" or not isinstance(trainer_result.get("output_state"), Mapping):
            return _failure(result_path, "invalid_trainer_result", "trainer result must contain success and output_state", elapsed)
        if parent_path is not None and parent_sha256 != _sha256_file(parent_path):
            return _failure(result_path, "parent_modified", "parent checkpoint hash changed during training", elapsed)
        try:
            training_metrics = _validate_training_evidence(operator, trainer_result, parent_sha256)
        except ValueError as exc:
            return _failure(result_path, "invalid_training_evidence", str(exc), elapsed)
        staged = _stage_child(trainer_result["output_state"], request, Path(str(request["artifacts_dir"])).resolve(), config)
        if operator in TRAINING_OPERATORS:
            staged_hash = _sha256_file(Path(staged["state"]["checkpoint_path"]))
            if staged_hash != training_metrics["child_sha256"]:
                return _failure(result_path, "invalid_training_evidence", "child hash does not match staged checkpoint", elapsed)
            if not staged["evidence_manifest"]:
                return _failure(result_path, "invalid_training_evidence", "training child evidence manifest is missing", elapsed)
        reported_cost = trainer_result.get("cost") if isinstance(trainer_result.get("cost"), Mapping) else {}
        reported_gpu_hours = float(reported_cost.get("gpu_hours", 0.0))
        if reported_gpu_hours < 0:
            raise ValueError("trainer gpu_hours must be non-negative")
        output = {
            "status": "success",
            "output_state": staged["state"],
            "cost": {
                "wall_time_s": max(elapsed, float(reported_cost.get("wall_time_s", 0.0))),
                "gpu_hours": reported_gpu_hours,
                "controller_calls": 0,
            },
            "metrics": {
                "real_worker": True,
                "offline_simulation": False,
                "operator": operator,
                "teacher_cache": teacher_cache,
                "deployment": staged["deployment"],
                "child_evidence_manifest": staged["evidence_manifest"],
                **dict(trainer_result.get("metrics") or {}),
            },
        }
        _write_json(result_path, output)
        return 0
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return _failure(result_path, "worker_contract", str(exc), time.monotonic() - started)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="safe real H3 model worker adapter")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--config", default=os.environ.get("H3_A1_WORKER_CONFIG", ""))
    args = parser.parse_args(argv)
    cwd = Path.cwd().resolve()
    result_path = _result_path(args.result, cwd)
    try:
        if not args.config:
            return _failure(result_path, "worker_config_invalid", "--config or H3_A1_WORKER_CONFIG is required")
        request_path = _result_path(args.request, cwd)
        config_path = _resolve_config(args.config, cwd)
        return run(request_path, result_path, config_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _failure(result_path, "worker_contract", str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
