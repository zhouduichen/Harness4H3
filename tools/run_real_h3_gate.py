#!/usr/bin/env python3
"""Fail-closed entry point for the first authentic MiniMax-H3 campaign.

The gate deliberately performs all cheap checks before spawning a distributed
worker.  It never substitutes TinyH3, fake metrics, or a CPU path when a real
H3 prerequisite is missing.  A JSON result is written on both success and
failure so a remote run can be audited without relying on terminal output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


# A gate is normally invoked as ``python tools/run_real_h3_gate.py``.  In
# that form Python puts ``tools/`` (and any editable install) ahead of the
# checkout root, which can accidentally import a different Harness4H3 tree
# on shared hosts.  Make the selected staging checkout authoritative before
# importing project modules below.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON object required: %s" % path)
    return value


def _resolve(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _check_endpoint(base_url: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/system_stats"
    try:
        request = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"name": "comfyui_system_stats", "passed": isinstance(payload, Mapping), "endpoint": endpoint}
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError) as exc:
        return {"name": "comfyui_system_stats", "passed": False, "endpoint": endpoint, "error": str(exc)}


def preflight(
    parent_checkpoint: Path,
    worker_config: Path,
    *,
    comfyui_root: Optional[Path] = None,
    tasks_path: Optional[Path] = None,
    base_url: Optional[str] = None,
    require_gpu: bool = True,
    check_endpoint: bool = True,
) -> Dict[str, Any]:
    """Return a machine-readable prerequisite report without mutating state."""

    parent_checkpoint = Path(parent_checkpoint).resolve()
    worker_config = Path(worker_config).resolve()
    checks = []
    errors = []

    def add(name: str, passed: bool, **details: Any) -> None:
        item = {"name": name, "passed": bool(passed), **details}
        checks.append(item)
        if not passed:
            errors.append(name)

    add("parent_checkpoint", parent_checkpoint.is_file() and parent_checkpoint.suffix.lower() == ".safetensors", path=str(parent_checkpoint))
    add("worker_config", worker_config.is_file(), path=str(worker_config))

    config: Mapping[str, Any] = {}
    if worker_config.is_file():
        try:
            config = _read_json(worker_config)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            add("worker_config_json", False, error=str(exc))
        else:
            add("worker_config_json", True)
    else:
        add("worker_config_json", False, error="worker config is missing")

    root_value = comfyui_root or (Path(str(config.get("comfyui_root"))).resolve() if config.get("comfyui_root") else None)
    add("comfyui_root", root_value is not None and root_value.is_dir(), path=str(root_value) if root_value else None)

    configured_parent = config.get("model_checkpoint")
    add(
        "configured_parent_matches",
        bool(configured_parent) and _resolve(configured_parent, worker_config.parent) == parent_checkpoint,
        configured_path=str(_resolve(configured_parent, worker_config.parent)) if configured_parent else None,
    )

    command = config.get("trainer_command")
    valid_command = isinstance(command, (list, tuple)) and bool(command) and all(isinstance(item, str) and item for item in command)
    command_text = " ".join(str(item) for item in command) if valid_command else ""
    add("trainer_command", valid_command, command=command if valid_command else None)
    add(
        "real_h3_trainer",
        valid_command and "h3_real_train_worker.py" in command_text and "tiny" not in command_text.lower() and "fake" not in command_text.lower(),
        command=command_text,
    )

    world_size = int(config.get("world_size", 4)) if str(config.get("world_size", "4")).isdigit() else 0
    if require_gpu:
        try:
            import torch

            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if available else 0
            add("cuda_available", available, device_count=count)
            add("cuda_world_size", available and count >= world_size and 2 <= world_size <= 4, world_size=world_size, device_count=count)
        except (ImportError, RuntimeError) as exc:
            add("cuda_available", False, error=str(exc), device_count=0)
            add("cuda_world_size", False, world_size=world_size, device_count=0)
    else:
        add("cuda_available", True, skipped=True)
        add("cuda_world_size", True, skipped=True, world_size=world_size)

    cache_dir = config.get("cache_dir")
    output_dir = config.get("output_dir")
    add("cache_dir_configured", bool(cache_dir), path=str(cache_dir) if cache_dir else None)
    add("output_dir_configured", bool(output_dir), path=str(output_dir) if output_dir else None)
    if tasks_path is not None:
        add("task_manifest", Path(tasks_path).is_file(), path=str(Path(tasks_path).resolve()))
    if base_url and check_endpoint:
        endpoint = _check_endpoint(base_url)
        checks.append(endpoint)
        if not endpoint["passed"]:
            errors.append(endpoint["name"])

    return {
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "parent_checkpoint": str(parent_checkpoint),
        "worker_config": str(worker_config),
        "real_evidence_required": True,
        "offline_simulation": False,
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_gate(args: argparse.Namespace) -> int:
    started = time.monotonic()
    output_root = Path(args.output_root).resolve()
    gate_path = Path(args.gate_result).resolve() if args.gate_result else output_root / "real_h3_gate.json"
    worker_config = Path(args.worker_config).resolve()
    report = preflight(
        Path(args.parent_checkpoint),
        worker_config,
        comfyui_root=Path(args.comfyui_root).resolve() if args.comfyui_root else None,
        tasks_path=Path(args.tasks).resolve() if args.tasks else None,
        base_url=args.base_url,
        require_gpu=not args.skip_gpu_check,
        check_endpoint=not args.skip_endpoint_check,
    )
    if not report["passed"]:
        report = {**report, "status": "blocked", "elapsed_s": time.monotonic() - started}
        _write(gate_path, report)
        print(json.dumps({"status": report["status"], "gate_result": str(gate_path), "errors": report["errors"]}, ensure_ascii=False))
        return 2

    config = _read_json(worker_config)
    wrapper = [sys.executable, str(Path(__file__).resolve().with_name("h3_model_worker.py")), "--config", str(worker_config)]
    from harness4h3.controller.provider import OllamaStructuredController
    from harness4h3.controller.schemas import HardwareMetrics
    from harness4h3.target.profile import load_target_profile
    from research.experiments.a0_model_evolution import A0RuleBasedController
    from research.experiments.a1_real_evolution import run_a1_active

    target = load_target_profile(Path(args.target).resolve())
    if args.controller == "rulebased":
        controller = A0RuleBasedController()
    else:
        controller = OllamaStructuredController(args.controller_model, args.controller_url, timeout_s=args.controller_timeout_s)
    campaign = run_a1_active(
        Path(args.parent_checkpoint),
        wrapper,
        Path(args.config).resolve(),
        target,
        output_root,
        args.base_url,
        args.baseline_quality,
        HardwareMetrics(
            model_size_gb=args.baseline_model_size_gb,
            latency_s=args.baseline_latency_s,
            peak_memory_gb=args.baseline_peak_memory_gb,
        ),
        controller,
        max_experiments=args.max_experiments,
        max_gpu_hours=args.max_gpu_hours,
        max_failed_experiments=args.max_failed_experiments,
        worker_timeout_s=args.worker_timeout_s,
    )
    sequence = campaign.report.get("full_autonomous_experiment_sequence", [])
    has_child = any(item.get("child_model_id") == "M0001" for item in sequence if isinstance(item, Mapping))
    has_second_plan = campaign.report.get("second_controller_plan") is not None
    final = {
        **report,
        "status": "passed" if has_child and has_second_plan else "failed",
        "campaign_status": campaign.status,
        "campaign_report": str(output_root / "report.json"),
        "has_m0001": has_child,
        "has_exp0002_plan": has_second_plan,
        "elapsed_s": time.monotonic() - started,
        "config": dict(config),
    }
    _write(gate_path, final)
    print(json.dumps({"status": final["status"], "gate_result": str(gate_path), "campaign_status": campaign.status}, ensure_ascii=False))
    return 0 if final["status"] == "passed" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="fail-closed Real MiniMax-H3 A1-T0 gate")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--worker-config", default="configs/a1-worker.l40x4-distill4.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", default="configs/targets/l40x4_h3_example.yaml")
    parser.add_argument("--tasks")
    parser.add_argument("--comfyui-root")
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output-root", default="var/a1-real-gate")
    parser.add_argument("--gate-result")
    parser.add_argument("--baseline-quality", type=float, required=True)
    parser.add_argument("--baseline-model-size-gb", type=float, required=True)
    parser.add_argument("--baseline-latency-s", type=float, required=True)
    parser.add_argument("--baseline-peak-memory-gb", type=float, required=True)
    parser.add_argument("--controller", choices=("rulebased", "ollama"), default="rulebased")
    parser.add_argument("--controller-model", default="qwen3.5:9b-q8_0")
    parser.add_argument("--controller-url", default="http://127.0.0.1:11434")
    parser.add_argument("--controller-timeout-s", type=float, default=180.0)
    parser.add_argument("--worker-timeout-s", type=float, default=10800.0)
    parser.add_argument("--max-experiments", type=int, default=2)
    parser.add_argument("--max-gpu-hours", type=float, default=8.0)
    parser.add_argument("--max-failed-experiments", type=int, default=2)
    parser.add_argument("--skip-gpu-check", action="store_true", help="for contract tests only; never use for a real campaign")
    parser.add_argument("--skip-endpoint-check", action="store_true", help="for preflight-only tests")
    return run_gate(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
