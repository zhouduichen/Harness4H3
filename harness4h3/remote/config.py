"""Validated configuration for a fixed remote H3 campaign."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

from ..config import Target, WorkflowConfig
from .reward import RewardWeights
from .ssh import RemoteConfig


class RemoteConfigError(ValueError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RemoteConfigError("%s must be a mapping" % name)
    return value


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RemoteConfigError("%s must be a path" % name)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _target(value: Any, name: str) -> Target:
    raw = _mapping(value, name)
    node_id, input_name = str(raw.get("node_id", "")).strip(), str(raw.get("input", "")).strip()
    if not node_id or not input_name:
        raise RemoteConfigError("%s requires node_id and input" % name)
    return Target(node_id, input_name)


def _positive(value: Any, name: str, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RemoteConfigError("%s must be numeric" % name)
    if parsed < 0 if allow_zero else parsed <= 0:
        raise RemoteConfigError("%s must be positive" % name)
    return parsed


@dataclass(frozen=True)
class RemoteRuntimeConfig:
    experience_path: Path
    output_root: Path
    tasks_path: Path


@dataclass(frozen=True)
class RemoteWorkerConfig:
    enabled: bool
    entrypoint: str
    config_template: str
    python: str
    allowed_operators: Tuple[str, ...]
    max_steps: int


@dataclass(frozen=True)
class RemoteCampaignConfig:
    remote: RemoteConfig
    workflow: WorkflowConfig
    runtime: RemoteRuntimeConfig
    reward: RewardWeights
    benchmark_splits: Tuple[str, ...]
    sampling_interval_s: float
    efficiency_thresholds: Mapping[str, float]
    black_frame_rate_threshold: float
    reset_backend_before_run: bool
    evaluator_command: Tuple[str, ...]
    evaluator_timeout_s: float
    quality_scope: str
    research_grade: bool
    worker: RemoteWorkerConfig


def load_remote_campaign_config(path: Path) -> RemoteCampaignConfig:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RemoteConfigError("unable to read remote config %s: %s" % (path, exc))
    root = _mapping(raw, "config")
    remote_raw = _mapping(root.get("remote"), "remote")
    workflow_raw = _mapping(root.get("workflow"), "workflow")
    runtime_raw = _mapping(root.get("runtime"), "runtime")
    benchmark_raw = _mapping(root.get("benchmark", {}), "benchmark")
    reward_raw = _mapping(root.get("reward"), "reward")
    worker_raw = _mapping(root.get("worker", {}), "worker")
    remote = RemoteConfig(
        host=str(remote_raw.get("host", "")).strip(),
        harness_root=str(remote_raw.get("harness_root", "")).strip(),
        model_root=str(remote_raw.get("model_root", "")).strip(),
        comfyui_root=str(remote_raw.get("comfyui_root", "")).strip(),
        comfyui_port=int(remote_raw.get("comfyui_port", 8188)),
        python=str(remote_raw.get("python", "python3")),
        results_root=str(remote_raw["results_root"]) if remote_raw.get("results_root") else None,
        deployment_dir=str(remote_raw["deployment_dir"]) if remote_raw.get("deployment_dir") else None,
        campaign_root=str(remote_raw["campaign_root"]) if remote_raw.get("campaign_root") else None,
        training_python=str(remote_raw["training_python"]) if remote_raw.get("training_python") else None,
    )
    mutable_raw = _mapping(workflow_raw.get("mutable", {}), "workflow.mutable")
    unknown = sorted(set(mutable_raw) - {"steps", "cfg"})
    if unknown:
        raise RemoteConfigError("unsupported remote mutable key(s): %s" % ", ".join(unknown))
    workflow = WorkflowConfig(
        template=_path(path.parent, workflow_raw.get("template"), "workflow.template"),
        prompt_target=_target(workflow_raw.get("prompt_target"), "workflow.prompt_target"),
        seed_target=_target(workflow_raw["seed_target"], "workflow.seed_target") if workflow_raw.get("seed_target") else None,
        mutable={key: _target(value, "workflow.mutable.%s" % key) for key, value in mutable_raw.items()},
    )
    runtime = RemoteRuntimeConfig(
        experience_path=_path(path.parent, runtime_raw.get("experience_path", "../var/remote-h3/experience.jsonl"), "runtime.experience_path"),
        output_root=_path(path.parent, runtime_raw.get("output_root", "../var/remote-h3/benchmark"), "runtime.output_root"),
        tasks_path=_path(path.parent, runtime_raw.get("tasks_path", "../examples/tasks.yaml"), "runtime.tasks_path"),
    )
    weights = RewardWeights(
        _positive(reward_raw.get("alpha", 1.0), "reward.alpha", allow_zero=True),
        _positive(reward_raw.get("beta", 0.2), "reward.beta", allow_zero=True),
        _positive(reward_raw.get("gamma", 0.2), "reward.gamma", allow_zero=True),
        _positive(reward_raw.get("delta", 0.2), "reward.delta", allow_zero=True),
    )
    splits_raw = benchmark_raw.get("splits", ["sanity", "dev", "heldout"])
    if not isinstance(splits_raw, list) or not splits_raw or not all(str(item).strip() for item in splits_raw):
        raise RemoteConfigError("benchmark.splits must be a non-empty list")
    thresholds_raw = _mapping(benchmark_raw.get("efficiency_thresholds", {}), "benchmark.efficiency_thresholds")
    thresholds = {str(key): _positive(value, "benchmark.efficiency_thresholds.%s" % key, allow_zero=True) for key, value in thresholds_raw.items()}
    if any(value > 1 for value in thresholds.values()):
        raise RemoteConfigError("efficiency thresholds must be between 0 and 1")
    command_raw = benchmark_raw.get("evaluator_command", [])
    if isinstance(command_raw, str):
        command = tuple(shlex.split(command_raw))
    elif isinstance(command_raw, list) and all(isinstance(item, str) for item in command_raw):
        command = tuple(command_raw)
    else:
        raise RemoteConfigError("benchmark.evaluator_command must be a string or list")
    max_steps = int(worker_raw.get("max_steps", 32))
    if max_steps <= 0 or max_steps > 32:
        raise RemoteConfigError("worker.max_steps must be between 1 and 32")
    worker = RemoteWorkerConfig(
        enabled=bool(worker_raw.get("enabled", False)),
        entrypoint=str(worker_raw.get("entrypoint", "")).strip(),
        config_template=str(worker_raw.get("config_template", "")).strip(),
        python=str(worker_raw.get("python") or remote.training_python or remote.python),
        allowed_operators=tuple(str(item) for item in worker_raw.get("allowed_operators", ["recovery_finetune", "step_distill"])),
        max_steps=max_steps,
    )
    if worker.enabled and (not worker.entrypoint or not worker.config_template):
        raise RemoteConfigError("enabled worker requires entrypoint and config_template")
    if not workflow.template.exists():
        raise RemoteConfigError("workflow template does not exist: %s" % workflow.template)
    return RemoteCampaignConfig(
        remote=remote,
        workflow=workflow,
        runtime=runtime,
        reward=weights,
        benchmark_splits=tuple(str(item) for item in splits_raw),
        sampling_interval_s=_positive(benchmark_raw.get("sampling_interval_s", 1.0), "benchmark.sampling_interval_s"),
        efficiency_thresholds=thresholds,
        black_frame_rate_threshold=_positive(benchmark_raw.get("black_frame_rate_threshold", 0.0), "benchmark.black_frame_rate_threshold", allow_zero=True),
        reset_backend_before_run=bool(benchmark_raw.get("reset_backend_before_run", True)),
        evaluator_command=command,
        evaluator_timeout_s=_positive(benchmark_raw.get("evaluator_timeout_s", 120), "benchmark.evaluator_timeout_s"),
        quality_scope=str(root.get("quality_scope", "structural_proxy")),
        research_grade=bool(root.get("research_grade", False)),
        worker=worker,
    )

