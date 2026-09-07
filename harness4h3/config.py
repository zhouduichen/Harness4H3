from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from . import Harness4H3Error


class ConfigError(Harness4H3Error):
    pass


@dataclass(frozen=True)
class Target:
    node_id: str
    input_name: str


@dataclass(frozen=True)
class BackendConfig:
    base_url: str
    request_timeout_s: float
    poll_interval_s: float
    task_timeout_s: float


@dataclass(frozen=True)
class WorkflowConfig:
    template: Path
    prompt_target: Target
    seed_target: Optional[Target]
    mutable: Mapping[str, Target]


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: Path
    trajectory_path: Path
    archive_dir: Path
    tasks_path: Path


@dataclass(frozen=True)
class EvaluatorConfig:
    command: Tuple[str, ...]
    timeout_s: float


@dataclass(frozen=True)
class EvolutionConfig:
    min_delta: float
    recurrence_threshold: int
    max_regressions: int
    low_score_threshold: float


@dataclass(frozen=True)
class AppConfig:
    backend: BackendConfig
    workflow: WorkflowConfig
    runtime: RuntimeConfig
    evaluator: EvaluatorConfig
    evolution: EvolutionConfig


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError("%s must be a mapping" % name)
    return value


def _target(value: Any, name: str) -> Target:
    raw = _mapping(value, name)
    node_id = str(raw.get("node_id", "")).strip()
    input_name = str(raw.get("input", "")).strip()
    if not node_id or not input_name:
        raise ConfigError("%s requires node_id and input" % name)
    return Target(node_id, input_name)


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("%s must be a path" % name)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _positive_float(value: Any, name: str, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ConfigError("%s must be a number" % name)
    if parsed < 0 if allow_zero else parsed <= 0:
        raise ConfigError("%s must be %s" % (name, "non-negative" if allow_zero else "positive"))
    return parsed


def _non_negative_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ConfigError("%s must be an integer" % name)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ConfigError("%s must be an integer" % name)
    if parsed < minimum:
        raise ConfigError("%s must be at least %d" % (name, minimum))
    return parsed


def load_config(path: Path) -> AppConfig:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("unable to read config %s: %s" % (path, exc))
    root = _mapping(raw, "config")
    backend_raw = _mapping(root.get("backend"), "backend")
    workflow_raw = _mapping(root.get("workflow"), "workflow")
    runtime_raw = _mapping(root.get("runtime"), "runtime")
    evaluator_raw = _mapping(root.get("evaluator", {}), "evaluator")
    evolution_raw = _mapping(root.get("evolution", {}), "evolution")

    mutable_raw = _mapping(workflow_raw.get("mutable", {}), "workflow.mutable")
    allowed = {"steps", "cfg", "stability"}
    unknown = sorted(set(mutable_raw) - allowed)
    if unknown:
        raise ConfigError("unsupported mutable workflow key(s): %s" % ", ".join(unknown))

    command_raw = evaluator_raw.get("command", [])
    if isinstance(command_raw, str):
        command = tuple(shlex.split(command_raw))
    elif isinstance(command_raw, list) and all(isinstance(item, str) for item in command_raw):
        command = tuple(command_raw)
    else:
        raise ConfigError("evaluator.command must be a string or list of strings")

    base_url = str(backend_raw.get("base_url", "")).strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError("backend.base_url must use http or https")

    config = AppConfig(
        backend=BackendConfig(
            base_url=base_url,
            request_timeout_s=_positive_float(backend_raw.get("request_timeout_s", 30), "backend.request_timeout_s"),
            poll_interval_s=_positive_float(backend_raw.get("poll_interval_s", 2), "backend.poll_interval_s", allow_zero=True),
            task_timeout_s=_positive_float(backend_raw.get("task_timeout_s", 3600), "backend.task_timeout_s"),
        ),
        workflow=WorkflowConfig(
            template=_path(path.parent, workflow_raw.get("template"), "workflow.template"),
            prompt_target=_target(workflow_raw.get("prompt_target"), "workflow.prompt_target"),
            seed_target=_target(workflow_raw["seed_target"], "workflow.seed_target") if workflow_raw.get("seed_target") else None,
            mutable={key: _target(value, "workflow.mutable.%s" % key) for key, value in mutable_raw.items()},
        ),
        runtime=RuntimeConfig(
            output_dir=_path(path.parent, runtime_raw.get("output_dir", "var/outputs"), "runtime.output_dir"),
            trajectory_path=_path(path.parent, runtime_raw.get("trajectory_path", "var/trajectories.jsonl"), "runtime.trajectory_path"),
            archive_dir=_path(path.parent, runtime_raw.get("archive_dir", "var/archive"), "runtime.archive_dir"),
            tasks_path=_path(path.parent, runtime_raw.get("tasks_path", "tasks.yaml"), "runtime.tasks_path"),
        ),
        evaluator=EvaluatorConfig(
            command=command,
            timeout_s=_positive_float(evaluator_raw.get("timeout_s", 120), "evaluator.timeout_s"),
        ),
        evolution=EvolutionConfig(
            min_delta=_positive_float(evolution_raw.get("min_delta", 0.01), "evolution.min_delta", allow_zero=True),
            recurrence_threshold=_non_negative_int(evolution_raw.get("recurrence_threshold", 2), "evolution.recurrence_threshold", 2),
            max_regressions=_non_negative_int(evolution_raw.get("max_regressions", 0), "evolution.max_regressions"),
            low_score_threshold=_positive_float(evolution_raw.get("low_score_threshold", 0.65), "evolution.low_score_threshold", allow_zero=True),
        ),
    )
    validate_config(config)
    return config


def _load_workflow(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("unable to read workflow %s: %s" % (path, exc))
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("workflow must be a non-empty API-format object")
    if "nodes" in raw or "links" in raw:
        raise ConfigError("workflow must be API format, not a frontend graph")
    for node_id, node in raw.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
            raise ConfigError("invalid API workflow node %r" % node_id)
    return raw


def _validate_target(workflow: Mapping[str, Any], target: Target, name: str) -> None:
    if target.node_id not in workflow:
        raise ConfigError("%s references missing node %s" % (name, target.node_id))
    if target.input_name not in workflow[target.node_id]["inputs"]:
        raise ConfigError("%s references missing input %s:%s" % (name, target.node_id, target.input_name))


def validate_config(config: AppConfig) -> None:
    workflow = _load_workflow(config.workflow.template)
    _validate_target(workflow, config.workflow.prompt_target, "workflow.prompt_target")
    if config.workflow.seed_target:
        _validate_target(workflow, config.workflow.seed_target, "workflow.seed_target")
    for key, target in config.workflow.mutable.items():
        _validate_target(workflow, target, "workflow.mutable.%s" % key)
    if config.evolution.low_score_threshold > 1:
        raise ConfigError("evolution.low_score_threshold must be between 0 and 1")
    resolved = {
        config.runtime.output_dir.resolve(),
        config.runtime.trajectory_path.resolve(),
        config.runtime.archive_dir.resolve(),
    }
    if len(resolved) != 3:
        raise ConfigError("runtime output, trajectory, and archive paths must be distinct")
