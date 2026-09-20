"""Trusted configuration for the autonomous Student campaign."""

from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Tuple

import yaml

from ..remote.ssh import RemoteConfig
from .proposal import StudentTarget


class StudentConfigError(ValueError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudentConfigError("%s must be a mapping" % name)
    return value


def _absolute_remote(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudentConfigError("%s must be a non-empty remote path" % name)
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise StudentConfigError("%s must be absolute" % name)
    return str(path)


def _local_path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StudentConfigError("%s must be a path" % name)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _command(value: Any, name: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        parsed = tuple(shlex.split(value))
    elif isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        parsed = tuple(value)
    else:
        raise StudentConfigError("%s must be a command string or list of strings" % name)
    if not parsed:
        raise StudentConfigError("%s must not be empty" % name)
    return parsed


@dataclass(frozen=True)
class StudentControllerConfig:
    provider: str
    model: str
    base_url: str
    timeout_s: float


@dataclass(frozen=True)
class StudentCampaignConfig:
    goal: str
    target: StudentTarget
    remote: RemoteConfig
    teacher_checkpoint: str
    h3_cache_dir: str
    worker_entrypoint: str
    worker_python: str
    remote_campaign_root: str
    local_output_root: Path
    experience_path: Path
    controller: StudentControllerConfig
    evaluation_command: Tuple[str, ...]
    max_rounds: int
    max_failures: int
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target"] = asdict(self.target)
        value["local_output_root"] = str(self.local_output_root)
        value["experience_path"] = str(self.experience_path)
        value["evaluation_command"] = list(self.evaluation_command)
        value["remote"] = asdict(self.remote)
        return value


def _remote_config(raw: Mapping[str, Any]) -> RemoteConfig:
    return RemoteConfig(
        host=str(raw.get("host", "")).strip(),
        harness_root=_absolute_remote(raw.get("harness_root"), "remote.harness_root"),
        model_root=_absolute_remote(raw.get("model_root"), "remote.model_root"),
        comfyui_root=_absolute_remote(raw.get("comfyui_root"), "remote.comfyui_root"),
        comfyui_port=int(raw.get("comfyui_port", 8188)),
        python=str(raw.get("python", "python3")),
        results_root=str(raw["results_root"]) if raw.get("results_root") else None,
        deployment_dir=str(raw["deployment_dir"]) if raw.get("deployment_dir") else None,
        campaign_root=str(raw["campaign_root"]) if raw.get("campaign_root") else None,
        training_python=str(raw["training_python"]) if raw.get("training_python") else None,
        ssh_port=int(raw.get("ssh_port", 22)),
    )


def _target(raw: Any) -> StudentTarget:
    value = _mapping(raw or {}, "student.target")
    allowed = set(StudentTarget.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise StudentConfigError("student.target has unknown field(s): %s" % ", ".join(unknown))
    defaults = asdict(StudentTarget())
    defaults.update(value)
    try:
        return StudentTarget(**defaults)
    except (TypeError, ValueError) as exc:
        raise StudentConfigError("invalid student.target: %s" % exc) from exc


def _under(remote: RemoteConfig, path: str, name: str) -> None:
    try:
        if not remote._under_any_root(path):
            raise StudentConfigError("%s escapes configured remote roots" % name)
    except ValueError as exc:
        raise StudentConfigError("invalid %s: %s" % (name, exc)) from exc


def load_student_campaign_config(path: Path) -> StudentCampaignConfig:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StudentConfigError("unable to read student config %s: %s" % (path, exc)) from exc
    root = _mapping(raw, "config")
    student = _mapping(root.get("student"), "student")
    remote = _remote_config(_mapping(root.get("remote"), "remote"))
    goal = str(student.get("goal", "")).strip()
    if not goal:
        raise StudentConfigError("student.goal must not be empty")
    target = _target(student.get("target", {}))
    teacher_checkpoint = _absolute_remote(student.get("teacher_checkpoint"), "student.teacher_checkpoint")
    h3_cache_dir = _absolute_remote(student.get("h3_cache_dir"), "student.h3_cache_dir")
    worker_entrypoint = _absolute_remote(student.get("worker_entrypoint"), "student.worker_entrypoint")
    remote_campaign_root = _absolute_remote(
        student.get("remote_campaign_root") or remote.resolved_campaign_root,
        "student.remote_campaign_root",
    )
    for name, value in (
        ("student.teacher_checkpoint", teacher_checkpoint),
        ("student.h3_cache_dir", h3_cache_dir),
        ("student.worker_entrypoint", worker_entrypoint),
        ("student.remote_campaign_root", remote_campaign_root),
    ):
        _under(remote, value, name)
    if not remote._under_root(remote_campaign_root, remote.harness_root):
        raise StudentConfigError("student.remote_campaign_root must be under remote.harness_root")
    worker_python = str(student.get("worker_python") or remote.training_python or remote.python).strip()
    if not worker_python:
        raise StudentConfigError("student.worker_python must not be empty")
    controller_raw = _mapping(student.get("controller", {}), "student.controller")
    controller = StudentControllerConfig(
        provider=str(controller_raw.get("provider", "ollama")).strip().lower(),
        model=str(controller_raw.get("model", "")).strip(),
        base_url=str(controller_raw.get("base_url", "http://127.0.0.1:11434")).strip().rstrip("/"),
        timeout_s=float(controller_raw.get("timeout_s", 180.0)),
    )
    if controller.provider != "ollama" or not controller.model:
        raise StudentConfigError("student.controller currently requires provider=ollama and a model")
    if controller.timeout_s <= 0:
        raise StudentConfigError("student.controller.timeout_s must be positive")
    evaluation_command = _command(student.get("evaluation_command"), "student.evaluation_command")
    try:
        max_rounds = int(student.get("max_rounds", 4))
        max_failures = int(student.get("max_failures", 4))
        max_steps = int(student.get("max_steps", 32))
    except (TypeError, ValueError) as exc:
        raise StudentConfigError("student round limits must be integers") from exc
    if max_rounds <= 0 or max_failures < 0 or max_steps <= 0:
        raise StudentConfigError("student round limits are invalid")
    local_output_root = _local_path(path.parent, student.get("local_output_root", "var/student-campaign"), "student.local_output_root")
    experience_path = _local_path(path.parent, student.get("experience_path", "var/student-campaign/experience.jsonl"), "student.experience_path")
    return StudentCampaignConfig(
        goal=goal,
        target=target,
        remote=remote,
        teacher_checkpoint=teacher_checkpoint,
        h3_cache_dir=h3_cache_dir,
        worker_entrypoint=worker_entrypoint,
        worker_python=worker_python,
        remote_campaign_root=remote_campaign_root,
        local_output_root=local_output_root,
        experience_path=experience_path,
        controller=controller,
        evaluation_command=evaluation_command,
        max_rounds=max_rounds,
        max_failures=max_failures,
        max_steps=max_steps,
    )


__all__ = ["StudentCampaignConfig", "StudentConfigError", "StudentControllerConfig", "load_student_campaign_config"]
