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
    worker_device: str
    worker_gpu_wait_s: int
    worker_min_free_memory_gb: float
    worker_student_min_free_memory_gb: float
    remote_campaign_root: str
    remote_config_path: str
    vae_name: str
    local_output_root: Path
    experience_path: Path
    controller: StudentControllerConfig
    evaluation_command: Tuple[str, ...]
    max_rounds: int
    max_failures: int
    max_steps: int
    min_rounds_before_success: int = 1
    teacher_world_size: int = 3
    teacher_rank_min_free_memory_gb: float = 20.0
    controller_hold_file: Optional[str] = None
    controller_release_file: Optional[str] = None
    controller_worker_lease_file: Optional[str] = None

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
    remote_config_path = _absolute_remote(
        student.get("remote_config_path") or str(PurePosixPath(remote.harness_root) / "configs" / path.name),
        "student.remote_config_path",
    )
    for name, value in (
        ("student.teacher_checkpoint", teacher_checkpoint),
        ("student.h3_cache_dir", h3_cache_dir),
        ("student.worker_entrypoint", worker_entrypoint),
        ("student.remote_campaign_root", remote_campaign_root),
        ("student.remote_config_path", remote_config_path),
    ):
        _under(remote, value, name)
    if not remote._under_root(remote_campaign_root, remote.harness_root):
        raise StudentConfigError("student.remote_campaign_root must be under remote.harness_root")
    if not remote._under_root(remote_config_path, remote.harness_root):
        raise StudentConfigError("student.remote_config_path must be under remote.harness_root")
    worker_python = str(student.get("worker_python") or remote.training_python or remote.python).strip()
    if not worker_python:
        raise StudentConfigError("student.worker_python must not be empty")
    worker_device = str(student.get("worker_device", "auto")).strip()
    if not worker_device or (worker_device != "auto" and not worker_device.startswith("cuda:")):
        raise StudentConfigError("student.worker_device must be auto or cuda:N")
    if worker_device != "auto":
        try:
            if int(worker_device.split(":", 1)[1]) < 0:
                raise ValueError
        except (IndexError, TypeError, ValueError) as exc:
            raise StudentConfigError("student.worker_device must be auto or cuda:N") from exc
    try:
        worker_gpu_wait_s = int(student.get("worker_gpu_wait_s", 1800))
        worker_min_free_memory_gb = float(student.get("worker_min_free_memory_gb", 44.3))
        worker_student_min_free_memory_gb = float(student.get("worker_student_min_free_memory_gb", 20.0))
    except (TypeError, ValueError) as exc:
        raise StudentConfigError("student GPU scheduling limits are invalid") from exc
    if worker_gpu_wait_s < 0 or worker_min_free_memory_gb <= 0 or worker_student_min_free_memory_gb <= 0:
        raise StudentConfigError("student GPU scheduling limits are invalid")
    controller_raw = _mapping(student.get("controller", {}), "student.controller")
    controller = StudentControllerConfig(
        provider=str(controller_raw.get("provider", "ollama")).strip().lower(),
        model=str(controller_raw.get("model", "")).strip(),
        base_url=str(controller_raw.get("base_url", "http://127.0.0.1:11434")).strip().rstrip("/"),
        timeout_s=float(controller_raw.get("timeout_s", 180.0)),
    )
    if controller.provider not in {"ollama", "vllm", "openai_compatible", "openai-compatible"} or not controller.model:
        raise StudentConfigError("student.controller requires provider=ollama or vllm and a model")
    if controller.timeout_s <= 0:
        raise StudentConfigError("student.controller.timeout_s must be positive")
    evaluation_command = _command(student.get("evaluation_command"), "student.evaluation_command")
    vae_name = str(student.get("vae_name", "minimax_h3_video_vae_fp16.safetensors")).strip()
    if not vae_name or "/" in vae_name or "\\" in vae_name:
        raise StudentConfigError("student.vae_name must be a model filename")
    try:
        max_rounds = int(student.get("max_rounds", 4))
        max_failures = int(student.get("max_failures", 4))
        max_steps = int(student.get("max_steps", 32))
        min_rounds_before_success = int(student.get("min_rounds_before_success", 1))
    except (TypeError, ValueError) as exc:
        raise StudentConfigError("student round limits must be integers") from exc
    if (
        max_rounds <= 0
        or max_failures < 0
        or max_steps <= 0
        or min_rounds_before_success <= 0
        or min_rounds_before_success > max_rounds
    ):
        raise StudentConfigError("student round limits are invalid")
    try:
        teacher_world_size = int(student.get("teacher_world_size", 3))
        teacher_rank_min_free_memory_gb = float(student.get("teacher_rank_min_free_memory_gb", 20.0))
    except (TypeError, ValueError) as exc:
        raise StudentConfigError("student distributed teacher limits are invalid") from exc
    if teacher_world_size < 2 or teacher_world_size > 3 or teacher_rank_min_free_memory_gb <= 0:
        raise StudentConfigError("student distributed teacher limits are invalid")
    handoff_raw = _mapping(student.get("controller_handoff", {}), "student.controller_handoff")
    controller_hold_file = handoff_raw.get("hold_file")
    controller_release_file = handoff_raw.get("release_file")
    controller_worker_lease_file = handoff_raw.get("worker_lease_file")
    handoff_values = {
        "student.controller_handoff.hold_file": controller_hold_file,
        "student.controller_handoff.release_file": controller_release_file,
        "student.controller_handoff.worker_lease_file": controller_worker_lease_file,
    }
    for name, value in handoff_values.items():
        if value is not None:
            if not isinstance(value, str) or not value.strip() or not PurePosixPath(value).is_absolute():
                raise StudentConfigError("%s must be an absolute remote path" % name)
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
        worker_device=worker_device,
        worker_gpu_wait_s=worker_gpu_wait_s,
        worker_min_free_memory_gb=worker_min_free_memory_gb,
        worker_student_min_free_memory_gb=worker_student_min_free_memory_gb,
        remote_campaign_root=remote_campaign_root,
        remote_config_path=remote_config_path,
        vae_name=vae_name,
        local_output_root=local_output_root,
        experience_path=experience_path,
        controller=controller,
        evaluation_command=evaluation_command,
        max_rounds=max_rounds,
        max_failures=max_failures,
        max_steps=max_steps,
        min_rounds_before_success=min_rounds_before_success,
        teacher_world_size=teacher_world_size,
        teacher_rank_min_free_memory_gb=teacher_rank_min_free_memory_gb,
        controller_hold_file=str(controller_hold_file) if controller_hold_file else None,
        controller_release_file=str(controller_release_file) if controller_release_file else None,
        controller_worker_lease_file=str(controller_worker_lease_file) if controller_worker_lease_file else None,
    )


__all__ = ["StudentCampaignConfig", "StudentConfigError", "StudentControllerConfig", "load_student_campaign_config"]
