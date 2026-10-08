"""Real remote-experiment boundary for the pretrained Wan Student baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

import yaml

from ..remote.ssh import RemoteConfig, SSHClient


class WanExperimentConfigError(ValueError):
    pass


def _absolute(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WanExperimentConfigError("%s must be a non-empty absolute path" % name)
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise WanExperimentConfigError("%s must be absolute" % name)
    return str(path)


def _remote(raw: Mapping[str, Any]) -> RemoteConfig:
    try:
        return RemoteConfig(
            host=str(raw.get("host", "")).strip(),
            ssh_port=int(raw.get("ssh_port", 22)),
            harness_root=_absolute(raw.get("harness_root"), "remote.harness_root"),
            model_root=_absolute(raw.get("model_root"), "remote.model_root"),
            comfyui_root=_absolute(raw.get("comfyui_root"), "remote.comfyui_root"),
            comfyui_port=int(raw.get("comfyui_port", 8188)),
            python=str(raw.get("python", "python3")),
            results_root=_absolute(raw["results_root"], "remote.results_root") if raw.get("results_root") else None,
            campaign_root=_absolute(raw["campaign_root"], "remote.campaign_root") if raw.get("campaign_root") else None,
        )
    except (TypeError, ValueError) as exc:
        raise WanExperimentConfigError("invalid remote config: %s" % exc) from exc


@dataclass(frozen=True)
class WanRealExperimentConfig:
    wan_config: str
    evaluation_manifest: str
    worker_entrypoint: str
    worker_python: str
    remote_output_root: str
    remote_result: str
    remote: RemoteConfig
    phase: str = "W0_pretrained_baseline"
    device: str = "cuda:3"
    local_files_only: bool = True
    max_cases: int = 4
    seed_count: int = 2
    clip_model_path: Optional[str] = None
    quality_backend: str = "clip_temporal"
    target_fps: float = 24.0
    target_frames: int = 720
    target_duration_s: float = 30.0
    target_window_count: int = 7
    target_overlap_latent_frames: int = 6
    target_window_stride_latent_frames: int = 30

    def __post_init__(self) -> None:
        if self.max_cases <= 0 or self.seed_count <= 0:
            raise WanExperimentConfigError("max_cases and seed_count must be positive")
        if self.phase != "W0_pretrained_baseline":
            raise WanExperimentConfigError("only W0_pretrained_baseline is implemented")
        if self.device == "auto":
            raise WanExperimentConfigError("device must be fixed for the real remote experiment")
        if self.target_fps <= 0 or self.target_frames <= 0 or self.target_duration_s <= 0:
            raise WanExperimentConfigError("target video contract values must be positive")
        if self.target_window_count <= 0 or self.target_overlap_latent_frames < 0 or self.target_window_stride_latent_frames <= 0:
            raise WanExperimentConfigError("target window contract is invalid")
        if self.quality_backend not in {"clip_temporal", "none"}:
            raise WanExperimentConfigError("quality_backend must be clip_temporal or none")
        for path_name in (
            "wan_config",
            "evaluation_manifest",
            "worker_entrypoint",
            "remote_output_root",
            "remote_result",
        ):
            path = str(getattr(self, path_name))
            if not PurePosixPath(path).is_absolute():
                raise WanExperimentConfigError("%s must be absolute" % path_name)
        if self.clip_model_path is not None and not PurePosixPath(self.clip_model_path).is_absolute():
            raise WanExperimentConfigError("clip_model_path must be absolute")
        if not self.remote._under_any_root(self.wan_config):
            raise WanExperimentConfigError("wan_config escapes configured remote roots")
        if not self.remote._under_any_root(self.evaluation_manifest):
            raise WanExperimentConfigError("evaluation_manifest escapes configured remote roots")
        if not self.remote._under_root(self.worker_entrypoint, self.remote.harness_root):
            raise WanExperimentConfigError("worker_entrypoint must be under remote.harness_root")
        if not self.remote._under_root(self.remote_output_root, self.remote.harness_root):
            raise WanExperimentConfigError("remote_output_root must be under remote.harness_root")
        if not self.remote._under_root(self.remote_result, self.remote_output_root):
            raise WanExperimentConfigError("remote_result must be under remote_output_root")
        if self.clip_model_path is not None and not self.remote._under_any_root(self.clip_model_path):
            raise WanExperimentConfigError("clip_model_path escapes configured remote roots")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["remote"] = asdict(self.remote)
        return value

    @classmethod
    def from_path(cls, path: Path) -> "WanRealExperimentConfig":
        config_path = Path(path).resolve()
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise WanExperimentConfigError("unable to read Wan experiment config: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise WanExperimentConfigError("Wan experiment config must be a mapping")
        experiment = raw.get("wan_experiment")
        remote_raw = raw.get("remote")
        if not isinstance(experiment, Mapping) or not isinstance(remote_raw, Mapping):
            raise WanExperimentConfigError("config requires wan_experiment and remote mappings")
        allowed = {
            "wan_config",
            "evaluation_manifest",
            "worker_entrypoint",
            "worker_python",
            "remote_output_root",
            "remote_result",
            "phase",
            "device",
            "local_files_only",
            "max_cases",
            "seed_count",
            "clip_model_path",
            "quality_backend",
            "target_fps",
            "target_frames",
            "target_duration_s",
            "target_window_count",
            "target_overlap_latent_frames",
            "target_window_stride_latent_frames",
        }
        unknown = sorted(set(experiment) - allowed)
        if unknown:
            raise WanExperimentConfigError("wan_experiment has unknown field(s): %s" % ", ".join(map(str, unknown)))
        try:
            return cls(remote=_remote(remote_raw), **dict(experiment))
        except (TypeError, ValueError) as exc:
            raise WanExperimentConfigError("invalid Wan experiment config: %s" % exc) from exc


class RemoteWanExperiment:
    """Run the fixed Wan W0 worker on the configured real experiment host."""

    def __init__(self, config: WanRealExperimentConfig, client: SSHClient):
        self.config = config
        self.client = client

    def run(self) -> Mapping[str, Any]:
        result_path = self.config.remote_result
        self.client.remove_file(result_path)
        command = (
            self.config.worker_python,
            self.config.worker_entrypoint,
            "--wan-config",
            self.config.wan_config,
            "--evaluation-manifest",
            self.config.evaluation_manifest,
            "--output",
            self.config.remote_output_root,
            "--result",
            result_path,
            "--phase",
            self.config.phase,
            "--device",
            self.config.device,
            "--max-cases",
            str(self.config.max_cases),
            "--seed-count",
            str(self.config.seed_count),
            "--quality-backend",
            self.config.quality_backend,
            "--target-fps",
            str(self.config.target_fps),
            "--target-frames",
            str(self.config.target_frames),
            "--target-duration-s",
            str(self.config.target_duration_s),
            "--target-window-count",
            str(self.config.target_window_count),
            "--target-overlap-latent-frames",
            str(self.config.target_overlap_latent_frames),
            "--target-window-stride-latent-frames",
            str(self.config.target_window_stride_latent_frames),
        )
        if self.config.local_files_only:
            command += ("--local-files-only",)
        if self.config.clip_model_path:
            command += ("--clip-model-path", self.config.clip_model_path)
        self.client.run(command, timeout_s=86400.0, check=False)
        raw = self.client.read_json(result_path)
        if not isinstance(raw, Mapping):
            raise WanExperimentConfigError("remote Wan result must be a JSON object")
        return dict(raw)


@dataclass(frozen=True)
class WanTrainingExperimentConfig:
    """Isolated remote command boundary for the real Wan 5-second worker."""

    training_config: str
    worker_entrypoint: str
    remote_output_root: str
    remote_result: str
    remote: RemoteConfig
    steps: int = 1

    def __post_init__(self) -> None:
        if int(self.steps) <= 0:
            raise WanExperimentConfigError("steps must be positive")
        for name in ("training_config", "worker_entrypoint", "remote_output_root", "remote_result"):
            value = str(getattr(self, name))
            if not PurePosixPath(value).is_absolute():
                raise WanExperimentConfigError("%s must be absolute" % name)
        if not self.remote._under_root(self.training_config, self.remote.harness_root):
            raise WanExperimentConfigError("training_config must be under remote.harness_root")
        if not self.remote._under_root(self.worker_entrypoint, self.remote.harness_root):
            raise WanExperimentConfigError("worker_entrypoint must be under remote.harness_root")
        if not self.remote._under_root(self.remote_output_root, self.remote.harness_root):
            raise WanExperimentConfigError("remote_output_root must be under remote.harness_root")
        if not self.remote._under_root(self.remote_result, self.remote_output_root):
            raise WanExperimentConfigError("remote_result must be under remote_output_root")

    @classmethod
    def from_path(cls, path: Path) -> "WanTrainingExperimentConfig":
        config_path = Path(path).resolve()
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise WanExperimentConfigError("unable to read Wan training config: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise WanExperimentConfigError("Wan training config must be a mapping")
        training = raw.get("wan_training")
        remote_raw = raw.get("remote")
        if not isinstance(training, Mapping) or not isinstance(remote_raw, Mapping):
            raise WanExperimentConfigError("config requires wan_training and remote mappings")
        remote = _remote(remote_raw)
        remote_config = str(PurePosixPath(remote.harness_root) / "configs" / config_path.name)
        worker = str(PurePosixPath(remote.harness_root) / "tools" / "wan_student_train_worker.py")
        output = training.get("remote_output_root", remote_raw.get("output_root"))
        result = training.get("remote_result", remote_raw.get("result_path"))
        if not isinstance(output, str) or not isinstance(result, str):
            raise WanExperimentConfigError("remote output_root and result_path are required")
        return cls(
            training_config=remote_config,
            worker_entrypoint=worker,
            remote_output_root=output,
            remote_result=result,
            remote=remote,
            steps=int(training.get("train_steps", 1)),
        )


class RemoteWanTrainingExperiment:
    """Start the fixed real Wan 5-second worker on the configured host."""

    def __init__(self, config: WanTrainingExperimentConfig, client: SSHClient):
        self.config = config
        self.client = client

    def run(self) -> Mapping[str, Any]:
        self.client.remove_file(self.config.remote_result)
        command = (
            self.config.remote.python,
            self.config.worker_entrypoint,
            "--config",
            self.config.training_config,
            "--output",
            self.config.remote_output_root,
            "--result",
            self.config.remote_result,
            "--steps",
            str(self.config.steps),
        )
        self.client.run(command, timeout_s=86400.0, check=False)
        raw = self.client.read_json(self.config.remote_result)
        if not isinstance(raw, Mapping):
            raise WanExperimentConfigError("remote Wan training result must be a JSON object")
        return dict(raw)


__all__ = [
    "RemoteWanExperiment",
    "RemoteWanTrainingExperiment",
    "WanExperimentConfigError",
    "WanRealExperimentConfig",
    "WanTrainingExperimentConfig",
]
