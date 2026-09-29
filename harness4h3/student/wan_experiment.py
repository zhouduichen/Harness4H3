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
    device: str = "auto"
    local_files_only: bool = True
    max_cases: int = 4
    seed_count: int = 2
    clip_model_path: Optional[str] = None
    quality_backend: str = "clip_temporal"

    def __post_init__(self) -> None:
        if self.max_cases <= 0 or self.seed_count <= 0:
            raise WanExperimentConfigError("max_cases and seed_count must be positive")
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
            "device",
            "local_files_only",
            "max_cases",
            "seed_count",
            "clip_model_path",
            "quality_backend",
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
            "--device",
            self.config.device,
            "--max-cases",
            str(self.config.max_cases),
            "--seed-count",
            str(self.config.seed_count),
            "--quality-backend",
            self.config.quality_backend,
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


__all__ = [
    "RemoteWanExperiment",
    "WanExperimentConfigError",
    "WanRealExperimentConfig",
]
