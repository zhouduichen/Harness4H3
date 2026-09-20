"""Trusted SSH adapters for one Student campaign round."""

from __future__ import annotations

import shlex
from dataclasses import fields
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..remote.ssh import SSHClient
from .compiler import CompileManifest
from .config import StudentCampaignConfig
from .evaluator import StudentEvaluation
from .worker import TrainingResult


class RemoteStudentWorker:
    def __init__(self, config: StudentCampaignConfig, client: SSHClient):
        self.config = config
        self.client = client

    def _remote_round_dir(self, round_dir: Path) -> str:
        name = round_dir.name
        if not name.startswith("student_") or "/" in name or "\\" in name:
            raise ValueError("invalid Student round directory")
        return str(PurePosixPath(self.config.remote_campaign_root) / name)

    def run(self, manifest: CompileManifest, round_dir: Path) -> TrainingResult:
        remote_dir = self._remote_round_dir(round_dir)
        remote_manifest = remote_dir + "/compile_manifest.json"
        remote_result = remote_dir + "/training-result.json"
        self.client.write_json(remote_manifest, manifest.to_dict())
        command = (
            self.config.worker_python,
            self.config.worker_entrypoint,
            "--manifest",
            remote_manifest,
            "--teacher",
            self.config.teacher_checkpoint,
            "--output",
            remote_dir,
            "--result",
            remote_result,
            "--comfyui-root",
            self.config.remote.comfyui_root,
            "--cache-dir",
            self.config.h3_cache_dir,
            "--max-steps",
            str(self.config.max_steps),
        )
        self.client.run(command, timeout_s=max(3600.0, self.config.controller.timeout_s * 20), check=False)
        raw = self.client.read_json(remote_result)
        if not isinstance(raw, Mapping):
            raise ValueError("remote Student worker result must be an object")
        allowed = {item.name for item in fields(TrainingResult)}
        payload = {name: raw[name] for name in allowed if name in raw}
        missing = sorted(allowed - set(payload) - {"failure_code", "message"})
        if missing:
            raise ValueError("remote Student worker result missing: %s" % ", ".join(missing))
        return TrainingResult(**payload)


class RemoteStudentEvaluator:
    """Invoke a fixed trusted evaluator command and normalize its JSON result."""

    def __init__(self, config: StudentCampaignConfig, client: SSHClient):
        self.config = config
        self.client = client

    def evaluate(self, checkpoint: Path, round_dir: Path) -> StudentEvaluation:
        remote_dir = str(PurePosixPath(self.config.remote_campaign_root) / round_dir.name)
        remote_result = remote_dir + "/evaluation-result.json"
        command = tuple(self.config.evaluation_command) + (
            "--checkpoint",
            str(checkpoint),
            "--output",
            remote_dir,
            "--result",
            remote_result,
        )
        self.client.run(command, timeout_s=max(3600.0, self.config.controller.timeout_s * 20), check=False)
        raw = self.client.read_json(remote_result)
        if not isinstance(raw, Mapping):
            raise ValueError("remote Student evaluator result must be an object")
        return StudentEvaluation(
            valid=bool(raw.get("valid", False)),
            promotable=bool(raw.get("promotable", False)),
            failure_code=str(raw["failure_code"]) if raw.get("failure_code") else None,
            message=str(raw.get("message", "")),
            video_path=str(raw.get("video_path", "")),
            validity=dict(raw.get("validity") or {}),
            quality_score=float(raw["quality_score"]) if raw.get("quality_score") is not None else None,
            quality_metrics=dict(raw.get("quality_metrics") or {}),
            hardware=dict(raw.get("hardware") or {}),
        )


__all__ = ["RemoteStudentEvaluator", "RemoteStudentWorker"]
