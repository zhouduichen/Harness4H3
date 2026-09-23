"""Trusted SSH adapters for one Student campaign round."""

from __future__ import annotations

import shlex
from dataclasses import MISSING, fields
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..remote.ssh import SSHClient
from .compiler import CompileManifest
from .config import StudentCampaignConfig
from .edge import EdgeEvidence, validate_edge_evidence
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

    def run(
        self,
        manifest: CompileManifest,
        round_dir: Path,
        *,
        train_steps: int | None = None,
        parent_checkpoint: str | Path | None = None,
        parent_candidate_id: str | None = None,
        fidelity: str = "F1",
    ) -> TrainingResult:
        remote_dir = self._remote_round_dir(round_dir)
        remote_manifest = remote_dir + "/compile_manifest.json"
        remote_result = remote_dir + "/training-result.json"
        # Never let a killed worker's previous result satisfy a new launch.
        self.client.remove_file(remote_result)
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
            "--device",
            self.config.worker_device,
            "--wait-for-gpu-s",
            str(self.config.worker_gpu_wait_s),
            "--min-free-memory-gb",
            str(self.config.worker_min_free_memory_gb),
            "--student-min-free-memory-gb",
            str(self.config.worker_student_min_free_memory_gb),
            "--student-memory-safety-margin-gb",
            str(self.config.worker_student_memory_safety_margin_gb),
            "--teacher-world-size",
            str(self.config.teacher_world_size),
            "--teacher-rank-min-free-memory-gb",
            str(self.config.teacher_rank_min_free_memory_gb),
            "--controller-hold-file",
            self.config.controller_hold_file or "",
            "--controller-release-file",
            self.config.controller_release_file or "",
            "--controller-worker-lease-file",
            self.config.controller_worker_lease_file or "",
            "--max-steps",
            str(self.config.max_steps),
            "--fidelity",
            str(fidelity),
            "--parent-candidate-id",
            str(parent_candidate_id or ""),
        )
        if train_steps is not None:
            command += ("--train-steps", str(int(train_steps)))
        if parent_checkpoint:
            command += ("--parent-checkpoint", str(parent_checkpoint))
        self.client.run(command, timeout_s=max(3600.0, self.config.controller.timeout_s * 20), check=False)
        raw = self.client.read_json(remote_result)
        if not isinstance(raw, Mapping):
            raise ValueError("remote Student worker result must be an object")
        allowed = {item.name for item in fields(TrainingResult)}
        payload = {name: raw[name] for name in allowed if name in raw}
        optional = {
            item.name
            for item in fields(TrainingResult)
            if item.default is not MISSING or item.default_factory is not MISSING
        }
        missing = sorted(allowed - set(payload) - optional)
        if missing:
            raise ValueError("remote Student worker result missing: %s" % ", ".join(missing))
        return TrainingResult(**payload)


class RemoteStudentEvaluator:
    """Invoke a fixed trusted evaluator command and normalize its JSON result."""

    def __init__(self, config: StudentCampaignConfig, client: SSHClient):
        self.config = config
        self.client = client

    def evaluate(
        self,
        checkpoint: Path,
        round_dir: Path,
        *,
        fidelity: str = "F3",
        evaluation_cases: int | None = None,
        seed_count: int | None = None,
        verifier_strength: str | None = None,
        timeout_s: float | None = None,
    ) -> StudentEvaluation:
        remote_dir = str(PurePosixPath(self.config.remote_campaign_root) / round_dir.name)
        remote_result = remote_dir + "/evaluation-result.json"
        command = tuple(self.config.evaluation_command) + (
            "--checkpoint",
            str(checkpoint),
            "--output",
            remote_dir,
            "--result",
            remote_result,
            "--comfyui-root",
            self.config.remote.comfyui_root,
            "--cache-dir",
            self.config.h3_cache_dir,
            "--vae-name",
            self.config.vae_name,
            "--device",
            self.config.worker_device,
            "--wait-for-gpu-s",
            "600",
            "--min-free-memory-gb",
            "8.0",
            "--controller-hold-file",
            self.config.controller_hold_file or "",
            "--controller-release-file",
            self.config.controller_release_file or "",
            "--controller-worker-lease-file",
            self.config.controller_worker_lease_file or "",
            "--release-controller-handoff",
        )
        if self.config.evaluation_manifest:
            command += ("--evaluation-manifest", self.config.evaluation_manifest)
        if self.config.clip_model_path:
            command += ("--clip-model-path", self.config.clip_model_path)
        command += ("--quality-backend", self.config.quality_backend)
        command += ("--fidelity", str(fidelity))
        if evaluation_cases is not None:
            command += ("--max-cases", str(int(evaluation_cases)))
        if seed_count is not None:
            command += ("--seed-count", str(int(seed_count)))
        if verifier_strength:
            command += ("--verifier-strength", str(verifier_strength))
        evaluation_timeout = float(timeout_s) if timeout_s is not None else max(3600.0, self.config.controller.timeout_s * 20)
        self.client.run(command, timeout_s=max(30.0, evaluation_timeout), check=False)
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
            metric_evidence=dict(raw.get("metric_evidence") or {}),
            reward=float(raw["reward"]) if raw.get("reward") is not None else None,
            reward_terms={str(key): float(value) for key, value in dict(raw.get("reward_terms") or {}).items()},
        )


class RemoteStudentBaseline:
    """Run the fixed trusted H3 calibration worker on the server."""

    def __init__(self, config: StudentCampaignConfig, client: SSHClient):
        self.config = config
        self.client = client

    def run(self) -> Mapping[str, Any]:
        root = PurePosixPath(self.config.remote_campaign_root)
        output = str(root / "teacher-baseline")
        result_path = str(root / "teacher-baseline-result.json")
        command = tuple(self.config.baseline_command) + (
            "--teacher",
            self.config.teacher_checkpoint,
            "--evaluation-manifest",
            str(self.config.evaluation_manifest),
            "--output",
            output,
            "--result",
            result_path,
            "--comfyui-root",
            self.config.remote.comfyui_root,
            "--cache-dir",
            self.config.h3_cache_dir,
            "--vae-name",
            self.config.vae_name,
            "--clip-model-path",
            str(self.config.clip_model_path),
            "--latent-channels",
            str(self.config.target.latent_channels),
            "--latent-frames",
            str(self.config.target.latent_frames),
            "--latent-height",
            str(self.config.target.latent_height),
            "--latent-width",
            str(self.config.target.latent_width),
            "--condition-dim",
            str(self.config.target.condition_dim),
            "--device",
            self.config.worker_device,
            "--wait-for-gpu-s",
            "600",
            "--teacher-world-size",
            str(self.config.teacher_world_size),
            "--teacher-rank-min-free-memory-gb",
            str(self.config.teacher_rank_min_free_memory_gb),
            "--controller-hold-file",
            self.config.controller_hold_file or "",
            "--controller-release-file",
            self.config.controller_release_file or "",
            "--controller-worker-lease-file",
            self.config.controller_worker_lease_file or "",
        )
        self.client.run(command, timeout_s=max(3600.0, self.config.controller.timeout_s * 20), check=False)
        raw = self.client.read_json(result_path)
        if not isinstance(raw, Mapping) or raw.get("status") != "success":
            raise RuntimeError("teacher baseline failed: %s" % dict(raw or {}))
        return dict(raw)


class RemoteTargetDeviceEvaluator:
    """Invoke the configured target-device export/deploy/benchmark command."""

    def __init__(self, config: StudentCampaignConfig, client: SSHClient):
        self.config = config
        self.client = client
        if not config.target_device_command or config.target_device is None:
            raise ValueError("target-device command and TargetDeviceProfile are required")

    def evaluate(self, checkpoint: Path, proposal: Any, round_dir: Path) -> tuple[EdgeEvidence, ...]:
        remote_dir = str(PurePosixPath(self.config.remote_campaign_root) / round_dir.name)
        result_path = remote_dir + "/target-device-result.json"
        proposal_path = remote_dir + "/target-device-proposal.json"
        self.client.remove_file(result_path)
        proposal_payload = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal)
        self.client.write_json(proposal_path, proposal_payload)
        command = tuple(self.config.target_device_command) + (
            "--checkpoint",
            str(checkpoint),
            "--output",
            remote_dir + "/target-device",
            "--proposal",
            proposal_path,
            "--result",
            result_path,
            "--target-device-id",
            self.config.target_device.id,
        )
        self.client.run(command, timeout_s=max(3600.0, self.config.controller.timeout_s * 20), check=False)
        raw = self.client.read_json(result_path)
        if not isinstance(raw, Mapping) or raw.get("status") != "success":
            raise RuntimeError("target-device evaluation failed: %s" % dict(raw or {}))
        raw_evidence = raw.get("evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError("target-device result must contain an evidence array")
        evidence = tuple(EdgeEvidence.from_dict(item) for item in raw_evidence)
        return validate_edge_evidence(evidence, target_device_id=self.config.target_device.id)


class RemoteStudentRetention:
    """Delete only rejected Student payloads under the configured campaign root."""

    _NAMES = {"student.safetensors", "student-int8.safetensors"}

    def __init__(self, config: StudentCampaignConfig, client: SSHClient):
        self.config = config
        self.client = client

    def retain(self, training: TrainingResult, _evaluation: StudentEvaluation, candidate_id: str, outcome: str) -> None:
        if outcome in {"accepted", "active", "best", "in_flight"}:
            return
        candidates = {
            str(value)
            for value in (training.child_checkpoint, training.full_precision_checkpoint, training.quantized_checkpoint)
            if value
        }
        root = PurePosixPath(self.config.remote_campaign_root)
        for value in sorted(candidates):
            path = PurePosixPath(value)
            if path.name not in self._NAMES or path.parent.name != candidate_id or root not in path.parents:
                continue
            self.client.remove_file(str(path))


class RemoteStudentSupervisor:
    """Launch a detached campaign process without keeping Codex attached."""

    def __init__(self, config: StudentCampaignConfig, client: SSHClient):
        self.config = config
        self.client = client
        root = PurePosixPath(config.remote_campaign_root)
        self.pid_path = str(root / "student-campaign.pid")
        self.log_path = str(root / "student-campaign.log")
        self.result_path = str(root / "campaign-result.json")
        self.entrypoint = str(PurePosixPath(config.remote.harness_root) / "tools" / "student_campaign_supervisor.py")

    @staticmethod
    def _q(value: str) -> str:
        return shlex.quote(str(value))

    def start(self, max_rounds: int) -> Mapping[str, Any]:
        rounds = int(max_rounds)
        if rounds <= 0:
            raise ValueError("max_rounds must be positive")
        root = self._q(self.config.remote_campaign_root)
        pid = self._q(self.pid_path)
        log = self._q(self.log_path)
        config = self._q(self.config.remote_config_path)
        python = self._q(self.config.worker_python)
        entrypoint = self._q(self.entrypoint)
        script = (
            "set -eu; mkdir -p {root}; "
            "if test -s {pid}; then old=$(cat {pid}); "
            "case $old in (*[!0-9]*|'') old='';; esac; "
            "if test -n \"$old\" && kill -0 \"$old\" 2>/dev/null; then "
            "echo running:$old; exit 0; fi; fi; "
            "nohup {python} {entrypoint} --config {config} --max-rounds {rounds} > {log} 2>&1 < /dev/null & "
            "new=$!; printf '%s\\n' \"$new\" > {pid}; echo started:$new"
        ).format(root=root, pid=pid, python=python, entrypoint=entrypoint, config=config, rounds=rounds, log=log)
        result = self.client.run(("bash", "-lc", script), timeout_s=30.0)
        output = str(getattr(result, "stdout", "")).strip()
        state, _, value = output.partition(":")
        return {
            "status": "already_running" if state == "running" else "started",
            "pid": int(value) if value.isdigit() else None,
            "pid_path": self.pid_path,
            "log_path": self.log_path,
            "result_path": self.result_path,
        }

    def status(self) -> Mapping[str, Any]:
        pid = self._q(self.pid_path)
        result = self.client.run(("bash", "-lc", "if test -s %s; then p=$(cat %s); if kill -0 \"$p\" 2>/dev/null; then echo running:$p; else echo stopped:$p; fi; else echo missing; fi" % (pid, pid)), check=False, timeout_s=30.0)
        value = str(getattr(result, "stdout", "")).strip()
        state, _, raw_pid = value.partition(":")
        payload: dict[str, Any] = {"status": state or "unknown", "pid": int(raw_pid) if raw_pid.isdigit() else None, "pid_path": self.pid_path, "log_path": self.log_path, "result_path": self.result_path}
        if state == "stopped":
            try:
                payload["result"] = self.client.read_json(self.result_path)
            except Exception:
                payload["result"] = None
        return payload


__all__ = ["RemoteStudentBaseline", "RemoteStudentEvaluator", "RemoteStudentRetention", "RemoteStudentSupervisor", "RemoteStudentWorker"]
