#!/usr/bin/env python3
"""Run the real Wan Student evaluation -> vLLM proposal -> training loop."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness4h3.student.wan_campaign import (
    WanProposalError,
    WanTrainingProposal,
    append_event,
    atomic_write_json,
    candidate_decision,
    load_campaign_state,
    load_complete_rounds,
    register_fingerprint,
    sha256_file,
    write_campaign_state,
)
from harness4h3.student.wan_training import sha256_path
from harness4h3.student.gpu import acquire_controller_handoff, release_controller_handoff
from harness4h3.memory.experience import ExperienceRecord, ExperienceStore
from harness4h3.memory.fingerprint import experiment_fingerprint


def proposal_json_schema() -> dict[str, Any]:
    positive = {"type": "number", "exclusiveMinimum": 0, "maximum": 0.01}
    non_negative = {"type": "number", "minimum": 0, "maximum": 10.0}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "proposal_id",
            "parent_checkpoint_sha256",
            "training",
            "initialization",
            "precision",
            "curriculum",
            "data",
            "continuation",
            "focus",
            "hypothesis",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": 2},
            "proposal_id": {"type": "string", "pattern": "^wan_[0-9]{4,}$"},
            "parent_checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "training": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "method", "source_steps", "target_steps", "train_steps", "learning_rate",
                    "critic_learning_rate", "endpoint_weight", "velocity_weight", "clean_anchor_weight", "rollout_steps",
                    "rollout_every", "rollout_endpoint_weight", "rollout_clean_weight", "trainable_scope",
                    "fp32_optimizer_state",
                ],
                "properties": {
                    "method": {"type": "string", "enum": ["progressive_distillation", "dmd2"]},
                    "source_steps": {"type": "integer", "minimum": 1, "maximum": 256},
                    "target_steps": {"type": "integer", "minimum": 1, "maximum": 64},
                    "train_steps": {"type": "integer", "minimum": 1, "maximum": 100000},
                    "learning_rate": positive,
                    "critic_learning_rate": positive,
                    "endpoint_weight": non_negative,
                    "velocity_weight": non_negative,
                    "clean_anchor_weight": non_negative,
                    "rollout_steps": {"type": "integer", "enum": [0, 2, 4]},
                    "rollout_every": {"type": "integer", "minimum": 1, "maximum": 16},
                    "rollout_endpoint_weight": non_negative,
                    "rollout_clean_weight": non_negative,
                    "trainable_scope": {"type": "string", "minLength": 1, "maxLength": 512},
                    "fp32_optimizer_state": {"type": "boolean"},
                },
            },
            "initialization": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mode"],
                "properties": {"mode": {"type": "string", "enum": ["inherit", "generation_head_reset", "dit_zero"]}},
            },
            "precision": {
                "type": "object",
                "additionalProperties": False,
                "required": ["loss_reduction", "optimizer_state"],
                "properties": {
                    "loss_reduction": {"type": "string", "enum": ["bf16", "fp32"]},
                    "optimizer_state": {"type": "string", "enum": ["bf16", "fp32"]},
                },
            },
            "curriculum": {
                "type": "object",
                "additionalProperties": False,
                "required": ["fidelity_schedule"],
                "properties": {
                    "fidelity_schedule": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "enum": ["F1", "F2", "F3"]},
                    }
                },
            },
            "data": {
                "type": "object",
                "additionalProperties": False,
                "required": ["manifest_digest", "prompt_policy", "seed_policy"],
                "properties": {
                    "manifest_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "prompt_policy": {"type": "string", "enum": ["balanced", "fixed_manifest"]},
                    "seed_policy": {"type": "string", "enum": ["fixed", "rotating"]},
                },
            },
            "continuation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["overlap_latent_frames", "anchor_schedule"],
                "properties": {
                    "overlap_latent_frames": {"type": "integer", "minimum": 6, "maximum": 12},
                    "anchor_schedule": {"type": "string", "enum": ["constant", "linear", "cosine"]},
                },
            },
            "focus": {"type": "string", "enum": ["semantic", "temporal", "stability", "efficiency"]},
            "hypothesis": {"type": "string", "minLength": 1, "maxLength": 512},
        },
    }


def extract_json_content(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0]["message"]
            content = message["content"] if isinstance(message, Mapping) else None
        else:
            message = payload["message"]
            content = message.get("content") if isinstance(message, Mapping) else None
    except (KeyError, TypeError, IndexError) as exc:
        raise WanProposalError("vLLM response has no assistant content") from exc
    if not isinstance(content, str):
        raise WanProposalError("vLLM assistant content must be a JSON string")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WanProposalError("vLLM assistant content is not valid JSON: %s" % exc) from exc
    if not isinstance(value, Mapping):
        raise WanProposalError("vLLM proposal must be a JSON object")
    return dict(value)


def load_loop_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("wan_loop"), Mapping):
        raise ValueError("config requires a wan_loop mapping")
    loop = dict(raw["wan_loop"])
    required = (
        "training_config",
        "wan_config",
        "evaluation_manifest",
        "clip_model_path",
        "teacher_baseline_result",
        "parent_checkpoint",
        "output_root",
        "worker_python",
        "training_entrypoint",
        "vllm_base_url",
        "vllm_model",
    )
    missing = [name for name in required if not str(loop.get(name, "")).strip()]
    if missing:
        raise ValueError("wan_loop is missing: %s" % ", ".join(missing))
    for name in required[:7] + ("worker_python", "training_entrypoint"):
        if not Path(str(loop[name])).is_absolute():
            raise ValueError("%s must be absolute" % name)
    for name, default in (
        ("max_rounds", 1),
        ("max_cases", 4),
        ("seed_count", 2),
        ("window_count", 7),
        ("overlap_latent_frames", 6),
        ("inference_steps", 30),
        ("vllm_timeout_s", 900),
        ("training_timeout_s", 86400),
        ("semantic_floor_ratio", 0.95),
        ("quality_floor_ratio", 0.90),
        ("min_semantic_delta", 0.005),
        ("no_improvement_patience", 2),
        ("max_train_steps", 100000),
        ("device", "cuda:3"),
        ("gpu_count", 4),
        ("min_free_memory_gb", 35.0),
        ("gpu_wait_s", 3600),
        ("controller_launcher", ""),
        ("controller_launcher_pid_file", ""),
        ("controller_launcher_log", ""),
        ("controller_hold_file", ""),
        ("controller_release_file", ""),
        ("controller_worker_lease_file", ""),
        ("controller_shutdown_on_exit", False),
        ("proposal_schema_version", 2),
        ("algorithm_capabilities", ["progressive_distillation"]),
        ("dmd2_memory_admitted", False),
        ("wan_dmd2_adapter_ready", False),
        ("fidelity_schedule", ["F1", "F2", "F3"]),
        ("continuation_contract", "30s"),
        ("training_mode", "legacy"),
        ("teacher_forced_intervals", 0),
        ("trajectory_record", ""),
        ("state_path", ""),
        ("fingerprint_path", ""),
        ("experience_path", ""),
        ("checkpoint_every_steps", 8),
        ("watchdog_restart_limit", 3),
        ("watchdog_backoff_seconds", 30),
        ("watchdog_stop_file", ""),
        ("watchdog_lock_file", ""),
        ("watchdog_child_pid_file", ""),
        ("watchdog_log", ""),
        ("offline_simulation", False),
        ("correctness_audit_passed", False),
        ("correctness_evidence_path", ""),
        ("correctness_evidence_sha256", ""),
        ("legacy_r2_promotion_frozen", True),
        ("legacy_r2_checkpoint_digests", []),
        ("controller_release_required", False),
        ("controller_release_evidence_path", ""),
        ("controller_release_evidence_sha256", ""),
        ("controller_release_contract", ""),
        ("adaptive_visual_retry_enabled", False),
        ("adaptive_retry_offsets", [1, 2]),
        ("evaluation_parallel_devices", []),
        ("controller_primary_objective", "semantic_alignment"),
        ("controller_secondary_objectives", ["temporal_consistency", "visual_quality", "motion_quality"]),
        ("controller_min_quality_delta", 0.0),
        ("controller_max_temporal_drop", 0.005),
        ("controller_memory_limit", 3),
        ("controller_search_space", {}),
        ("memory_seed_paths", []),
        (
            "visual_quality_gate",
            {
                "enabled": False,
                "version": "visual-structure-v1",
                "min_spatial_gradient": 0.30,
                "min_laplacian_std": 1.8,
                "min_block_variance": 12.0,
                "min_frame_difference": 0.01,
                "max_low_texture_frame_ratio": 0.25,
            },
        ),
    ):
        loop.setdefault(name, default)
    root = Path(str(loop["output_root"]))
    if not loop["controller_launcher_pid_file"]:
        loop["controller_launcher_pid_file"] = str(root / "controller-launcher.pid")
    if not loop["controller_launcher_log"]:
        loop["controller_launcher_log"] = str(root / "controller-launcher.log")
    if not loop["controller_hold_file"]:
        loop["controller_hold_file"] = str(root / ".controller-handoff-hold.json")
    if not loop["controller_release_file"]:
        loop["controller_release_file"] = str(root / ".controller-release.json")
    if not loop["controller_worker_lease_file"]:
        loop["controller_worker_lease_file"] = str(root / ".worker-gpu-lease.json")
    if not loop["state_path"]:
        loop["state_path"] = str(root / "campaign-state.json")
    if not loop["fingerprint_path"]:
        loop["fingerprint_path"] = str(root / "fingerprints.jsonl")
    if not loop["experience_path"]:
        loop["experience_path"] = str(root / "experiences.jsonl")
    if not loop["watchdog_stop_file"]:
        loop["watchdog_stop_file"] = str(root / ".wan-watchdog-stop")
    if not loop["watchdog_lock_file"]:
        loop["watchdog_lock_file"] = str(root / ".watchdog.lock")
    if not loop["watchdog_child_pid_file"]:
        loop["watchdog_child_pid_file"] = str(root / ".campaign-child.pid")
    if not loop["watchdog_log"]:
        loop["watchdog_log"] = str(root / "watchdog.log")
    for name in (
        "controller_launcher",
        "controller_launcher_pid_file",
        "controller_launcher_log",
        "controller_hold_file",
        "controller_release_file",
        "controller_worker_lease_file",
        "experience_path",
        "watchdog_stop_file",
        "watchdog_lock_file",
        "watchdog_child_pid_file",
        "watchdog_log",
        "controller_release_evidence_path",
    ):
        if loop[name] and not Path(str(loop[name])).is_absolute():
            raise ValueError("%s must be absolute" % name)
    if int(loop["max_rounds"]) <= 0:
        raise ValueError("max_rounds must be positive")
    if int(loop["max_cases"]) <= 0 or int(loop["seed_count"]) <= 0:
        raise ValueError("max_cases and seed_count must be positive")
    if int(loop["max_train_steps"]) <= 0:
        raise ValueError("max_train_steps must be positive")
    if not 0 < float(loop["semantic_floor_ratio"]) <= 1:
        raise ValueError("semantic_floor_ratio must be in (0, 1]")
    if not 0 < float(loop["quality_floor_ratio"]) <= 1:
        raise ValueError("quality_floor_ratio must be in (0, 1]")
    if int(loop["proposal_schema_version"]) != 2:
        raise ValueError("proposal_schema_version must be 2")
    if bool(loop["offline_simulation"]):
        raise ValueError("Wan vLLM loop requires offline_simulation=false")
    if str(loop["continuation_contract"]) not in {"30s", "5s"}:
        raise ValueError("continuation_contract must be 30s or 5s")
    if str(loop["training_mode"]) not in {"legacy", "native_rollout_state"}:
        raise ValueError("training_mode must be legacy or native_rollout_state")
    if int(loop["teacher_forced_intervals"]) < 0:
        raise ValueError("teacher_forced_intervals must be non-negative")
    if loop["trajectory_record"] and not Path(str(loop["trajectory_record"])).is_absolute():
        raise ValueError("trajectory_record must be absolute")
    if str(loop["training_mode"]) == "native_rollout_state":
        if str(loop["continuation_contract"]) != "5s":
            raise ValueError("native_rollout_state requires continuation_contract=5s")
        if int(loop["teacher_forced_intervals"]) != 16:
            raise ValueError("native_rollout_state requires teacher_forced_intervals=16")
        if not str(loop["trajectory_record"]).strip():
            raise ValueError("native_rollout_state requires trajectory_record")
        training_path = Path(str(loop["training_config"]))
        if training_path.is_file():
            training_raw = yaml.safe_load(training_path.read_text(encoding="utf-8"))
            training = training_raw.get("wan_training") if isinstance(training_raw, Mapping) else None
            if not isinstance(training, Mapping) or str(training.get("cache_contract", "")).strip() != "native_5s":
                raise ValueError("native_rollout_state requires training_config cache_contract=native_5s")
    capabilities = loop["algorithm_capabilities"]
    if not isinstance(capabilities, list) or not capabilities or any(str(item) not in {"progressive_distillation", "dmd2"} for item in capabilities):
        raise ValueError("algorithm_capabilities must list progressive_distillation and/or dmd2")
    if int(loop["checkpoint_every_steps"]) < 0:
        raise ValueError("checkpoint_every_steps must not be negative")
    if int(loop["watchdog_restart_limit"]) <= 0:
        raise ValueError("watchdog_restart_limit must be positive")
    if int(loop["watchdog_backoff_seconds"]) < 0:
        raise ValueError("watchdog_backoff_seconds must not be negative")
    if not str(loop["controller_primary_objective"]).strip():
        raise ValueError("controller_primary_objective must not be empty")
    secondary = loop["controller_secondary_objectives"]
    if not isinstance(secondary, list) or not secondary or any(not str(item).strip() for item in secondary):
        raise ValueError("controller_secondary_objectives must be a non-empty list")
    if float(loop["controller_min_quality_delta"]) < 0:
        raise ValueError("controller_min_quality_delta must be non-negative")
    if not 0 <= float(loop["controller_max_temporal_drop"]) <= 1:
        raise ValueError("controller_max_temporal_drop must be in [0, 1]")
    if int(loop["controller_memory_limit"]) <= 0:
        raise ValueError("controller_memory_limit must be positive")
    if not isinstance(loop["controller_search_space"], Mapping):
        raise ValueError("controller_search_space must be a mapping")
    if bool(loop["controller_release_required"]):
        if not str(loop["controller_release_evidence_path"]).strip():
            raise ValueError("controller release evidence is required when release gate is enabled")
        if not str(loop["controller_release_evidence_sha256"]).strip():
            raise ValueError("controller release evidence digest is required when release gate is enabled")
        if str(loop["controller_release_contract"]).strip() != "adaptive_visual_retry":
            raise ValueError("controller release contract must be adaptive_visual_retry")
    offsets = loop["adaptive_retry_offsets"]
    if not isinstance(offsets, list) or not offsets or any(isinstance(item, bool) or int(item) <= 0 for item in offsets):
        raise ValueError("adaptive_retry_offsets must contain positive integers")
    evaluation_devices = loop["evaluation_parallel_devices"]
    if not isinstance(evaluation_devices, list) or any(
        not str(item).strip() or not str(item).strip().startswith("cuda:") for item in evaluation_devices
    ):
        raise ValueError("evaluation_parallel_devices must contain explicit cuda:N device names")
    if not isinstance(loop["memory_seed_paths"], list):
        raise ValueError("memory_seed_paths must be a list")
    for seed_path in loop["memory_seed_paths"]:
        if not Path(str(seed_path)).is_absolute():
            raise ValueError("memory_seed_paths must contain absolute paths")
    if not isinstance(loop["fidelity_schedule"], list) or not loop["fidelity_schedule"]:
        raise ValueError("fidelity_schedule must be a non-empty list")
    if any(str(item) not in {"F1", "F2", "F3"} for item in loop["fidelity_schedule"]):
        raise ValueError("fidelity_schedule contains unsupported fidelity")
    visual_gate = loop["visual_quality_gate"]
    if not isinstance(visual_gate, Mapping):
        raise ValueError("visual_quality_gate must be a mapping")
    visual_gate = dict(visual_gate)
    for name, default in (
        ("enabled", False),
        ("version", "visual-structure-v1"),
        ("min_spatial_gradient", 0.30),
        ("min_laplacian_std", 1.8),
        ("min_block_variance", 12.0),
        ("min_frame_difference", 0.01),
        ("max_low_texture_frame_ratio", 0.25),
    ):
        visual_gate.setdefault(name, default)
    if not str(visual_gate["version"]).strip():
        raise ValueError("visual_quality_gate.version must not be empty")
    if bool(visual_gate["enabled"]):
        for name in ("min_spatial_gradient", "min_laplacian_std", "min_block_variance", "min_frame_difference"):
            value = float(visual_gate[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("visual_quality_gate.%s must be positive" % name)
        low_texture_ratio = float(visual_gate["max_low_texture_frame_ratio"])
        if not math.isfinite(low_texture_ratio) or not 0 <= low_texture_ratio < 1:
            raise ValueError("visual_quality_gate.max_low_texture_frame_ratio must be in [0, 1)")
    loop["visual_quality_gate"] = visual_gate
    validate_correctness_gate(loop)
    validate_controller_release_gate(loop)
    return loop


def validate_correctness_gate(config: Mapping[str, Any]) -> None:
    """Validate the immutable P0 evidence binding before campaign startup."""

    evidence_digests = config.get("legacy_r2_checkpoint_digests", [])
    if not isinstance(evidence_digests, list) or any(
        not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in evidence_digests
    ):
        raise ValueError("legacy_r2_checkpoint_digests must contain lowercase SHA-256 digests")
    if not bool(config.get("correctness_audit_passed", False)):
        return
    evidence_path = Path(str(config.get("correctness_evidence_path", ""))).resolve()
    expected_digest = str(config.get("correctness_evidence_sha256", "")).strip()
    if not evidence_path.is_file() or not expected_digest:
        raise ValueError("correctness evidence path and digest are required when audit is passed")
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError("correctness evidence digest must be a lowercase SHA-256")
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("correctness evidence is not readable JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "passed":
        raise ValueError("correctness evidence must be a JSON object with status=passed")
    if sha256_file(evidence_path) != expected_digest:
        raise ValueError("correctness evidence SHA-256 does not match configured digest")


def validate_controller_release_gate(config: Mapping[str, Any]) -> None:
    """Require immutable P0 + adaptive evidence before recipe search starts."""

    if not bool(config.get("controller_release_required", False)):
        return
    evidence_path = Path(str(config.get("controller_release_evidence_path", ""))).resolve()
    expected_digest = str(config.get("controller_release_evidence_sha256", "")).strip()
    if not evidence_path.is_file() or not expected_digest:
        raise ValueError("controller release evidence path and digest are required")
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError("controller release evidence digest must be a lowercase SHA-256")
    if sha256_file(evidence_path) != expected_digest:
        raise ValueError("controller release evidence SHA-256 does not match configured digest")
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("controller release evidence is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("controller release evidence must be a JSON object")
    if payload.get("status") != "passed" or payload.get("controller_search_authorized") is not True:
        raise ValueError("controller release evidence does not authorize recipe search")
    if payload.get("release_contract") != "adaptive_visual_retry":
        raise ValueError("controller release evidence contract mismatch")
    if payload.get("legacy_r2_promotion_frozen") is not True:
        raise ValueError("Controller release must keep legacy R2 promotion frozen")
    if payload.get("fixed_seed_promotion_authorized") is not False:
        raise ValueError("Controller release must keep fixed-seed promotion distinct")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks or any(not isinstance(item, Mapping) or item.get("passed") is not True for item in checks):
        raise ValueError("controller release evidence contains a failed or malformed check")


def correctness_gate_decision(
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the P0 gate result that must precede all quality decisions."""

    validate_correctness_gate(config)
    if not bool(config.get("correctness_audit_passed", False)):
        return {
            "passed": False,
            "legacy_r2_promotion_frozen": bool(config.get("legacy_r2_promotion_frozen", True)),
        }
    parent_digest = str(
        parent.get("student_checkpoint_sha256")
        or parent.get("checkpoint_sha256")
        or parent.get("parent_checkpoint_sha256")
        or ""
    )
    legacy_digests = {str(item) for item in config.get("legacy_r2_checkpoint_digests", [])}
    frozen = bool(config.get("legacy_r2_promotion_frozen", True)) and parent_digest in legacy_digests
    return {
        "passed": True,
        "legacy_r2_promotion_frozen": bool(config.get("legacy_r2_promotion_frozen", True)),
        "legacy_r2_frozen": frozen,
        "correctness_evidence_path": str(config.get("correctness_evidence_path", "")),
        "correctness_evidence_sha256": str(config.get("correctness_evidence_sha256", "")),
        "parent_checkpoint_sha256": parent_digest,
    }


def _controller_handoff(config: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        str(config.get("controller_hold_file") or "") or None,
        str(config.get("controller_release_file") or "") or None,
        str(config.get("controller_worker_lease_file") or "") or None,
    )


def _shutdown_controller_on_exit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Stop an owned persistent launcher when a one-shot campaign is done.

    The launcher is intentionally reusable during a multi-round campaign, but
    leaving it alive after a bounded job strands one L40 behind an idle vLLM
    reservation.  Only a PID whose command line still names the configured
    launcher is signalled; stale or reused PID files are ignored.
    """

    if not bool(config.get("controller_shutdown_on_exit", False)):
        return {"enabled": False, "requested": False, "terminated": False}
    pid_file = Path(str(config.get("controller_launcher_pid_file", "")))
    if not pid_file.is_file():
        return {"enabled": True, "requested": False, "terminated": True, "reason": "pid_file_missing"}
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, TypeError, ValueError):
        return {"enabled": True, "requested": False, "terminated": True, "reason": "invalid_pid_file"}
    try:
        cmdline = Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return {"enabled": True, "requested": False, "terminated": True, "reason": "launcher_not_alive", "pid": pid}
    launcher = Path(str(config.get("controller_launcher", ""))).name
    if not launcher or launcher not in cmdline or "controller-wait-launch" not in cmdline:
        return {"enabled": True, "requested": False, "terminated": False, "reason": "pid_not_owned", "pid": pid}
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return {"enabled": True, "requested": True, "terminated": True, "pid": pid, "reason": "already_exited"}
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return {"enabled": True, "requested": True, "terminated": True, "pid": pid}
        time.sleep(0.5)
    return {"enabled": True, "requested": True, "terminated": False, "pid": pid, "reason": "shutdown_timeout"}


def _controller_is_ready(config: Mapping[str, Any]) -> bool:
    endpoint = str(config["vllm_base_url"]).rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(endpoint, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return isinstance(payload, Mapping)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _controller_launcher_exists(launcher: str) -> bool:
    try:
        probe = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    script_name = Path(launcher).name
    for line in probe.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or fields[0] == str(os.getpid()):
            continue
        if script_name in fields[1] and "controller-wait-launch" in fields[1]:
            return True
    return False


def ensure_controller(config: Mapping[str, Any]) -> None:
    """Start or wait for the server-local controller without stealing worker GPUs."""

    if _controller_is_ready(config):
        return
    launcher = str(config.get("controller_launcher") or "")
    if not launcher:
        raise RuntimeError("vLLM endpoint is unavailable and no controller_launcher is configured")
    pid_file = Path(str(config["controller_launcher_pid_file"]))
    live = False
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            cmdline = Path("/proc/%d/cmdline" % pid).read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
            launcher_name = Path(launcher).name
            live = bool(cmdline.strip()) and launcher_name in cmdline and "controller-wait-launch" in cmdline
        except (OSError, TypeError, ValueError):
            live = False
    if not live:
        live = _controller_launcher_exists(launcher)
    if not live:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        log_path = Path(str(config["controller_launcher_log"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "CONTROLLER_HANDOFF_HOLD_FILE": str(config["controller_hold_file"]),
                "CONTROLLER_RELEASE_FILE": str(config["controller_release_file"]),
                "CONTROLLER_WORKER_LEASE_FILE": str(config["controller_worker_lease_file"]),
                "CONTROLLER_GPU_LEASE_FILE": str(Path(str(config["output_root"])) / ".controller-gpu-lease.json"),
                "CONTROLLER_PID_FILE": str(Path(str(config["output_root"])) / ".controller-vllm.pid"),
                "CONTROLLER_EXTERNAL_PORTS": "8001",
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                ["bash", launcher],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        pid_file.write_text(str(process.pid) + "\n", encoding="utf-8")
    deadline = time.monotonic() + float(config["vllm_timeout_s"])
    while time.monotonic() < deadline:
        if _controller_is_ready(config):
            return
        time.sleep(5.0)
    raise RuntimeError("vLLM controller did not become ready within the configured timeout")


def wait_for_gpu_quiet(config: Mapping[str, Any]) -> None:
    """Wait until all configured Wan role cards have enough free memory.

    Some remote hosts expose a working CUDA runtime while the ``nvidia-smi``
    client has a minor driver-library mismatch.  Use PyTorch's CUDA allocator
    query as a read-only fallback so the resource gate does not wait forever
    when the worker itself can still initialize CUDA.
    """

    deadline = time.monotonic() + float(config["gpu_wait_s"])
    minimum_mib = float(config["min_free_memory_gb"]) * 1024.0
    needed = int(config["gpu_count"])
    while time.monotonic() < deadline:
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            free = []
            for line in probe.stdout.splitlines():
                try:
                    free.append(float(line.strip()))
                except ValueError:
                    continue
            if len(free) < needed:
                try:
                    import torch

                    if torch.cuda.is_available():
                        free = [float(torch.cuda.mem_get_info(index)[0]) / (1024.0 * 1024.0) for index in range(needed)]
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
            if len(free) >= needed and sum(value >= minimum_mib for value in free) >= needed:
                return
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        time.sleep(15.0)
    raise RuntimeError("Wan GPU resource gate timed out after %ss" % int(config["gpu_wait_s"]))


def build_training_command(
    config: Mapping[str, Any],
    proposal: WanTrainingProposal,
    *,
    output_dir: Path,
    result_path: Path,
    parent_checkpoint: Path,
) -> tuple[str, ...]:
    command = [
        str(config["worker_python"]),
        str(config["training_entrypoint"]),
        "--config",
        str(config["training_config"]),
        "--output",
        str(output_dir),
        "--result",
        str(result_path),
        "--steps",
        str(proposal.steps),
        "--teacher-reuse-steps",
        str(proposal.teacher_reuse_steps),
        "--learning-rate",
        str(proposal.learning_rate),
        "--method",
        str(proposal.training.method),
        "--source-steps",
        str(proposal.training.source_steps),
        "--target-steps",
        str(proposal.training.target_steps),
        "--rollout-steps",
        str(proposal.training.rollout_steps),
        "--rollout-every",
        str(proposal.training.rollout_every),
        "--rollout-endpoint-weight",
        str(proposal.training.rollout_endpoint_weight),
        "--rollout-clean-weight",
        str(proposal.training.rollout_clean_weight),
        "--endpoint-weight",
        str(proposal.training.endpoint_weight),
        "--velocity-weight",
        str(proposal.training.velocity_weight),
        "--clean-anchor-weight",
        str(proposal.training.clean_anchor_weight),
        "--trainable-scope",
        str(proposal.training.trainable_scope),
        "--fp32-optimizer-state" if proposal.training.fp32_optimizer_state else "--no-fp32-optimizer-state",
        "--initialization-mode",
        str(proposal.initialization.mode),
        "--parent-checkpoint",
        str(parent_checkpoint),
        "--training-checkpoint",
        str(output_dir / "training-checkpoint.pt"),
        "--checkpoint-every-steps",
        str(int(config.get("checkpoint_every_steps", 8))),
        "--dmd2-memory-admitted" if bool(config.get("dmd2_memory_admitted", False)) else "--no-dmd2-memory-admitted",
        "--wan-dmd2-adapter-ready" if bool(config.get("wan_dmd2_adapter_ready", False)) else "--no-wan-dmd2-adapter-ready",
    ]
    if str(config.get("training_mode", "legacy")) == "native_rollout_state":
        command.extend(
            [
                "--mode",
                str(config["training_mode"]),
                "--teacher-forced-intervals",
                str(int(config["teacher_forced_intervals"])),
                "--trajectory-record",
                str(config["trajectory_record"]),
            ]
        )
    return tuple(command)


def _compact_controller_evaluation(value: Any) -> Mapping[str, Any]:
    """Keep only trusted aggregate evaluation fields in the LLM prompt.

    Full evaluation records contain one large entry per video (including paths
    and validity metadata).  They are useful on disk, but would crowd the
    Controller context and can exceed the vLLM context window after the first
    round.  The Controller only needs the aggregate gate metrics and digest.
    """

    if not isinstance(value, Mapping):
        return {}
    metrics = value.get("quality_metrics")
    compact: dict[str, Any] = {
        key: value.get(key)
        for key in (
            "status",
            "case_count",
            "valid",
            "promotable",
            "quality_score",
            "student_checkpoint_sha256",
            "evaluation_manifest_digest",
        )
        if key in value
    }
    if isinstance(metrics, Mapping):
        compact["quality_metrics"] = {
            key: metrics.get(key)
            for key in ("semantic", "temporal", "motion", "score_type", "visual_quality_gate")
            if key in metrics
        }
    if isinstance(value.get("visual_quality_gate"), Mapping):
        compact["visual_quality_gate"] = {
            key: value["visual_quality_gate"].get(key)
            for key in ("enabled", "passed", "failures", "version")
            if key in value["visual_quality_gate"]
        }
    contract = value.get("contract")
    if isinstance(contract, Mapping):
        compact["contract"] = {
            key: contract.get(key)
            for key in ("decoded_frames_min", "decoded_frames_max", "fps", "width", "height")
            if key in contract
        }
    return compact


def _compact_controller_training(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    loss = value.get("loss") if isinstance(value.get("loss"), Mapping) else {}
    return {
        key: value.get(key)
        for key in (
            "status",
            "optimizer_steps",
            "teacher_forward_count",
            "teacher_reuse_steps",
            "endpoint_weight",
            "velocity_weight",
            "clean_anchor_weight",
            "teacher_target_groups",
            "student_checkpoint_sha256",
            "parent_checkpoint_sha256",
            "offline_simulation",
            "algorithm_name",
            "trainable_scope",
        )
        if key in value
    } | ({"loss": {key: loss.get(key) for key in ("initial", "total_final") if key in loss}} if loss else {})


def _compact_controller_round(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    proposal = value.get("proposal")
    compact: dict[str, Any] = {
        key: value.get(key)
        for key in ("round", "fingerprint")
        if key in value
    }
    if isinstance(proposal, Mapping):
        compact["proposal"] = proposal
    if isinstance(value.get("training"), Mapping):
        compact["training"] = _compact_controller_training(value["training"])
    if isinstance(value.get("evaluation"), Mapping):
        compact["evaluation"] = _compact_controller_evaluation(value["evaluation"])
    if isinstance(value.get("decision"), Mapping):
        compact["decision"] = value["decision"]
    return compact


def _compact_controller_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build a bounded prompt view while leaving full artifacts on disk."""

    compact = dict(context)
    compact["parent_evaluation"] = _compact_controller_evaluation(context.get("parent_evaluation"))
    prior_rounds = context.get("prior_rounds")
    if isinstance(prior_rounds, list):
        compact["prior_rounds"] = [_compact_controller_round(item) for item in prior_rounds[-3:]]
    lessons = context.get("memory_lessons")
    if isinstance(lessons, list):
        compact["memory_lessons"] = lessons[-3:]
    return compact


def _request_vllm(config: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = proposal_json_schema()
    system = (
        "You are the Wan Student training controller. Return JSON only. "
        "Choose exactly one bounded training policy from the supplied capability snapshot, including method, "
        "rollout, initialization, precision, curriculum, data, continuation, focus, and hypothesis. "
        "Never output code, shell commands, paths, device IDs, architecture, or checkpoint content. "
        "The trusted runner will bind the parent digest and all execution paths. "
        "When the campaign is native 5s, search only the supplied native 32-to-16 rollout-state recipe fields; "
        "never propose a different cache, trajectory, duration, teacher, architecture, or algorithm."
    )
    contract = str(config.get("continuation_contract", "30s"))
    if contract == "5s":
        contract_text = "the Wan-native 81-frame, 16 FPS, 256x256, approximately 5.06-second contract"
    else:
        contract_text = "the 30-second overlapping-latent contract"
    smoke_budget = int(config.get("max_train_steps", 100000))
    source_steps = config.get("progressive_source_steps")
    target_steps = config.get("progressive_target_steps")
    trainable_scope = config.get("wan_trainable_scope")
    search_space = dict(config.get("controller_search_space") or {})
    step_text = (
        " For progressive_distillation use source_steps=%d and target_steps=%d exactly; train_steps must be <=%d."
        % (int(source_steps), int(target_steps), smoke_budget)
        if source_steps is not None and target_steps is not None
        else ""
    )
    scope_text = (
        " Use trainable_scope=%s exactly; this is a trusted Wan smoke constraint."
        % str(trainable_scope)
        if trainable_scope
        else ""
    )
    objective_text = (
        " Optimize %s first, then protect %s. Choose only values from SEARCH_SPACE and do not repeat a rejected recipe unless the hypothesis explicitly explains the change."
        % (
            str(config.get("controller_primary_objective", "semantic_alignment")),
            ", ".join(str(item) for item in config.get("controller_secondary_objectives", [])),
        )
    )
    controller_context = _compact_controller_context(context)
    user = (
        "Propose exactly one next Wan training policy. Preserve the real H3 Teacher, "
        "offline_simulation=false, %s, and the 1-2B Wan Student. " % contract_text
        +
        "Prefer semantic alignment when semantic is below the Teacher floor; otherwise protect continuity. "
        + (
            "This is a locked native rollout-state campaign: keep the fixed real 5s H3 cache, one immutable 32-step "
            "trajectory, target 16 steps, and the visual veto. Only change values explicitly listed in SEARCH_SPACE. "
            if contract == "5s"
            else ""
        )
        + step_text
        + scope_text
        + objective_text
        +
        "Return fields exactly matching this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
        + "\nSEARCH_SPACE=\n"
        + json.dumps(search_space, ensure_ascii=False, sort_keys=True)
        + "\nCONTEXT=\n"
        + json.dumps(dict(controller_context), ensure_ascii=False, sort_keys=True)
    )
    payload = {
        "model": str(config["vllm_model"]),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    base = str(config["vllm_base_url"]).rstrip("/")
    endpoint = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    deadline = time.monotonic() + float(config["vllm_timeout_s"])
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=min(180.0, max(1.0, deadline - time.monotonic()))) as response:
                raw = json.loads(response.read().decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise WanProposalError("vLLM response must be an object")
            return extract_json_content(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise WanProposalError("vLLM returned HTTP %s: %s" % (exc.code, detail)) from exc
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, WanProposalError) as exc:
            last_error = exc
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    raise WanProposalError("vLLM proposal request timed out: %s" % last_error)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON artifact must be an object: %s" % path)
    return dict(value)


def _write_failure(path: Path, code: str, message: str, **extra: Any) -> dict[str, Any]:
    value = {"status": "failed", "valid": False, "promotable": False, "failure_code": code, "message": message, **extra}
    atomic_write_json(path, value)
    return value


def _normalize_controller_recipe(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply only an explicit, audited compatibility repair to Controller JSON.

    The real Controller has been observed to abbreviate the required
    ``hypothesis`` key as ``hypo``.  This is a recipe-field spelling repair,
    not a policy or code mutation; every other unknown field remains fail
    closed in ``WanTrainingProposal.from_mapping``.
    """

    normalized = dict(raw)
    repair: dict[str, Any] = {"applied": False}
    if "hypothesis" not in normalized and "hypo" in normalized:
        value = normalized.pop("hypo")
        if isinstance(value, str) and value.strip():
            normalized["hypothesis"] = value
            repair = {"applied": True, "from": "hypo", "to": "hypothesis", "reason": "controller_schema_spelling"}
    return normalized, repair


def _validate_recipe_search_space(proposal: WanTrainingProposal, config: Mapping[str, Any]) -> None:
    """Fail closed when the Controller leaves the trusted recipe search space."""

    search_space = config.get("controller_search_space") or {}
    if not isinstance(search_space, Mapping):
        raise WanProposalError("controller_search_space must be a mapping")
    values = {
        "learning_rate": proposal.training.learning_rate,
        "critic_learning_rate": proposal.training.critic_learning_rate,
        "rollout_endpoint_weight": proposal.training.rollout_endpoint_weight,
        "rollout_clean_weight": proposal.training.rollout_clean_weight,
        "rollout_every": proposal.training.rollout_every,
        "rollout_steps": proposal.training.rollout_steps,
        "train_steps": proposal.training.train_steps,
        "teacher_reuse_steps": proposal.teacher_reuse_steps,
        "endpoint_weight": proposal.training.endpoint_weight,
        "velocity_weight": proposal.training.velocity_weight,
        "clean_anchor_weight": proposal.training.clean_anchor_weight,
    }
    for key, allowed in search_space.items():
        if key not in values:
            raise WanProposalError("controller_search_space contains unsupported field: %s" % key)
        if not isinstance(allowed, list) or not allowed:
            raise WanProposalError("controller_search_space.%s must be a non-empty list" % key)
        value = values[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            matches = any(
                isinstance(candidate, (int, float))
                and not isinstance(candidate, bool)
                and math.isclose(float(value), float(candidate), rel_tol=1e-9, abs_tol=1e-12)
                for candidate in allowed
            )
        else:
            matches = value in allowed
        if not matches:
            raise WanProposalError("%s=%s is outside the trusted Controller search space" % (key, value))


def _apply_controller_promotion_gate(
    parent_eval: Mapping[str, Any],
    candidate_eval: Mapping[str, Any],
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the paper campaign's trusted quality/continuity gate."""

    result = dict(decision)
    if not bool(result.get("accepted")):
        return result
    parent_metrics = parent_eval.get("quality_metrics") if isinstance(parent_eval.get("quality_metrics"), Mapping) else {}
    candidate_metrics = candidate_eval.get("quality_metrics") if isinstance(candidate_eval.get("quality_metrics"), Mapping) else {}
    parent_quality = float(parent_eval.get("quality_score", 0.0))
    candidate_quality = float(candidate_eval.get("quality_score", 0.0))
    parent_temporal = float(parent_metrics.get("temporal", 0.0))
    candidate_temporal = float(candidate_metrics.get("temporal", 0.0))
    quality_delta = candidate_quality - parent_quality
    temporal_delta = candidate_temporal - parent_temporal
    result["quality_delta"] = quality_delta
    result["temporal_delta"] = temporal_delta
    if quality_delta < float(config.get("controller_min_quality_delta", 0.0)):
        result.update({"accepted": False, "reason": "quality_no_improvement"})
    elif temporal_delta < -float(config.get("controller_max_temporal_drop", 0.005)):
        result.update({"accepted": False, "reason": "temporal_regression"})
    required_contract = str(config.get("controller_release_contract", "")).strip()
    if required_contract and candidate_eval.get("contract_mode") != required_contract:
        result.update({"accepted": False, "reason": "release_contract_mismatch"})
    if required_contract and candidate_eval.get("adaptive_retry_verified") is not True:
        result.update({"accepted": False, "reason": "adaptive_retry_not_verified"})
    return result


def _run_adaptive_visual_retry(
    config: Mapping[str, Any],
    strict_result_path: Path,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the explicit retry contract after strict evaluation completes."""

    try:
        from wan_visual_retry import run_config as run_visual_retry_config
    except ModuleNotFoundError:
        from tools.wan_visual_retry import run_config as run_visual_retry_config
    retry_config = {
        "evaluation_manifest": str(config["evaluation_manifest"]),
        "wan_config": str(config["wan_config"]),
        "clip_model_path": str(config["clip_model_path"]),
        "max_cases": int(config["max_cases"]),
        "seed_count": int(config["seed_count"]),
        "device": str(config.get("device", "cuda:0")),
        "visual_quality_gate": dict(config.get("visual_quality_gate") or {}),
    }
    offsets = tuple(int(value) for value in config.get("adaptive_retry_offsets", [1, 2]))
    return run_visual_retry_config(
        retry_config,
        strict_result_path,
        checkpoint,
        output_dir / "adaptive",
        offsets,
    )


def _evaluate_checkpoint_strict_single(config: Mapping[str, Any], checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Generate and evaluate all configured prompts/seeds in one Wan process."""

    from harness4h3.student.evaluation_manifest import EvaluationManifest
    from harness4h3.student.quality import ClipTemporalQualityBackend
    from harness4h3.student.wan import generate_wan_clip, load_pretrained_wan, load_wan_spec, write_wan_video
    from harness4h3.student.video_contract import VideoContractError, VideoOutputSpec, validate_video_output
    from harness4h3.student.wan_long_video import (
        WanLongVideoError,
        WanLongVideoSpec,
        generate_wan_long_video,
        load_wan_transformer_checkpoint,
        validate_wan_long_video,
    )
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "evaluation-result.json"
    pipeline = None
    try:
        manifest = EvaluationManifest.from_path(Path(str(config["evaluation_manifest"])))
        spec = load_wan_spec(Path(str(config["wan_config"])))
        short_contract = str(config.get("continuation_contract", "30s")) == "5s"
        if short_contract:
            contract = VideoOutputSpec(
                # Wan's 81 decoded frames correspond to 24 H3 latent frames
                # under the authenticated causal VAE geometry.
                latent_frames=24,
                decoded_frames_min=spec.num_frames,
                decoded_frames_max=spec.num_frames,
                fps=spec.native_fps,
                width=spec.width,
                height=spec.height,
                duration_min_s=5.0,
                duration_max_s=5.1,
                fps_tolerance=0.1,
                window_count=1,
                overlap_latent_frames=0,
            )
        else:
            contract = WanLongVideoSpec(
                window_frames=spec.num_frames,
                overlap_latent_frames=int(config["overlap_latent_frames"]),
                window_count=int(config["window_count"]),
                native_fps=spec.native_fps,
                output_fps=spec.output_fps,
                width=spec.width,
                height=spec.height,
                inference_steps=int(config["inference_steps"]),
                guidance_scale=spec.guidance_scale,
            )
        pipeline, device = load_pretrained_wan(spec, device=str(config.get("device", "cuda:3")), local_files_only=True)
        checkpoint_digest = load_wan_transformer_checkpoint(pipeline, checkpoint)
        quality_backend = ClipTemporalQualityBackend(
            str(config["clip_model_path"]),
            device="cpu",
            sample_frames=12,
            drift_penalty=0.25,
            visual_quality_gate=config.get("visual_quality_gate"),
        )
        cases: list[dict[str, Any]] = []
        for case in manifest.cases[: int(config["max_cases"])]:
            for seed in case.seeds[: int(config["seed_count"])]:
                case_dir = output_dir / "videos" / case.case_id / ("seed_%d" % int(seed))
                video_path = case_dir / ("wan-5s.mp4" if short_contract else "wan-long.mp4")
                case_started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    # CUDA kernels are asynchronous.  Synchronize the
                    # boundary so this hardware metric measures actual Wan
                    # inference rather than Python launch time.
                    torch.cuda.synchronize(device)
                inference_started = time.perf_counter()
                if short_contract:
                    frames, generation = generate_wan_clip(
                        pipeline,
                        spec,
                        case.caption,
                        seed=int(seed),
                        device=device,
                    )
                    generation = {**dict(generation), "continuation_verified": True, "offline_simulation": False}
                else:
                    frames, generation = generate_wan_long_video(
                        pipeline,
                        spec,
                        contract,
                        case.caption,
                        seed=int(seed),
                        device=device,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_wall_time = time.perf_counter() - inference_started
                io_started = time.perf_counter()
                if short_contract:
                    write_wan_video(frames, video_path, spec.native_fps)
                    validity = validate_video_output(video_path, contract)
                else:
                    write_wan_video(frames, video_path, contract.output_fps)
                    validity = validate_wan_long_video(video_path, contract)
                io_wall_time = time.perf_counter() - io_started
                quality_started = time.perf_counter()
                evidence = quality_backend.evaluate(video_path, case.caption)
                quality_wall_time = time.perf_counter() - quality_started
                case_wall_time = time.perf_counter() - case_started
                peak_memory_bytes = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                )
                generation = {
                    **dict(generation),
                    "peak_memory_bytes": peak_memory_bytes,
                    "peak_memory_gb": float(peak_memory_bytes / (1024**3)),
                }
                cases.append(
                    {
                        "case_id": case.case_id,
                        "seed": int(seed),
                        "caption": case.caption,
                        "video_path": str(video_path),
                        "generation_wall_time_s": float(inference_wall_time),
                        "inference_wall_time_s": float(inference_wall_time),
                        "video_io_wall_time_s": float(io_wall_time),
                        "quality_eval_wall_time_s": float(quality_wall_time),
                        "case_wall_time_s": float(case_wall_time),
                        "generation": dict(generation),
                        "validity": dict(validity),
                        "quality": {
                            "score": evidence.aggregate,
                            "semantic": evidence.semantic,
                            "semantic_mean": evidence.semantic_mean,
                            "semantic_p10": evidence.semantic_p10,
                            "temporal": evidence.temporal,
                            "motion": evidence.motion,
                            "visual_metrics": dict(evidence.visual_metrics),
                            "visual_quality_gate": dict(evidence.visual_quality_gate),
                            "score_type": evidence.backend,
                        },
                    }
                )
        if not cases:
            raise ValueError("evaluation manifest selected no cases")
        scores = [float(item["quality"]["score"]) for item in cases]
        semantics = [float(item["quality"]["semantic"]) for item in cases]
        visual_gate_results = [dict(item["quality"].get("visual_quality_gate") or {}) for item in cases]
        visual_gate_passed = all(
            not bool(item.get("enabled", False)) or bool(item.get("passed", False))
            for item in visual_gate_results
        )
        visual_failures = [
            {
                "case_id": item["case_id"],
                "seed": item["seed"],
                "failures": list(item["quality"].get("visual_quality_gate", {}).get("failures", [])),
            }
            for item in cases
            if bool(item["quality"].get("visual_quality_gate", {}).get("enabled", False))
            and not bool(item["quality"].get("visual_quality_gate", {}).get("passed", False))
        ]
        result = {
            "status": "success",
            "valid": visual_gate_passed,
            "promotable": visual_gate_passed,
            "offline_simulation": False,
            "continuation_verified": all(bool(item["generation"].get("continuation_verified")) for item in cases),
            "student_checkpoint": str(checkpoint.resolve()),
            "student_checkpoint_sha256": checkpoint_digest,
            "evaluation_manifest_digest": manifest.digest,
            "contract": contract.to_dict(),
            "case_count": len(cases),
            "quality_score": sum(scores) / len(scores),
            "quality_metrics": {
                "semantic": sum(semantics) / len(semantics),
                "temporal": sum(float(item["quality"]["temporal"]) for item in cases) / len(cases),
                "motion": sum(float(item["quality"]["motion"]) for item in cases) / len(cases),
                "visual_quality_gate": {
                    "enabled": any(bool(item.get("enabled", False)) for item in visual_gate_results),
                    "passed": visual_gate_passed,
                    "failure_count": len(visual_failures),
                    "version": next(
                        (str(item.get("version")) for item in visual_gate_results if item.get("version")),
                        "disabled",
                    ),
                },
                "score_type": "clip_temporal",
            },
            "hardware_metrics": {
                "generation_wall_time_s_total": sum(float(item["generation_wall_time_s"]) for item in cases),
                "generation_wall_time_s_mean": sum(float(item["generation_wall_time_s"]) for item in cases) / len(cases),
                "inference_wall_time_s_total": sum(float(item["inference_wall_time_s"]) for item in cases),
                "inference_wall_time_s_mean": sum(float(item["inference_wall_time_s"]) for item in cases) / len(cases),
                "video_io_wall_time_s_total": sum(float(item["video_io_wall_time_s"]) for item in cases),
                "quality_eval_wall_time_s_total": sum(float(item["quality_eval_wall_time_s"]) for item in cases),
                "case_wall_time_s_total": sum(float(item["case_wall_time_s"]) for item in cases),
                "case_wall_time_s_mean": sum(float(item["case_wall_time_s"]) for item in cases) / len(cases),
                "peak_memory_bytes_max": max(int(item["generation"].get("peak_memory_bytes", 0)) for item in cases),
                "peak_memory_gb_max": max(float(item["generation"].get("peak_memory_gb", 0.0)) for item in cases),
                "inference_steps_total": sum(int(item["generation"].get("num_inference_steps", 0)) for item in cases),
                "transformer_nfe_proxy": sum(int(item["generation"].get("num_inference_steps", 0)) for item in cases),
            },
            "visual_quality_gate": {
                "enabled": any(bool(item.get("enabled", False)) for item in visual_gate_results),
                "passed": visual_gate_passed,
                "failure_count": len(visual_failures),
                "failures": visual_failures,
                "version": next(
                    (str(item.get("version")) for item in visual_gate_results if item.get("version")),
                    "disabled",
                ),
            },
            "cases": cases,
        }
        if not visual_gate_passed:
            result["failure_code"] = "visual_collapse"
            result["failure_message"] = "one or more evaluated videos failed the visual information-content gate"
        atomic_write_json(result_path, result)
        return result
    except (OSError, RuntimeError, TypeError, ValueError, VideoContractError, WanLongVideoError) as exc:
        return _write_failure(result_path, getattr(exc, "code", "wan_evaluation_failed"), str(exc))
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _merge_parallel_strict_results(
    config: Mapping[str, Any],
    checkpoint: Path,
    output_dir: Path,
    manifest_digest: str,
    results: list[Mapping[str, Any]],
    devices: list[str],
) -> dict[str, Any]:
    """Merge independent one-GPU strict evaluations without changing metrics."""

    cases: list[dict[str, Any]] = []
    for result in results:
        cases.extend(dict(item) for item in result.get("cases", []))
    if not cases:
        raise ValueError("parallel evaluation produced no cases")
    cases.sort(key=lambda item: (str(item.get("case_id", "")), int(item.get("seed", 0))))
    gate_results = [dict(item.get("quality", {}).get("visual_quality_gate") or {}) for item in cases]
    visual_gate_passed = all(
        not bool(item.get("enabled", False)) or bool(item.get("passed", False)) for item in gate_results
    )
    visual_failures = [
        {
            "case_id": item["case_id"],
            "seed": item["seed"],
            "failures": list(item.get("quality", {}).get("visual_quality_gate", {}).get("failures", [])),
        }
        for item in cases
        if bool(item.get("quality", {}).get("visual_quality_gate", {}).get("enabled", False))
        and not bool(item.get("quality", {}).get("visual_quality_gate", {}).get("passed", False))
    ]
    scores = [float(item["quality"]["score"]) for item in cases]
    semantics = [float(item["quality"]["semantic"]) for item in cases]
    hardware = [item.get("hardware_metrics") for item in results if isinstance(item.get("hardware_metrics"), Mapping)]
    first = results[0]
    total_wall = sum(float(item.get("generation_wall_time_s_total", 0.0)) for item in hardware)
    total_inference_wall = sum(float(item.get("inference_wall_time_s_total", 0.0)) for item in hardware)
    total_io_wall = sum(float(item.get("video_io_wall_time_s_total", 0.0)) for item in hardware)
    total_quality_wall = sum(float(item.get("quality_eval_wall_time_s_total", 0.0)) for item in hardware)
    total_case_wall = sum(float(item.get("case_wall_time_s_total", 0.0)) for item in hardware)
    peak_vram = max((float(item.get("peak_memory_gb_max", 0.0)) for item in hardware), default=0.0)
    total_nfe = sum(int(item.get("inference_steps_total", 0)) for item in hardware)
    result = {
        "status": "success",
        "valid": visual_gate_passed,
        "promotable": visual_gate_passed,
        "offline_simulation": False,
        "continuation_verified": all(bool(item.get("generation", {}).get("continuation_verified")) for item in cases),
        "student_checkpoint": str(checkpoint.resolve()),
        "student_checkpoint_sha256": first.get("student_checkpoint_sha256"),
        "evaluation_manifest_digest": manifest_digest,
        "contract": first.get("contract"),
        "case_count": len(cases),
        "quality_score": sum(scores) / len(scores),
        "quality_metrics": {
            "semantic": sum(semantics) / len(semantics),
            "temporal": sum(float(item["quality"]["temporal"]) for item in cases) / len(cases),
            "motion": sum(float(item["quality"]["motion"]) for item in cases) / len(cases),
            "visual_quality_gate": {
                "enabled": any(bool(item.get("enabled", False)) for item in gate_results),
                "passed": visual_gate_passed,
                "failure_count": len(visual_failures),
                "version": next((str(item.get("version")) for item in gate_results if item.get("version")), "disabled"),
            },
            "score_type": "clip_temporal",
        },
        "hardware_metrics": {
            "generation_wall_time_s_total": total_wall,
            "generation_wall_time_s_mean": total_wall / len(cases),
            "inference_wall_time_s_total": total_inference_wall,
            "inference_wall_time_s_mean": total_inference_wall / len(cases),
            "video_io_wall_time_s_total": total_io_wall,
            "quality_eval_wall_time_s_total": total_quality_wall,
            "case_wall_time_s_total": total_case_wall,
            "case_wall_time_s_mean": total_case_wall / len(cases),
            "peak_memory_gb_max": peak_vram,
            "inference_steps_total": total_nfe,
            "transformer_nfe_proxy": total_nfe,
        },
        "visual_quality_gate": {
            "enabled": any(bool(item.get("enabled", False)) for item in gate_results),
            "passed": visual_gate_passed,
            "failure_count": len(visual_failures),
            "failures": visual_failures,
            "version": next((str(item.get("version")) for item in gate_results if item.get("version")), "disabled"),
        },
        "parallel_evaluation": {"enabled": True, "devices": list(devices), "worker_count": len(results)},
        "cases": cases,
    }
    if not visual_gate_passed:
        result["failure_code"] = "visual_collapse"
        result["failure_message"] = "one or more evaluated videos failed the visual information-content gate"
    atomic_write_json(output_dir / "evaluation-result.json", result)
    return result


def _evaluate_checkpoint_parallel_strict(
    config: Mapping[str, Any], checkpoint: Path, output_dir: Path, devices: list[str]
) -> dict[str, Any]:
    """Evaluate prompt shards concurrently, one full Wan pipeline per GPU."""

    from harness4h3.student.evaluation_manifest import EvaluationManifest

    manifest = EvaluationManifest.from_path(Path(str(config["evaluation_manifest"])))
    selected_cases = list(manifest.cases[: int(config["max_cases"])])
    if len(devices) <= 1 or len(selected_cases) <= 1:
        return _evaluate_checkpoint_strict_single(config, checkpoint, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parallel_dir = output_dir / "parallel"
    parallel_dir.mkdir(parents=True, exist_ok=True)
    worker_config = dict(config)
    worker_config["evaluation_parallel_devices"] = []
    worker_config["adaptive_visual_retry_enabled"] = False
    worker_config["max_cases"] = 1
    worker_config["seed_count"] = int(config["seed_count"])
    config_path = parallel_dir / "campaign-worker.yaml"
    config_path.write_text(yaml.safe_dump({"wan_loop": worker_config}, sort_keys=False), encoding="utf-8")
    worker_script = Path(__file__).resolve().with_name("wan_parallel_evaluate_worker.py")
    processes: list[tuple[subprocess.Popen[str], Path, str]] = []
    for index, case in enumerate(selected_cases):
        shard_manifest = EvaluationManifest(
            cases=(case,),
            unique_cache_items=1,
            source_cache_dir=manifest.source_cache_dir,
            digest=EvaluationManifest._digest_for(
                manifest.version,
                manifest.source_cache_dir,
                1,
                (case,),
                cache_contract=manifest.cache_contract,
            ),
            version=manifest.version,
            cache_contract=manifest.cache_contract,
        )
        shard_manifest_path = parallel_dir / ("manifest-%02d.json" % index)
        shard_manifest.write(shard_manifest_path)
        shard_output = parallel_dir / ("worker-%02d" % index)
        log_path = parallel_dir / ("worker-%02d.log" % index)
        log = log_path.open("w", encoding="utf-8")
        command = [
            str(config["worker_python"]),
            str(worker_script),
            "--config", str(config_path),
            "--manifest", str(shard_manifest_path),
            "--checkpoint", str(checkpoint),
            "--output", str(shard_output),
            "--device", str(devices[index % len(devices)]),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        processes.append((process, log_path, str(devices[index % len(devices)])))
        log.close()
    results: list[Mapping[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (process, log_path, device) in enumerate(processes):
        return_code = process.wait(timeout=float(config["training_timeout_s"]))
        result_path = parallel_dir / ("worker-%02d" % index) / "evaluation-result.json"
        if return_code != 0 or not result_path.is_file():
            failures.append({"worker": index, "device": device, "return_code": return_code, "log": str(log_path)})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "success":
            failures.append({"worker": index, "device": device, "result": result, "log": str(log_path)})
            continue
        results.append(result)
    if failures:
        return _write_failure(
            output_dir / "evaluation-result.json",
            "parallel_evaluation_failed",
            json.dumps(failures, ensure_ascii=False, sort_keys=True),
        )
    return _merge_parallel_strict_results(config, checkpoint, output_dir, manifest.digest, results, devices)


def evaluate_checkpoint(config: Mapping[str, Any], checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Run strict evaluation, optionally using all configured GPUs, then retry failures."""

    devices = [str(item) for item in config.get("evaluation_parallel_devices", []) if str(item).strip()]
    if len(devices) > 1:
        strict = _evaluate_checkpoint_parallel_strict(config, checkpoint, output_dir, devices)
    else:
        strict = _evaluate_checkpoint_strict_single(config, checkpoint, output_dir)
    if strict.get("status") != "success" or not bool(config.get("adaptive_visual_retry_enabled", False)):
        return strict
    adaptive = _run_adaptive_visual_retry(config, output_dir / "evaluation-result.json", checkpoint, output_dir)
    adaptive["strict_evaluation_result"] = str((output_dir / "evaluation-result.json").resolve())
    adaptive["strict_evaluation"] = strict
    return adaptive


def run_training(config: Mapping[str, Any], proposal: WanTrainingProposal, parent: Path, round_dir: Path) -> dict[str, Any]:
    train_dir = round_dir / "training"
    result_path = train_dir / "training-result.json"
    train_dir.mkdir(parents=True, exist_ok=True)
    command = build_training_command(config, proposal, output_dir=train_dir, result_path=result_path, parent_checkpoint=parent)
    log_path = train_dir / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=float(config["training_timeout_s"]),
        )
    if not result_path.is_file():
        return _write_failure(result_path, "training_result_missing", "Wan training worker exited without a result", exit_code=completed.returncode)
    result = _read_json(result_path)
    if completed.returncode != 0 or result.get("status") != "success":
        return result
    child = Path(str(result.get("student_checkpoint", "")))
    if not child.is_file():
        return _write_failure(result_path, "checkpoint_missing", "training result points to no checkpoint")
    child_digest = sha256_file(child)
    if child_digest != result.get("student_checkpoint_sha256"):
        return _write_failure(result_path, "checkpoint_digest_mismatch", "child checkpoint digest does not match training result")
    if result.get("parent_checkpoint_sha256") != sha256_file(parent):
        return _write_failure(result_path, "parent_digest_mismatch", "training result is not bound to the trusted parent")
    if result.get("offline_simulation") is not False:
        return _write_failure(result_path, "offline_simulation", "real Wan training must report offline_simulation=false")
    checkpoint_evidence = Path(str(result.get("training_checkpoint_evidence", "")))
    if not checkpoint_evidence.is_file():
        return _write_failure(result_path, "training_checkpoint_missing", "training result points to no step checkpoint")
    if sha256_file(checkpoint_evidence) != result.get("training_checkpoint_sha256"):
        return _write_failure(result_path, "training_checkpoint_digest_mismatch", "step checkpoint digest does not match training result")
    if not str(result.get("resume_config_digest", "")):
        return _write_failure(result_path, "training_checkpoint_config_missing", "training result lacks resume config digest")
    return result


def _teacher_semantic(config: Mapping[str, Any]) -> float:
    baseline = _read_json(Path(str(config["teacher_baseline_result"])))
    metrics = baseline.get("optimization_metrics")
    if baseline.get("status") != "success" or not isinstance(metrics, Mapping) or metrics.get("semantic") is None:
        raise ValueError("Teacher baseline lacks semantic evidence")
    return float(metrics["semantic"])


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _wan_experiment_fingerprint(
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    proposal: WanTrainingProposal,
    *,
    parent_digest: str,
    capability_digest: str,
    evaluator_digest: str,
) -> str:
    """Build the canonical identity used by the shared experiment memory."""

    return experiment_fingerprint(
        parent_model_id=parent_digest,
        parent_system_id=str(identity["base_digest"]),
        operator=proposal.training.method,
        operator_args=proposal.to_dict(),
        target_id=("wan-5s-native-clip-v1" if str(config.get("continuation_contract", "30s")) == "5s" else "wan-30s-overlap-window-v1"),
        device_id=str(config.get("device")),
        benchmark_recipe={
            "manifest_digest": str(identity["path_digests"]["evaluation_manifest_digest"]),
            "capability_digest": capability_digest,
            "evaluator_digest": evaluator_digest,
            "continuation_contract": str(config.get("continuation_contract", "30s")),
        },
    )


def _append_experience(
    config: Mapping[str, Any],
    round_dir: Path,
    round_index: int,
    fingerprint: str,
    proposal: WanTrainingProposal,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
    decision: Mapping[str, Any],
    parent_digest: str,
    identity: Mapping[str, Any],
    capability_digest: str,
    evaluator_digest: str,
) -> bool:
    """Persist one completed campaign round in the existing append-only memory."""

    source = round_dir / "decision.json"
    if not source.is_file():
        raise ValueError("cannot persist experience before decision artifact exists")
    accepted = bool(decision.get("accepted"))
    training_status = str(training.get("status", ""))
    status = "failed" if training_status == "failed" else ("accepted" if accepted else "rejected")
    semantic = decision.get("semantic")
    reward = float(semantic) if isinstance(semantic, (int, float)) else None
    record = ExperienceRecord(
        experience_id="wan-round-%04d" % int(round_index),
        source_uri=round_dir.resolve().as_uri(),
        source_sha256=sha256_file(source),
        source_kind="wan_vllm_round",
        experiment_id="wan-%s" % fingerprint[:16],
        parent_model_id=parent_digest,
        child_model_id=(str(training["student_checkpoint_sha256"]) if training.get("student_checkpoint_sha256") else None),
        operator=proposal.training.method,
        operator_args=proposal.to_dict(),
        training=dict(training),
        evaluation=dict(evaluation) if evaluation is not None else None,
        decision=dict(decision),
        reward=reward,
        status=status,
        provenance={
            "fingerprint": fingerprint,
            "base_digest": str(identity["base_digest"]),
            "capability_digest": capability_digest,
            "evaluator_digest": evaluator_digest,
            "device_id": str(config.get("device")),
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    experience_path = Path(str(config.get("experience_path") or (Path(str(config["output_root"])) / "experiences.jsonl")))
    return ExperienceStore(experience_path).append(record)


def _memory_lessons(config: Mapping[str, Any], rounds: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return compact, trusted lessons for the next Controller proposal.

    The Controller sees recipe/result summaries only.  It never receives raw
    code, evaluator internals, or arbitrary artifact contents.
    """

    sources: list[Mapping[str, Any]] = []
    for path_value in config.get("memory_seed_paths", []):
        path = Path(str(path_value))
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        sources.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    sources.extend(item for item in rounds if isinstance(item, Mapping))
    lessons: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sources:
        decision = item.get("decision") if isinstance(item.get("decision"), Mapping) else {}
        evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), Mapping) else {}
        metrics = evaluation.get("quality_metrics") if isinstance(evaluation.get("quality_metrics"), Mapping) else {}
        proposal = item.get("proposal") or item.get("operator_args")
        training = proposal.get("training") if isinstance(proposal, Mapping) and isinstance(proposal.get("training"), Mapping) else {}
        identity = str(item.get("experience_id") or item.get("round") or item.get("fingerprint") or "memory")
        if identity in seen:
            continue
        seen.add(identity)
        status = str(item.get("status") or ("accepted" if decision.get("accepted") else "rejected"))
        reason = str(decision.get("reason") or status)
        semantic = metrics.get("semantic")
        quality = evaluation.get("quality_score")
        temporal = metrics.get("temporal")
        recipe = {
            key: training[key]
            for key in (
                "learning_rate", "endpoint_weight", "velocity_weight", "clean_anchor_weight",
                "rollout_endpoint_weight", "rollout_clean_weight", "rollout_every", "rollout_steps", "train_steps",
            )
            if key in training
        }
        lesson = {
            "id": identity,
            "status": status,
            "decision": reason,
            "semantic": semantic,
            "quality": quality,
            "temporal": temporal,
            "recipe": recipe,
            "lesson": (
                "Avoid repeating this recipe because it was rejected: %s"
                % reason
                if status in {"rejected", "failed"}
                else "This recipe was accepted; use it as a positive prior."
            ),
        }
        lessons.append(lesson)
    limit = int(config.get("controller_memory_limit", 3))
    return lessons[-limit:]


def _campaign_capabilities(config: Mapping[str, Any]) -> dict[str, Any]:
    enabled = {str(item) for item in config.get("algorithm_capabilities", ("progressive_distillation",))}
    return {
        "capabilities": [
            {
                "name": "progressive_distillation",
                "available": "progressive_distillation" in enabled,
                "backend": "h3_training.algorithms.progressive_distillation.ProgressiveDistillation",
                "constraints": {"binary_stages": True, "rollout_steps": [0, 2, 4]},
                "evidence_level": "V4",
                "reason": "Wan adapter boundary and shared algorithm are loaded",
            },
            {
                "name": "dmd2",
                "available": (
                    "dmd2" in enabled
                    and bool(config.get("dmd2_memory_admitted", False))
                    and bool(config.get("wan_dmd2_adapter_ready", False))
                ),
                "backend": "h3_training.algorithms.dmd2.DMD2",
                "constraints": {"critic_memory_admitted": bool(config.get("dmd2_memory_admitted", False))},
                "evidence_level": "V4",
                "reason": "Wan critic requires both adapter evidence and explicit trusted memory admission",
            },
        ],
        "continuation": {
            "contract": str(config.get("continuation_contract", "30s")),
            "overlap_latent_frames": int(config.get("overlap_latent_frames", 7)),
            "offline_simulation": False,
        },
    }


def _campaign_identity(config: Mapping[str, Any], capabilities: Mapping[str, Any]) -> dict[str, Any]:
    path_digests = {
        "wan_config_digest": sha256_file(Path(str(config["wan_config"]))),
        "training_config_digest": sha256_file(Path(str(config["training_config"]))),
        "training_entrypoint_digest": sha256_file(Path(str(config["training_entrypoint"]))),
        "evaluation_manifest_digest": sha256_file(Path(str(config["evaluation_manifest"]))),
        "clip_model_digest": sha256_path(Path(str(config["clip_model_path"]))),
        "teacher_baseline_digest": sha256_file(Path(str(config["teacher_baseline_result"]))),
        "parent_checkpoint_digest": sha256_file(Path(str(config["parent_checkpoint"]))),
    }
    if str(config.get("training_mode", "legacy")) == "native_rollout_state":
        path_digests["trajectory_record_digest"] = sha256_path(Path(str(config["trajectory_record"])))
    code_files = [Path(__file__), Path(__file__).resolve().parents[1] / "harness4h3" / "student" / "wan_campaign.py"]
    code_files.append(Path(__file__).resolve().parents[1] / "h3_training" / "adapters" / "wan.py")
    path_digests["code_digest"] = _canonical_digest({str(path): sha256_file(path) for path in code_files})
    identity = {
        "schema_version": int(config.get("proposal_schema_version", 2)),
        "student_family": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "teacher_identity": "real_h3_teacher",
        "evaluator_contract": (
            "5s-native-clip-clip-temporal-v1"
            if str(config.get("continuation_contract", "30s")) == "5s"
            else "30s-overlap-window-clip-temporal-v1"
        ),
        "path_digests": path_digests,
        "capability_snapshot": dict(capabilities),
        "fidelity_schedule": list(config.get("fidelity_schedule", ["F1", "F2", "F3"])),
        "offline_simulation": False,
    }
    identity["base_digest"] = _canonical_digest(identity)
    return identity


def _load_or_create_campaign_base(root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(root) / "campaign-base.json"
    if path.is_file():
        stored = _read_json(path)
        if stored.get("base_digest") != identity.get("base_digest"):
            raise ValueError("campaign verification base changed; start a new campaign root")
        return stored
    value = dict(identity)
    atomic_write_json(path, value)
    return value


def run_campaign(config: Mapping[str, Any], *, max_rounds: int | None = None) -> dict[str, Any]:
    root = Path(str(config["output_root"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    capabilities = _campaign_capabilities(config)
    identity = _campaign_identity(config, capabilities)
    _load_or_create_campaign_base(root, identity)
    state = load_campaign_state(root)
    if state.get("base_digest") and state["base_digest"] != identity["base_digest"]:
        raise ValueError("campaign state belongs to a different campaign base")
    state["base_digest"] = identity["base_digest"]
    events_path = root / "events.jsonl"
    complete_rounds, incomplete_rounds = load_complete_rounds(root)
    failed_rounds = {int(value) for value in state.get("failed_rounds", [])}
    resumable_rounds: set[int] = set()
    for round_index in incomplete_rounds:
        if round_index in failed_rounds:
            continue
        round_dir = root / ("round-%04d" % round_index)
        has_proposal = (round_dir / "proposal.json").is_file()
        has_training_result = (round_dir / "training-result.json").is_file()
        has_step_checkpoint = any((round_dir / "training").glob("training-checkpoint*.pt"))
        if has_proposal and not has_training_result and has_step_checkpoint:
            resumable_rounds.add(round_index)
            append_event(
                events_path,
                {"event": "round.resume_ready", "round": round_index, "checkpoint_dir": str(round_dir / "training")},
            )
            continue
        decision = {"accepted": False, "reason": "incomplete_round"}
        atomic_write_json(round_dir / "decision.json", decision)
        failed_rounds.add(round_index)
        append_event(events_path, {"event": "round.incomplete", "round": round_index, "decision": decision})
    state["failed_rounds"] = sorted(failed_rounds)

    parent = Path(str(state.get("parent_checkpoint") or config["parent_checkpoint"])).resolve()
    if not parent.is_file():
        raise FileNotFoundError("parent checkpoint does not exist: %s" % parent)
    parent_digest = sha256_file(parent)
    stored_parent_digest = state.get("parent_checkpoint_sha256")
    if stored_parent_digest and str(stored_parent_digest) != parent_digest:
        raise ValueError("campaign state parent checkpoint digest does not match the file")
    teacher_semantic = _teacher_semantic(config)
    campaign_result_path = root / "campaign-result.json"
    parent_eval = state.get("parent_evaluation")
    if not isinstance(parent_eval, Mapping):
        cached_parent_eval = root / "parent-evaluation.json"
        parent_eval = _read_json(cached_parent_eval) if cached_parent_eval.is_file() else None
    if not isinstance(parent_eval, Mapping):
        acquire_controller_handoff(*_controller_handoff(config))
        try:
            wait_for_gpu_quiet(config)
            parent_eval = evaluate_checkpoint(config, parent, root / "parent-evaluation")
        finally:
            release_controller_handoff(*_controller_handoff(config))
        atomic_write_json(root / "parent-evaluation.json", parent_eval)
        append_event(events_path, {"event": "parent.evaluated", "parent": str(parent), "parent_sha256": parent_digest, "evaluation": parent_eval})
    parent_eval = dict(parent_eval)
    rounds: list[dict[str, Any]] = []
    for record in complete_rounds:
        rounds.append(
            {
                "round": int(record["round"]),
                "fingerprint": record.get("fingerprint"),
                "proposal": record["proposal"],
                "training": record["training"],
                "evaluation": record["evaluation"],
                "decision": record["decision"],
            }
        )
    no_improvement = 0
    round_limit = int(max_rounds if max_rounds is not None else config["max_rounds"])
    start_round = max(
        int(state.get("next_round", 1)),
        max((int(item["round"]) for item in complete_rounds), default=0) + 1,
    )
    if resumable_rounds:
        start_round = min(resumable_rounds)
    manifest_digest = str(identity["path_digests"]["evaluation_manifest_digest"])
    capability_digest = _canonical_digest(capabilities)
    evaluator_digest = _canonical_digest(
        {
            "manifest_digest": manifest_digest,
            "clip_model_digest": identity["path_digests"]["clip_model_digest"],
            "contract": identity["evaluator_contract"],
        }
    )
    code_digest = str(identity["path_digests"]["code_digest"])
    state.update(
        {
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": parent_digest,
            "parent_evaluation": parent_eval,
            "last_complete_round": max((int(item["round"]) for item in complete_rounds), default=0),
            "next_round": start_round,
        }
    )
    write_campaign_state(root, state)
    for round_index in range(start_round, round_limit + 1):
        if round_index in failed_rounds and round_index not in resumable_rounds:
            continue
        round_dir = root / ("round-%04d" % round_index)
        round_dir.mkdir(parents=True, exist_ok=True)
        context = {
            "goal": (
                "Improve Wan Student semantic prompt alignment while preserving the native 5-second contract."
                if str(config.get("continuation_contract", "30s")) == "5s"
                else "Improve Wan Student semantic prompt alignment while preserving continuous 30-second generation."
            ),
            "round": round_index,
            "parent_checkpoint_sha256": parent_digest,
            "parent_evaluation": parent_eval,
            "teacher_semantic": teacher_semantic,
            "semantic_floor": teacher_semantic * float(config["semantic_floor_ratio"]),
            "prior_rounds": rounds,
            "memory_lessons": _memory_lessons(config, rounds),
            "capability_snapshot": capabilities,
            "failed_rounds": sorted(failed_rounds),
            "trusted_constraints": {
                "offline_simulation": False,
                "frames": ("81" if str(config.get("continuation_contract", "30s")) == "5s" else "708-732"),
                "fps": (16 if str(config.get("continuation_contract", "30s")) == "5s" else 24),
                "resolution": "256x256",
                "visual_quality_gate": dict(config.get("visual_quality_gate") or {}),
                "no_code_edits": True,
                "max_train_steps": int(config.get("max_train_steps", 100000)),
                "progressive_source_steps": config.get("progressive_source_steps"),
                "progressive_target_steps": config.get("progressive_target_steps"),
                "wan_trainable_scope": config.get("wan_trainable_scope"),
            },
            "controller_contract": {
                "primary_objective": str(config.get("controller_primary_objective", "semantic_alignment")),
                "secondary_objectives": [str(item) for item in config.get("controller_secondary_objectives", [])],
                "promotion_gate": {
                    "min_semantic_delta": float(config.get("min_semantic_delta", 0.005)),
                    "min_quality_delta": float(config.get("controller_min_quality_delta", 0.0)),
                    "max_temporal_drop": float(config.get("controller_max_temporal_drop", 0.005)),
                    "contract_pass_required": True,
                    "visual_quality_pass_required": bool(
                        (config.get("visual_quality_gate") or {}).get("enabled", False)
                    ),
                    "checkpoint_digest_required": True,
                },
                "search_space": dict(config.get("controller_search_space") or {}),
            },
        }
        resuming_round = round_index in resumable_rounds
        if resuming_round:
            raw_proposal = _read_json(round_dir / "proposal.json")
            proposal = WanTrainingProposal.from_mapping(
                raw_proposal,
                expected_parent_sha256=parent_digest,
                capabilities=capabilities,
            )
        else:
            # A real language-model Controller can occasionally emit a one-key
            # schema typo (for example ``hypo`` instead of ``hypothesis``).
            # Retry the recipe request with bounded validation feedback; never
            # mutate the recipe silently or edit code/model state.
            ensure_controller(config)
            raw_proposal = None
            proposal = None
            normalization = {"applied": False}
            proposal_error: WanProposalError | None = None
            for attempt in range(1, 6):
                request_context = dict(context)
                if proposal_error is not None:
                    request_context["controller_feedback"] = (
                        "The previous JSON proposal was rejected by the trusted schema validator: %s. "
                        "Return the complete corrected object; do not abbreviate or rename fields."
                        % proposal_error
                    )
                raw_proposal = _request_vllm(config, request_context)
                atomic_write_json(round_dir / ("controller-attempt-%02d.json" % attempt), dict(raw_proposal))
                candidate_proposal, candidate_repair = _normalize_controller_recipe(raw_proposal)
                try:
                    proposal = WanTrainingProposal.from_mapping(
                        candidate_proposal,
                        expected_parent_sha256=parent_digest,
                        capabilities=capabilities,
                    )
                    max_train_steps = int(config.get("max_train_steps", 100000))
                    if proposal.training.train_steps > max_train_steps:
                        raise WanProposalError(
                            "training.train_steps=%d exceeds the trusted smoke budget max_train_steps=%d"
                            % (proposal.training.train_steps, max_train_steps)
                        )
                    expected_source = config.get("progressive_source_steps")
                    expected_target = config.get("progressive_target_steps")
                    if proposal.training.method == "progressive_distillation" and expected_source is not None and expected_target is not None:
                        if proposal.training.source_steps != int(expected_source) or proposal.training.target_steps != int(expected_target):
                            raise WanProposalError(
                                "progressive_distillation must use source_steps=%d and target_steps=%d"
                                % (int(expected_source), int(expected_target))
                            )
                    expected_scope = config.get("wan_trainable_scope")
                    if expected_scope is not None and proposal.training.trainable_scope != str(expected_scope):
                        raise WanProposalError(
                            "Wan smoke requires trainable_scope=%s" % str(expected_scope)
                        )
                    _validate_recipe_search_space(proposal, config)
                    normalization = candidate_repair
                    atomic_write_json(round_dir / "controller-normalization.json", normalization)
                    break
                except WanProposalError as exc:
                    proposal = None
                    proposal_error = exc
            if proposal is None:
                raise proposal_error or WanProposalError("Controller did not produce a valid proposal")
        fingerprint = _wan_experiment_fingerprint(
            config,
            identity,
            proposal,
            parent_digest=parent_digest,
            capability_digest=capability_digest,
            evaluator_digest=evaluator_digest,
        )
        existing_fingerprint = None
        fingerprint_artifact = round_dir / "fingerprint.json"
        if resuming_round and fingerprint_artifact.is_file():
            existing_fingerprint = _read_json(fingerprint_artifact).get("fingerprint")
            if existing_fingerprint != fingerprint:
                raise ValueError("resumable round fingerprint does not match trusted campaign identity")
        duplicate = False if resuming_round else register_fingerprint(
            root,
            fingerprint,
            {"round": round_index, "proposal_id": proposal.proposal_id, "parent_checkpoint_sha256": parent_digest},
        )
        atomic_write_json(round_dir / "proposal.json", proposal.to_dict())
        atomic_write_json(round_dir / "fingerprint.json", {"fingerprint": fingerprint, "duplicate": duplicate})
        append_event(
            events_path,
            {
                "event": "proposal.validated",
                "round": round_index,
                "proposal": proposal.to_dict(),
                "fingerprint": fingerprint,
                "duplicate": duplicate,
                "controller_normalization": normalization if not resuming_round else {"applied": False},
            },
        )
        if duplicate:
            decision = {"accepted": False, "reason": "duplicate_fingerprint", "fingerprint": fingerprint}
            atomic_write_json(round_dir / "training-result.json", {"status": "skipped", "reason": "duplicate_fingerprint"})
            atomic_write_json(round_dir / "evaluation-result.json", {"status": "skipped", "reason": "duplicate_fingerprint"})
            atomic_write_json(round_dir / "decision.json", decision)
            duplicate_training = {"status": "skipped", "reason": "duplicate_fingerprint"}
            duplicate_evaluation = {"status": "skipped", "reason": "duplicate_fingerprint"}
            _append_experience(
                config,
                round_dir,
                round_index,
                fingerprint,
                proposal,
                duplicate_training,
                duplicate_evaluation,
                decision,
                parent_digest,
                identity,
                capability_digest,
                evaluator_digest,
            )
            rounds.append({"round": round_index, "fingerprint": fingerprint, "proposal": proposal.to_dict(), "training": duplicate_training, "evaluation": duplicate_evaluation, "decision": decision})
            state["last_complete_round"] = round_index
            state["next_round"] = round_index + 1
            state["failed_rounds"] = sorted(failed_rounds)
            write_campaign_state(root, state)
            no_improvement += 1
            if no_improvement >= int(config["no_improvement_patience"]):
                break
            continue
        acquire_controller_handoff(*_controller_handoff(config))
        try:
            wait_for_gpu_quiet(config)
            training = run_training(config, proposal, parent, round_dir)
            atomic_write_json(round_dir / "training-result.json", training)
            append_event(events_path, {"event": "training.finished", "round": round_index, "training": training})
            if training.get("status") != "success":
                decision = {"accepted": False, "reason": "training_failed"}
                atomic_write_json(round_dir / "evaluation-result.json", {"status": "skipped", "reason": "training_failed"})
                atomic_write_json(round_dir / "decision.json", decision)
                _append_experience(
                    config,
                    round_dir,
                    round_index,
                    fingerprint,
                    proposal,
                    training,
                    {"status": "skipped", "reason": "training_failed"},
                    decision,
                    parent_digest,
                    identity,
                    capability_digest,
                    evaluator_digest,
                )
                rounds.append({"round": round_index, "fingerprint": fingerprint, "proposal": proposal.to_dict(), "training": training, "decision": decision})
                state["failed_rounds"] = sorted(set(state.get("failed_rounds", [])) | {round_index})
                state["last_complete_round"] = round_index
                state["next_round"] = round_index + 1
                write_campaign_state(root, state)
                no_improvement += 1
                if no_improvement >= int(config["no_improvement_patience"]):
                    break
                continue
            child = Path(str(training["student_checkpoint"])).resolve()
            evaluation = evaluate_checkpoint(config, child, round_dir / "evaluation")
            atomic_write_json(round_dir / "evaluation-result.json", evaluation)
            correctness_gate = (
                correctness_gate_decision(
                    config,
                    {**parent_eval, "parent_checkpoint_sha256": parent_digest},
                    evaluation,
                )
                if "correctness_audit_passed" in config or "legacy_r2_promotion_frozen" in config
                else None
            )
            decision = candidate_decision(
                parent_eval,
                evaluation,
                teacher_semantic=teacher_semantic,
                semantic_floor_ratio=float(config["semantic_floor_ratio"]),
                quality_floor_ratio=float(config["quality_floor_ratio"]),
                min_semantic_delta=float(config["min_semantic_delta"]),
                correctness_gate=correctness_gate,
            )
            decision = _apply_controller_promotion_gate(parent_eval, evaluation, decision, config)
            atomic_write_json(round_dir / "decision.json", decision)
            round_record = {
                "round": round_index,
                "fingerprint": fingerprint,
                "proposal": proposal.to_dict(),
                "parent_checkpoint": str(parent),
                "parent_checkpoint_sha256": parent_digest,
                "training": training,
                "evaluation": evaluation,
                "decision": decision,
            }
            _append_experience(
                config,
                round_dir,
                round_index,
                fingerprint,
                proposal,
                training,
                evaluation,
                decision,
                parent_digest,
                identity,
                capability_digest,
                evaluator_digest,
            )
            rounds.append(round_record)
            append_event(events_path, {"event": "round.decided", **round_record})
            state["last_complete_round"] = round_index
            state["next_round"] = round_index + 1
            state["accepted_history"] = [item for item in rounds if item.get("decision", {}).get("accepted")]
            state["parent_checkpoint"] = str(child) if decision.get("accepted") else str(parent)
            state["parent_checkpoint_sha256"] = sha256_file(child) if decision.get("accepted") else parent_digest
            state["parent_evaluation"] = evaluation if decision.get("accepted") else parent_eval
            write_campaign_state(root, state)
        finally:
            release_controller_handoff(*_controller_handoff(config))
        if decision.get("accepted"):
            parent = child
            parent_digest = sha256_file(parent)
            parent_eval = evaluation
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= int(config["no_improvement_patience"]):
            break
    controller_shutdown = _shutdown_controller_on_exit(config)
    result = {
        "status": "success",
        "optimization_status": "COMPLETED",
        "execution_status": "COMPLETED",
        "rounds_completed": len(rounds),
        "accepted_parent_checkpoint": str(parent),
        "accepted_parent_checkpoint_sha256": parent_digest,
        "teacher_semantic": teacher_semantic,
        "semantic_floor": teacher_semantic * float(config["semantic_floor_ratio"]),
        "rounds": rounds,
        "stop_reason": "max_rounds" if len(rounds) >= round_limit else "no_improvement_or_failure",
        "controller_shutdown": controller_shutdown,
    }
    atomic_write_json(campaign_result_path, result)
    append_event(events_path, {"event": "campaign.finished", "result": result})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        config = load_loop_config(args.config)
        result = run_campaign(config, max_rounds=args.max_rounds)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, WanProposalError) as exc:
        output = None
        try:
            output = Path(str(load_loop_config(args.config)["output_root"])) / "campaign-result.json"
            atomic_write_json(output, {"status": "failed", "optimization_status": "INFRA_FAILURE", "execution_status": "INFRA_FAILURE", "failure_code": getattr(exc, "code", "wan_campaign_failed"), "message": str(exc)})
        except Exception:
            pass
        print(json.dumps({"status": "failed", "failure_code": getattr(exc, "code", "wan_campaign_failed"), "message": str(exc), "result_path": str(output) if output else None}, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
