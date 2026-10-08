#!/usr/bin/env python3
"""Run the real Wan Student evaluation -> vLLM proposal -> training loop."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
                    "critic_learning_rate", "velocity_weight", "clean_anchor_weight", "rollout_steps",
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
        ("proposal_schema_version", 2),
        ("algorithm_capabilities", ["progressive_distillation"]),
        ("dmd2_memory_admitted", False),
        ("wan_dmd2_adapter_ready", False),
        ("fidelity_schedule", ["F1", "F2", "F3"]),
        ("continuation_contract", "30s"),
        ("state_path", ""),
        ("fingerprint_path", ""),
        ("experience_path", ""),
        ("checkpoint_every_steps", 8),
        ("offline_simulation", False),
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
    for name in (
        "controller_launcher",
        "controller_launcher_pid_file",
        "controller_launcher_log",
        "controller_hold_file",
        "controller_release_file",
        "controller_worker_lease_file",
        "experience_path",
    ):
        if loop[name] and not Path(str(loop[name])).is_absolute():
            raise ValueError("%s must be absolute" % name)
    if int(loop["max_rounds"]) <= 0:
        raise ValueError("max_rounds must be positive")
    if int(loop["max_cases"]) <= 0 or int(loop["seed_count"]) <= 0:
        raise ValueError("max_cases and seed_count must be positive")
    if not 0 < float(loop["semantic_floor_ratio"]) <= 1:
        raise ValueError("semantic_floor_ratio must be in (0, 1]")
    if not 0 < float(loop["quality_floor_ratio"]) <= 1:
        raise ValueError("quality_floor_ratio must be in (0, 1]")
    if int(loop["proposal_schema_version"]) != 2:
        raise ValueError("proposal_schema_version must be 2")
    if bool(loop["offline_simulation"]):
        raise ValueError("Wan vLLM loop requires offline_simulation=false")
    capabilities = loop["algorithm_capabilities"]
    if not isinstance(capabilities, list) or not capabilities or any(str(item) not in {"progressive_distillation", "dmd2"} for item in capabilities):
        raise ValueError("algorithm_capabilities must list progressive_distillation and/or dmd2")
    if int(loop["checkpoint_every_steps"]) < 0:
        raise ValueError("checkpoint_every_steps must not be negative")
    if not isinstance(loop["fidelity_schedule"], list) or not loop["fidelity_schedule"]:
        raise ValueError("fidelity_schedule must be a non-empty list")
    if any(str(item) not in {"F1", "F2", "F3"} for item in loop["fidelity_schedule"]):
        raise ValueError("fidelity_schedule contains unsupported fidelity")
    return loop


def _controller_handoff(config: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        str(config.get("controller_hold_file") or "") or None,
        str(config.get("controller_release_file") or "") or None,
        str(config.get("controller_worker_lease_file") or "") or None,
    )


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
            os.kill(pid, 0)
            live = True
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
    """Wait until all four Wan role cards have enough free memory."""

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
            free = [float(line.strip()) for line in probe.stdout.splitlines() if line.strip()]
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
    return (
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
    )


def _request_vllm(config: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = proposal_json_schema()
    system = (
        "You are the Wan Student training controller. Return JSON only. "
        "Choose exactly one bounded training policy from the supplied capability snapshot, including method, "
        "rollout, initialization, precision, curriculum, data, continuation, focus, and hypothesis. "
        "Never output code, shell commands, paths, device IDs, architecture, or checkpoint content. "
        "The trusted runner will bind the parent digest and all execution paths."
    )
    user = (
        "Propose exactly one next Wan training policy. Preserve the real H3 Teacher, "
        "offline_simulation=false, the 30-second overlapping-latent contract, and the 1-2B Wan Student. "
        "Prefer semantic alignment when semantic is below the Teacher floor; otherwise protect continuity. "
        "Return fields exactly matching this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
        + "\nCONTEXT=\n"
        + json.dumps(dict(context), ensure_ascii=False, sort_keys=True)
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


def evaluate_checkpoint(config: Mapping[str, Any], checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Generate and evaluate all configured prompts/seeds in one Wan process."""

    from harness4h3.student.evaluation_manifest import EvaluationManifest
    from harness4h3.student.quality import ClipTemporalQualityBackend
    from harness4h3.student.wan import load_pretrained_wan, load_wan_spec, write_wan_video
    from harness4h3.student.wan_long_video import (
        WanLongVideoError,
        WanLongVideoSpec,
        generate_wan_long_video,
        load_wan_transformer_checkpoint,
        validate_wan_long_video,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "evaluation-result.json"
    pipeline = None
    try:
        manifest = EvaluationManifest.from_path(Path(str(config["evaluation_manifest"])))
        spec = load_wan_spec(Path(str(config["wan_config"])))
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
            str(config["clip_model_path"]), device="cpu", sample_frames=12, drift_penalty=0.25
        )
        cases: list[dict[str, Any]] = []
        for case in manifest.cases[: int(config["max_cases"])]:
            for seed in case.seeds[: int(config["seed_count"])]:
                case_dir = output_dir / "videos" / case.case_id / ("seed_%d" % int(seed))
                video_path = case_dir / "wan-long.mp4"
                started = time.perf_counter()
                frames, generation = generate_wan_long_video(
                    pipeline,
                    spec,
                    contract,
                    case.caption,
                    seed=int(seed),
                    device=device,
                )
                write_wan_video(frames, video_path, contract.output_fps)
                validity = validate_wan_long_video(video_path, contract)
                evidence = quality_backend.evaluate(video_path, case.caption)
                cases.append(
                    {
                        "case_id": case.case_id,
                        "seed": int(seed),
                        "caption": case.caption,
                        "video_path": str(video_path),
                        "generation_wall_time_s": float(time.perf_counter() - started),
                        "generation": dict(generation),
                        "validity": dict(validity),
                        "quality": {
                            "score": evidence.aggregate,
                            "semantic": evidence.semantic,
                            "semantic_mean": evidence.semantic_mean,
                            "semantic_p10": evidence.semantic_p10,
                            "temporal": evidence.temporal,
                            "motion": evidence.motion,
                            "score_type": evidence.backend,
                        },
                    }
                )
        if not cases:
            raise ValueError("evaluation manifest selected no cases")
        scores = [float(item["quality"]["score"]) for item in cases]
        semantics = [float(item["quality"]["semantic"]) for item in cases]
        result = {
            "status": "success",
            "valid": True,
            "promotable": True,
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
                "score_type": "clip_temporal",
            },
            "cases": cases,
        }
        atomic_write_json(result_path, result)
        return result
    except (OSError, RuntimeError, TypeError, ValueError, WanLongVideoError) as exc:
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
        target_id="wan-30s-overlap-window-v1",
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
        "training_entrypoint_digest": sha256_file(Path(str(config["training_entrypoint"]))),
        "evaluation_manifest_digest": sha256_file(Path(str(config["evaluation_manifest"]))),
        "clip_model_digest": sha256_path(Path(str(config["clip_model_path"]))),
        "teacher_baseline_digest": sha256_file(Path(str(config["teacher_baseline_result"]))),
        "parent_checkpoint_digest": sha256_file(Path(str(config["parent_checkpoint"]))),
    }
    code_files = [Path(__file__), Path(__file__).resolve().parents[1] / "harness4h3" / "student" / "wan_campaign.py"]
    code_files.append(Path(__file__).resolve().parents[1] / "h3_training" / "adapters" / "wan.py")
    path_digests["code_digest"] = _canonical_digest({str(path): sha256_file(path) for path in code_files})
    identity = {
        "schema_version": int(config.get("proposal_schema_version", 2)),
        "student_family": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "teacher_identity": "real_h3_teacher",
        "evaluator_contract": "30s-overlap-window-clip-temporal-v1",
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
            "contract": "30s-overlap-window-clip-temporal-v1",
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
            "goal": "Improve Wan Student semantic prompt alignment while preserving continuous 30-second generation.",
            "round": round_index,
            "parent_checkpoint_sha256": parent_digest,
            "parent_evaluation": parent_eval,
            "teacher_semantic": teacher_semantic,
            "semantic_floor": teacher_semantic * float(config["semantic_floor_ratio"]),
            "prior_rounds": rounds,
            "capability_snapshot": capabilities,
            "failed_rounds": sorted(failed_rounds),
            "trusted_constraints": {
                "offline_simulation": False,
                "frames": "708-732",
                "fps": 24,
                "resolution": "256x256",
                "no_code_edits": True,
            },
        }
        resuming_round = round_index in resumable_rounds
        if resuming_round:
            raw_proposal = _read_json(round_dir / "proposal.json")
        else:
            ensure_controller(config)
            raw_proposal = _request_vllm(config, context)
        proposal = WanTrainingProposal.from_mapping(
            raw_proposal,
            expected_parent_sha256=parent_digest,
            capabilities=capabilities,
        )
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
        append_event(events_path, {"event": "proposal.validated", "round": round_index, "proposal": proposal.to_dict(), "fingerprint": fingerprint, "duplicate": duplicate})
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
            decision = candidate_decision(
                parent_eval,
                evaluation,
                teacher_semantic=teacher_semantic,
                semantic_floor_ratio=float(config["semantic_floor_ratio"]),
                quality_floor_ratio=float(config["quality_floor_ratio"]),
                min_semantic_delta=float(config["min_semantic_delta"]),
            )
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
