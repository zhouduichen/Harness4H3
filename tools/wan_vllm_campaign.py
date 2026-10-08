#!/usr/bin/env python3
"""Run the real Wan Student evaluation -> vLLM proposal -> training loop."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
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
    sha256_file,
)
from harness4h3.student.gpu import acquire_controller_handoff, release_controller_handoff


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
    for name in (
        "controller_launcher",
        "controller_launcher_pid_file",
        "controller_launcher_log",
        "controller_hold_file",
        "controller_release_file",
        "controller_worker_lease_file",
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
        "--parent-checkpoint",
        str(parent_checkpoint),
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
    return result


def _teacher_semantic(config: Mapping[str, Any]) -> float:
    baseline = _read_json(Path(str(config["teacher_baseline_result"])))
    metrics = baseline.get("optimization_metrics")
    if baseline.get("status") != "success" or not isinstance(metrics, Mapping) or metrics.get("semantic") is None:
        raise ValueError("Teacher baseline lacks semantic evidence")
    return float(metrics["semantic"])


def run_campaign(config: Mapping[str, Any], *, max_rounds: int | None = None) -> dict[str, Any]:
    root = Path(str(config["output_root"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent = Path(str(config["parent_checkpoint"])).resolve()
    if not parent.is_file():
        raise FileNotFoundError("parent checkpoint does not exist: %s" % parent)
    parent_digest = sha256_file(parent)
    teacher_semantic = _teacher_semantic(config)
    campaign_result_path = root / "campaign-result.json"
    events_path = root / "events.jsonl"
    acquire_controller_handoff(*_controller_handoff(config))
    try:
        wait_for_gpu_quiet(config)
        parent_eval = evaluate_checkpoint(config, parent, root / "parent-evaluation")
    finally:
        release_controller_handoff(*_controller_handoff(config))
    atomic_write_json(root / "parent-evaluation.json", parent_eval)
    append_event(events_path, {"event": "parent.evaluated", "parent": str(parent), "parent_sha256": parent_digest, "evaluation": parent_eval})
    rounds: list[dict[str, Any]] = []
    no_improvement = 0
    round_limit = int(max_rounds if max_rounds is not None else config["max_rounds"])
    for round_index in range(1, round_limit + 1):
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
            "trusted_constraints": {
                "offline_simulation": False,
                "frames": "708-732",
                "fps": 24,
                "resolution": "256x256",
                "no_code_edits": True,
            },
        }
        ensure_controller(config)
        raw_proposal = _request_vllm(config, context)
        proposal = WanTrainingProposal.from_mapping(raw_proposal, expected_parent_sha256=parent_digest)
        atomic_write_json(round_dir / "proposal.json", proposal.to_dict())
        append_event(events_path, {"event": "proposal.validated", "round": round_index, "proposal": proposal.to_dict()})
        acquire_controller_handoff(*_controller_handoff(config))
        try:
            wait_for_gpu_quiet(config)
            training = run_training(config, proposal, parent, round_dir)
            atomic_write_json(round_dir / "training-result.json", training)
            append_event(events_path, {"event": "training.finished", "round": round_index, "training": training})
            if training.get("status") != "success":
                decision = {"accepted": False, "reason": "training_failed"}
                atomic_write_json(round_dir / "decision.json", decision)
                rounds.append({"round": round_index, "proposal": proposal.to_dict(), "training": training, "decision": decision})
                break
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
                "proposal": proposal.to_dict(),
                "parent_checkpoint": str(parent),
                "parent_checkpoint_sha256": parent_digest,
                "training": training,
                "evaluation": evaluation,
                "decision": decision,
            }
            rounds.append(round_record)
            append_event(events_path, {"event": "round.decided", **round_record})
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
