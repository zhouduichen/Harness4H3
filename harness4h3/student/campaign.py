"""Resumable proposal→compile→train→evaluate→revise Student campaign."""

from __future__ import annotations

import json
import os
import time
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence

from .compiler import CompileError, CompileManifest, StudentCompiler
from .evaluator import StudentEvaluation, append_experience, make_experience_record
from .metrics import ParetoDecision, pareto_decision, teacher_relative_metrics
from .proposal import StudentProposal, StudentTarget
from .retention import apply_retention, retain_after_evaluation
from .worker import TrainingResult


def build_student_control_plane(
    config: Any,
    provider: StudentProposalProvider,
    output_root: Path,
    teacher_baseline: Optional[Mapping[str, Any]] = None,
):
    """Build the immutable Student control plane used by CLI and Detached runs."""

    from ..campaign.base import ActorIdentity, CampaignBase, canonical_digest, sha256_path
    from ..campaign.capabilities import Capability, CapabilitySnapshot
    from ..campaign.reviews import ReviewPipeline, StructuredLLMReviewAgent

    if str(getattr(config, "quality_backend", "")).lower() != "clip_temporal":
        raise ValueError("Student control plane requires the semantic clip_temporal verifier")
    if not getattr(config, "evaluation_manifest", None) or not getattr(config, "clip_model_path", None):
        raise ValueError("Student control plane requires evaluation_manifest and clip_model_path")

    target = asdict(config.target)
    baseline_metrics = dict((teacher_baseline or {}).get("optimization_metrics") or {})
    baseline_payload = dict(teacher_baseline or {})
    baseline_kind = str(baseline_payload.get("kind") or "")
    if baseline_kind and baseline_kind != "h3_teacher_generation_baseline":
        raise ValueError(
            "Student optimization requires h3_teacher_generation_baseline; "
            "reference_reconstruction_baseline is quality-only"
        )
    teacher_hash = baseline_payload.get("teacher_checkpoint_sha256")
    if teacher_hash is None:
        try:
            teacher_hash = sha256_path(Path(config.teacher_checkpoint))
        except Exception as exc:
            raise ValueError("teacher checkpoint content identity is unavailable") from exc
    manifest_digest = baseline_payload.get("evaluation_manifest_digest")
    if manifest_digest is None:
        try:
            manifest_digest = sha256_path(Path(config.evaluation_manifest))
        except Exception as exc:
            raise ValueError("evaluation manifest content identity is unavailable") from exc
    clip_model_hash = baseline_payload.get("clip_model_hash")
    if clip_model_hash is None:
        try:
            clip_model_hash = sha256_path(Path(config.clip_model_path))
        except Exception as exc:
            raise ValueError("CLIP model content identity is unavailable") from exc
    evaluator_version = "clip-temporal-v1"
    clip_model_identity = str(Path(config.clip_model_path).name)
    quality_floor = None
    if baseline_metrics.get("quality") is not None:
        quality_floor = float(baseline_metrics["quality"]) * float(config.quality_floor_ratio)
    target_profile = {
        "id": "student-target-v1",
        "goal": str(config.goal),
        "student_target": target,
        "constraints": {
            "graph_valid": True,
            "video_decodable": True,
            "evaluation_promotable": True,
            "semantic_verified": True,
            "algorithm_dispatch": True,
            "parent_checkpoint_bound": True,
            "fidelity_executed": True,
            "quality_verified": True,
            "latency_verified": True,
            "memory_verified": True,
            "model_size_verified": True,
            "max_peak_memory_gb": float(config.target.max_peak_memory_gb),
        },
        "objectives": {
            "quality": "maximize",
            "latency_s": "minimize",
            "peak_memory_gb": "minimize",
            "model_size_gb": "minimize",
        },
    }
    if quality_floor is not None:
        target_profile["constraints"]["min_quality_score"] = quality_floor
    verifier_bank = {
        "version": "student-verifier-bank-v2",
        "semantic": "clip_temporal",
        "evaluator_version": evaluator_version,
        "hard": ["graph_valid", "video_decodable", "semantic_verified", "algorithm_dispatch", "parent_checkpoint_bound", "fidelity_executed", "quality_verified", "latency_verified", "memory_verified", "model_size_verified"],
        "pareto": ["quality", "latency_s", "peak_memory_gb", "model_size_gb"],
    }
    capabilities = CapabilitySnapshot(
        (
            Capability("progressive_distillation", "training", "StudentTrainWorker.progressive_distillation", {"method": "progressive_distillation"}, "V4", True, "trusted algorithm dispatch with binary stage validation"),
            Capability("dmd2", "training", "StudentTrainWorker.dmd2", {"method": "dmd2"}, "V4", True, "trusted algorithm dispatch"),
            Capability("quantize", "quantization", "student.quantization.quantize_checkpoint", {"quantization": ["none", "int8"]}, "V4", True, "trusted artifact transform"),
            Capability("semantic_verify", "verification", "student_evaluate_worker.clip_temporal", {"backend": "clip_temporal"}, "V4", True, "fixed semantic verifier"),
        )
    )
    controller_identity = ActorIdentity(
        str(getattr(provider, "provider_name", "student-controller")),
        str(getattr(provider, "model_name", provider.__class__.__name__)),
        "student-controller-v1",
    )
    provider_name = str(getattr(provider, "provider_name", "openai_compatible"))
    model_name = str(getattr(provider, "model_name", "student-review-model"))
    base_url = str(getattr(provider, "base_url", "http://127.0.0.1:11434"))
    timeout_s = float(getattr(provider, "timeout_s", 180.0))
    advocate_identity = ActorIdentity("student-advocate", model_name, "llm-advocate-v1")
    critic_identity = ActorIdentity("student-critical", model_name, "llm-critical-v1")
    modifier_identity = ActorIdentity("student-revision", model_name, "llm-revision-v1")
    evaluator_identity = ActorIdentity("student-evaluator", "student-evaluate-worker", "clip-temporal-v1")
    base = CampaignBase(
        schema_version=1,
        campaign_id="student-%s" % canonical_digest({"goal": config.goal, "teacher_checkpoint_sha256": teacher_hash, "manifest_digest": manifest_digest})[7:23],
        target_profile=target_profile,
        target_profile_hash=canonical_digest(target_profile),
        verifier_bank=verifier_bank,
        verifier_bank_hash=canonical_digest(verifier_bank),
        dataset_manifest_hash=str(manifest_digest),
        evaluation_recipe_hash=canonical_digest({"evaluation_command": list(config.evaluation_command), "quality_backend": config.quality_backend, "evaluator_version": evaluator_version, "evaluation_manifest_digest": manifest_digest, "clip_model_identity": clip_model_identity, "clip_model_hash": clip_model_hash}),
        controller_identity=controller_identity,
        critic_identity=critic_identity,
        evaluator_identity=evaluator_identity,
        prompt_version="student-controller-prompt-v2-batch",
        capability_snapshot=capabilities.to_dict(),
        teacher_checkpoint_sha256=str(teacher_hash),
        evaluation_manifest_digest=str(manifest_digest),
        evaluator_version=evaluator_version,
        clip_model_identity=clip_model_identity,
        clip_model_hash=str(clip_model_hash),
        capability_snapshot_hash=capabilities.digest,
    )
    base_path = Path(output_root).resolve() / "campaign-base.json"
    if base_path.is_file():
        from ..campaign.base import CampaignBase as _CampaignBase

        stored = _CampaignBase.from_dict(json.loads(base_path.read_text(encoding="utf-8")))
        if stored.digest != base.digest:
            raise ValueError("campaign verification base changed; start a new Student campaign output root")
        base = stored
        capabilities = CapabilitySnapshot(tuple(
            Capability(
                str(item["name"]), str(item["category"]), str(item["backend"]), dict(item.get("schema") or {}),
                str(item["evidence_level"]), bool(item["available"]), str(item.get("reason", "")),
            )
            for item in base.capability_snapshot.get("capabilities", ())
        ))
    else:
        _atomic_json(base_path, base.to_dict())
    advocate = StructuredLLMReviewAgent(
        advocate_identity, "advocate", model_name=model_name, base_url=base_url,
        provider=provider_name, timeout_s=timeout_s,
    )
    critical = StructuredLLMReviewAgent(
        base.critic_identity, "critical", model_name=model_name, base_url=base_url,
        provider=provider_name, timeout_s=timeout_s,
    )
    modifier = StructuredLLMReviewAgent(
        modifier_identity, "revision", model_name=model_name, base_url=base_url,
        provider=provider_name, timeout_s=timeout_s,
    )
    return base, capabilities, ReviewPipeline(advocate, critical, modifier, base, max_rounds=1)


class StudentProposalProvider(Protocol):
    provider_name: str
    model_name: str

    def propose(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class StudentProposalBatchProvider(Protocol):
    provider_name: str
    model_name: str

    def propose_batch(self, context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        ...


class StudentRoundWorker(Protocol):
    def run(
        self,
        manifest: CompileManifest,
        round_dir: Path,
        *,
        parent_checkpoint: Optional[str | Path] = None,
        parent_candidate_id: Optional[str] = None,
        fidelity: str = "F1",
    ) -> TrainingResult:
        ...


class StudentRoundEvaluator(Protocol):
    def evaluate(self, checkpoint: Path, round_dir: Path) -> StudentEvaluation:
        ...


def student_proposal_json_schema(target: StudentTarget = StudentTarget()) -> Mapping[str, Any]:
    """Strict schema sent to a local structured-output provider."""

    integer = {"type": "integer", "minimum": 1}
    positive_number = {"type": "number", "exclusiveMinimum": 0}
    architecture = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "family": {"type": "string", "enum": ["video_latent_dit"]},
            "latent_channels": {"type": "integer", "const": target.latent_channels},
            "hidden_size": {"type": "integer", "minimum": target.min_hidden_size, "maximum": target.max_hidden_size},
            "depth": {"type": "integer", "minimum": target.min_depth, "maximum": target.max_depth},
            "num_heads": integer,
            "mlp_ratio": {"type": "number", "minimum": 1, "maximum": 8},
            "spatial_patch": {"type": "integer", "enum": [1, 2, 4]},
            "temporal_patch": {"type": "integer", "enum": [1, 5]},
            # vLLM's guided-json grammar does not implement uniqueItems; the
            # trusted Harness validator enforces uniqueness after decoding.
            "temporal_layers": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "conditioning": {"type": "string", "enum": ["ada_norm_zero"]},
            "norm": {"type": "string", "enum": ["rmsnorm", "layernorm"]},
            "activation": {"type": "string", "enum": ["silu", "gelu"]},
        },
        "required": [
            "family", "latent_channels", "hidden_size", "depth", "num_heads", "mlp_ratio",
            "spatial_patch", "temporal_patch", "temporal_layers", "conditioning", "norm", "activation",
        ],
    }
    training = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "method": {"type": "string", "enum": ["progressive_distillation", "dmd2"]},
            "source_steps": {**integer, "maximum": 256},
            "target_steps": {**integer, "maximum": 64},
            "learning_rate": {**positive_number, "maximum": 0.01},
            "critic_learning_rate": {**positive_number, "maximum": 0.01},
        },
        "required": ["method", "source_steps", "target_steps", "learning_rate", "critic_learning_rate"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "proposal_id": {"type": "string", "pattern": "^student_[0-9]{4,}$"},
            "parent_proposal_id": {"type": ["string", "null"]},
            "teacher": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"checkpoint": {"type": "string", "minLength": 1}, "adapter": {"type": "string", "enum": ["minimax_h3"]}},
                "required": ["checkpoint", "adapter"],
            },
            "architecture": architecture,
            "training": training,
            "deployment": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "precision": {"type": "string", "enum": ["bf16", "fp16"]},
                    "quantization": {"type": "string", "enum": ["none", "int8"]},
                },
                "required": ["precision", "quantization"],
            },
        },
        "required": ["schema_version", "proposal_id", "parent_proposal_id", "teacher", "architecture", "training", "deployment"],
    }


def student_proposal_batch_json_schema(target: StudentTarget = StudentTarget()) -> Mapping[str, Any]:
    """Strict schema for one controller response containing a candidate batch."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": student_proposal_json_schema(target),
            },
        },
        "required": ["proposals"],
    }


def _student_architect_prompt(context: Mapping[str, Any]) -> str:
    return (
        "You are the autonomous Student architect. Return exactly one JSON StudentProposal. "
        "Do not emit code, shell commands, markdown, or explanations outside JSON. "
        "The Harness will reject invalid shapes, memory, or 1B-2B parameter counts. "
        "Choose a legal 1B-2B design rather than a tiny 12-layer model. "
        "For this registered graph, hidden_size=2048 and depth=24 is a legal reference scale, "
        "but you may choose another width/depth/head/patch combination. "
        "temporal_layers must be unique layer indices within depth. "
        "Use method progressive_distillation or dmd2; keep source_steps<=256, target_steps<=64, "
        "and for progressive_distillation use only binary-halving-reachable step pairs. "
        "The trusted campaign config controls the optimizer budget; do not emit max_steps or batch_size. "
        "Keep learning rates<=0.01. "
        "For mlp_ratio=4.0 on this registered graph, hidden=1536/depth=24 "
        "is 727394400 parameters and invalid; hidden=1792/depth=24 is "
        "980747360 and still invalid. Legal reference points include "
        "hidden=1920/depth=24 (1121579616), hidden=1536/depth=36 "
        "(1067335776), or hidden=2048/depth=24 (1271849056). "
        "Estimate parameters before returning and never repeat a below-1B design. CONTEXT="
        + json.dumps(dict(context), ensure_ascii=False, sort_keys=True)
    )


def _student_architect_batch_prompt(context: Mapping[str, Any]) -> str:
    return (
        "You are the autonomous Student architect. Return one JSON object with exactly a "
        "proposals array containing 3 to 5 independent StudentProposal objects. "
        "Do not emit code, shell commands, markdown, or explanations outside JSON. "
        "Candidates must be materially different and must use only the registered graph, "
        "training, and deployment fields. The Harness will reject invalid shapes, memory, "
        "or 1B-2B parameter counts. Keep every candidate legal before returning. CONTEXT="
        + json.dumps(dict(context), ensure_ascii=False, sort_keys=True)
    )


def _parse_student_response(raw: Mapping[str, Any], *, source: str) -> Mapping[str, Any]:
    try:
        if isinstance(raw.get("choices"), list) and raw["choices"]:
            message = raw["choices"][0]["message"]
            content = message["content"]
        else:
            message = raw["message"]
            content = message["content"] if isinstance(message, Mapping) else None
        parsed = json.loads(content) if isinstance(content, str) else content
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("proposal_invalid: %s response was not valid StudentProposal JSON: %s" % (source, exc)) from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("proposal_invalid: %s response was not a JSON object" % source)
    return dict(parsed)


def _parse_student_batch_response(raw: Mapping[str, Any], *, source: str) -> Sequence[Mapping[str, Any]]:
    parsed = _parse_student_response(raw, source=source)
    proposals = parsed.get("proposals")
    if not isinstance(proposals, list) or not 3 <= len(proposals) <= 5:
        raise ValueError("proposal_invalid: %s response must contain 3 to 5 proposals" % source)
    if not all(isinstance(item, Mapping) for item in proposals):
        raise ValueError("proposal_invalid: %s proposals must be JSON objects" % source)
    return tuple(dict(item) for item in proposals)


def _request_json_with_retry(request: urllib.request.Request, timeout_s: float) -> Mapping[str, Any]:
    """Wait through a server-local LLM handoff/restart without losing a round."""

    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_error: Optional[BaseException] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_error is not None:
                raise last_error
            raise urllib.error.URLError("proposal request timeout")
        try:
            with urllib.request.urlopen(request, timeout=min(180.0, max(1.0, remaining))) as response:
                raw = json.loads(response.read().decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("proposal response must be a JSON object")
            return dict(raw)
        except urllib.error.HTTPError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))


class OllamaStudentProposalProvider:
    provider_name = "ollama"

    def __init__(self, model_name: str, target: StudentTarget, *, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 180.0):
        self.model_name = str(model_name)
        self.target = target
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)

    def propose(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        schema = student_proposal_json_schema(self.target)
        prompt = _student_architect_prompt(context)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0.2},
        }
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = _request_json_with_retry(request, self.timeout_s)
            parsed = _parse_student_response(raw, source="Ollama")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("proposal_invalid: Ollama proposal request failed: %s" % exc) from exc
        return dict(parsed)

    def propose_batch(self, context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": _student_architect_batch_prompt(context)}],
            "stream": False,
            "think": False,
            "format": student_proposal_batch_json_schema(self.target),
            "options": {"temperature": 0.4},
        }
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = _request_json_with_retry(request, self.timeout_s)
            return _parse_student_batch_response(raw, source="Ollama")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("proposal_invalid: Ollama proposal batch request failed: %s" % exc) from exc


class OpenAICompatibleStudentProposalProvider:
    """Use a local vLLM/OpenAI-compatible server for structured proposals."""

    provider_name = "openai_compatible"

    def __init__(self, model_name: str, target: StudentTarget, *, base_url: str, timeout_s: float = 180.0):
        self.model_name = str(model_name)
        self.target = target
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)

    @property
    def endpoint(self) -> str:
        return self.base_url + ("/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions")

    def propose(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        schema = student_proposal_json_schema(self.target)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": _student_architect_prompt(context)}],
            "temperature": 0.2,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "student_proposal", "strict": True, "schema": schema},
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = _request_json_with_retry(request, self.timeout_s)
            return dict(_parse_student_response(raw, source="OpenAI-compatible"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
            except OSError:
                detail = str(exc)
            raise ValueError("proposal_invalid: OpenAI-compatible proposal request failed: HTTP %s: %s" % (exc.code, detail)) from exc
        except (OSError, urllib.error.URLError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("proposal_invalid: OpenAI-compatible proposal request failed: %s" % exc) from exc

    def propose_batch(self, context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        schema = student_proposal_batch_json_schema(self.target)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": _student_architect_batch_prompt(context)}],
            "temperature": 0.4,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "student_proposal_batch", "strict": True, "schema": schema},
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = _request_json_with_retry(request, self.timeout_s)
            return _parse_student_batch_response(raw, source="OpenAI-compatible")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
            except OSError:
                detail = str(exc)
            raise ValueError("proposal_invalid: OpenAI-compatible proposal batch request failed: HTTP %s: %s" % (exc.code, detail)) from exc
        except (OSError, urllib.error.URLError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("proposal_invalid: OpenAI-compatible proposal batch request failed: %s" % exc) from exc


def build_student_proposal_provider(
    provider: str,
    model_name: str,
    target: StudentTarget,
    *,
    base_url: str,
    timeout_s: float,
) -> StudentProposalProvider:
    normalized = str(provider).strip().lower()
    if normalized == "ollama":
        return OllamaStudentProposalProvider(model_name, target, base_url=base_url, timeout_s=timeout_s)
    if normalized in {"vllm", "openai_compatible", "openai-compatible"}:
        return OpenAICompatibleStudentProposalProvider(model_name, target, base_url=base_url, timeout_s=timeout_s)
    raise ValueError("unsupported Student proposal provider: %s" % provider)


@dataclass(frozen=True)
class CampaignRound:
    round_index: int
    proposal: Optional[StudentProposal]
    compile: Optional[CompileManifest]
    training: Optional[TrainingResult]
    evaluation: Optional[StudentEvaluation]
    failure_code: Optional[str]
    message: str
    candidate_id: Optional[str] = None
    parent_candidate_id: Optional[str] = None


@dataclass(frozen=True)
class CampaignResult:
    status: str
    rounds_completed: int
    rounds: tuple[CampaignRound, ...]
    failure_code: Optional[str] = None
    message: str = ""
    promotable: bool = False
    target_satisfied: bool = False
    candidate_decisions: tuple[Mapping[str, Any], ...] = ()
    stop_reason: str = ""
    trace_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rounds_completed": self.rounds_completed,
            "failure_code": self.failure_code,
            "message": self.message,
            "promotable": self.promotable,
            "target_satisfied": self.target_satisfied,
            "candidate_decisions": [dict(item) for item in self.candidate_decisions],
            "stop_reason": self.stop_reason,
            "trace_path": self.trace_path,
            "rounds": [
                {
                    "round_index": item.round_index,
                    "proposal": item.proposal.to_dict() if item.proposal else None,
                    "compile": item.compile.to_dict() if item.compile else None,
                    "training": item.training.to_dict() if item.training else None,
                    "evaluation": item.evaluation.to_dict() if item.evaluation else None,
                    "failure_code": item.failure_code,
                    "message": item.message,
                    "candidate_id": item.candidate_id,
                    "parent_candidate_id": item.parent_candidate_id,
                }
                for item in self.rounds
            ],
        }


@dataclass(frozen=True)
class StudentQualityPolicy:
    quality_floor_ratio: float = 0.90
    min_reward_delta: float = 0.02
    material_efficiency_gain: float = 0.05
    max_metric_regression: float = 0.02
    no_improvement_patience: int = 3

    def __post_init__(self) -> None:
        if not 0 < float(self.quality_floor_ratio) <= 1:
            raise ValueError("quality_floor_ratio must be in (0, 1]")
        if any(float(value) < 0 for value in (self.min_reward_delta, self.material_efficiency_gain, self.max_metric_regression)):
            raise ValueError("quality policy margins must be non-negative")
        if int(self.no_improvement_patience) <= 0:
            raise ValueError("no_improvement_patience must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class StudentCampaign:
    def __init__(
        self,
        provider: StudentProposalProvider,
        compiler: StudentCompiler,
        worker: StudentRoundWorker,
        evaluator: StudentRoundEvaluator,
        *,
        goal: str = "Produce a runnable 1B-2B H3-distilled edge video generator.",
        target: Optional[StudentTarget] = None,
        output_root: Path,
        experience_path: Optional[Path] = None,
        max_failures: int = 4,
        min_rounds_before_success: int = 1,
        retention_handler: Optional[Callable[[TrainingResult, StudentEvaluation, str, str], None]] = None,
        campaign_base: Optional[Any] = None,
        capability_snapshot: Optional[Any] = None,
        review_pipeline: Optional[Any] = None,
        acceptance_gate: Optional[Any] = None,
        hard_constraints: Optional[Mapping[str, Any]] = None,
        objectives: Optional[Mapping[str, str]] = None,
        max_candidates: int = 5,
        min_candidates: int = 3,
        fidelity_schedule: Sequence[str] = ("F1",),
        initial_parent_checkpoint: Optional[str | Path] = None,
        teacher_baseline: Optional[Mapping[str, Any]] = None,
        quality_policy: Optional[Mapping[str, Any] | StudentQualityPolicy] = None,
        max_steps: int = 256,
    ):
        self.provider = provider
        self.compiler = compiler
        self.worker = worker
        self.evaluator = evaluator
        self.goal = str(goal)
        self.target = target or compiler.target
        self.output_root = Path(output_root).resolve()
        self.experience_path = Path(experience_path or self.output_root / "experience.jsonl").resolve()
        self.max_failures = int(max_failures)
        self.min_rounds_before_success = int(min_rounds_before_success)
        self.retention_handler = retention_handler
        self.campaign_base = campaign_base
        self.capability_snapshot = capability_snapshot
        self.review_pipeline = review_pipeline
        self.acceptance_gate = acceptance_gate
        self.hard_constraints = dict(hard_constraints or {})
        self.objectives = dict(objectives or {})
        self.max_candidates = int(max_candidates)
        self.min_candidates = int(min_candidates)
        self.fidelity_schedule = tuple(str(item) for item in fidelity_schedule)
        self.parent_checkpoint = str(initial_parent_checkpoint) if initial_parent_checkpoint else None
        self.teacher_baseline = dict(teacher_baseline or {})
        self.max_steps = int(max_steps)
        if isinstance(quality_policy, StudentQualityPolicy):
            self.quality_policy = quality_policy
        elif quality_policy is not None:
            self.quality_policy = StudentQualityPolicy(**dict(quality_policy))
        else:
            self.quality_policy = None
        if self.quality_policy is not None and not self.teacher_baseline:
            raise ValueError("quality policy requires teacher_baseline")
        self._incumbent: Optional[dict[str, Any]] = None
        self._frontier: list[dict[str, Any]] = []
        self._best_reward: Optional[float] = None
        self._no_improvement_rounds = 0
        self._improvement_count = 0
        self._control_parent_metrics: dict[str, float] = {}
        if self.max_failures < 0:
            raise ValueError("max_failures must be non-negative")
        if self.min_rounds_before_success <= 0:
            raise ValueError("min_rounds_before_success must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 1 <= self.min_candidates <= self.max_candidates:
            raise ValueError("candidate bounds are invalid")
        if not self.fidelity_schedule:
            raise ValueError("fidelity_schedule must not be empty")
        if self.fidelity_schedule != tuple(("F1", "F2", "F3")[: len(self.fidelity_schedule)]):
            raise ValueError("fidelity_schedule must be an ordered F1 -> F2 -> F3 prefix")
        self.events_path = self.output_root / "campaign-events.jsonl"
        self.resume_path = self.output_root / "resume.json"
        self.decision_trace_path = self.output_root / "decision-trace.jsonl"
        self.archive_path = self.output_root / "archive.jsonl"

    def _append_event(self, payload: Mapping[str, Any]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _context(
        self,
        round_index: int,
        failures: Sequence[Mapping[str, Any]],
        seen: Sequence[str],
        prior_rounds: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        bounded_failures = [dict(item) for item in failures[-8:]]
        context = {
            "goal": self.goal,
            "round": round_index,
            "target": asdict(self.target),
            "provider": {"name": self.provider.provider_name, "model": self.provider.model_name},
            "seen_proposal_digests": list(seen[-8:]),
            "failures": bounded_failures,
            "prior_rounds": [dict(item) for item in prior_rounds[-4:]],
        }
        if self.quality_policy is not None:
            context["optimization"] = {
                "teacher_baseline": dict(self.teacher_baseline),
                "incumbent": dict(self._incumbent or {}),
                "frontier": [dict(item) for item in self._frontier[-4:]],
                "best_reward": self._best_reward,
                "no_improvement_rounds": self._no_improvement_rounds,
                "quality_policy": self.quality_policy.to_dict(),
            }
        return context

    def _persist_resume(self, next_round: int, status: str, failures: Sequence[Mapping[str, Any]], seen: Sequence[str]) -> None:
        payload: dict[str, Any] = {
            "next_round": next_round,
            "status": status,
            "failures": list(failures[-16:]),
            "seen_proposal_digests": list(seen[-16:]),
        }
        if self.quality_policy is not None:
            payload["teacher_baseline"] = dict(self.teacher_baseline)
            payload["incumbent"] = dict(self._incumbent or {})
            payload["frontier"] = [dict(item) for item in self._frontier]
            payload["best_reward"] = self._best_reward
            payload["no_improvement_rounds"] = self._no_improvement_rounds
            payload["improvement_count"] = self._improvement_count
        _atomic_json(self.resume_path, payload)

    def _load_resume(self) -> tuple[int, list[Mapping[str, Any]], list[str]]:
        if not self.resume_path.is_file():
            return 1, [], []
        try:
            raw = json.loads(self.resume_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return 1, [], []
        if not isinstance(raw, Mapping) or raw.get("status") not in {"running", "worker_running", "failed"}:
            return 1, [], []
        if self.quality_policy is not None:
            stored_baseline = raw.get("teacher_baseline")
            if isinstance(stored_baseline, Mapping) and stored_baseline:
                self.teacher_baseline = dict(stored_baseline)
            stored_incumbent = raw.get("incumbent")
            self._incumbent = dict(stored_incumbent) if isinstance(stored_incumbent, Mapping) and stored_incumbent else None
            stored_frontier = raw.get("frontier")
            self._frontier = [dict(item) for item in stored_frontier if isinstance(item, Mapping)] if isinstance(stored_frontier, list) else []
            self._best_reward = float(raw["best_reward"]) if raw.get("best_reward") is not None else None
            self._no_improvement_rounds = int(raw.get("no_improvement_rounds", 0))
            self._improvement_count = int(raw.get("improvement_count", 0))
        try:
            next_round = max(1, int(raw.get("next_round", 1)))
        except (TypeError, ValueError):
            next_round = 1
        if raw.get("status") == "failed":
            # A new run after failure starts at round one but keeps the
            # durable failure context for the next LLM proposal.
            next_round = 1
        failures = raw.get("failures") if isinstance(raw.get("failures"), list) else []
        seen = raw.get("seen_proposal_digests") if isinstance(raw.get("seen_proposal_digests"), list) else []
        # A worker-running checkpoint from an older supervisor may contain
        # only its manifest. Reconstruct the bounded context from the durable
        # event stream so a restart does not erase the LLM's failure feedback.
        if not failures or not seen:
            try:
                for line in self.events_path.read_text(encoding="utf-8").splitlines()[-16:]:
                    event = json.loads(line)
                    if not isinstance(event, Mapping):
                        continue
                    if event.get("status") in {"failed", "rejected"}:
                        failures.append(
                            {
                                "round": event.get("round"),
                                "failure_code": event.get("failure_code"),
                                "message": event.get("message", ""),
                                "type": "optimization_rejected" if event.get("status") == "rejected" else "failure",
                            }
                        )
                    digest = event.get("proposal_digest")
                    if digest:
                        seen.append(str(digest))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return next_round, [dict(item) for item in failures if isinstance(item, Mapping)], [str(item) for item in seen]

    @property
    def strict_optimization(self) -> bool:
        return self.quality_policy is not None

    @staticmethod
    def _optimization_metrics(evaluation: StudentEvaluation) -> Mapping[str, float]:
        raw = evaluation.metric_evidence.get("optimization_metrics")
        if not isinstance(raw, Mapping):
            raise ValueError("metric_evidence_missing: optimization_metrics is required")
        result = {}
        for name in ("quality", "latency", "memory", "size", "energy"):
            if raw.get(name) is not None:
                result[name] = float(raw[name])
        required = ("quality", "latency", "memory", "size")
        if any(name not in result for name in required):
            raise ValueError("metric_evidence_missing: required optimization metric is absent")
        return result

    def _apply_quality_policy(self, evaluation: StudentEvaluation) -> tuple[StudentEvaluation, Optional[ParetoDecision], Optional[dict[str, float]]]:
        if not self.strict_optimization:
            return evaluation, None, None
        if not evaluation.valid:
            return evaluation, None, None
        try:
            candidate_metrics = dict(self._optimization_metrics(evaluation))
            teacher_metrics = dict(self.teacher_baseline.get("optimization_metrics") or self.teacher_baseline)
            relative = teacher_relative_metrics(candidate_metrics, teacher_metrics)
            candidate = {
                "quality_ratio": relative.quality_ratio,
                "latency": relative.latency,
                "memory": relative.memory,
                "size": relative.size,
            }
            if relative.energy is not None:
                candidate["energy"] = relative.energy
            policy = self.quality_policy
            assert policy is not None
            decision = pareto_decision(
                candidate=candidate,
                incumbent=self._incumbent,
                quality_floor_ratio=policy.quality_floor_ratio,
                min_reward_delta=policy.min_reward_delta,
                material_efficiency_gain=policy.material_efficiency_gain,
                max_regression=policy.max_metric_regression,
            )
        except (TypeError, ValueError, KeyError) as exc:
            return replace(evaluation, promotable=False, failure_code="metric_evidence_missing", message=str(exc)), None, None
        evidence = dict(evaluation.metric_evidence)
        evidence["teacher_relative"] = relative.to_dict()
        evidence["optimization_decision"] = decision.to_dict()
        return replace(
            evaluation,
            promotable=decision.promotable,
            failure_code=None if decision.promotable else decision.reason,
            message=decision.reason,
            metric_evidence=evidence,
            reward=decision.reward,
        ), decision, candidate

    def _update_quality_state(self, candidate: Mapping[str, float], decision: ParetoDecision) -> None:
        if decision.promotable:
            self._incumbent = dict(candidate)
            self._frontier.append(dict(candidate))
            if decision.reward is not None:
                self._best_reward = max(self._best_reward or decision.reward, decision.reward)
            if decision.reason == "pareto_improvement":
                self._improvement_count += 1
                self._no_improvement_rounds = 0
            else:
                self._no_improvement_rounds = 0
        else:
            self._no_improvement_rounds += 1

    @staticmethod
    def _round_failure(
        round_index: int,
        proposal: Optional[StudentProposal],
        compile_manifest: Optional[CompileManifest],
        training: Optional[TrainingResult],
        evaluation: Optional[StudentEvaluation],
        code: str,
        message: str,
    ) -> CampaignRound:
        return CampaignRound(round_index, proposal, compile_manifest, training, evaluation, code, message)

    def _control_snapshot(self) -> Any:
        from ..campaign.capabilities import CapabilitySnapshot

        if self.capability_snapshot is None or not isinstance(self.capability_snapshot, CapabilitySnapshot):
            raise ValueError("control-plane StudentCampaign requires a CapabilitySnapshot")
        declared = self.campaign_base.capability_snapshot.get("digest")
        if declared is not None and str(declared) != self.capability_snapshot.digest:
            raise ValueError("capability snapshot does not match immutable campaign base")
        return self.capability_snapshot

    def _control_hard_constraints(self) -> Dict[str, Any]:
        if self.hard_constraints:
            constraints = dict(self.hard_constraints)
        else:
            target_profile = self.campaign_base.target_profile
            raw_constraints = target_profile.get("constraints") if isinstance(target_profile, Mapping) else {}
            constraints = dict(raw_constraints) if isinstance(raw_constraints, Mapping) else {}
            quality = target_profile.get("quality") if isinstance(target_profile, Mapping) else {}
            if isinstance(quality, Mapping) and quality.get("min_quality_score") is not None:
                constraints["min_quality_score"] = quality["min_quality_score"]
        # These are campaign invariants, not controller-selected objectives.
        constraints.setdefault("video_decodable", True)
        constraints.setdefault("evaluation_promotable", True)
        return constraints

    def _control_objectives(self) -> Dict[str, str]:
        if self.objectives:
            return dict(self.objectives)
        raw = self.campaign_base.target_profile.get("objectives")
        result: Dict[str, str] = {}
        if isinstance(raw, Mapping):
            for name, value in raw.items():
                if isinstance(value, Mapping):
                    direction = value.get("direction")
                else:
                    direction = value
                if direction in {"maximize", "minimize"}:
                    result[str(name)] = str(direction)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, Mapping) and item.get("name") and item.get("direction") in {"maximize", "minimize"}:
                    result[str(item["name"])] = str(item["direction"])
        return result or {"quality": "maximize", "latency_s": "minimize", "peak_memory_gb": "minimize"}

    def _control_batch(self, context: Mapping[str, Any], *, round_index: int, parent_id: str, parent_generation: int) -> Any:
        from ..campaign.proposals import CandidateEnvelope, ProposalBatch

        producer = getattr(self.provider, "propose_batch", None)
        if callable(producer):
            raw_items = producer(context)
        else:
            # This fallback is intentionally only a compatibility bridge. A
            # real control-plane provider should implement propose_batch so the
            # review and diversity contract is visible to the model.
            raw_items = [self.provider.propose(context)]
        if isinstance(raw_items, Mapping):
            raw_items = raw_items.get("proposals")
        if not isinstance(raw_items, (list, tuple)):
            raise ValueError("proposal_invalid: batch provider must return a proposals array")
        proposals = []
        parse_errors = []
        for index, raw in enumerate(raw_items, 1):
            try:
                proposal = raw if isinstance(raw, StudentProposal) else StudentProposal.from_dict(raw)
                report = proposal.validate(self.target)
                if not report.ok:
                    raise ValueError("; ".join(report.errors))
                proposals.append(proposal)
            except (TypeError, ValueError, KeyError) as exc:
                parse_errors.append("candidate %d: %s" % (index, exc))
        if parse_errors:
            raise ValueError("proposal_invalid: " + " | ".join(parse_errors))
        candidates = []
        for index, proposal in enumerate(proposals, 1):
            # The campaign-selected parent is authoritative. A controller may
            # describe lineage in its proposal, but it cannot redirect the
            # executable checkpoint inheritance edge.
            proposal_parent = parent_id
            candidate_id = proposal.proposal_id
            candidates.append(
                CandidateEnvelope(
                    candidate_id=candidate_id,
                    parent_candidate_id=proposal_parent,
                    generation=parent_generation + 1,
                    experiment_id="%s:%s:%s" % (self.campaign_base.campaign_id, round_index, candidate_id),
                    proposal_digest=proposal.digest,
                    mutation_fields=(
                        "architecture.hidden_size",
                        "architecture.depth",
                        "training.method",
                        "deployment.precision",
                        "quantization",
                    ),
                    architecture=proposal.architecture.to_dict(),
                    training_recipe=proposal.training.to_dict(),
                    deployment_recipe=proposal.deployment.to_dict(),
                    provenance={
                        "source": "student",
                        "provider": str(getattr(self.provider, "provider_name", "unknown")),
                        "model": str(getattr(self.provider, "model_name", "unknown")),
                        "student_proposal": proposal.to_dict(),
                        "batch_index": index,
                    },
                    predicted_metric_delta={},
                )
            )
        return ProposalBatch(
            batch_id="batch:%s:%04d" % (self.campaign_base.campaign_id, round_index),
            round_id="R%04d" % round_index,
            diagnosis=str(context.get("diagnosis") or "controller proposal batch"),
            parent_selection_evidence_ids=("parent:%s" % parent_id,),
            candidates=tuple(candidates),
        )

    @staticmethod
    def _control_public_execution(execution: Mapping[str, Any]) -> Mapping[str, Any]:
        return {str(key): value for key, value in execution.items() if not str(key).startswith("_")}

    @staticmethod
    def _control_best(items: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        if not items:
            return None
        return max(
            items,
            key=lambda item: (
                bool(item.get("target_satisfied")),
                bool(item.get("promotable")),
                float((item.get("objective_values") or {}).get("quality", (item.get("objective_values") or {}).get("quality_score", 0.0)) or 0.0),
                -float((item.get("objective_values") or {}).get("latency_s", float("inf")) or float("inf")),
            ),
        )

    def _append_control_experience(
        self,
        candidate: Any,
        evidence: Mapping[str, Any],
        decision: Any,
        *,
        round_index: int,
        training: Optional[Mapping[str, Any]] = None,
        baseline_metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        actual = {
            name: item.value
            for name, item in evidence.items()
            if item.value is not None and item.metric_name not in {"video_decodable", "evaluation_promotable"}
        }
        predicted = dict(candidate.predicted_metric_delta)
        baseline = dict(baseline_metrics or {})
        actual_delta = {
            name: float(value) - float(baseline.get(name, 0.0))
            for name, value in actual.items()
        }
        prediction_error = {
            name: float(value) - float(predicted[name])
            for name, value in actual_delta.items()
            if name in predicted and isinstance(predicted[name], (int, float))
        }
        record = {
            "schema_version": 2,
            "experience_id": "%s:%04d:%s" % (self.campaign_base.campaign_id, round_index, candidate.candidate_id),
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": candidate.parent_candidate_id,
            "candidate_digest": candidate.digest(self.campaign_base),
            "predicted_metric_delta": predicted,
            "actual_metrics": actual,
            "actual_delta": actual_delta,
            "prediction_error": prediction_error,
            "gate": decision.to_dict(),
            "execution_evidence": {
                "algorithm_name": (training or {}).get("algorithm_name"),
                "algorithm_path": (training or {}).get("algorithm_path"),
                "algorithm_dispatch": actual.get("algorithm_dispatch"),
                "parent_checkpoint_bound": actual.get("parent_checkpoint_bound"),
                "parent_inherited": actual.get("parent_inherited"),
                "fidelity_executed": actual.get("fidelity_executed"),
                "semantic_verified": actual.get("semantic_verified"),
            },
            "promotable": bool(getattr(decision, "promotable", False)),
            "target_satisfied": bool(getattr(decision, "target_satisfied", False)),
            "failure_code": None if bool(getattr(decision, "feasible", getattr(decision, "passed", False))) else "gate_rejected",
        }
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        with self.experience_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _mapping_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        keys = set(left) | set(right)
        if not keys:
            return 0.0
        distances = []
        for key in keys:
            a, b = left.get(key), right.get(key)
            if isinstance(a, Mapping) and isinstance(b, Mapping):
                distances.append(StudentCampaign._mapping_distance(a, b))
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
                scale = max(1.0, abs(float(a)), abs(float(b)))
                distances.append(min(1.0, abs(float(a) - float(b)) / scale))
            else:
                distances.append(0.0 if a == b else 1.0)
        return sum(distances) / float(len(distances))

    def _control_novelty(self, candidate: Any, decision: Mapping[str, Any]) -> Mapping[str, float]:
        previous = []
        if self.archive_path.is_file():
            try:
                previous = [
                    json.loads(line)
                    for line in self.archive_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                previous = []
        previous = [item for item in previous if isinstance(item, Mapping) and item.get("candidate_digest") != candidate.digest(self.campaign_base)]
        if not previous:
            return {"architecture_distance": 1.0, "training_distance": 1.0, "behavior_distance": 1.0, "novelty": 1.0}
        architecture = dict(candidate.architecture)
        training = dict(candidate.training_recipe)
        behavior = dict(decision.get("objective_values") or {})
        distances = []
        for item in previous:
            distances.append(
                (
                    self._mapping_distance(architecture, dict(item.get("architecture") or {})),
                    self._mapping_distance(training, dict(item.get("training_recipe") or {})),
                    self._mapping_distance(behavior, dict(item.get("behavior") or {})),
                )
            )
        minimum = min(distances, key=lambda value: sum(value))
        return {
            "architecture_distance": float(minimum[0]),
            "training_distance": float(minimum[1]),
            "behavior_distance": float(minimum[2]),
            "novelty": float(sum(minimum) / 3.0),
        }

    def _append_control_archive(self, candidate: Any, decision: Mapping[str, Any]) -> Mapping[str, float]:
        if candidate is None:
            return {"architecture_distance": 0.0, "training_distance": 0.0, "behavior_distance": 0.0, "novelty": 0.0}
        mutation_fields = tuple(candidate.mutation_fields)
        novelty = self._control_novelty(candidate, decision)
        common = {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": candidate.parent_candidate_id,
            "generation": candidate.generation,
            "candidate_digest": candidate.digest(self.campaign_base),
            "mutation_fields": list(mutation_fields),
            **novelty,
            "architecture": dict(candidate.architecture),
            "training_recipe": dict(candidate.training_recipe),
            "behavior": dict(decision.get("objective_values") or {}),
            "decision": dict(decision),
            "target_profile_hash": self.campaign_base.target_profile_hash,
            "verifier_bank_hash": self.campaign_base.verifier_bank_hash,
        }
        kinds = ["pareto" if decision.get("promotable") else "failure", "novelty"]
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a", encoding="utf-8") as handle:
            for kind in kinds:
                handle.write(json.dumps({**common, "archive_kind": kind}, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return novelty

    def _control_round_dir(self, round_index: int, candidate_id: str, fidelity: str) -> Path:
        """Use a worker-safe flat name for local and SSH round adapters."""

        safe_candidate = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(candidate_id))
        safe_fidelity = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(fidelity))
        return self.output_root / ("student_%04d_%s_%s" % (round_index, safe_candidate, safe_fidelity))

    def _run_control_plane(self, *, max_rounds: int) -> CampaignResult:
        from ..campaign.adapters import StudentCampaignAdapter
        from ..campaign.base import ActorIdentity
        from ..campaign.events import DecisionTrace
        from ..campaign.failures import FailureAttributor
        from ..campaign.gates import AcceptanceGate, MetricEvidence, pareto_dominates
        from ..campaign.proposals import validate_batch
        from .fidelity import FidelityGate, stage_spec

        if self.campaign_base is None:
            raise ValueError("campaign base is required for control-plane execution")
        snapshot = self._control_snapshot()
        if self.review_pipeline is None:
            raise ValueError("control-plane StudentCampaign requires an independent review_pipeline")
        if getattr(getattr(self.review_pipeline, "critical", None), "identity", None) != self.campaign_base.critic_identity:
            raise ValueError("review pipeline Critical identity must match the immutable campaign critic identity")
        gate = self.acceptance_gate or AcceptanceGate()
        fidelity_gate = FidelityGate()
        trace = DecisionTrace(self.decision_trace_path, self.campaign_base)
        existing = trace.read()
        if not existing:
            trace.append(
                "campaign.created",
                round_id=None,
                experiment_id=None,
                candidate_id=None,
                parent_candidate_id=None,
                actor=self.campaign_base.controller_identity,
                payload={
                    "base": self.campaign_base.to_dict(),
                    "capability_snapshot_digest": snapshot.digest,
                    "control_plane": "campaign-v1",
                },
                evidence_ids=(),
            )
            existing = trace.read()
        completed_rounds = []
        candidate_decisions = []
        archive = []
        seen_candidates = set()
        parent_id = "M0000"
        parent_generation = 0
        parent_checkpoint = self.parent_checkpoint
        start_round = 1
        for event in existing:
            if event.round_id and event.round_id.startswith("R"):
                try:
                    start_round = max(start_round, int(event.round_id[1:]) + 1)
                except ValueError:
                    pass
            if event.event_type == "parent.selected":
                selected = event.payload.get("candidate_id")
                if selected:
                    parent_id = str(selected)
                    parent_generation = int(event.payload.get("generation", parent_generation))
                    seen_candidates.add(parent_id)
                    selected_checkpoint = (
                        event.payload.get("inheritance_checkpoint")
                        or event.payload.get("parent_checkpoint")
                        or event.payload.get("child_checkpoint")
                    )
                    if not selected_checkpoint:
                        raise ValueError("parent.selected is missing executable child_checkpoint")
                    parent_checkpoint = str(selected_checkpoint)
        adapter = StudentCampaignAdapter(self.compiler, self.worker, self.evaluator)
        attributor = FailureAttributor()
        hard_constraints = self._control_hard_constraints()
        objectives = self._control_objectives()
        last_failure = None
        failure_rounds = 0
        last_promotable = False
        last_target_satisfied = False
        stop_reason = "max_rounds"
        for round_index in range(start_round, int(max_rounds) + 1):
            round_id = "R%04d" % round_index
            context = dict(self._context(round_index, (), (), ()))
            context.update(
                {
                    "parent_candidate_id": parent_id,
                    "parent_generation": parent_generation,
                    "parent_checkpoint_bound": bool(parent_checkpoint),
                    "fidelity_schedule": list(self.fidelity_schedule),
                }
            )
            decision_start = len(candidate_decisions)
            try:
                batch = self._control_batch(
                    context,
                    round_index=round_index,
                    parent_id=parent_id,
                    parent_generation=parent_generation,
                )
                trace.append(
                    "proposal.generated",
                    round_id=round_id,
                    experiment_id=batch.batch_id,
                    candidate_id=None,
                    parent_candidate_id=parent_id,
                    actor=self.campaign_base.controller_identity,
                    payload=batch.to_dict(),
                    evidence_ids=batch.parent_selection_evidence_ids,
                )
                report = validate_batch(
                    batch,
                    base=self.campaign_base,
                    snapshot=snapshot,
                    parent_ids={parent_id: parent_generation},
                    min_candidates=self.min_candidates,
                    max_candidates=self.max_candidates,
                )
                trace.append(
                    "proposal.validated",
                    round_id=round_id,
                    experiment_id=batch.batch_id,
                    candidate_id=None,
                    parent_candidate_id=parent_id,
                    actor=self.campaign_base.controller_identity,
                    payload={"ok": report.ok, "errors": list(report.errors), "candidate_errors": {key: list(value) for key, value in report.candidate_errors.items()}},
                    evidence_ids=batch.parent_selection_evidence_ids,
                )
                if not report.ok:
                    raise ValueError("proposal_invalid: " + "; ".join(report.errors or ("candidate validation failed",)))
            except (TypeError, ValueError, KeyError, OSError) as exc:
                failure_rounds += 1
                last_failure = "proposal_invalid"
                report = attributor.attribute("proposal", {"failure_code": "proposal_invalid", "message": str(exc)}, ())
                trace.append(
                    "campaign.replanned",
                    round_id=round_id,
                    experiment_id=None,
                    candidate_id=None,
                    parent_candidate_id=parent_id,
                    actor=self.campaign_base.controller_identity,
                    payload={"failure": report.to_dict(), "action": "repropose_batch"},
                    evidence_ids=(),
                )
                if failure_rounds > self.max_failures:
                    stop_reason = "failure_budget_exhausted"
                    break
                continue

            round_results = []
            round_evidence = {}
            for candidate in batch.candidates:
                if candidate.candidate_id in seen_candidates:
                    candidate_decisions.append({"candidate_id": candidate.candidate_id, "promotable": False, "target_satisfied": False, "failure_code": "duplicate_candidate"})
                    continue
                seen_candidates.add(candidate.candidate_id)
                try:
                    review = self.review_pipeline.review(
                        candidate,
                        {
                            "campaign_id": self.campaign_base.campaign_id,
                            "evidence_ids": list(batch.parent_selection_evidence_ids),
                            "hard_constraints": hard_constraints,
                            "objectives": objectives,
                            "diagnosis": batch.diagnosis,
                            "fidelity_schedule": list(self.fidelity_schedule),
                            "capability_snapshot_digest": snapshot.digest,
                        },
                    )
                    trace.append(
                        "critic.completed",
                        round_id=round_id,
                        experiment_id=candidate.experiment_id,
                        candidate_id=candidate.candidate_id,
                        parent_candidate_id=candidate.parent_candidate_id,
                        actor=self.campaign_base.critic_identity,
                        payload=review.to_dict(),
                        evidence_ids=review.advocate.supporting_evidence_ids,
                    )
                    reviewed_candidate = review.final_candidate or (
                        review.revision.candidate if review.revision is not None else candidate
                    )
                    if review.revision is not None:
                        trace.append(
                            "proposal.revised",
                            round_id=round_id,
                            experiment_id=reviewed_candidate.experiment_id,
                            candidate_id=reviewed_candidate.candidate_id,
                            parent_candidate_id=reviewed_candidate.parent_candidate_id,
                            actor=self.campaign_base.critic_identity,
                            payload=review.revision.to_dict(),
                            evidence_ids=review.revision.resolved_objection_ids,
                        )
                    if not review.approved:
                        decision = {
                            "candidate_id": candidate.candidate_id,
                            "parent_candidate_id": candidate.parent_candidate_id,
                            "promotable": False,
                            "target_satisfied": False,
                            "failure_code": "review_rejected",
                            "review": review.to_dict(),
                        }
                        candidate_decisions.append(decision)
                        continue
                    canonical_proposal = None
                    if reviewed_candidate.provenance.get("student_proposal") is not None:
                        canonical_proposal = self._revalidate_reviewed_student_candidate(
                            reviewed_candidate,
                            batch,
                            snapshot,
                            parent_id,
                            parent_generation,
                        )
                        trace.append(
                            "proposal.revalidated",
                            round_id=round_id,
                            experiment_id=reviewed_candidate.experiment_id,
                            candidate_id=reviewed_candidate.candidate_id,
                            parent_candidate_id=reviewed_candidate.parent_candidate_id,
                            actor=self.campaign_base.controller_identity,
                            payload={
                                "proposal_digest": canonical_proposal.digest,
                                "candidate_proposal_digest": reviewed_candidate.proposal_digest,
                                "validation": {
                                    "schema": True,
                                    "capability": True,
                                    "parent_generation": True,
                                    "compiler_pending": True,
                                },
                            },
                            evidence_ids=(),
                        )
                    validation = adapter.validate(
                        reviewed_candidate,
                        self._control_round_dir(round_index, reviewed_candidate.candidate_id, "validate"),
                    )
                    execution = None
                    public_execution = {}
                    training_obj = None
                    evaluation_obj = None
                    fidelity_parent_checkpoint = parent_checkpoint
                    fidelity_parent_candidate_id = parent_id
                    fidelity_history = []
                    fidelity_gate_decision = None
                    for fidelity in self.fidelity_schedule:
                        spec = stage_spec(self.max_steps, fidelity)
                        trace.append(
                            "training.started",
                            round_id=round_id,
                            experiment_id=reviewed_candidate.experiment_id,
                            candidate_id=reviewed_candidate.candidate_id,
                            parent_candidate_id=reviewed_candidate.parent_candidate_id,
                            actor=self.campaign_base.controller_identity,
                            payload={
                                "fidelity": fidelity,
                                "parent_checkpoint": fidelity_parent_checkpoint,
                                "parent_candidate_id": fidelity_parent_candidate_id,
                                "fidelity_spec": {
                                    "train_steps": spec.train_steps,
                                    "cumulative_train_steps": spec.cumulative_train_steps,
                                    "evaluation_cases": spec.evaluation_cases,
                                    "seed_count": spec.seed_count,
                                    "verifier_strength": spec.verifier_strength,
                                    "timeout_s": spec.timeout_s,
                                    "gpu_budget": spec.gpu_budget,
                                },
                                "validation": dict(validation),
                            },
                            evidence_ids=(str(validation.get("manifest_digest")),),
                        )
                        execution = adapter.execute(
                            reviewed_candidate,
                            fidelity,
                            self._control_round_dir(round_index, reviewed_candidate.candidate_id, fidelity),
                            train_steps=spec.train_steps,
                            parent_checkpoint=fidelity_parent_checkpoint,
                            parent_candidate_id=fidelity_parent_candidate_id,
                        )
                        public_execution = self._control_public_execution(execution)
                        training_obj = execution.get("_training_result")
                        evaluation_obj = execution.get("_evaluation")
                        stage_evidence = adapter.verify(
                            reviewed_candidate,
                            execution,
                            self._control_round_dir(round_index, reviewed_candidate.candidate_id, fidelity),
                        )
                        fidelity_gate_decision = fidelity_gate.evaluate(spec, stage_evidence)
                        fidelity_history.append(
                            {
                                "fidelity": fidelity,
                                "fidelity_spec": {
                                    "train_steps": spec.train_steps,
                                    "cumulative_train_steps": spec.cumulative_train_steps,
                                    "evaluation_cases": spec.evaluation_cases,
                                    "seed_count": spec.seed_count,
                                    "verifier_strength": spec.verifier_strength,
                                    "timeout_s": spec.timeout_s,
                                    "gpu_budget": spec.gpu_budget,
                                },
                                "parent_checkpoint": fidelity_parent_checkpoint,
                                "parent_candidate_id": fidelity_parent_candidate_id,
                                "training": dict(public_execution.get("training") or {}),
                                "evaluation": dict(public_execution.get("evaluation") or {}),
                                "gate": fidelity_gate_decision.to_dict(),
                            }
                        )
                        trace.append(
                            "fidelity.gate",
                            round_id=round_id,
                            experiment_id=reviewed_candidate.experiment_id,
                            candidate_id=reviewed_candidate.candidate_id,
                            parent_candidate_id=reviewed_candidate.parent_candidate_id,
                            actor=self.campaign_base.evaluator_identity,
                            payload=fidelity_gate_decision.to_dict(),
                            evidence_ids=tuple(item.input_reference for item in stage_evidence),
                        )
                        if training_obj is not None and getattr(training_obj, "status", "failed") == "success":
                            fidelity_parent_checkpoint = (
                                getattr(training_obj, "full_precision_checkpoint", None)
                                or getattr(training_obj, "child_checkpoint", None)
                            )
                            fidelity_parent_candidate_id = reviewed_candidate.candidate_id
                        trace.append(
                            "training.completed",
                            round_id=round_id,
                            experiment_id=reviewed_candidate.experiment_id,
                            candidate_id=reviewed_candidate.candidate_id,
                            parent_candidate_id=reviewed_candidate.parent_candidate_id,
                            actor=self.campaign_base.controller_identity,
                            payload={"fidelity": fidelity, **(public_execution.get("training") or {})},
                            evidence_ids=(str(public_execution.get("training", {}).get("child_checkpoint") or "training")),
                        )
                        if training_obj is None or getattr(training_obj, "status", "failed") != "success":
                            break
                        if not fidelity_gate_decision.passed:
                            break
                    if (
                        training_obj is not None
                        and getattr(training_obj, "status", "failed") == "success"
                        and (fidelity_gate_decision is None or not fidelity_gate_decision.passed)
                    ):
                        self._append_control_experience(
                            reviewed_candidate,
                            {item.metric_name: item for item in stage_evidence},
                            fidelity_gate_decision,
                            round_index=round_index,
                            training=training_obj.to_dict(),
                            baseline_metrics=self._control_parent_metrics,
                        )
                        failure = attributor.attribute(
                            "fidelity",
                            {
                                "failure_code": "fidelity_gate_failed",
                                "message": "fidelity %s failed: %s" % (
                                    fidelity_gate_decision.fidelity if fidelity_gate_decision else self.fidelity_schedule[0],
                                    ", ".join(fidelity_gate_decision.violations if fidelity_gate_decision else ("no_stage_result",)),
                                ),
                            },
                            tuple(fidelity_gate_decision.required_evidence if fidelity_gate_decision else ()),
                        )
                        decision = {
                            "candidate_id": reviewed_candidate.candidate_id,
                            "parent_candidate_id": reviewed_candidate.parent_candidate_id,
                            "promotable": False,
                            "target_satisfied": False,
                            "failure_code": failure.failure_code,
                            "failure": failure.to_dict(),
                            "fidelity_history": fidelity_history,
                            "review": review.to_dict(),
                        }
                        candidate_decisions.append(decision)
                        trace.append(
                            "campaign.replanned",
                            round_id=round_id,
                            experiment_id=reviewed_candidate.experiment_id,
                            candidate_id=reviewed_candidate.candidate_id,
                            parent_candidate_id=reviewed_candidate.parent_candidate_id,
                            actor=self.campaign_base.controller_identity,
                            payload={"failure": failure.to_dict(), "action": "retain_parent"},
                            evidence_ids=failure.evidence_ids,
                        )
                        continue
                    training_payload = public_execution.get("training") or {}
                    trace.append(
                        "training.metric",
                        round_id=round_id,
                        experiment_id=reviewed_candidate.experiment_id,
                        candidate_id=reviewed_candidate.candidate_id,
                        parent_candidate_id=reviewed_candidate.parent_candidate_id,
                        actor=self.campaign_base.evaluator_identity,
                        payload={
                            key: training_payload.get(key)
                            for key in ("optimizer_steps", "initial_loss", "final_loss", "gradient_norm", "peak_memory_gb")
                            if training_payload.get(key) is not None
                        },
                        evidence_ids=(str(public_execution.get("training", {}).get("compiler_digest") or "training")),
                    )
                    if training_obj is None or getattr(training_obj, "status", "failed") != "success":
                        failure = attributor.attribute("training", public_execution.get("training") or {"failure_code": "training_failed"}, ())
                        decision = {"candidate_id": reviewed_candidate.candidate_id, "promotable": False, "target_satisfied": False, "failure_code": failure.failure_code, "failure": failure.to_dict(), "review": review.to_dict()}
                        candidate_decisions.append(decision)
                        trace.append(
                            "campaign.replanned",
                            round_id=round_id,
                            experiment_id=reviewed_candidate.experiment_id,
                            candidate_id=reviewed_candidate.candidate_id,
                            parent_candidate_id=reviewed_candidate.parent_candidate_id,
                            actor=self.campaign_base.controller_identity,
                            payload={"failure": failure.to_dict(), "action": "retain_parent"},
                            evidence_ids=failure.evidence_ids,
                        )
                        continue
                    trace.append(
                        "evaluation.started",
                        round_id=round_id,
                        experiment_id=reviewed_candidate.experiment_id,
                        candidate_id=reviewed_candidate.candidate_id,
                        parent_candidate_id=reviewed_candidate.parent_candidate_id,
                        actor=self.campaign_base.evaluator_identity,
                        payload={"fidelity": public_execution.get("fidelity", self.fidelity_schedule[-1])},
                        evidence_ids=(),
                    )
                    trace.append(
                        "evaluation.completed",
                        round_id=round_id,
                        experiment_id=reviewed_candidate.experiment_id,
                        candidate_id=reviewed_candidate.candidate_id,
                        parent_candidate_id=reviewed_candidate.parent_candidate_id,
                        actor=self.campaign_base.evaluator_identity,
                        payload=public_execution.get("evaluation") or {},
                        evidence_ids=(str(public_execution.get("evaluation", {}).get("video_path") or "evaluation")),
                    )
                    evidence_items = list(
                        adapter.verify(
                            reviewed_candidate,
                            execution,
                            self._control_round_dir(round_index, reviewed_candidate.candidate_id, execution.get("fidelity", self.fidelity_schedule[-1])),
                        )
                    )
                    evaluation_valid = bool(getattr(evaluation_obj, "valid", False))
                    evaluation_promotable = bool(getattr(evaluation_obj, "promotable", False))
                    evidence_items.append(MetricEvidence("evaluation_promotable", "student-adapter-v1", str(self.output_root / round_id / candidate.candidate_id), 1.0 if evaluation_promotable else 0.0, evaluation_promotable, "student-evaluator", "server", True))
                    evidence_map = {item.metric_name: item for item in evidence_items}
                    if validation.get("graph_status") != "compiled":
                        evidence_map["graph_valid"] = MetricEvidence("graph_valid", "student-adapter-v1", str(self.output_root / round_id / candidate.candidate_id), 0.0, False, "student-compiler", "server", True)
                    else:
                        evidence_map["graph_valid"] = MetricEvidence("graph_valid", "student-adapter-v1", str(self.output_root / round_id / candidate.candidate_id), 1.0, True, "student-compiler", "server", True)
                    constraints = dict(hard_constraints)
                    constraints.setdefault("graph_valid", True)
                    decision_obj = gate.evaluate(
                        reviewed_candidate,
                        evidence_map,
                        hard_constraints=constraints,
                        objectives=objectives,
                        min_rounds_met=round_index >= self.min_rounds_before_success,
                    )
                    evaluation_failure = None
                    if not evaluation_valid or not evaluation_promotable:
                        evaluation_failure = attributor.attribute(
                            "evaluation",
                            {
                                "failure_code": getattr(evaluation_obj, "failure_code", None) or "metric_missing",
                                "message": getattr(evaluation_obj, "message", "evaluation did not produce promotable evidence"),
                            },
                            decision_obj.evidence_ids,
                        )
                    decision = {
                        "candidate_id": reviewed_candidate.candidate_id,
                        "parent_candidate_id": reviewed_candidate.parent_candidate_id,
                        "generation": reviewed_candidate.generation,
                        "promotable": decision_obj.promotable,
                        "target_satisfied": decision_obj.target_satisfied,
                        "objective_values": dict(decision_obj.objective_values),
                        "violations": list(decision_obj.violations),
                        "reason": decision_obj.reason,
                        "evidence": {key: value.to_dict() for key, value in evidence_map.items()},
                        "predicted_metric_delta": dict(reviewed_candidate.predicted_metric_delta),
                        "review": review.to_dict(),
                    }
                    if evaluation_failure is not None:
                        decision["failure_code"] = evaluation_failure.failure_code
                        decision["failure"] = evaluation_failure.to_dict()
                    decision["fidelity_history"] = fidelity_history
                    candidate_decisions.append(decision)
                    round_evidence[reviewed_candidate.candidate_id] = evidence_map
                    trace.append(
                        "gate.decided",
                        round_id=round_id,
                        experiment_id=reviewed_candidate.experiment_id,
                        candidate_id=reviewed_candidate.candidate_id,
                        parent_candidate_id=reviewed_candidate.parent_candidate_id,
                        actor=self.campaign_base.evaluator_identity,
                        payload={"decision": decision_obj.to_dict(), "candidate": reviewed_candidate.to_dict()},
                        evidence_ids=decision_obj.evidence_ids,
                    )
                    round_results.append((reviewed_candidate, decision_obj, training_obj, evaluation_obj, validation, evidence_map))
                except (TypeError, ValueError, KeyError, OSError) as exc:
                    failure = attributor.attribute("execution", {"failure_code": "campaign_error", "message": str(exc)}, ())
                    candidate_decisions.append({"candidate_id": candidate.candidate_id, "promotable": False, "target_satisfied": False, "failure_code": failure.failure_code, "failure": failure.to_dict()})
                    trace.append(
                        "campaign.replanned",
                        round_id=round_id,
                        experiment_id=candidate.experiment_id,
                        candidate_id=candidate.candidate_id,
                        parent_candidate_id=candidate.parent_candidate_id,
                        actor=self.campaign_base.controller_identity,
                        payload={"failure": failure.to_dict(), "action": "retain_parent"},
                        evidence_ids=(),
                    )

            # Apply the Pareto gate only after every candidate in the batch has
            # fixed verifier evidence. A merely feasible candidate is not
            # promotable when another feasible candidate dominates it.
            decision_by_id = {
                item.get("candidate_id"): item
                for item in candidate_decisions[decision_start:]
                if item.get("candidate_id")
            }
            updated_round_results = []
            feasible_ids = [
                item[0].candidate_id
                for item in round_results
                if decision_by_id.get(item[0].candidate_id, {}).get("promotable")
            ]
            for item in round_results:
                reviewed_candidate, decision_obj, training_obj, evaluation_obj, validation, evidence_map = item
                decision = decision_by_id.get(reviewed_candidate.candidate_id)
                if decision is None:
                    updated_round_results.append(item)
                    continue
                dominated_by = [
                    other_id
                    for other_id in feasible_ids
                    if other_id != reviewed_candidate.candidate_id
                    and pareto_dominates(round_evidence[other_id], evidence_map, objectives)
                ]
                decision["pareto_feasible"] = not dominated_by and bool(decision_obj.feasible)
                decision["pareto_dominated_by"] = dominated_by
                if dominated_by and decision.get("promotable"):
                    decision["promotable"] = False
                    decision["target_satisfied"] = False
                    decision["reason"] = "pareto_dominated"
                final_gate = replace(
                    decision_obj,
                    promotable=bool(decision.get("promotable")),
                    target_satisfied=bool(decision.get("target_satisfied")),
                    reason=str(decision.get("reason", decision_obj.reason)),
                )
                self._append_control_experience(
                    reviewed_candidate,
                    evidence_map,
                    final_gate,
                    round_index=round_index,
                    training=training_obj.to_dict() if training_obj is not None else None,
                    baseline_metrics=self._control_parent_metrics,
                )
                updated_round_results.append((reviewed_candidate, final_gate, training_obj, evaluation_obj, validation, evidence_map))
            round_results = updated_round_results

            candidate_by_id = {item.candidate_id: item for item in batch.candidates}
            for decision in candidate_decisions[decision_start:]:
                archived_candidate = candidate_by_id.get(decision.get("candidate_id"))
                novelty = self._append_control_archive(archived_candidate, decision)
                trace.append(
                    "archive.updated",
                    round_id=round_id,
                    experiment_id=None,
                    candidate_id=decision.get("candidate_id"),
                    parent_candidate_id=archived_candidate.parent_candidate_id if archived_candidate else None,
                    actor=self.campaign_base.controller_identity,
                    payload={
                        "archive_kinds": ["pareto" if decision.get("promotable") else "failure", "novelty"],
                        "novelty": dict(novelty),
                        "pareto_feasible": bool(decision.get("pareto_feasible", False)),
                        "pareto_dominated_by": list(decision.get("pareto_dominated_by") or ()),
                    },
                    evidence_ids=(),
                )

            best = self._control_best([item for item in candidate_decisions if item.get("candidate_id") in {value.candidate_id for value in batch.candidates} and item.get("promotable")])
            if best is not None:
                last_promotable = True
                selected = next(item for item in round_results if item[0].candidate_id == best["candidate_id"])
                selected_candidate, selected_gate, selected_training, selected_evaluation, selected_validation, _selected_evidence = selected
                parent_id = selected_candidate.candidate_id
                parent_generation = selected_candidate.generation
                parent_checkpoint = (
                    getattr(selected_training, "full_precision_checkpoint", None)
                    or getattr(selected_training, "child_checkpoint", None)
                )
                self.parent_checkpoint = parent_checkpoint
                self._control_parent_metrics = {
                    str(key): float(value) for key, value in selected_gate.objective_values.items()
                    if isinstance(value, (int, float))
                }
                trace.append(
                    "parent.selected",
                    round_id=round_id,
                    experiment_id=selected_candidate.experiment_id,
                    candidate_id=selected_candidate.candidate_id,
                    parent_candidate_id=selected_candidate.parent_candidate_id,
                    actor=self.campaign_base.controller_identity,
                    payload={
                        "candidate_id": parent_id,
                        "generation": parent_generation,
                        "reason": "best_feasible_candidate",
                        "parent_checkpoint": getattr(selected_training, "parent_checkpoint", None),
                        "inheritance_checkpoint": parent_checkpoint,
                        "child_checkpoint": getattr(selected_training, "child_checkpoint", None),
                        "evaluation_checkpoint": getattr(selected_training, "child_checkpoint", None),
                        "child_sha256": getattr(selected_training, "child_sha256", None),
                        "fidelity": getattr(selected_training, "fidelity", None),
                    },
                    evidence_ids=selected_gate.evidence_ids,
                )
                completed_rounds.append(CampaignRound(round_index, self._proposal(selected_candidate.provenance.get("student_proposal")), None, selected_training, selected_evaluation, None, selected_gate.reason, selected_candidate.candidate_id, selected_candidate.parent_candidate_id))
                last_target_satisfied = bool(best.get("target_satisfied"))
                if last_target_satisfied:
                    stop_reason = "target_satisfied"
                    break
            else:
                failure_rounds += 1
                last_failure = "candidate_gate_failed"
                trace.append(
                    "campaign.replanned",
                    round_id=round_id,
                    experiment_id=batch.batch_id,
                    candidate_id=None,
                    parent_candidate_id=parent_id,
                    actor=self.campaign_base.controller_identity,
                    payload={"action": "retain_parent_and_revise", "candidate_count": len(batch.candidates)},
                    evidence_ids=(),
                )
                if failure_rounds > self.max_failures:
                    stop_reason = "failure_budget_exhausted"
                    break
        trace.append(
            "campaign.stopped",
            round_id="R%04d" % min(int(max_rounds), max(1, start_round + len(completed_rounds))),
            experiment_id=None,
            candidate_id=parent_id if parent_id != "M0000" else None,
            parent_candidate_id=None,
            actor=self.campaign_base.controller_identity,
            payload={"reason": stop_reason, "promotable": last_promotable, "target_satisfied": last_target_satisfied},
            evidence_ids=(),
        )
        terminal_status = (
            "TARGET_SATISFIED" if last_target_satisfied else
            "PROMOTABLE" if last_promotable else
            "BUDGET_EXHAUSTED" if stop_reason in {"failure_budget_exhausted", "max_rounds"} else
            "NO_PROGRESS"
        )
        return CampaignResult(
            terminal_status,
            len(completed_rounds),
            tuple(completed_rounds),
            last_failure,
            stop_reason,
            last_promotable,
            last_target_satisfied,
            tuple(dict(item) for item in candidate_decisions),
            stop_reason,
            str(self.decision_trace_path),
        )

    @staticmethod
    def _proposal(value: Any) -> Optional[StudentProposal]:
        if value is None:
            return None
        return value if isinstance(value, StudentProposal) else StudentProposal.from_dict(value)

    def _revalidate_reviewed_student_candidate(
        self,
        candidate: Any,
        batch: Any,
        snapshot: Any,
        parent_id: str,
        parent_generation: int,
    ) -> StudentProposal:
        from ..campaign.proposals import ProposalBatch, validate_batch
        from ..campaign.revision import proposal_from_candidate

        try:
            canonical_proposal = proposal_from_candidate(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError("revision_integrity_failure: %s" % exc) from exc
        proposal_report = canonical_proposal.validate(self.target)
        if not proposal_report.ok:
            raise ValueError("revision_proposal_invalid: %s" % "; ".join(proposal_report.errors))
        post_review_batch = ProposalBatch(
            batch_id=batch.batch_id + ":reviewed",
            round_id=batch.round_id,
            diagnosis=batch.diagnosis,
            parent_selection_evidence_ids=batch.parent_selection_evidence_ids,
            candidates=(candidate,),
        )
        post_review_report = validate_batch(
            post_review_batch,
            base=self.campaign_base,
            snapshot=snapshot,
            parent_ids={parent_id: parent_generation},
            min_candidates=1,
            max_candidates=1,
        )
        if not post_review_report.ok:
            raise ValueError(
                "revision_candidate_invalid: %s"
                % "; ".join(post_review_report.errors or (str(post_review_report.candidate_errors),))
            )
        return canonical_proposal

    def _compile_round(self, proposal: StudentProposal, round_dir: Path) -> CompileManifest:
        compile_dir = round_dir / "compile"
        manifest_path = compile_dir / "compile_manifest.json"
        if manifest_path.is_file():
            try:
                existing = CompileManifest.from_path(manifest_path)
            except CompileError:
                existing = None
            if existing is None or existing.proposal_digest != proposal.digest:
                # A previous crashed/restarted attempt may have used the same
                # numeric round for another proposal. Never overwrite it.
                compile_dir = round_dir / ("compile-" + proposal.digest[:12])
        return self.compiler.compile(proposal, compile_dir)

    def run(self, *, max_rounds: int) -> CampaignResult:
        if int(max_rounds) <= 0:
            raise ValueError("max_rounds must be positive")
        if self.campaign_base is not None:
            from ..campaign.events import TraceIntegrityError

            try:
                return self._run_control_plane(max_rounds=int(max_rounds))
            except TraceIntegrityError as exc:
                return CampaignResult(
                    "INTEGRITY_FAILURE",
                    0,
                    (),
                    "trace_integrity_failure",
                    str(exc),
                    False,
                    False,
                    (),
                    "integrity_failure",
                    str(self.decision_trace_path),
                )
            except Exception as exc:
                return CampaignResult(
                    "INFRA_FAILURE",
                    0,
                    (),
                    "campaign_infrastructure_failure",
                    str(exc),
                    False,
                    False,
                    (),
                    "infrastructure_failure",
                    str(self.decision_trace_path),
                )
        self.output_root.mkdir(parents=True, exist_ok=True)
        start_round, persisted_failures, persisted_seen = self._load_resume()
        failures: list[Mapping[str, Any]] = persisted_failures
        seen: list[str] = persisted_seen
        rounds: list[CampaignRound] = []
        failure_count = 0
        last_failure: Optional[str] = None
        prior_rounds: list[Mapping[str, Any]] = []
        for round_index in range(start_round, int(max_rounds) + 1):
            context = self._context(round_index, failures, seen, prior_rounds)
            self._persist_resume(round_index, "running", failures, seen)
            proposal = None
            compile_manifest = None
            training = None
            evaluation = None
            failure_code = None
            message = ""
            round_dir = self.output_root / ("student_%04d" % round_index)
            try:
                raw = self.provider.propose(context)
                proposal = raw if isinstance(raw, StudentProposal) else StudentProposal.from_dict(raw)
                report = proposal.validate(self.target)
                if not report.ok:
                    raise ValueError("proposal_invalid: %s" % "; ".join(report.errors))
                if proposal.digest in seen:
                    failure_code = "duplicate_proposal"
                    message = "proposal digest already appeared in this campaign"
                else:
                    seen.append(proposal.digest)
                    compile_manifest = self._compile_round(proposal, round_dir)
                    worker_resume: dict[str, Any] = {
                        "next_round": round_index,
                        "status": "worker_running",
                        "manifest": compile_manifest.to_dict(),
                        "failures": list(failures[-16:]),
                        "seen_proposal_digests": list(seen[-16:]),
                    }
                    if self.quality_policy is not None:
                        worker_resume.update(
                            {
                                "teacher_baseline": dict(self.teacher_baseline),
                                "incumbent": dict(self._incumbent or {}),
                                "frontier": [dict(item) for item in self._frontier],
                                "best_reward": self._best_reward,
                                "no_improvement_rounds": self._no_improvement_rounds,
                                "improvement_count": self._improvement_count,
                            }
                        )
                    _atomic_json(self.resume_path, worker_resume)
                    training = self.worker.run(compile_manifest, round_dir)
                    if training.status != "success":
                        failure_code = training.failure_code or "training_failed"
                        message = training.message
                    else:
                        if not training.child_checkpoint:
                            failure_code = "checkpoint_missing"
                            message = "worker reported success without a child checkpoint"
                        else:
                            evaluation = self.evaluator.evaluate(Path(training.child_checkpoint), round_dir)
                            evaluation, policy_decision, policy_candidate = self._apply_quality_policy(evaluation)
                            if self.strict_optimization and evaluation.valid and policy_decision is None:
                                self._no_improvement_rounds += 1
                            failure_code = evaluation.failure_code
                            message = evaluation.message
                            outcome = "accepted" if evaluation.promotable else "rejected"
                            training_dict = training.to_dict()
                            experience = make_experience_record(
                                experience_id="student-exp-%04d" % round_index,
                                proposal_digest=proposal.digest,
                                compiler_digest=compile_manifest.manifest_digest,
                                teacher_sha256=training.parent_sha256 or ("0" * 64),
                                parent_checkpoint=None,
                                child_checkpoint=training.child_checkpoint,
                                training=training_dict,
                                evaluation=evaluation,
                                outcome=outcome,
                                diagnosis=message,
                                next_round_hints=(failure_code or "evaluation_failed",),
                            )
                            append_experience(self.experience_path, experience)
                            if self.retention_handler is not None:
                                self.retention_handler(training, evaluation, "student_%04d" % round_index, outcome)
                            else:
                                decision = retain_after_evaluation(
                                    Path(training.child_checkpoint),
                                    "student_%04d" % round_index,
                                    outcome=outcome,
                                    protected=(),
                                )
                                if outcome != "accepted":
                                    apply_retention(decision)
                            if evaluation.promotable:
                                if self.strict_optimization and policy_decision is not None and policy_candidate is not None:
                                    self._update_quality_state(policy_candidate, policy_decision)
                                rounds.append(CampaignRound(round_index, proposal, compile_manifest, training, evaluation, None, message))
                                prior_rounds.append(
                                    {
                                        "round": round_index,
                                        "status": "accepted",
                                        "proposal_digest": proposal.digest,
                                        "training": {
                                            "optimizer_steps": training.optimizer_steps,
                                            "final_loss": training.final_loss,
                                            "peak_memory_gb": training.peak_memory_gb,
                                        },
                                        "evaluation": {
                                            "quality_score": evaluation.quality_score,
                                            "reward": evaluation.reward,
                                            "failure_code": evaluation.failure_code,
                                            "message": evaluation.message,
                                            "metrics": {
                                                name: value.get("value")
                                                for name, value in evaluation.metric_evidence.get("metrics", {}).items()
                                                if isinstance(value, Mapping) and value.get("value") is not None
                                            },
                                        },
                                    }
                                )
                                if self.strict_optimization:
                                    self._persist_resume(round_index + 1, "running", failures, seen)
                                    if self._no_improvement_rounds >= self.quality_policy.no_improvement_patience:
                                        status = "success" if self._improvement_count > 0 else "no_pareto_improvement"
                                        stop_reason = "no_improvement_patience"
                                        self._append_event({"round": round_index, "status": status, "failure_code": None if status == "success" else "no_pareto_improvement", "proposal_digest": proposal.digest, "llm_context": context, "optimization": {"incumbent": self._incumbent, "frontier": self._frontier}})
                                        self._persist_resume(round_index + 1, status, failures, seen)
                                        terminal = "PROMOTABLE" if status == "success" else "NO_PROGRESS"
                                        return CampaignResult(
                                            terminal,
                                            len(rounds),
                                            tuple(rounds),
                                            None if status == "success" else "no_pareto_improvement",
                                            "effective optimization stopped after no-improvement patience",
                                            terminal == "PROMOTABLE",
                                            False,
                                            stop_reason=stop_reason,
                                        )
                                    self._append_event({"round": round_index, "status": "accepted_intermediate", "failure_code": None, "proposal_digest": proposal.digest, "llm_context": context, "optimization": {"incumbent": self._incumbent, "frontier": self._frontier}})
                                    continue
                                if round_index >= self.min_rounds_before_success:
                                    self._append_event({"round": round_index, "status": "success", "failure_code": None, "proposal_digest": proposal.digest, "llm_context": context})
                                    self._persist_resume(round_index + 1, "success", failures, seen)
                                    return CampaignResult("PROMOTABLE", round_index, tuple(rounds), None, "server evidence is promotable; edge evidence is still required", True, False)
                                self._append_event({"round": round_index, "status": "accepted_intermediate", "failure_code": None, "proposal_digest": proposal.digest, "llm_context": context})
                                self._persist_resume(round_index + 1, "running", failures, seen)
                                continue
                            if self.strict_optimization and evaluation.valid:
                                failures.append(
                                    {
                                        "round": round_index,
                                        "failure_code": failure_code or "pareto_rejected",
                                        "message": message,
                                        "type": "optimization_rejected",
                                        "teacher_relative": evaluation.metric_evidence.get("teacher_relative", {}),
                                        "optimization_decision": evaluation.metric_evidence.get("optimization_decision", {}),
                                    }
                                )
                                rounds.append(self._round_failure(round_index, proposal, compile_manifest, training, evaluation, failure_code or "pareto_rejected", message))
                                self._append_event({"round": round_index, "status": "rejected", "failure_code": failure_code or "pareto_rejected", "message": message, "proposal_digest": proposal.digest, "llm_context": context, "optimization": {"incumbent": self._incumbent, "frontier": self._frontier}})
                                if self._no_improvement_rounds >= self.quality_policy.no_improvement_patience:
                                    status = "success" if self._improvement_count > 0 else "no_pareto_improvement"
                                    stop_reason = "no_improvement_patience"
                                    self._persist_resume(round_index + 1, status, failures, seen)
                                    terminal = "PROMOTABLE" if status == "success" else "NO_PROGRESS"
                                    return CampaignResult(
                                        terminal,
                                        len(rounds),
                                        tuple(rounds),
                                        None if status == "success" else "no_pareto_improvement",
                                        "effective optimization stopped after no-improvement patience",
                                        terminal == "PROMOTABLE",
                                        False,
                                        stop_reason=stop_reason,
                                    )
                                self._persist_resume(round_index + 1, "running", failures, seen)
                                continue
            except (ValueError, CompileError, OSError, TypeError, KeyError) as exc:
                failure_code = "proposal_invalid" if str(exc).startswith("proposal_invalid") else "campaign_error"
                message = str(exc)
            if failure_code is None:
                failure_code = "campaign_error"
            last_failure = failure_code
            failure_count += 1
            failure_record = {"round": round_index, "failure_code": failure_code, "message": message}
            failures.append(failure_record)
            rounds.append(self._round_failure(round_index, proposal, compile_manifest, training, evaluation, failure_code, message))
            self._append_event({"round": round_index, "status": "failed", "failure_code": failure_code, "message": message, "proposal_digest": proposal.digest if proposal else None, "llm_context": context})
            self._persist_resume(round_index + 1, "running", failures, seen)
            if failure_count > self.max_failures:
                break
        if self.strict_optimization:
            status = "success" if self._improvement_count > 0 else "no_pareto_improvement"
            terminal = "PROMOTABLE" if status == "success" else "NO_PROGRESS"
            self._persist_resume(int(max_rounds) + 1, terminal, failures, seen)
            return CampaignResult(
                terminal,
                len(rounds),
                tuple(rounds),
                None if status == "success" else "no_pareto_improvement",
                "effective optimization reached the round budget",
                terminal == "PROMOTABLE",
                False,
                stop_reason="campaign_budget_exhausted",
            )
        status = "BUDGET_EXHAUSTED" if rounds else "INFRA_FAILURE"
        self._persist_resume(int(max_rounds) + 1, status, failures, seen)
        return CampaignResult(status, len(rounds), tuple(rounds), last_failure, "campaign did not reach an accepted student")


__all__ = [
    "CampaignResult",
    "CampaignRound",
    "OpenAICompatibleStudentProposalProvider",
    "OllamaStudentProposalProvider",
    "build_student_proposal_provider",
    "build_student_control_plane",
    "StudentCampaign",
    "StudentProposalBatchProvider",
    "StudentProposalProvider",
    "StudentRoundEvaluator",
    "StudentRoundWorker",
    "student_proposal_json_schema",
    "student_proposal_batch_json_schema",
]
