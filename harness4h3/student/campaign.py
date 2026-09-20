"""Resumable proposal→compile→train→evaluate→revise Student campaign."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .compiler import CompileError, CompileManifest, StudentCompiler
from .evaluator import StudentEvaluation, append_experience, make_experience_record
from .proposal import StudentProposal, StudentTarget
from .retention import apply_retention, retain_after_evaluation
from .worker import TrainingResult


class StudentProposalProvider(Protocol):
    provider_name: str
    model_name: str

    def propose(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class StudentRoundWorker(Protocol):
    def run(self, manifest: CompileManifest, round_dir: Path) -> TrainingResult:
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
            "method": {"type": "string", "enum": ["velocity_distill", "dmd2"]},
            "source_steps": integer,
            "target_steps": integer,
            "max_steps": integer,
            "learning_rate": positive_number,
            "critic_learning_rate": positive_number,
            "batch_size": integer,
        },
        "required": ["method", "source_steps", "target_steps", "max_steps", "learning_rate", "critic_learning_rate", "batch_size"],
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


class OllamaStudentProposalProvider:
    provider_name = "ollama"

    def __init__(self, model_name: str, target: StudentTarget, *, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 180.0):
        self.model_name = str(model_name)
        self.target = target
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)

    def propose(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        schema = student_proposal_json_schema(self.target)
        prompt = (
            "You are the autonomous Student architect. Return exactly one JSON StudentProposal. "
            "Do not emit code, shell commands, markdown, or explanations outside JSON. "
            "The Harness will reject invalid shapes, memory, or 1B-2B parameter counts. CONTEXT="
            + json.dumps(dict(context), ensure_ascii=False, sort_keys=True)
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("proposal_invalid: Ollama proposal request failed: %s" % exc) from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("proposal_invalid: Ollama response was not a JSON object")
        return dict(parsed)


@dataclass(frozen=True)
class CampaignRound:
    round_index: int
    proposal: Optional[StudentProposal]
    compile: Optional[CompileManifest]
    training: Optional[TrainingResult]
    evaluation: Optional[StudentEvaluation]
    failure_code: Optional[str]
    message: str


@dataclass(frozen=True)
class CampaignResult:
    status: str
    rounds_completed: int
    rounds: tuple[CampaignRound, ...]
    failure_code: Optional[str] = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rounds_completed": self.rounds_completed,
            "failure_code": self.failure_code,
            "message": self.message,
            "rounds": [
                {
                    "round_index": item.round_index,
                    "proposal": item.proposal.to_dict() if item.proposal else None,
                    "compile": item.compile.to_dict() if item.compile else None,
                    "training": item.training.to_dict() if item.training else None,
                    "evaluation": item.evaluation.to_dict() if item.evaluation else None,
                    "failure_code": item.failure_code,
                    "message": item.message,
                }
                for item in self.rounds
            ],
        }


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
        retention_handler: Optional[Callable[[TrainingResult, StudentEvaluation, str, str], None]] = None,
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
        self.retention_handler = retention_handler
        if self.max_failures < 0:
            raise ValueError("max_failures must be non-negative")
        self.events_path = self.output_root / "campaign-events.jsonl"
        self.resume_path = self.output_root / "resume.json"

    def _append_event(self, payload: Mapping[str, Any]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _context(self, round_index: int, failures: Sequence[Mapping[str, Any]], seen: Sequence[str]) -> Mapping[str, Any]:
        bounded_failures = [dict(item) for item in failures[-8:]]
        return {
            "goal": self.goal,
            "round": round_index,
            "target": asdict(self.target),
            "provider": {"name": self.provider.provider_name, "model": self.provider.model_name},
            "seen_proposal_digests": list(seen[-8:]),
            "failures": bounded_failures,
        }

    def _persist_resume(self, next_round: int, status: str, failures: Sequence[Mapping[str, Any]], seen: Sequence[str]) -> None:
        _atomic_json(
            self.resume_path,
            {"next_round": next_round, "status": status, "failures": list(failures[-16:]), "seen_proposal_digests": list(seen[-16:])},
        )

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

    def run(self, *, max_rounds: int) -> CampaignResult:
        if int(max_rounds) <= 0:
            raise ValueError("max_rounds must be positive")
        self.output_root.mkdir(parents=True, exist_ok=True)
        failures: list[Mapping[str, Any]] = []
        seen: list[str] = []
        rounds: list[CampaignRound] = []
        failure_count = 0
        last_failure: Optional[str] = None
        for round_index in range(1, int(max_rounds) + 1):
            context = self._context(round_index, failures, seen)
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
                    compile_manifest = self.compiler.compile(proposal, round_dir / "compile")
                    _atomic_json(self.resume_path, {"next_round": round_index, "status": "worker_running", "manifest": compile_manifest.to_dict()})
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
                                rounds.append(CampaignRound(round_index, proposal, compile_manifest, training, evaluation, None, message))
                                self._append_event({"round": round_index, "status": "success", "failure_code": None, "proposal_digest": proposal.digest, "llm_context": context})
                                self._persist_resume(round_index + 1, "success", failures, seen)
                                return CampaignResult("success", round_index, tuple(rounds), None, "student accepted")
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
        status = "failed" if rounds else "no_rounds"
        self._persist_resume(int(max_rounds) + 1, status, failures, seen)
        return CampaignResult(status, len(rounds), tuple(rounds), last_failure, "campaign did not reach an accepted student")


__all__ = [
    "CampaignResult",
    "CampaignRound",
    "OllamaStudentProposalProvider",
    "StudentCampaign",
    "StudentProposalProvider",
    "StudentRoundEvaluator",
    "StudentRoundWorker",
    "student_proposal_json_schema",
]
