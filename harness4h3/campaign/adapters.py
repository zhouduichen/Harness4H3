"""Adapters from existing Student/H3 execution records to campaign evidence."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from ..student.compiler import CompileManifest
from ..student.evaluator import StudentEvaluation
from ..student.proposal import StudentProposal
from ..student.worker import TrainingResult
from .base import ActorIdentity, CampaignBase, canonical_digest
from .events import DecisionTrace
from .gates import MetricEvidence
from .proposals import CandidateEnvelope


class CandidateExecutor(Protocol):
    def validate(self, candidate: CandidateEnvelope, round_dir: Path) -> Mapping[str, Any]:
        ...

    def execute(
        self,
        candidate: CandidateEnvelope,
        fidelity: str,
        round_dir: Path,
        train_steps: int | None = None,
        *,
        parent_checkpoint: str | Path | None = None,
        parent_candidate_id: str | None = None,
    ) -> Mapping[str, Any]:
        ...

    def verify(
        self, candidate: CandidateEnvelope, execution: Mapping[str, Any], round_dir: Path
    ) -> Tuple[MetricEvidence, ...]:
        ...


class StudentCampaignAdapter:
    """Keep Student compiler/worker/evaluator outside the control plane."""

    def __init__(self, compiler: Any, worker: Any, evaluator: Any):
        self.compiler = compiler
        self.worker = worker
        self.evaluator = evaluator

    @staticmethod
    def _proposal(value: Any) -> StudentProposal:
        if isinstance(value, StudentProposal):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("student proposal must be a mapping")
        return StudentProposal.from_dict(value)

    def _run_worker(
        self,
        compile_manifest: CompileManifest,
        round_dir: Path,
        *,
        fidelity: str,
        train_steps: int | None = None,
        parent_checkpoint: str | Path | None,
        parent_candidate_id: str | None,
    ) -> TrainingResult:
        """Call old test doubles and the new parent-aware worker safely."""

        run = self.worker.run
        parameters = inspect.signature(run).parameters
        kwargs = {}
        if "fidelity" in parameters:
            kwargs["fidelity"] = fidelity
        if "train_steps" in parameters and train_steps is not None:
            kwargs["train_steps"] = int(train_steps)
        if "parent_checkpoint" in parameters:
            kwargs["parent_checkpoint"] = parent_checkpoint
        if "parent_candidate_id" in parameters:
            kwargs["parent_candidate_id"] = parent_candidate_id
        return run(compile_manifest, round_dir, **kwargs)

    def _run_evaluator(
        self,
        checkpoint: Path,
        round_dir: Path,
        *,
        fidelity: str,
        training: TrainingResult,
    ) -> StudentEvaluation:
        evaluate = self.evaluator.evaluate
        parameters = inspect.signature(evaluate).parameters
        kwargs = {}
        optional = {
            "fidelity": fidelity,
            "evaluation_cases": int(training.evaluation_cases) if training.evaluation_cases else None,
            "seed_count": int(training.seed_count) if training.seed_count else None,
            "verifier_strength": training.verifier_strength or None,
            "timeout_s": float(training.timeout_s) if training.timeout_s else None,
        }
        for name, value in optional.items():
            if name in parameters and value is not None:
                kwargs[name] = value
        return evaluate(checkpoint, round_dir, **kwargs)

    def run_proposal(
        self,
        proposal: Any,
        round_dir: Path,
        *,
        fidelity: str = "F1",
        train_steps: int | None = None,
        parent_checkpoint: str | Path | None = None,
        parent_candidate_id: str | None = None,
    ) -> Mapping[str, Any]:
        parsed = self._proposal(proposal)
        round_dir = Path(round_dir)
        compile_manifest = self.compiler.compile(parsed, round_dir / "compile")
        training = self._run_worker(
            compile_manifest,
            round_dir,
            fidelity=fidelity,
            train_steps=train_steps,
            parent_checkpoint=parent_checkpoint,
            parent_candidate_id=parent_candidate_id,
        )
        if not isinstance(training, TrainingResult):
            raise ValueError("Student worker must return TrainingResult")
        if training.status != "success" or not training.child_checkpoint:
            return {
                "fidelity": fidelity,
                "proposal": parsed.to_dict(),
                "compile": compile_manifest.to_dict(),
                "training": training.to_dict(),
                "evaluation": None,
                "v1": {"graph_status": compile_manifest.graph_status, "parameter_count": compile_manifest.parameter_count},
                "v2": training.to_dict(),
                "v3": {"video_decodable": False},
                "parent": {
                    "candidate_id": parent_candidate_id,
                    "checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
                    "fidelity": fidelity,
                },
                "_compile_manifest": compile_manifest,
                "_training_result": training,
                "_evaluation": None,
            }
        evaluation = self._run_evaluator(
            Path(training.child_checkpoint), round_dir, fidelity=fidelity, training=training
        )
        return {
            "fidelity": fidelity,
            "proposal": parsed.to_dict(),
            "compile": compile_manifest.to_dict(),
            "training": training.to_dict(),
            "evaluation": evaluation.to_dict(),
            "v1": {"graph_status": compile_manifest.graph_status, "parameter_count": compile_manifest.parameter_count},
            "v2": training.to_dict(),
            "v3": {
                "video_decodable": bool(evaluation.valid),
                "validity": copy.deepcopy(dict(evaluation.validity)),
            },
            "parent": {
                "candidate_id": parent_candidate_id,
                "checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
                "inherited": bool(training.parent_inherited),
                "fidelity": fidelity,
            },
            "v4": {
                "quality_score": evaluation.quality_score,
                "quality_metrics": copy.deepcopy(dict(evaluation.quality_metrics)),
                "hardware": copy.deepcopy(dict(evaluation.hardware)),
                "metric_evidence": copy.deepcopy(dict(evaluation.metric_evidence)),
            },
            "_compile_manifest": compile_manifest,
            "_training_result": training,
            "_evaluation": evaluation,
        }

    def run_candidate(
        self,
        candidate: CandidateEnvelope,
        *,
        fidelity: str,
        round_dir: Path,
        train_steps: int | None = None,
        parent_checkpoint: str | Path | None = None,
        parent_candidate_id: str | None = None,
    ) -> Mapping[str, Any]:
        raw = candidate.provenance.get("student_proposal")
        if raw is None:
            raise ValueError("Student candidate provenance must contain student_proposal")
        return self.run_proposal(
            raw,
            round_dir,
            fidelity=fidelity,
            train_steps=train_steps,
            parent_checkpoint=parent_checkpoint,
            parent_candidate_id=parent_candidate_id,
        )

    def validate(self, candidate: CandidateEnvelope, round_dir: Path) -> Mapping[str, Any]:
        parsed = self._proposal(candidate.provenance.get("student_proposal"))
        manifest = self.compiler.compile(parsed, Path(round_dir) / "compile")
        return {
            "graph_status": manifest.graph_status,
            "parameter_count": manifest.parameter_count,
            "manifest_digest": manifest.manifest_digest,
            "proposal_digest": manifest.proposal_digest,
        }

    def execute(
        self,
        candidate: CandidateEnvelope,
        fidelity: str,
        round_dir: Path,
        train_steps: int | None = None,
        *,
        parent_checkpoint: str | Path | None = None,
        parent_candidate_id: str | None = None,
    ) -> Mapping[str, Any]:
        return self.run_candidate(
            candidate,
            fidelity=fidelity,
            train_steps=train_steps,
            round_dir=round_dir,
            parent_checkpoint=parent_checkpoint,
            parent_candidate_id=parent_candidate_id,
        )

    def verify(self, candidate: CandidateEnvelope, execution: Mapping[str, Any], round_dir: Path) -> Tuple[MetricEvidence, ...]:
        training = execution.get("training") or {}
        evaluation = execution.get("evaluation") or {}
        validity = execution.get("v3") or {}
        fidelity = str(execution.get("fidelity") or "")
        algorithm_dispatch = str(training.get("algorithm_dispatch") or "")
        # Legacy test doubles predate the explicit dispatch field.  Real
        # workers always persist it; infer only the conservative success case
        # so old fixtures remain readable without weakening a failed result.
        if algorithm_dispatch in {"", "not_started"} and training.get("status") == "success" and int(training.get("optimizer_steps", 0) or 0) > 0:
            algorithm_dispatch = "executed"
        quality_metrics = evaluation.get("quality_metrics") or {}
        semantic_verified = (
            quality_metrics.get("semantic") is not None
            and str(quality_metrics.get("score_type", "")) == "clip_temporal"
        )
        evidence = [
            MetricEvidence(
                "video_decodable", "student-adapter-v1", str(round_dir),
                1.0 if validity.get("video_decodable") else 0.0,
                bool(validity.get("video_decodable")), "student-evaluator", "server", True,
            )
        ]
        evidence.extend(
            (
                MetricEvidence(
                    "algorithm_dispatch", "student-adapter-v1", str(round_dir),
                    1.0 if algorithm_dispatch == "executed" else 0.0,
                    algorithm_dispatch == "executed", "student-worker", "server", True,
                ),
                MetricEvidence(
                    "parent_checkpoint_bound", "student-adapter-v1", str(round_dir),
                    1.0 if (training.get("parent_sha256") or training.get("initialization_mode") == "fresh_init") else 0.0,
                    bool(training.get("parent_sha256") or training.get("initialization_mode") == "fresh_init"),
                    "student-worker", "server", True,
                ),
                MetricEvidence(
                    "parent_inherited", "student-adapter-v1", str(round_dir),
                    1.0 if training.get("parent_inherited") else 0.0,
                    bool(training.get("parent_inherited")), "student-worker", "server", False,
                ),
                MetricEvidence(
                    "inherited_parameter_count", "student-adapter-v2", str(round_dir),
                    float(training.get("inherited_parameter_count", 0)),
                    training.get("inherited_parameter_count") is not None, "student-worker", "server", False,
                ),
                MetricEvidence(
                    "total_parameter_count", "student-adapter-v2", str(round_dir),
                    float(training.get("total_parameter_count", 0)),
                    training.get("total_parameter_count") is not None, "student-worker", "server", False,
                ),
                MetricEvidence(
                    "inheritance_ratio", "student-adapter-v2", str(round_dir),
                    float(training.get("inheritance_ratio", 0.0)),
                    training.get("inheritance_ratio") is not None, "student-worker", "server", False,
                ),
                MetricEvidence(
                    "fidelity_executed", "student-adapter-v1", str(round_dir),
                    1.0 if fidelity else 0.0,
                    bool(fidelity), "student-worker", "server", True,
                ),
                MetricEvidence(
                    "semantic_verified", "student-adapter-v1", str(round_dir),
                    1.0 if semantic_verified else 0.0,
                    semantic_verified, "student-evaluator", "server", True,
                ),
            )
        )
        if training.get("peak_memory_gb") is not None:
            evidence.append(MetricEvidence(
                "peak_memory_gb", "student-adapter-v1", str(round_dir),
                float(training["peak_memory_gb"]), 1.0, "student-worker", "server", False,
            ))
        if evaluation.get("quality_score") is not None:
            evidence.append(MetricEvidence(
                "quality", "student-adapter-v1", str(round_dir),
                float(evaluation["quality_score"]), 1.0, "student-evaluator", "server", False,
            ))
        hardware = evaluation.get("hardware") or {}
        evidence.extend(
            (
                MetricEvidence(
                    "quality_verified", "student-adapter-v1", str(round_dir),
                    1.0 if evaluation.get("quality_score") is not None else 0.0,
                    evaluation.get("quality_score") is not None, "student-evaluator", "server", True,
                ),
                MetricEvidence(
                    "latency_verified", "student-adapter-v1", str(round_dir),
                    1.0 if hardware.get("latency_s") is not None else 0.0,
                    hardware.get("latency_s") is not None, "student-evaluator", "server", True,
                ),
                MetricEvidence(
                    "memory_verified", "student-adapter-v1", str(round_dir),
                    1.0 if (training.get("peak_memory_gb") is not None or hardware.get("peak_memory_gb") is not None) else 0.0,
                    training.get("peak_memory_gb") is not None or hardware.get("peak_memory_gb") is not None,
                    "student-evaluator", "server", True,
                ),
                MetricEvidence(
                    "model_size_verified", "student-adapter-v1", str(round_dir),
                    1.0 if (training.get("model_size_bytes") is not None or hardware.get("model_size_gb") is not None) else 0.0,
                    training.get("model_size_bytes") is not None or hardware.get("model_size_gb") is not None,
                    "student-evaluator", "server", True,
                ),
                MetricEvidence(
                    "promotable_for_edge_test", "student-adapter-v1", str(round_dir),
                    1.0 if evaluation.get("promotable") else 0.0,
                    bool(evaluation.get("promotable")), "student-evaluator", "server", False,
                ),
                MetricEvidence(
                    "edge_evidence_complete", "student-adapter-v1", str(round_dir),
                    0.0, False, "student-evaluator", "server", False,
                ),
            )
        )
        for metric_name in ("latency_s", "energy_j", "model_size_gb"):
            value = hardware.get(metric_name)
            if value is not None:
                evidence.append(MetricEvidence(
                    metric_name, "student-adapter-v1", str(round_dir), float(value), 1.0,
                    "student-evaluator", "server", False,
                ))
        model_size_bytes = training.get("model_size_bytes")
        if model_size_bytes is not None:
            evidence.append(MetricEvidence(
                "model_size_gb", "student-adapter-v1", str(round_dir), float(model_size_bytes) / float(1024 ** 3), 1.0,
                "student-worker", "server", False,
            ))
        return tuple(evidence)


class LegacyH3CampaignAdapter:
    """Identity/evidence adapter for the existing OperatorRegistry loop."""

    def __init__(self, registry: Any, output_root: Path, target: Any):
        self.registry = registry
        self.output_root = Path(output_root)
        self.target = target

    def build_base(self, session_id: str, controller: Any, evaluator: Any) -> CampaignBase:
        """Load one immutable base or create it before the legacy loop starts."""

        base_path = self.output_root / ("campaign-%s-base.json" % str(session_id))
        snapshot = self.capability_snapshot(self.registry, {})
        controller_identity = ActorIdentity(
            str(getattr(controller, "provider_name", "legacy-controller")),
            str(getattr(controller, "model_name", controller.__class__.__name__)),
            str(getattr(controller, "version", "legacy-v1")),
        )
        evaluator_identity = ActorIdentity(
            "legacy-evaluator",
            evaluator.__class__.__name__,
            str(getattr(evaluator, "version", "legacy-v1")),
        )
        critic_identity = ActorIdentity("legacy-critic", "fixed-critical-v1", "1")
        if critic_identity in {controller_identity, evaluator_identity}:
            critic_identity = ActorIdentity("legacy-critic", "fixed-critical-v2", "1")
        if base_path.is_file():
            base = CampaignBase.from_dict(json.loads(base_path.read_text(encoding="utf-8")))
            if base.target_profile_hash != canonical_digest(self.target.to_dict()):
                raise ValueError("campaign verification base changed; start a new legacy campaign")
            return base
        base = CampaignBase(
            schema_version=1,
            campaign_id="legacy-%s" % str(session_id),
            target_profile=self.target.to_dict(),
            target_profile_hash=canonical_digest(self.target.to_dict()),
            verifier_bank={"version": "legacy-composite-v1", "evaluator": evaluator.__class__.__name__},
            verifier_bank_hash=canonical_digest({"version": "legacy-composite-v1", "evaluator": evaluator.__class__.__name__}),
            dataset_manifest_hash=canonical_digest({"dataset": "legacy-evaluator-inputs"}),
            evaluation_recipe_hash=canonical_digest({"recipe": "legacy-composite-v1", "target": self.target.id}),
            controller_identity=controller_identity,
            critic_identity=critic_identity,
            evaluator_identity=evaluator_identity,
            prompt_version="legacy-controller-v1",
            capability_snapshot=snapshot.to_dict(),
        )
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(base.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        return base

    def trace(self, session_id: str, controller: Any, evaluator: Any) -> DecisionTrace:
        base = self.build_base(session_id, controller, evaluator)
        trace = DecisionTrace(self.output_root / ("campaign-%s-decision-trace.jsonl" % str(session_id)), base)
        if not trace.read():
            trace.append(
                "campaign.created",
                round_id=None,
                experiment_id=None,
                candidate_id=None,
                parent_candidate_id=None,
                actor=base.controller_identity,
                payload={"base_digest": base.digest, "adapter": "legacy-h3", "session_id": str(session_id)},
                evidence_ids=(),
            )
        return trace

    @staticmethod
    def capability_snapshot(registry: Any, backend_status: Optional[Mapping[str, Mapping[str, Any]]] = None):
        from .capabilities import CapabilityRegistry

        return CapabilityRegistry.from_operator_registry(registry, backend_status or {})

    @staticmethod
    def evidence_from_operator_result(result: Mapping[str, Any], experiment_id: str) -> Tuple[MetricEvidence, ...]:
        status = str(result.get("status", "failed"))
        return (
            MetricEvidence(
                "training_valid", "legacy-h3-adapter-v1", experiment_id,
                1.0 if status == "success" else 0.0,
                status == "success", "operator-result", "server", True,
            ),
        )


__all__ = ["CandidateExecutor", "LegacyH3CampaignAdapter", "StudentCampaignAdapter"]
