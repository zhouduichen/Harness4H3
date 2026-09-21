"""Adapters from existing Student/H3 execution records to campaign evidence."""

from __future__ import annotations

import copy
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

    def execute(self, candidate: CandidateEnvelope, fidelity: str, round_dir: Path) -> Mapping[str, Any]:
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

    def run_proposal(self, proposal: Any, round_dir: Path, *, fidelity: str = "F1") -> Mapping[str, Any]:
        parsed = self._proposal(proposal)
        round_dir = Path(round_dir)
        compile_manifest = self.compiler.compile(parsed, round_dir / "compile")
        training = self.worker.run(compile_manifest, round_dir)
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
                "_compile_manifest": compile_manifest,
                "_training_result": training,
                "_evaluation": None,
            }
        evaluation = self.evaluator.evaluate(Path(training.child_checkpoint), round_dir)
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

    def run_candidate(self, candidate: CandidateEnvelope, *, fidelity: str, round_dir: Path) -> Mapping[str, Any]:
        raw = candidate.provenance.get("student_proposal")
        if raw is None:
            raise ValueError("Student candidate provenance must contain student_proposal")
        return self.run_proposal(raw, round_dir, fidelity=fidelity)

    def validate(self, candidate: CandidateEnvelope, round_dir: Path) -> Mapping[str, Any]:
        parsed = self._proposal(candidate.provenance.get("student_proposal"))
        manifest = self.compiler.compile(parsed, Path(round_dir) / "compile")
        return {
            "graph_status": manifest.graph_status,
            "parameter_count": manifest.parameter_count,
            "manifest_digest": manifest.manifest_digest,
            "proposal_digest": manifest.proposal_digest,
        }

    def execute(self, candidate: CandidateEnvelope, fidelity: str, round_dir: Path) -> Mapping[str, Any]:
        return self.run_candidate(candidate, fidelity=fidelity, round_dir=round_dir)

    def verify(self, candidate: CandidateEnvelope, execution: Mapping[str, Any], round_dir: Path) -> Tuple[MetricEvidence, ...]:
        training = execution.get("training") or {}
        evaluation = execution.get("evaluation") or {}
        validity = execution.get("v3") or {}
        evidence = [
            MetricEvidence(
                "video_decodable", "student-adapter-v1", str(round_dir),
                1.0 if validity.get("video_decodable") else 0.0,
                bool(validity.get("video_decodable")), "student-evaluator", "server", True,
            )
        ]
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
