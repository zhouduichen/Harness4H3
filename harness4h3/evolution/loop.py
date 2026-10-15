from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact, CompressionResult, ModelStructure, WanModelAdapter
from ..archive.model_candidate import ModelCandidate
from ..archive.model_store import ModelStore
from ..h3.state import ModelState
from ..memory.design_gene import DesignGene
from ..memory.experience import ExperienceRecord, ExperienceStore
from ..profiler import CalibrationCorpus, CalibrationSampleSpec, FakeWanCalibrationProvider, ProfileReport, profile_model
from ..operators.wan import OperatorSpec, SearchSpaceManager
from ..sandbox.runner import AlgorithmSandbox, CompressionPlan as SandboxCompressionPlan, SandboxResult
from .agent import AlgorithmAgent
from .candidate import AlgorithmCandidate
from .design import build_design_record
from .evaluator import PromotionPolicy
from .improvement import AlgorithmValidationRecord, AlgorithmValidator
from .policy import SearchPolicy
from .screening import CandidateScreeningPipeline, ScreeningContext
from .skill_hub import SkillHub
from .store import AlgorithmStore
from ..operators.wan import WanCompressionRegistry, build_wan_compression_registry


EvaluationFn = Callable[[CheckpointArtifact, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class EvolutionRound:
    round_id: str
    parent_model_id: str
    algorithm_id: str
    status: str
    child_model_id: Optional[str]
    decision: Mapping[str, Any]
    paths: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvolutionLoop:
    """One bounded, restartable algorithm-generation/compression round."""

    def __init__(
        self,
        *,
        adapter: WanModelAdapter,
        parent: CheckpointArtifact,
        output_root: Path,
        model_store: Optional[ModelStore] = None,
        algorithm_store: Optional[AlgorithmStore] = None,
        experience_store: Optional[ExperienceStore] = None,
        skill_hub: Optional[SkillHub] = None,
        agent: Optional[AlgorithmAgent] = None,
        sandbox: Optional[AlgorithmSandbox] = None,
        registry: Optional[WanCompressionRegistry] = None,
        evaluator: Optional[EvaluationFn] = None,
        search_policy: Optional[SearchPolicy] = None,
        memory_enabled: bool = True,
        adapt_search_policy: bool = True,
        promotion_policy: Optional[PromotionPolicy] = None,
        advanced_profiling: bool = False,
        algorithm_improvement: bool = False,
        search_space_evolution: bool = False,
        candidate_screening: bool = False,
        search_space_manager: Optional[SearchSpaceManager] = None,
    ) -> None:
        self.adapter = adapter
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rounds_root = self.output_root / "rounds"
        self.rounds_root.mkdir(parents=True, exist_ok=True)
        self.model_store = model_store or ModelStore(self.output_root / "models")
        self.algorithm_store = algorithm_store or AlgorithmStore(self.output_root / "algorithms")
        self.experience_store = experience_store or ExperienceStore(self.output_root / "experiences.jsonl")
        self.skill_hub = skill_hub or SkillHub(self.experience_store.path, self.output_root / "design-genes.jsonl")
        self.agent = agent or AlgorithmAgent()
        self.sandbox = sandbox or AlgorithmSandbox(self.output_root / "sandbox")
        self.registry = registry or build_wan_compression_registry()
        self.evaluator = evaluator or self._default_evaluator
        self.search_policy = search_policy or self._load_search_policy()
        self.memory_enabled = bool(memory_enabled)
        self.adapt_search_policy = bool(adapt_search_policy)
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.advanced_profiling = bool(advanced_profiling)
        self.algorithm_improvement = bool(algorithm_improvement)
        self.search_space_evolution = bool(search_space_evolution)
        self.candidate_screening = bool(candidate_screening)
        self.search_space_manager: Optional[SearchSpaceManager] = search_space_manager
        self.search_space_version = None
        self.search_space_proposal = None
        self.pareto_path = self.output_root / "pareto-front.json"
        self.parent = parent
        self._initialize_evolution_features()
        self._initialize_parent()

    def _initialize_evolution_features(self) -> None:
        """Initialize additive features without changing legacy defaults."""

        if not self.search_space_evolution:
            return
        self.search_space_manager = self.search_space_manager or SearchSpaceManager(self.output_root / "search-space")
        self.search_space_version = self.search_space_manager.active
        # The first extension is deliberately an offline CPU demonstration.
        # Real Wan runs only consume an extension after an explicit validation
        # artifact exists; no new operator is silently trusted on GPU.
        if bool(self.parent.metadata.get("offline_simulation", False)) and "prune_ffn_by_weight_threshold" not in self.search_space_version.operator_names():
            spec = OperatorSpec(
                name="prune_ffn_by_weight_threshold",
                version="1",
                argument_schema={"block_ids": "list[str]", "threshold": "float"},
                granularity="channel",
                constraints=("wan_ffn_width_aligned", "threshold_must_change_width"),
                limitations=("weight_proxy_only", "not_real_wan_validated"),
                trusted_factory_key="prune_ffn_by_weight_threshold",
            )
            existing = next((item for item in self.search_space_manager.store.read_proposals() if item.spec.name == spec.name), None)
            if existing is None:
                existing = self.search_space_manager.propose(spec, "CPU Fake threshold proxy for a channel-importance plateau", ("offline-bootstrap",))
            if existing.status == "proposed":
                existing = self.search_space_manager.validate(
                    existing,
                    self.adapter,
                    self.parent,
                    output_dir=self.output_root / "search-space" / "validation" / existing.proposal_id,
                )
            self.search_space_proposal = existing
            if existing.status == "validated":
                self.search_space_version = self.search_space_manager.register(existing)
        self.registry = self.search_space_manager.registry_for(self.search_space_version, self.registry)
        self.sandbox.operator_specs.update(
            {
                item.name: item.to_dict()
                for item in self.search_space_version.operators
                if item.trusted_factory_key
            }
        )
        self._record_search_space_gene()

    def _record_search_space_gene(self) -> None:
        if self.search_space_version is None or self.skill_hub.genes is None:
            return
        proposal = self.search_space_proposal
        if proposal is None or proposal.status != "validated":
            return
        gene_id = "G-SS-" + proposal.proposal_id
        if any(item.gene_id == gene_id for item in self.skill_hub.genes.read()):
            return
        spec = proposal.spec
        validation = dict(proposal.validation)
        self.skill_hub.genes.append(
            DesignGene(
                gene_id=gene_id,
                status="accepted",
                state={
                    "architecture": self.parent.structure.architecture,
                    "search_space_version": self.search_space_version.version_id,
                    "offline_simulation": bool(validation.get("offline_simulation", True)),
                },
                bottleneck={"kind": "search_space_extension", "evidence_refs": list(proposal.evidence_ids)},
                intervention={"operator": spec.name, "strategy": "validated_search_space_extension", "argument_schema": spec.argument_schema},
                controlled_conditions={"proposal_id": proposal.proposal_id, "parent_version_id": proposal.parent_version_id},
                benefit=dict(validation.get("metrics") or {}),
                remaining_limitation={"reason": "not_real_wan_validated", "limitations": list(spec.limitations)},
                risks_lessons=["new execution capability is trusted only after interface, reload, and finite-value checks"],
                evidence={"proposal_id": proposal.proposal_id, "search_space_version": self.search_space_version.version_id, "validation": validation},
                confidence="partial",
            )
        )

    def _initialize_parent(self) -> None:
        if self.model_store.lineage():
            return
        self._verify_checkpoint(self.parent, expected_sha256=self.parent.sha256)
        structure = self.parent.structure
        state = ModelState(
            model_id="M0000",
            parent_model_id=None,
            checkpoint_path=self.parent.path,
            architecture_name=structure.architecture,
            parameter_count=structure.parameter_count,
            num_blocks=structure.block_count,
            hidden_size=structure.hidden_size,
            ffn_width=structure.ffn_width,
            components={"adapter": "wan"},
            measured_metrics={"offline_simulation": bool(self.parent.metadata.get("offline_simulation", False))},
            provenance={"checkpoint_sha256": self.parent.sha256, "offline_simulation": bool(self.parent.metadata.get("offline_simulation", False))},
        )
        self.model_store.initialize(ModelCandidate("M0000", None, 0, self.parent.path, state, None, "active", {"adapter": "wan"}))
        _write_json(self.output_root / "initial-baseline.json", self.parent.to_dict())

    def run_round(self) -> EvolutionRound:
        round_id, round_dir, resuming = self._acquire_round()
        round_state = self._read_json(round_dir / "round.json") if resuming else {}
        parent_model_id = str(round_state.get("parent_model_id", "")).strip()
        parent_candidate = self.model_store.get(parent_model_id) if parent_model_id else self.model_store.active()
        parent_artifact = self._artifact_for_candidate(parent_candidate)
        if not resuming:
            self._write_round_state(round_dir, {"round_id": round_id, "parent_model_id": parent_candidate.id, "status": "in_progress"})
        profile_path = round_dir / "profile.json"
        profile = self._load_or_profile(profile_path, parent_artifact)
        search_space_path = round_dir / "search-space.json"
        if self.search_space_version is not None and not search_space_path.exists():
            _write_json(search_space_path, self.search_space_version.to_dict())
        architecture = str(profile.model["architecture"])
        skills = self.skill_hub.search(architecture=architecture, limit=8) if self.memory_enabled else []
        pareto_front = self._load_pareto_front()
        selection_history = self._selection_history()
        # Local round history is not evaluator Memory.  It only lets a
        # deterministic baseline avoid proposing the same selection logic
        # after a rejected round, while Memory remains disabled for that
        # baseline.  Retrieved skills are the separate, optional evidence
        # channel used by memory-enabled runs.
        agent_history = [*selection_history, *skills]
        algorithm_path = round_dir / "algorithm.json"
        persisted_algorithm = self._read_json(algorithm_path) if algorithm_path.exists() else None
        algorithm_id = str(persisted_algorithm.get("algorithm_id", "")).strip() if persisted_algorithm else self.algorithm_store.next_id()
        # The initial parent already has a measured structural state.  Let the
        # policy decide whether to exploit it or open an exploration branch.
        search_choice = str(round_state.get("search_choice", "")).strip() or self.search_policy.choose(has_best=bool(parent_candidate.state.measured_metrics))
        # Algorithm lineage is independent from model lineage.  A rejected
        # child must not become the next model parent, but its algorithm is
        # still the version that the next algorithm should inherit from.
        latest_algorithm = self.algorithm_store.latest()
        parent_algorithm_id = latest_algorithm.algorithm_id if latest_algorithm is not None else None
        improvement_parent = latest_algorithm
        improved_algorithm = False
        if persisted_algorithm:
            algorithm = AlgorithmCandidate.from_dict(persisted_algorithm)
            if self.algorithm_improvement and algorithm.parent_algorithm_id and algorithm.parent_source_sha256:
                try:
                    improvement_parent = self.algorithm_store.get(algorithm.parent_algorithm_id)
                except ValueError:
                    improvement_parent = None
                improved_algorithm = improvement_parent is not None
            if not any(item.algorithm_id == algorithm.algorithm_id for item in self.algorithm_store.read()):
                if algorithm.parent_algorithm_id and algorithm.parent_source_sha256:
                    self.algorithm_store.create_child(algorithm, algorithm.parent_source_sha256)
                else:
                    self.algorithm_store.create(algorithm)
        else:
            try:
                if self.algorithm_improvement and improvement_parent is not None:
                    algorithm = self.agent.improve(
                        algorithm_id,
                        parent=improvement_parent,
                        parent_id=parent_candidate.id,
                        profile=profile.to_dict(),
                        history=agent_history,
                        failures=_history_failure_reasons(agent_history),
                        skills=skills,
                        search_space=self.search_space_version.to_dict() if self.search_space_version is not None else {},
                    )
                    improved_algorithm = True
                else:
                    algorithm = self.agent.generate(
                        algorithm_id,
                        parent_id=parent_candidate.id,
                        parent_algorithm_id=parent_algorithm_id,
                        profile=profile.to_dict(),
                        history=agent_history,
                        skills=skills,
                        pareto_front=pareto_front,
                        search_choice=search_choice,
                    )
            except (RuntimeError, TypeError, ValueError) as exc:
                algorithm = AlgorithmCandidate.from_source(
                    algorithm_id,
                    "# algorithm generation failed\ndef propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': []}}\n",
                    parent_id=parent_candidate.id,
                    parent_algorithm_id=parent_algorithm_id,
                    hypothesis="generation failed: %s" % exc,
                    metadata={"generator_failure": True},
                )
                _write_json(algorithm_path, algorithm.to_dict())
                return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, search_choice, "generation", "algorithm_generation_failed", str(exc), {"profile": str(profile_path), "algorithm": str(algorithm_path)}, algorithm_persisted=False)
            _write_json(algorithm_path, algorithm.to_dict())
            try:
                if improved_algorithm:
                    self.algorithm_store.create_child(algorithm, improvement_parent.source_sha256)  # type: ignore[union-attr]
                else:
                    self.algorithm_store.create(algorithm)
            except ValueError as exc:
                return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, search_choice, "generation", "duplicate_algorithm", str(exc), {"profile": str(profile_path), "algorithm": str(algorithm_path)}, algorithm_persisted=False)
        self._write_round_state(round_dir, {**round_state, "round_id": round_id, "parent_model_id": parent_candidate.id, "search_choice": search_choice, "algorithm_id": algorithm.algorithm_id, "status": "in_progress"})

        sandbox_path = round_dir / "sandbox.json"
        sandbox_result = self._load_sandbox(sandbox_path)
        if sandbox_result is None:
            sandbox_result = self.sandbox.run(algorithm.source, profile.to_dict(), skills)
            _write_json(sandbox_path, _sandbox_dict(sandbox_result))
        if sandbox_result.status != "success" or sandbox_result.plan is None:
            return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, search_choice, "sandbox", sandbox_result.failure_type or "sandbox_failed", sandbox_result.message, {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path)})

        validation_path = round_dir / "algorithm-validation.json"
        if improved_algorithm and improvement_parent is not None:
            validation = self._load_algorithm_validation(validation_path)
            if validation is None:
                validation = AlgorithmValidator().validate(
                    improvement_parent,
                    algorithm,
                    self.sandbox,
                    profile.to_dict(),
                    agent_history,
                    self._algorithm_proxy_evaluator,
                    budget={"sandbox_timeout_s": self.sandbox.timeout_s, "mode": "cpu_fake"},
                )
                _write_json(validation_path, validation.to_dict())
            algorithm = replace(
                algorithm,
                validation_status="validated" if validation.valid else "rejected",
                validation_record_path=str(validation_path),
            )
            self.algorithm_store.update(algorithm)
            _write_json(algorithm_path, algorithm.to_dict())

        design_path = round_dir / "design.json"
        design = self._read_json(design_path) if design_path.exists() else {}
        if not design:
            design = build_design_record(
                algorithm_id=algorithm.algorithm_id,
                parent_algorithm_id=algorithm.parent_algorithm_id,
                search_choice=search_choice,
                profile=profile.to_dict(),
                history=agent_history,
                plan=sandbox_result.plan.to_dict(),
                hypothesis=algorithm.hypothesis,
            )
            _write_json(design_path, design)

        screening_path = round_dir / "screening"
        if self.candidate_screening:
            screening = CandidateScreeningPipeline(screening_path)
            pre_results = screening.run(self._screening_context(parent_artifact, profile, algorithm, sandbox_result.plan))
            if pre_results and pre_results[-1].status != "passed":
                return self._finish_failure(
                    round_id,
                    round_dir,
                    parent_candidate,
                    algorithm,
                    search_choice,
                    "screening",
                    "candidate_screening_rejected",
                    "; ".join(pre_results[-1].reasons) or "candidate screening rejected",
                    {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "design": str(design_path), "screening": str(screening.path)},
                    design=design,
                )

        compression_dir = round_dir / "checkpoint"
        compression_path = round_dir / "compression.json"
        compression = self._load_compression(compression_path)
        if compression is None:
            compression = self.registry.execute(algorithm_plan_operator(sandbox_result), self.adapter, parent_artifact, sandbox_result.plan.args, compression_dir)
            _write_json(compression_path, compression.to_dict())
        if compression.status != "success" or compression.child_checkpoint is None:
            return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, search_choice, "compression", compression.failure_type or "compression_failed", compression.message, {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "design": str(design_path), "compression": str(compression_path)}, design=design)

        try:
            child_integrity = self._verify_checkpoint(compression.child_checkpoint, expected_sha256=compression.child_checkpoint.sha256, parent_sha256=parent_artifact.sha256)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            child_integrity = False
            integrity_error = str(exc)
        else:
            integrity_error = None
        evaluation_path = round_dir / "evaluation.json"
        evaluation = self._read_json(evaluation_path) if evaluation_path.exists() else None
        if evaluation is None:
            try:
                evaluation = dict(self.evaluator(compression.child_checkpoint, profile.to_dict()))
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                evaluation = {"valid": False, "evaluation_status": "failed", "failure_type": "evaluator_failed", "failure_message": str(exc), "offline_simulation": False}
        if not child_integrity:
            evaluation = {
                **evaluation,
                "valid": False,
                "evaluation_status": "child_integrity_failed",
                "failure_type": "child_integrity_failed",
                "failure_message": integrity_error or "child checkpoint is identical to parent",
            }
        if not evaluation_path.exists() or not child_integrity:
            _write_json(evaluation_path, evaluation)
        if self.candidate_screening:
            screening = CandidateScreeningPipeline(screening_path)
            full_results = screening.run(
                self._screening_context(
                    parent_artifact,
                    profile,
                    algorithm,
                    sandbox_result.plan,
                    full_evaluator=lambda _: dict(evaluation),
                )
            )
            full = next((item for item in full_results if item.stage == "full"), None)
            if full is not None and full.status == "passed":
                evaluation = dict(full.metrics)
                if not evaluation_path.exists() or not child_integrity:
                    _write_json(evaluation_path, evaluation)
        child_candidate = self._existing_or_create_child(parent_candidate, compression.child_checkpoint, algorithm, evaluation)
        evaluation_offline = bool(evaluation.get("offline_simulation", False))
        parent_offline = bool(parent_candidate.state.provenance.get("offline_simulation", False))
        offline_mismatch = evaluation_offline and not parent_offline
        promotion = self.promotion_policy.decide(
            parent=parent_candidate.state.measured_metrics,
            evaluation=evaluation,
            child_integrity=child_integrity,
            pareto_front=tuple(self._load_pareto_front()),
        )
        feasible = bool(promotion.get("accepted", False)) and not offline_mismatch and not bool(evaluation.get("critical_regression", False))
        decision_status = "promoted" if feasible else "rejected"
        previous_score = float(parent_candidate.state.measured_metrics.get("search_score", parent_candidate.state.measured_metrics.get("quality_score", 0.0)))
        current_score = float(evaluation.get("search_score", evaluation.get("quality_score", 0.0)))
        feedback_failure = str(evaluation.get("failure_type", "")) or None
        if not feasible and feedback_failure is None:
            reasons = promotion.get("reasons")
            if isinstance(reasons, list) and reasons:
                feedback_failure = str(reasons[0])
        self._update_search_policy(feasible=feasible, improved=current_score > previous_score, failure_type=feedback_failure)
        self._persist_search_policy()
        decision = {
            "status": decision_status,
            "parent_model_id": parent_candidate.id,
            "child_model_id": child_candidate.id,
            "algorithm_id": algorithm.algorithm_id,
            "operator": sandbox_result.plan.operator,
            "operator_args": dict(sandbox_result.plan.args),
            "algorithm_plan": sandbox_result.plan.to_dict(),
            "design": design,
            "design_fingerprint": design.get("fingerprint"),
            "feasible": feasible,
            "evaluation_offline": evaluation_offline,
            "offline_mismatch": offline_mismatch,
            "evaluation": evaluation,
            "promotion": dict(promotion),
            "pareto_front_before": self._load_pareto_front(),
            "search_policy": {"choice": search_choice, **self.search_policy.to_dict()},
        }
        if self.search_space_version is not None:
            decision["search_space_version"] = self.search_space_version.version_id
        if self.candidate_screening:
            decision["screening"] = [item.to_dict() for item in CandidateScreeningPipeline(screening_path).load()]
        decision_path = round_dir / "decision.json"
        _write_json(decision_path, decision)
        self._finalize_round_state(
            round_dir,
            round_id=round_id,
            parent_model_id=parent_candidate.id,
            algorithm_id=algorithm.algorithm_id,
            child_model_id=child_candidate.id,
            status=decision_status,
            decision_path=decision_path,
        )
        status = "accepted" if feasible else "rejected"
        self._record_experience(round_id, parent_candidate, child_candidate, algorithm, evaluation, decision, status, decision_path, design=design)
        self.algorithm_store.update(replace(algorithm, status=status, experiment_ids=(round_id,)))
        if feasible:
            self._update_pareto_front(child_candidate.id, evaluation)
            self.model_store.set_active(child_candidate.id)
        paths = {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "design": str(design_path), "compression": str(compression_path), "evaluation": str(evaluation_path), "decision": str(decision_path)}
        if self.search_space_version is not None:
            paths["search_space"] = str(search_space_path)
        if self.candidate_screening:
            paths["screening"] = str(screening_path / "screening.json")
        if validation_path.exists():
            paths["algorithm_validation"] = str(validation_path)
        return EvolutionRound(round_id, parent_candidate.id, algorithm.algorithm_id, decision_status, child_candidate.id, decision, paths)

    def _finish_failure(self, round_id: str, round_dir: Path, parent: ModelCandidate, algorithm: AlgorithmCandidate, search_choice: str, stage: str, failure_type: str, message: str, paths: Mapping[str, str], *, algorithm_persisted: bool = True, design: Optional[Mapping[str, Any]] = None) -> EvolutionRound:
        self._update_search_policy(feasible=False, improved=False, failure_type=failure_type)
        decision = {
            "status": "failed",
            "stage": stage,
            "failure_type": failure_type,
            "message": message,
            "parent_model_id": parent.id,
            "algorithm_id": algorithm.algorithm_id,
            "search_policy": {"choice": search_choice, **self.search_policy.to_dict()},
        }
        if design:
            decision["design"] = dict(design)
            decision["design_fingerprint"] = design.get("fingerprint")
        decision_path = round_dir / "decision.json"
        _write_json(decision_path, decision)
        self._finalize_round_state(
            round_dir,
            round_id=round_id,
            parent_model_id=parent.id,
            algorithm_id=algorithm.algorithm_id,
            status="failed",
            failure_type=failure_type,
            decision_path=decision_path,
        )
        self._record_experience(round_id, parent, None, algorithm, None, decision, "failed", decision_path, design=design)
        if algorithm_persisted:
            self.algorithm_store.update(replace(algorithm, status="failed", experiment_ids=(round_id,), failure_reasons=(failure_type,)))
        all_paths = dict(paths)
        all_paths["decision"] = str(decision_path)
        return EvolutionRound(round_id, parent.id, algorithm.algorithm_id, "failed", None, decision, all_paths)

    def _create_child(self, parent: ModelCandidate, artifact: CheckpointArtifact, algorithm: AlgorithmCandidate, evaluation: Mapping[str, Any]) -> ModelCandidate:
        structure = artifact.structure
        model_id = self.model_store.next_id()
        state = ModelState(
            model_id=model_id,
            parent_model_id=parent.id,
            checkpoint_path=artifact.path,
            architecture_name=structure.architecture,
            parameter_count=structure.parameter_count,
            num_blocks=structure.block_count,
            hidden_size=structure.hidden_size,
            ffn_width=structure.ffn_width,
            measured_metrics=dict(evaluation),
            provenance={"checkpoint_sha256": artifact.sha256, "algorithm_id": algorithm.algorithm_id, "algorithm_parent_id": algorithm.parent_algorithm_id, "parent_checkpoint_sha256": parent.state.provenance.get("checkpoint_sha256"), "offline_simulation": bool(artifact.metadata.get("offline_simulation", False))},
        )
        child = ModelCandidate(model_id, parent.id, parent.generation + 1, artifact.path, state, algorithm.algorithm_id, "evaluated", {"algorithm_id": algorithm.algorithm_id})
        self.model_store.create(child)
        return child

    def _record_experience(
        self,
        round_id: str,
        parent: ModelCandidate,
        child: Optional[ModelCandidate],
        algorithm: AlgorithmCandidate,
        evaluation: Optional[Mapping[str, Any]],
        decision: Mapping[str, Any],
        status: str,
        decision_path: Path,
        *,
        design: Optional[Mapping[str, Any]] = None,
    ) -> None:
        source = decision_path.read_bytes()
        training = {
            "architecture": parent.state.architecture_name,
            "algorithm_id": algorithm.algorithm_id,
            "algorithm_parent_id": algorithm.parent_algorithm_id,
            "hypothesis": algorithm.hypothesis,
            "predicted_gain": dict(algorithm.metadata.get("predicted_gain") or {}),
        }
        if design:
            training.update(
                {
                    "design_fingerprint": design.get("fingerprint"),
                    "design_strategy": design.get("strategy"),
                    "design_feedback": list(design.get("feedback") or []),
                    "design": dict(design),
                }
            )
        record = ExperienceRecord(
            experience_id=round_id,
            source_uri=str(decision_path.resolve()),
            source_sha256=hashlib.sha256(source).hexdigest(),
            source_kind="compression_round",
            experiment_id=round_id,
            parent_model_id=parent.id,
            child_model_id=child.id if child else None,
            operator=str(decision.get("operator")) if decision.get("operator") else None,
            operator_args=dict(decision.get("operator_args") or {}),
            training=training,
            evaluation=dict(evaluation) if evaluation is not None else None,
            decision=dict(decision),
            reward=float(evaluation.get("search_score", evaluation.get("quality_score", 0.0))) if evaluation else None,
            status=status,
            provenance={"architecture": parent.state.architecture_name, "offline_simulation": bool(parent.state.provenance.get("offline_simulation", False))},
            created_at="",
        )
        self.experience_store.append(record)
        if design:
            self._record_design_gene(round_id, parent, child, evaluation, decision, status, decision_path, design)

    def _record_design_gene(
        self,
        round_id: str,
        parent: ModelCandidate,
        child: Optional[ModelCandidate],
        evaluation: Optional[Mapping[str, Any]],
        decision: Mapping[str, Any],
        status: str,
        decision_path: Path,
        design: Mapping[str, Any],
    ) -> None:
        genes = self.skill_hub.genes
        if genes is None:
            return
        gene_id = "G-" + str(round_id)
        if any(item.gene_id == gene_id for item in genes.read()):
            return
        measured = dict(evaluation or {})
        benefit = {
            key: measured[key]
            for key in ("quality_score", "search_score", "hardware_metrics", "quality_metrics")
            if key in measured
        }
        promotion = decision.get("promotion")
        reasons = list(promotion.get("reasons") or []) if isinstance(promotion, Mapping) else []
        if not reasons and decision.get("failure_type"):
            reasons.append(str(decision["failure_type"]))
        first_reason = str(reasons[0]) if reasons else None
        plan = dict(design.get("plan") or {})
        gene = DesignGene(
            gene_id=gene_id,
            status=status,
            state={
                "architecture": parent.state.architecture_name,
                "parent_model_id": parent.id,
                "offline_simulation": bool(parent.state.provenance.get("offline_simulation", False)),
            },
            bottleneck={
                "kind": design.get("bottleneck", "unknown"),
                "evidence_refs": list(design.get("evidence_refs") or []),
            },
            intervention={
                "operator": plan.get("operator"),
                "steps": list((plan.get("args") or {}).get("steps") or []),
                "strategy": design.get("strategy"),
            },
            controlled_conditions={
                "search_choice": design.get("search_choice"),
                "evaluation_status": measured.get("evaluation_status"),
                "source_revision": measured.get("source_revision"),
            },
            benefit=benefit,
            remaining_limitation={"reason": first_reason, "reasons": reasons},
            risks_lessons=[str(item) for item in list(design.get("feedback") or []) + reasons],
            evidence={
                "architecture": parent.state.architecture_name,
                "round_id": round_id,
                "decision_path": str(decision_path),
                "design_fingerprint": design.get("fingerprint"),
                "parent_model_id": parent.id,
                "child_model_id": child.id if child else None,
            },
            confidence="measured" if measured.get("valid") is True else "partial",
            created_at="",
        )
        genes.append(gene)

    def _artifact_for_candidate(self, candidate: ModelCandidate) -> CheckpointArtifact:
        model = self.adapter.load_model(candidate.checkpoint_path)
        structure = self.adapter.inspect(model)
        digest = structure.checkpoint_sha256
        expected = str(candidate.state.provenance.get("checkpoint_sha256", digest))
        if digest != expected:
            raise ValueError("model_checkpoint_hash_mismatch: %s" % candidate.id)
        return CheckpointArtifact(candidate.checkpoint_path, digest, structure.architecture, structure, {"offline_simulation": bool(candidate.state.provenance.get("offline_simulation", False))})

    def _artifact_for_path(self, path: str, *, metadata: Optional[Mapping[str, Any]] = None) -> CheckpointArtifact:
        model = self.adapter.load_model(str(path))
        structure = self.adapter.inspect(model)
        return CheckpointArtifact(str(path), structure.checkpoint_sha256, structure.architecture, structure, dict(metadata or {}))

    def _existing_or_create_child(self, parent: ModelCandidate, artifact: CheckpointArtifact, algorithm: AlgorithmCandidate, evaluation: Mapping[str, Any]) -> ModelCandidate:
        for candidate in self.model_store.lineage():
            if candidate.parent_id == parent.id and candidate.created_by_experiment_id == algorithm.algorithm_id and candidate.checkpoint_path == artifact.path:
                return candidate
        return self._create_child(parent, artifact, algorithm, evaluation)

    def _verify_checkpoint(self, artifact: CheckpointArtifact, *, expected_sha256: str, parent_sha256: Optional[str] = None) -> bool:
        if parent_sha256 is not None and artifact.sha256 == parent_sha256:
            return False
        model = self.adapter.reload_model(artifact)
        structure = self.adapter.inspect(model)
        if structure.checkpoint_sha256 != expected_sha256 or structure.checkpoint_sha256 != artifact.sha256:
            raise ValueError("checkpoint_content_hash_mismatch")
        return True

    def _load_pareto_front(self) -> list[Mapping[str, Any]]:
        if not self.pareto_path.exists():
            return []
        try:
            raw = json.loads(self.pareto_path.read_text(encoding="utf-8"))
            return [dict(item) for item in raw.get("entries", []) if isinstance(item, Mapping)]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []

    def _update_pareto_front(self, candidate_id: str, evaluation: Mapping[str, Any]) -> None:
        entries = self._load_pareto_front()
        entries.append({"candidate_id": candidate_id, **dict(evaluation)})
        front = [
            item for item in entries
            if not any(
                other.get("candidate_id") != item.get("candidate_id")
                and PromotionPolicy._dominates(other, item)
                for other in entries
            )
        ]
        _write_json(self.pareto_path, {"entries": front})

    def _acquire_round(self) -> tuple[str, Path, bool]:
        """Reuse the oldest round without a decision before allocating a new one."""

        incomplete: list[tuple[int, Path]] = []
        for path in self.rounds_root.glob("R*"):
            if not path.is_dir() or (path / "decision.json").exists():
                continue
            try:
                incomplete.append((int(path.name[1:]), path))
            except ValueError:
                continue
        if incomplete:
            _, path = sorted(incomplete, key=lambda item: item[0])[0]
            return path.name, path, True
        round_id = self._next_round_id()
        round_dir = self.rounds_root / round_id
        round_dir.mkdir(parents=True, exist_ok=False)
        return round_id, round_dir, False

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}

    def _write_round_state(self, round_dir: Path, state: Mapping[str, Any]) -> None:
        _write_json(round_dir / "round.json", dict(state))

    def _finalize_round_state(
        self,
        round_dir: Path,
        *,
        round_id: str,
        parent_model_id: str,
        algorithm_id: str,
        status: str,
        decision_path: Path,
        child_model_id: Optional[str] = None,
        failure_type: Optional[str] = None,
    ) -> None:
        """Persist an auditable terminal state after the decision is durable.

        ``decision.json`` remains the source of the full decision contract, but
        the round marker must not remain ``in_progress`` after a recoverable
        round has reached a terminal outcome.  Reading the existing state keeps
        intermediate resume metadata such as ``search_choice`` intact.
        """

        state = self._read_json(round_dir / "round.json")
        state.update(
            {
                "round_id": round_id,
                "parent_model_id": parent_model_id,
                "algorithm_id": algorithm_id,
                "status": status,
                "decision_path": str(decision_path),
            }
        )
        if child_model_id is not None:
            state["child_model_id"] = child_model_id
        if failure_type is not None:
            state["failure_type"] = failure_type
        self._write_round_state(round_dir, state)

    def _load_or_profile(self, path: Path, parent: CheckpointArtifact) -> ProfileReport:
        if path.exists():
            try:
                profile = ProfileReport.read_json(path)
                if str(profile.model.get("checkpoint_sha256", "")) == parent.sha256:
                    return profile
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        model = self.adapter.reload_model(parent)
        if self.advanced_profiling:
            calibration = None
            if bool(parent.metadata.get("offline_simulation", False)):
                calibration = CalibrationCorpus(
                    specs=(
                        CalibrationSampleSpec(
                            prompt="a representative Wan video calibration prompt",
                            seed=7,
                            latent_shape=(2, int(parent.structure.hidden_size)),
                            timesteps=(0.9, 0.5, 0.1),
                        ),
                    ),
                    provider=FakeWanCalibrationProvider(int(parent.structure.hidden_size)),
                )
            profile = profile_model(
                self.adapter,
                model,
                seed=7,
                calibration=calibration,
                advanced=True,
                channel_chunk_size=16,
            )
        else:
            profile = profile_model(self.adapter, model, seed=7)
        profile.write_json(path)
        return profile

    def _screening_context(
        self,
        parent: CheckpointArtifact,
        profile: ProfileReport,
        algorithm: AlgorithmCandidate,
        plan: SandboxCompressionPlan,
        *,
        full_evaluator: Optional[Callable[[Any], Mapping[str, Any]]] = None,
    ) -> ScreeningContext:
        return ScreeningContext(
            parent=parent,
            adapter=self.adapter,
            profile=profile.to_dict(),
            algorithm=algorithm,
            plan=plan,
            registry=self.registry,
            search_space_version=self.search_space_version.to_dict() if self.search_space_version is not None else None,
            full_evaluator=full_evaluator,
        )

    def _algorithm_proxy_evaluator(self, result: SandboxResult) -> Mapping[str, Any]:
        """Compare algorithm plans with an offline proxy, never promotion evidence."""

        if result.plan is None:
            return {"valid": False, "offline_simulation": True}
        operator = result.plan.operator
        scores = {
            "prune_blocks": (0.78, 0.82),
            "prune_ffn_channels": (0.88, 0.68),
            "compose": (0.92, 0.55),
            "prune_ffn_by_weight_threshold": (0.90, 0.66),
        }
        score, cost = scores.get(operator, (0.50, 1.0))
        return {
            "valid": True,
            "offline_simulation": True,
            "evaluation_status": "cpu_fake_algorithm_proxy",
            "search_score": score,
            "model_size_gb": cost,
            "cost_units": cost,
        }

    @staticmethod
    def _load_algorithm_validation(path: Path) -> Optional[AlgorithmValidationRecord]:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return AlgorithmValidationRecord(
                algorithm_id=str(raw["algorithm_id"]),
                parent_algorithm_id=str(raw["parent_algorithm_id"]),
                parent_source_sha256=str(raw["parent_source_sha256"]),
                candidate_source_sha256=str(raw["candidate_source_sha256"]),
                valid=bool(raw["valid"]),
                evidence_level=str(raw["evidence_level"]),
                controlled_conditions_digest=str(raw["controlled_conditions_digest"]),
                parent_plan=raw.get("parent_plan"),
                candidate_plan=raw.get("candidate_plan"),
                parent_metrics=dict(raw.get("parent_metrics") or {}),
                candidate_metrics=dict(raw.get("candidate_metrics") or {}),
                reasons=tuple(str(item) for item in raw.get("reasons", [])),
                created_at=str(raw.get("created_at", "")),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _selection_history(self) -> list[Mapping[str, Any]]:
        """Return local operator selections without exposing evaluator evidence.

        A fixed deterministic baseline still needs to know which structural
        choices it already tried; otherwise duplicate-source protection ends
        the campaign after the first rejected round.  This is deliberately a
        narrow artifact view: quality, hardware, reward, and other experience
        fields are supplied only through ``skills`` when Memory is enabled.
        """

        history: list[Mapping[str, Any]] = []
        for decision_path in sorted(self.rounds_root.glob("R*/decision.json")):
            try:
                decision = self._read_json(decision_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            history.append(
                {
                    "round_id": decision_path.parent.name,
                    "algorithm_id": decision.get("algorithm_id"),
                    "status": decision.get("status"),
                    "decision": {
                        "operator": decision.get("operator"),
                        "operator_args": dict(decision.get("operator_args") or {}),
                    },
                }
            )
        return history

    def _load_sandbox(self, path: Path) -> Optional[SandboxResult]:
        if not path.exists():
            return None
        raw = self._read_json(path)
        status = str(raw.get("status", ""))
        plan = None
        if raw.get("plan") is not None:
            plan = SandboxCompressionPlan.from_mapping(raw["plan"], operator_specs=self.sandbox.operator_specs)
        return SandboxResult(
            status=status,
            source_sha256=str(raw.get("source_sha256", "")),
            plan=plan,
            failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
            message=str(raw.get("message", "")),
            stdout_path=str(raw["stdout_path"]) if raw.get("stdout_path") else None,
            stderr_path=str(raw["stderr_path"]) if raw.get("stderr_path") else None,
            metadata=dict(raw.get("metadata") or {}),
        )

    def _load_compression(self, path: Path) -> Optional[CompressionResult]:
        if not path.exists():
            return None
        raw = self._read_json(path)
        status = str(raw.get("status", ""))
        child_raw = raw.get("child_checkpoint")
        child = None
        if status == "success":
            if not isinstance(child_raw, Mapping) or not str(child_raw.get("path", "")).strip():
                raise ValueError("incomplete compression artifact")
            child = self._artifact_for_path(str(child_raw["path"]), metadata=dict(child_raw.get("metadata") or {}))
            if child.sha256 != str(child_raw.get("sha256", "")):
                raise ValueError("resumed child checkpoint hash mismatch")
        return CompressionResult(
            status=status,
            parent_checkpoint_sha256=str(raw.get("parent_checkpoint_sha256", "")),
            child_checkpoint=child,
            structural_change=dict(raw.get("structural_change") or {}),
            failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
            message=str(raw.get("message", "")),
            wall_time_s=float(raw.get("wall_time_s", 0.0)),
        )

    def _next_round_id(self) -> str:
        numbers = []
        for path in self.rounds_root.glob("R*"):
            try:
                numbers.append(int(path.name[1:]))
            except ValueError:
                continue
        return "R%04d" % (max(numbers, default=0) + 1)

    def _load_search_policy(self) -> SearchPolicy:
        path = self.output_root / "search-policy.json"
        if not path.exists():
            return SearchPolicy()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return SearchPolicy.from_dict(dict(raw))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return SearchPolicy()

    def _persist_search_policy(self) -> None:
        _write_json(self.output_root / "search-policy.json", self.search_policy.to_dict())

    def _update_search_policy(self, *, feasible: bool, improved: bool, failure_type: Optional[str] = None) -> None:
        if self.adapt_search_policy:
            self.search_policy.update(feasible=feasible, improved=improved, failure_type=failure_type)
        self._persist_search_policy()

    @staticmethod
    def _default_evaluator(child: CheckpointArtifact, profile: Mapping[str, Any]) -> Mapping[str, Any]:
        before = int(profile["model"]["block_count"])
        after = int(child.structure.block_count)
        ratio = float(after) / float(max(before, 1))
        baseline_quality = 0.99
        quality = max(0.0, baseline_quality - (1.0 - ratio) * 0.01)
        baseline_hardware = {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0}
        return {
            "evaluation_status": "structural_proxy",
            "quality_score": quality,
            "baseline_quality_score": baseline_quality,
            "baseline_hardware_metrics": baseline_hardware,
            "latency_s": 1.0 * ratio,
            "peak_memory_gb": 1.0 * ratio,
            "model_size_gb": max(0.01, 1.0 * ratio),
            "hardware_metrics": {"latency_s": 1.0 * ratio, "peak_memory_gb": 1.0 * ratio, "model_size_gb": max(0.01, 1.0 * ratio)},
            "valid": True,
            "feasible": True,
            "critical_regression": False,
            "offline_simulation": True,
            "search_score": max(0.0, 0.98 - (1.0 - ratio) * 0.06),
        }


def algorithm_plan_operator(result: SandboxResult) -> str:
    if result.plan is None:
        raise ValueError("algorithm plan is unavailable")
    return result.plan.operator


def _history_failure_reasons(history: list[Mapping[str, Any]]) -> tuple[str, ...]:
    values: list[str] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        decision = item.get("decision")
        if isinstance(decision, Mapping):
            if decision.get("failure_type"):
                values.append(str(decision["failure_type"]))
            promotion = decision.get("promotion")
            if isinstance(promotion, Mapping):
                reasons = promotion.get("reasons", [])
                if isinstance(reasons, (list, tuple)):
                    values.extend(str(reason) for reason in reasons if str(reason).strip())
        if item.get("failure_reason"):
            values.append(str(item["failure_reason"]))
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique[:16])


def _sandbox_dict(result: SandboxResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "source_sha256": result.source_sha256,
        "plan": result.plan.to_dict() if result.plan else None,
        "failure_type": result.failure_type,
        "message": result.message,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
        "metadata": dict(result.metadata),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
