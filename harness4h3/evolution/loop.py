from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact, CompressionResult, ModelStructure, ModelVariant, WanModelAdapter, stable_digest
from ..archive.model_candidate import ModelCandidate
from ..archive.model_store import ModelStore
from ..h3.state import ModelState
from ..memory.design_gene import DesignGene
from ..memory.experience import ExperienceRecord, ExperienceStore
from ..profiler import CalibrationCorpus, CalibrationSampleSpec, FakeWanCalibrationProvider, ProfileReport, profile_model
from ..operators.wan import OperatorSpec, SearchSpaceManager, SearchSpaceVersion, default_search_space
from ..sandbox.policy import SourcePolicyError, validate_source
from ..sandbox.runner import AlgorithmSandbox, CompressionPlan as SandboxCompressionPlan, SandboxResult
from .agent import AlgorithmAgent
from .algorithm_evaluation import AlgorithmModelEvaluator
from .candidate import AlgorithmCandidate
from .design import build_design_record
from .evaluator import PromotionPolicy
from .improvement import AlgorithmValidationRecord, AlgorithmValidator
from .policy import SearchPolicy
from .screening import CandidateScreeningPipeline, ScreeningContext
from .search_ranking import SEARCH_RANKING_VERSION, reusable_measured_trace, budgeted_candidate_order, candidate_family, efficiency_reduction_ratio, measured_candidate_key, measured_pareto_key, measured_recovery_key, measured_search_score, pareto_front_indices
from .skill_hub import SkillHub
from .store import AlgorithmStore
from .taxonomy import classify_algorithm_change, classify_search_space_proposal
from .tica_calibration import TICACalibrationManifest, TICAEvaluationSplits
from .tica_prune import TICARecipe, TICAPruner, _operator_specs_from_search_space, _run_algorithm_search_program, build_human_tica_source, collect_tica_profile, tica_operator_spec, tica_score
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
        channel_method: str = "weight",
        algorithm_improvement: bool = False,
        search_space_evolution: bool = False,
        frozen_expanded_search_space: bool = False,
        candidate_screening: bool = False,
        algorithm_probe_samples: int = 3,
        phase: str = "evolution",
        search_space_manager: Optional[SearchSpaceManager] = None,
        tica_prune: bool = False,
        tica_manifest: Optional[TICACalibrationManifest] = None,
        tica_splits: Optional[TICAEvaluationSplits] = None,
        tica_recipe: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.adapter = adapter
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rounds_root = self.output_root / "rounds"
        self.rounds_root.mkdir(parents=True, exist_ok=True)
        self.model_store = model_store or ModelStore(self.output_root / "models")
        self.algorithm_store = algorithm_store or AlgorithmStore(self.output_root / "algorithms")
        self.experience_store = experience_store or ExperienceStore(self.output_root / "experiences.jsonl")
        self.skill_hub = skill_hub or SkillHub(
            self.experience_store.path,
            self.output_root / "design-genes.jsonl",
            self.output_root / "algorithm-skills.jsonl",
        )
        self.agent = agent or AlgorithmAgent()
        self.sandbox = sandbox or AlgorithmSandbox(self.output_root / "sandbox")
        self.tica_prune = bool(tica_prune)
        self.tica_splits = tica_splits
        self.tica_manifest = tica_splits.calibration if tica_splits is not None else tica_manifest
        if self.tica_prune and self.tica_splits is None and bool(parent.metadata.get("offline_simulation", False)):
            self.tica_splits = TICAEvaluationSplits(
                calibration=self.tica_manifest or TICACalibrationManifest.from_fake(hidden_size=int(parent.structure.hidden_size), sample_prefix="calibration"),
                validation=TICACalibrationManifest.from_fake(hidden_size=int(parent.structure.hidden_size), seed=1007, sample_prefix="validation"),
            )
            self.tica_manifest = self.tica_splits.calibration
        if self.tica_prune and self.tica_manifest is None:
            raise ValueError("TICA requires explicit Calibration/Validation/Test manifests")
        if self.tica_prune and self.tica_splits is None:
            raise ValueError("TICA Evolution requires Calibration and Validation split manifests")
        if self.tica_prune:
            # Evolution may inspect Calibration and Validation sources only.
            # The Test manifest remains reserved for FinalEvaluation and is
            # deliberately not opened or hash-verified on this path.
            if any(manifest.provider_name != "fake_wan_cpu" for manifest in (self.tica_splits.calibration, self.tica_splits.validation)):  # type: ignore[union-attr]
                for manifest in (self.tica_splits.calibration, self.tica_splits.validation):  # type: ignore[union-attr]
                    manifest.verify_cached_sources()
            split_payload = self.tica_splits.to_dict()  # type: ignore[union-attr]
            split_payload.pop("test", None)
            _write_json(self.output_root / "tica-splits.json", split_payload)
        self.tica_recipe = TICARecipe.from_mapping(tica_recipe)
        self._custom_evaluator = evaluator is not None
        self.registry = registry or build_wan_compression_registry(include_tica=self.tica_prune, include_multigranular=True)
        self.evaluator = evaluator or self._default_evaluator
        self.search_policy = search_policy or self._load_search_policy()
        self.memory_enabled = bool(memory_enabled)
        self.adapt_search_policy = bool(adapt_search_policy)
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.advanced_profiling = bool(advanced_profiling)
        self.channel_method = str(channel_method).strip() or "weight"
        self.algorithm_improvement = bool(algorithm_improvement)
        self.algorithm_probe_samples = max(1, int(algorithm_probe_samples))
        self.search_space_evolution = bool(search_space_evolution)
        self.frozen_expanded_search_space = bool(frozen_expanded_search_space)
        self.candidate_screening = bool(candidate_screening)
        self.phase = str(phase).strip().lower() or "evolution"
        if self.phase != "evolution":
            raise ValueError("EvolutionLoop only supports phase=evolution; use FinalEvaluation for Test")
        self.search_space_manager: Optional[SearchSpaceManager] = search_space_manager
        self.search_space_version = None
        self.search_space_proposal = None
        self.search_space_feedback: Mapping[str, Any] = {}
        if self.tica_prune:
            self.sandbox.operator_specs["tica_prune"] = tica_operator_spec()
        self.pareto_path = self.output_root / "pareto-front.json"
        self.parent = parent
        self._initialize_evolution_features()
        self._initialize_parent()

    def _initialize_evolution_features(self) -> None:
        """Initialize additive features without changing legacy defaults."""

        if not (self.search_space_evolution or self.frozen_expanded_search_space):
            return
        self.search_space_manager = self.search_space_manager or SearchSpaceManager(self.output_root / "search-space")
        self.search_space_version = self.search_space_manager.active
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
                evidence={
                    "proposal_id": proposal.proposal_id,
                    "search_space_version": self.search_space_version.version_id,
                    "implementation_source_sha256": validation.get("implementation_source_sha256"),
                    "validation": validation,
                },
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
        profile_payload: Mapping[str, Any] = profile.to_dict()
        tica_profile_path = round_dir / "tica-profile.json"
        tica_search_path = round_dir / "tica-search.json"
        algorithm_trace_path = round_dir / "algorithm-program-trace.json"
        if self.tica_prune:
            tica_profile = self._load_or_collect_tica_profile(tica_profile_path, parent_artifact)
            profile_payload = {**dict(profile_payload), **dict(tica_profile)}
        search_space_path = round_dir / "search-space.json"
        pinned_search_space = False
        if resuming and search_space_path.exists() and self.search_space_manager is not None:
            try:
                pinned = SearchSpaceVersion.from_dict(self._read_json(search_space_path))
                if pinned.status == "validated":
                    self.search_space_version = pinned
                    self.registry = self.search_space_manager.registry_for(pinned, self.registry)
                    self.sandbox.operator_specs.update(
                        {
                            item.name: item.to_dict()
                            for item in pinned.operators
                            if item.trusted_factory_key
                        }
                    )
                    pinned_search_space = True
            except (KeyError, TypeError, ValueError):
                pinned_search_space = False
        if self.search_space_version is not None and not search_space_path.exists():
            _write_json(search_space_path, self.search_space_version.to_dict())
        architecture = str(profile_payload["model"]["architecture"])
        skills = self.skill_hub.search(architecture=architecture, limit=8) if self.memory_enabled else []
        pareto_front = self._load_pareto_front()
        selection_history = self._selection_history()
        algorithm_feedback = self._algorithm_search_feedback_history(
            limit=4,
            parent_checkpoint_sha256=parent_artifact.sha256,
        )
        # Saved measured-search traces are authoritative cross-round evidence;
        # they remain available even when optional SkillHub retrieval is off.
        # Put them first so bounded Agent context cannot truncate them behind
        # selection-only history or unrelated skills.
        agent_history = [*algorithm_feedback, *selection_history, *skills]
        algorithm_path = round_dir / "algorithm.json"
        persisted_algorithm = self._read_json(algorithm_path) if algorithm_path.exists() else None
        algorithm_id = str(persisted_algorithm.get("algorithm_id", "")).strip() if persisted_algorithm else self.algorithm_store.next_id()
        # The initial parent already has a measured structural state.  Let the
        # policy decide whether to exploit it or open an exploration branch.
        search_choice = str(round_state.get("search_choice", "")).strip() or self.search_policy.choose(has_best=bool(parent_candidate.state.measured_metrics))
        if self.search_space_evolution and not pinned_search_space:
            self._maybe_evolve_search_space()
            self.registry = self.search_space_manager.registry_for(self.search_space_version, self.registry)  # type: ignore[union-attr]
            self.sandbox.operator_specs.update(
                {
                    item.name: item.to_dict()
                    for item in self.search_space_version.operators  # type: ignore[union-attr]
                    if item.trusted_factory_key
                }
            )
            if self.search_space_version is not None and not search_space_path.exists():
                _write_json(search_space_path, self.search_space_version.to_dict())
            elif self.search_space_version is not None:
                _write_json(search_space_path, self.search_space_version.to_dict())
        # Algorithm lineage is independent from model lineage.  A rejected
        # child must not become the next model parent, and a rejected latest
        # algorithm must not automatically become the next algorithm parent.
        latest_algorithm = self.algorithm_store.latest()
        parent_selection = self._algorithm_parent_selection()
        improvement_parent = parent_selection.get("effective")
        parent_algorithm_id = improvement_parent.algorithm_id if improvement_parent is not None else None
        algorithm_generation_search_space = (
            self.search_space_version.to_dict()
            if self.search_space_version is not None
            else default_search_space().to_dict()
        )
        improved_algorithm = False
        algorithm_reused_after_duplicate = False
        if persisted_algorithm:
            algorithm = AlgorithmCandidate.from_dict(persisted_algorithm)
            algorithm_reused_after_duplicate = bool(
                algorithm.metadata.get("reused_existing_algorithm_after_duplicate")
            )
            if self.algorithm_improvement and algorithm.parent_algorithm_id and algorithm.parent_source_sha256:
                try:
                    improvement_parent = self.algorithm_store.get(algorithm.parent_algorithm_id)
                except ValueError:
                    improvement_parent = None
                improved_algorithm = improvement_parent is not None and not algorithm_reused_after_duplicate
            if not any(item.algorithm_id == algorithm.algorithm_id for item in self.algorithm_store.read()):
                if algorithm.parent_algorithm_id and algorithm.parent_source_sha256:
                    algorithm = self.algorithm_store.create_child(algorithm, algorithm.parent_source_sha256)
                else:
                    algorithm = self.algorithm_store.create(algorithm)
                _write_json(algorithm_path, algorithm.to_dict())
        else:
            try:
                if self.algorithm_improvement and improvement_parent is not None:
                    algorithm, improvement_parent = self._improve_unique_algorithm(
                        algorithm_id=algorithm_id,
                        parent=improvement_parent,
                        parent_id=parent_candidate.id,
                        profile=profile_payload,
                        history=agent_history,
                        failures=_history_failure_reasons(agent_history),
                        skills=skills,
                        search_space=algorithm_generation_search_space,
                        protocol_mode="tica" if self.tica_prune else "standard",
                        round_id=round_id,
                    )
                    algorithm_reused_after_duplicate = bool(
                        algorithm.metadata.get("reused_existing_algorithm_after_duplicate")
                    )
                    improved_algorithm = not algorithm_reused_after_duplicate
                elif self.tica_prune and not list(self.algorithm_store.read()):
                    algorithm = AlgorithmCandidate.from_source(
                        algorithm_id,
                        build_human_tica_source(algorithm_id),
                        parent_id=parent_candidate.id,
                        parent_algorithm_id=parent_algorithm_id,
                        hypothesis="human-designed TICA-Prune baseline using multi-timestep local risk and activation contribution",
                        metadata={
                            "origin": "human_baseline",
                            "tica_baseline": True,
                            "not_agent_discovered": True,
                            "design_strategy": "tica_prune",
                            "calibration_digest": self.tica_manifest.digest() if self.tica_manifest is not None else None,
                        },
                    )
                else:
                    algorithm = self.agent.generate(
                        algorithm_id,
                        parent_id=parent_candidate.id,
                        parent_algorithm_id=parent_algorithm_id,
                        profile=profile_payload,
                        history=agent_history,
                        skills=skills,
                        pareto_front=pareto_front,
                        search_space=algorithm_generation_search_space,
                        search_choice=search_choice,
                        protocol_mode="tica" if self.tica_prune else "standard",
                    )
            except (RuntimeError, TypeError, ValueError) as exc:
                failure_type = (
                    "duplicate_algorithm"
                    if str(exc).startswith("duplicate_algorithm_source")
                    else "algorithm_generation_failed"
                )
                algorithm = AlgorithmCandidate.from_source(
                    algorithm_id,
                    "# algorithm generation failed\ndef propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': []}}\n",
                    parent_id=parent_candidate.id,
                    parent_algorithm_id=parent_algorithm_id,
                    hypothesis="generation failed: %s" % exc,
                    metadata={"generator_failure": True},
                )
                _write_json(algorithm_path, algorithm.to_dict())
                return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, search_choice, "generation", failure_type, str(exc), {"profile": str(profile_path), "algorithm": str(algorithm_path)}, algorithm_persisted=False)
            try:
                algorithm = replace(
                    algorithm,
                    metadata={
                        **dict(algorithm.metadata),
                        "evaluation_parent_checkpoint_sha256": parent_artifact.sha256,
                        "evaluation_parent_model_id": parent_candidate.id,
                    },
                )
                _write_json(algorithm_path, algorithm.to_dict())
                if algorithm_reused_after_duplicate:
                    # The source already belongs to the evidence-selected
                    # parent. Keep reuse metadata local to this round; the
                    # immutable AlgorithmStore entry is not a new version.
                    pass
                elif improved_algorithm:
                    algorithm = self.algorithm_store.create_child(algorithm, improvement_parent.source_sha256)  # type: ignore[union-attr]
                else:
                    algorithm = self.algorithm_store.create(algorithm)
            except ValueError as exc:
                return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, search_choice, "generation", "duplicate_algorithm", str(exc), {"profile": str(profile_path), "algorithm": str(algorithm_path)}, algorithm_persisted=False)
            _write_json(algorithm_path, algorithm.to_dict())
        self._write_round_state(round_dir, {**round_state, "round_id": round_id, "parent_model_id": parent_candidate.id, "search_choice": search_choice, "algorithm_id": algorithm.algorithm_id, "status": "in_progress"})

        sandbox_path = round_dir / "sandbox.json"
        sandbox_result = self._load_sandbox(sandbox_path)
        if sandbox_result is None:
            sandbox_result = self.sandbox.run(algorithm.source, profile_payload, skills)
            _write_json(sandbox_path, _sandbox_dict(sandbox_result))
        if sandbox_result.status != "success" or sandbox_result.plan is None:
            return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, search_choice, "sandbox", sandbox_result.failure_type or "sandbox_failed", sandbox_result.message, {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path)})

        # Non-TICA algorithms use the same Algorithm Program contract.  A
        # source-defined ``search`` entrypoint is executed against the active
        # operator registry before model mutation; the first ranked plan is
        # the actual plan sent to compression/evaluation.  TICA keeps its
        # deeper calibration-budgeted path below.
        if not self.tica_prune:
            try:
                validated_source = validate_source(algorithm.source)
            except SourcePolicyError as exc:
                return self._finish_failure(
                    round_id,
                    round_dir,
                    parent_candidate,
                    algorithm,
                    search_choice,
                    "algorithm_search",
                    "algorithm_search_source_invalid",
                    str(exc),
                    {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path)},
                )
            if "search" in validated_source.functions:
                search_space_payload = (
                    self.search_space_version.to_dict()
                    if self.search_space_version is not None
                    else default_search_space().to_dict()
                )
                block_search_space_payload = algorithm_generation_search_space
                if not block_search_space_payload["operators"]:
                    raise ValueError("standard_search_space_has_no_operators")
                cached_trace = self._read_json(algorithm_trace_path) if algorithm_trace_path.exists() else None
                selected_plan: Optional[SandboxCompressionPlan] = None
                trace_payload: Mapping[str, Any] | None = None
                if (
                    isinstance(cached_trace, Mapping)
                    and cached_trace.get("algorithm_source_sha256") == algorithm.source_sha256
                    and isinstance(cached_trace.get("selected_plan"), Mapping)
                    and cached_trace.get("search_space_version") == search_space_payload.get("version_id")
                    and (not self.algorithm_improvement or reusable_measured_trace(cached_trace))
                ):
                    try:
                        selected_plan = SandboxCompressionPlan.from_mapping(
                            cached_trace["selected_plan"],
                            operator_specs=self.sandbox.operator_specs,
                        )
                        trace_payload = cached_trace
                    except (TypeError, ValueError):
                        selected_plan = None
                if selected_plan is None:
                    try:
                        decision = _run_algorithm_search_program(
                            algorithm.source,
                            self.sandbox,
                            profile_payload,
                            agent_history,
                            {
                                "depth": 1,
                                "parent_state_id": "uncompressed",
                                "proposed_plan": sandbox_result.plan.to_dict(),
                                "budget": {"max_candidates": 16},
                                "measured_candidates": [],
                                "frontier": [],
                            },
                            block_search_space_payload,
                        )
                        measured_search: Mapping[str, Any] | None = None
                        if self.algorithm_improvement:
                            seed_value = profile_payload.get("provenance", {}).get("seed", 7) if isinstance(profile_payload.get("provenance"), Mapping) else 7
                            measured_search = self._complete_standard_algorithm_search(
                                source=algorithm.source,
                                proposed_result=sandbox_result,
                                profile=profile_payload,
                                history=agent_history,
                                parent=parent_artifact,
                                output_root=round_dir / "algorithm-search",
                                seed=int(seed_value or 7),
                                search_budget={
                                    "max_candidates": 16,
                                    "max_full_evals": 8,
                                    "max_interaction_evals": 0,
                                    "max_search_depth": 2,
                                    "quality_floor_ratio": self.promotion_policy.quality_floor_ratio,
                                },
                                search_space_payload=block_search_space_payload,
                            )
                            selected_raw = measured_search.get("selected_plan")
                            if measured_search.get("valid") is not True or not isinstance(selected_raw, Mapping):
                                trace_payload = {
                                    "algorithm_id": algorithm.algorithm_id,
                                    "algorithm_source_sha256": algorithm.source_sha256,
                                    "trace": [{
                                        "depth": 1,
                                        "source_sha256": decision.source_sha256,
                                        "payload_sha256": decision.payload_sha256,
                                        "plan_digests": [stable_digest(item) for item in decision.plans],
                                        "rank": list(decision.rank),
                                        "stop": decision.stop,
                                        "search_space_version": search_space_payload.get("version_id"),
                                        "search_space_operator_names": [str(item.get("name", "")) for item in block_search_space_payload.get("operators", ()) if isinstance(item, Mapping)],
                                    }],
                                    "decision": decision.to_dict(),
                                    "selected_plan": None,
                                    "selected_rank_index": None,
                                    "measured_search": dict(measured_search.get("algorithm_search") or {}),
                                    "search_space_version": search_space_payload.get("version_id"),
                                }
                                _write_json(algorithm_trace_path, trace_payload)
                                raise ValueError(str(measured_search.get("failure_reason") or "standard_algorithm_search_no_valid_candidate"))
                        else:
                            selected_raw = next(
                                (
                                    decision.plans[index]
                                    for index in decision.rank
                                    if decision.plans[index].get("operator") == "prune_blocks"
                                ),
                                None,
                            )
                            if not isinstance(selected_raw, Mapping):
                                raise ValueError("standard_block_search_no_candidate")
                        selected_plan = SandboxCompressionPlan.from_mapping(selected_raw, operator_specs=self.sandbox.operator_specs)
                        trace_payload = {
                            "algorithm_id": algorithm.algorithm_id,
                            "algorithm_source_sha256": algorithm.source_sha256,
                            "trace": [{
                                "depth": 1,
                                "source_sha256": decision.source_sha256,
                                "payload_sha256": decision.payload_sha256,
                                "plan_digests": [stable_digest(item) for item in decision.plans],
                                "rank": list(decision.rank),
                                "evaluation_order": list((measured_search or {}).get("algorithm_search", {}).get("trace", [{}])[0].get("evaluation_order", [])),
                                "stop": decision.stop,
                                "search_space_version": search_space_payload.get("version_id"),
                                "search_space_operator_names": [str(item.get("name", "")) for item in block_search_space_payload.get("operators", ()) if isinstance(item, Mapping)],
                            }],
                            "decision": decision.to_dict(),
                            "selected_plan": selected_plan.to_dict(),
                            "selected_rank_index": int((measured_search or {}).get("algorithm_search", {}).get("selected_rank_index", decision.rank[0])),
                            "selected_rank_position": int((measured_search or {}).get("algorithm_search", {}).get("selected_rank_position", 0)),
                            "search_space_version": search_space_payload.get("version_id"),
                        }
                        if measured_search is not None:
                            trace_payload = {
                                **trace_payload,
                                "measured_search": dict(measured_search.get("algorithm_search") or {}),
                                "measured_selected_plan": dict(measured_search.get("selected_plan") or {}),
                            }
                        _write_json(algorithm_trace_path, trace_payload)
                    except (RuntimeError, TypeError, ValueError) as exc:
                        return self._finish_failure(
                            round_id,
                            round_dir,
                            parent_candidate,
                            algorithm,
                            search_choice,
                            "algorithm_search",
                            "algorithm_search_failed",
                            str(exc),
                            {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "algorithm_program_trace": str(algorithm_trace_path)},
                        )
                sandbox_result = SandboxResult(
                    "success",
                    sandbox_result.source_sha256,
                    selected_plan,
                    metadata={
                        **dict(sandbox_result.metadata),
                        "algorithm_program_search": True,
                        "algorithm_program_trace": str(algorithm_trace_path),
                        "algorithm_program_selected_plan_digest": stable_digest(selected_plan.to_dict()),
                        "algorithm_program_measured_search": dict((trace_payload or {}).get("measured_search") or {}),
                    },
                )
                _write_json(sandbox_path, _sandbox_dict(sandbox_result))

        tica_search_payload: Mapping[str, Any] | None = None
        active_tica_recipe: Optional[TICARecipe] = None
        tica_agent_rule_digest: Optional[str] = None
        if self.tica_prune:
            # The Agent source is a bounded algorithm-rule proposal.  Its
            # returned recipe fields configure the actual search; the Agent
            # does not get to bypass trusted operators or execute a model.
            tica_recipe = self._tica_recipe_from_plan(sandbox_result.plan.args)
            active_tica_recipe = tica_recipe
            tica_agent_rule_digest = stable_digest(dict(sandbox_result.plan.args))
            if tica_search_path.exists():
                cached_search = self._read_json(tica_search_path)
                if (
                    cached_search.get("recipe") == tica_recipe.to_dict()
                    and cached_search.get("algorithm_id") == algorithm.algorithm_id
                    and cached_search.get("algorithm_source_sha256") == algorithm.source_sha256
                    and cached_search.get("agent_rule_digest") == tica_agent_rule_digest
                    and cached_search.get("selected_candidate_id")
                    and cached_search.get("selected_plan_digest")
                ):
                    tica_search_payload = cached_search
            if tica_search_payload is None:
                search = self._run_tica_search(
                    parent_artifact,
                    profile_payload,
                    round_dir,
                    recipe=tica_recipe,
                    seed_plan=sandbox_result.plan.args,
                    algorithm_source=algorithm.source,
                    algorithm_history=agent_history,
                    search_space=self.search_space_version.to_dict() if self.search_space_version is not None else {"operators": []},
                )
                tica_search_payload = {
                    **search.to_dict(),
                    "algorithm_id": algorithm.algorithm_id,
                    "algorithm_source_sha256": algorithm.source_sha256,
                    "agent_rule_digest": tica_agent_rule_digest,
                }
                _write_json(tica_search_path, tica_search_payload)
            _write_json(
                algorithm_trace_path,
                {
                    "algorithm_id": algorithm.algorithm_id,
                    "algorithm_source_sha256": algorithm.source_sha256,
                    "trace": list(tica_search_payload.get("algorithm_trace", [])),
                    "trace_digest": stable_digest(tica_search_payload.get("algorithm_trace", [])),
                    "search_space_version": self.search_space_version.version_id if self.search_space_version is not None else "default",
                },
            )
            selected_id = str(tica_search_payload.get("selected_candidate_id", ""))
            selected_digest = str(tica_search_payload.get("selected_plan_digest", ""))
            candidates = tica_search_payload.get("candidates")
            selected = next(
                (item for item in candidates if isinstance(item, Mapping) and str(item.get("candidate_id", "")) == selected_id),
                None,
            ) if isinstance(candidates, list) else None
            selected_plan = selected.get("plan") if isinstance(selected, Mapping) else None
            if not isinstance(selected_plan, Mapping) or not selected_digest:
                return self._finish_failure(
                    round_id,
                    round_dir,
                    parent_candidate,
                    algorithm,
                    search_choice,
                    "tica_search",
                    "tica_no_feasible_selected_candidate",
                    "TICA search produced no measured feasible compression plan",
                    {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "tica_search": str(tica_search_path)},
                )
            if isinstance(selected_plan.get("operator"), str) and isinstance(selected_plan.get("args"), Mapping):
                selected_operator = str(selected_plan["operator"])
                selected_args = dict(selected_plan["args"])
                computed_digest = stable_digest(selected_plan)
            else:
                selected_operator = "tica_prune"
                selected_args = dict(selected_plan)
                computed_digest = TICAPruner._plan_digest(selected_args)
            if computed_digest != selected_digest:
                return self._finish_failure(
                    round_id,
                    round_dir,
                    parent_candidate,
                    algorithm,
                    search_choice,
                    "tica_search",
                    "tica_selected_plan_digest_mismatch",
                    "selected TICA plan digest does not match its candidate record",
                    {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "tica_search": str(tica_search_path)},
                )
            sandbox_result = SandboxResult(
                "success",
                sandbox_result.source_sha256,
                SandboxCompressionPlan(selected_operator, selected_args),
                metadata={
                    **dict(sandbox_result.metadata),
                    "tica_selected_candidate_id": selected_id,
                    "tica_selected_plan_digest": selected_digest,
                    "tica_agent_rule_digest": tica_agent_rule_digest,
                },
            )

        validation_path = round_dir / "algorithm-validation.json"
        validation_profiles_path = round_dir / "algorithm-validation-profiles.json"
        if improved_algorithm and improvement_parent is not None:
            validation = self._load_algorithm_validation(validation_path)
            if validation is None:
                validation_parent = self._algorithm_validation_parent(improvement_parent, parent_artifact)
                validation_profiles = self._load_validation_profiles(validation_profiles_path)
                if validation_profiles is None:
                    validation_profiles = self._build_validation_profiles(validation_parent, profile_payload)
                    _write_json(validation_profiles_path, {"profiles": list(validation_profiles)})
                measurement_evaluator = self._algorithm_measurement_evaluator(
                    validation_parent,
                    validation_profiles,
                    round_dir / "algorithm-validation-measurements",
                    recipe=active_tica_recipe,
                    algorithm_sources={
                        improvement_parent.source_sha256: improvement_parent.source,
                        algorithm.source_sha256: algorithm.source,
                    },
                    algorithm_history=agent_history,
                )
                validation = AlgorithmValidator().validate(
                    improvement_parent,
                    algorithm,
                    self.sandbox,
                    validation_profiles[0],
                    agent_history,
                    measurement_evaluator,
                    budget={
                        "sandbox_timeout_s": self.sandbox.timeout_s,
                        "mode": "cpu_fake" if bool(validation_parent.metadata.get("offline_simulation", False)) else "real_wan_structure",
                        "tica_search_budget": (active_tica_recipe or self.tica_recipe).budget() if self.tica_prune else None,
                    },
                    profiles=validation_profiles,
                    # Both TICA and standard Algorithm Programs may define a
                    # source-controlled ``search`` entrypoint.  The
                    # validation callback reruns that search for parent and
                    # child under identical conditions, so an improvement is
                    # never accepted merely because its first ``propose``
                    # result looked better.
                    complete_search=measurement_evaluator,
                    algorithm_profile=profile_payload if self.tica_prune else None,
                )
                _write_json(validation_path, validation.to_dict())
            algorithm = replace(
                algorithm,
                validation_status="validated" if validation.valid else "rejected",
                validation_record_path=str(validation_path),
            )
            self.algorithm_store.update(algorithm)
            _write_json(algorithm_path, algorithm.to_dict())

        validation_record = self._load_algorithm_validation(validation_path) if validation_path.exists() else None
        parent_plan_for_taxonomy = validation_record.parent_plan if validation_record is not None and validation_record.parent_plan else sandbox_result.plan.to_dict()
        parent_source_for_taxonomy = improvement_parent.source if improvement_parent is not None and improved_algorithm else algorithm.source
        change_taxonomy = classify_algorithm_change(
            parent_source_for_taxonomy,
            algorithm.source,
            parent_plan_for_taxonomy,
            sandbox_result.plan.to_dict(),
            self.search_space_version.to_dict() if self.search_space_version is not None else default_search_space().to_dict(),
        )
        if self.search_space_proposal is not None:
            change_taxonomy["search_space_proposal"] = classify_search_space_proposal(
                self.search_space_proposal.spec.to_dict(),
                self.search_space_version.to_dict() if self.search_space_version is not None else default_search_space().to_dict(),
                self.search_space_proposal.validation,
            )

        design_path = round_dir / "design.json"
        design = self._read_json(design_path) if design_path.exists() else {}
        if not design:
            design = build_design_record(
                algorithm_id=algorithm.algorithm_id,
                parent_algorithm_id=algorithm.parent_algorithm_id,
                search_choice=search_choice,
                profile=profile_payload,
                history=agent_history,
                plan=sandbox_result.plan.to_dict(),
                hypothesis=algorithm.hypothesis,
                change_summary=algorithm.change_summary,
                change_rationale=algorithm.change_rationale,
                source_sha256=algorithm.source_sha256,
                parent_source_sha256=algorithm.parent_source_sha256,
                change_taxonomy=change_taxonomy,
            )
            _write_json(design_path, design)

        screening_path = round_dir / "screening"
        if self.candidate_screening:
            screening = CandidateScreeningPipeline(screening_path)
            pre_results = screening.run(self._screening_context(parent_artifact, profile_payload, algorithm, sandbox_result.plan))
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

        # Keep Model evolution (the compressed checkpoint) separate from the
        # runtime/sampler policy identity.  The sidecar is written outside
        # the checkpoint directory so it cannot change the artifact digest;
        # resuming a round therefore restores the same variant description
        # without re-saving or re-counting the model.
        variant_path = round_dir / "variant_manifest.json"
        if not variant_path.exists():
            variant = ModelVariant.from_compression(
                "M-" + round_id,
                parent_artifact,
                compression,
                sandbox_result.plan.to_dict(),
                algorithm_id=algorithm.algorithm_id,
                search_space_version=self.search_space_version.version_id if self.search_space_version is not None else None,
                source_revision=algorithm.source_sha256,
                runtime_revision=self.adapter.__class__.__name__,
            )
            variant.write_manifest(variant_path)

        change_taxonomy = classify_algorithm_change(
            parent_source_for_taxonomy,
            algorithm.source,
            parent_plan_for_taxonomy,
            sandbox_result.plan.to_dict(),
            self.search_space_version.to_dict() if self.search_space_version is not None else default_search_space().to_dict(),
            structural_change=compression.structural_change,
        )
        if self.search_space_proposal is not None:
            change_taxonomy["search_space_proposal"] = classify_search_space_proposal(
                self.search_space_proposal.spec.to_dict(),
                self.search_space_version.to_dict() if self.search_space_version is not None else default_search_space().to_dict(),
                self.search_space_proposal.validation,
            )
        design = build_design_record(
            algorithm_id=algorithm.algorithm_id,
            parent_algorithm_id=algorithm.parent_algorithm_id,
            search_choice=search_choice,
            profile=profile_payload,
            history=agent_history,
            plan=sandbox_result.plan.to_dict(),
            hypothesis=algorithm.hypothesis,
            change_summary=algorithm.change_summary,
            change_rationale=algorithm.change_rationale,
            source_sha256=algorithm.source_sha256,
            parent_source_sha256=algorithm.parent_source_sha256,
            change_taxonomy=change_taxonomy,
        )
        _write_json(design_path, design)

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
                if self.tica_prune and not self._custom_evaluator:
                    # Calibration is search-only.  Model-level selection and
                    # promotion must use the held-out Validation split; the
                    # Test split is reserved for a later frozen evaluation.
                    promotion_profile = {**dict(profile_payload), "evaluation_split": "validation"}
                    measurement = AlgorithmModelEvaluator(
                        self.adapter,
                        self.registry,
                        parent_artifact,
                        promotion_profile,
                        round_dir / "model-evaluation",
                        seed=int(profile_payload.get("provenance", {}).get("seed", 7)),
                        calibration_manifest=self._tica_manifest_for_split("validation"),
                        search_budget=(active_tica_recipe or self.tica_recipe).budget(),
                    )
                    evaluation = dict(measurement.evaluate(sandbox_result))
                else:
                    evaluation = dict(self.evaluator(compression.child_checkpoint, profile_payload))
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
        plan_digest_mismatch = False
        if tica_search_payload is not None:
            final_plan_digest = stable_digest(sandbox_result.plan.to_dict())
            measured_plan_digest = evaluation.get("plan_digest")
            if measured_plan_digest is not None and str(measured_plan_digest) != final_plan_digest:
                plan_digest_mismatch = True
                evaluation = {
                    **evaluation,
                    "valid": False,
                    "evaluation_status": "plan_digest_mismatch",
                    "failure_type": "tica_plan_digest_mismatch",
                    "failure_message": "measured evaluation plan digest does not match final execution plan",
                }
        if not evaluation_path.exists() or not child_integrity or plan_digest_mismatch:
            _write_json(evaluation_path, evaluation)
        if self.candidate_screening:
            screening = CandidateScreeningPipeline(screening_path)
            full_results = screening.run(
                    self._screening_context(
                        parent_artifact,
                        profile_payload,
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
            "algorithm_source_sha256": algorithm.source_sha256,
            "algorithm_parent_source_sha256": algorithm.parent_source_sha256,
            "algorithm_source_diff": algorithm.source_diff,
            "operator": sandbox_result.plan.operator,
            "operator_args": dict(sandbox_result.plan.args),
            "variant_manifest": str(variant_path),
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
            "algorithm_parent_selection": {
                key: (
                    [item.algorithm_id for item in value]
                    if isinstance(value, (list, tuple))
                    else value.algorithm_id if isinstance(value, AlgorithmCandidate) else value
                )
                for key, value in parent_selection.items()
            },
        }
        if self.search_space_version is not None:
            decision["search_space_version"] = self.search_space_version.version_id
        decision["change_taxonomy"] = change_taxonomy
        if tica_search_payload is not None:
            decision["selected_candidate_id"] = tica_search_payload.get("selected_candidate_id")
            decision["selected_plan_digest"] = tica_search_payload.get("selected_plan_digest")
            decision["final_plan_digest"] = stable_digest(sandbox_result.plan.to_dict())
            decision["tica_plan_digest_match"] = decision["selected_plan_digest"] == decision["final_plan_digest"]
            decision["tica_agent_rule_digest"] = tica_search_payload.get("agent_rule_digest")
            decision["tica_search_algorithm_id"] = tica_search_payload.get("algorithm_id")
            decision["algorithm_program_trace_path"] = str(algorithm_trace_path)
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
        if algorithm.validation_status == "validated":
            self.skill_hub.record_algorithm(
                algorithm,
                architecture=parent_candidate.state.architecture_name,
                validation_status=algorithm.validation_status,
                evidence={
                    "round_id": round_id,
                    "decision_path": str(decision_path),
                    "offline_simulation": evaluation_offline,
                    "change_taxonomy": change_taxonomy,
                },
            )
        if feasible:
            self._update_pareto_front(child_candidate.id, evaluation)
            self.model_store.set_active(child_candidate.id)
        paths = {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "design": str(design_path), "compression": str(compression_path), "variant_manifest": str(variant_path), "evaluation": str(evaluation_path), "decision": str(decision_path)}
        if self.tica_prune:
            paths["tica_profile"] = str(tica_profile_path)
            paths["tica_search"] = str(tica_search_path)
        if algorithm_trace_path.exists():
            paths["algorithm_program_trace"] = str(algorithm_trace_path)
        if self.search_space_version is not None:
            paths["search_space"] = str(search_space_path)
        if self.candidate_screening:
            paths["screening"] = str(screening_path / "screening.json")
        if validation_path.exists():
            paths["algorithm_validation"] = str(validation_path)
        if validation_profiles_path.exists():
            paths["algorithm_validation_profiles"] = str(validation_profiles_path)
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
            "algorithm_source_sha256": algorithm.source_sha256,
            "algorithm_parent_source_sha256": algorithm.parent_source_sha256,
            "algorithm_source_diff": algorithm.source_diff,
            "algorithm_change_summary": algorithm.change_summary,
            "algorithm_change_rationale": algorithm.change_rationale,
            "algorithm_validation_status": algorithm.validation_status,
            "change_taxonomy": dict(decision.get("change_taxonomy") or {}),
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
                "algorithm_id": decision.get("algorithm_id"),
                "algorithm_parent_id": design.get("parent_algorithm_id"),
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
                "algorithm_change_summary": design.get("change_summary"),
                "algorithm_change_rationale": design.get("change_rationale"),
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
                "algorithm_id": decision.get("algorithm_id"),
                "algorithm_source_sha256": decision.get("algorithm_source_sha256"),
                "change_taxonomy": dict(decision.get("change_taxonomy") or {}),
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
                channel_method=getattr(self, "channel_method", "weight"),
            )
        else:
            profile = profile_model(self.adapter, model, seed=7, channel_method=getattr(self, "channel_method", "weight"))
        profile.write_json(path)
        return profile

    def _load_or_collect_tica_profile(self, path: Path, parent: CheckpointArtifact) -> Mapping[str, Any]:
        """Load one immutable calibration profile or collect it once.

        The profile is separate from the legacy ProfileReport so old runs and
        their JSON schema remain untouched.  A cached profile is accepted only
        when it belongs to the same calibration manifest and parent checkpoint.
        """

        expected_digest = self.tica_manifest.digest() if self.tica_manifest is not None else None
        if path.exists():
            cached = self._read_json(path)
            tica = cached.get("tica") if isinstance(cached.get("tica"), Mapping) else {}
            if tica.get("calibration_digest") == expected_digest and cached.get("parent_checkpoint_sha256") == parent.sha256:
                if "evaluation_split" not in tica:
                    cached["tica"] = {**dict(tica), "evaluation_split": "calibration"}
                cached.setdefault("evaluation_split", "calibration")
                return cached
        model = self.adapter.reload_model(parent)
        if self.tica_manifest is None:
            raise ValueError("TICA calibration manifest is unavailable")
        collected = dict(collect_tica_profile(self.adapter, model, self.tica_manifest))
        tica_metadata = dict(collected.get("tica") or {})
        tica_metadata["evaluation_split"] = "calibration"
        collected["tica"] = tica_metadata
        collected["evaluation_split"] = "calibration"
        collected["parent_checkpoint_sha256"] = parent.sha256
        collected["source_revision"] = "tica-prune-v1"
        _write_json(path, collected)
        return collected

    def _tica_manifest_for_split(self, split: str) -> TICACalibrationManifest:
        if str(split).lower() == "test":
            raise ValueError("Test split is unavailable during Evolution phase")
        if self.tica_splits is None:
            if self.tica_manifest is None:
                raise ValueError("TICA split manifest is unavailable")
            return self.tica_manifest
        return self.tica_splits.manifest(split)

    def _tica_recipe_from_plan(self, plan: Mapping[str, Any]) -> TICARecipe:
        """Bind sandbox-approved Agent rule fields to the TICA search recipe."""

        raw = self.tica_recipe.to_dict()
        for name in (
            "time_weight_formula",
            "risk_limit",
            "fast_risk_limit",
            "activation_weight",
            "interaction_weight",
            "beam_width",
            "max_fast_evals",
            "max_full_evals",
            "max_interaction_evals",
            "strategy",
            "max_search_depth",
        ):
            if name in plan:
                raw[name] = plan[name]
        # A plan emitted by an Agent remains one algorithm-rule proposal; the
        # fair ablation suite is controlled by the outer recipe and is not
        # silently changed by a structural plan field.
        return TICARecipe.from_mapping(raw)

    def _run_tica_search(
        self,
        parent: CheckpointArtifact,
        profile: Mapping[str, Any],
        round_dir: Path,
        *,
        recipe: Optional[TICARecipe] = None,
        seed_plan: Optional[Mapping[str, Any]] = None,
        manifest: Optional[TICACalibrationManifest] = None,
        algorithm_source: Optional[str] = None,
        algorithm_history: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
        search_space: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Run stateful TICA search under one bounded, shared budget.

        Fast measurements come from the already executed local sensitivity
        probes. Full and interaction measurements execute actual structural
        candidates through AlgorithmModelEvaluator. No quality or interaction
        constants are introduced here.
        """

        manifest = manifest or self.tica_manifest
        if manifest is None:
            raise ValueError("TICA search requires an evaluation split manifest")
        recipe = recipe or self.tica_recipe
        score = tica_score(profile, time_weight_formula=recipe.time_weight_formula, time_weights=recipe.time_weights or None)
        evaluator = AlgorithmModelEvaluator(
            self.adapter,
            self.registry,
            parent,
            profile,
            round_dir / "tica-search-measurements",
            seed=int(profile.get("provenance", {}).get("seed", 7)) if isinstance(profile.get("provenance"), Mapping) else 7,
            calibration_manifest=manifest,
            search_budget=recipe.budget(),
        )

        def measure(plan: Mapping[str, Any], full: bool, stage: str) -> Mapping[str, Any]:
            operator = str(plan.get("operator", "tica_prune"))
            plan_args = plan.get("args") if isinstance(plan.get("args"), Mapping) else plan
            if not full:
                selected = [str(item) for item in plan_args.get("block_ids", [])]
                if operator == "timestep_block_gate":
                    selected = [
                        str(item.get("block_id"))
                        for item in plan_args.get("rules", [])
                        if isinstance(item, Mapping) and item.get("block_id")
                    ]
                risks = [score.block_risk.get(item) for item in selected if item in score.block_risk]
                if not risks and operator == "timestep_block_gate":
                    risks = [float(value) for value in score.block_risk.values()]
                return {
                    "valid": bool(risks),
                    "evidence_level": "cpu_fake_local_output" if manifest.provider_name == "fake_wan_cpu" else "real_wan_local_output",
                    "evaluation_status": "local_output_screen",
                    "quality_error": sum(risks) / max(len(risks), 1) if risks else None,
                    "risk_metric": "local_relative_output_error",
                    "risk_value": sum(risks) / max(len(risks), 1) if risks else None,
                    "risk_scale": "profile_relative_error",
                    "risk_is_approximate": True,
                    "selected_blocks": selected,
                    "time_weight_formula": recipe.time_weight_formula,
                    "stage": stage,
                    "operator": operator,
                }
            return evaluator.evaluate(
                SandboxResult(
                    "success",
                    "0" * 64,
                    SandboxCompressionPlan(operator, dict(plan_args)),
                    metadata={"stage": stage, "trusted_tica_baseline": operator == "tica_prune"},
                )
            )

        return TICAPruner().search(
            self.adapter,
            parent,
            manifest,
            profile,
            recipe,
            evaluate_candidate=measure,
            seed_plan=seed_plan,
            algorithm_source=algorithm_source,
            algorithm_history=algorithm_history,
            search_space=search_space,
            sandbox=self.sandbox,
        )

    def _screening_context(
        self,
        parent: CheckpointArtifact,
        profile: Mapping[str, Any],
        algorithm: AlgorithmCandidate,
        plan: SandboxCompressionPlan,
        *,
        full_evaluator: Optional[Callable[[Any], Mapping[str, Any]]] = None,
    ) -> ScreeningContext:
        search_space = self.search_space_version.to_dict() if self.search_space_version is not None else None
        if self.tica_prune:
            search_space = dict(search_space or {"operators": []})
            search_space["operators"] = [
                *list(search_space.get("operators", [])),
                tica_operator_spec(),
            ]
        return ScreeningContext(
            parent=parent,
            adapter=self.adapter,
            profile=profile,
            algorithm=algorithm,
            plan=plan,
            registry=self.registry,
            search_space_version=search_space,
            full_evaluator=full_evaluator,
        )

    def _algorithm_measurement_evaluator(
        self,
        parent: CheckpointArtifact,
        profiles: tuple[Mapping[str, Any], ...],
        output_root: Path,
        *,
        recipe: Optional[TICARecipe] = None,
        algorithm_sources: Optional[Mapping[str, str]] = None,
        algorithm_history: Sequence[Mapping[str, Any]] = (),
    ) -> Callable[..., Mapping[str, Any]]:
        """Build a scenario-aware callback for AlgorithmValidator.

        Each scenario gets one evaluator and therefore one fixed probe. The
        parent checkpoint is immutable; both parent and child plans are run
        from that same artifact, independent of the model-promotion path.
        """

        if self.tica_prune:
            return self._complete_tica_algorithm_measurement_evaluator(
                parent,
                profiles,
                output_root,
                recipe=recipe,
                algorithm_sources=algorithm_sources,
                algorithm_history=algorithm_history,
            )

        evaluator_recipe = recipe or self.tica_recipe
        algorithm_search_cache: dict[str, Mapping[str, Any]] = {}

        def evaluate(result: SandboxResult, scenario_profile: Mapping[str, Any]) -> Mapping[str, Any]:
            split = str(scenario_profile.get("evaluation_split", "validation")).lower()
            if split == "test":
                raise ValueError("TICA algorithm validation cannot consume Test split")
            scenario_manifest = self._tica_manifest_for_split(split) if self.tica_prune else None
            digest = stable_digest({"profile": scenario_profile, "manifest": scenario_manifest.digest() if scenario_manifest is not None else None})
            # Parent and child must not share a mutable evaluator/search
            # ledger.  Keying by source hash gives each algorithm its own
            # complete search while retaining identical model/input
            # conditions.
            if self.tica_prune and result.plan is not None and result.plan.operator == "tica_prune":
                return self._evaluate_tica_algorithm_search(
                    parent,
                    scenario_profile,
                    result,
                    output_root / ("algorithm-" + result.source_sha256[:16]) / digest,
                    evaluator_recipe=evaluator_recipe,
                )
            source = (algorithm_sources or {}).get(result.source_sha256)
            if source is not None:
                validated = validate_source(source)
                if "search" in validated.functions:
                    search_space_payload = (
                        self.search_space_version.to_dict()
                        if self.search_space_version is not None
                        else default_search_space().to_dict()
                    )
                    search_key = stable_digest({
                        "source_sha256": result.source_sha256,
                        "profile": scenario_profile,
                        "search_space": search_space_payload,
                    })
                    cached = algorithm_search_cache.get(search_key)
                    if cached is None:
                        seed_value = scenario_profile.get("provenance", {}).get("seed", 7) if isinstance(scenario_profile.get("provenance"), Mapping) else 7
                        cached = self._complete_standard_algorithm_search(
                            source=source,
                            proposed_result=result,
                            profile=scenario_profile,
                            history=algorithm_history,
                            parent=parent,
                            output_root=output_root / ("standard-" + result.source_sha256[:16]) / digest,
                            seed=int(seed_value or 7),
                            search_budget={
                                "max_candidates": 16,
                                "max_full_evals": 8,
                                "max_interaction_evals": 0,
                                "max_search_depth": 2,
                            },
                            search_space_payload=search_space_payload,
                        )
                        cached = {
                            **dict(cached),
                            "search_space_version": search_space_payload.get("version_id"),
                        }
                        algorithm_search_cache[search_key] = cached
                    return dict(cached)

            seed_value = scenario_profile.get("provenance", {}).get("seed", 7) if isinstance(scenario_profile.get("provenance"), Mapping) else 7
            evaluator = AlgorithmModelEvaluator(
                self.adapter,
                self.registry,
                parent,
                scenario_profile,
                output_root / digest,
                seed=int(seed_value or 7),
                calibration_manifest=scenario_manifest if self.tica_prune else None,
                search_budget=evaluator_recipe.budget() if self.tica_prune else {"max_fast_evals": 8, "max_full_evals": 4, "max_interaction_evals": 2},
            )
            return dict(evaluator.evaluate(result))

        return evaluate

    def _complete_standard_algorithm_search(
        self,
        *,
        source: str,
        proposed_result: SandboxResult,
        profile: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        parent: CheckpointArtifact,
        output_root: Path,
        seed: int,
        search_budget: Mapping[str, int],
        search_space_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Run a bounded, source-controlled multi-candidate standard search.

        The Agent owns Block-subset generation and ranking. The Harness owns
        equal-budget measurement from one immutable parent, quality-floor
        enforcement, duplicate filtering, and final measured selection. A
        second depth is reserved for quality-guided Block restoration or
        replacement.
        """

        max_candidates = int(search_budget.get("max_candidates", 16))
        max_full_evals = int(search_budget.get("max_full_evals", max_candidates))
        max_depth = max(1, min(3, int(search_budget.get("max_search_depth", 2))))
        if max_candidates <= 0 or max_full_evals <= 0:
            raise ValueError("standard_algorithm_search_budget_invalid")
        block_search_space = dict(search_space_payload)
        block_search_space["operators"] = [
            dict(item) for item in search_space_payload.get("operators", ()) if isinstance(item, Mapping)
        ]
        if not block_search_space["operators"]:
            raise ValueError("standard_search_space_has_no_operators")
        quality_floor_ratio = float(search_budget.get("quality_floor_ratio", self.promotion_policy.quality_floor_ratio))
        if not math.isfinite(quality_floor_ratio) or not 0.0 <= quality_floor_ratio <= 1.0:
            raise ValueError("standard_algorithm_search_quality_floor_invalid")
        evaluator = AlgorithmModelEvaluator(
            self.adapter,
            self.registry,
            parent,
            profile,
            Path(output_root),
            seed=int(seed),
            search_budget={
                "max_candidates": max_candidates,
                "max_full_evals": max_full_evals,
                "max_interaction_evals": int(search_budget.get("max_interaction_evals", 0)),
            },
            probe_sample_count=int(search_budget.get("probe_sample_count", self.algorithm_probe_samples)),
        )
        previous_digests = self._previously_measured_plan_digests(
            parent.sha256,
            exclude_round_id=Path(output_root).parent.name,
        )
        prior_feedback = self._algorithm_search_feedback_history(
            limit=4,
            parent_checkpoint_sha256=parent.sha256,
        )
        prior_candidates: list[Mapping[str, Any]] = []
        prior_plans: list[Mapping[str, Any]] = []
        for feedback in prior_feedback:
            for raw in feedback.get("measured_candidates", ()):
                if not isinstance(raw, Mapping) or raw.get("valid") is not True:
                    continue
                plan = raw.get("plan")
                if not isinstance(plan, Mapping):
                    continue
                recovery_record = {
                    **dict(raw),
                    "evaluation_position": len(prior_candidates),
                }
                if measured_recovery_key(recovery_record) is None:
                    continue
                prior_candidates.append(recovery_record)
                prior_plans.append(dict(plan))

        prior_best = max(prior_candidates, key=measured_recovery_key, default=None)
        initial_depth = min(2, max_depth) if prior_best is not None else 1
        seen: set[str] = set(previous_digests)
        measured: list[Mapping[str, Any]] = []
        decisions: list[Mapping[str, Any]] = []
        skipped: list[Mapping[str, Any]] = []
        remaining = max_full_evals
        state: Mapping[str, Any] = {
            # The Agent's depth-2 recovery policy consumes measured near misses.
            # Seed it from recent rounds on this exact immutable parent so a
            # new round can propose unseen restores/replacements immediately.
            "depth": initial_depth,
            "parent_state_id": (
                "cross_round_feedback:%s" % str(prior_feedback[0].get("round_id", ""))
                if prior_best is not None and prior_feedback
                else "uncompressed"
            ),
            "proposed_plan": (
                dict(prior_best["plan"])
                if prior_best is not None
                else proposed_result.plan.to_dict() if proposed_result.plan is not None else None
            ),
            "budget": {"max_candidates": max_candidates, "remaining_full_evals": max_full_evals, "max_full_evals": max_full_evals},
            "quality_floor_ratio": quality_floor_ratio,
            "quality_floor_basis": "parent_relative_probe_similarity_with_parent_baseline_1.0",
            "algorithm_improvement": bool(self.algorithm_improvement),
            "measured_candidates": prior_candidates,
            "frontier": prior_plans,
        }
        selected_plan: Optional[SandboxCompressionPlan] = None
        selected_metrics: Optional[Mapping[str, Any]] = None
        selected_record: Optional[Mapping[str, Any]] = None
        metrics_by_digest: dict[str, Mapping[str, Any]] = {}

        for depth in range(initial_depth, max_depth + 1):
            if remaining <= 0:
                break
            decision = _run_algorithm_search_program(source, self.sandbox, profile, history, state, block_search_space)
            decisions.append({
                "depth": depth,
                "source_sha256": decision.source_sha256,
                "payload_sha256": decision.payload_sha256,
                "plan_digests": [stable_digest(item) for item in decision.plans],
                "rank": list(decision.rank),
                "stop": decision.stop,
            })
            depth_records: list[Mapping[str, Any]] = []
            operator_specs = _operator_specs_from_search_space(block_search_space)
            eligible = []
            unique = set(seen)
            for index in decision.rank:
                digest = stable_digest(decision.plans[index])
                if digest in unique:
                    skipped.append({
                        "depth": depth,
                        "candidate_index": index,
                        "plan": decision.plans[index],
                        "plan_digest": digest,
                        "reason": "previously_measured_same_parent" if digest in previous_digests else "duplicate",
                    })
                else:
                    unique.add(digest)
                    eligible.append(index)
            family_counts: dict[str, int] = {}
            for feedback in prior_feedback:
                counts = feedback.get("measured_family_counts", {})
                if isinstance(counts, Mapping):
                    for family, count in counts.items():
                        family_counts[str(family)] = family_counts.get(str(family), 0) + int(count)
            for previous in measured:
                previous_plan = previous.get("plan") if isinstance(previous, Mapping) else None
                if isinstance(previous_plan, Mapping):
                    family = candidate_family(previous_plan, operator_specs)
                    family_counts[family] = family_counts.get(family, 0) + 1
            evaluation_order = budgeted_candidate_order(
                decision.plans, eligible, operator_specs, remaining, family_counts=family_counts,
            )
            skipped.extend(
                {"depth": depth, "candidate_index": index, "plan": decision.plans[index], "plan_digest": stable_digest(decision.plans[index]), "reason": "budget"}
                for index in eligible if index not in evaluation_order
            )
            decisions[-1] = {
                **dict(decisions[-1]),
                "diagnostics": dict(decision.diagnostics),
                "evaluation_order": [
                    {
                        "rank_position": rank_position,
                        "candidate_index": index,
                        "family": candidate_family(decision.plans[index], operator_specs),
                        "operator": str(decision.plans[index].get("operator", "")),
                    }
                    for rank_position, index in enumerate(evaluation_order)
                ],
            }
            for rank_position, rank_index in enumerate(evaluation_order):
                if remaining <= 0:
                    break
                raw_plan = decision.plans[rank_index]
                plan_digest = stable_digest(raw_plan)
                if plan_digest in seen:
                    continue
                seen.add(plan_digest)
                plan = SandboxCompressionPlan.from_mapping(raw_plan, operator_specs=self.sandbox.operator_specs)
                candidate = SandboxResult(
                    "success",
                    decision.source_sha256,
                    plan,
                    metadata={"algorithm_program_search": True, "search_depth": depth, "search_rank_index": rank_index},
                )
                metrics = dict(evaluator.evaluate(candidate))
                metrics_by_digest[plan_digest] = metrics
                remaining -= 1
                efficiency = efficiency_reduction_ratio(metrics)
                score = measured_search_score(metrics)
                record = {
                    "depth": depth,
                    "rank_index": int(rank_index),
                    "rank_position": int(decision.rank.index(rank_index)),
                    "evaluation_position": int(rank_position),
                    "plan_digest": plan_digest,
                    "plan": plan.to_dict(),
                    "family": candidate_family(plan.to_dict(), operator_specs),
                    "parent_checkpoint_sha256": parent.sha256,
                    "recovery_action": (
                        decision.diagnostics.get("candidate_actions", [])[rank_index]
                        if isinstance(decision.diagnostics.get("candidate_actions"), list)
                        and rank_index < len(decision.diagnostics.get("candidate_actions", []))
                        else None
                    ),
                    "valid": bool(metrics.get("valid") is True),
                    "quality_score": metrics.get("quality_score"),
                    "quality_floor_ratio": quality_floor_ratio,
                    "quality_floor_basis": "parent_relative_probe_similarity_with_parent_baseline_1.0",
                    "quality_feasible": bool(
                        metrics.get("valid") is True
                        and isinstance(metrics.get("quality_score"), (int, float))
                        and not isinstance(metrics.get("quality_score"), bool)
                        and math.isfinite(float(metrics["quality_score"]))
                        and float(metrics["quality_score"]) >= quality_floor_ratio
                    ),
                    "search_score": score,
                    "efficiency_reduction_ratio": efficiency,
                    "risk_value": metrics.get("risk_value", metrics.get("quality_error")),
                    "quality_error": metrics.get("quality_error"),
                    "failure_reason": metrics.get("failure_reason"),
                }
                # Keep actual candidate execution evidence in the search
                # trace.  This makes runtime-only candidates comparable and
                # preserves the explicit boundary between CPU counters and
                # independent GPU timing.
                for key in (
                    "evidence_level",
                    "structural_change",
                    "parameter_reduction_ratio",
                    "compute_metrics",
                    "estimated_flops",
                    "runtime_execution",
                    "hardware_metrics",
                    "probe_coverage",
                    "ranking_evidence",
                ):
                    if key in metrics:
                        record[key] = metrics[key]
                depth_records.append(record)
                measured.append(record)

                record_key = measured_candidate_key(record)
                selected_key = measured_candidate_key(selected_record) if selected_record is not None else None
                if record_key is not None and (selected_key is None or record_key > selected_key):
                    selected_metrics = metrics
                    selected_plan = plan
                    selected_record = record

            if decision.stop or remaining <= 0:
                break
            recoverable = [item for item in depth_records if measured_recovery_key(item) is not None]
            if not recoverable:
                break
            best = max(recoverable, key=measured_recovery_key)
            state = {
                "depth": depth + 1,
                "parent_state_id": str(best["plan_digest"]),
                "proposed_plan": dict(best["plan"]),
                "budget": {"max_candidates": min(max_candidates, remaining), "remaining_full_evals": remaining, "max_full_evals": max_full_evals},
                "quality_floor_ratio": quality_floor_ratio,
                "quality_floor_basis": "parent_relative_probe_similarity_with_parent_baseline_1.0",
                "algorithm_improvement": bool(self.algorithm_improvement),
                "measured_candidates": depth_records,
                "frontier": [dict(item["plan"]) for item in recoverable],
            }

        pareto_indices = pareto_front_indices(measured)
        pareto_rows = [measured[index] for index in pareto_indices]
        feasible_rows = [item for item in pareto_rows if item.get("quality_feasible") is True]
        selection_pool = feasible_rows if feasible_rows else pareto_rows
        selected_record = max(
            selection_pool,
            key=lambda item: measured_pareto_key(item) or tuple(),
            default=None,
        )
        if selected_record is not None:
            selected_plan = SandboxCompressionPlan.from_mapping(
                selected_record["plan"], operator_specs=self.sandbox.operator_specs
            )
            selected_metrics = metrics_by_digest.get(str(selected_record.get("plan_digest", "")))
        selected_status = "quality_feasible" if feasible_rows else "validation_only_near_miss"
        algorithm_search = {
            "protocol": "standard_hbps",
            "ranking_version": SEARCH_RANKING_VERSION,
            "candidate_scope": "registered_multigranular",
            "parent_checkpoint_sha256": parent.sha256,
            "quality_floor_ratio": quality_floor_ratio,
            "quality_floor_basis": "parent_relative_probe_similarity_with_parent_baseline_1.0",
            "complete": True,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "candidate_count": sum(len(item.get("plan_digests", ())) for item in decisions),
            "evaluated_candidate_count": len(measured),
            "previously_measured_plan_count": len(previous_digests),
            "prior_measurement_round_ids": [str(item.get("round_id", "")) for item in prior_feedback],
            "prior_measurement_candidate_count": len(prior_candidates),
            "initial_search_depth": initial_depth,
            "candidate_results": measured,
            "valid_candidate_count": sum(row.get("valid") is True for row in measured),
            "quality_feasible_candidate_count": sum(measured_candidate_key(row) is not None for row in measured),
            "pareto_front_indices": list(pareto_indices),
            "pareto_front_plan_digests": [str(measured[index].get("plan_digest", "")) for index in pareto_indices],
            "pareto_objectives": ["quality_score:max", "efficiency_reduction_ratio:max", "risk_value:min"],
            "selection_status": selected_status if selected_record is not None else "no_valid_measurement",
            "not_evaluated": skipped,
            "selection_reason": "feasibility_then_pareto_quality_efficiency_risk_then_source_rank",
            "budget": {**dict(search_budget), "full_evals_used": len(measured)},
            "trace": decisions,
        }
        if selected_plan is None or selected_metrics is None or selected_record is None:
            return {
                "valid": False,
                "failure_reason": "standard_algorithm_search_no_valid_candidate",
                "algorithm_search": {**algorithm_search, "complete": False},
            }
        result = {**dict(selected_metrics), "search_score": selected_record.get("search_score")}
        if selected_status != "quality_feasible":
            result["validation_only"] = True
            result["validation_only_reason"] = "no_calibration_candidate_met_quality_floor"
        result["algorithm_search"] = {
            **algorithm_search,
            "selected_plan_digest": stable_digest(selected_plan.to_dict()),
            "selected_rank_index": int(selected_record["rank_index"]),
            "selected_rank_position": int(selected_record["rank_position"]),
        }
        result["selected_plan"] = selected_plan.to_dict()
        return result

    def _complete_tica_algorithm_measurement_evaluator(
        self,
        parent: CheckpointArtifact,
        profiles: tuple[Mapping[str, Any], ...],
        output_root: Path,
        *,
        recipe: Optional[TICARecipe] = None,
        algorithm_sources: Optional[Mapping[str, str]] = None,
        algorithm_history: Sequence[Mapping[str, Any]] = (),
    ) -> Callable[..., Mapping[str, Any]]:
        """Compare TICA rules by running each rule's complete Calibration search.

        Sandbox only validates the rule boundary.  The returned callback then
        executes an independent TICAPruner search for each source, from the
        same immutable checkpoint, calibration manifest, search-space version,
        seed and budget.  Only the selected Calibration candidate is measured
        on the independent Validation manifest; Test is unreachable here.
        """

        if self.tica_manifest is None or self.tica_splits is None:
            raise ValueError("complete TICA algorithm validation requires split manifests")
        evaluator_recipe = recipe or self.tica_recipe
        calibration_cache: dict[str, Mapping[str, Any]] = {}
        search_cache: dict[str, Any] = {}
        validation_evaluators: dict[str, AlgorithmModelEvaluator] = {}
        output_root = Path(output_root)

        def calibration_profile() -> Mapping[str, Any]:
            key = stable_digest({"parent": parent.sha256, "manifest": self.tica_manifest.digest()})
            if key in calibration_cache:
                return calibration_cache[key]
            model = self.adapter.reload_model(parent)
            profile = dict(collect_tica_profile(self.adapter, model, self.tica_manifest))
            profile["evaluation_split"] = "calibration"
            profile["parent_checkpoint_sha256"] = parent.sha256
            profile["profiling_digest"] = stable_digest({"profile": profile, "manifest": self.tica_manifest.digest()})
            calibration_cache[key] = profile
            return profile

        def evaluate(result: SandboxResult, scenario_profile: Mapping[str, Any]) -> Mapping[str, Any]:
            if result.plan is None:
                return {"valid": False, "failure_reason": "algorithm_plan_missing"}
            if str(scenario_profile.get("evaluation_split", "validation")).lower() == "test":
                raise ValueError("TICA algorithm validation cannot consume Test split")
            rule = self._tica_recipe_from_plan(result.plan.args)
            rule_digest = stable_digest(
                {
                    "source_sha256": result.source_sha256,
                    "plan": result.plan.to_dict(),
                    "recipe": rule.to_dict(),
                    "parent": parent.sha256,
                    "calibration": self.tica_manifest.digest(),
                }
            )
            search = search_cache.get(rule_digest)
            if search is None:
                search = self._run_tica_search(
                    parent,
                    calibration_profile(),
                    output_root / "search" / rule_digest,
                    recipe=rule,
                    seed_plan=result.plan.args,
                    algorithm_source=(algorithm_sources or {}).get(result.source_sha256),
                    algorithm_history=algorithm_history,
                    search_space=self.search_space_version.to_dict() if self.search_space_version is not None else {"operators": []},
                )
                search_cache[rule_digest] = search
            selected = next(
                (item for item in search.candidates if item.candidate_id == search.selected_candidate_id),
                None,
            )
            base = {
                "parent_checkpoint_sha256": parent.sha256,
                "calibration_digest": search.calibration_digest,
                "search_space_version": self.search_space_version.version_id if self.search_space_version is not None else "default",
                "search_seed": int((calibration_profile().get("provenance") or {}).get("seed", 7)) if isinstance(calibration_profile().get("provenance"), Mapping) else 7,
                "profiling_digest": calibration_profile().get("profiling_digest"),
                "search_budget": rule.budget(),
                "search_usage": dict(search.budget),
                "search_cost": {
                    "observed_depth": int(search.observed_search_depth),
                    "termination_reason": search.termination_reason,
                    "candidate_count": len(search.candidates),
                    "feasible_count": sum(1 for item in search.candidates if item.feasible),
                    "level_stats": list(search.level_stats),
                    "budget_audit": list(search.budget_audit),
                },
                "profiling_cost": dict(search.profiling_cost),
                "total_cost": dict(search.total_cost),
                "pareto_front": list(search.pareto_front),
                "selected_candidate_id": search.selected_candidate_id,
                "selected_plan_digest": search.selected_plan_digest,
                "search_result_digest": stable_digest(search.to_dict()),
                "algorithm_search": {
                    "complete": selected is not None,
                    "source_sha256": result.source_sha256,
                    "candidate_count": len(search.candidates),
                    "selected_candidate_id": search.selected_candidate_id,
                    "candidates": [item.to_dict() for item in search.candidates],
                },
            }
            if selected is None or not selected.feasible:
                return {**base, "valid": False, "failure_reason": "no_feasible_calibration_candidate"}
            validation_digest = stable_digest({"profile": scenario_profile, "manifest": self.tica_splits.validation.digest()})
            evaluator = validation_evaluators.get(validation_digest)
            if evaluator is None:
                seed_value = scenario_profile.get("provenance", {}).get("seed", 7) if isinstance(scenario_profile.get("provenance"), Mapping) else 7
                evaluator = AlgorithmModelEvaluator(
                    self.adapter,
                    self.registry,
                    parent,
                    scenario_profile,
                    output_root / "validation" / validation_digest,
                    seed=int(seed_value or 7),
                    calibration_manifest=self.tica_splits.validation,
                    search_budget=rule.budget(),
                )
                validation_evaluators[validation_digest] = evaluator
            validation_result = dict(
                evaluator.evaluate(
                    SandboxResult(
                        "success",
                        result.source_sha256,
                        SandboxCompressionPlan(
                            str(selected.plan.get("operator", "tica_prune"))
                            if isinstance(selected.plan, Mapping) and isinstance(selected.plan.get("args"), Mapping)
                            else "tica_prune",
                            dict(selected.plan.get("args", {}))
                            if isinstance(selected.plan, Mapping) and isinstance(selected.plan.get("args"), Mapping)
                            else dict(selected.plan),
                        ),
                        metadata={"complete_tica_search": True, "calibration_search_digest": base["search_result_digest"]},
                    )
                )
            )
            return {
                **validation_result,
                **base,
                "valid": bool(validation_result.get("valid") is True and validation_result.get("output_metrics")),
                "validation_selected_candidate": selected.to_dict(),
                "validation_evaluation": validation_result,
            }

        return evaluate

    def _maybe_evolve_search_space(self) -> None:
        """Propose an extension only after persisted experiments show a bottleneck."""

        if self.search_space_manager is None or self.search_space_version is None:
            return
        feedback = self._search_space_feedback()
        self.search_space_feedback = feedback
        _write_json(self.output_root / "search-space-feedback.json", feedback)
        if not bool(feedback.get("ready")):
            return
        if any(item.status in {"proposed", "validating"} for item in self.search_space_manager.store.read_proposals()):
            return
        # A validated active extension is already the result of an earlier
        # evidence gate. Do not repeatedly rediscover the same capability.
        # Permit the first two evidence-backed extensions (the original
        # timestep gate and then a timestep-conditioned attention budget).
        # Later generations must not grow the registry without a new audit
        # rule, otherwise a noisy CPU run could keep adding duplicate ideas.
        if len(self.search_space_version.operators) > len(default_search_space().operators) + 1:
            return
        history = self._feedback_history()
        failures = tuple(str(item) for item in feedback.get("signals", ()))
        suggestion = self.agent.propose_search_space_change(
            search_space=self.search_space_version.to_dict(),
            history=history,
            failures=failures,
        )
        spec = OperatorSpec.from_dict(suggestion["spec"])
        evidence_ids = tuple(dict.fromkeys([*feedback.get("evidence_ids", ()), *suggestion.get("evidence_ids", ())]))
        proposal = self.search_space_manager.propose(
            spec,
            str(suggestion.get("rationale", "search-space extension")),
            evidence_ids,
            trigger_evidence=feedback,
        )
        proposal = self.search_space_manager.validate(
            proposal,
            self.adapter,
            self.parent,
            output_dir=self.output_root / "search-space" / "validation" / proposal.proposal_id,
        )
        proposal = replace(
            proposal,
            validation={
                **dict(proposal.validation),
                "change_taxonomy": classify_search_space_proposal(
                    proposal.spec.to_dict(),
                    self.search_space_version.to_dict(),
                    proposal.validation,
                ),
            },
        )
        self.search_space_manager.store.write_proposal(proposal)
        self.search_space_proposal = proposal
        if proposal.status == "validated":
            self.search_space_version = self.search_space_manager.register(proposal)
            self._record_search_space_gene()

    def _feedback_history(self) -> list[Mapping[str, Any]]:
        records: list[Mapping[str, Any]] = []
        for item in sorted(self.experience_store.read(), key=lambda value: str(value.experience_id)):
            records.append(
                {
                    "round_id": item.experience_id,
                    "algorithm_id": item.training.get("algorithm_id"),
                    "decision": dict(item.decision),
                    "evaluation": dict(item.evaluation or {}),
                    "training": dict(item.training),
                    "failure_reason": item.decision.get("failure_type"),
                }
            )
        return records[-16:]

    def _algorithm_search_feedback_history(
        self,
        *,
        limit: Optional[int] = 4,
        parent_checkpoint_sha256: Optional[str] = None,
    ) -> list[Mapping[str, Any]]:
        """Read compactable, measured Block outcomes from prior round traces."""

        round_dirs = sorted(
            self.rounds_root.glob("R*"),
            key=lambda path: int(path.name[1:]) if path.name[1:].isdigit() else -1,
        )
        records: list[Mapping[str, Any]] = []
        for round_dir in round_dirs:
            trace_path = round_dir / "algorithm-program-trace.json"
            if not trace_path.exists():
                continue
            try:
                trace = self._read_json(trace_path)
                measured = trace.get("measured_search")
                if not isinstance(measured, Mapping):
                    continue
                raw_candidates = measured.get("candidate_results")
                if not isinstance(raw_candidates, list):
                    continue
                floor = measured.get("quality_floor_ratio")
                if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not math.isfinite(float(floor)):
                    floor = None
                candidates: list[Mapping[str, Any]] = []
                for raw in raw_candidates:
                    if not isinstance(raw, Mapping):
                        continue
                    plan = raw.get("plan")
                    if not isinstance(plan, Mapping) or not isinstance(plan.get("args"), Mapping):
                        continue

                    def finite_metric(name: str) -> Optional[float]:
                        value = raw.get(name)
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                            return float(value)
                        return None

                    plan_copy = {"operator": str(plan.get("operator", "")), "args": dict(plan.get("args", {}))}
                    probe_coverage = raw.get("probe_coverage")
                    candidates.append(
                        {
                            "plan": plan_copy,
                            "plan_digest": str(raw.get("plan_digest") or stable_digest(plan_copy)),
                            "candidate_index": raw.get("rank_index"),
                            "rank_position": raw.get("rank_position"),
                            "operator": plan_copy["operator"],
                            "family": candidate_family(plan_copy, {}),
                            "valid": raw.get("valid") is True,
                            "quality_score": finite_metric("quality_score"),
                            "quality_floor_ratio": finite_metric("quality_floor_ratio") if finite_metric("quality_floor_ratio") is not None else floor,
                            "quality_feasible": raw.get("quality_feasible") is True,
                            "search_score": finite_metric("search_score"),
                            "efficiency_reduction_ratio": finite_metric("efficiency_reduction_ratio"),
                            "risk_value": finite_metric("risk_value"),
                            "quality_error": finite_metric("quality_error"),
                            "failure_reason": str(raw.get("failure_reason")) if raw.get("failure_reason") else None,
                            "recovery_action": str(raw.get("recovery_action")) if raw.get("recovery_action") else None,
                            "video_quality_evaluated": bool(
                                isinstance(probe_coverage, Mapping)
                                and probe_coverage.get("video_quality") is True
                            ),
                            "depth": int(raw.get("depth", 1)) if isinstance(raw.get("depth", 1), int) else 1,
                        }
                    )
                if not candidates:
                    continue
                decision_path = round_dir / "decision.json"
                decision: Mapping[str, Any] = {}
                if decision_path.exists():
                    try:
                        decision = self._read_json(decision_path)
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        decision = {}
                feasible_count = sum(item.get("quality_feasible") is True for item in candidates)
                tested_block_sets: list[list[str]] = []
                seen_block_sets: set[tuple[str, ...]] = set()
                for item in candidates:
                    if item.get("operator") != "prune_blocks":
                        continue
                    block_ids = item["plan"]["args"]["block_ids"]
                    block_key = tuple(block_ids)
                    if block_key not in seen_block_sets:
                        seen_block_sets.add(block_key)
                        tested_block_sets.append(list(block_ids))
                summarized_candidates = sorted(
                    candidates,
                    key=lambda item: (
                        item.get("quality_feasible") is True,
                        item.get("valid") is True,
                        item.get("quality_score") if isinstance(item.get("quality_score"), (int, float)) else float("-inf"),
                        item.get("search_score") if isinstance(item.get("search_score"), (int, float)) else float("-inf"),
                        item.get("efficiency_reduction_ratio") if isinstance(item.get("efficiency_reduction_ratio"), (int, float)) else float("-inf"),
                    ),
                    reverse=True,
                )
                family_counts: dict[str, int] = {}
                for item in candidates:
                    family = str(item.get("family", "other"))
                    family_counts[family] = family_counts.get(family, 0) + 1
                round_id = round_dir.name
                records.append(
                    {
                        "memory_kind": "algorithm_search_feedback",
                        "experience_id": round_id,
                        "round_id": round_id,
                        "algorithm_id": str(trace.get("algorithm_id", decision.get("algorithm_id", ""))),
                        "parent_checkpoint_sha256": str(measured.get("parent_checkpoint_sha256", "")),
                        "quality_floor_ratio": floor,
                        "quality_floor_basis": str(measured.get("quality_floor_basis", "")),
                        "candidate_count": len(candidates),
                        "measured_candidate_summary_count": len(summarized_candidates),
                        "measured_family_counts": family_counts,
                        "quality_feasible_candidate_count": feasible_count,
                        "search_outcome": "quality_feasible_candidate_found" if feasible_count else "no_quality_feasible_candidate",
                        "failure_reason": str(decision.get("failure_type", "quality_floor_not_met" if feasible_count == 0 else "")) or None,
                        "tested_block_sets": tested_block_sets,
                        "all_plan_digests": [str(item["plan_digest"]) for item in candidates],
                        "measured_candidates": summarized_candidates,
                    }
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue

        records.reverse()
        if parent_checkpoint_sha256 is not None:
            records = [
                item for item in records
                if str(item.get("parent_checkpoint_sha256", "")) == str(parent_checkpoint_sha256)
            ]
        if limit is None:
            return records
        return records[:max(0, int(limit))]

    def _previously_measured_plan_digests(
        self,
        parent_checkpoint_sha256: str,
        *,
        exclude_round_id: Optional[str] = None,
    ) -> set[str]:
        """Return plans from any granularity already probed on this exact parent."""

        digests: set[str] = set()
        for feedback in self._algorithm_search_feedback_history(limit=None):
            if str(feedback.get("round_id", "")) == str(exclude_round_id or ""):
                continue
            if str(feedback.get("parent_checkpoint_sha256", "")) != str(parent_checkpoint_sha256):
                continue
            for digest in feedback.get("all_plan_digests", ()):
                if digest:
                    digests.add(str(digest))
            for candidate in feedback.get("measured_candidates", ()):
                if not isinstance(candidate, Mapping):
                    continue
                digest = candidate.get("plan_digest")
                plan = candidate.get("plan")
                if digest:
                    digests.add(str(digest))
                elif isinstance(plan, Mapping):
                    digests.add(stable_digest(plan))
        return digests

    def _improve_unique_algorithm(
        self,
        *,
        algorithm_id: str,
        parent: AlgorithmCandidate,
        parent_id: Optional[str],
        profile: Mapping[str, Any],
        history: list[Mapping[str, Any]],
        failures: tuple[str, ...],
        skills: list[Mapping[str, Any]],
        search_space: Mapping[str, Any],
        protocol_mode: str,
        round_id: str,
    ) -> tuple[AlgorithmCandidate, AlgorithmCandidate]:
        """Retry one duplicate source from its existing lineage with explicit feedback."""

        current_parent = parent
        current_history = list(history)
        current_failures = tuple(failures)
        duplicate_attempts: list[dict[str, Any]] = []
        for attempt in range(2):
            candidate = self.agent.improve(
                algorithm_id,
                parent=current_parent,
                parent_id=parent_id,
                profile=profile,
                history=current_history,
                failures=current_failures,
                skills=skills,
                search_space=search_space,
                protocol_mode=protocol_mode,
            )
            duplicate = self.algorithm_store.duplicate_of(candidate)
            if duplicate is None:
                if duplicate_attempts:
                    candidate = replace(
                        candidate,
                        metadata={
                            **dict(candidate.metadata),
                            "duplicate_source_attempts": duplicate_attempts,
                        },
                    )
                return candidate, current_parent
            duplicate_attempts.append(
                {
                    "attempt": attempt + 1,
                    "candidate_source_sha256": candidate.source_sha256,
                    "duplicate_algorithm_id": duplicate.algorithm_id,
                    "duplicate_algorithm_source_sha256": duplicate.source_sha256,
                }
            )
            if attempt == 1:
                # Reuse the evidence-selected AI-authored program with the
                # current measured history. Candidate-level dedupe prevents
                # repeating plans already tested on this parent checkpoint.
                return (
                    replace(
                        parent,
                        metadata={
                            **dict(parent.metadata),
                            "reused_existing_algorithm_after_duplicate": True,
                            "duplicate_source_attempts": duplicate_attempts,
                        },
                    ),
                    parent,
                )
            retry_feedback = {
                "memory_kind": "algorithm_duplicate_feedback",
                "experience_id": round_id,
                "round_id": round_id,
                "algorithm_id": duplicate.algorithm_id,
                "duplicate_algorithm_id": duplicate.algorithm_id,
                "candidate_source_sha256": candidate.source_sha256,
                "duplicate_algorithm_source_sha256": duplicate.source_sha256,
                "duplicate_attempt": attempt + 1,
                "failure_reason": "duplicate_algorithm_source",
            }
            current_parent = duplicate
            current_history = [retry_feedback, *history]
            current_failures = tuple(dict.fromkeys([*failures, "duplicate_algorithm_source"]))
        raise RuntimeError("algorithm_improvement_retry_exhausted")

    def _search_space_feedback(self) -> dict[str, Any]:
        rounds = self._feedback_history()
        evidence_ids = [str(item.get("round_id")) for item in rounds if item.get("round_id")]
        plan_digests: dict[str, list[str]] = {}
        failure_counts: dict[str, int] = {}
        stagnant_rounds = 0
        diminishing_gain_rounds: list[str] = []
        for item in reversed(rounds):
            decision = item.get("decision") if isinstance(item.get("decision"), Mapping) else {}
            evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), Mapping) else {}
            plan = decision.get("algorithm_plan") or {"operator": decision.get("operator"), "args": decision.get("operator_args")}
            if plan and plan.get("operator"):
                digest = stable_digest(plan)
                plan_digests.setdefault(digest, []).append(str(item.get("round_id")))
            reasons = []
            promotion = decision.get("promotion") if isinstance(decision.get("promotion"), Mapping) else {}
            reasons.extend(str(value) for value in promotion.get("reasons", []) if str(value).strip())
            if decision.get("failure_type"):
                reasons.append(str(decision["failure_type"]))
            for reason in set(reasons):
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            if evaluation.get("failure_type"):
                reasons.append(str(evaluation["failure_type"]))
            for reason in reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            improved = bool(promotion.get("accepted")) and not bool(evaluation.get("critical_regression"))
            if not improved:
                stagnant_rounds += 1
                hardware = evaluation.get("hardware_metrics") if isinstance(evaluation.get("hardware_metrics"), Mapping) else {}
                has_efficiency_signal = any(
                    isinstance(value, (int, float)) and float(value) < 1.0
                    for value in hardware.values()
                ) or float(evaluation.get("estimated_compute_ratio", 1.0) or 1.0) < 1.0
                if has_efficiency_signal and item.get("round_id"):
                    diminishing_gain_rounds.append(str(item["round_id"]))
            else:
                break
        duplicate_plans = [round_ids for round_ids in plan_digests.values() if len(round_ids) >= 2]
        repeated_failures = {reason: count for reason, count in failure_counts.items() if count >= 2}
        signals: list[str] = []
        if stagnant_rounds >= 2:
            signals.append("pareto_stagnation")
        if duplicate_plans:
            signals.append("repeated_algorithm_plan")
        if repeated_failures:
            signals.append("repeated_failure")
        # A candidate can make a real structural/runtime improvement while
        # still missing the promotion threshold.  Treat repeated evidence of
        # that diminishing return as a bottleneck instead of waiting for two
        # completely failed rounds; this is especially important after the
        # default evaluator starts measuring head/FFN/runtime changes fairly.
        if diminishing_gain_rounds:
            signals.append("latency_diminishing_efficiency_gain")
        ready = len(rounds) >= 2 and bool(signals)
        return {
            "ready": ready,
            "required_rounds": 2,
            "observed_rounds": len(rounds),
            "stagnant_rounds": stagnant_rounds,
            "signals": signals,
            "repeated_failures": repeated_failures,
            "diminishing_gain_rounds": list(reversed(diminishing_gain_rounds)),
            "duplicate_plan_rounds": duplicate_plans,
            "evidence_ids": evidence_ids[-16:],
            "bottleneck": ";".join(signals) or "insufficient_evidence",
            "hypothesis": "the active operator closure cannot improve the observed bottleneck under the current budget",
            "expected_gain": "increase reachable compression/quality trade-offs only if validation supplies a structural counterexample",
        }

    def _build_validation_profiles(self, parent: CheckpointArtifact, profile: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """Create fixed, same-parent diagnostic scenarios for algorithm comparison."""

        model = self.adapter.reload_model(parent)
        provenance = profile.get("provenance") if isinstance(profile, Mapping) else None
        seed = int(provenance.get("seed", 7) or 7) if isinstance(provenance, Mapping) else 7
        if str(profile.get("model", {}).get("checkpoint_sha256", "")) == parent.sha256:
            base = dict(profile)
        else:
            base = profile_model(
                self.adapter,
                model,
                seed=seed,
                advanced=self.advanced_profiling,
                channel_chunk_size=16 if self.advanced_profiling else None,
                channel_method=getattr(self, "channel_method", "weight"),
            ).to_dict()
        alternate = profile_model(
            self.adapter,
            model,
            seed=seed + 1,
            advanced=self.advanced_profiling,
            channel_chunk_size=16 if self.advanced_profiling else None,
            channel_method=getattr(self, "channel_method", "weight"),
        ).to_dict()
        if self.tica_prune:
            # Validation is an independent comparison/evaluation input.  It
            # must not be profiled into a new TICA search profile: all search
            # rules, candidate generation, and search scoring are bound to the
            # immutable Calibration profile collected by the complete-search
            # evaluator.  Keep only structural/provenance metadata here.
            def validation_view(value: Mapping[str, Any]) -> dict[str, Any]:
                view = {
                    key: item
                    for key, item in dict(value).items()
                    if key not in {"tica", "tica_block_sensitivity", "tica_channel_sensitivity", "profiling_cost"}
                }
                view["evaluation_split"] = "validation"
                provenance_view = dict(view.get("provenance") or {})
                provenance_view["search_profile_source"] = "calibration_only"
                view["provenance"] = provenance_view
                return view

            base = validation_view(base)
            alternate = validation_view(alternate)
        alternate_provenance = dict(alternate.get("provenance") or {})
        alternate_provenance["validation_scenario"] = "alternate_seed"
        alternate["provenance"] = alternate_provenance
        return (base, alternate)

    def _algorithm_validation_parent(self, algorithm: AlgorithmCandidate, fallback: CheckpointArtifact) -> CheckpointArtifact:
        """Resolve the immutable model on which this algorithm was generated."""

        reference_model_id = str(algorithm.metadata.get("evaluation_parent_model_id", "")).strip()
        if not reference_model_id and algorithm.parent_id:
            reference_model_id = str(algorithm.parent_id)
        if reference_model_id:
            try:
                return self._artifact_for_candidate(self.model_store.get(reference_model_id))
            except (OSError, RuntimeError, TypeError, ValueError):
                pass
        expected_sha256 = str(algorithm.metadata.get("evaluation_parent_checkpoint_sha256", "")).strip()
        if expected_sha256 and expected_sha256 == fallback.sha256:
            return fallback
        return fallback

    @staticmethod
    def _load_validation_profiles(path: Path) -> Optional[tuple[Mapping[str, Any], ...]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            profiles = payload.get("profiles") if isinstance(payload, Mapping) else None
            if not isinstance(profiles, list) or len(profiles) < 2:
                return None
            return tuple(dict(item) for item in profiles if isinstance(item, Mapping))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

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
                scenario_results=tuple(dict(item) for item in raw.get("scenario_results", []) if isinstance(item, Mapping)),
                reasons=tuple(str(item) for item in raw.get("reasons", [])),
                created_at=str(raw.get("created_at", "")),
                calibration_digest=str(raw["calibration_digest"]) if raw.get("calibration_digest") else None,
                search_budget=dict(raw.get("search_budget") or {}),
                interaction_evidence=dict(raw.get("interaction_evidence") or {}),
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

    def _algorithm_parent_selection(self) -> dict[str, Any]:
        """Separate latest, effective, historical-best, and pending branches."""

        candidates = list(self.algorithm_store.read())
        latest = max(candidates, key=lambda item: int(item.algorithm_id[1:])) if candidates else None
        validated = [item for item in candidates if item.validation_status == "validated"]
        accepted = [item for item in candidates if item.status == "accepted"]

        # A failed model-search round still has useful algorithm evidence in
        # its candidate trace. Rank algorithm lineage separately from model
        # promotion: feasibility leads, then quality, measured score, and
        # efficiency. Under-floor near misses can select an algorithm parent,
        # but never a model candidate.
        scores: dict[str, tuple[float, float, float, float, float]] = {}

        def record_score(algorithm_id: str, row: Mapping[str, Any], *, accepted_result: bool = False) -> None:
            quality = row.get("quality_score")
            score = row.get("search_score", quality)
            efficiency = row.get("efficiency_reduction_ratio", row.get("parameter_reduction_ratio", 0.0))
            values = (quality, score, efficiency)
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
                return
            feasible = row.get("quality_feasible") is True or row.get("feasible") is True or accepted_result
            position = row.get("rank_position", row.get("evaluation_position", 0))
            if not isinstance(position, (int, float)) or isinstance(position, bool) or not math.isfinite(float(position)):
                position = 0
            key = (
                1.0 if feasible else 0.0,
                float(quality),
                float(score),
                float(efficiency),
                -float(position),
            )
            if algorithm_id and key > scores.get(algorithm_id, (-1.0, float("-inf"), float("-inf"), float("-inf"), float("-inf"))):
                scores[algorithm_id] = key

        for feedback in self._algorithm_search_feedback_history(limit=None):
            algorithm_id = str(feedback.get("algorithm_id", ""))
            for row in feedback.get("measured_candidates", ()):
                if isinstance(row, Mapping) and row.get("valid") is True:
                    record_score(algorithm_id, row)

        for record in self.experience_store.read():
            algorithm_id = str(record.training.get("algorithm_id", ""))
            evaluation = record.evaluation or {}
            record_score(algorithm_id, evaluation, accepted_result=record.status == "accepted")

        def algorithm_rank(item: AlgorithmCandidate) -> tuple[Any, ...]:
            return (
                *scores.get(item.algorithm_id, (-1.0, float("-inf"), float("-inf"), float("-inf"), float("-inf"))),
                item.validation_status == "validated",
                item.status == "accepted",
                int(item.algorithm_id[1:]),
            )

        historical_best = max(
            candidates,
            key=algorithm_rank,
            default=None,
        )
        effective = max(
            validated,
            key=algorithm_rank,
            default=None,
        )
        if effective is None:
            effective = historical_best or (accepted[-1] if accepted else latest)
        pending = [item for item in candidates if item.validation_status in {"pending", "unvalidated"} or item.status in {"proposed", "failed"}]
        rejected = [item for item in candidates if item.validation_status in {"rejected", "failed"} or item.status == "rejected"]
        return {
            "latest": latest,
            "effective": effective,
            "historical_best": historical_best,
            "pending": tuple(pending),
            "rejected": tuple(rejected),
            "selection_reason": "validated_then_measured_block_search_evidence_then_latest_fallback",
        }

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
        model_profile = profile.get("model", {}) if isinstance(profile, Mapping) else {}
        before = int(model_profile.get("block_count", 1))
        after = int(child.structure.block_count)
        block_ratio = float(after) / float(max(before, 1))
        parent_parameters = int(model_profile.get("parameter_count", child.structure.parameter_count))
        parameter_ratio = float(child.structure.parameter_count) / float(max(parent_parameters, 1))
        # Block pruning has a direct layer-count proxy; other structural
        # operators (FFN/head packing) are represented by the actual child
        # parameter count.  This keeps the Fake evaluator from silently
        # treating real head/FFN mutations as zero compute change.
        ratio = block_ratio if after < before else parameter_ratio
        runtime_ratio = 1.0
        runtime_operations = child.structure.metadata.get("runtime_operations", [])
        if isinstance(runtime_operations, (list, tuple)):
            for item in runtime_operations:
                if not isinstance(item, Mapping):
                    continue
                operator = str(item.get("operator", ""))
                args = item.get("args", {})
                if not isinstance(args, Mapping):
                    continue
                if operator in {"prune_tokens", "sparse_runtime"}:
                    runtime_ratio *= float(args.get("keep_ratio", 1.0))
                elif operator == "kv_token_select":
                    runtime_ratio *= float(args.get("keep_ratio", 1.0))
                elif operator == "token_merge":
                    runtime_ratio *= float(args.get("keep_ratio", 1.0)) ** 2
                elif operator == "attention_window":
                    runtime_ratio *= min(1.0, float(args.get("window_size", 64)) / 64.0)
                elif operator == "head_gating":
                    runtime_ratio *= float(args.get("keep_ratio", 1.0))
                elif operator == "sampling_step_skip":
                    runtime_ratio *= min(1.0, float(args.get("target_steps", 16)) / 16.0)
        static_ratio = max(0.01, min(1.0, ratio))
        ratio = max(0.01, min(1.0, static_ratio * runtime_ratio))
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
            # Runtime-only policies reduce executed work but do not remove
            # checkpoint bytes.  Keep the model-size proxy tied to the
            # structural ratio so a head/token gate cannot claim parameter
            # or storage reduction.
            "model_size_gb": max(0.01, 1.0 * static_ratio),
            "hardware_metrics": {"latency_s": 1.0 * ratio, "peak_memory_gb": 1.0 * ratio, "model_size_gb": max(0.01, 1.0 * static_ratio)},
            "parameter_reduction_ratio": max(0.0, 1.0 - parameter_ratio),
            "estimated_compute_ratio": ratio,
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
