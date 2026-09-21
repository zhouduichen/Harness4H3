from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..archive.model_candidate import ModelCandidate
from ..archive.model_store import ModelStore
from ..archive.pareto import ParetoArchive
from ..archive.system_candidate import SystemCandidate
from ..archive.system_store import SystemCandidateStore
from ..evaluator.composite import CompositeEvaluator
from ..memory.experiment_store import ExperimentRecord, ExperimentStore
from ..memory.fingerprint import experiment_fingerprint
from ..memory.design_gene import DesignGeneStore
from ..operators.base import ExecutionContext, OperatorRegistry
from ..target.profile import TargetProfile
from .context import ControllerContext
from .continuation import ContinuationDecision, ContinuationPolicy, ContinuationStatus
from .policy import PlanValidationError, ValidationPipeline
from .provider import ControllerProvider, ControllerProviderError
from .schemas import BudgetState, CostEstimate, EvaluationResult, ExperimentPlan, OperatorResult


class LoopState(str, Enum):
    INIT = "INIT"
    LOAD_TARGET = "LOAD_TARGET"
    LOAD_MODEL = "LOAD_MODEL"
    INSPECT = "INSPECT"
    BUILD_CONTEXT = "BUILD_CONTEXT"
    DIAGNOSE_AND_PLAN = "DIAGNOSE_AND_PLAN"
    VALIDATE = "VALIDATE"
    REPLAN = "REPLAN"
    ESTIMATE_COST = "ESTIMATE_COST"
    EXECUTE = "EXECUTE"
    RECORD_FAILURE = "RECORD_FAILURE"
    REGISTER_CHILD = "REGISTER_CHILD"
    EVALUATE = "EVALUATE"
    UPDATE_PARETO = "UPDATE_PARETO"
    SUMMARIZE_EXPERIMENT = "SUMMARIZE_EXPERIMENT"
    CHECK_STOP = "CHECK_STOP"


@dataclass(frozen=True)
class SessionState:
    session_id: str
    target_profile_id: str
    current_model_id: str
    baseline_quality: float
    budget: BudgetState
    status: str = "running"
    transitions: List[str] = field(default_factory=list)
    failure_counts: Mapping[str, int] = field(default_factory=dict)
    current_system_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SessionState":
        return cls(
            session_id=str(raw["session_id"]),
            target_profile_id=str(raw["target_profile_id"]),
            current_model_id=str(raw["current_model_id"]),
            baseline_quality=float(raw["baseline_quality"]),
            budget=BudgetState.from_dict(raw["budget"]),
            status=str(raw.get("status", "running")),
            transitions=[str(item) for item in raw.get("transitions", [])],
            failure_counts={str(key): int(value) for key, value in (raw.get("failure_counts") or {}).items()},
            current_system_id=str(raw["current_system_id"]) if raw.get("current_system_id") else None,
        )


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    current_model_id: str
    budget: BudgetState
    transitions: List[str]
    current_system_id: Optional[str] = None


class OptimizationLoop:
    def __init__(
        self,
        controller: ControllerProvider,
        operators: OperatorRegistry,
        evaluator: CompositeEvaluator,
        models: ModelStore,
        pareto: ParetoArchive,
        experiments: ExperimentStore,
        run_root: Path,
        checkpoint_path: Path,
        max_repeated_failures: int = 2,
        design_genes: Optional[DesignGeneStore] = None,
        systems: Optional[SystemCandidateStore] = None,
        exploration_enabled: bool = False,
        require_real_evidence: bool = False,
    ):
        self.controller = controller
        self.operators = operators
        self.evaluator = evaluator
        self.models = models
        self.pareto = pareto
        self.experiments = experiments
        self.run_root = Path(run_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.max_repeated_failures = max_repeated_failures
        self.design_genes = design_genes
        self.systems = systems or SystemCandidateStore(self.models.root.parent / "systems", model_store=self.models)
        self.continuation = ContinuationPolicy(exploration_enabled=exploration_enabled)
        self.require_real_evidence = bool(require_real_evidence)
        self.validation = ValidationPipeline()

    def run(
        self,
        session_id: str,
        target: TargetProfile,
        budget: BudgetState,
        initial_candidate: ModelCandidate,
        checkpoint_hook: Optional[Callable[[SessionState], None]] = None,
    ) -> OptimizationResult:
        state = self._load_or_initialize(session_id, target, budget, initial_candidate)
        current = self.models.get(state.current_model_id)
        current_system = self.systems.get(state.current_system_id)
        initial_evaluation = self.evaluator.evaluate(
            current.state,
            target,
            state.baseline_quality,
            system=current_system,
            device_id=self._device_id(current_system),
            task_split=self._task_split(current_system),
            benchmark_recipe=self._benchmark_recipe(current_system),
        )
        if initial_evaluation.feasible:
            return self._finish(state, "target_satisfied")

        while state.status == "running":
            stop = state.budget.stop_reason()
            if stop:
                return self._finish(state, stop)
            current = self.models.get(state.current_model_id)
            current_system = self.systems.get(state.current_system_id)
            state = self._transition(state, LoopState.BUILD_CONTEXT)
            context = self._context(target, current, current_system, state)
            state = self._transition(state, LoopState.DIAGNOSE_AND_PLAN)
            try:
                raw_plan = self.controller.plan(context)
            except ControllerProviderError as exc:
                state = self._transition(state, LoopState.VALIDATE)
                error = PlanValidationError("controller_failure", str(exc))
                state = self._record_validation_failure(state, target, current, {}, error)
                self._checkpoint(state)
                if checkpoint_hook:
                    checkpoint_hook(state)
                if state.failure_counts.get("controller_failure", 0) >= self.max_repeated_failures:
                    return self._finish(state, "no_valid_plan")
                state = self._transition(state, LoopState.REPLAN)
                continue
            state = self._transition(state, LoopState.VALIDATE)
            try:
                validated = self.validation.validate(raw_plan, current.state, target, state.budget, self.operators)
                if validated.plan.parent_system_id and validated.plan.parent_system_id != current_system.id:
                    raise PlanValidationError("policy_invalid", "plan parent does not match current system")
            except PlanValidationError as exc:
                state = self._record_validation_failure(state, target, current, raw_plan, exc)
                self._checkpoint(state)
                if checkpoint_hook:
                    checkpoint_hook(state)
                if exc.code == "budget_exhausted":
                    return self._finish(state, "budget_exhausted")
                if state.failure_counts.get(exc.code, 0) >= self.max_repeated_failures:
                    return self._finish(state, "no_valid_plan")
                state = self._transition(state, LoopState.REPLAN)
                continue

            plan = validated.plan
            fingerprint = self._experiment_fingerprint(current_system, plan, target)
            if not plan.repeat_for_statistics:
                prior_fingerprints = {
                    record.fingerprint
                    for record in self.experiments.read()
                    # A failed operator attempt is retryable: the Controller
                    # must be able to recover from transient OOM/process
                    # failures without requiring a statistical-repeat flag.
                    # Successful executions (including evaluation-rejected
                    # children) remain protected against accidental repeats.
                    if record.fingerprint and record.execution.get("status") != "failed"
                }
                if fingerprint in prior_fingerprints:
                    error = PlanValidationError(
                        "duplicate_experiment",
                        "experiment fingerprint already exists; set repeat_for_statistics=true for a statistical repeat",
                    )
                    state = self._record_validation_failure(
                        state, target, current, plan, error, fingerprint=fingerprint
                    )
                    self._checkpoint(state)
                    if checkpoint_hook:
                        checkpoint_hook(state)
                    if state.failure_counts.get("duplicate_experiment", 0) >= self.max_repeated_failures:
                        return self._finish(state, "no_valid_plan")
                    state = self._transition(state, LoopState.REPLAN)
                    continue
            state = self._transition(state, LoopState.ESTIMATE_COST)
            experiment_dir = self.run_root / plan.experiment_id
            try:
                experiment_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                error = PlanValidationError("experiment_exists", "experiment directory already exists and will not be overwritten")
                state = self._record_validation_failure(state, target, current, plan, error)
                self._checkpoint(state)
                if checkpoint_hook:
                    checkpoint_hook(state)
                if state.failure_counts.get("experiment_exists", 0) >= self.max_repeated_failures:
                    return self._finish(state, "no_valid_plan")
                state = self._transition(state, LoopState.REPLAN)
                continue
            self._write_json(experiment_dir / "controller_request.json", context.to_dict())
            self._write_json(experiment_dir / "controller_response.json", plan.to_dict())
            self._write_json(experiment_dir / "plan.json", plan.to_dict())
            self._write_json(experiment_dir / "validated_plan.json", {**plan.to_dict(), "estimated_cost": asdict(validated.estimated_cost)})
            state = self._transition(state, LoopState.EXECUTE)
            child_id = self.models.next_id()
            child_system_id = self.systems.next_id()
            operator_result = self.operators.execute(
                plan.operator,
                current,
                plan.operator_args,
                target,
                ExecutionContext(
                    experiment_dir,
                    child_id,
                    self.models,
                    child_system_id=child_system_id,
                    parent_system=current_system,
                    system_store=self.systems,
                ),
            )
            self._write_json(experiment_dir / "operator_result.json", operator_result.to_dict())
            if operator_result.ok and self.require_real_evidence and not self._has_real_evidence(operator_result):
                operator_result = OperatorResult(
                    "failed",
                    None,
                    operator_result.cost,
                    artifacts=list(operator_result.artifacts),
                    metrics=dict(operator_result.metrics),
                    failure_type="non_real_evidence",
                    message="real campaign requires evidence_kind=real_h3 and offline_simulation=false",
                )
                self._write_json(experiment_dir / "operator_result.json", operator_result.to_dict())
            if not operator_result.ok:
                state = self._record_operator_failure(
                    state, target, current, plan, operator_result, fingerprint=fingerprint
                )
                self._checkpoint(state)
                if checkpoint_hook:
                    checkpoint_hook(state)
                key = operator_result.failure_type or "operator_failure"
                if state.failure_counts.get(key, 0) >= self.max_repeated_failures:
                    return self._finish(state, "no_valid_plan")
                continue

            evaluation: Optional[EvaluationResult] = None
            child: Optional[ModelCandidate] = None
            child_system: Optional[SystemCandidate] = None
            front_ids: List[str] = [entry.candidate_id for entry in self.pareto.front()]
            continuation: Optional[ContinuationDecision] = None
            if operator_result.output_state is not None:
                state = self._transition(state, LoopState.REGISTER_CHILD)
                child_state = operator_result.output_state
                child = ModelCandidate(
                    id=child_id,
                    parent_id=current.id,
                    generation=current.generation + 1,
                    checkpoint_path=child_state.checkpoint_path,
                    state=child_state,
                    created_by_experiment_id=plan.experiment_id,
                    status="candidate",
                    metadata={"operator": plan.operator},
                )
                self.models.create(child)
                child_system = SystemCandidate.from_model_candidate(
                    child_system_id,
                    child,
                    parent_id=current_system.id,
                    generation=current_system.generation + 1,
                    algorithm_state=child_state.algorithm_state,
                    runtime_state=current_system.runtime_state,
                    created_by_experiment_id=plan.experiment_id,
                    status="candidate",
                    metadata={"operator": plan.operator, "model_id": child.id},
                )
                self.systems.create(child_system)
                state = self._transition(state, LoopState.EVALUATE)
                evaluation = self.evaluator.evaluate(
                    child.state,
                    target,
                    state.baseline_quality,
                    system=child_system,
                    device_id=self._device_id(child_system),
                    task_split=self._task_split(child_system),
                    benchmark_recipe=self._benchmark_recipe(child_system),
                )
                self._write_json(experiment_dir / "evaluation.json", evaluation.to_dict())
                state = self._transition(state, LoopState.UPDATE_PARETO)
                front_ids = [
                    entry.candidate_id
                    for entry in self.pareto.update(child_system.id, evaluation, target.objectives)
                ]
            elif operator_result.output_system is not None:
                state = self._transition(state, LoopState.REGISTER_CHILD)
                child_system = operator_result.output_system
                if child_system.model_ref != current.id:
                    raise ValueError(
                        "runtime operator returned model reference %s for current model %s"
                        % (child_system.model_ref, current.id)
                    )
                self.systems.create(child_system)
                state = self._transition(state, LoopState.EVALUATE)
                evaluation = self.evaluator.evaluate(
                    current.state,
                    target,
                    state.baseline_quality,
                    system=child_system,
                    device_id=self._device_id(child_system),
                    task_split=self._task_split(child_system),
                    benchmark_recipe=self._benchmark_recipe(child_system),
                )
                self._write_json(experiment_dir / "evaluation.json", evaluation.to_dict())
                state = self._transition(state, LoopState.UPDATE_PARETO)
                front_ids = [
                    entry.candidate_id
                    for entry in self.pareto.update(child_system.id, evaluation, target.objectives)
                ]

            if evaluation is not None:
                evaluation = replace(evaluation, evaluation_id=self._next_evaluation_id())
                self._write_json(experiment_dir / "evaluation.json", evaluation.to_dict())

            if evaluation is not None and child_system is not None:
                continuation = self.continuation.decide(
                    evaluation,
                    on_front=child_system.id in front_ids,
                    target=target,
                    plan_acceptance=plan.acceptance,
                    baseline_quality=state.baseline_quality,
                    parent_metrics=self._metrics(current.state),
                )
            else:
                continuation = ContinuationDecision(
                    status=ContinuationStatus.PARETO_KEEP,
                    advance=True,
                    reasons=["operator_returned_no_candidate"],
                )

            state = self._transition(state, LoopState.SUMMARIZE_EXPERIMENT)
            keep = continuation.advance
            decision_failure = None
            if not keep and evaluation is not None:
                decision_failure = (
                    "quality_critical_regression"
                    if evaluation.critical_regression
                    else "continuation_rejected"
                )
            next_model_id = child.id if child is not None and keep else current.id
            next_system_id = child_system.id if child_system is not None and keep else current_system.id
            if keep:
                self.models.set_active(next_model_id)
                self.systems.set_active(next_system_id)
            failed = not keep
            new_budget = state.budget.consume(operator_result.cost, failed=failed, controller_calls=1)
            decision = self._decision(
                keep,
                next_model_id,
                evaluation,
                next_system_id=next_system_id,
                status=continuation.status.value,
                reasons=continuation.reasons,
            )
            record = self._record(
                state,
                target,
                current,
                plan,
                operator_result,
                child.id if child else None,
                evaluation,
                decision_failure,
                decision,
                front_ids,
                child_system_id=child_system.id if child_system else None,
                fingerprint=fingerprint,
                repeat_for_statistics=plan.repeat_for_statistics,
            )
            self.experiments.append(record)
            failure_counts = dict(state.failure_counts)
            if decision_failure:
                failure_counts[decision_failure] = failure_counts.get(decision_failure, 0) + 1
            state = SessionState(
                state.session_id,
                state.target_profile_id,
                next_model_id,
                state.baseline_quality,
                new_budget,
                transitions=state.transitions,
                failure_counts=failure_counts,
                current_system_id=next_system_id,
            )
            state = self._transition(state, LoopState.CHECK_STOP)
            self._checkpoint(state)
            if checkpoint_hook:
                checkpoint_hook(state)
            if evaluation is not None and continuation.status.value == "final_accept" and keep:
                return self._finish(state, "target_satisfied")
            if evaluation is not None and evaluation.critical_regression and plan.stop_conditions.get("critical_regression", False):
                return self._finish(state, "critical_failure")
            if decision_failure and state.failure_counts.get(decision_failure, 0) >= self.max_repeated_failures:
                return self._finish(state, "no_valid_plan")

        return self._finish(state, state.status)

    def _ensure_system_for_model(self, model_id: str) -> SystemCandidate:
        """Return a persisted system for a model, migrating old checkpoints in memory."""

        try:
            active = self.systems.active()
            if active.model_ref == model_id:
                return active
        except Exception:
            pass
        for candidate in reversed(self.systems.lineage()):
            if candidate.model_ref == model_id:
                return candidate
        model = self.models.get(model_id)
        existing = self.systems.lineage()
        if not existing:
            system = SystemCandidate.from_model_candidate("S0000", model, status="baseline")
            self.systems.initialize(system)
            return system
        try:
            parent = self.systems.active()
        except Exception:
            parent = existing[-1]
        system = SystemCandidate.from_model_candidate(
            self.systems.next_id(),
            model,
            parent_id=parent.id,
            generation=parent.generation + 1,
            runtime_state=parent.runtime_state,
            status="migration",
            metadata={"migration": "model-only-checkpoint"},
        )
        self.systems.create(system)
        self.systems.set_active(system.id)
        return system

    def _load_or_initialize(
        self,
        session_id: str,
        target: TargetProfile,
        budget: BudgetState,
        initial: ModelCandidate,
    ) -> SessionState:
        if self.checkpoint_path.exists():
            state = SessionState.from_dict(json.loads(self.checkpoint_path.read_text(encoding="utf-8")))
            if state.session_id != session_id or state.target_profile_id != target.id:
                raise ValueError("checkpoint belongs to a different optimization session or target")
            current_system_id = state.current_system_id
            if current_system_id:
                try:
                    system = self.systems.get(current_system_id)
                except Exception:
                    system = None
                if system is not None and system.model_ref == state.current_model_id:
                    migrated_system_id = system.id
                else:
                    migrated_system_id = self._ensure_system_for_model(state.current_model_id).id
            else:
                migrated_system_id = self._ensure_system_for_model(state.current_model_id).id
            return SessionState(
                state.session_id,
                state.target_profile_id,
                state.current_model_id,
                state.baseline_quality,
                state.budget,
                status="running" if state.status == "running" else state.status,
                transitions=state.transitions,
                failure_counts=state.failure_counts,
                current_system_id=migrated_system_id,
            )
        self.models.initialize(initial)
        current = self.models.active()
        system = self._ensure_system_for_model(current.id)
        baseline_quality = float(current.state.measured_metrics["quality_score"])
        initial_evaluation = self.evaluator.evaluate(
            current.state,
            target,
            baseline_quality,
            system=system,
            device_id=self._device_id(system),
            task_split=self._task_split(system),
            benchmark_recipe=self._benchmark_recipe(system),
        )
        if not self.pareto.entries():
            self.pareto.update(system.id, initial_evaluation, target.objectives)
        state = SessionState(
            session_id,
            target.id,
            current.id,
            baseline_quality,
            budget,
            current_system_id=system.id,
        )
        for transition in (LoopState.INIT, LoopState.LOAD_TARGET, LoopState.LOAD_MODEL, LoopState.INSPECT):
            state = self._transition(state, transition)
        self._checkpoint(state)
        return state

    def _context(
        self,
        target: TargetProfile,
        current: ModelCandidate,
        current_system: SystemCandidate,
        state: SessionState,
    ) -> ControllerContext:
        recent = [item.to_dict() for item in list(self.experiments.read())[-8:]]
        failures = [item for item in recent if item.get("failure_type")]
        front = [entry.to_dict() for entry in self.pareto.front()]
        genes = []
        if self.design_genes is not None:
            genes = [gene.to_dict() for gene in list(self.design_genes.read())[-8:] if gene.status in {"validated", "validated_m5_5", "transferred"}]
        records = list(self.experiments.read())
        recent_ids = {str(item.get("experiment_id")) for item in recent}
        relevant = []
        for record in reversed(records):
            if record.experiment_id in recent_ids:
                continue
            same_model = record.parent_model_id == current.id or record.child_model_id == current.id
            same_system = record.parent_system_id == current_system.id or record.child_system_id == current_system.id
            same_target = record.target_profile_id == target.id
            if same_model or same_system or same_target:
                relevant.append(record.to_dict())
            if len(relevant) >= 8:
                break
        operator_counts: Dict[str, int] = {}
        failure_counts: Dict[str, int] = {}
        decision_counts: Dict[str, int] = {}
        objective_ranges: Dict[str, Dict[str, float]] = {}
        search_score_range: Dict[str, float] = {}
        for record in records:
            operator = str(record.plan.get("operator", "unknown"))
            operator_counts[operator] = operator_counts.get(operator, 0) + 1
            if record.failure_type:
                failure_counts[record.failure_type] = failure_counts.get(record.failure_type, 0) + 1
            status = str(record.decision.get("status", "unknown"))
            decision_counts[status] = decision_counts.get(status, 0) + 1
            evaluation = record.evaluation or {}
            hardware = evaluation.get("hardware") if isinstance(evaluation, Mapping) else {}
            values = {
                "quality_score": evaluation.get("quality_score") if isinstance(evaluation, Mapping) else None,
                **(dict(hardware) if isinstance(hardware, Mapping) else {}),
            }
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    current_range = objective_ranges.setdefault(name, {"min": float(value), "max": float(value)})
                    current_range["min"] = min(current_range["min"], float(value))
                    current_range["max"] = max(current_range["max"], float(value))
            search_score = evaluation.get("search_score") if isinstance(evaluation, Mapping) else None
            if isinstance(search_score, (int, float)) and not isinstance(search_score, bool):
                if not search_score_range:
                    search_score_range = {"min": float(search_score), "max": float(search_score)}
                else:
                    search_score_range["min"] = min(search_score_range["min"], float(search_score))
                    search_score_range["max"] = max(search_score_range["max"], float(search_score))
        campaign_summary = {
            "experiment_count": len(records),
            "operator_counts": operator_counts,
            "failure_counts": failure_counts,
            "decision_counts": decision_counts,
            "objective_ranges": objective_ranges,
            "search_score_range": search_score_range,
            "current_search_score": target.objective_score(self._metrics(current.state)),
        }
        return ControllerContext(
            target,
            current.state,
            state.budget,
            self.operators.visible(),
            recent,
            failures,
            front,
            genes,
            current_system=current_system.to_dict(),
            campaign_summary=campaign_summary,
            retrieved_relevant_experiments=relevant,
        )

    def _next_evaluation_id(self) -> str:
        highest = 0
        for record in self.experiments.read():
            value = (record.evaluation or {}).get("evaluation_id") if isinstance(record.evaluation, Mapping) else None
            if isinstance(value, str) and value.startswith("E"):
                try:
                    highest = max(highest, int(value[1:]))
                except ValueError:
                    continue
        return "E%04d" % (highest + 1)

    @staticmethod
    def _device_id(system: SystemCandidate) -> Optional[str]:
        for source in (system.runtime_state, system.metadata):
            value = source.get("device_id") if isinstance(source, Mapping) else None
            if value:
                return str(value)
        return None

    @staticmethod
    def _has_real_evidence(result: OperatorResult) -> bool:
        metrics = result.metrics if isinstance(result.metrics, Mapping) else {}
        state = result.output_state
        provenance = state.provenance if state is not None and isinstance(state.provenance, Mapping) else {}
        return (
            metrics.get("real_worker") is True
            and metrics.get("offline_simulation") is False
            and (metrics.get("evidence_kind") == "real_h3" or provenance.get("real_h3") is True)
        )

    @classmethod
    def _benchmark_recipe(cls, system: SystemCandidate) -> Mapping[str, Any]:
        for source in (system.runtime_state, system.metadata):
            value = source.get("benchmark_recipe") if isinstance(source, Mapping) else None
            if isinstance(value, Mapping):
                return dict(value)
        return {}

    @classmethod
    def _task_split(cls, system: SystemCandidate) -> Optional[str]:
        recipe = cls._benchmark_recipe(system)
        value = recipe.get("split")
        return str(value) if value else None

    @classmethod
    def _experiment_fingerprint(
        cls,
        system: SystemCandidate,
        plan: ExperimentPlan,
        target: TargetProfile,
    ) -> str:
        return experiment_fingerprint(
            plan.parent_model_id,
            system.id,
            plan.operator,
            plan.operator_args,
            target.id,
            cls._device_id(system),
            cls._benchmark_recipe(system),
        )

    @staticmethod
    def _metrics(state: Any) -> Mapping[str, Any]:
        hardware = state.measured_metrics
        return {
            "quality_score": hardware.get("quality_score"),
            "latency_s": hardware.get("latency_s"),
            "peak_memory_gb": hardware.get("peak_memory_gb"),
            "model_size_gb": hardware.get("model_size_gb"),
            "energy_j": hardware.get("energy_j"),
        }

    def _record_validation_failure(
        self,
        state: SessionState,
        target: TargetProfile,
        current: ModelCandidate,
        raw_plan: Any,
        error: PlanValidationError,
        *,
        fingerprint: str = "",
    ) -> SessionState:
        state = self._transition(state, LoopState.RECORD_FAILURE)
        number = state.budget.used_iterations + 1
        plan_dict = raw_plan.to_dict() if isinstance(raw_plan, ExperimentPlan) else (dict(raw_plan) if isinstance(raw_plan, Mapping) else {"raw": repr(raw_plan)})
        record = ExperimentRecord(
            experiment_id=str(plan_dict.get("experiment_id") or "invalid_%04d" % number),
            session_id=state.session_id,
            target_profile_id=target.id,
            controller=self._controller_metadata(),
            parent_model_id=current.id,
            child_model_id=None,
            state_digest=self._state_digest(current),
            diagnosis={"text": plan_dict.get("diagnosis", "")},
            plan=plan_dict,
            execution={"status": "rejected", "message": str(error)},
            training_logs=[],
            cost=asdict(CostEstimate()),
            evaluation=None,
            failure_type=error.code,
            decision=self._decision(False, current.id),
            pareto_update={"front": [entry.candidate_id for entry in self.pareto.front()]},
            created_at=self._now(),
            parent_system_id=state.current_system_id,
            fingerprint=fingerprint,
            repeat_for_statistics=bool(raw_plan.get("repeat_for_statistics", False)) if isinstance(raw_plan, Mapping) else False,
        )
        self.experiments.append(record)
        counts = dict(state.failure_counts)
        counts[error.code] = counts.get(error.code, 0) + 1
        return SessionState(
            state.session_id,
            state.target_profile_id,
            current.id,
            state.baseline_quality,
            state.budget.consume(CostEstimate(), failed=True, controller_calls=1),
            transitions=state.transitions,
            failure_counts=counts,
            current_system_id=state.current_system_id,
        )

    def _record_operator_failure(
        self,
        state: SessionState,
        target: TargetProfile,
        current: ModelCandidate,
        plan: ExperimentPlan,
        result: OperatorResult,
        *,
        fingerprint: str = "",
    ) -> SessionState:
        state = self._transition(state, LoopState.RECORD_FAILURE)
        key = result.failure_type or "operator_failure"
        record = self._record(
            state,
            target,
            current,
            plan,
            result,
            None,
            None,
            key,
            self._decision(False, current.id),
            [entry.candidate_id for entry in self.pareto.front()],
            fingerprint=fingerprint,
            repeat_for_statistics=plan.repeat_for_statistics,
        )
        self.experiments.append(record)
        counts = dict(state.failure_counts)
        counts[key] = counts.get(key, 0) + 1
        return SessionState(
            state.session_id,
            state.target_profile_id,
            current.id,
            state.baseline_quality,
            state.budget.consume(result.cost, failed=True, controller_calls=1),
            transitions=state.transitions,
            failure_counts=counts,
            current_system_id=state.current_system_id,
        )

    def _record(
        self,
        state: SessionState,
        target: TargetProfile,
        parent: ModelCandidate,
        plan: ExperimentPlan,
        operator_result: OperatorResult,
        child_id: Optional[str],
        evaluation: Optional[EvaluationResult],
        failure_type: Optional[str],
        decision: Mapping[str, Any],
        front_ids: List[str],
        *,
        child_system_id: Optional[str] = None,
        fingerprint: str = "",
        repeat_for_statistics: bool = False,
    ) -> ExperimentRecord:
        system_state_digest = ""
        if state.current_system_id:
            try:
                system_state_digest = self._system_state_digest(self.systems.get(state.current_system_id))
            except Exception:
                system_state_digest = ""
        return ExperimentRecord(
            experiment_id=plan.experiment_id,
            session_id=state.session_id,
            target_profile_id=target.id,
            controller=self._controller_metadata(),
            parent_model_id=parent.id,
            child_model_id=child_id,
            state_digest=self._state_digest(parent),
            diagnosis={"text": plan.diagnosis},
            plan=plan.to_dict(),
            execution=operator_result.to_dict(),
            training_logs=list(operator_result.artifacts),
            cost=asdict(operator_result.cost),
            evaluation=evaluation.to_dict() if evaluation else None,
            failure_type=failure_type,
            decision=dict(decision),
            pareto_update={"front": front_ids},
            created_at=self._now(),
            parent_system_id=state.current_system_id,
            child_system_id=child_system_id,
            system_state_digest=system_state_digest,
            fingerprint=fingerprint,
            repeat_for_statistics=repeat_for_statistics,
        )

    @staticmethod
    def _decision(
        keep: bool,
        next_model_id: str,
        evaluation: Optional[EvaluationResult] = None,
        *,
        next_system_id: Optional[str] = None,
        status: Optional[str] = None,
        reasons: Optional[List[str]] = None,
    ) -> Mapping[str, Any]:
        """Expose semantic decisions while preserving the legacy keep flag.

        ``keep`` and ``continue_from`` remain authoritative for the existing
        protocol. The explicit status separates a retained search point from a
        terminally accepted candidate and from a rejection.
        """
        final_accept = bool(keep and evaluation is not None and evaluation.feasible)
        status = status or ("final_accept" if final_accept else "search_keep" if keep else "reject")
        return {
            "keep": keep,
            "advance": keep,
            "continue_from": next_model_id,
            "continue_model_id": next_model_id,
            "continue_system_id": next_system_id,
            "status": status,
            "decision_status": status,
            "search_keep": status == "search_keep",
            "exploratory_keep": status == "exploratory_keep",
            "pareto_keep": status == "pareto_keep",
            "final_accept": final_accept,
            "reject": status == "reject",
            "reasons": list(reasons or []),
        }

    def _controller_metadata(self) -> Mapping[str, Any]:
        return {
            "provider": getattr(self.controller, "provider_name", "unknown"),
            "model": getattr(self.controller, "model_name", "unknown"),
            "request_id": getattr(self.controller, "last_request_id", None),
        }

    @staticmethod
    def _state_digest(candidate: ModelCandidate) -> str:
        payload = json.dumps(candidate.state.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _system_state_digest(candidate: SystemCandidate) -> str:
        payload = json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _transition(state: SessionState, transition: LoopState) -> SessionState:
        return SessionState(
            state.session_id,
            state.target_profile_id,
            state.current_model_id,
            state.baseline_quality,
            state.budget,
            status=state.status,
            transitions=state.transitions + [transition.value],
            failure_counts=state.failure_counts,
            current_system_id=state.current_system_id,
        )

    def _finish(self, state: SessionState, status: str) -> OptimizationResult:
        finished = SessionState(
            state.session_id,
            state.target_profile_id,
            state.current_model_id,
            state.baseline_quality,
            state.budget,
            status=status,
            transitions=state.transitions,
            failure_counts=state.failure_counts,
            current_system_id=state.current_system_id,
        )
        self._checkpoint(finished)
        return OptimizationResult(
            status,
            finished.current_model_id,
            finished.budget,
            finished.transitions,
            finished.current_system_id,
        )

    def _checkpoint(self, state: SessionState) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="session-", suffix=".json", dir=str(self.checkpoint_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.checkpoint_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
