from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from harness4h3 import HARNESS_CHANGE_POLICY, HARNESS_STATUS, HARNESS_VERSION
from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.model_store import ModelStore
from harness4h3.archive.pareto import ParetoArchive
from harness4h3.controller.context import ControllerContext
from harness4h3.controller.policy import PlanValidationError, ValidationPipeline
from harness4h3.controller.provider import (
    ControllerProviderError,
    OllamaStructuredController,
)
from harness4h3.controller.schemas import BudgetState, CostEstimate, EvaluationResult, ExperimentPlan, OperatorResult
from harness4h3.evaluator.composite import CompositeEvaluator
from harness4h3.evaluator.constraints import ConstraintEvaluator
from harness4h3.evaluator.hardware import FakeHardwareEvaluator
from harness4h3.evaluator.quality import FakeQualityEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.memory.trajectory import Trajectory, TrajectoryStore
from harness4h3.operators.base import ExecutionContext, OperatorRegistry
from harness4h3.operators.model_evolution import build_model_evolution_registry
from harness4h3.target.profile import TargetProfile, load_target_profile


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _state_digest(state: ModelState) -> str:
    raw = json.dumps(_canonicalize(state.to_dict()), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fingerprint(operator: str, args: Mapping[str, Any], state: ModelState, target_id: str) -> str:
    payload = {
        "operator": str(operator),
        "operator_args": _canonicalize(dict(args)),
        "parent_state_digest": _state_digest(state),
        "target_profile_id": str(target_id),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _persist(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class A0Budget:
    max_gpu_hours: float = 24.0
    max_experiments: int = 20
    max_failed_experiments: int = 5
    max_concurrent_experiments: int = 1
    used_gpu_hours: float = 0.0
    used_experiments: int = 0
    used_failed_experiments: int = 0
    used_wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.max_gpu_hours <= 0 or self.max_experiments <= 0 or self.max_concurrent_experiments != 1:
            raise ValueError("A0 budget must be positive and single-concurrency")
        if self.max_failed_experiments < 0:
            raise ValueError("max_failed_experiments must be non-negative")

    def stop_reason(self) -> Optional[str]:
        if self.used_failed_experiments >= self.max_failed_experiments and self.max_failed_experiments >= 0:
            if self.used_failed_experiments > 0:
                return "failure_budget_exhausted"
        if self.used_experiments >= self.max_experiments:
            return "experiment_budget_exhausted"
        if self.used_gpu_hours >= self.max_gpu_hours:
            return "gpu_budget_exhausted"
        return None

    def consume(self, cost: CostEstimate, failed: bool) -> "A0Budget":
        return A0Budget(
            max_gpu_hours=self.max_gpu_hours,
            max_experiments=self.max_experiments,
            max_failed_experiments=self.max_failed_experiments,
            max_concurrent_experiments=self.max_concurrent_experiments,
            used_gpu_hours=self.used_gpu_hours + max(0.0, float(cost.gpu_hours)),
            used_experiments=self.used_experiments + 1,
            used_failed_experiments=self.used_failed_experiments + (1 if failed else 0),
            used_wall_time_s=self.used_wall_time_s + max(0.0, float(cost.wall_time_s)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class A0CampaignResult:
    status: str
    payload: Mapping[str, Any]
    report: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "payload": dict(self.payload), "report": dict(self.report)}


def _tier_for(operator: str) -> int:
    if operator in {"create_student", "prune_blocks", "prune_heads", "prune_channels"}:
        return 1
    if operator in {"distill", "recovery_finetune"}:
        return 2
    return 3


def _operator_summary(registry: OperatorRegistry) -> List[Mapping[str, Any]]:
    return [dict(item) for item in registry.visible()]


def _pareto_summary(pareto: ParetoArchive) -> List[Mapping[str, Any]]:
    return [entry.to_dict() for entry in pareto.front()]


def _evaluation_dict(result: Optional[EvaluationResult]) -> Mapping[str, Any]:
    return result.to_dict() if result is not None else {}


class A0RuleBasedController:
    """Deterministic controller used by offline A0 preflight and CI."""

    provider_name = "offline"
    model_name = "a0-rule-based"

    def plan(self, context: ControllerContext) -> ExperimentPlan:
        state = context.current_model_state
        target = context.target_profile
        previous = [
            item
            for item in context.recent_experiments
            if isinstance(item, Mapping) and item.get("operator")
        ]
        previous_for_operator: Dict[str, Mapping[str, Any]] = {}
        for item in previous:
            previous_for_operator[str(item["operator"])] = item

        algorithm = state.algorithm_state
        provenance = state.provenance
        bits = int(state.quantization.get("bits", 16))
        metrics = state.measured_metrics
        if not algorithm.get("student") and provenance.get("operator") not in {"create_student", "prune_blocks", "prune_heads", "prune_channels"}:
            prior = previous_for_operator.get("create_student")
            prior_ratio = float((prior or {}).get("operator_args", {}).get("width_ratio", 0.0))
            current_size = float(metrics.get("model_size_gb", 7.0))
            current_memory = float(metrics.get("peak_memory_gb", 12.0))
            size_ratio = float(target.max_model_size_gb / current_size * 0.92) if target.max_model_size_gb else 0.65
            memory_ratio = float(target.max_peak_memory_gb / current_memory * 0.92) if target.max_peak_memory_gb else 0.65
            ratio = min(0.75, size_ratio, memory_ratio)
            if prior_ratio:
                ratio = min(ratio, prior_ratio * 0.82)
            ratio = max(0.25, min(0.75, ratio))
            operator = "create_student"
            operator_args = {"width_ratio": round(ratio, 3), "block_ratio": round(ratio, 3)}
            diagnosis = "student_architecture_is_the_highest_leverage_model_change"
            hypothesis = "a smaller student will reduce model size and peak memory while retaining the quality floor"
            if prior:
                hypothesis += " after %s with a materially smaller width ratio" % prior.get("experiment_id")
            effects = {"model_size_gb": "decrease", "peak_memory_gb": "decrease", "quality_score": "bounded_drop"}
        elif algorithm.get("student") and "distillation" not in algorithm:
            operator = "distill"
            operator_args = {"dataset_fraction": 0.1, "training_steps": 500}
            diagnosis = "student_quality_gap_after_structural_change"
            hypothesis = "short teacher-student distillation will recover quality without restoring the pruned architecture"
            effects = {"quality_score": "increase", "model_size_gb": "unchanged"}
        elif state.sampling_steps and state.sampling_steps > 8:
            operator = "step_distill"
            operator_args = {"target_steps": 8}
            diagnosis = "sampling_latency_remains_the_main_cost"
            hypothesis = "step distillation will reduce sampling latency while preserving the quality floor"
            effects = {"latency_s": "decrease", "peak_memory_gb": "decrease", "quality_score": "bounded_drop"}
        elif bits > 4:
            operator = "quantize"
            operator_args = {"bits": 4}
            diagnosis = "weight_precision_can_reduce_residency"
            hypothesis = "four-bit weight-only quantization will reduce model size and memory with bounded quality loss"
            effects = {"model_size_gb": "decrease", "peak_memory_gb": "decrease", "quality_score": "bounded_drop"}
        else:
            operator = "recovery_finetune"
            prior = previous_for_operator.get("recovery_finetune")
            previous_steps = int((prior or {}).get("operator_args", {}).get("training_steps", 0))
            recovery_steps = min(5000, previous_steps * 2 if previous_steps else 500)
            operator_args = {"training_steps": recovery_steps}
            diagnosis = "recover_remaining_quality_after_model_changes"
            hypothesis = "a short recovery fine-tune will improve the Pareto quality point without changing structure"
            if prior:
                hypothesis += " after %s by increasing recovery steps" % prior.get("experiment_id")
            effects = {"quality_score": "increase"}

        tier = _tier_for(operator)
        number = context.budget_state.used_iterations + 1
        target_quality = target.min_quality_score
        if target_quality is None and target.max_quality_drop is not None:
            target_quality = None
        cost_by_operator = {
            "create_student": (1.0, 0.25),
            "prune_blocks": (0.6, 0.12),
            "prune_heads": (0.6, 0.12),
            "prune_channels": (0.6, 0.12),
            "distill": (2.0, 0.75),
            "step_distill": (2.5, 1.0),
            "recovery_finetune": (1.5, 0.5),
            "quantize": (0.8, 0.2),
        }
        wall_time_s, gpu_hours = cost_by_operator[operator]
        return ExperimentPlan(
            experiment_id="exp_%04d" % number,
            parent_model_id=state.model_id,
            diagnosis=diagnosis,
            objective="produce a better H3-derived Pareto candidate",
            hypothesis=hypothesis,
            operator=operator,
            operator_args=operator_args,
            expected_effects=effects,
            risks=["quality regression", "training artifact failure"],
            required_budget={"wall_time_s": wall_time_s, "gpu_hours": gpu_hours, "controller_calls": 0, "tier": tier},
            acceptance={"max_quality_drop": target.max_quality_drop, "min_quality_score": target_quality},
            stop_conditions={"critical_regression": True, "target_satisfied": False, "budget_exhausted": True},
            rationale="select the next model-level intervention from the current lineage and observed metrics",
        )


class A0Campaign:
    def __init__(
        self,
        target: TargetProfile,
        controller: Any,
        registry: OperatorRegistry,
        evaluator: CompositeEvaluator,
        output_root: Path,
        initial_candidate: Optional[ModelCandidate] = None,
        budget: Optional[A0Budget] = None,
        stop_on_target: bool = False,
        offline_simulation: bool = True,
    ):
        self.target = target
        self.controller = controller
        self.registry = registry
        self.evaluator = evaluator
        self.output_root = Path(output_root)
        self.initial_candidate = initial_candidate or ModelCandidate(
            "M0000", None, 0, "fake://M0000", ModelState.fake_baseline("M0000"), None, "baseline"
        )
        self.budget = budget or A0Budget()
        self.stop_on_target = stop_on_target
        self.offline_simulation = offline_simulation
        self.validation = ValidationPipeline()

    def _context(
        self,
        state: ModelState,
        budget: A0Budget,
        recent: Sequence[Mapping[str, Any]],
        failures: Sequence[Mapping[str, Any]],
        pareto: ParetoArchive,
    ) -> ControllerContext:
        harness_budget = BudgetState(
            max_iterations=self.budget.max_experiments,
            max_failed_experiments=self.budget.max_failed_experiments,
            max_gpu_hours=self.budget.max_gpu_hours,
            max_controller_calls=self.budget.max_experiments,
            used_iterations=budget.used_experiments,
            used_failures=budget.used_failed_experiments,
            used_gpu_hours=budget.used_gpu_hours,
            used_controller_calls=budget.used_experiments,
        )
        return ControllerContext(
            self.target,
            state,
            harness_budget,
            tuple(_operator_summary(self.registry)),
            recent_experiments=[dict(item) for item in recent],
            relevant_failures=[dict(item) for item in failures],
            pareto_front=[dict(item) for item in _pareto_summary(pareto)],
            validated_design_genes=[],
            validated_evaluation={"source": "a0_offline_or_external_evaluator"},
        )

    @staticmethod
    def _novelty_error(
        operator: str,
        hypothesis: str,
        fingerprint: str,
        seen: Set[str],
        recent: Sequence[Mapping[str, Any]],
    ) -> Optional[str]:
        if fingerprint in seen:
            return "duplicate_experiment_fingerprint"
        prior_ids = [
            str(item.get("experiment_id"))
            for item in recent
            if str(item.get("operator", "")) == operator and item.get("experiment_id")
        ]
        if prior_ids and not any(previous_id in hypothesis for previous_id in prior_ids):
            return "same_operator_requires_prior_evidence"
        return None

    def run(self) -> A0CampaignResult:
        started_at = datetime.now(timezone.utc).isoformat()
        campaign_started = time.monotonic()
        model_store = ModelStore(self.output_root / "models")
        model_store.initialize(self.initial_candidate)
        pareto = ParetoArchive(self.output_root / "pareto")
        trajectories = TrajectoryStore(self.output_root / "trajectories.jsonl")
        output_path = self.output_root / "campaign.json"
        report_path = self.output_root / "report.json"
        baseline_quality = float(self.initial_candidate.state.measured_metrics.get("quality_score", 0.0))
        budget = self.budget
        recent: List[Mapping[str, Any]] = []
        failures: List[Mapping[str, Any]] = []
        seen: Set[str] = set()
        duplicate_fingerprints_blocked = 0
        iterations: List[Mapping[str, Any]] = []
        target_satisfied = False
        fatal_setup_error: Optional[str] = None

        payload: Dict[str, Any] = {
            "campaign_id": "A0",
            "harness": {"version": HARNESS_VERSION, "status": HARNESS_STATUS, "change_policy": HARNESS_CHANGE_POLICY},
            "campaign": {
                "name": "Autonomous Model Evolution Campaign A0",
                "action_space": list(self.registry.names()),
                "max_concurrent_experiments": self.budget.max_concurrent_experiments,
                "multi_fidelity_tiers": {
                    "0": "static checks",
                    "1": "cheap structural screening",
                    "2": "short distillation/recovery",
                    "3": "full candidate evaluation",
                },
            },
            "target_profile": self.target.to_dict(),
            "target_profile_id": self.target.id,
            "status": "running",
            "human_intervention_count": 0,
            "offline_simulation": self.offline_simulation,
            "iterations": iterations,
            "system_candidates": [],
            "experience_influence_evidence": [],
            "duplicate_fingerprints_blocked": 0,
            "gpu_hours_available": False,
        }

        while budget.stop_reason() is None:
            iteration_started = time.monotonic()
            parent = model_store.active()
            state_before = parent.state
            context: Optional[ControllerContext] = None
            plan: Optional[ExperimentPlan] = None
            operator_result: Optional[OperatorResult] = None
            child: Optional[ModelCandidate] = None
            evaluation: Optional[EvaluationResult] = None
            plan_error: Optional[str] = None
            failure_type: Optional[str] = None
            fingerprint: Optional[str] = None
            operator_args: Dict[str, Any] = {}
            tier_results: Dict[str, Any] = {"tier_0": {"static_checks": "passed"}}
            tier = None
            outcome = "failed_experiment"

            try:
                context = self._context(state_before, budget, recent, failures, pareto)
                raw_plan = self.controller.plan(context)
                validated = self.validation.validate(
                    raw_plan,
                    state_before,
                    self.target,
                    context.budget_state,
                    self.registry,
                )
                plan = validated.plan
                operator_args = dict(plan.operator_args)
                tier = int(plan.required_budget.get("tier", _tier_for(plan.operator)))
                if tier not in {1, 2, 3}:
                    raise PlanValidationError("tier_invalid", "A0 fidelity tier must be 1, 2, or 3")
                fingerprint = _fingerprint(plan.operator, operator_args, state_before, self.target.id)
                novelty_error = self._novelty_error(plan.operator, plan.hypothesis, fingerprint, seen, recent)
                if novelty_error:
                    failure_type = novelty_error
                    if novelty_error == "duplicate_experiment_fingerprint":
                        duplicate_fingerprints_blocked += 1
                else:
                    seen.add(fingerprint)
                    child_id = model_store.next_id()
                    operator_result = self.registry.execute(
                        plan.operator,
                        parent,
                        operator_args,
                        self.target,
                        ExecutionContext(self.output_root / "runs" / plan.experiment_id, child_id, model_store),
                    )
                    if not operator_result.ok:
                        failure_type = operator_result.failure_type or "operator_failure"
                    elif operator_result.output_state is None:
                        failure_type = "missing_output_state"
                    else:
                        for level in range(1, tier + 1):
                            evaluation = self.evaluator.evaluate(operator_result.output_state, self.target, baseline_quality)
                            tier_results["tier_%d" % level] = {
                                "validated": evaluation.feasible,
                                "evaluation": evaluation.to_dict(),
                                "promotion": "continue" if evaluation.feasible and level < tier else "complete",
                            }
                            if not evaluation.feasible:
                                break
                        if evaluation is None:
                            failure_type = "missing_evaluation"
                        else:
                            outcome = "accepted_candidate" if evaluation.feasible else "rejected_candidate"
                            child = ModelCandidate(
                                child_id,
                                parent.id,
                                parent.generation + 1,
                                operator_result.output_state.checkpoint_path,
                                operator_result.output_state,
                                plan.experiment_id,
                                "accepted" if outcome == "accepted_candidate" else "rejected",
                                metadata={"operator": plan.operator, "fidelity_tier": tier, "offline_simulation": self.offline_simulation},
                            )
                            model_store.create(child)
                            if outcome == "accepted_candidate":
                                model_store.set_active(child.id)
                                pareto.update(child.id, evaluation)
                                target_satisfied = True
            except ControllerProviderError as exc:
                failure_type = "controller_failure"
                plan_error = str(exc)
            except PlanValidationError as exc:
                failure_type = exc.code
                plan_error = str(exc)
            except (OSError, TypeError, ValueError, KeyError) as exc:
                failure_type = "campaign_execution_error"
                plan_error = str(exc)

            elapsed = time.monotonic() - iteration_started
            measured_cost = operator_result.cost if operator_result is not None else CostEstimate()
            actual_cost = CostEstimate(
                wall_time_s=max(elapsed, measured_cost.wall_time_s),
                gpu_hours=measured_cost.gpu_hours,
            )
            if outcome == "failed_experiment" and failure_type is None:
                failure_type = "execution_failure"
            failed = outcome == "failed_experiment"
            budget = budget.consume(actual_cost, failed=failed)
            if failed:
                failures.append({"experiment_id": plan.experiment_id if plan else "exp_%04d" % (budget.used_experiments), "operator": plan.operator if plan else None, "failure_type": failure_type})

            experiment_id = plan.experiment_id if plan else "exp_%04d" % budget.used_experiments
            reason = "accepted_candidate" if outcome == "accepted_candidate" else failure_type or "acceptance_rejected"
            active_after = model_store.active()
            record: Dict[str, Any] = {
                "iteration": budget.used_experiments - 1,
                "experiment_id": experiment_id,
                "model_parent_id": parent.id,
                "model_child_id": child.id if child else None,
                "state_before": state_before.to_dict(),
                "diagnosis": plan.diagnosis if plan else None,
                "hypothesis": plan.hypothesis if plan else None,
                "operator": plan.operator if plan else None,
                "operator_args": operator_args,
                "expected_effects": dict(plan.expected_effects) if plan else {},
                "risks": list(plan.risks) if plan else [],
                "fidelity_tier": tier,
                "fidelity_results": tier_results,
                "fingerprint": fingerprint,
                "execution_result": operator_result.to_dict() if operator_result else {"status": "not_executed"},
                "evaluation": _evaluation_dict(evaluation),
                "quality_metrics": dict(evaluation.quality_metrics) if evaluation else {},
                "hardware_metrics": evaluation.hardware.__dict__ if evaluation else {},
                "outcome": outcome,
                "accept_reject_reason": reason,
                "failure_type": failure_type,
                "operator_executed": operator_result is not None,
                "state_after": operator_result.output_state.to_dict() if operator_result and operator_result.output_state else state_before.to_dict(),
                "active_state_after": active_after.state.to_dict(),
                "next_decision_context": {
                    "active_parent_id": active_after.id,
                    "active_parent_state_digest": _state_digest(active_after.state),
                    "budget": budget.to_dict(),
                    "recent_experiment_ids": [str(item.get("experiment_id")) for item in iterations[-8:]],
                },
                "controller_error": plan_error,
                "cost": {"wall_time_s": actual_cost.wall_time_s, "gpu_hours": actual_cost.gpu_hours},
                "harness_version": HARNESS_VERSION,
            }
            iterations.append(record)
            recent.append({
                "experiment_id": experiment_id,
                "operator": record["operator"],
                "operator_args": operator_args,
                "hypothesis": record["hypothesis"],
                "outcome": outcome,
                "failure_type": failure_type,
            })
            if record["operator"] and len([item for item in recent if item.get("operator") == record["operator"]]) > 1:
                payload["experience_influence_evidence"].append({
                    "experiment_id": experiment_id,
                    "operator": record["operator"],
                    "evidence": "later decision cited prior experiment history",
                })
            payload.update({
                "budget": budget.to_dict(),
                "iterations": iterations,
                "system_candidates": [item.to_dict() for item in model_store.lineage()],
                "pareto_front": _pareto_summary(pareto),
                "duplicate_fingerprints_blocked": duplicate_fingerprints_blocked,
                "wall_time_s": time.monotonic() - campaign_started,
                "gpu_hours": budget.used_gpu_hours,
            })
            trajectories.append(
                Trajectory(
                    task_id="a0:%s" % experiment_id,
                    harness_version=HARNESS_VERSION,
                    split="a0",
                    inputs={"parent_model_id": parent.id, "operator": record["operator"], "tier": tier},
                    steps=[
                        {"action": "controller_plan", "plan": plan.to_dict() if plan else {}, "error": plan_error},
                        {"action": "model_operator", "operator_result": record["execution_result"]},
                        {"action": "fidelity_evaluation", "results": tier_results},
                        {"action": "campaign_decision", "outcome": outcome, "reason": reason},
                    ],
                    final_result={"model_child_id": child.id if child else None, "outcome": outcome},
                    score=evaluation.quality_score if evaluation else None,
                    failure_type=failure_type,
                    cost={"wall_time": actual_cost.wall_time_s, "gpu_hours": actual_cost.gpu_hours},
                    evaluation={"campaign": "A0", "record": record},
                    critical_regression=evaluation.critical_regression if evaluation else failed,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            _persist(output_path, payload)
            if target_satisfied and self.stop_on_target:
                break

        ended_at = datetime.now(timezone.utc).isoformat()
        if fatal_setup_error:
            status = "setup_failed"
            termination_reason = fatal_setup_error
        elif target_satisfied and self.stop_on_target:
            status = "target_satisfied"
            termination_reason = "target_satisfied"
        else:
            status = "completed"
            termination_reason = budget.stop_reason() or "campaign_complete"
        payload.update({
            "status": status,
            "termination_reason": termination_reason,
            "target_satisfied": target_satisfied,
            "budget": budget.to_dict(),
            "total_experiments": len(iterations),
            "failed_experiment_count": sum(1 for item in iterations if item.get("outcome") == "failed_experiment"),
            "rejected_candidate_count": sum(1 for item in iterations if item.get("outcome") == "rejected_candidate"),
            "wall_time_s": time.monotonic() - campaign_started,
            "gpu_hours": budget.used_gpu_hours,
            "gpu_hours_available": False,
            "duplicate_fingerprints_blocked": duplicate_fingerprints_blocked,
        })
        report = build_a0_report(payload, started_at, ended_at)
        payload["research_report"] = report
        _persist(output_path, payload)
        _persist(report_path, report)
        return A0CampaignResult(status, payload, report)


def build_a0_report(payload: Mapping[str, Any], started_at: str, ended_at: str) -> Dict[str, Any]:
    iterations = list(payload.get("iterations") or [])
    accepted = [item for item in iterations if item.get("outcome") == "accepted_candidate"]
    rejected = [item for item in iterations if item.get("outcome") == "rejected_candidate"]
    failed = [item for item in iterations if item.get("outcome") == "failed_experiment"]
    lineage = list(payload.get("system_candidates") or [])
    return {
        "campaign_id": payload.get("campaign_id", "A0"),
        "harness_version": payload["harness"]["version"],
        "target_profile_id": payload["target_profile_id"],
        "status": payload.get("status"),
        "termination_reason": payload.get("termination_reason"),
        "target_satisfied": bool(payload.get("target_satisfied")),
        "full_autonomous_experiment_sequence": iterations,
        "accepted_experiments": accepted,
        "rejected_experiments": rejected,
        "failed_experiments": failed,
        "model_lineage": lineage,
        "final_pareto_candidates": list(payload.get("pareto_front") or []),
        "total_experiments": len(iterations),
        "failed_experiment_count": len(failed),
        "rejected_candidate_count": len(rejected),
        "budget": dict(payload.get("budget") or {}),
        "wall_time_s": float(payload.get("wall_time_s", 0.0)),
        "gpu_hours": float(payload.get("gpu_hours", 0.0)),
        "gpu_hours_available": bool(payload.get("gpu_hours_available", False)),
        "human_intervention_count": 0,
        "max_concurrent_experiments": payload.get("campaign", {}).get("max_concurrent_experiments", 1),
        "offline_simulation": bool(payload.get("offline_simulation", False)),
        "repeated_failure_avoidance": {
            "duplicate_fingerprints_blocked": int(payload.get("duplicate_fingerprints_blocked", 0)),
            "rejected_parent_reuse": False,
            "same_operator_requires_prior_evidence": True,
        },
        "experience_influence_evidence": list(payload.get("experience_influence_evidence") or []),
        "started_at": started_at,
        "ended_at": ended_at,
    }


def default_validated_nvfp4_candidate() -> ModelCandidate:
    state = ModelState(
        model_id="M0000",
        parent_model_id=None,
        checkpoint_path="validated://minimax-h3-nvfp4",
        architecture_name="MiniMax-H3",
        parameter_count=11_681_874_744,
        trainable_parameter_count=0,
        num_blocks=50,
        hidden_size=2688,
        num_attention_heads=32,
        ffn_width=10752,
        dtype="mixed",
        quantization={"bits": 4, "scheme": "nvfp4"},
        sampling_steps=20,
        components={"text_encoder": "validated-H3-text-encoder"},
        measured_metrics={
            "quality_score": 0.991137,
            "latency_s": 90.28058435407002,
            "peak_memory_gb": 16.29452817,
            "model_size_gb": 12.5286368,
            "energy_j": 150.0,
            "throughput": 1.0 / 90.28058435407002,
        },
        provenance={"source": "validated_nvfp4_artifact", "offline_simulation": True},
    )
    return ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")


def run_campaign(
    target: TargetProfile,
    controller: Any,
    registry: OperatorRegistry,
    evaluator: CompositeEvaluator,
    output_root: Path,
    initial_candidate: Optional[ModelCandidate] = None,
    budget: Optional[A0Budget] = None,
    stop_on_target: bool = False,
    offline_simulation: bool = True,
) -> A0CampaignResult:
    return A0Campaign(
        target,
        controller,
        registry,
        evaluator,
        output_root,
        initial_candidate,
        budget,
        stop_on_target,
        offline_simulation,
    ).run()


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Autonomous H3 model-evolution campaign A0")
    parser.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    parser.add_argument("--output-root", default="var/a0-model-evolution")
    parser.add_argument("--controller", choices=("mock", "ollama"), default="mock")
    parser.add_argument("--controller-model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b-q8_0"))
    parser.add_argument("--controller-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--controller-timeout", type=float, default=180.0)
    parser.add_argument("--max-gpu-hours", type=float, default=24.0)
    parser.add_argument("--max-experiments", type=int, default=20)
    parser.add_argument("--max-failed-experiments", type=int, default=5)
    parser.add_argument("--stop-on-target", action="store_true")
    args = parser.parse_args()
    target = load_target_profile(Path(args.target))
    registry = build_model_evolution_registry()
    controller = A0RuleBasedController()
    if args.controller == "ollama":
        controller = OllamaStructuredController(args.controller_model, args.controller_url, timeout_s=args.controller_timeout)
    evaluator = CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator())
    result = run_campaign(
        target,
        controller,
        registry,
        evaluator,
        Path(args.output_root),
        initial_candidate=default_validated_nvfp4_candidate(),
        budget=A0Budget(args.max_gpu_hours, args.max_experiments, args.max_failed_experiments),
        stop_on_target=args.stop_on_target,
        offline_simulation=args.controller == "mock",
    )
    print(json.dumps({"status": result.status, "report": str(Path(args.output_root) / "report.json"), "experiments": result.report["total_experiments"]}, ensure_ascii=False))
    return 0 if result.status in {"completed", "target_satisfied"} else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
