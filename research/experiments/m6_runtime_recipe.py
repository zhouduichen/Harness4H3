from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from harness4h3 import HARNESS_CHANGE_POLICY, HARNESS_STATUS, HARNESS_VERSION
from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.archive.system_store import SystemCandidateStore
from harness4h3.backends.comfyui import BackendError, MiniMaxH3Adapter
from harness4h3.benchmark.h3 import H3BenchmarkRunner
from harness4h3.benchmark.m6 import M6ValidationRunner
from harness4h3.config import load_config
from harness4h3.controller.context import ControllerContext
from harness4h3.controller.provider import ControllerProviderError, OllamaStructuredController, RuleBasedMockController
from harness4h3.controller.schemas import BudgetState, CostEstimate, OperatorResult
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.harness.loop import load_workflow
from harness4h3.harness.state import load_tasks
from harness4h3.memory.trajectory import Trajectory, TrajectoryStore
from harness4h3.operators.base import ExecutionContext
from harness4h3.operators.runtime_memory import build_runtime_registry
from harness4h3.target.profile import load_target_profile

from research.experiments.m6_campaign import (
    build_research_report,
    classify_outcome,
    experiment_fingerprint,
    state_digest,
    validate_novelty,
)
from research.experiments.m6_runtime_memory import (
    REFERENCE_METRICS,
    _effective_operator_args,
    _load_gene,
    _load_m55_evaluation,
    _persist,
    _state,
)


def _load_runtime_gene(root: Path) -> Mapping[str, Any]:
    path = root / "research/evidence/design-genes/design-gene-m6-vae-tiling.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return raw if isinstance(raw, Mapping) and raw.get("status") else {}


def _load_prior_campaign(path_value: Optional[str]) -> Mapping[str, Any]:
    if not path_value:
        return {}
    try:
        raw = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(raw, Mapping):
        return {}
    iterations = []
    for item in raw.get("iterations") or []:
        if not isinstance(item, Mapping):
            continue
        if not item.get("operator"):
            continue
        not_evaluated = not item.get("split_results") or bool(item.get("split_errors"))
        iterations.append(
            {
                "experiment_id": item.get("experiment_id"),
                "operator": item.get("operator"),
                "operator_args": dict(item.get("operator_args") or {}),
                "outcome": "not_evaluated" if not_evaluated else item.get("outcome"),
                "failure_type": item.get("failure_type"),
                "optimization_conclusion": "none" if not_evaluated else "evaluated",
                "reason": item.get("accept_reject_reason"),
            }
        )
    return {
        "campaign_id": raw.get("campaign_id") or Path(path_value).parent.name,
        "status": raw.get("campaign_classification") or raw.get("status"),
        "termination_reason": raw.get("termination_reason"),
        "optimization_conclusion": (
            "inconclusive_no_optimization_conclusion"
            if not any(item.get("split_results") for item in raw.get("iterations") or [])
            else "evaluated"
        ),
        "source_path": str(Path(path_value)),
        "experiments": iterations,
    }


def _controller_plan(
    args: argparse.Namespace,
    target: Any,
    current: Any,
    operators: Sequence[Mapping[str, Any]],
    genes: Sequence[Mapping[str, Any]],
    validated_evaluation: Mapping[str, Any],
    recent: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    pareto_front: Sequence[Mapping[str, Any]],
    budget: BudgetState,
):
    context = ControllerContext(
        target,
        current,
        budget,
        tuple(operators),
        recent_experiments=[dict(item) for item in recent],
        relevant_failures=[dict(item) for item in failures],
        pareto_front=[dict(item) for item in pareto_front],
        validated_design_genes=[dict(item) for item in genes if item],
        validated_evaluation=validated_evaluation,
    )
    if args.controller == "mock":
        plan = RuleBasedMockController().plan(context)
    else:
        plan = OllamaStructuredController(
            args.controller_model,
            args.controller_url,
            timeout_s=args.controller_timeout,
        ).plan(context)
    return plan, context


def _archive_summary(store: SystemCandidateStore) -> List[Mapping[str, Any]]:
    return [
        {
            "candidate_id": candidate.id,
            "parent_id": candidate.parent_id,
            "generation": candidate.generation,
            "model_ref": candidate.model_ref,
            "status": candidate.status,
            "evaluation": dict(candidate.evaluation),
        }
        for candidate in store.lineage()
    ]


def _acceptance_reason(
    outcome: str,
    failure_type: Optional[str],
    split_results: Mapping[str, Any],
    split_errors: Mapping[str, Any],
) -> str:
    if outcome == "accepted_candidate":
        return "target_profile_satisfied"
    if outcome == "failed_experiment":
        if failure_type:
            return failure_type
        if split_errors:
            return "validation_failure:" + ",".join(
                sorted(str(item.get("failure_type", "m6_validation_failure")) for item in split_errors.values())
            )
        return "execution_failure"
    failed_gates = {
        split: {
            key: value
            for key, value in (result.get("gates") or {}).items()
            if value is False
        }
        for split, result in split_results.items()
        if isinstance(result, Mapping)
    }
    return "acceptance_rejected:" + json.dumps(failed_gates, sort_keys=True)


def _next_decision_context(
    active_system: SystemCandidate,
    current_state: Any,
    budget: BudgetState,
    experiment_history: Sequence[Mapping[str, Any]],
    failed_history: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "active_system_id": active_system.id,
        "active_parent_state_digest": state_digest(current_state),
        "budget": budget.to_dict(),
        "recent_experiment_ids": [str(item.get("experiment_id")) for item in experiment_history[-8:]],
        "failure_count": len(failed_history),
        "rejected_parent_reuse": False,
    }


def _compact_split(result: Mapping[str, Any]) -> Mapping[str, Any]:
    gates = result.get("gates") if isinstance(result, Mapping) else {}
    aggregates = result.get("aggregates") if isinstance(result, Mapping) else {}
    branch = aggregates.get("branch") if isinstance(aggregates, Mapping) else {}
    return {
        "label": result.get("label"),
        "validated": bool(result.get("validated")),
        "gates": dict(gates) if isinstance(gates, Mapping) else {},
        "branch_metrics": {
            name: stats.get("mean")
            for name, stats in (branch.items() if isinstance(branch, Mapping) else [])
            if isinstance(stats, Mapping) and "mean" in stats
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="M6 bounded autonomous runtime-memory campaign")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    parser.add_argument("--base-url", default=os.environ.get("COMFYUI_BASE_URL", "http://100.88.143.10:8188"))
    parser.add_argument("--output", default="var/m6-runtime/m6-recipe-validation.json")
    parser.add_argument("--report-output", default="var/m6-runtime/m6-recipe-report.json")
    parser.add_argument("--benchmark-output", default="var/m6-runtime/recipe-runs")
    parser.add_argument("--trajectory-output", default="var/m6-runtime/recipe-trajectories.jsonl")
    parser.add_argument("--system-store-output", default="var/m6-runtime/system-candidates")
    parser.add_argument("--controller", choices=("ollama", "mock"), default="ollama")
    parser.add_argument("--controller-model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b-q8_0"))
    parser.add_argument("--controller-url", default=os.environ.get("OLLAMA_BASE_URL", "http://100.88.143.10:11434"))
    parser.add_argument("--controller-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--splits", default="dev,heldout")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-failed-experiments", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--campaign-id", default="M6-Campaign")
    parser.add_argument("--prior-campaign", default=None)
    parser.add_argument("--preflight-record", default=None)
    args = parser.parse_args()
    if args.max_iterations <= 0 or args.max_failed_experiments < 0:
        raise ValueError("iteration and failure budgets must be positive/non-negative")

    root = Path(__file__).resolve().parents[2]
    prior_campaign = _load_prior_campaign(args.prior_campaign)
    preflight: Mapping[str, Any] = {}
    if args.preflight_record:
        try:
            loaded_preflight = json.loads(Path(args.preflight_record).read_text(encoding="utf-8"))
            if isinstance(loaded_preflight, Mapping):
                preflight = loaded_preflight
        except (OSError, ValueError, TypeError):
            preflight = {"status": "unreadable", "path": str(args.preflight_record)}
    config = load_config(root / args.config)
    target = load_target_profile(root / args.target)
    tasks = load_tasks(config.runtime.tasks_path)
    split_tasks = {split: [task for task in tasks if task.split == split] for split in ("dev", "heldout")}
    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not requested_splits or any(split not in split_tasks for split in requested_splits):
        raise ValueError("--splits must contain only dev and/or heldout")
    if any(not split_tasks[split] for split in requested_splits):
        raise ValueError("selected M6 split has no tasks")

    base_model = _state("M0001")
    base_candidate = ModelCandidate("M0001", "M0000", 1, base_model.checkpoint_path, base_model, "m5.5", "candidate")
    system_store = SystemCandidateStore(root / args.system_store_output)
    system_store.initialize(
        SystemCandidate.from_model_candidate(
            "C0000",
            base_candidate,
            runtime_state={
                **dict(base_model.runtime_state),
                "component_lifecycle": {
                    "denoiser_loaded": True,
                    "text_encoder_loaded": True,
                    "vae_loaded": True,
                    "lora_loaded": True,
                    "cache_state": "resident",
                    "offload_policy": "none",
                    "unload_points": [],
                },
            },
            status="baseline",
            metadata={"model_stage": "m5.5", "model_ref": base_candidate.id},
        )
    )
    registry = build_runtime_registry()
    runtime_operators = [
        item for item in registry.visible()
        if str(item["name"]).startswith(("runtime_", "vae_", "inference_", "component_", "cache_"))
    ]
    design_gene = _load_gene(root)
    negative_gene = _load_runtime_gene(root)
    genes = [gene for gene in (design_gene, negative_gene) if gene]
    m55_evaluation = _load_m55_evaluation(root, design_gene)
    runner = H3BenchmarkRunner(
        MiniMaxH3Adapter(args.base_url, request_timeout_s=args.request_timeout, poll_interval_s=2, task_timeout_s=1800),
        SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s),
        load_workflow(config.workflow.template),
        config.workflow,
        root / args.benchmark_output,
        system_sample_interval_s=1.0,
    )
    validator = M6ValidationRunner(runner)
    trajectories = TrajectoryStore(root / args.trajectory_output)
    attribution = {
        "primary_intervention": "runtime_memory",
        "secondary_changes": [],
        "controlled_variables": [
            "sampling_steps", "seed", "prompt", "scheduler", "resolution", "cfg",
            "vae", "text_encoder", "lora", "cache_reset_before_each_run", "peak_memory_max_gate",
        ],
        "diagnosis": "vae_tiling was rejected; search lifecycle/offload/cache policies without changing weights",
        "design_gene_read_only": [gene.get("gene_id") for gene in genes if gene.get("gene_id")],
    }
    output = root / args.output
    report_output = root / args.report_output
    started_at = datetime.now(timezone.utc).isoformat()
    campaign_started = time.monotonic()
    budget_state = BudgetState(
        max_iterations=args.max_iterations,
        max_failed_experiments=args.max_failed_experiments,
        max_controller_calls=args.max_iterations,
    )
    seen_fingerprints = set()
    experiment_history: List[Mapping[str, Any]] = []
    failed_history: List[Mapping[str, Any]] = []
    duplicate_fingerprints_blocked = 0
    experience_influence_evidence: List[Mapping[str, Any]] = []
    active_system = system_store.active()
    baseline_state = active_system.evaluation_state(base_model)
    if negative_gene:
        intervention = negative_gene.get("intervention") or {}
        negative_operator = str(intervention.get("operator", ""))
        negative_args = intervention.get("args") or {}
        if negative_operator and isinstance(negative_args, Mapping):
            negative_fingerprint = experiment_fingerprint(negative_operator, negative_args, baseline_state, target.id)
            seen_fingerprints.add(negative_fingerprint)
            experiment_history.append(
                {
                    "experiment_id": "gene:%s" % negative_gene.get("gene_id", "m6-negative"),
                    "operator": negative_operator,
                    "operator_args": dict(negative_args),
                    "outcome": "rejected_candidate",
                    "fingerprint": negative_fingerprint,
                    "source": "validated_negative_experience",
                }
            )
            experience_influence_evidence.append(
                {
                    "source_gene_id": negative_gene.get("gene_id"),
                    "source_status": negative_gene.get("status"),
                    "operator": negative_operator,
                    "operator_args": dict(negative_args),
                    "fingerprint_reserved": negative_fingerprint,
                    "lesson": "same operator/configuration is not retried after clean VRAM rejection",
                }
            )

    payload: Dict[str, Any] = {
        "campaign_id": args.campaign_id,
        "harness": {
            "version": HARNESS_VERSION,
            "status": HARNESS_STATUS,
            "change_policy": HARNESS_CHANGE_POLICY,
        },
        "optimization_campaign": {
            "name": "m6_runtime_memory",
            "mode": "autonomous_inner_loop",
            "controller_selects_operator": True,
            "stop_conditions": ["target_profile_satisfied", "iteration_budget_exhausted", "failure_budget_exhausted"],
            "max_experiments": args.max_iterations,
            "max_failed_experiments": args.max_failed_experiments,
            "failure_definition": "controller/backend/operator/artifact execution failure only",
            "rejection_definition": "valid execution with one or more strict acceptance gates false",
        },
        "target_profile_id": target.id,
        "controller": {"provider": args.controller, "model": args.controller_model},
        "reference_metrics": REFERENCE_METRICS,
        "m5_5_evaluation": m55_evaluation,
        "acceptance_splits": requested_splits,
        "status": "running",
        "human_intervention_count": 0,
        "system_candidates": [candidate.to_dict() for candidate in system_store.lineage()],
        "iterations": [],
        "prior_campaigns": [dict(prior_campaign)] if prior_campaign else [],
        "preflight": dict(preflight),
        "experience_influence_evidence": experience_influence_evidence,
        "gpu_hours_available": False,
    }
    recent: List[Mapping[str, Any]] = [{"stage": "M5.5", "evaluation": dict(m55_evaluation)}]
    if prior_campaign:
        recent.append({"stage": "prior_campaign", **dict(prior_campaign)})
        experience_influence_evidence.append(
            {
                "source_campaign_id": prior_campaign.get("campaign_id"),
                "source_status": prior_campaign.get("status"),
                "lesson": "backend timeout produced no optimization conclusion; prior operator/config remains eligible",
            }
        )
    recent.extend(experiment_history)
    accepted = False
    accepted_system_candidate_id: Optional[str] = None
    accepted_recipe: Optional[Any] = None

    while not budget_state.stop_reason():
        iteration = budget_state.used_iterations
        iteration_started = time.monotonic()
        parent_system = system_store.active()
        current_state = parent_system.evaluation_state(base_model)
        plan = None
        context = None
        plan_dict: Dict[str, Any] = {}
        requested_operator: Optional[str] = None
        operator_args: Dict[str, Any] = {}
        fingerprint: Optional[str] = None
        operator_result: Optional[OperatorResult] = None
        branch_state = current_state
        system_child: Optional[SystemCandidate] = None
        split_results: Dict[str, Any] = {}
        split_errors: Dict[str, Any] = {}
        failure_type: Optional[str] = None
        controller_error: Optional[str] = None

        try:
            plan, context = _controller_plan(
                args,
                target,
                current_state,
                runtime_operators,
                genes,
                m55_evaluation,
                recent,
                failed_history,
                _archive_summary(system_store),
                budget_state,
            )
            plan_dict = plan.to_dict()
            requested_operator = plan.operator
            try:
                operator_args = _effective_operator_args(requested_operator, requested_operator, plan.operator_args)
                fingerprint = experiment_fingerprint(requested_operator, operator_args, current_state, target.id)
                novelty_error = validate_novelty(
                    requested_operator,
                    plan.hypothesis,
                    fingerprint,
                    seen_fingerprints,
                    experiment_history,
                )
                if novelty_error:
                    failure_type = novelty_error
                    if novelty_error == "duplicate_experiment_fingerprint":
                        duplicate_fingerprints_blocked += 1
                else:
                    seen_fingerprints.add(fingerprint)
                    parent_candidate = ModelCandidate(
                        base_candidate.id,
                        base_candidate.parent_id,
                        base_candidate.generation,
                        base_candidate.checkpoint_path,
                        current_state,
                        base_candidate.created_by_experiment_id,
                        base_candidate.status,
                    )
                    operator_result = registry.execute(
                        requested_operator,
                        parent_candidate,
                        operator_args,
                        target,
                        ExecutionContext(
                            root / args.benchmark_output / plan.experiment_id,
                            "M%04d" % (iteration + 2),
                        ),
                    )
            except Exception as exc:
                failure_type = "runtime_policy_invalid"
                controller_error = str(exc)
        except ControllerProviderError as exc:
            failure_type = "controller_failure"
            controller_error = str(exc)

        if operator_result is not None and operator_result.ok and operator_result.output_state is not None:
            branch_state = replace(
                current_state,
                algorithm_state=operator_result.output_state.algorithm_state,
                runtime_state=operator_result.output_state.runtime_state,
                provenance=operator_result.output_state.provenance,
            )
            for split in requested_splits:
                try:
                    result = validator.run(
                        current_state,
                        branch_state,
                        split_tasks[split],
                        label=f"recipe-{iteration + 1}-{requested_operator}-{split}",
                        target=target,
                        reference_metrics=REFERENCE_METRICS,
                        operator_attribution={**attribution, "recipe_iteration": iteration},
                        repetitions=args.repetitions,
                    )
                    split_results[split] = result.to_dict()
                except (BackendError, OSError, ValueError) as exc:
                    split_errors[split] = {
                        "failure_type": getattr(exc, "failure_type", "m6_validation_failure"),
                        "message": str(exc),
                    }
            split_validated = bool(split_results) and not split_errors and all(
                bool(result.get("validated")) for result in split_results.values()
            )
            system_child = SystemCandidate(
                system_store.next_id(),
                parent_system.id,
                parent_system.generation + 1,
                base_candidate.id,
                algorithm_state=branch_state.algorithm_state,
                runtime_state=branch_state.runtime_state,
                evaluation={
                    "operator": requested_operator,
                    "operator_args": operator_args,
                    "fingerprint": fingerprint,
                    "splits": {key: _compact_split(value) for key, value in split_results.items()},
                    "errors": split_errors,
                },
                created_by_experiment_id=plan.experiment_id if plan else None,
                status="accepted" if split_validated else ("failed" if split_errors else "rejected"),
                metadata={"model_ref": base_candidate.id, "runtime_only": True, "parent_system_id": parent_system.id},
            )
            system_store.create(system_child)

        outcome = classify_outcome(
            operator_result.ok if operator_result is not None else False,
            operator_result is not None and operator_result.output_state is not None,
            split_results,
            split_errors,
            bool(split_results) and not split_errors and all(bool(result.get("validated")) for result in split_results.values()),
        )
        if outcome == "failed_experiment" and failure_type is None:
            failure_type = (
                operator_result.failure_type
                if operator_result is not None and operator_result.failure_type
                else next(iter(split_errors.values()), {}).get("failure_type")
                if split_errors
                else "m6_validation_failure"
            )
        if outcome != "failed_experiment":
            failure_type = None
        reason = _acceptance_reason(outcome, failure_type, split_results, split_errors)
        state_after = branch_state if system_child is not None else current_state
        elapsed = time.monotonic() - iteration_started
        actual_cost = CostEstimate(
            wall_time_s=elapsed,
            gpu_hours=operator_result.cost.gpu_hours if operator_result is not None else 0.0,
        )
        next_budget = budget_state.consume(
            actual_cost,
            failed=outcome == "failed_experiment",
            controller_calls=1,
        )
        if outcome == "accepted_candidate" and system_child is not None:
            system_store.set_active(system_child.id)
            active_for_next = system_child
            accepted = True
            accepted_system_candidate_id = system_child.id
            accepted_recipe = list(system_child.runtime_state.get("runtime_recipe") or [])
        else:
            active_for_next = parent_system

        next_context = _next_decision_context(
            active_for_next,
            active_for_next.evaluation_state(base_model),
            next_budget,
            experiment_history,
            failed_history,
        )
        experiment_id = plan.experiment_id if plan is not None else "exp_%04d" % (iteration + 1)
        record: Dict[str, Any] = {
            "iteration": iteration,
            "experiment_id": experiment_id,
            "system_parent_id": parent_system.id,
            "controller_context": context.to_dict() if context is not None else None,
            "controller_plan": plan_dict,
            "controller_error": controller_error,
            "state_before": current_state.to_dict(),
            "diagnosis": plan.diagnosis if plan is not None else None,
            "hypothesis": plan.hypothesis if plan is not None else None,
            "operator": requested_operator,
            "operator_args": operator_args,
            "expected_effects": dict(plan.expected_effects) if plan is not None else {},
            "risks": list(plan.risks) if plan is not None else [],
            "fingerprint": fingerprint,
            "execution_result": operator_result.to_dict() if operator_result is not None else {"status": "not_executed"},
            "operator_executed": operator_result is not None,
            "system_child_id": system_child.id if system_child is not None else None,
            "split_results": split_results,
            "split_errors": split_errors,
            "quality_metrics": {
                split: _compact_split(value).get("branch_metrics", {}) for split, value in split_results.items()
            },
            "hardware_metrics": {
                split: _compact_split(value).get("gates", {}) for split, value in split_results.items()
            },
            "outcome": outcome,
            "accept_reject_reason": reason,
            "failure_type": failure_type,
            "state_after": state_after.to_dict(),
            "active_state_after": active_for_next.evaluation_state(base_model).to_dict(),
            "next_decision_context": next_context,
            "harness_version": HARNESS_VERSION,
            "wall_time_s": elapsed,
            "gpu_hours": actual_cost.gpu_hours,
        }
        payload["iterations"].append(record)
        if system_child is not None:
            payload["system_candidates"].append(system_child.to_dict())
        history_entry = {
            "experiment_id": experiment_id,
            "operator": requested_operator,
            "operator_args": dict(operator_args),
            "hypothesis": record["hypothesis"],
            "diagnosis": record["diagnosis"],
            "outcome": outcome,
            "accept_reject_reason": reason,
            "fingerprint": fingerprint,
            "quality_metrics": record["quality_metrics"],
            "hardware_metrics": record["hardware_metrics"],
        }
        experiment_history.append(history_entry)
        recent.append({"stage": "M6", **history_entry})
        if outcome == "failed_experiment":
            failed_history.append({"stage": "M6", **history_entry, "failure_type": failure_type})
        if requested_operator and requested_operator != "vae_tiling":
            experience_influence_evidence.append(
                {
                    "source_gene_id": negative_gene.get("gene_id") if negative_gene else None,
                    "subsequent_experiment_id": experiment_id,
                    "selected_operator": requested_operator,
                    "reason": "Controller selected a different registered runtime strategy after vae_tiling negative evidence",
                }
            )
        budget_state = next_budget
        payload.update(
            {
                "budget": budget_state.to_dict(),
                "failure_count": budget_state.used_failures,
                "rejected_candidate_count": sum(
                    1 for item in payload["iterations"] if item.get("outcome") == "rejected_candidate"
                ),
                "total_experiments": len(payload["iterations"]),
                "wall_time_s": time.monotonic() - campaign_started,
                "gpu_hours": budget_state.used_gpu_hours,
                "duplicate_fingerprints_blocked": duplicate_fingerprints_blocked,
                "experience_influence_evidence": experience_influence_evidence,
            }
        )
        trajectories.append(
            Trajectory(
                task_id="m6:recipe:%d" % (iteration + 1),
                harness_version=HARNESS_VERSION,
                split="m6",
                inputs={
                    "system_parent_id": parent_system.id,
                    "system_child_id": system_child.id if system_child is not None else None,
                    "state_before": current_state.to_dict(),
                    "controller_context": context.to_dict() if context is not None else None,
                    "controller_plan": plan_dict,
                    "fingerprint": fingerprint,
                    "harness_version": HARNESS_VERSION,
                },
                steps=[
                    {"action": "controller_plan", "plan": plan_dict, "error": controller_error},
                    {"action": "runtime_operator", "operator": requested_operator, "operator_result": record["execution_result"]},
                    {"action": "m6_validation", "splits": split_results, "errors": split_errors},
                    {"action": "campaign_decision", "outcome": outcome, "reason": reason, "next_context": next_context},
                ],
                final_result={"system_child_id": system_child.id if system_child else None, "outcome": outcome},
                score=None,
                failure_type=failure_type,
                cost={"tokens": 0.0, "wall_time": elapsed, "gpu_hours": actual_cost.gpu_hours},
                evaluation={
                    "target_profile_id": target.id,
                    "operator_attribution": attribution,
                    "splits": split_results,
                    "campaign_record": record,
                },
                critical_regression=any(
                    bool((result.get("gates") or {}).get("critical_regression"))
                    for result in split_results.values()
                    if isinstance(result, Mapping)
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        _persist(output, payload)
        if accepted:
            break

    ended_at = datetime.now(timezone.utc).isoformat()
    if accepted:
        termination_reason = "target_satisfied"
        payload["status"] = "accepted"
    elif budget_state.used_failures >= args.max_failed_experiments:
        termination_reason = "failure_budget_exhausted"
        payload["status"] = "failure_budget_exhausted"
    elif budget_state.used_iterations >= args.max_iterations:
        termination_reason = "experiment_budget_exhausted"
        payload["status"] = "experiment_budget_exhausted"
    else:
        termination_reason = "budget_exhausted"
        payload["status"] = "budget_exhausted"
    payload.update(
        {
            "termination_reason": termination_reason,
            "accepted_system_candidate_id": accepted_system_candidate_id,
            "accepted_recipe": accepted_recipe,
            "failure_count": budget_state.used_failures,
            "rejected_candidate_count": sum(
                1 for item in payload["iterations"] if item.get("outcome") == "rejected_candidate"
            ),
            "total_experiments": len(payload["iterations"]),
            "wall_time_s": time.monotonic() - campaign_started,
            "gpu_hours": budget_state.used_gpu_hours,
            "duplicate_fingerprints_blocked": duplicate_fingerprints_blocked,
            "experience_influence_evidence": experience_influence_evidence,
        }
    )
    report = build_research_report(payload, started_at, ended_at)
    payload["research_report"] = report
    _persist(output, payload)
    _persist(report_output, report)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "system_candidate": payload["accepted_system_candidate_id"],
                "experiments": payload["total_experiments"],
                "failed_experiments": payload["failure_count"],
                "rejected_candidates": payload["rejected_candidate_count"],
            },
            ensure_ascii=False,
        )
    )
    print("RESULT_PATH", output.resolve())
    print("REPORT_PATH", report_output.resolve())
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
