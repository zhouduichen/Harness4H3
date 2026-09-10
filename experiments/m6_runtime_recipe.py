from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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

from experiments.m6_runtime_memory import (
    REFERENCE_METRICS,
    _effective_operator_args,
    _load_gene,
    _load_m55_evaluation,
    _persist,
    _state,
)


def _load_runtime_gene(root: Path) -> Mapping[str, Any]:
    path = root / "docs/experience/design-gene-m6-vae-tiling.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return raw if isinstance(raw, Mapping) and raw.get("status") else {}


def _controller_plan(
    args: argparse.Namespace,
    target: Any,
    current: Any,
    operators: Sequence[Mapping[str, Any]],
    genes: Sequence[Mapping[str, Any]],
    validated_evaluation: Mapping[str, Any],
    recent: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    budget: BudgetState,
):
    context = ControllerContext(
        target,
        current,
        budget,
        tuple(operators),
        recent_experiments=[dict(item) for item in recent],
        relevant_failures=[dict(item) for item in failures],
        pareto_front=[],
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
    parser = argparse.ArgumentParser(description="M6 bounded runtime-recipe search")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    parser.add_argument("--base-url", default=os.environ.get("COMFYUI_BASE_URL", "http://100.88.143.10:8188"))
    parser.add_argument("--output", default="var/m6-runtime/m6-recipe-validation.json")
    parser.add_argument("--benchmark-output", default="var/m6-runtime/recipe-runs")
    parser.add_argument("--trajectory-output", default="var/m6-runtime/recipe-trajectories.jsonl")
    parser.add_argument("--system-store-output", default="var/m6-runtime/system-candidates")
    parser.add_argument("--controller", choices=("ollama", "mock"), default="ollama")
    parser.add_argument("--controller-model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b-q8_0"))
    parser.add_argument("--controller-url", default=os.environ.get("OLLAMA_BASE_URL", "http://100.88.143.10:11434"))
    parser.add_argument("--controller-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--splits", default="dev,heldout")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--max-failed-experiments", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.max_iterations <= 0 or args.max_failed_experiments < 0:
        raise ValueError("iteration and failure budgets must be positive/non-negative")

    root = Path(__file__).resolve().parents[1]
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
    system = system_store.initialize(
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
    payload: Dict[str, Any] = {
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
        },
        "target_profile_id": target.id,
        "controller": {"provider": args.controller, "model": args.controller_model},
        "reference_metrics": REFERENCE_METRICS,
        "m5_5_evaluation": m55_evaluation,
        "acceptance_splits": requested_splits,
        "status": "running",
        "system_candidates": [system.to_dict()],
        "iterations": [],
    }
    recent: List[Mapping[str, Any]] = [{"stage": "M5.5", "evaluation": dict(m55_evaluation)}]
    failures: List[Mapping[str, Any]] = []
    accepted = False
    for iteration in range(args.max_iterations):
        current_state = system.evaluation_state(base_model)
        budget = BudgetState(
            max_iterations=args.max_iterations,
            max_failed_experiments=args.max_failed_experiments,
            max_controller_calls=args.max_iterations,
            used_iterations=iteration,
            used_failures=len(failures),
            used_controller_calls=iteration,
        )
        try:
            plan, context = _controller_plan(
                args, target, current_state, runtime_operators, genes, m55_evaluation,
                recent, failures, budget,
            )
        except ControllerProviderError as exc:
            failure = {"failure_type": "controller_failure", "message": str(exc), "iteration": iteration}
            failures.append(failure)
            payload["iterations"].append(failure)
            _persist(output, payload)
            if len(failures) >= args.max_failed_experiments:
                break
            continue

        plan_dict = plan.to_dict()
        requested_operator = plan.operator
        operator_args: Dict[str, Any] = {}
        try:
            operator_args = _effective_operator_args(requested_operator, requested_operator, plan.operator_args)
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
                ExecutionContext(root / args.benchmark_output / requested_operator, "M%04d" % (iteration + 2)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            operator_result = OperatorResult(
                status="failed",
                output_state=None,
                cost=CostEstimate(),
                failure_type="runtime_policy_invalid",
                message=str(exc),
            )

        iteration_payload: Dict[str, Any] = {
            "iteration": iteration,
            "system_parent_id": system.id,
            "controller_context": context.to_dict(),
            "controller_plan": plan_dict,
            "operator": requested_operator,
            "operator_args": operator_args,
            "operator_result": operator_result.to_dict(),
        }
        split_results: Dict[str, Any] = {}
        split_errors: Dict[str, Any] = {}
        system_child: SystemCandidate | None = None
        if operator_result.ok and operator_result.output_state is not None:
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
                system.id,
                system.generation + 1,
                base_candidate.id,
                algorithm_state=branch_state.algorithm_state,
                runtime_state=branch_state.runtime_state,
                evaluation={
                    "operator": requested_operator,
                    "operator_args": operator_args,
                    "splits": {key: _compact_split(value) for key, value in split_results.items()},
                    "errors": split_errors,
                },
                created_by_experiment_id=plan.experiment_id,
                status="accepted" if split_validated else "rejected",
                metadata={"model_ref": base_candidate.id, "runtime_only": True},
            )
            system_store.create(system_child)
            iteration_payload["system_child"] = system_child.to_dict()
            if split_validated:
                system_store.set_active(system_child.id)
                accepted = True
            else:
                failures.append({
                    "failure_type": "acceptance_rejected",
                    "iteration": iteration,
                    "operator": requested_operator,
                    "splits": {key: _compact_split(value) for key, value in split_results.items()},
                    "errors": split_errors,
                })
                # Continue from the rejected runtime composition so the next
                # Controller can append a complementary recipe step.
                system = system_child
                recent.append({"stage": "M6", "evaluation": iteration_payload["system_child"]["evaluation"]})
        else:
            failures.append({
                "failure_type": operator_result.failure_type or "runtime_policy_invalid",
                "iteration": iteration,
                "operator": requested_operator,
                "message": operator_result.message,
            })
        iteration_payload["splits"] = split_results
        if split_errors:
            iteration_payload["errors"] = split_errors
        payload["iterations"].append(iteration_payload)
        if system_child is not None:
            payload["system_candidates"].append(system_child.to_dict())
        _persist(output, payload)
        trajectories.append(
            Trajectory(
                task_id="m6:recipe:%d" % (iteration + 1),
                harness_version=HARNESS_VERSION,
                split="m6",
                inputs={
                    "system_parent_id": iteration_payload["system_parent_id"],
                    "system_child_id": system_child.id if system_child is not None else None,
                    "controller_plan": plan_dict,
                    "harness_version": HARNESS_VERSION,
                },
                steps=[
                    {"action": "controller_plan", "operator": requested_operator, "operator_args": operator_args},
                    {"action": "runtime_operator", "operator_result": operator_result.to_dict()},
                    {"action": "m6_validation", "splits": split_results, "errors": split_errors},
                ],
                final_result={"system_child_id": system_child.id if system_child else None, "validated": accepted},
                score=None,
                failure_type=None if accepted else (failures[-1].get("failure_type") if failures else None),
                cost={"tokens": 0.0, "wall_time": float(operator_result.cost.wall_time_s)},
                evaluation={"target_profile_id": target.id, "operator_attribution": attribution, "splits": split_results},
                critical_regression=not accepted,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        if accepted or len(failures) >= args.max_failed_experiments:
            break

    payload["status"] = "accepted" if accepted else "budget_exhausted"
    payload["accepted_system_candidate_id"] = system.id if accepted else None
    payload["failure_count"] = len(failures)
    payload["failures"] = failures
    _persist(output, payload)
    print(json.dumps({"status": payload["status"], "system_candidate": payload["accepted_system_candidate_id"]}, ensure_ascii=False))
    print("RESULT_PATH", output.resolve())
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
