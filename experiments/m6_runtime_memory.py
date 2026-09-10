from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.backends.comfyui import BackendError
from harness4h3.backends.comfyui import MiniMaxH3Adapter
from harness4h3.benchmark.h3 import H3BenchmarkRunner
from harness4h3.benchmark.m6 import M6ValidationRunner
from harness4h3.config import load_config
from harness4h3.controller.context import ControllerContext
from harness4h3.controller.provider import OllamaStructuredController, RuleBasedMockController
from harness4h3.controller.schemas import BudgetState
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.harness.loop import load_workflow
from harness4h3.harness.state import Task, load_tasks
from harness4h3.memory.trajectory import Trajectory, TrajectoryStore
from harness4h3.operators.base import ExecutionContext
from harness4h3.operators.runtime_memory import build_runtime_registry
from harness4h3.target.profile import load_target_profile


REFERENCE_METRICS = {"model_size_gb": 20.970379616, "latency_s": 205.33523804112338}


def _state(model_id: str) -> ModelState:
    return ModelState(
        model_id=model_id,
        parent_model_id="M0000" if model_id == "M0001" else "M0001",
        checkpoint_path=r"D:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_nvfp4.safetensors",
        architecture_name="MiniMax-H3",
        parameter_count=11_681_874_744,
        num_blocks=50,
        hidden_size=2688,
        num_attention_heads=32,
        ffn_width=10752,
        dtype="mixed",
        quantization={"bits": 4, "scheme": "nvfp4"},
        sampling_steps=20,
        components={"text_encoder": "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"},
        runtime_state={"backend": "comfyui", "m5_stage": "m5.5", "metrics_stale": False},
        measured_metrics={
            "quality_score": 0.991137,
            "latency_s": 90.28058435407002,
            "peak_memory_gb": 16.29452817,
            "model_size_gb": 12.5286368,
        },
        provenance={"source": "remote Windows RTX 5080", "validated_design_gene": "H3-NVFP4-Quantization-001"},
    )


def _load_gene(root: Path) -> Mapping[str, Any]:
    path = root / "docs/experience/design-gene-h3-nvfp4.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, Mapping) and raw.get("status") in {"validated", "validated_m5_5", "transferred"} else {}


def _load_m55_evaluation(root: Path, gene: Mapping[str, Any]) -> Mapping[str, Any]:
    path = root / "var/m5-controlled/m5.5-validation.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        raw = {}
    if isinstance(raw, Mapping) and isinstance(raw.get("results"), Mapping):
        compact: Dict[str, Any] = {}
        for label, result in raw["results"].items():
            if not isinstance(result, Mapping):
                continue
            compact_result: Dict[str, Any] = {
                "label": result.get("label", label),
                "validated": bool(result.get("validated")),
                "repetitions": result.get("repetitions"),
                "pairwise": result.get("pairwise", []),
            }
            aggregates = result.get("aggregates")
            if isinstance(aggregates, Mapping):
                compact_result["aggregates"] = {
                    str(role): {
                        str(metric): {
                            key: stats.get(key)
                            for key in ("count", "mean", "median", "minimum", "maximum", "stddev")
                            if isinstance(stats, Mapping) and key in stats
                        }
                        for metric, stats in values.items()
                        if isinstance(values, Mapping)
                    }
                    for role, values in aggregates.items()
                    if isinstance(values, Mapping)
                }
            compact[str(label)] = compact_result
        return {"stage": "M5.5", "source": str(path), "results": compact}
    return {
        "stage": "M5.5",
        "source": "validated_design_gene",
        "validated": bool(gene),
        "benefit": dict(gene.get("benefit") or {}) if isinstance(gene, Mapping) else {},
        "remaining_limitation": dict(gene.get("remaining_limitation") or {}) if isinstance(gene, Mapping) else {},
    }


def _controller_plan(
    args: argparse.Namespace,
    target: Any,
    current: ModelState,
    operators: Sequence[Mapping[str, Any]],
    gene: Mapping[str, Any],
    validated_evaluation: Mapping[str, Any],
):
    recent = [{"stage": "M5.5", "evaluation": dict(validated_evaluation)}]
    lessons = list(gene.get("risks_lessons") or []) if isinstance(gene, Mapping) else []
    failures = [{"source": "validated_design_gene", "lessons": lessons, "remaining_limitation": gene.get("remaining_limitation", {})}] if lessons else []
    context = ControllerContext(
        target,
        current,
        BudgetState(max_iterations=4, max_failed_experiments=4, max_controller_calls=4),
        tuple(operators),
        recent_experiments=recent,
        relevant_failures=failures,
        pareto_front=[],
        validated_design_genes=[gene] if gene else [],
        validated_evaluation=validated_evaluation,
    )
    if args.controller == "mock":
        plan = RuleBasedMockController().plan(context)
    else:
        plan = OllamaStructuredController(args.controller_model, args.controller_url, timeout_s=args.controller_timeout).plan(context)
    return plan, context


def _persist(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="M6 cross-layer runtime-memory validation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    parser.add_argument("--base-url", default=os.environ.get("COMFYUI_BASE_URL", "http://100.88.143.10:8188"))
    parser.add_argument("--output", default="var/m6-runtime/m6-validation.json")
    parser.add_argument("--benchmark-output", default="var/m6-runtime/runs")
    parser.add_argument("--trajectory-output", default="var/m6-runtime/trajectories.jsonl")
    parser.add_argument("--controller", choices=("ollama", "mock"), default="ollama")
    parser.add_argument("--controller-model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b-q8_0"))
    parser.add_argument("--controller-url", default=os.environ.get("OLLAMA_BASE_URL", "http://100.88.143.10:11434"))
    parser.add_argument("--controller-timeout", type=float, default=180.0)
    parser.add_argument("--branches", default="controller", help="controller or comma-separated runtime operator names")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    target = load_target_profile(root / args.target)
    tasks = load_tasks(config.runtime.tasks_path)
    split_tasks = {split: [task for task in tasks if task.split == split] for split in ("dev", "heldout")}
    if not split_tasks["dev"] or not split_tasks["heldout"]:
        raise ValueError("M6 requires both dev and heldout tasks")
    parent = _state("M0001")
    registry = build_runtime_registry()
    runtime_operators = [item for item in registry.visible() if str(item["name"]).startswith(("runtime_", "vae_", "inference_"))]
    gene = _load_gene(root)
    m55_evaluation = _load_m55_evaluation(root, gene)
    plan, context = _controller_plan(args, target, parent, runtime_operators, gene, m55_evaluation)
    requested = [item.strip() for item in args.branches.split(",") if item.strip()]
    if requested == ["controller"]:
        requested = [plan.operator]
    known = {str(item["name"]) for item in runtime_operators}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError("unknown runtime branch(es): " + ", ".join(unknown))
    runner = H3BenchmarkRunner(
        MiniMaxH3Adapter(args.base_url, request_timeout_s=30, poll_interval_s=2, task_timeout_s=1800),
        SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s),
        load_workflow(config.workflow.template),
        config.workflow,
        root / args.benchmark_output,
        system_sample_interval_s=1.0,
    )
    validator = M6ValidationRunner(runner)
    trajectory_store = TrajectoryStore(root / args.trajectory_output)
    attribution = {
        "primary_intervention": "runtime_memory",
        "secondary_changes": [],
        "controlled_variables": [
            "sampling_steps", "seed", "prompt", "scheduler", "resolution", "cfg",
            "vae", "text_encoder", "lora", "cache_reset_before_each_run", "peak_memory_max_gate",
        ],
        "diagnosis": "storage and latency are validated; peak runtime VRAM is the sole remaining hard violation",
        "design_gene_read_only": gene.get("gene_id") if gene else None,
    }
    parent_candidate = ModelCandidate("M0001", "M0000", 1, parent.checkpoint_path, parent, "m5.5", "candidate")
    output = root / args.output
    payload: Dict[str, Any] = {
        "target_profile_id": target.id,
        "controller": {"provider": args.controller, "model": args.controller_model, "context": context.to_dict(), "plan": plan.to_dict()},
        "reference_metrics": REFERENCE_METRICS,
        "m5_5_evaluation": m55_evaluation,
        "branches": {},
    }
    for number, operator_name in enumerate(requested, 2):
        operator_args = dict(plan.operator_args) if operator_name == plan.operator else {
            "mode": "aggressive" if operator_name == "runtime_offload" else 4
        }
        if operator_name == "vae_tiling":
            operator_args = {"tile_size": 256, "overlap": 32}
        if operator_name == "inference_chunking":
            operator_args = {"chunk_size": 4}
        child_id = "M%04d" % number
        operator_result = registry.execute(
            operator_name,
            parent_candidate,
            operator_args,
            target,
            ExecutionContext(root / args.benchmark_output / operator_name, child_id),
        )
        branch_payload: Dict[str, Any] = {"operator": operator_name, "operator_args": operator_args, "operator_result": operator_result.to_dict()}
        if operator_result.ok and operator_result.output_state is not None:
            branch_candidate = ModelCandidate(
                child_id,
                parent_candidate.id,
                parent_candidate.generation + 1,
                operator_result.output_state.checkpoint_path,
                operator_result.output_state,
                "%s:%s" % (plan.experiment_id, operator_name),
                "candidate",
                metadata={"operator": operator_name, "operator_args": operator_args, "parent_model_id": parent_candidate.id},
            )
            branch_payload["branch_candidate"] = branch_candidate.to_dict()
            split_results = {}
            split_errors: Dict[str, Any] = {}
            for split, selected in split_tasks.items():
                try:
                    result = validator.run(
                        parent,
                        branch_candidate.state,
                        selected,
                        label=f"{operator_name}-{split}",
                        target=target,
                        reference_metrics=REFERENCE_METRICS,
                        operator_attribution=attribution,
                        repetitions=args.repetitions,
                    )
                    split_results[split] = result.to_dict()
                except (BackendError, OSError, ValueError) as exc:
                    split_errors[split] = {
                        "failure_type": getattr(exc, "failure_type", "m6_validation_failure"),
                        "message": str(exc),
                    }
            branch_payload["splits"] = split_results
            if split_errors:
                branch_payload["errors"] = split_errors
            branch_payload["validated"] = bool(split_results) and not split_errors and all(
                item["validated"] for item in split_results.values()
            )
        else:
            branch_payload["splits"] = {}
            branch_payload["validated"] = False
        payload["branches"][operator_name] = branch_payload
        split_scores = [
            result.get("aggregates", {}).get("branch", {}).get("quality_score", {}).get("mean")
            for result in branch_payload.get("splits", {}).values()
            if isinstance(result, Mapping)
        ]
        score_values = [float(value) for value in split_scores if isinstance(value, (int, float))]
        branch_failure = None
        if not branch_payload["validated"]:
            branch_failure = operator_result.failure_type
            if branch_failure is None:
                errors = branch_payload.get("errors", {})
                if isinstance(errors, Mapping):
                    branch_failure = next(
                        (str(item.get("failure_type")) for item in errors.values() if isinstance(item, Mapping) and item.get("failure_type")),
                        None,
                    )
            branch_failure = branch_failure or "m6_gate_failed"
        trajectory_store.append(
            Trajectory(
                task_id="m6:%s" % operator_name,
                harness_version=child_id,
                split="m6",
                inputs={
                    "parent_model_id": parent.model_id,
                    "operator": operator_name,
                    "operator_args": operator_args,
                    "target_profile_id": target.id,
                    "controller_plan": plan.to_dict(),
                },
                steps=[
                    {"action": "controller_plan", "operator": plan.operator, "operator_args": dict(plan.operator_args)},
                    {"action": "runtime_operator", "operator": operator_name, "operator_result": operator_result.to_dict()},
                    {"action": "m6_validation", "splits": branch_payload.get("splits", {}), "errors": branch_payload.get("errors", {})},
                ],
                final_result={"branch_model_id": child_id, "validated": branch_payload["validated"]},
                score=(sum(score_values) / len(score_values)) if score_values else None,
                failure_type=branch_failure,
                cost={"tokens": 0.0, "wall_time": float(operator_result.cost.wall_time_s)},
                evaluation={
                    "target_profile_id": target.id,
                    "operator_attribution": attribution,
                    "splits": branch_payload.get("splits", {}),
                    "errors": branch_payload.get("errors", {}),
                },
                critical_regression=not branch_payload["validated"],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        _persist(output, payload)
        print(json.dumps({"branch": operator_name, "validated": branch_payload["validated"]}, ensure_ascii=False))
    _persist(output, payload)
    print("RESULT_PATH", output.resolve())
    return 0 if payload["branches"] and all(item["validated"] for item in payload["branches"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
