from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import Harness4H3Error
from .archive.store import CandidateStore
from .archive.model_candidate import ModelCandidate
from .archive.model_store import ModelStore
from .archive.pareto import ParetoArchive
from .benchmark.h3 import H3BenchmarkRunner
from .config import AppConfig, ConfigError, load_config
from .controller.loop import OptimizationLoop
from .controller.provider import OllamaStructuredController, OpenAIResponsesController, RuleBasedMockController
from .controller.schemas import BudgetState
from .evaluator.composite import CompositeEvaluator
from .evaluator.constraints import ConstraintEvaluator
from .evaluator.evaluator import SubprocessEvaluator, make_request
from .evaluator.hardware import FakeHardwareEvaluator
from .evaluator.quality import FakeQualityEvaluator
from .harness.loop import HarnessRunner, load_workflow
from .harness.state import Task, load_tasks
from .h3.fake import FakeH3Model
from .h3.inspector import H3Inspector
from .memory.experiment_store import ExperimentStore
from .memory.trajectory import TrajectoryStore
from .model.minimax_h3 import MiniMaxH3Adapter
from .operators.fake import FakeOperatorBackend, build_fake_registry
from .self_improve.evolve import EvolutionController
from .target.profile import load_target_profile


def _emit(value: Any, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    elif isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _config(args: argparse.Namespace) -> AppConfig:
    return load_config(Path(args.config))


def _tasks(config: AppConfig, override: Optional[str]) -> List[Task]:
    return load_tasks(Path(override).resolve() if override else config.runtime.tasks_path)


def _components(config: AppConfig):
    base_url = os.environ.get("COMFYUI_BASE_URL", config.backend.base_url).strip().rstrip("/")
    backend = MiniMaxH3Adapter(
        base_url=base_url,
        request_timeout_s=config.backend.request_timeout_s,
        poll_interval_s=config.backend.poll_interval_s,
        task_timeout_s=config.backend.task_timeout_s,
    )
    evaluator = SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s)
    trajectories = TrajectoryStore(config.runtime.trajectory_path)
    archive = CandidateStore(config.runtime.archive_dir)
    workflow_template = load_workflow(config.workflow.template)
    baseline_policy: Dict[str, Any] = {
        "prompt": {"prefix": "", "suffix": ""},
        "context": {"include_constraints": True, "max_prompt_chars": 4000},
        "workflow": {
            key: workflow_template[target.node_id]["inputs"][target.input_name]
            for key, target in config.workflow.mutable.items()
        },
    }
    archive.initialize(baseline_policy)
    runner = HarnessRunner(
        workflow_template=workflow_template,
        workflow_config=config.workflow,
        backend=backend,
        evaluator=evaluator,
        trajectories=trajectories,
        output_root=config.runtime.output_dir,
    )
    return runner, trajectories, archive, evaluator


def cmd_validate(args: argparse.Namespace) -> int:
    config = _config(args)
    tasks = _tasks(config, args.tasks)
    _emit(
        {
            "valid": True,
            "workflow": str(config.workflow.template),
            "tasks": len(tasks),
            "splits": sorted(set(task.split for task in tasks)),
        },
        args.json,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _config(args)
    tasks = [task for task in _tasks(config, args.tasks) if args.split == "all" or task.split == args.split]
    if not tasks:
        raise ConfigError("no tasks selected for split %s" % args.split)
    runner, _, archive, _ = _components(config)
    candidate = archive.get(args.candidate) if args.candidate else archive.active()
    results = runner.run_batch(tasks, candidate)
    summary = {
        "candidate": candidate.id,
        "tasks": len(results),
        "passed": sum(1 for item in results if item.score is not None and item.score > 0 and not item.critical_regression),
        "mean_score": sum(float(item.score or 0) for item in results) / len(results),
        "failures": [{"task_id": item.task_id, "failure_type": item.failure_type} for item in results if item.failure_type],
    }
    _emit(summary, args.json)
    return 0 if summary["passed"] == len(results) else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    tasks_by_id = {task.id: task for task in _tasks(config, args.tasks)}
    source = TrajectoryStore(Path(args.trajectory).resolve() if args.trajectory else config.runtime.trajectory_path)
    evaluator = SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s)
    output = Path(args.output).resolve() if args.output else source.path.with_suffix(source.path.suffix + ".reevaluated")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("a", encoding="utf-8") as handle:
        for trajectory in source.read():
            task = tasks_by_id.get(trajectory.task_id)
            if task is None:
                raise ConfigError("trajectory references unknown task %s" % trajectory.task_id)
            result_data = trajectory.final_result if isinstance(trajectory.final_result, dict) else {}
            artifacts = result_data.get("artifacts") or []
            evaluation = evaluator.evaluate(
                make_request(
                    task.id,
                    task.expected,
                    artifacts,
                    float(trajectory.cost.get("wall_time", 0)),
                    backend_success=bool(artifacts),
                )
            )
            handle.write(json.dumps({"task_id": task.id, "harness_version": trajectory.harness_version, **evaluation.__dict__}, ensure_ascii=False, default=dict) + "\n")
            count += 1
    _emit({"evaluated": count, "output": str(output)}, args.json)
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    config = _config(args)
    tasks = _tasks(config, args.tasks)
    runner, trajectories, archive, _ = _components(config)
    controller = EvolutionController(archive, config.evolution, runner.run_batch)
    outcome = controller.evolve(tasks, list(trajectories.read()))
    _emit(outcome.to_dict(), args.json)
    return 0


def cmd_legacy_lineage(args: argparse.Namespace) -> int:
    config = _config(args)
    archive = CandidateStore(config.runtime.archive_dir)
    if not archive.active_path.exists():
        _emit({"active": None, "lineage": []}, args.json)
        return 0
    active = archive.active_id
    items = []
    for candidate in archive.lineage():
        outcome = archive.outcome(candidate.id)
        items.append(
            {
                "id": candidate.id,
                "parent": candidate.parent,
                "generation": candidate.generation,
                "mutation_type": candidate.mutation_type,
                "active": candidate.id == active,
                "status": "active" if candidate.id == active else ((outcome or {}).get("status") or "archived"),
                "score": (outcome or {}).get("candidate_score"),
            }
        )
    _emit({"active": active, "lineage": items}, args.json)
    return 0


def _session_root(args: argparse.Namespace) -> Path:
    return Path(args.session_dir).resolve()


def _fake_optimization_loop(root: Path, controller=None) -> OptimizationLoop:
    return OptimizationLoop(
        controller=controller or RuleBasedMockController(),
        operators=build_fake_registry(FakeOperatorBackend()),
        evaluator=CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator()),
        models=ModelStore(root / "models"),
        pareto=ParetoArchive(root / "pareto"),
        experiments=ExperimentStore(root / "experiments.jsonl"),
        run_root=root / "runs",
        checkpoint_path=root / "session.json",
    )


def cmd_validate_target(args: argparse.Namespace) -> int:
    target = load_target_profile(Path(args.target).resolve())
    _emit({"valid": True, "target": target.to_dict()}, args.json)
    return 0


def cmd_inspect_model(args: argparse.Namespace) -> int:
    if args.checkpoint:
        state = H3Inspector().inspect(
            Path(args.checkpoint),
            model_id=args.model_id,
            parent_model_id=args.parent_model_id,
            architecture_name=args.architecture,
            sampling_steps=args.sampling_steps,
            include_file_sha256=args.sha256,
        )
        backend = "checkpoint"
    else:
        state = FakeH3Model.baseline().state
        backend = "fake"
    _emit({"backend": backend, "model_state": state.to_dict()}, args.json)
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    target = load_target_profile(Path(args.target).resolve())
    root = _session_root(args)
    if args.controller == "ollama":
        controller = OllamaStructuredController(args.controller_model, args.controller_url, args.controller_timeout_s)
    elif args.controller == "openai":
        controller = OpenAIResponsesController(
            args.controller_model,
            args.controller_url,
            args.controller_api_key_env,
            args.controller_timeout_s,
        )
    else:
        controller = RuleBasedMockController()
    loop = _fake_optimization_loop(root, controller)
    initial = FakeH3Model.baseline().candidate()
    budget = BudgetState(
        max_iterations=args.max_iterations,
        max_failed_experiments=args.max_failed_experiments,
        max_wall_time_s=args.max_wall_time_s,
        max_gpu_hours=args.max_gpu_hours,
        max_controller_calls=args.max_controller_calls,
    )
    result = loop.run(args.session_id, target, budget, initial)
    payload = {
        "session_id": args.session_id,
        "status": result.status,
        "current_model_id": result.current_model_id,
        "budget": result.budget.to_dict(),
        "search_metrics": {
            "experiments_to_target": result.budget.used_iterations if result.status == "target_satisfied" else None,
            "gpu_hours_to_target": result.budget.used_gpu_hours if result.status == "target_satisfied" else None,
            "wall_time_to_target_s": result.budget.used_wall_time_s if result.status == "target_satisfied" else None,
            "failed_experiments": result.budget.used_failures,
            "human_intervention_count": 0,
        },
        "lineage": [candidate.to_dict() for candidate in loop.models.lineage()],
        "pareto_front": [entry.to_dict() for entry in loop.pareto.front()],
    }
    _emit(payload, args.json)
    return 0 if result.status == "target_satisfied" else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = _config(args)
    target = load_target_profile(Path(args.target).resolve()) if args.target else None
    state = H3Inspector().inspect(
        Path(args.checkpoint),
        model_id=args.model_id,
        parent_model_id=args.parent_model_id,
        sampling_steps=args.sampling_steps,
        include_file_sha256=args.sha256,
    )
    tasks = [task for task in _tasks(config, args.tasks) if args.split == "all" or task.split == args.split]
    if not tasks:
        raise ConfigError("no tasks selected for split %s" % args.split)
    base_url = (args.base_url or os.environ.get("COMFYUI_BASE_URL", config.backend.base_url)).strip().rstrip("/")
    backend = MiniMaxH3Adapter(
        base_url=base_url,
        request_timeout_s=config.backend.request_timeout_s,
        poll_interval_s=config.backend.poll_interval_s,
        task_timeout_s=args.task_timeout_s,
    )
    runner = H3BenchmarkRunner(
        backend,
        SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s),
        load_workflow(config.workflow.template),
        config.workflow,
        Path(args.output_root),
        system_sample_interval_s=args.sample_interval_s,
    )
    summary = runner.run(state, tasks, baseline_quality=args.baseline_quality, target=target)
    payload = {"target_profile_id": target.id if target else None, **summary.to_dict()}
    if args.result:
        result_path = Path(args.result).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["result"] = str(result_path)
    _emit(payload, args.json)
    return 0 if summary.feasible is not False else 1


def cmd_model_lineage(args: argparse.Namespace) -> int:
    store = ModelStore(_session_root(args) / "models")
    items = store.lineage()
    active = store.active_id if items else None
    _emit(
        {
            "active": active,
            "lineage": [
                {
                    "id": item.id,
                    "parent_id": item.parent_id,
                    "generation": item.generation,
                    "operator": item.metadata.get("operator"),
                    "active": item.id == active,
                }
                for item in items
            ],
        },
        args.json,
    )
    return 0


def cmd_pareto(args: argparse.Namespace) -> int:
    entries = ParetoArchive(_session_root(args) / "pareto").front()
    _emit({"pareto_front": [entry.to_dict() for entry in entries]}, args.json)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    records = list(ExperimentStore(_session_root(args) / "experiments.jsonl").read())
    if args.experiment_id:
        records = [item for item in records if item.experiment_id == args.experiment_id]
    _emit({"experiments": [item.to_dict() for item in records]}, args.json)
    return 0


def _json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="emit one machine-readable JSON object")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvoGen-RSI Phase-I model optimization harness for MiniMax H3")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--json", action="store_true", help="emit one machine-readable JSON object")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate configuration, workflow, and tasks without calling ComfyUI")
    validate.add_argument("--tasks")
    _json_flag(validate)
    validate.set_defaults(handler=cmd_validate)

    run = subparsers.add_parser("run", help="run a task split and append trajectories")
    run.add_argument("--tasks")
    run.add_argument("--split", choices=("sanity", "dev", "heldout", "all"), default="dev")
    run.add_argument("--candidate")
    _json_flag(run)
    run.set_defaults(handler=cmd_run)

    evaluate = subparsers.add_parser("evaluate", help="reevaluate recorded artifacts without rewriting trajectories")
    evaluate.add_argument("--tasks")
    evaluate.add_argument("--trajectory")
    evaluate.add_argument("--output")
    _json_flag(evaluate)
    evaluate.set_defaults(handler=cmd_evaluate)

    evolve = subparsers.add_parser("evolve", help="diagnose failures and gate an isolated candidate")
    evolve.add_argument("--tasks")
    _json_flag(evolve)
    evolve.set_defaults(handler=cmd_evolve)

    validate_target = subparsers.add_parser("validate", help="validate an EvoGen TargetProfile")
    validate_target.add_argument("--target", default="configs/targets/mobile_example.yaml")
    _json_flag(validate_target)
    validate_target.set_defaults(handler=cmd_validate_target)

    inspect = subparsers.add_parser("inspect", help="inspect a real H3 safetensors checkpoint or the Fake H3 baseline")
    inspect.add_argument("--checkpoint", help="local .safetensors checkpoint; weights are never loaded")
    inspect.add_argument("--model-id", default="M0000")
    inspect.add_argument("--parent-model-id")
    inspect.add_argument("--architecture", help="explicit architecture label when metadata is insufficient")
    inspect.add_argument("--sampling-steps", type=int)
    inspect.add_argument("--sha256", action="store_true", help="stream the complete file to calculate SHA256")
    _json_flag(inspect)
    inspect.set_defaults(handler=cmd_inspect_model)

    optimize = subparsers.add_parser("optimize", help="run the fully offline Fake H3 optimization loop")
    optimize.add_argument("--target", default="configs/targets/mobile_example.yaml")
    optimize.add_argument("--session-dir", default="var/evogen")
    optimize.add_argument("--session-id", default="mobile_h3_fake_001")
    optimize.add_argument("--controller", choices=("mock", "ollama", "openai"), default="mock")
    optimize.add_argument("--controller-model", default="qwen3.5:4b")
    optimize.add_argument("--controller-url", default="http://127.0.0.1:11434")
    optimize.add_argument("--controller-api-key-env", default="OPENAI_API_KEY")
    optimize.add_argument("--controller-timeout-s", type=float, default=180.0)
    optimize.add_argument("--max-iterations", type=int, default=5)
    optimize.add_argument("--max-failed-experiments", type=int, default=4)
    optimize.add_argument("--max-wall-time-s", type=float)
    optimize.add_argument("--max-gpu-hours", type=float, default=1.0)
    optimize.add_argument("--max-controller-calls", type=int, default=5)
    _json_flag(optimize)
    optimize.set_defaults(handler=cmd_optimize)

    benchmark = subparsers.add_parser("benchmark", help="run the independent real H3/ComfyUI benchmark")
    benchmark.add_argument("--checkpoint", required=True, help="local H3 .safetensors or .gguf checkpoint")
    benchmark.add_argument("--model-id", default="M0000")
    benchmark.add_argument("--parent-model-id")
    benchmark.add_argument("--sampling-steps", type=int)
    benchmark.add_argument("--sha256", action="store_true")
    benchmark.add_argument("--target", help="optional immutable TargetProfile for feasibility evaluation")
    benchmark.add_argument("--tasks")
    benchmark.add_argument("--split", choices=("sanity", "dev", "heldout", "all"), default="sanity")
    benchmark.add_argument("--base-url")
    benchmark.add_argument("--output-root", default="var/benchmark")
    benchmark.add_argument("--result")
    benchmark.add_argument("--baseline-quality", type=float)
    benchmark.add_argument("--sample-interval-s", type=float, default=1.0)
    benchmark.add_argument("--task-timeout-s", type=float, default=3600.0)
    _json_flag(benchmark)
    benchmark.set_defaults(handler=cmd_benchmark)

    lineage = subparsers.add_parser("lineage", help="show immutable model-candidate lineage")
    lineage.add_argument("--session-dir", default="var/evogen")
    _json_flag(lineage)
    lineage.set_defaults(handler=cmd_model_lineage)

    pareto = subparsers.add_parser("pareto", help="show the model Pareto front")
    pareto.add_argument("--session-dir", default="var/evogen")
    _json_flag(pareto)
    pareto.set_defaults(handler=cmd_pareto)

    replay = subparsers.add_parser("replay", help="read append-only model optimization experiments")
    replay.add_argument("--session-dir", default="var/evogen")
    replay.add_argument("--experiment-id")
    _json_flag(replay)
    replay.set_defaults(handler=cmd_replay)

    legacy_lineage = subparsers.add_parser("legacy-lineage", help="show frozen workflow-candidate history")
    _json_flag(legacy_lineage)
    legacy_lineage.set_defaults(handler=cmd_legacy_lineage)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (Harness4H3Error, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc), "type": exc.__class__.__name__}, ensure_ascii=False))
        else:
            print("harness4h3: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
