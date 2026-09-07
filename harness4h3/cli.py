from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import Harness4H3Error
from .archive.store import CandidateStore
from .config import AppConfig, ConfigError, load_config
from .evaluator.evaluator import SubprocessEvaluator, make_request
from .harness.loop import HarnessRunner, load_workflow
from .harness.state import Task, load_tasks
from .memory.trajectory import TrajectoryStore
from .model.minimax_h3 import MiniMaxH3Adapter
from .self_improve.evolve import EvolutionController


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


def cmd_lineage(args: argparse.Namespace) -> int:
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


def _json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="emit one machine-readable JSON object")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight self-improving harness for MiniMax-H3 video generation")
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

    lineage = subparsers.add_parser("lineage", help="show candidate parent-child history")
    _json_flag(lineage)
    lineage.set_defaults(handler=cmd_lineage)
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
