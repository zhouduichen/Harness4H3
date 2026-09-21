from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from .controller.provider import (
    OllamaStructuredController,
    OpenAIResponsesController,
    RuleBasedMockController,
    build_controller_from_config,
)
from .controller.directive import submit_directive
from .controller.schemas import BudgetState, HardwareMetrics
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
from .memory.observation import ControllerEventStore, ObservationStore
from .memory.trajectory import TrajectoryStore
from .model.minimax_h3 import MiniMaxH3Adapter
from .remote.config import load_remote_campaign_config
from .remote.importer import RemoteResultImporter
from .remote.ssh import LocalCommandClient, RemoteError, SSHClient
from .operators.fake import FakeOperatorBackend, build_fake_registry
from .self_improve.evolve import EvolutionController
from research.experiments.a0_model_evolution import (
    A0Budget,
    A0RuleBasedController,
    default_validated_nvfp4_candidate,
    run_campaign,
)
from research.experiments.a1_real_evolution import run_a1_active
from .operators.model_evolution import build_external_model_evolution_registry, build_model_evolution_registry
from .executor.local import LocalProcessExecutor
from .target.profile import load_target_profile
from research.experiments.remote_h3_closed_loop import RemoteCampaign
from .student.campaign import StudentCampaign, build_student_control_plane, build_student_proposal_provider
from .student.compiler import StudentCompiler
from .student.config import StudentConfigError, load_student_campaign_config
from .student.proposal import StudentProposal
from .student.remote import RemoteStudentBaseline, RemoteStudentEvaluator, RemoteStudentRetention, RemoteStudentSupervisor, RemoteStudentWorker


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
        "current_system_id": result.current_system_id,
        "budget": result.budget.to_dict(),
        "search_metrics": {
            "experiments_to_target": result.budget.used_iterations if result.status == "target_satisfied" else None,
            "gpu_hours_to_target": result.budget.used_gpu_hours if result.status == "target_satisfied" else None,
            "wall_time_to_target_s": result.budget.used_wall_time_s if result.status == "target_satisfied" else None,
            "failed_experiments": result.budget.used_failures,
            "human_intervention_count": 0,
        },
        "lineage": [candidate.to_dict() for candidate in loop.models.lineage()],
        "system_lineage": [candidate.to_dict() for candidate in loop.systems.lineage()],
        "pareto_front": [entry.to_dict() for entry in loop.pareto.front()],
    }
    _emit(payload, args.json)
    return 0 if result.status == "target_satisfied" else 1


def cmd_a0_evolve(args: argparse.Namespace) -> int:
    target = load_target_profile(Path(args.target).resolve())
    if args.controller == "ollama":
        controller = OllamaStructuredController(args.controller_model, args.controller_url, args.controller_timeout_s)
    else:
        controller = A0RuleBasedController()
    evaluator = CompositeEvaluator(FakeQualityEvaluator(), FakeHardwareEvaluator(), ConstraintEvaluator())
    output_root = Path(args.output_root).resolve()
    if args.external_operator_command:
        registry = build_external_model_evolution_registry(
            args.external_operator_command,
            executor=LocalProcessExecutor(timeout_s=args.external_operator_timeout_s),
            timeout_s=args.external_operator_timeout_s,
        )
        offline_simulation = False
    else:
        registry = build_model_evolution_registry()
        offline_simulation = True
    result = run_campaign(
        target=target,
        controller=controller,
        registry=registry,
        evaluator=evaluator,
        output_root=output_root,
        initial_candidate=default_validated_nvfp4_candidate(),
        budget=A0Budget(
            max_gpu_hours=args.max_gpu_hours,
            max_experiments=args.max_experiments,
            max_failed_experiments=args.max_failed_experiments,
        ),
        stop_on_target=args.stop_on_target,
        offline_simulation=offline_simulation,
    )
    if args.output:
        result_path = Path(args.output).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_output:
        report_path = Path(args.report_output).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "campaign": "A0",
        "status": result.status,
        "target_satisfied": result.report["target_satisfied"],
        "experiments": result.report["total_experiments"],
        "accepted_candidates": len(result.report["accepted_experiments"]),
        "rejected_candidates": result.report["rejected_candidate_count"],
        "failed_experiments": result.report["failed_experiment_count"],
        "output_root": str(output_root),
        "report": str(Path(args.report_output).resolve()) if args.report_output else str(output_root / "report.json"),
        "offline_simulation": result.report["offline_simulation"],
    }
    _emit(summary, args.json)
    return 0 if result.status in {"completed", "target_satisfied"} else 1


def cmd_a1_evolve(args: argparse.Namespace) -> int:
    target = load_target_profile(Path(args.target).resolve())
    if args.controller == "mock":
        controller = A0RuleBasedController()
    else:
        controller = OllamaStructuredController(args.controller_model, args.controller_url, args.controller_timeout_s)
    result = run_a1_active(
        parent_checkpoint=Path(args.parent_checkpoint),
        worker_command=tuple(args.worker_command) + ("--config", str(Path(args.worker_config).resolve())),
        config_path=Path(args.config).resolve(),
        target=target,
        output_root=Path(args.output_root).resolve(),
        base_url=args.base_url,
        baseline_quality=args.baseline_quality,
        baseline_hardware=HardwareMetrics(
            model_size_gb=args.baseline_model_size_gb,
            latency_s=args.baseline_latency_s,
            peak_memory_gb=args.baseline_peak_memory_gb,
        ),
        controller=controller,
        max_experiments=args.max_experiments,
        max_gpu_hours=args.max_gpu_hours,
        max_failed_experiments=args.max_failed_experiments,
        worker_timeout_s=args.worker_timeout_s,
    )
    _emit(
        {
            "campaign": "A1",
            "status": result.status,
            "target_satisfied": result.report["target_satisfied"],
            "experiments": result.report["total_experiments"],
            "output_root": str(Path(args.output_root).resolve()),
            "report": str(Path(args.output_root).resolve() / "report.json"),
            "offline_simulation": False,
        },
        args.json,
    )
    return 0 if result.status in {"completed", "target_satisfied"} else 1


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
    attribution = {
        "primary_intervention": args.primary_intervention,
        "secondary_changes": list(args.secondary_change or []),
        "controlled_variables": list(args.controlled_variable or []),
        "rationale": args.rationale,
    }
    baseline_hardware = None
    if any(value is not None for value in (args.baseline_model_size_gb, args.baseline_latency_s, args.baseline_peak_memory_gb)):
        baseline_hardware = HardwareMetrics(
            model_size_gb=args.baseline_model_size_gb,
            latency_s=args.baseline_latency_s,
            peak_memory_gb=args.baseline_peak_memory_gb,
        )
    summary = runner.run(
        state,
        tasks,
        baseline_quality=args.baseline_quality,
        target=target,
        operator_attribution=attribution,
        baseline_hardware=baseline_hardware,
        black_frame_rate_threshold=args.black_frame_rate_threshold,
        reset_backend_before_run=args.reset_backend_before_run,
    )
    payload = {"target_profile_id": target.id if target else None, **summary.to_dict()}
    if args.result:
        result_path = Path(args.result).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["result"] = str(result_path)
    _emit(payload, args.json)
    return 0 if summary.feasible is not False else 1


class _NoWriteExperienceStore:
    def append(self, record):
        return True


def cmd_import_experience(args: argparse.Namespace) -> int:
    config = load_remote_campaign_config(Path(args.remote_config))
    output = Path(args.output).resolve()
    store = _NoWriteExperienceStore()
    if not args.dry_run:
        from .memory.experience import ExperienceStore

        store = ExperienceStore(output)
    summary = RemoteResultImporter(store, SSHClient(config.remote)).discover_remote(root=config.remote.results_root)
    payload = summary.to_dict()
    payload.update({"output": str(output), "dry_run": bool(args.dry_run)})
    _emit(payload, args.json)
    return 0


def cmd_directive(args: argparse.Namespace) -> int:
    """Append a human objective for the next Controller planning boundary."""

    output_root = Path(args.output_root).resolve()
    directive, written = submit_directive(
        ObservationStore(output_root / "observations.jsonl"),
        args.text,
        directive_id=args.directive_id,
    )
    _emit(
        {
            "status": "submitted" if written else "already_present",
            "output_root": str(output_root),
            "observation_id": "obs-%s" % directive.directive_id,
            **directive.to_dict(),
        },
        args.json,
    )
    return 0


def cmd_remote_campaign(args: argparse.Namespace) -> int:
    config = load_remote_campaign_config(Path(args.remote_config))
    target = load_target_profile(Path(args.target).resolve()) if args.target else None
    controller = build_controller_from_config(
        Path(args.controller_config),
        provider_name=args.controller_provider,
        model_name=args.controller_model,
        remote_port=args.controller_remote_port,
        timeout_s=args.controller_timeout_s,
    )
    ssh = LocalCommandClient(config.remote) if args.local_resources else None
    campaign = RemoteCampaign(
        config,
        controller=controller,
        ssh=ssh,
        target=target,
        output_root=Path(args.output_root).resolve() if args.output_root else None,
    )
    result = campaign.run_loop(
        resume=args.resume,
        max_iterations=args.max_experiments,
        split=args.split,
        resource_poll_interval_s=args.resource_poll_interval_s,
    )
    _emit(result.to_dict(), args.json)
    return 0 if result.status in {"accepted", "completed", "target_satisfied"} else 1


def _student_config(args: argparse.Namespace):
    return load_student_campaign_config(Path(args.student_config).resolve())


def cmd_student_validate(args: argparse.Namespace) -> int:
    config = _student_config(args)
    _emit(
        {
            "valid": True,
            "goal": config.goal,
            "target": config.to_dict()["target"],
            "remote_host": config.remote.host,
            "worker_entrypoint": config.worker_entrypoint,
            "evaluation_command": list(config.evaluation_command),
        },
        args.json,
    )
    return 0


def cmd_student_compile(args: argparse.Namespace) -> int:
    config = _student_config(args)
    proposal = StudentProposal.from_dict(json.loads(Path(args.proposal).read_text(encoding="utf-8")))
    manifest = StudentCompiler(config.target).compile(proposal, Path(args.output))
    _emit(manifest.to_dict(), args.json)
    return 0


def cmd_student_run(args: argparse.Namespace) -> int:
    config = _student_config(args)
    client = LocalCommandClient(config.remote) if args.local_resources else SSHClient(config.remote)
    if args.detach:
        payload = RemoteStudentSupervisor(config, client).start(args.max_rounds or config.max_rounds)
        _emit(payload, args.json)
        return 0
    provider = build_student_proposal_provider(
        config.controller.provider,
        config.controller.model,
        config.target,
        base_url=config.controller.base_url,
        timeout_s=config.controller.timeout_s,
    )
    if config.quality_backend != "clip_temporal":
        raise StudentConfigError("student run requires quality_backend=clip_temporal for the semantic verifier")
    teacher_baseline = RemoteStudentBaseline(config, client).run()
    campaign_base, capability_snapshot, review_pipeline = build_student_control_plane(
        config, provider, config.local_output_root, teacher_baseline
    )
    campaign = StudentCampaign(
        provider,
        StudentCompiler(config.target),
        RemoteStudentWorker(config, client),
        RemoteStudentEvaluator(config, client),
        goal=config.goal,
        target=config.target,
        output_root=config.local_output_root,
        experience_path=config.experience_path,
        max_failures=config.max_failures,
        min_rounds_before_success=config.min_rounds_before_success,
        retention_handler=RemoteStudentRetention(config, client).retain,
        campaign_base=campaign_base,
        capability_snapshot=capability_snapshot,
        review_pipeline=review_pipeline,
        initial_parent_checkpoint=config.teacher_checkpoint,
        fidelity_schedule=config.fidelity_schedule,
        teacher_baseline=teacher_baseline,
        quality_policy={
            "quality_floor_ratio": config.quality_floor_ratio,
            "min_reward_delta": config.min_reward_delta,
            "material_efficiency_gain": config.material_efficiency_gain,
            "max_metric_regression": config.max_metric_regression,
            "no_improvement_patience": config.no_improvement_patience,
        },
    )
    result = campaign.run(max_rounds=args.max_rounds or config.max_rounds)
    _emit(result.to_dict(), args.json)
    return 0 if result.status == "success" else 1


def cmd_student_status(args: argparse.Namespace) -> int:
    config = _student_config(args)
    client = LocalCommandClient(config.remote) if args.local_resources else SSHClient(config.remote)
    _emit(RemoteStudentSupervisor(config, client).status(), args.json)
    return 0


def cmd_controller_status(args: argparse.Namespace) -> int:
    """Read or follow the append-only Controller lifecycle stream."""
    path = Path(args.output_root).resolve() / "controller-events.jsonl"
    if not args.follow:
        events = list(ControllerEventStore(path).read())
        _emit(
            {
                "output_root": str(path.parent),
                "events_path": str(path),
                "event_count": len(events),
                "latest": events[-1] if events else None,
                "events": events,
            },
            args.json,
        )
        return 0

    offset = 0
    try:
        while True:
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for line in handle:
                        if line.strip():
                            _emit(json.loads(line), args.json)
                    offset = handle.tell()
            time.sleep(args.poll_interval_s)
    except KeyboardInterrupt:
        return 0


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
    parser = argparse.ArgumentParser(description="Device-aware research harness for reproducible MiniMax H3 optimization")
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

    validate_target = subparsers.add_parser("validate", help="validate an immutable optimization TargetProfile")
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

    a0 = subparsers.add_parser("a0-evolve", help="run the A0 model-evolution research protocol")
    a0.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    a0.add_argument("--output-root", default="var/a0-model-evolution")
    a0.add_argument("--output", help="optional copy of the complete campaign JSON")
    a0.add_argument("--report-output", help="optional copy of the research report JSON")
    a0.add_argument("--controller", choices=("mock", "ollama"), default="mock")
    a0.add_argument("--controller-model", default="qwen3.5:9b-q8_0")
    a0.add_argument("--controller-url", default="http://127.0.0.1:11434")
    a0.add_argument("--controller-timeout-s", type=float, default=180.0)
    a0.add_argument("--max-gpu-hours", type=float, default=24.0)
    a0.add_argument("--max-experiments", type=int, default=20)
    a0.add_argument("--max-failed-experiments", type=int, default=5)
    a0.add_argument("--stop-on-target", action="store_true")
    a0.add_argument("--external-operator-command", nargs="+", help="fixed argv for real model/training worker")
    a0.add_argument("--external-operator-timeout-s", type=float, default=3600.0)
    _json_flag(a0)
    a0.set_defaults(handler=cmd_a0_evolve)

    a1 = subparsers.add_parser("a1-evolve", help="run the A1 real-worker research protocol")
    a1.add_argument("--parent-checkpoint", required=True, help="local checkpoint accessible to the worker")
    a1.add_argument("--worker-command", nargs="+", required=True, help="fixed argv for h3_model_worker.py")
    a1.add_argument("--worker-config", required=True, help="trusted worker config containing trainer argv")
    a1.add_argument("--worker-timeout-s", type=float, default=7200.0)
    a1.add_argument("--config", default="configs/default.yaml")
    a1.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    a1.add_argument("--base-url", default="http://100.88.143.10:8188")
    a1.add_argument("--output-root", default="var/a1-real-evolution")
    a1.add_argument("--baseline-quality", type=float, required=True)
    a1.add_argument("--baseline-model-size-gb", type=float, required=True)
    a1.add_argument("--baseline-latency-s", type=float, required=True)
    a1.add_argument("--baseline-peak-memory-gb", type=float, required=True)
    a1.add_argument("--controller", choices=("mock", "ollama"), default="ollama")
    a1.add_argument("--controller-model", default="qwen3.5:9b-q8_0")
    a1.add_argument("--controller-url", default="http://100.88.143.10:11434")
    a1.add_argument("--controller-timeout-s", type=float, default=180.0)
    a1.add_argument("--max-experiments", type=int, default=2)
    a1.add_argument("--max-gpu-hours", type=float, default=8.0)
    a1.add_argument("--max-failed-experiments", type=int, default=2)
    _json_flag(a1)
    a1.set_defaults(handler=cmd_a1_evolve)

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
    benchmark.add_argument("--baseline-model-size-gb", type=float)
    benchmark.add_argument("--baseline-latency-s", type=float)
    benchmark.add_argument("--baseline-peak-memory-gb", type=float)
    benchmark.add_argument("--black-frame-rate-threshold", type=float, default=0.0)
    benchmark.add_argument("--primary-intervention", default="quantization")
    benchmark.add_argument("--secondary-change", action="append", default=[])
    benchmark.add_argument("--controlled-variable", action="append", default=[])
    benchmark.add_argument("--rationale", default="")
    benchmark.add_argument("--reset-backend-before-run", action="store_true", help="POST /free before running to isolate model/cache state")
    benchmark.add_argument("--sample-interval-s", type=float, default=1.0)
    benchmark.add_argument("--task-timeout-s", type=float, default=3600.0)
    _json_flag(benchmark)
    benchmark.set_defaults(handler=cmd_benchmark)

    import_experience = subparsers.add_parser("import-experience", help="read remote H3 trainer results into local experience memory")
    import_experience.add_argument("--remote-config", required=True)
    import_experience.add_argument("--output", required=True)
    import_experience.add_argument("--dry-run", action="store_true", help="read and normalize without writing the JSONL store")
    _json_flag(import_experience)
    import_experience.set_defaults(handler=cmd_import_experience)

    directive = subparsers.add_parser(
        "directive",
        help="submit a bounded human objective for the next Controller plan",
    )
    directive.add_argument("--output-root", required=True, help="campaign output root containing observations.jsonl")
    directive.add_argument("--text", required=True, help="next-round optimization objective")
    directive.add_argument("--directive-id", help="stable ID for retries or intentionally repeated objectives")
    _json_flag(directive)
    directive.set_defaults(handler=cmd_directive)

    remote_campaign = subparsers.add_parser("remote-campaign", help="run repeated Controller-driven SSH H3 optimization loops")
    remote_campaign.add_argument("--remote-config", required=True)
    remote_campaign.add_argument("--controller-config", default="configs/controller.yaml")
    remote_campaign.add_argument("--target")
    remote_campaign.add_argument("--output-root")
    remote_campaign.add_argument(
        "--max-iterations",
        "--max-experiments",
        dest="max_experiments",
        type=int,
        default=1,
        help="maximum Controller -> operator -> evaluation loop iterations",
    )
    remote_campaign.add_argument("--split", choices=("sanity", "dev", "heldout", "all"), default=None)
    remote_campaign.add_argument(
        "--controller-provider",
        choices=("vllm", "rulebased"),
        default="vllm",
        help="Controller backend; vllm is the default and never falls back when unavailable",
    )
    remote_campaign.add_argument("--controller-model", default=None)
    remote_campaign.add_argument("--controller-remote-port", type=int, default=None)
    remote_campaign.add_argument("--controller-timeout-s", type=float, default=None)
    remote_campaign.add_argument(
        "--resource-poll-interval-s",
        type=float,
        default=5.0,
        help="seconds between retries of a Controller plan waiting for GPUs",
    )
    remote_campaign.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    remote_campaign.add_argument(
        "--local-resources",
        "--on-server",
        action="store_true",
        help="run trusted commands and loopback services directly on this server (no nested SSH)",
    )
    _json_flag(remote_campaign)
    remote_campaign.set_defaults(handler=cmd_remote_campaign)

    student_campaign = subparsers.add_parser(
        "student-campaign",
        help="autonomously propose, compile, train, evaluate, and revise a 1B-2B Student",
    )
    student_campaign.add_argument("--student-config", default="configs/student-campaign.example.yaml")
    student_subparsers = student_campaign.add_subparsers(dest="student_command", required=True)

    student_validate = student_subparsers.add_parser("validate", help="validate Student campaign and SSH trust boundaries")
    _json_flag(student_validate)
    student_validate.set_defaults(handler=cmd_student_validate)

    student_compile = student_subparsers.add_parser("compile", help="compile one JSON Student proposal on meta tensors")
    student_compile.add_argument("--proposal", required=True)
    student_compile.add_argument("--output", required=True)
    _json_flag(student_compile)
    student_compile.set_defaults(handler=cmd_student_compile)

    student_run = student_subparsers.add_parser("run", help="run the autonomous Student campaign through SSH")
    student_run.add_argument("--max-rounds", type=int)
    student_run.add_argument("--detach", action="store_true", help="start the remote campaign and return immediately")
    student_run.add_argument("--local-resources", action="store_true", help="execute trusted worker commands on this host")
    _json_flag(student_run)
    student_run.set_defaults(handler=cmd_student_run)

    student_status = student_subparsers.add_parser("status", help="read one remote detached Student supervisor snapshot")
    student_status.add_argument("--local-resources", action="store_true")
    _json_flag(student_status)
    student_status.set_defaults(handler=cmd_student_status)

    controller_status = subparsers.add_parser("controller-status", help="show or follow the Controller event stream")
    controller_status.add_argument("--output-root", default="var/remote-h3")
    controller_status.add_argument("--follow", action="store_true", help="follow new Controller events until interrupted")
    controller_status.add_argument("--poll-interval-s", type=float, default=1.0)
    _json_flag(controller_status)
    controller_status.set_defaults(handler=cmd_controller_status)

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
    except (Harness4H3Error, RemoteError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc), "type": exc.__class__.__name__}, ensure_ascii=False))
        else:
            print("harness4h3: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
