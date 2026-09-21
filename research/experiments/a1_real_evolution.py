from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.archive.model_store import ModelStore
from harness4h3.archive.pareto import ParetoArchive
from harness4h3.archive.system_store import SystemCandidateStore
from harness4h3 import HARNESS_VERSION
from harness4h3.backends.comfyui import MiniMaxH3Adapter
from harness4h3.benchmark.h3 import BenchmarkSummary, H3BenchmarkRunner
from harness4h3.config import load_config
from harness4h3.controller.provider import OllamaStructuredController
from harness4h3.controller.loop import OptimizationLoop
from harness4h3.controller.schemas import BudgetState, EvaluationResult, HardwareMetrics, ExperimentPlan
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.h3.inspector import H3Inspector
from harness4h3.harness.loop import load_workflow
from harness4h3.harness.state import Task, load_tasks
from harness4h3.operators.base import OperatorRegistry
from harness4h3.operators.model_evolution import build_external_model_evolution_registry
from harness4h3.memory.experiment_store import ExperimentStore
from harness4h3.target.profile import TargetProfile, load_target_profile

from research.experiments.a0_model_evolution import (
    A0Budget,
    A0CampaignResult,
    A0RuleBasedController,
    run_campaign,
)


class TieredRealBenchmarkEvaluator:
    """Adapt the real H3 benchmark to the evaluator interface used by A0.

    A candidate is measured on one progressively stronger task set per call:
    Tier 1 uses a cheap dev sample, Tier 2 uses all dev tasks, and Tier 3 uses
    held-out tasks. The call count is keyed by immutable child model id, so the
    Controller cannot accidentally reuse an earlier candidate's measurement.
    """

    def __init__(
        self,
        benchmark: H3BenchmarkRunner,
        tasks_by_tier: Mapping[int, Sequence[Task]],
        target: TargetProfile,
        baseline_quality: float,
        baseline_hardware: HardwareMetrics,
        force_full_fidelity: bool = False,
    ):
        self.benchmark = benchmark
        self.tasks_by_tier = {int(key): tuple(value) for key, value in tasks_by_tier.items()}
        self.target = target
        self.baseline_quality = float(baseline_quality)
        self.baseline_hardware = baseline_hardware
        self.force_full_fidelity = bool(force_full_fidelity)
        self.calls: Dict[str, int] = {}
        self.summaries: Dict[Tuple[str, Optional[str], int], BenchmarkSummary] = {}

    def evaluate(
        self,
        state,
        target: TargetProfile,
        baseline_quality: float,
        system: Optional[SystemCandidate] = None,
        device_id: Optional[str] = None,
        task_split: Optional[str] = None,
        benchmark_recipe: Optional[Mapping[str, Any]] = None,
    ) -> EvaluationResult:
        level = 3 if self.force_full_fidelity else min(self.calls.get(state.model_id, 0) + 1, 3)
        self.calls[state.model_id] = level
        tasks = self.tasks_by_tier.get(level) or self.tasks_by_tier.get(3) or self.tasks_by_tier.get(2) or ()
        if not tasks:
            raise ValueError("real A1 evaluator has no tasks for tier %d" % level)
        summary = self.benchmark.run(
            state,
            tasks,
            baseline_quality=self.baseline_quality,
            target=target,
            baseline_hardware=self.baseline_hardware,
            reset_backend_before_run=True,
            system=system,
            device_id=device_id,
            task_split=task_split,
            benchmark_recipe=benchmark_recipe,
        )
        self.summaries[(state.model_id, summary.system_id, level)] = summary
        quality_score = float(summary.quality_score) if summary.quality_score is not None else float("nan")
        violations = list(summary.violations)
        if summary.accepted is False:
            violations.extend(
                "hard_gate:%s" % key
                for key, value in summary.hard_gates.items()
                if value is False
            )
        critical = any(run.critical_regression for run in summary.runs) or not summary.runs
        metrics = {
            **dict(summary.quality_metrics),
            "real_benchmark": True,
            "offline_simulation": False,
            "fidelity_tier": level,
            "benchmark_summary": summary.to_dict(),
        }
        # The real benchmark is authoritative. ``accepted`` includes the
        # semantic/decode/quality/efficiency gates when baseline metrics exist.
        feasible = bool(summary.feasible is True and summary.accepted is True and not critical)
        validity = {
            "benchmark_ran": bool(summary.runs),
            "generation_valid": bool(summary.hard_gates.get("generation_valid", False)),
            "decode_success": bool(summary.hard_gates.get("decode_success", False)),
            "semantic_generation_valid": all(run.semantic_generation_valid for run in summary.runs),
        }
        provenance = {
            "evaluator": self.__class__.__name__,
            "evaluator_version": summary.evaluator_version,
            "benchmark_summary": summary.to_dict(),
            "benchmark_recipe": dict(summary.benchmark_recipe),
            "real_benchmark": True,
            "offline_simulation": False,
        }
        return EvaluationResult(
            quality_score=quality_score,
            quality_metrics=metrics,
            hardware=summary.hardware,
            feasible=feasible,
            violations=violations,
            critical_regression=critical,
            model_id=state.model_id,
            system_id=summary.system_id,
            device_id=summary.device_id,
            task_split=summary.task_split,
            validity=validity,
            provenance=provenance,
            search_score=target.objective_score(
                {
                    "quality_score": quality_score,
                    "latency_s": summary.hardware.latency_s,
                    "peak_memory_gb": summary.hardware.peak_memory_gb,
                    "model_size_gb": summary.hardware.model_size_gb,
                    "energy_j": summary.hardware.energy_j,
                }
            ),
        )


class A1BootstrapController:
    """Force A1's first real child through the complete fidelity ladder."""

    provider_name = "a1-bootstrap"

    def __init__(self, inner: Any):
        self.inner = inner
        self.model_name = getattr(inner, "model_name", "unknown")

    def plan(self, context):
        plan = self.inner.plan(context)
        required_budget = dict(plan.required_budget)
        required_budget["tier"] = 3
        return replace(
            plan,
            required_budget=required_budget,
            parent_system_id=str(context.current_system.get("id")) if context.current_system.get("id") else plan.parent_system_id,
        )


class A1RealBootstrapController(A1BootstrapController):
    """Force the first active real-H3 plan to be recovery fine-tuning."""

    provider_name = "a1-real-bootstrap"

    def plan(self, context):
        plan = super().plan(context)
        if context.budget_state.used_iterations != 0:
            # P0 deliberately exposes only the one real operator.  The second
            # Controller turn must still be a valid real-H3 plan after E0001;
            # do not let the generic offline search policy invent a structural
            # or quantization operator before the first closed loop is proven.
            return replace(
                plan,
                operator="recovery_finetune",
                operator_args={"training_steps": 1},
                objective="repeat one bounded real-H3 recovery update after independent evaluation",
                expected_effects={"quality_score": "measure independently after training"},
                required_budget={"wall_time_s": 3600.0, "gpu_hours": 2.0, "controller_calls": 0, "tier": 3},
                stop_conditions={"critical_regression": False, "target_satisfied": False, "budget_exhausted": True},
                repeat_for_statistics=True,
                rationale="P0 permits only recovery_finetune until the first real H3 trajectory has completed",
            )
        return replace(
            plan,
            operator="recovery_finetune",
            operator_args={"training_steps": 1},
            objective="perform one authentic MiniMax-H3 recovery update",
            diagnosis="establish the real H3 training path before broader search",
            hypothesis="one bounded recovery step will produce a verifiable child checkpoint without changing runtime configuration",
            expected_effects={"quality_score": "measure independently after training"},
            stop_conditions={"critical_regression": False, "target_satisfied": False, "budget_exhausted": True},
            required_budget={"wall_time_s": 3600.0, "gpu_hours": 2.0, "controller_calls": 0, "tier": 3},
            rationale="A1-T0 must prove forward, backward, optimizer, save, reload, and independent benchmark evidence before other operators are considered",
        )


def _hardware_metrics(values: Mapping[str, Optional[float]]) -> HardwareMetrics:
    required = {key: values.get(key) for key in ("model_size_gb", "latency_s", "peak_memory_gb")}
    if any(value is None for value in required.values()):
        raise ValueError("A1 requires baseline model_size_gb, latency_s, and peak_memory_gb")
    return HardwareMetrics(
        model_size_gb=float(required["model_size_gb"]),
        latency_s=float(required["latency_s"]),
        peak_memory_gb=float(required["peak_memory_gb"]),
    )


def build_real_evaluator(
    config_path: Path,
    base_url: str,
    benchmark_output: Path,
    target: TargetProfile,
    baseline_quality: float,
    baseline_hardware: HardwareMetrics,
    request_timeout_s: float = 120.0,
    task_timeout_s: float = 3600.0,
    sample_interval_s: float = 1.0,
    force_full_fidelity: bool = False,
) -> TieredRealBenchmarkEvaluator:
    config = load_config(config_path)
    tasks = load_tasks(config.runtime.tasks_path)
    dev = [task for task in tasks if task.split == "dev"]
    heldout = [task for task in tasks if task.split == "heldout"]
    if not dev or not heldout:
        raise ValueError("A1 real evaluation requires both dev and heldout tasks")
    backend = MiniMaxH3Adapter(
        base_url=base_url.rstrip("/"),
        request_timeout_s=request_timeout_s,
        poll_interval_s=config.backend.poll_interval_s,
        task_timeout_s=task_timeout_s,
    )
    benchmark = H3BenchmarkRunner(
        backend,
        SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s),
        load_workflow(config.workflow.template),
        config.workflow,
        benchmark_output,
        system_sample_interval_s=sample_interval_s,
    )
    return TieredRealBenchmarkEvaluator(
        benchmark,
        {1: dev[:1], 2: dev, 3: heldout},
        target,
        baseline_quality,
        baseline_hardware,
        force_full_fidelity=force_full_fidelity,
    )


def run_a1(
    parent_checkpoint: Path,
    worker_command: Sequence[str],
    config_path: Path,
    target: TargetProfile,
    output_root: Path,
    base_url: str,
    baseline_quality: float,
    baseline_hardware: HardwareMetrics,
    controller: Any,
    max_experiments: int = 2,
    max_gpu_hours: float = 8.0,
    max_failed_experiments: int = 2,
    worker_timeout_s: float = 7200.0,
) -> A0CampaignResult:
    parent_checkpoint = Path(parent_checkpoint).resolve()
    if not parent_checkpoint.is_file():
        raise ValueError("A1 parent checkpoint does not exist: %s" % parent_checkpoint)
    state = H3Inspector().inspect(parent_checkpoint, model_id="M0000", sampling_steps=20)
    state = replace(
        state,
        measured_metrics={
            **dict(state.measured_metrics),
            "quality_score": float(baseline_quality),
            "latency_s": baseline_hardware.latency_s,
            "peak_memory_gb": baseline_hardware.peak_memory_gb,
            "model_size_gb": baseline_hardware.model_size_gb,
        },
        provenance={**dict(state.provenance), "a1_parent": True, "offline_simulation": False},
    )
    initial = ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")
    registry = build_external_model_evolution_registry(
        worker_command,
        timeout_s=worker_timeout_s,
        allowed_operators=("recovery_finetune",),
    )
    evaluator = build_real_evaluator(
        config_path,
        base_url,
        output_root / "benchmark",
        target,
        baseline_quality,
        baseline_hardware,
    )
    return run_campaign(
        target,
        A1BootstrapController(controller),
        registry,
        evaluator,
        output_root,
        initial_candidate=initial,
        budget=A0Budget(max_gpu_hours, max_experiments, max_failed_experiments),
        stop_on_target=False,
        offline_simulation=False,
    )


def run_a1_active(
    parent_checkpoint: Path,
    worker_command: Sequence[str],
    config_path: Path,
    target: TargetProfile,
    output_root: Path,
    base_url: str,
    baseline_quality: float,
    baseline_hardware: HardwareMetrics,
    controller: Any,
    max_experiments: int = 2,
    max_gpu_hours: float = 8.0,
    max_failed_experiments: int = 2,
    worker_timeout_s: float = 7200.0,
) -> A0CampaignResult:
    """Run A1 with the pair-aware Harness loop and canonical evidence records."""

    parent_checkpoint = Path(parent_checkpoint).resolve()
    output_root = Path(output_root).resolve()
    if not parent_checkpoint.is_file():
        raise ValueError("A1 parent checkpoint does not exist: %s" % parent_checkpoint)
    state = H3Inspector().inspect(parent_checkpoint, model_id="M0000", sampling_steps=20)
    state = replace(
        state,
        measured_metrics={
            **dict(state.measured_metrics),
            "quality_score": float(baseline_quality),
            "latency_s": baseline_hardware.latency_s,
            "peak_memory_gb": baseline_hardware.peak_memory_gb,
            "model_size_gb": baseline_hardware.model_size_gb,
        },
        provenance={
            **dict(state.provenance),
            "a1_parent": True,
            "real_h3": True,
            "offline_simulation": False,
        },
    )
    initial = ModelCandidate("M0000", None, 0, state.checkpoint_path, state, None, "baseline")
    models = ModelStore(output_root / "models")
    models.initialize(initial)
    systems = SystemCandidateStore(output_root / "systems", model_store=models)
    baseline_system = SystemCandidate.from_model_candidate(
        "S0000",
        initial,
        runtime_state={
            **dict(state.runtime_state),
            "backend": "comfyui",
            "device_id": target.hardware_name,
            "benchmark_recipe": {"config": str(Path(config_path).resolve()), "split": "heldout"},
        },
        status="baseline",
        metadata={"real_h3": True, "offline_simulation": False},
    )
    systems.initialize(baseline_system)
    registry = build_external_model_evolution_registry(
        worker_command,
        timeout_s=worker_timeout_s,
        allowed_operators=("recovery_finetune",),
    )
    evaluator = build_real_evaluator(
        config_path,
        base_url,
        output_root / "benchmark",
        target,
        baseline_quality,
        baseline_hardware,
        force_full_fidelity=True,
    )
    experiments = ExperimentStore(output_root / "experiments.jsonl")
    loop = OptimizationLoop(
        controller=A1RealBootstrapController(controller),
        operators=registry,
        evaluator=evaluator,
        models=models,
        pareto=ParetoArchive(output_root / "pareto"),
        experiments=experiments,
        run_root=output_root / "runs",
        checkpoint_path=output_root / "session.json",
        systems=systems,
        require_real_evidence=True,
    )
    result = loop.run(
        "A1-real",
        target,
        BudgetState(
            max_iterations=max_experiments,
            max_failed_experiments=max_failed_experiments,
            max_gpu_hours=max_gpu_hours,
            max_controller_calls=max_experiments,
        ),
        initial,
    )
    records = [record.to_dict() for record in experiments.read()]
    report = {
        "campaign_id": "A1",
        "harness_version": HARNESS_VERSION,
        "status": result.status,
        "target_satisfied": result.status == "target_satisfied",
        "offline_simulation": False,
        "total_experiments": len(records),
        "failed_experiment_count": sum(1 for item in records if item.get("failure_type")),
        "rejected_candidate_count": sum(1 for item in records if item.get("decision", {}).get("status") == "reject"),
        "accepted_experiments": [item for item in records if item.get("decision", {}).get("keep")],
        "full_autonomous_experiment_sequence": records,
        "model_lineage": [candidate.to_dict() for candidate in models.lineage()],
        "system_lineage": [candidate.to_dict() for candidate in systems.lineage()],
        "pareto_front": [entry.to_dict() for entry in loop.pareto.front()],
        "duplicate_fingerprints_blocked": sum(1 for item in records if item.get("failure_type") == "duplicate_experiment"),
        "second_controller_plan": (
            json.loads((output_root / "runs" / "exp_0002" / "plan.json").read_text(encoding="utf-8"))
            if (output_root / "runs" / "exp_0002" / "plan.json").is_file()
            else None
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "campaign.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return A0CampaignResult(result.status, report, report)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="A1 first real H3 model-evolution campaign")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--worker-command", nargs="+", required=True)
    parser.add_argument("--worker-timeout-s", type=float, default=7200.0)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target", default="configs/targets/rtx5080_example.yaml")
    parser.add_argument("--base-url", default="http://100.88.143.10:8188")
    parser.add_argument("--output-root", default="var/a1-real-evolution")
    parser.add_argument("--baseline-quality", type=float, required=True)
    parser.add_argument("--baseline-model-size-gb", type=float, required=True)
    parser.add_argument("--baseline-latency-s", type=float, required=True)
    parser.add_argument("--baseline-peak-memory-gb", type=float, required=True)
    parser.add_argument("--controller", choices=("mock", "ollama"), default="ollama")
    parser.add_argument("--controller-model", default="qwen3.5:9b-q8_0")
    parser.add_argument("--controller-url", default="http://100.88.143.10:11434")
    parser.add_argument("--controller-timeout-s", type=float, default=180.0)
    parser.add_argument("--max-experiments", type=int, default=2)
    parser.add_argument("--max-gpu-hours", type=float, default=8.0)
    parser.add_argument("--max-failed-experiments", type=int, default=2)
    args = parser.parse_args()
    target = load_target_profile(Path(args.target).resolve())
    if args.controller == "mock":
        controller = A0RuleBasedController()
    else:
        controller = OllamaStructuredController(args.controller_model, args.controller_url, timeout_s=args.controller_timeout_s)
    result = run_a1(
        Path(args.parent_checkpoint),
        args.worker_command,
        Path(args.config).resolve(),
        target,
        Path(args.output_root).resolve(),
        args.base_url,
        args.baseline_quality,
        _hardware_metrics(
            {
                "model_size_gb": args.baseline_model_size_gb,
                "latency_s": args.baseline_latency_s,
                "peak_memory_gb": args.baseline_peak_memory_gb,
            }
        ),
        controller,
        args.max_experiments,
        args.max_gpu_hours,
        args.max_failed_experiments,
        args.worker_timeout_s,
    )
    print(json.dumps({"status": result.status, "output_root": str(Path(args.output_root).resolve()), "experiments": result.report["total_experiments"]}, ensure_ascii=False))
    return 0 if result.status in {"completed", "target_satisfied"} else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
