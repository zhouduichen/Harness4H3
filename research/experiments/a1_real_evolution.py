from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.backends.comfyui import MiniMaxH3Adapter
from harness4h3.benchmark.h3 import BenchmarkSummary, H3BenchmarkRunner
from harness4h3.config import load_config
from harness4h3.controller.provider import OllamaStructuredController
from harness4h3.controller.schemas import BudgetState, EvaluationResult, HardwareMetrics, ExperimentPlan
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.h3.inspector import H3Inspector
from harness4h3.harness.loop import load_workflow
from harness4h3.harness.state import Task, load_tasks
from harness4h3.operators.base import OperatorRegistry
from harness4h3.operators.model_evolution import build_external_model_evolution_registry
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
    ):
        self.benchmark = benchmark
        self.tasks_by_tier = {int(key): tuple(value) for key, value in tasks_by_tier.items()}
        self.target = target
        self.baseline_quality = float(baseline_quality)
        self.baseline_hardware = baseline_hardware
        self.calls: Dict[str, int] = {}
        self.summaries: Dict[Tuple[str, int], BenchmarkSummary] = {}

    def evaluate(self, state, target: TargetProfile, baseline_quality: float) -> EvaluationResult:
        level = min(self.calls.get(state.model_id, 0) + 1, 3)
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
        )
        self.summaries[(state.model_id, level)] = summary
        quality_score = float(summary.quality_score) if summary.quality_score is not None else 0.0
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
        return EvaluationResult(
            quality_score=quality_score,
            quality_metrics=metrics,
            hardware=summary.hardware,
            feasible=feasible,
            violations=violations,
            critical_regression=critical,
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
        return replace(plan, required_budget=required_budget)


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
    registry = build_external_model_evolution_registry(worker_command, timeout_s=worker_timeout_s)
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
