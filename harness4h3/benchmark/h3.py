from __future__ import annotations

import copy
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..backends.comfyui import BackendError, MiniMaxH3Adapter
from ..config import WorkflowConfig
from ..controller.schemas import HardwareMetrics
from ..evaluator.evaluator import EvaluationResult, EvaluatorError, SubprocessEvaluator, make_request
from ..h3.state import ModelState
from ..harness.context import _render_prompt, _set_target
from ..harness.state import Task
from .runtime_policy import apply_runtime_policy


@dataclass(frozen=True)
class BenchmarkTaskResult:
    task_id: str
    status: str
    prompt_id: Optional[str]
    artifacts: Tuple[str, ...]
    wall_time_s: float
    quality_score: Optional[float]
    quality_metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    message: str = ""
    operator_execution_success: bool = False
    artifact_generation_success: bool = False
    semantic_generation_valid: bool = False
    critical_regression: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    model_id: str
    task_count: int
    quality_score: Optional[float]
    quality_metrics: Mapping[str, Any]
    hardware: HardwareMetrics
    feasible: Optional[bool]
    violations: Tuple[str, ...]
    runs: Tuple[BenchmarkTaskResult, ...]
    hardware_samples: Tuple[Mapping[str, Any], ...] = ()
    operator_attribution: Mapping[str, Any] = field(default_factory=dict)
    efficiency_improvements: Mapping[str, Any] = field(default_factory=dict)
    hard_gates: Mapping[str, Any] = field(default_factory=dict)
    accepted: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _checkpoint_name(value: str) -> str:
    return PureWindowsPath(value).name if "\\" in value else Path(value).name


class _SystemSampler:
    def __init__(self, endpoint: str, interval_s: float = 1.0):
        self.endpoint = endpoint.rstrip("/")
        self.interval_s = interval_s
        self.samples: List[Mapping[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> Optional[Mapping[str, Any]]:
        try:
            request = urllib.request.Request(self.endpoint + "/system_stats", headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, Mapping) else None
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError, json.JSONDecodeError):
            return None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)

    def __enter__(self) -> "_SystemSampler":
        initial = self._sample()
        if initial is not None:
            self.samples.append(initial)
        self._thread = threading.Thread(target=self._run, name="harness4h3-system-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s + 1.0))
        final = self._sample()
        if final is not None:
            self.samples.append(final)


def _metric_values(samples: Sequence[Mapping[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    peak_ram = None
    peak_vram = None
    for sample in samples:
        system = sample.get("system") if isinstance(sample, Mapping) else None
        if isinstance(system, Mapping):
            total = system.get("ram_total")
            free = system.get("ram_free")
            if isinstance(total, (int, float)) and isinstance(free, (int, float)) and total >= free:
                peak_ram = max(peak_ram or 0.0, (float(total) - float(free)) / 1_000_000_000)
        devices = sample.get("devices") if isinstance(sample, Mapping) else None
        if isinstance(devices, list):
            for device in devices:
                if not isinstance(device, Mapping):
                    continue
                total = device.get("vram_total")
                free = device.get("vram_free")
                if isinstance(total, (int, float)) and isinstance(free, (int, float)) and total >= free:
                    peak_vram = max(peak_vram or 0.0, (float(total) - float(free)) / 1_000_000_000)
    return peak_ram, peak_vram


class H3BenchmarkRunner:
    """Run fixed H3 tasks and return quality/hardware evidence for one ModelState."""

    def __init__(
        self,
        backend: MiniMaxH3Adapter,
        evaluator: SubprocessEvaluator,
        workflow_template: Mapping[str, Any],
        workflow_config: WorkflowConfig,
        output_root: Path,
        system_sample_interval_s: float = 1.0,
    ):
        self.backend = backend
        self.evaluator = evaluator
        self.workflow_template = copy.deepcopy(dict(workflow_template))
        self.workflow_config = workflow_config
        self.output_root = Path(output_root)
        self.system_sample_interval_s = float(system_sample_interval_s)

    def _workflow(self, state: ModelState, task: Task) -> Mapping[str, Any]:
        workflow = copy.deepcopy(self.workflow_template)
        policy = {"prompt": {"prefix": "", "suffix": ""}, "context": {"include_constraints": True, "max_prompt_chars": 4000}}
        prompt = _render_prompt(task, policy)
        _set_target(workflow, self.workflow_config.prompt_target.node_id, self.workflow_config.prompt_target.input_name, prompt)
        if self.workflow_config.seed_target is not None:
            _set_target(workflow, self.workflow_config.seed_target.node_id, self.workflow_config.seed_target.input_name, task.seed)
        if state.sampling_steps is not None and "steps" in self.workflow_config.mutable:
            target = self.workflow_config.mutable["steps"]
            _set_target(workflow, target.node_id, target.input_name, state.sampling_steps)
        checkpoint_name = _checkpoint_name(state.checkpoint_path)
        model_node_id = "127"
        model_node = workflow.get(model_node_id)
        if not isinstance(model_node, Mapping) or str(model_node.get("class_type", "")).lower() not in {"unetloader", "unetloadergguf"}:
            model_node = None
            for candidate_id, candidate in workflow.items():
                if isinstance(candidate, Mapping) and str(candidate.get("class_type", "")).lower() in {"unetloader", "unetloadergguf"}:
                    model_node_id, model_node = str(candidate_id), candidate
                    break
        if isinstance(model_node, Mapping) and checkpoint_name.lower().endswith((".gguf", ".safetensors", ".ckpt", ".pt")):
            model_node["class_type"] = "UnetLoaderGGUF" if checkpoint_name.lower().endswith(".gguf") else "UNETLoader"
            inputs = model_node.setdefault("inputs", {})
            inputs["unet_name"] = checkpoint_name
            workflow[model_node_id] = model_node
        runtime_state = state.runtime_state if isinstance(state.runtime_state, Mapping) else {}
        runtime_recipe = runtime_state.get("runtime_recipe")
        if isinstance(runtime_recipe, (list, tuple)):
            for runtime_policy in runtime_recipe:
                apply_runtime_policy(workflow, runtime_policy)
        else:
            runtime_policy = runtime_state.get("runtime_policy")
            if runtime_policy is not None:
                apply_runtime_policy(workflow, runtime_policy)
        return workflow

    @staticmethod
    def _requires_task_cache_release(state: ModelState) -> bool:
        runtime_state = state.runtime_state if isinstance(state.runtime_state, Mapping) else {}
        policies = runtime_state.get("runtime_recipe")
        if not isinstance(policies, (list, tuple)):
            policies = [runtime_state.get("runtime_policy")]
        for policy in policies:
            if not isinstance(policy, Mapping):
                continue
            kind = str(policy.get("kind", ""))
            args = policy.get("args") if isinstance(policy.get("args"), Mapping) else {}
            if kind == "cache_release":
                return True
            if kind == "component_lifecycle_optimize" and args.get("free_cache_before_decode") is True:
                return True
        return False

    def run(
        self,
        state: ModelState,
        tasks: Sequence[Task],
        baseline_quality: Optional[float] = None,
        target: Any = None,
        operator_attribution: Optional[Mapping[str, Any]] = None,
        baseline_hardware: Optional[HardwareMetrics] = None,
        efficiency_thresholds: Optional[Mapping[str, float]] = None,
        black_frame_rate_threshold: float = 0.0,
        reset_backend_before_run: bool = False,
        power_sampler: Any = None,
    ) -> BenchmarkSummary:
        if not 0.0 <= float(black_frame_rate_threshold) <= 1.0:
            raise ValueError("black_frame_rate_threshold must be between 0 and 1")
        runs: List[BenchmarkTaskResult] = []
        release_each_task = self._requires_task_cache_release(state)
        reset = getattr(self.backend, "free", None)
        if release_each_task and not callable(reset):
            raise ValueError("cache-release runtime policy requires a backend.free() method")
        if reset_backend_before_run:
            if not callable(reset):
                raise ValueError("reset_backend_before_run requires a backend.free() method")
            reset()
        with _SystemSampler(self.backend.base_url, self.system_sample_interval_s) as sampler:
            if power_sampler is not None:
                power_sampler.start()
            try:
                for task in tasks:
                    started = time.monotonic()
                    try:
                        if release_each_task:
                            reset()
                        result = self.backend.run(self._workflow(state, task), self.output_root / state.model_id / task.id)
                        artifacts = tuple(str(path.resolve()) for path in result.artifacts)
                        evaluation: EvaluationResult = self.evaluator.evaluate(
                            make_request(task.id, task.expected, artifacts, result.wall_time_s, backend_success=True)
                        )
                        metrics = dict(evaluation.metrics)
                        failure_type = evaluation.failure_type
                        if failure_type == "low_luma" and metrics.get("all_black") in (1, 1.0, True):
                            failure_type = "degenerate_output_black_frame"
                        operator_success = bool(metrics.get("operator_execution_success", True))
                        artifact_success = bool(metrics.get("artifact_generation_success", bool(artifacts)))
                        semantic_valid = bool(
                            metrics.get(
                                "semantic_generation_valid",
                                failure_type not in {"decode_failed", "degenerate_output_black_frame", "low_luma", "output_mismatch"},
                            )
                        )
                        metrics.setdefault("operator_execution_success", 1.0 if operator_success else 0.0)
                        metrics.setdefault("artifact_generation_success", 1.0 if artifact_success else 0.0)
                        metrics.setdefault("semantic_generation_valid", 1.0 if semantic_valid else 0.0)
                        runs.append(
                            BenchmarkTaskResult(
                                task_id=task.id,
                                status="success",
                                prompt_id=result.prompt_id,
                                artifacts=artifacts,
                                wall_time_s=result.wall_time_s,
                                quality_score=evaluation.score,
                                quality_metrics=metrics,
                                failure_type=failure_type,
                                operator_execution_success=operator_success,
                                artifact_generation_success=artifact_success,
                                semantic_generation_valid=semantic_valid,
                                critical_regression=evaluation.critical_regression,
                            )
                        )
                    except (BackendError, EvaluatorError, OSError, ValueError) as exc:
                        failure_type = getattr(exc, "failure_type", "benchmark_failure")
                        runs.append(
                            BenchmarkTaskResult(
                                task_id=task.id,
                                status="failed",
                                prompt_id=None,
                                artifacts=(),
                                wall_time_s=time.monotonic() - started,
                                quality_score=None,
                                quality_metrics={
                                    "operator_execution_success": 0.0,
                                    "artifact_generation_success": 0.0,
                                    "semantic_generation_valid": 0.0,
                                },
                                failure_type=failure_type,
                                message=str(exc),
                            )
                        )
            finally:
                if power_sampler is not None:
                    power_sampler.stop()
        successful = [run for run in runs if run.quality_score is not None]
        quality_score = mean([float(run.quality_score) for run in successful]) if successful else None
        quality_metrics = {
            "backend": "comfyui",
            "successful_tasks": len(successful),
            "failed_tasks": len(runs) - len(successful),
            "task_scores": {run.task_id: run.quality_score for run in runs},
            "operator_execution_success_tasks": sum(1 for run in runs if run.operator_execution_success),
            "artifact_generation_success_tasks": sum(1 for run in runs if run.artifact_generation_success),
            "semantic_generation_valid_tasks": sum(1 for run in runs if run.semantic_generation_valid),
            "failure_types": {run.task_id: run.failure_type for run in runs if run.failure_type},
        }
        latencies = [run.wall_time_s for run in successful]
        measured = state.measured_metrics
        model_size = measured.get("model_size_gb")
        if model_size is None and state.provenance.get("size_bytes") is not None:
            model_size = float(state.provenance["size_bytes"]) / 1_000_000_000
        peak_ram, peak_vram = _metric_values(sampler.samples)
        power_summary = power_sampler.summary() if power_sampler is not None else {}
        sampled_energy = power_summary.get("energy_j") if isinstance(power_summary, Mapping) else None
        measured_energy = measured.get("energy_j")
        if power_sampler is not None:
            quality_metrics["power_sampling"] = dict(power_summary)
        hardware = HardwareMetrics(
            latency_s=mean(latencies) if latencies else None,
            peak_memory_gb=peak_vram if peak_vram is not None else peak_ram,
            model_size_gb=float(model_size) if model_size is not None else None,
            energy_j=float(sampled_energy) if sampled_energy is not None else (float(measured_energy) if measured_energy is not None else None),
            throughput=(1.0 / mean(latencies)) if latencies and mean(latencies) > 0 else None,
            thermal=None,
        )
        feasible: Optional[bool] = None
        violations: Tuple[str, ...] = ()
        if target is not None and quality_score is not None and baseline_quality is not None:
            from ..evaluator.constraints import ConstraintEvaluator

            feasible_value, violation_list, _ = ConstraintEvaluator().evaluate(
                quality_score, hardware, target, float(baseline_quality)
            )
            feasible, violations = feasible_value, tuple(violation_list)

        attribution = dict(operator_attribution or {})
        attribution.setdefault("primary_intervention", "unspecified")
        attribution.setdefault("secondary_changes", [])
        attribution.setdefault("controlled_variables", [])

        thresholds = {
            "model_size_gb": 0.20,
            "latency_s": 0.15,
            "peak_memory_gb": 0.15,
        }
        if efficiency_thresholds:
            thresholds.update({str(key): float(value) for key, value in efficiency_thresholds.items()})
        if any(value < 0.0 or value > 1.0 for value in thresholds.values()):
            raise ValueError("efficiency thresholds must be between 0 and 1")
        efficiency_improvements: Dict[str, Any] = {}
        for name, threshold in thresholds.items():
            before = getattr(baseline_hardware, name, None) if baseline_hardware is not None else None
            after = getattr(hardware, name, None)
            reduction = None
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and float(before) > 0:
                reduction = (float(before) - float(after)) / float(before)
            efficiency_improvements[name] = {
                "baseline": before,
                "candidate": after,
                "reduction": reduction,
                "threshold": threshold,
                "passed": bool(reduction is not None and reduction >= threshold),
            }
        efficiency_gate = None if baseline_hardware is None else any(item["passed"] for item in efficiency_improvements.values())
        black_rates = []
        for run in runs:
            if run.quality_score is None:
                continue
            value = run.quality_metrics.get("black_frame_ratio", 0.0)
            if isinstance(value, (int, float)):
                black_rates.append(float(value))
        max_black_rate = max(black_rates) if black_rates else None
        generation_valid = bool(runs) and all(run.semantic_generation_valid for run in runs)
        decode_success = bool(successful) and all(float(run.quality_metrics.get("decodable", 0.0)) == 1.0 for run in successful)
        quality_drop = None if baseline_quality is None or quality_score is None else float(baseline_quality) - float(quality_score)
        quality_limit = getattr(target, "max_quality_drop", None) if target is not None else None
        quality_gate = None if quality_drop is None or quality_limit is None else quality_drop <= float(quality_limit)
        temporal_ok = bool(successful) and all(
            run.failure_type != "temporal_instability" and not run.critical_regression for run in successful
        )
        hard_gates = {
            "generation_valid": generation_valid,
            "black_frame_rate": max_black_rate,
            "black_frame_rate_threshold": float(black_frame_rate_threshold),
            "black_frame_gate": None if max_black_rate is None else max_black_rate <= float(black_frame_rate_threshold),
            "decode_success": decode_success,
            "no_critical_temporal_collapse": temporal_ok,
            "quality_drop": quality_drop,
            "quality_drop_limit": quality_limit,
            "quality_gate": quality_gate,
            "efficiency_gate": efficiency_gate,
            "target_feasible": feasible,
        }
        gate_values = [generation_valid, decode_success, temporal_ok]
        if hard_gates["black_frame_gate"] is not None:
            gate_values.append(bool(hard_gates["black_frame_gate"]))
        if quality_gate is not None:
            gate_values.append(bool(quality_gate))
        if efficiency_gate is not None:
            gate_values.append(bool(efficiency_gate))
        accepted = bool(all(gate_values)) if baseline_hardware is not None and baseline_quality is not None else None
        return BenchmarkSummary(
            state.model_id,
            len(runs),
            quality_score,
            quality_metrics,
            hardware,
            feasible,
            violations,
            tuple(runs),
            tuple(sampler.samples),
            attribution,
            efficiency_improvements,
            hard_gates,
            accepted,
        )
