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
        model_node = workflow.get("127")
        if isinstance(model_node, Mapping) and checkpoint_name.lower().endswith(".gguf"):
            model_node["class_type"] = "UnetLoaderGGUF"
            inputs = model_node.setdefault("inputs", {})
            inputs["unet_name"] = checkpoint_name
            workflow["127"] = model_node
        return workflow

    def run(
        self,
        state: ModelState,
        tasks: Sequence[Task],
        baseline_quality: Optional[float] = None,
        target: Any = None,
    ) -> BenchmarkSummary:
        runs: List[BenchmarkTaskResult] = []
        with _SystemSampler(self.backend.base_url, self.system_sample_interval_s) as sampler:
            for task in tasks:
                started = time.monotonic()
                try:
                    result = self.backend.run(self._workflow(state, task), self.output_root / state.model_id / task.id)
                    artifacts = tuple(str(path.resolve()) for path in result.artifacts)
                    evaluation: EvaluationResult = self.evaluator.evaluate(
                        make_request(task.id, task.expected, artifacts, result.wall_time_s, backend_success=True)
                    )
                    runs.append(
                        BenchmarkTaskResult(
                            task.id,
                            "success",
                            result.prompt_id,
                            artifacts,
                            result.wall_time_s,
                            evaluation.score,
                            dict(evaluation.metrics),
                            evaluation.failure_type,
                        )
                    )
                except (BackendError, EvaluatorError, OSError, ValueError) as exc:
                    failure_type = getattr(exc, "failure_type", "benchmark_failure")
                    runs.append(BenchmarkTaskResult(task.id, "failed", None, (), time.monotonic() - started, None, {}, failure_type, str(exc)))
        successful = [run for run in runs if run.quality_score is not None]
        quality_score = mean([float(run.quality_score) for run in successful]) if successful else None
        quality_metrics = {
            "backend": "comfyui",
            "successful_tasks": len(successful),
            "failed_tasks": len(runs) - len(successful),
            "task_scores": {run.task_id: run.quality_score for run in runs},
        }
        latencies = [run.wall_time_s for run in successful]
        measured = state.measured_metrics
        model_size = measured.get("model_size_gb")
        if model_size is None and state.provenance.get("size_bytes") is not None:
            model_size = float(state.provenance["size_bytes"]) / 1_000_000_000
        peak_ram, peak_vram = _metric_values(sampler.samples)
        hardware = HardwareMetrics(
            latency_s=mean(latencies) if latencies else None,
            peak_memory_gb=peak_vram if peak_vram is not None else peak_ram,
            model_size_gb=float(model_size) if model_size is not None else None,
            energy_j=float(measured["energy_j"]) if measured.get("energy_j") is not None else None,
            throughput=(1.0 / mean(latencies)) if latencies and mean(latencies) > 0 else None,
            thermal=None,
        )
        feasible: Optional[bool] = None
        violations: Tuple[str, ...] = ()
        if target is not None and quality_score is not None:
            from ..evaluator.constraints import ConstraintEvaluator

            feasible_value, violation_list, _ = ConstraintEvaluator().evaluate(
                quality_score, hardware, target, float(baseline_quality if baseline_quality is not None else quality_score)
            )
            feasible, violations = feasible_value, tuple(violation_list)
        return BenchmarkSummary(state.model_id, len(runs), quality_score, quality_metrics, hardware, feasible, violations, tuple(runs), tuple(sampler.samples))
