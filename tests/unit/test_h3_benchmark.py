from __future__ import annotations

from pathlib import Path

from harness4h3.backends.comfyui import BackendResult
from harness4h3.benchmark.h3 import H3BenchmarkRunner
from harness4h3.config import Target, WorkflowConfig
from harness4h3.controller.schemas import HardwareMetrics
from harness4h3.evaluator.evaluator import EvaluationResult
from harness4h3.h3.state import ModelState
from harness4h3.harness.state import Task


class FakeBackend:
    base_url = "http://127.0.0.1:1"

    def __init__(self):
        self.workflows = []

    def run(self, workflow, output_dir):
        self.workflows.append(workflow)
        path = Path(output_dir) / "out.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return BackendResult("prompt-1", (path,), {}, 2.0)


class FakeEvaluator:
    def evaluate(self, request):
        return EvaluationResult(0.95, {"frames": 22}, False, None)


def test_h3_benchmark_uses_state_steps_and_reports_quality_latency(tmp_path):
    workflow = {
        "127": {"class_type": "UNETLoader", "inputs": {"unet_name": "baseline.safetensors"}},
        "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "131": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": "old"}},
        "124": {"class_type": "BasicScheduler", "inputs": {"steps": 20}},
    }
    config = WorkflowConfig(
        template=tmp_path / "workflow.json",
        prompt_target=Target("131", "prompt"),
        seed_target=Target("129", "noise_seed"),
        mutable={"steps": Target("124", "steps")},
    )
    state = ModelState.from_dict({**ModelState.fake_baseline().to_dict(), "model_id": "M0001", "checkpoint_path": "D:\\models\\minimax_h3-Q4_K.gguf", "sampling_steps": 4, "measured_metrics": {"model_size_gb": 11.4}})
    task = Task("t1", "A visible dragon", "sanity", 42, {}, {"frames": 22})
    backend = FakeBackend()
    summary = H3BenchmarkRunner(backend, FakeEvaluator(), workflow, config, tmp_path / "outputs", 0.01).run(state, [task])
    rendered = backend.workflows[0]
    assert rendered["124"]["inputs"]["steps"] == 4
    assert rendered["129"]["inputs"]["noise_seed"] == 42
    assert rendered["127"]["class_type"] == "UnetLoaderGGUF"
    assert rendered["127"]["inputs"]["unet_name"] == "minimax_h3-Q4_K.gguf"
    assert summary.quality_score == 0.95
    assert summary.hardware == HardwareMetrics(latency_s=2.0, peak_memory_gb=None, model_size_gb=11.4, energy_j=None, throughput=0.5, thermal=None)
    assert summary.runs[0].status == "success"
