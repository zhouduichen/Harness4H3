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


class ResettableFakeBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.free_calls = 0

    def free(self):
        self.free_calls += 1
        return {}


class FakeEvaluator:
    def evaluate(self, request):
        return EvaluationResult(0.95, {"frames": 22}, False, None)


class BlackFrameEvaluator:
    def evaluate(self, request):
        return EvaluationResult(
            0.49,
            {
                "decodable": 1.0,
                "black_frame_ratio": 1.0,
                "all_black": 1.0,
                "artifact_generation_success": 1.0,
                "operator_execution_success": 1.0,
            },
            True,
            "low_luma",
        )


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


def test_h3_benchmark_switches_safetensors_loader_and_records_black_frame_semantics(tmp_path):
    workflow = {
        "127": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "old.gguf"}},
        "131": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": "old"}},
    }
    config = WorkflowConfig(tmp_path / "workflow.json", Target("131", "prompt"), None, {})
    state = ModelState.from_dict({**ModelState.fake_baseline().to_dict(), "checkpoint_path": "D:\\models\\parent-int8.safetensors"})
    task = Task("t1", "A visible dragon", "sanity")
    backend = FakeBackend()
    summary = H3BenchmarkRunner(backend, BlackFrameEvaluator(), workflow, config, tmp_path / "outputs", 0.01).run(
        state,
        [task],
        operator_attribution={"primary_intervention": "quantization", "secondary_changes": [], "controlled_variables": ["seed"]},
    )
    rendered = backend.workflows[0]
    assert rendered["127"]["class_type"] == "UNETLoader"
    assert rendered["127"]["inputs"]["unet_name"] == "parent-int8.safetensors"
    run = summary.runs[0]
    assert run.operator_execution_success is True
    assert run.artifact_generation_success is True
    assert run.semantic_generation_valid is False
    assert run.failure_type == "degenerate_output_black_frame"
    assert summary.operator_attribution["primary_intervention"] == "quantization"


def test_benchmark_does_not_claim_feasibility_without_baseline_quality(tmp_path):
    workflow = {
        "131": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": "old"}},
    }
    config = WorkflowConfig(tmp_path / "workflow.json", Target("131", "prompt"), None, {})
    state = ModelState.from_dict({**ModelState.fake_baseline().to_dict(), "measured_metrics": {"model_size_gb": 1.0}})
    task = Task("t1", "A visible dragon", "sanity")
    summary = H3BenchmarkRunner(FakeBackend(), FakeEvaluator(), workflow, config, tmp_path / "outputs", 0.01).run(
        state, [task], target=type("Target", (), {"max_model_size_gb": 10, "max_peak_memory_gb": 10, "max_latency_s": 10, "max_energy_j": None, "max_quality_drop": 0.05, "min_quality_score": None})()
    )
    assert summary.feasible is None


def test_benchmark_can_reset_backend_before_controlled_run(tmp_path):
    config = WorkflowConfig(tmp_path / "workflow.json", Target("131", "prompt"), None, {})
    backend = ResettableFakeBackend()
    state = ModelState.fake_baseline()
    task = Task("t1", "A visible dragon", "sanity")
    H3BenchmarkRunner(backend, FakeEvaluator(), {"131": {"class_type": "Prompt", "inputs": {"prompt": "old"}}}, config, tmp_path / "outputs", 0.01).run(
        state, [task], reset_backend_before_run=True
    )
    assert backend.free_calls == 1


def test_cache_release_policy_frees_backend_at_each_task_boundary(tmp_path):
    config = WorkflowConfig(tmp_path / "workflow.json", Target("131", "prompt"), None, {})
    backend = ResettableFakeBackend()
    state = ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "runtime_state": {"runtime_recipe": [{"kind": "cache_release", "args": {"stage": "always"}}]},
        }
    )
    tasks = [Task("t1", "A visible dragon", "dev"), Task("t2", "A red ball", "dev")]
    H3BenchmarkRunner(backend, FakeEvaluator(), {"131": {"class_type": "Prompt", "inputs": {"prompt": "old"}}}, config, tmp_path / "outputs", 0.01).run(
        state, tasks
    )
    assert backend.free_calls == 2
