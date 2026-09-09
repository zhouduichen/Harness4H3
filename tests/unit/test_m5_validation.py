from __future__ import annotations

from pathlib import Path

from harness4h3.backends.comfyui import BackendResult
from harness4h3.benchmark.h3 import H3BenchmarkRunner
from harness4h3.benchmark.validation import M5ValidationRunner, aggregate_summaries
from harness4h3.config import Target, WorkflowConfig
from harness4h3.evaluator.evaluator import EvaluationResult
from harness4h3.h3.state import ModelState
from harness4h3.harness.state import Task


class Backend:
    base_url = "http://127.0.0.1:1"

    def __init__(self):
        self.free_calls = 0
        self.workflows = []

    def free(self):
        self.free_calls += 1
        return {}

    def run(self, workflow, output_dir):
        self.workflows.append(workflow)
        path = Path(output_dir) / "out.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return BackendResult("p", (path,), {}, 2.0)


class Evaluator:
    def evaluate(self, request):
        return EvaluationResult(0.95, {"decodable": 1.0, "black_frame_ratio": 0.0}, False, None)


def state(model_id: str, size: float) -> ModelState:
    return ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "model_id": model_id,
            "checkpoint_path": f"D:\\models\\{model_id}.safetensors",
            "sampling_steps": 20,
            "measured_metrics": {"model_size_gb": size},
        }
    )


def test_aggregate_summaries_reports_reproducibility_statistics(tmp_path):
    config = WorkflowConfig(tmp_path / "workflow.json", Target("131", "prompt"), None, {})
    runner = H3BenchmarkRunner(Backend(), Evaluator(), {"131": {"class_type": "Prompt", "inputs": {"prompt": "old"}}}, config, tmp_path / "out", 0.01)
    task = Task("t1", "visible dragon", "dev")
    summaries = [runner.run(state("M0", 20.0), [task]), runner.run(state("M0", 20.0), [task])]
    stats = aggregate_summaries(summaries)
    assert stats["quality_score"]["count"] == 2
    assert stats["latency_s"]["median"] == 2.0
    assert stats["latency_s"]["ci95_low"] == 2.0


def test_m5_validation_flips_order_and_records_pairwise_gate(tmp_path):
    config = WorkflowConfig(tmp_path / "workflow.json", Target("131", "prompt"), None, {})
    backend = Backend()
    benchmark = H3BenchmarkRunner(backend, Evaluator(), {"131": {"class_type": "Prompt", "inputs": {"prompt": "old"}}}, config, tmp_path / "out", 0.01)
    target = type(
        "Target",
        (),
        {
            "id": "rtx",
            "max_quality_drop": 0.05,
            "max_model_size_gb": None,
            "max_peak_memory_gb": None,
            "max_latency_s": None,
            "max_energy_j": None,
            "min_quality_score": None,
        },
    )()
    result = M5ValidationRunner(benchmark).run(
        state("M0", 20.0),
        state("M1", 12.0),
        [Task("t1", "visible dragon", "sanity")],
        label="sanity",
        repetitions=2,
        target=target,
        operator_attribution={"primary_intervention": "quantization", "secondary_changes": [], "controlled_variables": ["seed"]},
    )
    assert backend.free_calls == 4
    assert [item["order"] for item in result.pairwise] == [["parent", "child"], ["child", "parent"]]
    assert all(item["validated"] for item in result.pairwise)
    assert result.validated is True
    assert result.aggregates["child"]["model_size_gb"]["mean"] == 12.0
