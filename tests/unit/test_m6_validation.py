from __future__ import annotations

from pathlib import Path

from harness4h3.benchmark.h3 import BenchmarkSummary, BenchmarkTaskResult
from harness4h3.benchmark.m6 import M6ValidationRunner
from harness4h3.controller.schemas import HardwareMetrics
from harness4h3.h3.state import ModelState
from harness4h3.harness.state import Task


def state(model_id: str, policy=None) -> ModelState:
    runtime_state = {"runtime_policy": policy} if policy else {}
    return ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "model_id": model_id,
            "parent_model_id": "M0001" if model_id != "M0001" else "M0000",
            "checkpoint_path": f"D:\\models\\{model_id}.safetensors",
            "architecture_name": "MiniMax-H3",
            "quantization": {"bits": 4, "scheme": "nvfp4"},
            "runtime_state": runtime_state,
            "measured_metrics": {"model_size_gb": 12.5286368},
        }
    )


def summary(model_id: str, quality: float, latency: float, peak: float) -> BenchmarkSummary:
    run = BenchmarkTaskResult(
        "t1", "success", "p", ("/tmp/out.mp4",), latency, quality,
        {"decodable": 1.0, "black_frame_ratio": 0.0}, None, "", True, True, True, False,
    )
    return BenchmarkSummary(
        model_id, 1, quality, {}, HardwareMetrics(latency, peak, 12.5286368), None, (), (run,),
        ({"devices": [{"vram_total": 17_000_000_000, "vram_free": 17_000_000_000 - int(peak * 1_000_000_000)}]},),
        {}, {}, {"generation_valid": True, "black_frame_gate": True, "decode_success": True}, None,
    )


class DummyBenchmark:
    def __init__(self, peak: float):
        self.peak = peak
        self.calls = []

    def run(self, state, tasks, **kwargs):
        self.calls.append((state.model_id, kwargs.get("reset_backend_before_run")))
        if state.runtime_state.get("runtime_policy"):
            return summary(state.model_id, 0.94, 100.0, self.peak)
        return summary(state.model_id, 0.95, 90.0, 16.2)


def test_m6_acceptance_uses_peak_max_and_preserves_reference_efficiency():
    benchmark = DummyBenchmark(15.8)
    result = M6ValidationRunner(benchmark).run(
        state("M0001"),
        state("M0002", {"kind": "vae_tiling", "args": {"tile_size": 256, "overlap": 32}}),
        [Task("t1", "dragon", "dev")],
        label="vae_tiling",
        target=type("Target", (), {"id": "rtx", "max_peak_memory_gb": 16.0, "max_quality_drop": 0.05})(),
        reference_metrics={"model_size_gb": 20.970379616, "latency_s": 205.0},
    )
    assert result.validated is True
    assert result.gates["peak_memory_max_gb"] == 15.8
    assert result.gates["peak_memory_gate"] is True
    assert result.gates["peak_vram_max_gb"] == 15.8
    assert result.aggregates["branch"]["peak_vram_gb"]["mean"] == 15.8
    assert benchmark.calls == [("M0001-m6-vae_tiling-r0", True), ("M0002-m6-vae_tiling-r0", True)]


def test_m6_rejects_branch_when_any_peak_sample_exceeds_target():
    result = M6ValidationRunner(DummyBenchmark(16.2)).run(
        state("M0001"), state("M0002", {"kind": "runtime_offload", "args": {"mode": "aggressive"}}),
        [Task("t1", "dragon", "heldout")], label="runtime_offload", target=type(
            "Target", (), {"id": "rtx", "max_peak_memory_gb": 16.0, "max_quality_drop": 0.05}
        )(), reference_metrics={"model_size_gb": 20.970379616, "latency_s": 205.0},
    )
    assert result.validated is False
    assert result.gates["peak_memory_gate"] is False
