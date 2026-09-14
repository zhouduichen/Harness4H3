from dataclasses import replace
from types import SimpleNamespace

from harness4h3.benchmark.h3 import BenchmarkSummary
from harness4h3.config import WorkflowConfig
from harness4h3.remote.config import load_remote_campaign_config
from harness4h3.remote.ssh import RemoteConfig
from harness4h3.target.profile import load_target_profile
from research.experiments.remote_h3_closed_loop import RemoteCampaign


def _training_result(model_id, parent_id, checkpoint, operator, target_steps):
    return {
        "status": "success",
        "metrics": {
            "optimizer_steps": 1,
            "gradient_norm": 0.2,
            "changed_trainable_tensors": 4,
            "unchanged_frozen_tensors": 531,
            "child_reloaded": True,
            "parent_sha256_before": "a" * 64,
            "parent_sha256_after": "a" * 64,
            "parent_sha256": "a" * 64,
            "child_sha256": ("b" if model_id.endswith("5") else "c") * 64,
        },
        "output_state": {
            "model_id": model_id,
            "parent_model_id": parent_id,
            "checkpoint_path": checkpoint,
            "architecture_name": "MiniMax-H3-FL2VA",
            "dtype": "bfloat16",
            "sampling_steps": target_steps,
            "algorithm_state": {"operator": operator, "target_steps": target_steps},
        },
    }


class _FakeSSH:
    def __init__(self, results):
        self.config = RemoteConfig(
            "fake",
            "/srv/harness",
            "/srv/models",
            "/srv/comfy",
            results_root="/srv/models/results",
            deployment_dir="/srv/comfy/models/diffusion_models",
            campaign_root="/srv/harness/campaign",
        )
        self.results = results

    def find(self, pattern, root=None):
        return list(self.results)

    def read_json(self, path):
        if path not in self.results:
            from harness4h3.remote.ssh import RemoteError

            raise RemoteError("missing fake metadata")
        return self.results[path]["result"]

    def sha256(self, path):
        return ("d" if path.endswith("m0005.json") else "e") * 64

    def ensure_model_link(self, model_path, model_id):
        return "/srv/comfy/models/diffusion_models/%s.safetensors" % model_id


class _FakeTunnel:
    base_url = "http://fake"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class _FakeBenchmark:
    def __init__(self, scores, hardware):
        self.scores = scores
        self.hardware = hardware

    def run(self, state, tasks, **kwargs):
        score = self.scores.get(state.model_id, 0.86)
        hardware = self.hardware.get(state.model_id, {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120})
        runs = tuple(
            {
                "task_id": task.id,
                "quality_score": score,
                "wall_time_s": hardware["latency_s"],
                "quality_metrics": {"decodable": 1.0},
                "failure_type": None,
                "critical_regression": False,
            }
            for task in tasks
        )
        return {
            "model_id": state.model_id,
            "task_count": len(runs),
            "quality_score": score,
            "quality_metrics": {},
            "hardware": {**hardware, "model_size_gb": 66.0},
            "feasible": True,
            "violations": [],
            "runs": runs,
            "hard_gates": {"generation_valid": True, "decode_success": True, "no_critical_temporal_collapse": True},
        }


def build_campaign(tmp_path, results, scores, hardware):
    config = load_remote_campaign_config("configs/remote-l40-h3.yaml")
    config = replace(
        config,
        remote=RemoteConfig(
            "fake",
            "/srv/harness",
            "/srv/models",
            "/srv/comfy",
            results_root="/srv/models/results",
            deployment_dir="/srv/comfy/models/diffusion_models",
            campaign_root="/srv/harness/campaign",
        ),
        runtime=replace(config.runtime, target_path="configs/targets/l40x4_h3_example.yaml"),
        worker=replace(config.worker, enabled=False),
        research_grade=True,
    )
    remote = _FakeSSH(results)
    return RemoteCampaign(
        config,
        ssh=remote,
        target=load_target_profile("configs/targets/l40x4_h3_example.yaml"),
        output_root=tmp_path,
        benchmark_factory=lambda base_url, evaluator: _FakeBenchmark(scores, hardware),
        tunnel_factory=lambda client: _FakeTunnel(),
    )


def test_campaign_imports_chain_evaluates_and_resumes_without_retraining(tmp_path):
    results = {
        "/srv/models/results/trainer_result_m0005.json": {
            "result": _training_result("M0005", "M0000", "/srv/models/results/M0005.safetensors", "recovery_finetune", 32)
        },
        "/srv/models/results/trainer_result_m0006.json": {
            "result": _training_result("M0006", "M0005", "/srv/models/results/M0006.safetensors", "step_distill", 16)
        },
    }
    scores = {"M0000": 0.86, "M0005": 0.86, "M0006": 0.85}
    hardware = {
        "M0000": {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120},
        "M0005": {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120},
        "M0006": {"latency_s": 18, "peak_memory_gb": 36, "energy_j": 90},
    }
    first = build_campaign(tmp_path, results, scores, hardware).run(resume=False, max_experiments=2)
    assert first.records["M0006"].decision["status"] in {"accepted", "rejected"}
    assert first.report["metrics"]["M0006"]["Q"] == 0.85
    second = build_campaign(tmp_path, results, scores, hardware).run(resume=True, max_experiments=2)
    assert second.report["imported_duplicates"] >= 2
    assert second.current_model_id == first.current_model_id
