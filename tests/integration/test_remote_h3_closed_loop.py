from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

from harness4h3.benchmark.h3 import BenchmarkSummary
from harness4h3.config import WorkflowConfig
from harness4h3.controller.schemas import ExperimentPlan
from harness4h3.controller.directive import submit_directive
from harness4h3.controller.provider import ControllerUnavailableError, experiment_plan_json_schema
from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.h3.state import ModelState
from harness4h3.memory.experience import ExperienceRecord
from harness4h3.memory.observation import make_observation
from harness4h3.remote.config import load_remote_campaign_config
from harness4h3.remote.ssh import RemoteConfig, RemoteError, SSHClient
from harness4h3.remote.scheduler import ResourceDecision
from harness4h3.controller.reviewer import ReviewDecision
from harness4h3.target.profile import load_target_profile
from research.experiments.remote_h3_closed_loop import CampaignResult, RemoteCampaign


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
        self.writes = {}
        self.commands = []

    def run(self, command, **kwargs):
        command = tuple(command)
        self.commands.append(command)
        if command and command[0] == "test" and any(
            "gpu-lease.json" in item or "comfyui-lease.json" in item for item in command
        ):
            return SimpleNamespace(stdout="", stderr="", returncode=1)
        if any("/v1/models" in item for item in command):
            return SimpleNamespace(
                stdout=json.dumps({"data": [{"id": "qwen3.5-controller"}]}),
                stderr="",
                returncode=0,
            )
        if any("/queue" in item for item in command):
            return SimpleNamespace(stdout=json.dumps({"queue_running": [], "queue_pending": []}), stderr="", returncode=0)
        if any("/free" in item for item in command):
            return SimpleNamespace(stdout=json.dumps({"ok": True}), stderr="", returncode=0)
        if any("query-compute-apps" in item for item in command):
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return SimpleNamespace(
            stdout="0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
            stderr="",
            returncode=0,
        )

    def find(self, pattern, root=None):
        return list(self.results)

    def read_json(self, path):
        if path in self.results:
            return self.results[path]["result"]
        if path.endswith("-config.json") or "a1-worker" in path:
            return {"max_steps": 1, "world_size": 4}
        if "trainer_result_" in path:
            return _training_result("M0001", "M0000", "/srv/models/results/M0001.safetensors", "recovery_finetune", 32)
        from harness4h3.remote.ssh import RemoteError

        raise RemoteError("missing fake metadata")

    def write_json(self, path, value):
        self.writes[path] = value

    def sha256(self, path):
        return ("d" if path.endswith("m0005.json") else "e") * 64

    def remove_file(self, path):
        return {"status": "missing"}

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
    observations = list(build_campaign(tmp_path, results, scores, hardware).observations.read())
    assert any(item.kind == "experience" and "#benchmark" in item.source_uri for item in observations)
    second = build_campaign(tmp_path, results, scores, hardware).run(resume=True, max_experiments=2)
    assert second.report["imported_duplicates"] >= 2
    assert second.current_model_id == first.current_model_id


def test_discovery_digest_carries_bounded_pipeline_telemetry(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._pipeline_telemetry = lambda: {
        "window_events": 3,
        "underutilized_gpu_indices": ["2"],
        "power_target_w": 300.0,
        "last_power_feedback": {
            "target_power_w": 300.0,
            "per_gpu": {
                "2": {
                    "power_w_avg": 120.0,
                    "utilization_gpu_pct_avg": 45.0,
                    "lane": "controller",
                }
            },
        },
    }

    digest = campaign._build_discovery_digest()

    item = digest.to_context()["telemetry"]["items"][0]
    assert item["window_events"] == 3
    assert item["last_power_feedback"]["per_gpu"]["2"]["lane"] == "controller"


def test_campaign_records_comfyui_lease_release_after_evaluation(tmp_path):
    results = {
        "/srv/models/results/trainer_result_m0005.json": {
            "result": _training_result(
                "M0005",
                "M0000",
                "/srv/models/results/M0005.safetensors",
                "recovery_finetune",
                32,
            )
        }
    }
    campaign = build_campaign(
        tmp_path,
        results,
        {"M0000": 0.86, "M0005": 0.86},
        {
            "M0000": {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120},
            "M0005": {"latency_s": 30, "peak_memory_gb": 42, "energy_j": 120},
        },
    )
    result = campaign.run(resume=False, max_experiments=1)
    events = [json.loads(line) for line in (tmp_path / "controller-events.jsonl").read_text().splitlines()]
    names = [event["event_type"] for event in events]
    assert "comfyui_lease_reserved" in names
    assert "comfyui_cache_release_requested" in names
    assert "comfyui_cache_released" in names
    lane_events = [event for event in events if event["event_type"] == "lane_allocation"]
    assert lane_events
    assert lane_events[-1]["stage"] == "evaluating"
    assert lane_events[-1]["lanes"][0]["lane"] == "comfyui"
    assert lane_events[-1]["lanes"][0]["lease_file"]
    assert names.index("evaluation_decision") < names.index("comfyui_cache_release_requested")
    assert names.index("comfyui_cache_release_requested") < names.index("comfyui_cache_released")
    assert campaign.scheduler.reserved_gpu_indices == ()
    assert result.report["benchmark_cache_policy"] == "idle_release"
    recipe = campaign._load_evaluations()["M0005"]["benchmark_recipe"]
    assert recipe["comfyui_lease_release"]["success"] is True


def test_multiple_independent_tasks_use_parallel_comfyui_workers(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, benchmark_splits=("dev",))
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)
    active = 0
    maximum = 0
    lock = threading.Lock()

    class ParallelBenchmark:
        def run(self, state, tasks, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.05)
                return _FakeBenchmark({}, {}).run(state, tasks, **kwargs)
            finally:
                with lock:
                    active -= 1

    campaign.benchmark_factory = lambda _base_url, _evaluator: ParallelBenchmark()
    summary = campaign._evaluate(parent, None, "dev")
    campaign._release_comfyui_if_idle("parallel_eval_test")

    started = [item for item in campaign.events.read() if item.get("event_type") == "evaluation_started"][-1]
    assert started["evaluation_workers"] == 2
    assert maximum == 2
    assert summary["task_count"] == 2
    assert campaign.scheduler.reserved_gpu_indices == ()


def test_evaluation_started_precedes_overlap_callback_events(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, benchmark_splits=("dev",))
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)

    campaign._evaluate(
        parent,
        None,
        "dev",
        on_benchmark_reserved=lambda: campaign.events.append(
            "speculative_worker_started",
            {"allocated_gpus": [1, 2]},
        ),
    )

    event_types = [item.get("event_type") for item in campaign.events.read()]
    assert event_types.index("evaluation_started") < event_types.index("speculative_worker_started")


def test_pipeline_evaluation_fans_out_independent_tasks(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        pipeline_enabled=True,
    )

    specs = campaign._evaluation_worker_specs(campaign._tasks("dev"))

    assert len(specs) == 2
    assert specs[0].gpu_index == 0
    overlap = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "evaluation_overlap_budget"
    ]
    assert overlap[-1]["training_overlap_reserved"] == 0
    assert overlap[-1]["reason"] == "fan_out_independent_evaluation_tasks"


def test_evaluation_releases_campaign_controller_when_initial_selection_is_empty(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, benchmark_splits=("dev",))
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)

    class Scheduler:
        gpu_count = 4
        reserved_gpu_indices = ()
        controller_reserved_gpu_indices = (0, 1, 2, 3)

        def reserve_gpu(self, index):
            self.reserved_gpu_indices = tuple(sorted(set(self.reserved_gpu_indices) | {int(index)}))

        def release_gpu(self, index):
            self.reserved_gpu_indices = tuple(item for item in self.reserved_gpu_indices if item != int(index))

        def snapshot(self):
            return ({index: (0, 46080) for index in range(4)}, "")

        def meets_memory_waterline(self, _index, _snapshot):
            return True

    scheduler = Scheduler()
    campaign.scheduler = scheduler
    for lease in campaign.comfyui_leases.values():
        lease.scheduler = scheduler
    selected_worker = SimpleNamespace(gpu_index=0, port=8188)
    selection_calls = []

    def select_workers(_tasks):
        selection_calls.append(True)
        return () if len(selection_calls) == 1 else (selected_worker,)

    campaign._evaluation_worker_specs = select_workers
    releases = []

    def release_controller(reason, wait_s=30.0, **_kwargs):
        releases.append((reason, wait_s))
        return {"status": "released", "reason": reason}

    campaign._request_controller_release = release_controller
    summary = campaign._evaluate(parent, None, "dev")

    assert summary["task_count"] == 2
    assert len(selection_calls) == 2
    assert releases == [("evaluation_gpu_allocation", 30.0)]
    event = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "evaluation_controller_release"
    ][-1]
    assert event["status"] == "released"
    assert event["controller_gpus"] == [0, 1, 2, 3]


def test_pipeline_evaluation_filters_busy_secondary_lanes(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        pipeline_enabled=True,
    )

    class SnapshotScheduler:
        gpu_count = 4
        reserved_gpu_indices = ()
        last_compute_gpu_indices = (1, 3)
        controller_reserved_gpu_indices = (3,)

        def snapshot(self):
            return (
                {
                    index: (0, 46080)
                    for index in range(self.gpu_count)
                },
                "GPU-1, 100, 4096\nGPU-3, 101, 4096\n",
            )

        def meets_memory_waterline(self, index, _snapshot):
            return index not in {1, 3}

    campaign.scheduler = SnapshotScheduler()
    specs = campaign._evaluation_worker_specs(campaign._tasks("dev"))

    assert [worker.gpu_index for worker in specs] == [0, 2]
    overlap = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "evaluation_overlap_budget"
    ]
    assert overlap[-1]["evaluation_workers"] == 2


def test_speculative_overlap_hands_off_multi_gpu_controller_lane(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        pipeline_enabled=True,
        controller_overlap_gpus=1,
    )

    class MultiGpuControllerScheduler:
        controller_reserved_gpu_indices = (1, 2)

    campaign.scheduler = MultiGpuControllerScheduler()

    assert campaign._preserve_controller_lane_for_speculative_worker() is False
    handoff = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "controller_lane_handoff_requested"
    ]
    assert handoff[-1]["controller_gpus"] == [1, 2]
    assert handoff[-1]["configured_overlap_gpus"] == 1


def test_evaluation_successor_hands_off_tp1_controller_for_three_worker_gpus(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        pipeline_enabled=True,
        controller_overlap_gpus=1,
    )
    campaign._active_evaluation_gpu_indices = (0,)
    campaign._comfyui_lease_active = lambda: True

    class EvaluationOverlapScheduler:
        controller_reserved_gpu_indices = (1,)

    campaign.scheduler = EvaluationOverlapScheduler()

    assert campaign._preserve_controller_lane_for_speculative_worker() is False
    handoff = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "controller_lane_handoff_requested"
    ]
    assert handoff[-1]["reason"] == "evaluation_successor_ready_use_all_non_evaluator_gpus"
    assert handoff[-1]["evaluator_gpus"] == [0]


def test_real_ssh_prefers_fresh_secondary_when_primary_lacks_h3_extension(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        comfyui_workers=(
            SimpleNamespace(gpu_index=0, port=8188),
            SimpleNamespace(gpu_index=1, port=8189),
        ),
        worker=replace(campaign.config.worker, enabled=True),
        pipeline_enabled=True,
    )

    class LiveSSH(SSHClient):
        def __init__(self, config):
            self.config = config

    class FreeScheduler:
        gpu_count = 4
        reserved_gpu_indices = ()
        last_compute_gpu_indices = ()
        controller_reserved_gpu_indices = ()

        def snapshot(self):
            return ({index: (0, 46080) for index in range(4)}, "")

        def meets_memory_waterline(self, _index, _snapshot):
            return True

    campaign.ssh = LiveSSH(campaign.ssh.config)
    campaign.scheduler = FreeScheduler()
    campaign.optimization_capabilities = {
        "lpl": {"safe_to_plan": False},
        "tdtm": {"safe_to_plan": False},
    }
    specs = campaign._evaluation_worker_specs(campaign._tasks("dev"))

    assert [worker.gpu_index for worker in specs] == [1, 0]
    preferred = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "evaluation_worker_preferred"
    ]
    assert preferred[-1]["gpu_index"] == 1


def test_comfyui_primary_lease_uses_controller_compatible_marker(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.ssh.remove_file = lambda _path: {"status": "deleted"}
    campaign.comfyui_leases.pop((0, 8188), None)

    primary = campaign._comfyui_lease_for(SimpleNamespace(gpu_index=0, port=8188))
    secondary = campaign._comfyui_lease_for(SimpleNamespace(gpu_index=2, port=8189))

    assert primary.lease_path == "/srv/harness/campaign/.comfyui-gpu-lease.json"
    assert secondary.lease_path == "/srv/harness/campaign/.comfyui-gpu-lease-2.json"


def test_controller_context_bounds_observations_but_keeps_unconsumed_ids(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)
    total = campaign.config.max_context_observations + 12
    for index in range(total):
        campaign._append_observation(
            make_observation(
                "obs-context-%03d" % index,
                "experience",
                "ssh://fake/context-%03d.json" % index,
                ("%064x" % (index + 1))[-64:],
                experiment_id="exp_%04d" % (index + 1),
                model_id="M0000",
                summary={"recipe": {"operator": "step_distill", "index": index}},
            )
        )

    context = campaign._controller_context(parent, 0)
    visible_ids = {item["observation_id"] for item in context.observations}

    assert len(context.observations) == campaign.config.max_context_observations
    assert set(context.unconsumed_observation_ids) == visible_ids
    assert "obs-context-%03d" % (total - 1) in visible_ids
    assert "obs-context-000" not in visible_ids
    assert "obs-goal-l40x4_h3_v1" in visible_ids
    assert len(list(campaign.observations.read())) == total + 1  # includes the durable goal observation


def test_controller_trace_observation_ids_follow_bounded_context_window(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)
    total = campaign.config.max_context_observations + 12
    for index in range(total):
        campaign._append_observation(
            make_observation(
                "obs-trace-%03d" % index,
                "experience",
                "ssh://fake/trace-%03d.json" % index,
                ("%064x" % (index + 101))[-64:],
                experiment_id="exp_%04d" % (index + 101),
                model_id="M0000",
                summary={"recipe": {"operator": "step_distill", "index": index}},
            )
        )

    class Controller:
        provider_name = "test"
        model_name = "bounded-trace-controller"

        def __init__(self):
            self.context = None

        def plan(self, context):
            self.context = context
            return ExperimentPlan(
                experiment_id="exp_0001",
                parent_model_id=context.current_model_state.model_id,
                diagnosis="bounded trace",
                objective="keep controller evidence bounded",
                hypothesis="the bounded window is sufficient",
                operator="prune_blocks",
                operator_args={"ratio": 0.1},
                expected_effects={"quality_score": "preserve"},
                risks=["none"],
                required_budget={"wall_time_s": 1.0, "gpu_hours": 0.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="bounded trace test",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids),
                resource_request={
                    "gpu_count": 0,
                    "distributed": False,
                    "exclusive": False,
                    "evaluation_workers": 0,
                    "on_unavailable": "replan",
                },
            )

    campaign.controller = Controller()
    plan = campaign._controller_plan(parent, 0)

    assert plan is not None
    trace_ids = campaign.controller_trace[-1]["input_observation_ids"]
    context_ids = [item["observation_id"] for item in campaign.controller.context.observations]
    assert len(trace_ids) == campaign.config.max_context_observations
    assert trace_ids == context_ids
    assert "obs-trace-000" not in trace_ids


def test_controller_experiment_id_skips_recovered_experience(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)
    campaign.experience.append(
        ExperienceRecord.minimal(
            "exp_0023",
            "ssh://fake/recovered-m0022.json",
            "a" * 64,
        )
    )

    context = campaign._controller_context(parent, training_calls=22)
    schema = experiment_plan_json_schema(context)

    assert context.budget_state.used_iterations == 23
    assert schema["properties"]["experiment_id"]["const"] == "exp_0024"


def test_missing_candidate_checkpoint_skips_comfyui_without_reserving_gpu(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    candidate = ModelCandidate(
        "M0001",
        "M0000",
        1,
        "fake://missing/M0001.safetensors",
        replace(
            ModelState.fake_baseline("M0001"),
            parent_model_id="M0000",
            checkpoint_path="fake://missing/M0001.safetensors",
        ),
        "exp_missing",
        "candidate",
    )
    original_run = campaign.ssh.run

    def run(command, **kwargs):
        if tuple(command[:2]) == ("test", "-e"):
            return SimpleNamespace(stdout="", stderr="", returncode=1)
        return original_run(command, **kwargs)

    campaign.ssh.run = run
    summary = campaign._evaluate(candidate, None, "dev")

    assert summary["violations"] == ["checkpoint_missing"]
    assert summary["benchmark_workers"] == []
    assert campaign.scheduler.reserved_gpu_indices == ()
    assert any(
        item.get("event_type") == "evaluation_skipped_missing_checkpoint"
        for item in campaign.events.read()
    )


def test_campaign_does_not_release_comfyui_after_a_cycle_without_evaluation(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})

    result = campaign.run(resume=False, max_experiments=1)

    assert result.status == "completed"
    assert campaign.scheduler.reserved_gpu_indices == ()
    events = [json.loads(line) for line in (tmp_path / "controller-events.jsonl").read_text().splitlines()]
    names = [event["event_type"] for event in events]
    assert "comfyui_cache_release_requested" not in names
    assert "comfyui_cache_released" not in names


def test_releasing_all_comfyui_leases_clears_evaluation_lane_telemetry(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._active_evaluation_gpu_indices = (0,)
    campaign._prepare_comfyui_lease()

    result = campaign._release_comfyui_if_idle("evaluation_phase_completed")

    assert result.success is True
    assert campaign._active_evaluation_gpu_indices == ()
    assert campaign.scheduler.reserved_gpu_indices == ()


def test_evaluation_observation_identity_ignores_post_measurement_bookkeeping(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    summary = {
        "model_id": "M0001",
        "evaluation_id": "eval-M0001",
        "system_id": "S0001",
        "device_id": "l40x4",
        "task_split": "heldout",
        "benchmark_recipe": {
            "target_profile_id": "l40x4_h3_v1",
            "split": "heldout",
            "workflow_template": "workflow.json",
            "quality_scope": "structural_proxy",
            "comfyui_cache_policy": "idle_release",
            "comfyui_lease_state": "reserved_for_benchmark",
        },
        "quality_score": 0.9,
        "quality_metrics": {"successful_tasks": 1},
        "hardware": {"latency_s": 30.0, "peak_memory_gb": 42.0, "model_size_gb": 66.0},
        "hard_gates": {"generation_valid": True},
        "feasible": True,
        "violations": [],
        "task_count": 1,
        "runs": [{"task_id": "t1", "quality_score": 0.9}],
    }
    first = campaign._evaluation_observation("M0001", summary)
    enriched = dict(summary)
    enriched["checkpoint_retention"] = {"retained": True, "reason": "accepted_candidate_protected"}
    enriched["benchmark_recipe"] = {
        **summary["benchmark_recipe"],
        "comfyui_lease_release": {"state": "released_for_other_work", "success": True},
    }
    second = campaign._evaluation_observation("M0001", enriched)

    assert second.observation_id == first.observation_id
    assert second.source_sha256 == first.source_sha256
    assert campaign.observations.append(first) is True
    assert campaign.observations.append(second) is False


def test_remote_root_state_records_checkpoint_size_for_hardware_objective(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    original_run = campaign.ssh.run

    def run(command, **kwargs):
        if tuple(command[:2]) == ("stat", "-c"):
            return SimpleNamespace(stdout="66280487368\n", stderr="", returncode=0)
        return original_run(command, **kwargs)

    campaign.ssh.run = run
    campaign._register_candidates([])
    state = campaign.models.get("M0000").state

    assert state.provenance["size_bytes"] == 66280487368
    assert state.checkpoint_path.endswith("minimax_h3_fl2va_bf16.safetensors")


def test_checkpoint_retention_caps_completed_weights_but_keeps_recent_rollback_set(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, max_retained_checkpoints=3)
    root = ModelCandidate(
        "M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline"
    )
    campaign.models.initialize(root)
    candidates = {root.id: root}
    for number in range(1, 6):
        model_id = "M%04d" % number
        parent_id = "M%04d" % (number - 1)
        state = replace(
            ModelState.fake_baseline(model_id),
            parent_model_id=parent_id,
            checkpoint_path="fake://%s" % model_id,
        )
        candidate = ModelCandidate(
            model_id,
            parent_id,
            number,
            state.checkpoint_path,
            state,
            "exp_%04d" % number,
            "candidate",
        )
        campaign.models.create(candidate)
        candidates[model_id] = candidate
    campaign.models.set_active("M0005")
    signature = campaign._evaluation_signature("heldout")
    evaluations = {
        model_id: {"evaluation_signature": signature, "quality_score": 0.8}
        for model_id in candidates
        if model_id != "M0000"
    }
    deleted = []
    campaign._retain_checkpoint = lambda checkpoint, model_id, outcome, parent_checkpoint_path=None: (
        deleted.append(model_id)
        or {"model_id": model_id, "outcome": outcome, "retained": False, "deleted": True}
    )

    reclaimed = campaign._reclaim_superseded_checkpoints(candidates, evaluations, "heldout")

    assert [item["model_id"] for item in reclaimed] == ["M0001", "M0002"]
    assert deleted == ["M0001", "M0002"]
    assert all(
        evaluations[model_id]["checkpoint_retention"]["retained"] is False
        for model_id in ("M0001", "M0002")
    )
    assert "checkpoint_retention" not in evaluations["M0003"]
    events = [json.loads(line) for line in (tmp_path / "controller-events.jsonl").read_text().splitlines()]
    cap = [event for event in events if event["event_type"] == "checkpoint_retention_cap"][-1]
    assert cap["max_retained_checkpoints"] == 3
    assert cap["kept_model_ids"] == ["M0005", "M0004", "M0003"]
    assert cap["metadata_preserved"] is True


def test_controller_plan_is_the_only_training_plan_source(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    directive, written = submit_directive(
        campaign.observations,
        "下一轮优先降低 peak_memory，质量下降不得超过 2%",
        directive_id="goal-001",
    )
    assert written is True

    class Controller:
        provider_name = "test"
        model_name = "controller"

        def __init__(self):
            self.contexts = []

        def plan(self, context):
            self.contexts.append(context)
            return ExperimentPlan(
                experiment_id="exp_0001",
                parent_model_id=context.current_model_state.model_id,
                diagnosis="test_diagnosis",
                objective="test_objective",
                hypothesis="test_hypothesis",
                operator="recovery_finetune",
                operator_args={"training_steps": 1},
                expected_effects={"quality_score": "preserve"},
                risks=["quality regression"],
                required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="controller test plan",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids),
                resource_request={
                    "gpu_count": 4,
                    "distributed": True,
                    "exclusive": True,
                    "evaluation_workers": 1,
                    "on_unavailable": "wait",
                },
            )

    controller = Controller()
    campaign.controller = controller
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    plan = campaign._controller_plan(parent, 0)

    assert plan is not None
    assert controller.contexts[0].current_model_state.model_id == "M0000"
    assert directive.to_observation().observation_id in controller.contexts[0].unconsumed_observation_ids
    assert directive.to_observation().observation_id in plan.consumed_observation_ids
    assert campaign.controller_trace[0]["status"] == "validated"
    assert campaign.controller_trace[0]["plan"]["operator"] == "recovery_finetune"
    input_events = [
        item for item in campaign.events.read() if item.get("event_type") == "controller_input"
    ]
    assert len(input_events) == 1
    assert "summary_keys" in input_events[0]["observation_summaries"][0]
    assert "summary" not in input_events[0]["observation_summaries"][0]


def test_primary_plan_activates_round_policy_and_enforces_it(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    def policy(campaign_arg, allowed, max_gpu_hours=2.0):
        return {
            "schema_version": 1,
            "round_id": "R0001",
            "substrate_digest": campaign_arg._round_policy_substrate_digest(),
            "search_mode": "runtime_efficiency",
            "allowed_operators": list(allowed),
            "axis_budget": {"max_trials": 2, "max_gpu_hours": max_gpu_hours},
            "objective": {"quality_floor": 0.8},
            "fixed_evaluation": {
                "split": "heldout",
                "recipe_digest": campaign_arg._evaluation_signature("heldout"),
            },
            "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
            "stop_conditions": ["critical_regression", "budget_exhausted"],
            "source_observation_ids": [],
            "created_at": "2026-09-19T00:00:00+00:00",
        }

    class Controller:
        provider_name = "test"
        model_name = "policy-controller"

        def __init__(self):
            self.operator = "recovery_finetune"
            self.include_policy = True
            self.calls = 0

        def plan(self, context):
            self.calls += 1
            operator = self.operator
            return ExperimentPlan(
                experiment_id="exp_%04d" % (context.budget_state.used_iterations + 1),
                parent_model_id=context.current_model_state.model_id,
                parent_system_id=str(context.current_system.get("id") or "S0000"),
                diagnosis="policy test",
                objective="exercise policy lifecycle",
                hypothesis="the fixed policy bounds the next operator",
                operator=operator,
                operator_args={"training_steps": 1},
                expected_effects={"quality_score": "preserve"},
                risks=["quality regression"],
                required_budget={"wall_time_s": 1.0, "gpu_hours": 1.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="policy lifecycle test",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids) or [
                    "goal:l40x4_h3_v1"
                ],
                resource_request={
                    "gpu_count": 4,
                    "min_gpu_count": 2,
                    "max_gpu_count": 4,
                    "elastic": True,
                    "distributed": True,
                    "exclusive": False,
                    "evaluation_workers": 1,
                    "on_unavailable": "wait",
                },
                round_policy=(
                    policy(campaign, ("recovery_finetune",))
                    if self.include_policy
                    else None
                ),
            )

    controller = Controller()
    campaign.controller = controller
    first = campaign._controller_plan(root, 0)
    assert first is not None
    assert first.round_policy["round_id"] == "R0001"
    assert json.loads((tmp_path / "active-round-policy.json").read_text())["round_id"] == "R0001"
    assert campaign._load_campaign_state()["active_round_policy"]["round_id"] == "R0001"
    assert any(item["event_type"] == "round_policy_activated" for item in campaign.events.read())

    controller.include_policy = False
    controller.operator = "distill"
    second = campaign._controller_plan(root, 1)
    assert second is None
    assert "does not allow operator distill" in campaign.controller_trace[-1]["error"]

    exhausted_state = campaign._load_campaign_state()
    exhausted_state["round_policy_progress"] = {
        "round_id": "R0001",
        "trials_completed": 2,
        "gpu_hours_used": 1.0,
        "completed_experiment_ids": ["exp_0001", "exp_0002"],
        "stop_reason": None,
    }
    campaign._save_campaign_state(exhausted_state)
    controller.operator = "recovery_finetune"
    assert campaign._controller_plan(root, 2) is None
    assert campaign.controller_trace[-1]["status"] == "round_policy_stopped"
    assert campaign.controller_trace[-1]["reason"] == "budget_exhausted"


def test_prefetch_plan_cannot_activate_round_policy(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")
    policy = {
        "schema_version": 1,
        "round_id": "R0001",
        "substrate_digest": campaign._round_policy_substrate_digest(),
        "search_mode": "runtime_efficiency",
        "allowed_operators": ["recovery_finetune"],
        "axis_budget": {"max_trials": 2, "max_gpu_hours": 2.0},
        "objective": {"quality_floor": 0.8},
        "fixed_evaluation": {
            "split": "heldout",
            "recipe_digest": campaign._evaluation_signature("heldout"),
        },
        "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
        "stop_conditions": ["critical_regression"],
        "source_observation_ids": [],
        "created_at": "2026-09-19T00:00:00+00:00",
    }

    class Controller:
        provider_name = "test"
        model_name = "prefetch-policy-controller"

        def plan(self, context):
            return ExperimentPlan(
                experiment_id="exp_0001",
                parent_model_id=context.current_model_state.model_id,
                parent_system_id=str(context.current_system.get("id") or "S0000"),
                diagnosis="parallel policy",
                objective="fill GPUs",
                hypothesis="parallel work removes idle time",
                operator="recovery_finetune",
                operator_args={"training_steps": 1},
                expected_effects={"quality_score": "preserve"},
                risks=["speculative"],
                required_budget={"wall_time_s": 1.0, "gpu_hours": 1.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="parallel policy test",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids) or ["goal:l40x4_h3_v1"],
                resource_request={
                    "gpu_count": 2,
                    "min_gpu_count": 2,
                    "max_gpu_count": 2,
                    "elastic": True,
                    "distributed": True,
                    "exclusive": False,
                    "evaluation_workers": 1,
                    "on_unavailable": "wait",
                },
                round_policy=policy,
            )

    campaign.controller = Controller()
    assert campaign._controller_plan(root, 0, prefetch=True, planning_intent="parallel_gpu_fill") is None
    assert "may only be activated" in campaign.controller_prefetch_trace[-1]["error"]
    assert not (tmp_path / "active-round-policy.json").exists()


def test_controller_plan_prefetch_is_persisted_separately(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    class Controller:
        provider_name = "test"
        model_name = "prefetch-controller"
        timeout_s = 1.0

        def plan(self, context):
            return ExperimentPlan(
                experiment_id="exp_%04d" % (context.budget_state.used_iterations + 1),
                parent_model_id=context.current_model_state.model_id,
                parent_system_id="S0000",
                diagnosis="prefetch diagnosis",
                objective="prepare the next branch while the worker runs",
                hypothesis="overlap removes the controller gap",
                operator="recovery_finetune",
                operator_args={"training_steps": 1},
                expected_effects={"quality_score": "preserve"},
                risks=["speculative plan is discarded when stale"],
                required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="prefetch for the next safe boundary",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids),
                resource_request={
                    # Explicitly request the ordinary overlap mode.  A plan
                    # that requests all four cards is a deliberate handoff
                    # mode and must release the Controller first.
                    "gpu_count": 3,
                    "min_gpu_count": 2,
                    "max_gpu_count": 4,
                    "elastic": True,
                    "distributed": True,
                    "exclusive": False,
                    "evaluation_workers": 1,
                    "on_unavailable": "wait",
                },
            )

    campaign.controller = Controller()
    handle = campaign._start_controller_prefetch(root, 0)

    # The worker boundary has not been reached yet.  A completed LLM call
    # must nevertheless publish the plan so a restart/reviewer can reuse it
    # without opening another Controller request.
    deadline = time.monotonic() + 1.0
    state = {}
    while time.monotonic() < deadline:
        state = campaign._load_campaign_state()
        if isinstance(state.get("prefetched_plan"), dict):
            break
        time.sleep(0.01)
    assert state["prefetched_plan"]["plan"]["experiment_id"] == "exp_0002"

    plan = campaign._finish_controller_prefetch(handle)

    assert plan is not None
    assert plan.experiment_id == "exp_0002"
    assert campaign.controller_trace == []
    assert campaign.controller_prefetch_trace[-1]["prefetch"] is True
    state = campaign._load_campaign_state()
    assert state["prefetched_plan"]["plan"]["experiment_id"] == "exp_0002"
    assert state["prefetched_training_calls"] == 1
    assert len(
        [item for item in campaign.events.read() if item.get("event_type") == "controller_plan_prefetch_ready"]
    ) == 1


def test_failed_child_clears_only_its_prefetch_cursors(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    state = {
        "prefetched_source_child_model_id": "M0001",
        "prefetched_plan": {"plan": {"experiment_id": "exp_0002"}},
        "prefetched_parent_model_id": "M0000",
        "pending_plan": {"experiment_id": "exp_0001"},
        "parallel_prefetched_source_child_model_id": "M0001",
        "parallel_prefetched_plan": {"plan": {"experiment_id": "exp_0003"}},
        "parallel_prefetched_parent_model_id": "M0001",
        "unrelated": "keep",
    }

    primary_stale, parallel_stale = campaign._clear_prefetch_state_for_child(state, "M0001")

    assert (primary_stale, parallel_stale) == (True, True)
    assert "prefetched_plan" not in state
    assert "pending_plan" not in state
    assert "parallel_prefetched_plan" not in state
    assert state["unrelated"] == "keep"


def test_primary_prefetch_persists_one_gpu_sibling_candidate(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    primary = ExperimentPlan(
        experiment_id="exp_0002",
        parent_model_id=root.id,
        parent_system_id="S0000",
        diagnosis="reduce structure first",
        objective="make a safe CPU-only structural change",
        hypothesis="pruning reduces memory before GPU training",
        operator="prune_blocks",
        operator_args={"ratio": 0.1},
        expected_effects={"quality_score": "preserve"},
        risks=["quality regression"],
        required_budget={"wall_time_s": 1.0, "gpu_hours": 0.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="CPU primary with a durable GPU sibling",
        consumed_observation_ids=["obs-goal-l40x4_h3_v1"],
        diagnosis_evidence=["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 0,
            "min_gpu_count": 0,
            "max_gpu_count": 0,
            "elastic": False,
            "distributed": False,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "replan",
        },
    )
    sibling = replace(
        primary,
        experiment_id="exp_0999",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        required_budget={"wall_time_s": 1.0, "gpu_hours": 0.5, "controller_calls": 0},
        resource_request={
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 2,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )

    calls = []

    class Controller:
        provider_name = "test"
        model_name = "persisted-batch"
        timeout_s = 1.0

        def plan(self, _context):
            calls.append(True)
            self.last_eligible_candidates = [primary, sibling]
            return primary

    campaign.controller = Controller()
    handle = campaign._start_controller_prefetch(root, 0)
    assert campaign._finish_controller_prefetch(handle) == primary
    assert campaign._finish_controller_prefetch(handle) == primary
    assert len(calls) == 1

    state = campaign._load_campaign_state()
    assert state["prefetched_plan"]["plan"]["experiment_id"] == "exp_0002"
    assert state["parallel_prefetched_plan"]["plan"]["experiment_id"] == "exp_0003"
    assert state["parallel_prefetched_plan"]["plan"]["operator"] == "recovery_finetune"
    assert state["parallel_prefetched_training_calls"] == 1
    parallel_plan = campaign._parallel_prefetched_plan(state)
    assert campaign._parallel_prefetch_is_current(state, parallel_plan, root.id, 1) is True
    assert campaign._ready_parallel_prefetched_plan(state, root.id, 1) == parallel_plan
    stale_state = dict(state)
    stale_state["parallel_prefetched_training_calls"] = 2
    assert campaign._parallel_prefetch_is_current(stale_state, parallel_plan, root.id, 1) is False
    assert campaign._ready_parallel_prefetched_plan(stale_state, root.id, 1) is None
    stale_state = dict(state)
    stale_state["parallel_prefetched_source_child_model_id"] = "M0099"
    assert campaign._parallel_prefetch_is_current(stale_state, parallel_plan, root.id, 1) is False

    campaign._active_evaluation_gpu_indices = (0,)
    wide_parallel = replace(
        sibling,
        resource_request={
            **sibling.resource_request,
            "gpu_count": 4,
            "max_gpu_count": 4,
        },
    )
    packed_parallel = campaign._parallel_gpu_fill_execution_plan(wide_parallel)
    assert packed_parallel.resource_request["gpu_count"] == 3
    assert packed_parallel.resource_request["max_gpu_count"] == 3
    assert any(
        item.get("event_type") == "controller_plan_parallel_prefetch_armed"
        and item.get("reason") == "primary_prefetch_persisted_gpu_candidate"
        for item in campaign.events.read()
    )


def test_parallel_prefetch_uses_one_fast_controller_candidate(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    class Controller:
        provider_name = "vllm"
        model_name = "parallel-fast-controller"
        timeout_s = 300.0
        candidate_count = 4
        candidate_temperature = 0.25

        def plan(self, _context):
            raise AssertionError("the campaign stub should intercept the plan call")

    campaign.controller = Controller()
    captured = []

    def intercept(_parent, _training_calls, *, controller=None, **kwargs):
        captured.append(
            (
                int(controller.candidate_count),
                float(controller.candidate_temperature),
                float(controller.timeout_s),
                kwargs.get("planning_intent"),
            )
        )
        return None

    campaign._controller_plan = intercept
    handle = campaign._start_controller_prefetch(
        root,
        0,
        planning_intent="parallel_gpu_fill",
        operator_filter=("recovery_finetune",),
    )
    assert handle is not None
    handle["thread"].join(timeout=2.0)

    assert captured == [(1, 0.0, 180.0, "parallel_gpu_fill")]
    started = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "controller_plan_prefetch_started"
    ][-1]
    assert started["candidate_count"] == 1


def test_parallel_prefetch_rebases_stale_parent_system_id(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")
    campaign.systems.create(
        SystemCandidate.from_model_candidate(
            "S0056",
            root,
            parent_id="S0000",
            generation=1,
            status="candidate",
        )
    )

    class Controller:
        provider_name = "test"
        model_name = "parallel-stale-system-controller"

        def plan(self, context):
            evidence_ids = list(context.unconsumed_observation_ids) or [
                item["observation_id"] for item in context.observations
            ]
            return ExperimentPlan(
                experiment_id="exp_0001",
                parent_model_id="M0000",
                parent_system_id="S0000",
                diagnosis="parallel branch has stale system identity",
                objective="fill the evaluator overlap GPUs",
                hypothesis="same-parent system rebase preserves the branch",
                operator="recovery_finetune",
                operator_args={"training_steps": 1},
                expected_effects={"quality_score": "preserve"},
                risks=["parallel branch may be discarded"],
                required_budget={"wall_time_s": 1.0, "gpu_hours": 1.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="parallel overlap test",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=evidence_ids,
                resource_request={
                    "gpu_count": 2,
                    "min_gpu_count": 2,
                    "max_gpu_count": 2,
                    "elastic": True,
                    "distributed": True,
                    "exclusive": False,
                    "evaluation_workers": 1,
                    "on_unavailable": "wait",
                },
            )

    campaign.controller = Controller()
    plan = campaign._controller_plan(
        root,
        0,
        prefetch=True,
        planning_intent="parallel_gpu_fill",
    )

    assert plan is not None
    assert plan.parent_system_id == "S0056"
    assert any(
        item.get("event_type") == "controller_plan_parallel_system_rebased"
        for item in campaign.events.read()
    )
    primary_prefetch = campaign._controller_plan(root, 0, prefetch=True, planning_intent="primary")
    assert primary_prefetch is not None
    assert primary_prefetch.parent_system_id == "S0056"
    assert any(
        item.get("event_type") == "controller_plan_prefetch_system_rebased"
        for item in campaign.events.read()
    )
    strict = campaign._controller_plan(root, 0, prefetch=False, planning_intent="primary")
    assert strict is None
    assert "parent_system_id S0000 does not match current system S0056" in campaign.controller_trace[-1]["error"]


def test_pipeline_telemetry_exposes_overlap_wait_and_underutilized_cards(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.events.append(
        "controller_plan_prefetch_ready",
        {"elapsed_s": 2.5, "planning_intent": "evaluation_overlap"},
    )
    campaign.events.append(
        "controller_plan_overlap_ready",
        {"wait_s": 1.25, "experiment_id": "exp_0002"},
    )
    campaign.events.append(
        "controller_plan_parallel_prefetch_armed",
        {"reason": "cpu_only_worker_eager_parallel_prefetch"},
    )
    campaign.events.append(
        "speculative_worker_started",
        {"allocated_gpus": [2, 3]},
    )
    campaign.events.append(
        "training_power_sampled",
        {
            "summary": {
                "per_gpu": {
                    "0": {"power_w_avg": 297.0, "utilization_gpu_pct_avg": 95.0, "lane": "worker"},
                    "1": {"power_w_avg": 44.0, "utilization_gpu_pct_avg": 22.0, "lane": "idle"},
                }
            }
        },
    )

    telemetry = campaign._pipeline_telemetry()

    assert telemetry["prefetch_latency_s"]["last_s"] == 2.5
    assert telemetry["overlap_wait_s"]["last_s"] == 1.25
    assert telemetry["event_counts"]["controller_plan_parallel_prefetch_armed"] == 1
    assert telemetry["underutilized_gpu_indices"] == ["1"]
    assert telemetry["recent_speculative_worker_gpu_sets"][-1] == [2, 3]
    assert telemetry["power_target_w"] == 300.0
    assert telemetry["last_power_feedback"]["under_target_gpu_indices"] == ["1"]
    assert telemetry["last_power_feedback"]["per_gpu"]["0"]["lane"] == "worker"


def test_pipeline_telemetry_measures_evaluation_to_gpu_fill_gap(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.events.append("evaluation_started", {"model_id": "M0000"})
    campaign.events.append("speculative_worker_started", {"allocated_gpus": [1, 2, 3]})
    campaign.events.append("evaluation_completed", {"model_id": "M0000"})

    telemetry = campaign._pipeline_telemetry()

    assert telemetry["evaluation_gpu_fill_wait_s"]["samples"] == 1
    assert telemetry["evaluation_gpu_fill_wait_s"]["last_s"] >= 0.0
    assert telemetry["evaluation_gpu_fill_ready_count"] == 1
    assert telemetry["evaluation_gpu_fill_missing_count"] == 0


def test_pipeline_telemetry_counts_evaluation_without_gpu_fill(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.events.append("evaluation_started", {"model_id": "M0000"})
    campaign.events.append("speculative_worker_started", {"allocated_gpus": []})
    campaign.events.append("evaluation_completed", {"model_id": "M0000"})

    telemetry = campaign._pipeline_telemetry()

    assert telemetry["evaluation_gpu_fill_wait_s"]["samples"] == 0
    assert telemetry["evaluation_gpu_fill_ready_count"] == 0
    assert telemetry["evaluation_gpu_fill_missing_count"] == 1


def test_pipeline_telemetry_counts_bounded_evaluation_refill(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.events.append(
        "controller_plan_evaluation_refill_started",
        {"candidate_model_id": "M0001", "primary_experiment_id": "exp_0001"},
    )
    campaign.events.append(
        "controller_plan_evaluation_refill_ready",
        {"candidate_model_id": "M0001", "experiment_id": "exp_0002"},
    )

    telemetry = campaign._pipeline_telemetry()

    assert telemetry["evaluation_refill_started_count"] == 1
    assert telemetry["evaluation_refill_ready_count"] == 1
    assert telemetry["event_counts"]["controller_plan_evaluation_refill_ready"] == 1


def test_cpu_primary_prefetch_arms_parallel_gpu_plan_during_training(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        pipeline_enabled=True,
        pipeline_max_inflight=2,
    )
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    class Controller:
        provider_name = "test"
        model_name = "early-parallel-controller"
        timeout_s = 1.0

        def plan(self, context):
            parallel = context.planning_intent == "parallel_gpu_fill"
            evidence_ids = [item["observation_id"] for item in context.observations]
            primary = ExperimentPlan(
                experiment_id="exp_%04d" % (context.budget_state.used_iterations + 1),
                # Reproduce the real LLM behavior seen in the remote log: a
                # primary prefetch can echo the old parent even though its
                # context was built from the deterministic child state.
                parent_model_id=(
                    context.current_model_state.model_id if parallel else "M0000"
                ),
                parent_system_id="S0000",
                diagnosis="prepare an independent branch",
                objective="keep available cards useful during evaluation",
                hypothesis="the second branch removes the plan-generation gap",
                operator="recovery_finetune" if parallel else "prune_blocks",
                operator_args={"training_steps": 1} if parallel else {"ratio": 0.1},
                expected_effects={"quality_score": "preserve"},
                risks=["speculative branch is benchmarked independently"],
                required_budget={"wall_time_s": 1.0, "gpu_hours": 0.1, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="early parallel prefetch test",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=evidence_ids,
                resource_request=(
                    {
                        "gpu_count": 2,
                        "min_gpu_count": 2,
                        "max_gpu_count": 2,
                        "elastic": True,
                        "distributed": True,
                        "exclusive": False,
                        "evaluation_workers": 1,
                        "on_unavailable": "wait",
                    }
                    if parallel
                    else {
                        "gpu_count": 0,
                        "min_gpu_count": 0,
                        "max_gpu_count": 0,
                        "elastic": False,
                        "distributed": False,
                        "exclusive": False,
                        "evaluation_workers": 1,
                        "on_unavailable": "wait",
                    }
                ),
            )
            if not parallel:
                # A real OpenAI-compatible Controller exposes these locally
                # eligible alternatives after one batched n-way response.
                # The overlap path should reuse the GPU candidate instead of
                # opening a second parallel_gpu_fill request.
                self.last_eligible_candidates = [
                    primary,
                    replace(
                        primary,
                        operator="recovery_finetune",
                        operator_args={"training_steps": 1},
                        resource_request={
                            "gpu_count": 2,
                            "min_gpu_count": 2,
                            "max_gpu_count": 2,
                            "elastic": True,
                            "distributed": True,
                            "exclusive": False,
                            "evaluation_workers": 1,
                            "on_unavailable": "wait",
                        },
                    ),
                ]
            return primary

    campaign.controller = Controller()
    current_plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="current plan",
        objective="make a CPU-only primary successor",
        hypothesis="prune first",
        operator="prune_blocks",
        operator_args={"ratio": 0.1},
        expected_effects={"quality_score": "preserve"},
        risks=["quality regression"],
        required_budget={"wall_time_s": 1.0, "gpu_hours": 0.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="current plan",
        consumed_observation_ids=[],
        diagnosis_evidence=["goal:l40x4_h3_v1"],
        resource_request={
            "gpu_count": 0,
            "min_gpu_count": 0,
            "max_gpu_count": 0,
            "elastic": False,
            "distributed": False,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    primary = campaign._start_controller_prefetch(
        root,
        0,
    )
    arm_thread = campaign._arm_parallel_prefetch_after_primary(
        root,
        current_plan,
        "M0001",
        0,
        primary,
    )

    deadline = time.monotonic() + 2.0
    state = {}
    while time.monotonic() < deadline:
        state = campaign._load_campaign_state()
        if isinstance(state.get("parallel_prefetched_plan"), dict):
            break
        time.sleep(0.01)
    if arm_thread is not None:
        arm_thread.join(timeout=1.0)

    parallel = campaign._parallel_prefetched_plan(state)
    assert parallel is not None
    assert parallel.experiment_id == "exp_0003"
    assert parallel.parent_model_id == "M0001"
    assert parallel.operator == "recovery_finetune"
    assert any(
        item.get("event_type") == "controller_plan_parallel_prefetch_armed"
        for item in campaign.events.read()
    )
    assert any(
        item.get("event_type") == "controller_plan_prefetch_rebased_for_parallel"
        for item in campaign.events.read()
    )


def test_eager_parallel_gpu_prefetch_starts_with_cpu_only_worker(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        pipeline_enabled=True,
        pipeline_max_inflight=2,
    )
    campaign._register_candidates([])
    root = campaign.models.get("M0000")
    current_plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id=root.id,
        parent_system_id="S0000",
        diagnosis="current CPU-only worker",
        objective="prepare a GPU branch before evaluation",
        hypothesis="the GPU branch removes the evaluation idle window",
        operator="prune_blocks",
        operator_args={"ratio": 0.1},
        expected_effects={"quality_score": "preserve"},
        risks=["parallel branch may be rejected by its own gate"],
        required_budget={"wall_time_s": 1.0, "gpu_hours": 0.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="eager GPU-fill queue test",
        consumed_observation_ids=[],
        diagnosis_evidence=["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 0,
            "min_gpu_count": 0,
            "max_gpu_count": 0,
            "elastic": False,
            "distributed": False,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    calls = []

    def fake_start(*_args, **kwargs):
        calls.append(kwargs)
        return {"thread": None, "result": {}}

    campaign._start_controller_prefetch = fake_start
    job = campaign._arm_eager_parallel_gpu_prefetch(root, current_plan, "M0001", 0)

    assert job is not None
    assert calls == [
        {
            "source_plan": current_plan,
            "source_experiment_id": "exp_0001",
            "source_child_model_id": "M0001",
            "planning_intent": "parallel_gpu_fill",
            "operator_filter": ("recovery_finetune", "distill", "step_distill", "dmd2"),
            "prefetch_state_key": "parallel",
            "experiment_cursor": 2,
        }
    ]
    assert job["parallel_handle"]["thread"] is None
    assert job["done"].is_set()
    assert any(
        item.get("event_type") == "controller_plan_parallel_prefetch_armed"
        and item.get("reason") == "cpu_only_worker_eager_parallel_prefetch"
        for item in campaign.events.read()
    )


def test_evaluation_setup_arms_parallel_gpu_prefetch_before_callback(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        pipeline_enabled=True,
        pipeline_max_inflight=2,
    )
    campaign._register_candidates([])
    root = campaign.models.get("M0000")
    candidate = replace(root, created_by_experiment_id="exp_0001")
    calls = []

    def fake_start(_candidate, _training_calls, **kwargs):
        calls.append(kwargs)
        return {
            "thread": None,
            "training_calls": 2,
            "source_child_model_id": "M0000",
            "prefetch_state_key": "parallel",
        }

    campaign._start_controller_prefetch = fake_start
    handle = campaign._arm_evaluation_parallel_prefetch(candidate, 1, "M0000")

    assert handle is not None
    assert calls == [
        {
            "source_experiment_id": "exp_0001",
            "source_child_model_id": "M0000",
            "planning_intent": "parallel_gpu_fill",
            "operator_filter": ("recovery_finetune", "distill", "step_distill", "dmd2"),
            "prefetch_state_key": "parallel",
            "experiment_cursor": 2,
        }
    ]
    assert any(
        item.get("event_type") == "controller_plan_parallel_prefetch_armed"
        and item.get("reason") == "evaluation_setup_before_benchmark_callback"
        for item in campaign.events.read()
    )


def test_primary_controller_batch_is_consumed_before_second_gpu_plan_call(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        pipeline_enabled=True,
        pipeline_max_inflight=2,
    )
    campaign._register_candidates([])
    root = campaign.models.get("M0000")
    primary = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id=root.id,
        parent_system_id="S0000",
        diagnosis="reduce memory first",
        objective="prune structural blocks",
        hypothesis="a bounded prune reduces resident memory",
        operator="prune_blocks",
        operator_args={"ratio": 0.1},
        expected_effects={"quality_score": "preserve"},
        risks=["quality regression"],
        required_budget={"wall_time_s": 1.0, "gpu_hours": 0.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="CPU primary with a batched GPU sibling",
        consumed_observation_ids=[],
        diagnosis_evidence=["goal:l40x4_h3_v1"],
        resource_request={
            "gpu_count": 0,
            "min_gpu_count": 0,
            "max_gpu_count": 0,
            "elastic": False,
            "distributed": False,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "replan",
        },
    )
    parallel = replace(
        primary,
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        required_budget={"wall_time_s": 1.0, "gpu_hours": 0.5, "controller_calls": 0},
        resource_request={
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    narrow = replace(
        parallel,
        experiment_id="exp_narrow",
        resource_request={
            **parallel.resource_request,
            "gpu_count": 2,
            "max_gpu_count": 2,
        },
    )
    campaign._last_primary_controller_batch = {
        "experiment_id": primary.experiment_id,
        "parent_model_id": primary.parent_model_id,
        "training_calls": 0,
        "candidates": (primary, narrow, parallel),
    }

    selected = campaign._consume_primary_batch_parallel_candidate(primary, 0)

    assert selected is not None
    assert selected.experiment_id == "exp_0002"
    assert selected.parent_model_id == root.id
    assert selected.operator == "recovery_finetune"
    assert selected.resource_request["max_gpu_count"] == 4
    assert campaign._last_primary_controller_batch is None
    assert any(
        item.get("event_type") == "controller_plan_parallel_reused"
        and item.get("reason") == "primary_batched_candidate_before_cpu_worker"
        for item in campaign.events.read()
    )


def test_prefetch_cursor_advances_past_inflight_source_plan(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    context = campaign._controller_context(root, 1, experiment_cursor=7)

    assert context.budget_state.used_iterations == 7


def test_parallel_gpu_fill_context_filters_out_cpu_operators(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    context = campaign._controller_context(
        root,
        0,
        planning_intent="parallel_gpu_fill",
        operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
    )

    assert context.planning_intent == "parallel_gpu_fill"
    assert {item["name"] for item in context.available_operators} == {
        "recovery_finetune",
        "distill",
        "step_distill",
        "dmd2",
    }
    assert "parallel_gpu_fill" in context.campaign_summary["planning_intent"]


def test_controller_context_excludes_live_compute_from_gpu_capacity(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    class SnapshotScheduler:
        gpu_count = 4
        min_free_memory_mb = 26624
        reserved_gpu_indices = ()
        last_compute_gpu_indices = (0, 2)
        last_gpu_telemetry = {
            0: {"power_w": 290.0, "utilization_gpu_pct": 96.0},
            1: {"power_w": 120.0, "utilization_gpu_pct": 44.0},
            2: {"power_w": 285.0, "utilization_gpu_pct": 92.0},
            3: {"power_w": 0.0, "utilization_gpu_pct": 0.0},
        }

        def snapshot(self):
            return ({index: (0, 46080) for index in range(4)}, "GPU-0, 100, 4096\nGPU-2, 101, 4096\n")

        def _controller_reserved_gpu_indices(self):
            return (3,)

        def meets_memory_waterline(self, _index, _snapshot):
            return True

    campaign.scheduler = SnapshotScheduler()
    context = campaign._controller_context(root, 0)
    capacity = context.campaign_summary["gpu_capacity"]

    assert capacity["free_above_waterline_indices"] == [1]
    assert capacity["compute_process_gpu_indices"] == [0, 2]
    assert capacity["compute_process_mapping_unknown"] is False
    assert capacity["per_gpu"]["1"]["power_w"] == 120.0
    assert capacity["per_gpu"]["1"]["utilization_gpu_pct"] == 44.0


def test_parallel_gpu_fill_plan_is_rejected_if_controller_ignores_operator_filter(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    root = campaign.models.get("M0000")

    class CpuController:
        provider_name = "test"
        model_name = "cpu-controller"

        def plan(self, context):
            return ExperimentPlan(
                experiment_id="exp_%04d" % (context.budget_state.used_iterations + 1),
                parent_model_id=context.current_model_state.model_id,
                parent_system_id="S0000",
                diagnosis="bad fill plan",
                objective="should be rejected",
                hypothesis="a CPU plan cannot fill GPU capacity",
                operator="prune_blocks",
                operator_args={"ratio": 0.1},
                expected_effects={"quality_score": "preserve"},
                risks=["none"],
                required_budget={"wall_time_s": 1.0, "gpu_hours": 0.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="test filter",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids) or ["goal:l40x4_h3_v1"],
                resource_request={
                    "gpu_count": 0,
                    "min_gpu_count": 0,
                    "max_gpu_count": 0,
                    "elastic": False,
                    "distributed": False,
                    "exclusive": False,
                    "evaluation_workers": 1,
                    "on_unavailable": "replan",
                },
            )

    campaign.controller = CpuController()
    assert campaign._controller_plan(
        root,
        0,
        planning_intent="parallel_gpu_fill",
        operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
        prefetch=True,
    ) is None
    assert campaign.controller_prefetch_trace[-1]["status"] == "rejected"
    assert "operator_not_allowed" in campaign.controller_prefetch_trace[-1]["error"]


def test_prefetch_context_models_deterministic_child_state(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    parent = ModelCandidate(
        "M0000",
        None,
        0,
        "fake://M0000",
        replace(ModelState.fake_baseline(), sampling_steps=32),
        None,
        "baseline",
    )
    source_plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="reduce sampling steps",
        objective="halve sampling steps",
        hypothesis="step distillation preserves quality",
        operator="step_distill",
        operator_args={"target_steps": 16},
        expected_effects={"quality_score": "preserve"},
        risks=["quality regression"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="predict the next worker state",
        consumed_observation_ids=[],
        diagnosis_evidence=[],
        resource_request={"gpu_count": 2},
    )

    predicted = campaign._prefetch_parent_candidate(parent, source_plan, "M0001")

    assert predicted.id == "M0001"
    assert predicted.parent_id == "M0000"
    assert predicted.state.sampling_steps == 16
    assert predicted.state.algorithm_state["source_steps"] == 32
    assert predicted.state.provenance["prefetch_prediction"] is True


def test_prefetched_plan_is_reused_at_the_next_training_boundary(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)

    class NoPlanCallController:
        provider_name = "test"
        model_name = "prefetch-only"

        def plan(self, _context):
            raise AssertionError("a current prefetched plan should avoid a second LLM call")

    campaign.controller = NoPlanCallController()
    plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="reuse persisted plan",
        objective="continue without a controller gap",
        hypothesis="the worker-overlapped plan is still current",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["speculative plan may be invalidated"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="reuse the validated overlap plan",
        consumed_observation_ids=[],
        diagnosis_evidence=["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    campaign._save_campaign_state(
        {
            "consumed_observation_ids": [],
            "prefetched_plan": {"plan": plan.to_dict()},
            "prefetched_parent_model_id": "M0000",
            "prefetched_training_calls": 0,
        }
    )

    record = campaign._train_one(parent, 0)

    assert record is not None
    assert campaign.controller_trace[-1]["prefetch_reuse"] is True
    assert any(
        item.get("event_type") == "controller_plan_prefetch_reused"
        for item in campaign.events.read()
    )


def test_run_loop_does_not_stop_before_persisted_plan_boundary(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    campaign.controller = SimpleNamespace(provider_name="test", model_name="test-controller")
    campaign.models.initialize(
        ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    )
    campaign._save_campaign_state({
        "prefetched_plan": {"plan": {"experiment_id": "exp_0002"}},
    })
    calls = []

    def fake_run(**_kwargs):
        calls.append(True)
        trace = [] if len(calls) == 1 else [{"status": "validated"}]
        return CampaignResult(
            "completed",
            campaign.models.active_id,
            {},
            {"controller": {"calls": 0, "trace": trace, "prefetch_trace": []}},
        )

    campaign.run = fake_run
    result = campaign.run_loop(max_iterations=2, resource_poll_interval_s=0)

    assert len(calls) == 2
    assert result.report["loop"]["iterations"] == 2
    assert any(
        item.get("event_type") == "loop_continuation_pending_plan"
        for item in campaign.events.read()
    )


def test_speculative_plan_rebases_only_to_its_source_child(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._register_candidates([])
    parent = campaign.models.get("M0000")
    child = ModelCandidate(
        "M0001",
        "M0000",
        1,
        "fake://M0001",
        replace(
            ModelState.fake_baseline(),
            model_id="M0001",
            parent_model_id="M0000",
            checkpoint_path="fake://M0001",
        ),
        "exp_0001",
        "candidate",
    )
    campaign.models.create(child)
    plan = ExperimentPlan(
        experiment_id="exp_0002",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="prepare the next branch",
        objective="overlap training with evaluation",
        hypothesis="the source child remains the next safe parent",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["speculative result is discarded if the source child is invalid"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="test speculative lineage",
        consumed_observation_ids=[],
        diagnosis_evidence=["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    campaign._save_campaign_state(
        {
            "consumed_observation_ids": [],
            "prefetched_plan": {"plan": plan.to_dict()},
            "prefetched_parent_model_id": parent.id,
            "prefetched_source_child_model_id": child.id,
            "prefetched_training_calls": 0,
        }
    )

    rebased = campaign._speculative_plan_for_candidate(child, 0)

    assert rebased is not None
    assert rebased.parent_model_id == "M0001"
    assert rebased.experiment_id == "exp_0002"
    unrelated = ModelCandidate(
        "M0002",
        "M0001",
        2,
        "fake://M0002",
        replace(
            ModelState.fake_baseline(),
            model_id="M0002",
            parent_model_id="M0001",
            checkpoint_path="fake://M0002",
        ),
        "exp_0002",
        "candidate",
    )
    assert campaign._speculative_plan_for_candidate(unrelated, 0) is None


def test_speculative_worker_uses_non_comfyui_gpus_without_promoting_child(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    campaign._register_candidates([])
    parent = campaign.models.get("M0000")
    plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="speculative worker test",
        objective="use free cards during benchmark",
        hypothesis="two remaining cards can train the next branch",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["discard on failed current evaluation"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="speculative worker test",
        consumed_observation_ids=[],
        diagnosis_evidence=["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    planned, effective = campaign._effective_worker_request(plan)
    assert planned["max_gpu_count"] == 4
    assert effective["max_gpu_count"] == 3
    assert effective["elastic"] is True
    assert effective["exclusive"] is False
    full_card_plan = replace(
        plan,
        resource_request={
            **plan.resource_request,
            "gpu_count": 4,
        },
    )
    planned_full, effective_full = campaign._effective_worker_request(full_card_plan)
    assert planned_full["gpu_count"] == 4
    assert planned_full["max_gpu_count"] == 4
    assert effective_full["gpu_count"] == 4
    assert effective_full["max_gpu_count"] == 4
    # Simulate ComfyUI and the one-card Controller reservation.  The worker
    # must choose exactly the two remaining cards in this fake snapshot.
    campaign.scheduler.reserve_gpu(0)
    campaign.scheduler.reserve_gpu(1)
    release_reasons = []
    campaign._request_controller_release = lambda reason, **_kwargs: release_reasons.append(reason)
    handle = campaign._start_speculative_worker(
        parent,
        plan,
        preserve_controller_lane=True,
    )

    assert handle is not None
    handle["thread"].join(timeout=2.0)
    assert not handle["thread"].is_alive()
    assert campaign.models.active_id == "M0000"
    state = campaign._load_speculative_state()
    assert state["status"] == "completed"
    started = [item for item in campaign.events.read() if item.get("event_type") == "speculative_worker_started"]
    assert started[-1]["allocated_gpus"] == [2, 3]
    lane_events = [item for item in campaign.events.read() if item.get("event_type") == "lane_allocation"]
    worker_lanes = [
        lane
        for lane in lane_events[-1]["lanes"]
        if lane.get("lane") == "worker"
    ]
    assert lane_events[-1]["stage"] == "training"
    assert worker_lanes[-1]["allocated_gpus"] == [2, 3]
    assert any("--nproc_per_node=2" in command for command in campaign.ssh.commands)
    assert release_reasons == []


def test_evaluation_speculative_worker_uses_three_cards_after_controller_handoff(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    campaign._register_candidates([])
    parent = campaign.models.get("M0000")
    plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="evaluation overlap test",
        objective="use all non-evaluator cards",
        hypothesis="a ready successor does not need a resident Controller card",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["discard if evaluation rejects the branch"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="evaluation overlap test",
        consumed_observation_ids=[],
        diagnosis_evidence=["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    campaign._active_evaluation_gpu_indices = (0,)
    campaign._comfyui_lease_active = lambda: True
    campaign.scheduler.reserve_gpu(0)
    campaign.scheduler.allocation_recheck_s = 0
    controller_released = {"value": False}
    campaign.scheduler._controller_reserved_gpu_indices = (
        lambda: () if controller_released["value"] else (1,)
    )

    def release_controller(_reason, **_kwargs):
        controller_released["value"] = True
        return {"status": "released"}

    campaign._request_controller_release = release_controller
    preserve = campaign._preserve_controller_lane_for_speculative_worker()
    handle = campaign._start_speculative_worker(
        parent,
        plan,
        preserve_controller_lane=preserve,
    )

    assert preserve is False
    assert handle is not None
    handle["thread"].join(timeout=2.0)
    assert not handle["thread"].is_alive()
    started = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "speculative_worker_started"
    ]
    assert started[-1]["allocated_gpus"] == [1, 2, 3]
    assert any(
        "--nproc_per_node=3" in command
        for command in campaign.ssh.commands
    )


def test_cpu_worker_cannot_release_concurrent_speculative_gpu_lease(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    released = []

    class SharedLeaseScheduler:
        lease_active = True

        @staticmethod
        def normalize_request(request):
            return {
                "gpu_count": int(request.get("gpu_count", 0)),
                "min_gpu_count": int(request.get("min_gpu_count", request.get("gpu_count", 0))),
                "max_gpu_count": int(request.get("max_gpu_count", request.get("gpu_count", 0))),
                "elastic": bool(request.get("elastic", False)),
            }

        def acquire(self, _request):
            raise AssertionError("a second worker must not overwrite the active shared lease")

        def release(self):
            released.append(True)
            self.lease_active = False
            return {"status": "deleted"}

    campaign.scheduler = SharedLeaseScheduler()
    campaign._worker_lease_experiment_id = "exp_parallel"

    campaign._release_worker_gpu_lease("exp_cpu", "worker_completed")

    assert released == []
    assert campaign._worker_lease_experiment_id == "exp_parallel"
    skipped = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "worker_gpu_lease_release_skipped"
    ]
    assert skipped[-1]["lease_owner_experiment_id"] == "exp_parallel"

    blocked = campaign._acquire_worker_gpu_lease(
        "exp_primary",
        {
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
        },
    )
    assert blocked.status == "wait"
    assert "exp_parallel" in blocked.reason

    campaign._release_worker_gpu_lease("exp_parallel", "speculative_worker_completed")
    assert released == [True]
    assert campaign._worker_lease_experiment_id is None


def test_speculative_finalization_does_not_block_benchmark_cache_release(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    finished = threading.Event()

    def long_checkpoint_write():
        time.sleep(0.25)
        finished.set()

    worker = threading.Thread(target=long_checkpoint_write)
    worker.start()
    handle = {
        "thread": worker,
        "status": "running",
        "experiment_id": "exp_0001",
        "child_model_id": "M0001",
        "parent_model_id": "M0000",
        "worker_timeout_s": 3600.0,
    }
    started = time.monotonic()

    result = campaign._finish_speculative_worker(
        handle,
        keep=True,
        reason="preserve_speculative_child_for_next_evaluation",
    )

    elapsed = time.monotonic() - started
    assert result is None
    assert worker.is_alive()
    assert elapsed < 0.15
    assert campaign._load_speculative_state()["finalization_keep"] is True
    worker.join(timeout=1.0)
    assert finished.is_set()


def test_prefetched_plan_rebases_when_the_worker_child_becomes_active(tmp_path):
    results = {
        "/srv/harness/campaign/trainer_result_m0002.json": {
            "result": _training_result(
                "M0002",
                "M0001",
                "/srv/models/results/M0002.safetensors",
                "recovery_finetune",
                32,
            )
        }
    }
    campaign = build_campaign(tmp_path, results, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    root = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(root)
    child_state = replace(
        ModelState.fake_baseline("M0001"),
        parent_model_id="M0000",
        checkpoint_path="fake://M0001",
    )
    child = ModelCandidate("M0001", "M0000", 1, "fake://M0001", child_state, "exp_0001", "candidate")
    campaign.models.create(child)
    campaign.models.set_active(child.id)

    class NoPlanCallController:
        provider_name = "test"
        model_name = "prefetch-rebase-only"

        def plan(self, _context):
            raise AssertionError("the speculative plan should be rebased without a new LLM call")

    campaign.controller = NoPlanCallController()
    plan = ExperimentPlan(
        experiment_id="exp_0002",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="reuse after promotion",
        objective="continue directly from the promoted worker child",
        hypothesis="the worker child is the expected next parent",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["speculative plan may be invalidated"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="rebase the validated plan to the promoted child",
        consumed_observation_ids=[],
        diagnosis_evidence=["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    campaign._save_campaign_state(
        {
            "consumed_observation_ids": [],
            "prefetched_plan": {"plan": plan.to_dict()},
            "prefetched_parent_model_id": "M0000",
            "prefetched_training_calls": 0,
            "prefetched_source_experiment_id": "exp_0001",
            "prefetched_source_child_model_id": "M0001",
        }
    )

    record = campaign._train_one(child, 0)

    assert record is not None
    assert record.child_model_id == "M0002"
    assert campaign.controller_trace[-1]["prefetch_reuse"] is True
    assert campaign.controller_trace[-1]["prefetch_rebased"] is True
    assert campaign.controller_trace[-1]["plan"]["parent_model_id"] == "M0001"
    events = [item for item in campaign.events.read() if item.get("event_type") == "controller_plan_prefetch_reused"]
    assert events[-1]["reason"] == "worker_child_promoted_without_replan_request"


def test_remote_campaign_loop_reasks_controller_until_goal_or_bound(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    calls = []

    def cycle(*, resume, max_experiments, split):
        calls.append((resume, max_experiments, split))
        return CampaignResult(
            "completed",
            "M0000",
            {},
            {"controller": {"calls": 1, "trace": [{"status": "validated"}]}},
        )

    campaign.run = cycle
    campaign._goal_satisfied = lambda split: len(calls) >= 2
    result = campaign.run_loop(resume=False, max_iterations=4, split="sanity")

    assert len(calls) == 2
    assert calls == [(False, 1, "sanity"), (True, 1, "sanity")]
    assert result.report["goal"]["goal_id"] == "l40x4_h3_v1"
    assert result.report["loop"]["stop_reason"] == "target_satisfied"


def test_remote_campaign_loop_stops_at_iteration_boundary(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    stop_file = tmp_path / "stop-requested"
    calls = []

    def cycle(*, resume, max_experiments, split):
        calls.append((resume, max_experiments, split))
        stop_file.touch()
        return CampaignResult(
            "completed",
            "M0000",
            {},
            {"controller": {"calls": 1, "trace": [{"status": "validated"}]}},
        )

    campaign.run = cycle
    campaign._goal_satisfied = lambda split: False
    result = campaign.run_loop(
        resume=False,
        max_iterations=4,
        split="sanity",
        resource_poll_interval_s=0,
        stop_file=stop_file,
    )

    assert calls == [(False, 1, "sanity")]
    assert result.status == "stop_requested"
    assert result.report["loop"]["stop_reason"] == "stop_requested"
    names = [item["event_type"] for item in campaign.events.read()]
    assert "campaign_stop_boundary_reached" in names


def test_dependency_waits_do_not_consume_optimization_iterations(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    campaign.controller = SimpleNamespace(
        provider_name="vllm",
        model_name="test-controller",
        remote_port=8000,
    )
    preflight_calls = []
    run_calls = []

    def preflight():
        preflight_calls.append(True)
        if len(preflight_calls) <= 2:
            raise ControllerUnavailableError("controller is still loading")

    def fake_run(**_kwargs):
        run_calls.append(True)
        return CampaignResult(
            "completed",
            "M0000",
            {},
            {"controller": {"calls": 1, "trace": [{"status": "validated"}]}},
        )

    campaign._remote_controller_preflight = preflight
    campaign._remote_comfyui_preflight = lambda: None
    campaign.run = fake_run
    campaign._goal_satisfied = lambda _split: False

    result = campaign.run_loop(
        resume=False,
        max_iterations=1,
        split="sanity",
        resource_poll_interval_s=0,
    )

    assert len(preflight_calls) == 3
    assert len(run_calls) == 1
    assert result.report["loop"]["iterations"] == 1
    assert len(result.report["loop"]["dependency_waits"]) == 2
    state = campaign._load_campaign_state()
    assert state["pipeline"]["stage"] == "waiting"
    assert state["pipeline_waiting_on"] == "controller"


def test_controller_preflight_uses_only_explicit_healthy_fallback_port(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})

    class FallbackSSH(_FakeSSH):
        def run(self, command, **kwargs):
            command = tuple(command)
            if any("/v1/models" in item for item in command):
                endpoint = next(item for item in command if "/v1/models" in item)
                if ":8000/" in endpoint:
                    return SimpleNamespace(stdout="", stderr="connection refused", returncode=7)
                return SimpleNamespace(
                    stdout=json.dumps({"data": [{"id": "qwen3.5-controller"}]}),
                    stderr="",
                    returncode=0,
                )
            return super().run(command, **kwargs)

    campaign.ssh = FallbackSSH({})
    controller = SimpleNamespace(
        provider_name="vllm",
        model_name="qwen3.5-controller",
        remote_port=8000,
        preferred_remote_port=8000,
        fallback_remote_ports=(8001,),
    )

    campaign._remote_controller_preflight(controller)

    assert controller.remote_port == 8001
    events = [item for item in campaign.events.read() if item.get("event_type") == "controller_preflight"]
    assert events[-1]["fallback_selected"] is True
    assert events[-1]["attempted_ports"] == [8000, 8001]


def test_pipeline_waiting_event_is_idempotent_for_repeated_polling(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign._set_pipeline_waiting("controller", current_model_id="M0000")
    campaign._set_pipeline_waiting("controller", current_model_id="M0000")

    waits = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "pipeline_waiting"
    ]
    assert len(waits) == 1


def test_pending_controller_plan_retries_without_a_second_controller_call(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)

    class Controller:
        provider_name = "test"
        model_name = "controller"

        def __init__(self):
            self.calls = 0

        def plan(self, context):
            self.calls += 1
            return ExperimentPlan(
                experiment_id="exp_0001",
                parent_model_id=context.current_model_state.model_id,
                diagnosis="resource test",
                objective="exercise queued training",
                hypothesis="a later retry can use newly available GPUs",
                operator="recovery_finetune",
                operator_args={"training_steps": 1},
                expected_effects={"quality_score": "preserve"},
                risks=["resource contention"],
                required_budget={"wall_time_s": 2.0, "gpu_hours": 4.0, "controller_calls": 0},
                acceptance={"max_quality_drop": 0.05},
                stop_conditions={"critical_regression": True},
                rationale="controller resource test",
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids),
                resource_request={
                    # This test exercises the normal overlap mode.  A
                    # validated gpu_count=4 request is reserved for the
                    # explicit full-card handoff test below.
                    "gpu_count": 3,
                    "min_gpu_count": 2,
                    "max_gpu_count": 4,
                    "elastic": True,
                    "distributed": True,
                    "exclusive": False,
                    "evaluation_workers": 1,
                    "on_unavailable": "wait",
                },
            )

    class Scheduler:
        def __init__(self):
            self.calls = 0

        @property
        def controller_reserved_gpu_indices(self):
            # A TP1 Controller on one card can coexist with the worker's
            # two-card minimum; releasing it would only create a reload gap.
            return (3,)

        def normalize_request(self, value):
            return {
                **dict(value),
                "min_gpu_count": value.get("min_gpu_count", value["gpu_count"]),
                "max_gpu_count": value.get("max_gpu_count", value["gpu_count"]),
                "elastic": value.get("elastic", False),
            }

        def acquire(self, request):
            self.calls += 1
            if self.calls == 1:
                return ResourceDecision("wait", 3, (), "busy", 2, 3, True, 1)
            return ResourceDecision("ready", 3, (1, 3), "elastic allocation", 2, 3, True, 2)

    controller = Controller()
    campaign.controller = controller
    campaign.scheduler = Scheduler()
    first = campaign._train_one(parent, 0)
    assert first is None
    assert controller.calls == 1
    assert campaign._load_campaign_state()["pending_plan"]["experiment_id"] == "exp_0001"
    assert any(
        item.get("event_type") == "controller_release_skipped"
        for item in campaign.events.read()
    )

    directive, written = submit_directive(
        campaign.observations,
        "下一轮先尝试降低 peak_memory",
        directive_id="pending-goal-001",
    )
    assert written is True
    second = campaign._train_one(parent, 0)
    assert second is not None
    assert controller.calls == 2
    assert "pending_plan" not in campaign._load_campaign_state()
    assert directive.to_observation().observation_id in campaign.controller_trace[-1]["plan"]["consumed_observation_ids"]
    assert any("--nproc_per_node=2" in command for command in campaign.ssh.commands)
    assert any(value.get("world_size") == 2 for value in campaign.ssh.writes.values() if isinstance(value, dict))


def test_full_card_worker_hands_off_controller_before_acquire(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    parent = ModelCandidate("M0000", None, 0, "fake://M0000", ModelState.fake_baseline(), None, "baseline")
    campaign.models.initialize(parent)

    plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        diagnosis="use a full-card training boundary",
        objective="exercise the four-GPU handoff",
        hypothesis="a successor plan is prepared before all cards train",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["resource contention"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 4.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="full-card handoff test",
        consumed_observation_ids=[],
        diagnosis_evidence=[],
        resource_request={
            "gpu_count": 4,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )

    class Controller:
        provider_name = "test"
        model_name = "controller"

        def __init__(self):
            self.calls = 0

        def plan(self, context):
            self.calls += 1
            return replace(
                plan,
                consumed_observation_ids=list(context.unconsumed_observation_ids),
                diagnosis_evidence=list(context.unconsumed_observation_ids),
            )

    class Scheduler:
        gpu_count = 4
        controller_reserved_gpu_indices = (3,)
        reserved_gpu_indices = ()
        lease_active = False

        def normalize_request(self, value):
            return {
                **dict(value),
                "min_gpu_count": value.get("min_gpu_count", value["gpu_count"]),
                "max_gpu_count": value.get("max_gpu_count", value["gpu_count"]),
                "elastic": value.get("elastic", False),
            }

        def acquire(self, request):
            assert request["gpu_count"] == 4
            assert request["max_gpu_count"] == 4
            return ResourceDecision("ready", 4, (0, 1, 2, 3), "full-card allocation", 2, 4, True, 4)

    controller = Controller()
    campaign.controller = controller
    campaign.scheduler = Scheduler()
    order = []
    campaign._start_controller_prefetch = lambda *args, **kwargs: order.append("prefetch") or {"fake": True}
    campaign._finish_controller_prefetch = (
        lambda handle, **kwargs: order.append("finish_prefetch") or plan
    )
    campaign._request_controller_release = (
        lambda reason, **kwargs: order.append(("release", reason, kwargs)) or {"status": "released"}
    )

    result = campaign._train_one(parent, 0)

    assert result is not None
    assert controller.calls == 1
    assert order[0] == "prefetch"
    assert order[1] == "finish_prefetch"
    assert order[2][0:2] == ("release", "training_gpu_allocation")
    assert order[2][2]["full_card_training"] is True
    resource_events = [
        item for item in campaign.events.read() if item.get("event_type") == "resource_scheduled"
    ]
    assert resource_events[-1]["full_card_training"] is True
    worker_events = [
        item for item in campaign.events.read() if item.get("event_type") == "worker_started"
    ]
    assert worker_events[-1]["full_card_training"] is True
    assert worker_events[-1]["allocated_gpus"] == [0, 1, 2, 3]
    assert not any(
        item.get("event_type") == "controller_release_skipped"
        for item in campaign.events.read()
    )


def test_waiting_worker_plan_is_replanned_after_bounded_attempts(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        resource_wait_replan_after=2,
    )
    campaign._register_candidates([])
    parent = campaign.models.get("M0000")
    context = campaign._controller_context(parent, 0)
    plan = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="wait only within a bounded resource window",
        objective="continue optimization without an infinite GPU wait",
        hypothesis="a fresh Controller decision can select a safe available lane",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["foreign GPU contention"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="replan after the resource wait budget is exhausted",
        consumed_observation_ids=list(context.unconsumed_observation_ids),
        diagnosis_evidence=list(context.unconsumed_observation_ids) or ["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 3,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    campaign._save_campaign_state(
        {
            "consumed_observation_ids": list(context.unconsumed_observation_ids),
            "pending_plan": plan.to_dict(),
            "pending_parent_model_id": "M0000",
            "pending_training_calls": 0,
            "pending_attempts": 1,
        }
    )

    class AlwaysWaitingScheduler:
        gpu_count = 4
        min_free_memory_mb = 26624
        reserved_gpu_indices = ()

        @property
        def controller_reserved_gpu_indices(self):
            return (3,)

        def snapshot(self):
            return ({index: (0, 46080) for index in range(4)}, "")

        def _controller_reserved_gpu_indices(self):
            return (3,)

        def meets_memory_waterline(self, _index, _snapshot):
            return True

        def normalize_request(self, value):
            return {
                **dict(value),
                "min_gpu_count": value.get("min_gpu_count", value["gpu_count"]),
                "max_gpu_count": value.get("max_gpu_count", value["gpu_count"]),
                "elastic": value.get("elastic", False),
            }

        def acquire(self, request):
            return ResourceDecision("wait", int(request["gpu_count"]), (), "foreign job remains active", 2, 3, True, 0)

    campaign.scheduler = AlwaysWaitingScheduler()
    campaign._review_now = lambda *_args, **_kwargs: None

    assert campaign._train_one(parent, 0) is None
    state = campaign._load_campaign_state()
    assert "pending_plan" not in state
    event = [
        item
        for item in campaign.events.read()
        if item.get("event_type") == "resource_wait_replan_requested"
    ][-1]
    assert event["attempts"] == 2
    assert event["threshold"] == 2
    assert campaign.controller_trace[-1]["execution_status"] == "resource_unavailable_replan"


def test_bounded_gpu_wait_persists_cpu_recovery_intent_for_next_controller_call(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(
        campaign.config,
        worker=replace(campaign.config.worker, enabled=True),
        resource_wait_replan_after=2,
    )
    campaign._register_candidates([])
    parent = campaign.models.get("M0000")
    context = campaign._controller_context(parent, 0)
    pending = ExperimentPlan(
        experiment_id="exp_0001",
        parent_model_id="M0000",
        parent_system_id="S0000",
        diagnosis="distributed worker cannot be placed",
        objective="wait for a safe training allocation",
        hypothesis="foreign jobs may release GPUs later",
        operator="recovery_finetune",
        operator_args={"training_steps": 1},
        expected_effects={"quality_score": "preserve"},
        risks=["foreign GPU contention"],
        required_budget={"wall_time_s": 2.0, "gpu_hours": 1.0, "controller_calls": 0},
        acceptance={"max_quality_drop": 0.05},
        stop_conditions={"critical_regression": True},
        rationale="resource recovery test",
        consumed_observation_ids=list(context.unconsumed_observation_ids),
        diagnosis_evidence=list(context.unconsumed_observation_ids) or ["obs-goal-l40x4_h3_v1"],
        resource_request={
            "gpu_count": 3,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    )
    campaign._save_campaign_state(
        {
            "consumed_observation_ids": list(context.unconsumed_observation_ids),
            "pending_plan": pending.to_dict(),
            "pending_parent_model_id": "M0000",
            "pending_training_calls": 0,
            "pending_attempts": 1,
        }
    )

    class AlwaysWaitingScheduler:
        gpu_count = 4
        min_free_memory_mb = 26624
        reserved_gpu_indices = ()

        @property
        def controller_reserved_gpu_indices(self):
            return (3,)

        def snapshot(self):
            return ({index: (0, 46080) for index in range(4)}, "")

        def _controller_reserved_gpu_indices(self):
            return (3,)

        def meets_memory_waterline(self, _index, _snapshot):
            return True

        def normalize_request(self, value):
            return {
                **dict(value),
                "min_gpu_count": value.get("min_gpu_count", value["gpu_count"]),
                "max_gpu_count": value.get("max_gpu_count", value["gpu_count"]),
                "elastic": value.get("elastic", False),
            }

        def acquire(self, request):
            return ResourceDecision("wait", int(request["gpu_count"]), (), "foreign job remains active", 2, 3, True, 0)

    campaign.scheduler = AlwaysWaitingScheduler()
    campaign._review_now = lambda *_args, **_kwargs: None
    calls = []

    def controller_plan(parent_arg, training_calls_arg, **kwargs):
        calls.append((parent_arg.id, training_calls_arg, kwargs))
        return None

    campaign._controller_plan = controller_plan

    assert campaign._train_one(parent, 0) is None
    state = campaign._load_campaign_state()
    assert state["resource_replan_intent"]["source_experiment_id"] == "exp_0001"
    assert campaign._train_one(parent, 0) is None
    assert calls == [
        (
            "M0000",
            0,
            {
                "planning_intent": "resource_recovery_cpu",
                "operator_filter": ("prune_blocks", "quantize"),
            },
        )
    ]
    assert any(
        item.get("event_type") == "controller_resource_recovery_requested"
        for item in campaign.events.read()
    )


def test_worker_command_uses_actual_elastic_gpu_count(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    command = campaign._worker_command(
        "recovery_finetune",
        "/srv/harness/campaign/config.json",
        "/srv/harness/campaign/request.json",
        "/srv/harness/campaign/result.json",
        allocated_gpu_count=2,
    )
    assert "--nproc_per_node=2" in command
    assert "--nproc_per_node=4" not in command


def test_concurrent_speculative_workers_reserve_distinct_child_ids(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})

    first = campaign._reserve_next_child_id()
    second = campaign._reserve_next_child_id()

    assert first != second
    assert {first, second} == {"M0000", "M0001"}
    campaign._release_child_id_reservation(first)
    campaign._release_child_id_reservation(second)


def test_loop_retries_a_waiting_plan_at_the_next_boundary(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    calls = []

    def cycle(*, resume, max_experiments, split):
        calls.append((resume, max_experiments, split))
        if len(calls) == 1:
            trace = {
                "controller_call": True,
                "status": "validated",
                "execution_status": "waiting_for_resources",
                "plan": {"resource_request": {"on_unavailable": "wait"}},
            }
        else:
            trace = {"controller_call": True, "status": "validated"}
        return CampaignResult("completed", "M0000", {}, {"controller": {"calls": 1, "trace": [trace]}})

    campaign.run = cycle
    campaign._goal_satisfied = lambda split: False
    result = campaign.run_loop(resume=False, max_iterations=2, split="sanity", resource_poll_interval_s=0)

    assert len(calls) == 3
    assert calls == [(False, 1, "sanity"), (True, 1, "sanity"), (True, 1, "sanity")]
    assert result.report["loop"]["iterations"] == 2


def test_loop_reasks_controller_when_resource_policy_requests_replan(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    calls = []

    def cycle(*, resume, max_experiments, split):
        calls.append((resume, max_experiments, split))
        if len(calls) == 1:
            trace = {
                "controller_call": True,
                "status": "validated",
                "execution_status": "resource_unavailable_replan",
                "plan": {"resource_request": {"on_unavailable": "replan"}},
            }
        else:
            trace = {"controller_call": True, "status": "validated"}
        return CampaignResult("completed", "M0000", {}, {"controller": {"calls": 1, "trace": [trace]}})

    campaign.run = cycle
    campaign._goal_satisfied = lambda split: False
    result = campaign.run_loop(resume=False, max_iterations=2, split="sanity", resource_poll_interval_s=0)

    assert len(calls) == 3
    assert result.report["loop"]["iterations"] == 2


def test_loop_retries_when_remote_controller_is_unavailable(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    stop_file = tmp_path / "stop-requested"
    calls = []

    def cycle(*, resume, max_experiments, split):
        calls.append((resume, max_experiments, split))
        if len(calls) >= 2:
            stop_file.write_text("stop\n", encoding="utf-8")
        trace = {
            "controller_call": True,
            "provider": "vllm",
            "model": "qwen3.5-controller",
            "status": "controller_unavailable",
        }
        return CampaignResult("completed", "M0000", {}, {"controller": {"calls": 1, "trace": [trace]}})

    campaign.run = cycle
    campaign._goal_satisfied = lambda split: False
    result = campaign.run_loop(
        resume=False,
        max_iterations=2,
        split="sanity",
        resource_poll_interval_s=0,
        stop_file=stop_file,
    )

    assert len(calls) == 2
    assert calls == [(False, 1, "sanity"), (True, 1, "sanity")]
    assert result.report["loop"]["iterations"] == 0
    assert result.report["loop"]["stop_reason"] == "stop_requested"
    assert len(result.report["loop"]["dependency_waits"]) == 2


def test_loop_retries_a_rejected_real_controller_plan(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    calls = []

    def cycle(*, resume, max_experiments, split):
        calls.append((resume, max_experiments, split))
        trace = (
            {"controller_call": True, "status": "rejected", "error": "missing observation"}
            if len(calls) == 1
            else {"controller_call": True, "status": "validated"}
        )
        return CampaignResult("completed", "M0000", {}, {"controller": {"calls": 1, "trace": [trace]}})

    campaign.run = cycle
    campaign._goal_satisfied = lambda split: False
    result = campaign.run_loop(resume=False, max_iterations=2, split="sanity", resource_poll_interval_s=0)

    assert calls == [(False, 1, "sanity"), (True, 1, "sanity")]
    assert result.report["loop"]["iterations"] == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "controller-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event_type"] == "controller_plan_rejected_retry" for event in events)


def test_loop_skips_full_cycle_when_remote_controller_preflight_is_unavailable(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    stop_file = tmp_path / "stop-requested"
    attempts = []

    def unavailable():
        from harness4h3.controller.provider import ControllerUnavailableError

        attempts.append(True)
        if len(attempts) >= 2:
            stop_file.write_text("stop\n", encoding="utf-8")
        raise ControllerUnavailableError("vLLM is waiting for a free GPU group")

    campaign._remote_controller_preflight = unavailable

    def cycle(**kwargs):
        raise AssertionError("full campaign cycle must not run while Controller preflight is unavailable")

    campaign.run = cycle
    result = campaign.run_loop(
        resume=False,
        max_iterations=2,
        split="sanity",
        resource_poll_interval_s=0,
        stop_file=stop_file,
    )

    assert result.report["loop"]["iterations"] == 0
    assert result.report["loop"]["stop_reason"] == "stop_requested"
    assert result.report["controller"]["calls"] == 0
    events = [
        json.loads(line)
        for line in (tmp_path / "controller-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    waits = [event for event in events if event["event_type"] == "loop_dependency_unavailable"]
    assert len(waits) == 2
    assert all(event["dependency"] == "controller" for event in waits)
    state = campaign._load_campaign_state()
    assert state["pipeline"]["stage"] == "waiting"
    assert state["pipeline_waiting_on"] == "controller"


def test_loop_waits_for_comfyui_before_evaluating_pending_candidates(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, worker=replace(campaign.config.worker, enabled=True))
    stop_file = tmp_path / "stop-requested"
    attempts = []

    def unavailable():
        attempts.append(True)
        if len(attempts) >= 2:
            stop_file.write_text("stop\n", encoding="utf-8")
        raise RemoteError("ComfyUI is still starting")

    campaign._remote_comfyui_preflight = unavailable
    result = campaign.run_loop(
        resume=False,
        max_iterations=2,
        split="sanity",
        resource_poll_interval_s=0,
        stop_file=stop_file,
    )

    assert result.status == "stop_requested"
    assert result.report["loop"]["stop_reason"] == "stop_requested"
    assert result.report["loop"]["iterations"] == 0
    state = campaign._load_campaign_state()
    assert state["pipeline"]["stage"] == "waiting"
    assert state["pipeline_waiting_on"] == "comfyui"


def test_campaign_reviewer_runs_immediate_and_heartbeat_reviews(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, review_interval_s=0.01, max_review_calls=4)

    class Reviewer:
        provider_name = "test"
        model_name = "reviewer"
        last_request_id = "review-1"

        def __init__(self):
            self.requests = []

        def review(self, request):
            self.requests.append(request)
            return ReviewDecision(
                action="continue",
                reason="test telemetry is healthy",
                evidence_ids=("evt-test",),
                confidence=0.9,
                next_review_after_s=0.01,
                risks=(),
            )

    reviewer = Reviewer()
    campaign.controller = reviewer
    campaign._review_now("training", "worker_started", {"experiment_id": "exp_0001"})
    heartbeat = campaign._start_review_heartbeat(
        "exp_0001",
        "training",
        lambda: {"telemetry": {"gpu": [{"index": 2, "utilization_gpu_pct": 80}]}},
    )
    time.sleep(0.04)
    campaign._stop_review_heartbeat(heartbeat)

    assert len(reviewer.requests) >= 2
    events = list(campaign.events.read())
    completed = [item for item in events if item.get("event_type") == "controller_review_completed"]
    assert any(item.get("trigger") == "heartbeat" for item in completed)
    assert all(item.get("provider") == "test" for item in completed)


def test_campaign_reviewer_stop_is_recorded_without_changing_worker_evidence(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, max_review_calls=2)

    class Reviewer:
        provider_name = "test"
        model_name = "reviewer"

        def review(self, request):
            return ReviewDecision(
                action="stop",
                reason="critical regression detected",
                evidence_ids=(),
                confidence=1.0,
                next_review_after_s=60,
                risks=("stop at safe boundary",),
            )

    campaign.controller = Reviewer()
    decision = campaign._review_now("training", "worker_completed", {"experiment_id": "exp_0001"})
    assert decision is not None and decision.action == "stop"
    assert campaign.review_stop_requested is True
    assert list(campaign.events.read())[-1]["event_type"] == "controller_review_completed"


def test_recoverable_worker_failure_routes_review_to_replan(tmp_path):
    campaign = build_campaign(tmp_path, {}, {}, {})
    campaign.config = replace(campaign.config, max_review_calls=2)

    class Reviewer:
        provider_name = "test"
        model_name = "reviewer"

        def review(self, request):
            return ReviewDecision(
                action="stop",
                reason="the worker failed and should be reconsidered",
                evidence_ids=(),
                confidence=0.9,
                next_review_after_s=60,
                risks=("change operator",),
            )

    campaign.controller = Reviewer()
    decision = campaign._review_now(
        "training",
        "worker_completed",
        {"experiment_id": "exp_0001", "status": "failed", "failure_type": "checkpoint_corrupt"},
    )
    assert decision is not None and decision.action == "stop"
    assert campaign.review_stop_requested is False
    assert campaign.review_replan_requested is True
    event = list(campaign.events.read())[-1]
    assert event["action"] == "stop"
    assert event["applied_action"] == "replan"
