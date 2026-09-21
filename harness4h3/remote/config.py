"""Validated configuration for a fixed remote H3 campaign."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

from ..config import Target, WorkflowConfig
from .reward import RewardWeights
from .ssh import RemoteConfig


class RemoteConfigError(ValueError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RemoteConfigError("%s must be a mapping" % name)
    return value


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RemoteConfigError("%s must be a path" % name)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _target(value: Any, name: str) -> Target:
    raw = _mapping(value, name)
    node_id, input_name = str(raw.get("node_id", "")).strip(), str(raw.get("input", "")).strip()
    if not node_id or not input_name:
        raise RemoteConfigError("%s requires node_id and input" % name)
    return Target(node_id, input_name)


def _positive(value: Any, name: str, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RemoteConfigError("%s must be numeric" % name)
    if parsed < 0 if allow_zero else parsed <= 0:
        raise RemoteConfigError("%s must be positive" % name)
    return parsed


@dataclass(frozen=True)
class RemoteRuntimeConfig:
    experience_path: Path
    output_root: Path
    tasks_path: Path
    target_path: Path


@dataclass(frozen=True)
class RemoteWorkerConfig:
    enabled: bool
    entrypoint: str
    config_template: str
    python: str
    allowed_operators: Tuple[str, ...]
    max_steps: int
    quantized_bits: Tuple[int, ...] = ()
    operator_entrypoints: Mapping[str, str] = field(default_factory=dict)
    operator_launchers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComfyUIWorkerConfig:
    """One independently schedulable ComfyUI evaluator process."""

    gpu_index: int
    port: int


@dataclass(frozen=True)
class RemoteCampaignConfig:
    remote: RemoteConfig
    workflow: WorkflowConfig
    runtime: RemoteRuntimeConfig
    reward: RewardWeights
    benchmark_splits: Tuple[str, ...]
    sampling_interval_s: float
    efficiency_thresholds: Mapping[str, float]
    black_frame_rate_threshold: float
    reset_backend_before_run: bool
    evaluator_command: Tuple[str, ...]
    evaluator_timeout_s: float
    benchmark_task_timeout_s: float
    quality_scope: str
    research_grade: bool
    worker: RemoteWorkerConfig
    review_interval_s: float = 60.0
    max_review_calls: int = 120
    # Maximum observation records copied into one Controller context.  The
    # append-only observation store remains complete on disk; this is only a
    # prompt-size guard for long-running campaigns.
    max_context_observations: int = 24
    controller_max_iterations: int = 64
    comfyui_cache_policy: str = "idle_release"
    checkpoint_retention_policy: str = "rejected_candidate_v1"
    # Number of completed non-root model weights kept for rollback.  The
    # append-only experiment/evaluation/experience records are not capped.
    max_retained_checkpoints: int = 3
    # Multiple workers are used only when the selected benchmark contains
    # multiple independent tasks.  A single video remains on one ComfyUI
    # process because duplicating it would not reduce diffusion latency.
    comfyui_workers: Tuple[ComfyUIWorkerConfig, ...] = (
        ComfyUIWorkerConfig(gpu_index=0, port=8188),
    )
    # Keep one primary successor and at most one isolated GPU-fill successor
    # in flight.  More branches would require a larger lineage arbitration
    # protocol and are intentionally not enabled by this campaign.
    pipeline_enabled: bool = True
    pipeline_max_inflight: int = 1
    # Keep one GPU available for a TP=1 Controller while a distributed worker
    # trains.  The scheduler still verifies live memory before allocation.
    controller_overlap_gpus: int = 1
    prefetch_before_full_training: bool = True
    # Number of consecutive scheduler waits before returning the decision to
    # the remote Controller for a fresh resource-aware plan.
    resource_wait_replan_after: int = 12
    comfyui_process_policy: str = "persistent_api"
    comfyui_idle_shutdown_s: float = 30.0
    # Maximum age of a campaign benchmark lease before an on-demand ComfyUI
    # launcher treats it as orphaned and exits.  This is deliberately much
    # longer than one normal task, but finite so an SSH crash cannot strand a
    # GPU indefinitely.
    comfyui_lease_max_age_s: float = 21600.0
    power_target_w: float = 300.0


def load_remote_campaign_config(path: Path) -> RemoteCampaignConfig:
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RemoteConfigError("unable to read remote config %s: %s" % (path, exc))
    root = _mapping(raw, "config")
    remote_raw = _mapping(root.get("remote"), "remote")
    workflow_raw = _mapping(root.get("workflow"), "workflow")
    runtime_raw = _mapping(root.get("runtime"), "runtime")
    benchmark_raw = _mapping(root.get("benchmark", {}), "benchmark")
    pipeline_raw = _mapping(root.get("pipeline", {}), "pipeline")
    review_raw = _mapping(root.get("controller_review", {}), "controller_review")
    reward_raw = _mapping(root.get("reward"), "reward")
    worker_raw = _mapping(root.get("worker", {}), "worker")
    retention_raw = _mapping(root.get("checkpoint_retention", {}), "checkpoint_retention")
    remote = RemoteConfig(
        host=str(remote_raw.get("host", "")).strip(),
        harness_root=str(remote_raw.get("harness_root", "")).strip(),
        model_root=str(remote_raw.get("model_root", "")).strip(),
        comfyui_root=str(remote_raw.get("comfyui_root", "")).strip(),
        comfyui_port=int(remote_raw.get("comfyui_port", 8188)),
        python=str(remote_raw.get("python", "python3")),
        results_root=str(remote_raw["results_root"]) if remote_raw.get("results_root") else None,
        deployment_dir=str(remote_raw["deployment_dir"]) if remote_raw.get("deployment_dir") else None,
        campaign_root=str(remote_raw["campaign_root"]) if remote_raw.get("campaign_root") else None,
        training_python=str(remote_raw["training_python"]) if remote_raw.get("training_python") else None,
        ssh_port=int(remote_raw.get("ssh_port", 22)),
    )
    mutable_raw = _mapping(workflow_raw.get("mutable", {}), "workflow.mutable")
    unknown = sorted(set(mutable_raw) - {"steps", "cfg"})
    if unknown:
        raise RemoteConfigError("unsupported remote mutable key(s): %s" % ", ".join(unknown))
    workflow = WorkflowConfig(
        template=_path(path.parent, workflow_raw.get("template"), "workflow.template"),
        prompt_target=_target(workflow_raw.get("prompt_target"), "workflow.prompt_target"),
        seed_target=_target(workflow_raw["seed_target"], "workflow.seed_target") if workflow_raw.get("seed_target") else None,
        mutable={key: _target(value, "workflow.mutable.%s" % key) for key, value in mutable_raw.items()},
    )
    runtime = RemoteRuntimeConfig(
        experience_path=_path(path.parent, runtime_raw.get("experience_path", "../var/remote-h3/experience.jsonl"), "runtime.experience_path"),
        output_root=_path(path.parent, runtime_raw.get("output_root", "../var/remote-h3/benchmark"), "runtime.output_root"),
        tasks_path=_path(path.parent, runtime_raw.get("tasks_path", "../examples/tasks.yaml"), "runtime.tasks_path"),
        target_path=_path(path.parent, runtime_raw.get("target_path", "targets/l40x4_h3_example.yaml"), "runtime.target_path"),
    )
    weights = RewardWeights(
        _positive(reward_raw.get("alpha", 1.0), "reward.alpha", allow_zero=True),
        _positive(reward_raw.get("beta", 0.2), "reward.beta", allow_zero=True),
        _positive(reward_raw.get("gamma", 0.2), "reward.gamma", allow_zero=True),
        _positive(reward_raw.get("delta", 0.2), "reward.delta", allow_zero=True),
    )
    splits_raw = benchmark_raw.get("splits", ["sanity", "dev", "heldout"])
    if not isinstance(splits_raw, list) or not splits_raw or not all(str(item).strip() for item in splits_raw):
        raise RemoteConfigError("benchmark.splits must be a non-empty list")
    thresholds_raw = _mapping(benchmark_raw.get("efficiency_thresholds", {}), "benchmark.efficiency_thresholds")
    thresholds = {str(key): _positive(value, "benchmark.efficiency_thresholds.%s" % key, allow_zero=True) for key, value in thresholds_raw.items()}
    if any(value > 1 for value in thresholds.values()):
        raise RemoteConfigError("efficiency thresholds must be between 0 and 1")
    comfyui_workers_raw = benchmark_raw.get(
        "comfyui_workers",
        [{"gpu_index": 0, "port": remote.comfyui_port}],
    )
    if not isinstance(comfyui_workers_raw, list) or not comfyui_workers_raw:
        raise RemoteConfigError("benchmark.comfyui_workers must be a non-empty list")
    comfyui_workers = []
    seen_gpus = set()
    seen_ports = set()
    for index, item in enumerate(comfyui_workers_raw):
        raw_worker = _mapping(item, "benchmark.comfyui_workers[%d]" % index)
        try:
            gpu_index = int(raw_worker.get("gpu_index"))
            port = int(raw_worker.get("port"))
        except (TypeError, ValueError):
            raise RemoteConfigError("benchmark.comfyui_workers[%d] requires integer gpu_index and port" % index)
        if gpu_index < 0 or gpu_index >= 4:
            raise RemoteConfigError("benchmark.comfyui_workers[%d].gpu_index must be in [0, 3]" % index)
        if port <= 0 or port > 65535:
            raise RemoteConfigError("benchmark.comfyui_workers[%d].port must be in [1, 65535]" % index)
        if gpu_index in seen_gpus:
            raise RemoteConfigError("benchmark.comfyui_workers cannot reuse gpu_index %d" % gpu_index)
        if port in seen_ports:
            raise RemoteConfigError("benchmark.comfyui_workers cannot reuse port %d" % port)
        seen_gpus.add(gpu_index)
        seen_ports.add(port)
        comfyui_workers.append(ComfyUIWorkerConfig(gpu_index=gpu_index, port=port))
    if comfyui_workers[0] != ComfyUIWorkerConfig(gpu_index=0, port=remote.comfyui_port):
        raise RemoteConfigError(
            "benchmark.comfyui_workers must start with the primary worker at GPU 0 and remote.comfyui_port"
        )
    command_raw = benchmark_raw.get("evaluator_command", [])
    if isinstance(command_raw, str):
        command = tuple(shlex.split(command_raw))
    elif isinstance(command_raw, list) and all(isinstance(item, str) for item in command_raw):
        command = tuple(command_raw)
    else:
        raise RemoteConfigError("benchmark.evaluator_command must be a string or list")
    max_steps = int(worker_raw.get("max_steps", 32))
    if max_steps <= 0 or max_steps > 32:
        raise RemoteConfigError("worker.max_steps must be between 1 and 32")
    allowed_operators = tuple(str(item) for item in worker_raw.get("allowed_operators", ["recovery_finetune", "step_distill"]))
    quantized_bits_raw = worker_raw.get("quantized_bits", [])
    if not isinstance(quantized_bits_raw, list):
        raise RemoteConfigError("worker.quantized_bits must be a list")
    try:
        quantized_bits = tuple(sorted(set(int(item) for item in quantized_bits_raw)))
    except (TypeError, ValueError):
        raise RemoteConfigError("worker.quantized_bits must contain integers")
    if any(bits not in {4, 8} for bits in quantized_bits):
        raise RemoteConfigError("worker.quantized_bits entries must be 4 or 8")
    entrypoints_raw = worker_raw.get("operator_entrypoints", {})
    launchers_raw = worker_raw.get("operator_launchers", {})
    if not isinstance(entrypoints_raw, Mapping) or not all(isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in entrypoints_raw.items()):
        raise RemoteConfigError("worker.operator_entrypoints must be a mapping of operator to path")
    if not isinstance(launchers_raw, Mapping) or not all(isinstance(key, str) and isinstance(value, str) and value in {"python", "torchrun"} for key, value in launchers_raw.items()):
        raise RemoteConfigError("worker.operator_launchers must map operators to python or torchrun")
    unknown_entrypoints = sorted(set(entrypoints_raw) - set(allowed_operators))
    unknown_launchers = sorted(set(launchers_raw) - set(allowed_operators))
    if unknown_entrypoints or unknown_launchers:
        raise RemoteConfigError("worker operator runner is configured for a disallowed operator")
    worker = RemoteWorkerConfig(
        enabled=bool(worker_raw.get("enabled", False)),
        entrypoint=str(worker_raw.get("entrypoint", "")).strip(),
        config_template=str(worker_raw.get("config_template", "")).strip(),
        python=str(worker_raw.get("python") or remote.training_python or remote.python),
        allowed_operators=allowed_operators,
        max_steps=max_steps,
        quantized_bits=quantized_bits,
        operator_entrypoints=dict(entrypoints_raw),
        operator_launchers=dict(launchers_raw),
    )
    if worker.enabled and (not worker.entrypoint or not worker.config_template):
        raise RemoteConfigError("enabled worker requires entrypoint and config_template")
    review_interval_s = _positive(review_raw.get("interval_s", 60.0), "controller_review.interval_s")
    try:
        max_review_calls = int(review_raw.get("max_calls", 120))
    except (TypeError, ValueError):
        raise RemoteConfigError("controller_review.max_calls must be an integer")
    if max_review_calls <= 0:
        raise RemoteConfigError("controller_review.max_calls must be positive")
    try:
        max_context_observations = int(review_raw.get("max_context_observations", 24))
    except (TypeError, ValueError):
        raise RemoteConfigError("controller_review.max_context_observations must be an integer")
    if isinstance(review_raw.get("max_context_observations", 24), bool) or not 1 <= max_context_observations <= 128:
        raise RemoteConfigError("controller_review.max_context_observations must be between 1 and 128")
    try:
        controller_max_iterations = int(review_raw.get("max_iterations", 64))
    except (TypeError, ValueError):
        raise RemoteConfigError("controller_review.max_iterations must be an integer")
    if controller_max_iterations <= 0:
        raise RemoteConfigError("controller_review.max_iterations must be positive")
    comfyui_cache_policy = str(benchmark_raw.get("comfyui_cache_policy", "idle_release")).strip()
    if comfyui_cache_policy not in {"idle_release", "warm_cache", "cold_cache"}:
        raise RemoteConfigError(
            "benchmark.comfyui_cache_policy must be idle_release, warm_cache, or cold_cache"
        )
    checkpoint_retention_policy = str(
        retention_raw.get("policy", "rejected_candidate_v1")
    ).strip()
    if checkpoint_retention_policy not in {"rejected_candidate_v1", "keep_all"}:
        raise RemoteConfigError(
            "checkpoint_retention.policy must be rejected_candidate_v1 or keep_all"
        )
    try:
        max_retained_checkpoints = int(retention_raw.get("max_retained_checkpoints", 3))
    except (TypeError, ValueError):
        raise RemoteConfigError("checkpoint_retention.max_retained_checkpoints must be an integer")
    if max_retained_checkpoints <= 0:
        raise RemoteConfigError("checkpoint_retention.max_retained_checkpoints must be positive")
    pipeline_enabled = pipeline_raw.get("enabled", True)
    if not isinstance(pipeline_enabled, bool):
        raise RemoteConfigError("pipeline.enabled must be a boolean")
    try:
        pipeline_max_inflight = int(pipeline_raw.get("max_inflight", 1))
    except (TypeError, ValueError):
        raise RemoteConfigError("pipeline.max_inflight must be an integer")
    if isinstance(pipeline_raw.get("max_inflight", 1), bool) or not 1 <= pipeline_max_inflight <= 2:
        raise RemoteConfigError("pipeline.max_inflight must be between 1 and 2")
    try:
        controller_overlap_gpus = int(pipeline_raw.get("controller_overlap_gpus", 1))
    except (TypeError, ValueError):
        raise RemoteConfigError("pipeline.controller_overlap_gpus must be an integer")
    if (
        isinstance(pipeline_raw.get("controller_overlap_gpus", 1), bool)
        or not 0 <= controller_overlap_gpus < 4
    ):
        raise RemoteConfigError("pipeline.controller_overlap_gpus must be between 0 and 3")
    prefetch_before_full_training = pipeline_raw.get("prefetch_before_full_training", True)
    if not isinstance(prefetch_before_full_training, bool):
        raise RemoteConfigError("pipeline.prefetch_before_full_training must be a boolean")
    try:
        resource_wait_replan_after = int(pipeline_raw.get("resource_wait_replan_after", 12))
    except (TypeError, ValueError):
        raise RemoteConfigError("pipeline.resource_wait_replan_after must be an integer")
    if (
        isinstance(pipeline_raw.get("resource_wait_replan_after", 12), bool)
        or resource_wait_replan_after <= 0
    ):
        raise RemoteConfigError("pipeline.resource_wait_replan_after must be positive")
    comfyui_process_policy = str(
        pipeline_raw.get("comfyui_process_policy", "persistent_api")
    ).strip()
    if comfyui_process_policy not in {"on_demand", "persistent_api"}:
        raise RemoteConfigError(
            "pipeline.comfyui_process_policy must be on_demand or persistent_api"
        )
    comfyui_idle_shutdown_s = _positive(
        pipeline_raw.get("comfyui_idle_shutdown_s", 30.0),
        "pipeline.comfyui_idle_shutdown_s",
    )
    comfyui_lease_max_age_s = _positive(
        pipeline_raw.get("comfyui_lease_max_age_s", 21600.0),
        "pipeline.comfyui_lease_max_age_s",
    )
    power_target_w = _positive(
        pipeline_raw.get("power_target_w", 300.0),
        "pipeline.power_target_w",
    )
    if not workflow.template.exists():
        raise RemoteConfigError("workflow template does not exist: %s" % workflow.template)
    return RemoteCampaignConfig(
        remote=remote,
        workflow=workflow,
        runtime=runtime,
        reward=weights,
        benchmark_splits=tuple(str(item) for item in splits_raw),
        sampling_interval_s=_positive(benchmark_raw.get("sampling_interval_s", 1.0), "benchmark.sampling_interval_s"),
        efficiency_thresholds=thresholds,
        black_frame_rate_threshold=_positive(benchmark_raw.get("black_frame_rate_threshold", 0.0), "benchmark.black_frame_rate_threshold", allow_zero=True),
        reset_backend_before_run=bool(benchmark_raw.get("reset_backend_before_run", True)),
        evaluator_command=command,
        evaluator_timeout_s=_positive(benchmark_raw.get("evaluator_timeout_s", 120), "benchmark.evaluator_timeout_s"),
        benchmark_task_timeout_s=_positive(
            benchmark_raw.get("task_timeout_s", 1200),
            "benchmark.task_timeout_s",
        ),
        quality_scope=str(root.get("quality_scope", "structural_proxy")),
        research_grade=bool(root.get("research_grade", False)),
        worker=worker,
        review_interval_s=review_interval_s,
        max_review_calls=max_review_calls,
        max_context_observations=max_context_observations,
        controller_max_iterations=controller_max_iterations,
        comfyui_cache_policy=comfyui_cache_policy,
        checkpoint_retention_policy=checkpoint_retention_policy,
        max_retained_checkpoints=max_retained_checkpoints,
        comfyui_workers=tuple(comfyui_workers),
        pipeline_enabled=pipeline_enabled,
        pipeline_max_inflight=pipeline_max_inflight,
        controller_overlap_gpus=controller_overlap_gpus,
        prefetch_before_full_training=prefetch_before_full_training,
        resource_wait_replan_after=resource_wait_replan_after,
        comfyui_process_policy=comfyui_process_policy,
        comfyui_idle_shutdown_s=comfyui_idle_shutdown_s,
        comfyui_lease_max_age_s=comfyui_lease_max_age_s,
        power_target_w=power_target_w,
    )
