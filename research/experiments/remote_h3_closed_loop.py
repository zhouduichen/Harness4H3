"""SSH-backed MiniMax-H3 experience import, evaluation, and promotion loop.

This module is deliberately a coordinator. The Controller only proposes an
operator; trusted remote configuration owns every executable, path, threshold,
and benchmark setting.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.model_store import ModelCandidateExists, ModelStore, ModelStoreError
from harness4h3.archive.pareto import ParetoArchive
from harness4h3.archive.system_candidate import SystemCandidate
from harness4h3.archive.system_store import SystemCandidateStore, SystemStoreError
from harness4h3.backends.comfyui import MiniMaxH3Adapter
from harness4h3.benchmark.capabilities import probe_minimax_h3_capabilities
from harness4h3.benchmark.h3 import BenchmarkSummary, H3BenchmarkRunner
from harness4h3.config import Target, WorkflowConfig
from harness4h3.controller.context import ControllerContext
from harness4h3.controller.policy import ValidationPipeline
from harness4h3.controller.provider import (
    ControllerProvider,
    ControllerProviderError,
    ControllerUnavailableError,
    build_controller_from_config,
)
from harness4h3.controller.round_policy import (
    RoundPolicy,
    mark_round_policy_stop,
    normalize_round_policy_progress,
    record_round_policy_trial,
    round_policy_budget_status,
    validate_round_policy,
)
from harness4h3.controller.reviewer import ReviewDecision
from harness4h3.controller.schemas import BudgetState, EvaluationResult, ExperimentPlan, HardwareMetrics
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.harness.state import Task, load_tasks
from harness4h3.memory.experience import ExperienceRecord, ExperienceStore
from harness4h3.memory.discovery_digest import DiscoveryDigest
from harness4h3.memory.fingerprint import experiment_fingerprint
from harness4h3.memory.observation import (
    ControllerEventStore,
    ObservationStore,
    artifact_reference,
    make_observation,
)
from harness4h3.memory.experiment_store import ExperimentRecord, ExperimentStore
from harness4h3.remote.config import RemoteCampaignConfig, load_remote_campaign_config
from harness4h3.remote.comfyui_lease import ComfyUILeaseManager, ComfyUILeaseResult
from harness4h3.remote.checkpoint_retention import RemoteCheckpointRetention
from harness4h3.remote.decision import AcceptanceInput, DecisionResult, decide
from harness4h3.remote.importer import ImportSummary, RemoteResultImporter
from harness4h3.remote.lane_packer import pack_worker_request
from harness4h3.remote.power import RemotePowerSampler
from harness4h3.remote.pipeline import (
    PipelineStage,
    PipelineState,
    evaluation_gpu_count,
    pack_evaluation_overlap,
)
from harness4h3.remote.round_gate import RoundGateResult, evaluate_round_gate
from harness4h3.remote.scheduler import RemoteResourceScheduler, ResourceDecision
from harness4h3.remote.ssh import RemoteError, RemotePortForward, SSHClient
from harness4h3.target.profile import TargetProfile, load_target_profile
from harness4h3.operators.model_evolution import build_model_evolution_registry


@dataclass(frozen=True)
class CampaignResult:
    status: str
    current_model_id: str
    records: Mapping[str, ExperienceRecord]
    report: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "current_model_id": self.current_model_id,
            "records": {key: value.to_dict() for key, value in self.records.items()},
            "report": copy.deepcopy(dict(self.report)),
        }


class RemoteResourceWaitError(RemoteError):
    """A recoverable remote resource wait, not a benchmark failure."""


def _summary_dict(summary: Any) -> Dict[str, Any]:
    if isinstance(summary, BenchmarkSummary):
        return summary.to_dict()
    if hasattr(summary, "to_dict"):
        raw = summary.to_dict()
        return dict(raw) if isinstance(raw, Mapping) else {}
    return dict(summary) if isinstance(summary, Mapping) else {}


def _inherit_quantization_metadata(state: ModelState, parent: ModelCandidate) -> ModelState:
    """Keep a quantized checkpoint's precision label across head-only updates."""

    parent_quantization = parent.state.quantization
    child_quantization = state.quantization
    parent_bits = parent_quantization.get("bits") if isinstance(parent_quantization, Mapping) else None
    child_bits = child_quantization.get("bits") if isinstance(child_quantization, Mapping) else None
    if (
        isinstance(parent_bits, int)
        and not isinstance(parent_bits, bool)
        and parent_bits in {4, 8}
        and (not isinstance(child_bits, int) or isinstance(child_bits, bool) or child_bits > parent_bits)
    ):
        return replace(
            state,
            dtype=parent.state.dtype,
            quantization=dict(parent_quantization),
        )
    return state


def _hardware(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, HardwareMetrics):
        return asdict(raw)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        name: getattr(raw, name, None)
        for name in ("latency_s", "peak_memory_gb", "model_size_gb", "energy_j", "throughput", "thermal")
    }


def _h3_runtime_optimizations(operator: Any, args: Any) -> Mapping[str, Any]:
    """Translate a validated training recipe into measured H3 runtime state.

    LPL/TDTM are inference controls, so they must not create another full
    checkpoint.  They travel with the step-distilled lineage as a bounded
    recipe and are materialized by ``H3BenchmarkRunner`` only when the remote
    ComfyUI capability probe has verified the extension.
    """

    if str(operator) != "step_distill" or not isinstance(args, Mapping):
        return {}
    optimizations: Dict[str, Dict[str, Any]] = {}
    lpl_target = args.get("lpl_target_steps")
    if isinstance(lpl_target, int) and not isinstance(lpl_target, bool) and lpl_target > 0:
        optimizations["lpl"] = {"target_steps": int(lpl_target)}
    merge_steps = args.get("tdtm_merge_steps")
    threshold = args.get("tdtm_similarity_threshold", 0.985)
    if (
        isinstance(merge_steps, int)
        and not isinstance(merge_steps, bool)
        and merge_steps > 0
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
    ):
        optimizations["tdtm"] = {
            "merge_steps": int(merge_steps),
            "similarity_threshold": float(threshold),
        }
    return {"h3_optimizations": optimizations} if optimizations else {}


def _numeric(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if converted == converted else float(default)


def _compact_controller_input_observation(value: Any) -> Mapping[str, Any]:
    """Keep controller-input audit events bounded without losing traceability.

    The append-only observation store is the source of truth and contains the
    complete recipe/evaluation evidence. Re-embedding those nested payloads in
    every ``controller_input`` event made the audit log grow rapidly and was
    never needed by the Controller, whose prompt has its own compaction path.
    """

    item = value if isinstance(value, Mapping) else {}
    compact: Dict[str, Any] = {
        key: item[key]
        for key in ("observation_id", "kind", "experiment_id", "model_id")
        if key in item
    }
    summary = item.get("summary")
    if not isinstance(summary, Mapping):
        return compact

    compact["summary_keys"] = sorted(str(key) for key in summary)[:32]
    for key in (
        "goal_id",
        "objective",
        "quality_score",
        "evaluation_id",
        "system_id",
        "device_id",
        "feasible",
        "operator",
        "status",
    ):
        if key in summary:
            value = summary[key]
            compact[key] = value[:800] if isinstance(value, str) else value
    instruction = summary.get("instruction")
    if isinstance(instruction, str):
        compact["instruction"] = instruction[:800]
        if len(instruction) > 800:
            compact["instruction_truncated"] = True
    violations = summary.get("violations")
    if isinstance(violations, (list, tuple)):
        compact["violation_count"] = len(violations)
        compact["violations"] = [str(entry)[:300] for entry in violations[:8]]
    hard_gates = summary.get("hard_gates")
    if isinstance(hard_gates, Mapping):
        compact["hard_gate_keys"] = sorted(str(key) for key in hard_gates)[:32]
    artifacts = item.get("artifacts")
    if isinstance(artifacts, (list, tuple)):
        compact["artifact_count"] = len(artifacts)
        compact["artifacts"] = [str(entry)[:300] for entry in artifacts[:8]]
    return compact


def _evaluation_result(
    summary: Mapping[str, Any],
    *,
    model_id: Optional[str] = None,
    system_id: Optional[str] = None,
    device_id: Optional[str] = None,
    task_split: Optional[str] = None,
) -> Optional[EvaluationResult]:
    stored = summary.get("evaluation_record")
    if isinstance(stored, Mapping):
        try:
            return EvaluationResult.from_dict(stored)
        except (TypeError, KeyError, ValueError):
            pass
    quality = summary.get("quality_score")
    if not isinstance(quality, (int, float)):
        return None
    hardware = HardwareMetrics(**{key: value for key, value in _hardware(summary.get("hardware", {})).items() if key in HardwareMetrics.__dataclass_fields__})
    return EvaluationResult(
        quality_score=float(quality),
        quality_metrics=dict(summary.get("quality_metrics") or {}),
        hardware=hardware,
        feasible=bool(summary.get("feasible", False)),
        violations=[str(item) for item in summary.get("violations", [])],
        critical_regression=bool(summary.get("hard_gates", {}).get("no_critical_temporal_collapse") is False),
        model_id=model_id or (str(summary["model_id"]) if summary.get("model_id") else None),
        system_id=system_id or (str(summary["system_id"]) if summary.get("system_id") else None),
        device_id=device_id or (str(summary["device_id"]) if summary.get("device_id") else None),
        task_split=task_split or (str(summary["task_split"]) if summary.get("task_split") else None),
        validity={
            "quality_measured": quality is not None,
            "hardware_measured": bool(hardware.latency_s is not None and hardware.peak_memory_gb is not None),
            "generation_valid": (summary.get("hard_gates") or {}).get("generation_valid") is True,
            "benchmark_valid": bool(summary.get("task_count", 0)),
        },
        provenance={
            "evaluator": str(summary.get("evaluator_version") or "remote_h3_benchmark"),
            "benchmark_recipe": dict(summary.get("benchmark_recipe") or {}),
            "quality_scope": summary.get("quality_scope"),
            "offline_simulation": False,
        },
        search_score=summary.get("search_score") if isinstance(summary.get("search_score"), (int, float)) else None,
        evaluation_id=str(summary.get("evaluation_id") or ""),
    )


def _state_digest(state: ModelState) -> str:
    payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RemoteCampaign:
    def __init__(
        self,
        config: RemoteCampaignConfig,
        controller: Optional[ControllerProvider] = None,
        ssh: Optional[SSHClient] = None,
        evaluator_factory: Optional[Callable[..., Any]] = None,
        target: Optional[TargetProfile] = None,
        output_root: Optional[Path] = None,
        benchmark_factory: Optional[Callable[..., Any]] = None,
        tunnel_factory: Optional[Callable[[SSHClient], Any]] = None,
    ):
        self.config = config
        # The remote campaign is a real Controller workflow by default.  The
        # offline RuleBased controller is available only when a caller passes
        # it explicitly (the CLI exposes that as an explicit test option).
        self.controller = controller or build_controller_from_config()
        self.ssh = ssh or SSHClient(config.remote)
        self.target = target or load_target_profile(config.runtime.target_path)
        self.output_root = Path(output_root or config.runtime.output_root).resolve()
        self.experience_path = config.runtime.experience_path if output_root is None else self.output_root / "experience.jsonl"
        self.experience = ExperienceStore(self.experience_path)
        self.observations = ObservationStore(self.output_root / "observations.jsonl")
        self.events = ControllerEventStore(self.output_root / "controller-events.jsonl")
        self.experiments = ExperimentStore(self.output_root / "experiments.jsonl")
        self.models = ModelStore(self.output_root / "models")
        self.systems = SystemCandidateStore(self.output_root / "systems", model_store=self.models)
        self.pareto = ParetoArchive(self.output_root / "pareto")
        self.evaluations_path = self.output_root / "evaluations.json"
        self.campaign_state_path = self.output_root / "campaign_state.json"
        # Keep the active search policy in its own atomic file.  Prefetch calls
        # update the campaign cursor concurrently, so a sidecar prevents a
        # stale speculative state write from erasing the primary policy.
        self.round_policy_path = self.output_root / "active-round-policy.json"
        # Keep in-flight speculative execution separate from the main
        # campaign snapshot.  Evaluation/reviewer writes cannot overwrite a
        # worker handle while the benchmark is running.
        self.speculative_state_path = self.output_root / "speculative-worker.json"
        self.evaluator_factory = evaluator_factory
        self.benchmark_factory = benchmark_factory
        self.tunnel_factory = tunnel_factory
        self.validation = ValidationPipeline()
        self.registry = build_model_evolution_registry()
        self.optimization_capabilities = probe_minimax_h3_capabilities(
            self.config.workflow.template,
            self.config.remote.comfyui_root,
        )
        self.controller_trace: List[Dict[str, Any]] = []
        # A speculative next plan is generated on a cloned Controller while
        # the trusted worker is running. Keep it out of the ordinary trace so
        # evaluation bookkeeping cannot accidentally attribute the shadow
        # plan to the worker that is currently being measured.
        self.controller_prefetch_trace: List[Dict[str, Any]] = []
        # A short CPU-only worker can finish before the cloned Controller
        # request. Keep that request alive through evaluation instead of
        # blocking the worker boundary for the full LLM timeout.
        self._deferred_prefetch_handles: List[Dict[str, Any]] = []
        # The primary OpenAI-compatible request can return several locally
        # eligible candidates in one n-way response. Keep that batch only
        # until the corresponding primary plan reaches _train_one(); a
        # CPU-only primary can then launch a GPU sibling immediately instead
        # of opening a second LLM request and leaving overlap cards idle.
        self._last_primary_controller_batch: Optional[Dict[str, Any]] = None
        # When the next primary plan is CPU-only, arm its independent GPU
        # alternative as soon as that plan is ready during training. The
        # evaluation callback can then launch from a ready plan instead of
        # spending the first minutes of the benchmark waiting for a second
        # Controller request.
        self._early_parallel_prefetch_lock = threading.Lock()
        self._early_parallel_prefetch_jobs: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # Checkpoint staging is a separate CPU/I/O pipeline. It starts while
        # the candidate is benchmarked and is joined before any training GPU
        # lease is acquired, so a slow NFS read cannot leave allocated GPUs
        # waiting at the worker barrier.
        self._checkpoint_prefetch_handles: Dict[str, Dict[str, Any]] = {}
        # Exact evaluator cards are captured only after ComfyUI leases are
        # selected.  They are passed into overlap evidence at worker launch;
        # no later status poll is allowed to infer historical ownership.
        self._active_evaluation_gpu_indices: Tuple[int, ...] = ()
        self._review_lock = threading.Lock()
        # Evaluation can launch the primary and parallel speculative workers
        # from different threads.  ModelStore.next_id() only sees imported
        # lineage, so two in-flight branches could otherwise select the same
        # child directory and result JSON.  Keep a process-local reservation
        # and also replay durable worker records after a restart.
        self._child_id_lock = threading.Lock()
        self._reserved_child_ids: set[str] = set()
        # The remote worker lease is shared by the main worker and an
        # optional speculative sibling.  Track its logical owner locally so
        # completion of a CPU-only main worker cannot release a still-running
        # GPU sibling's lease.  The lock also serializes two speculative
        # launch threads before they can overwrite the one shared lease file.
        self._worker_lease_lock = threading.Lock()
        self._worker_lease_experiment_id: Optional[str] = None
        self.review_calls = 0
        self.review_trace: List[Dict[str, Any]] = []
        self.review_stop_requested = False
        self.review_replan_requested = False
        self._resource_wait_review_signature: Optional[str] = None
        # ``worker.max_steps`` is the per-child training budget, not the
        # campaign budget. Keep the Controller budget independent so a long
        # RSI run can produce many bounded children without stopping at the
        # first child's step count.
        self._controller_iteration_budget = max(1, int(config.controller_max_iterations))
        # GPU 0 is not permanently reserved.  The ComfyUI lease adds the
        # reservation immediately before a real benchmark, while the live
        # nvidia-smi waterline excludes any resident/foreign process during
        # planning and training.  This lets the scheduler reuse GPU 0 after
        # ComfyUI has unloaded its cache.
        self.scheduler = RemoteResourceScheduler(
            self.ssh,
            gpu_count=4,
            min_free_memory_mb=26 * 1024,
            reserved_gpu_indices=(),
        )
        self.comfyui_lease = ComfyUILeaseManager(
            self.ssh,
            self.scheduler,
            port=self.config.remote.comfyui_port,
            gpu_index=0,
            release_wait_s=float(self.config.comfyui_idle_shutdown_s),
        )
        self.comfyui_leases: Dict[Tuple[int, int], ComfyUILeaseManager] = {
            (0, self.config.remote.comfyui_port): self.comfyui_lease,
        }
        self.checkpoint_retention = RemoteCheckpointRetention(
            self.ssh,
            self.config.remote.results_root or self.config.remote.model_root,
            policy=self.config.checkpoint_retention_policy,
        )
        self.last_comfyui_lease: Optional[ComfyUILeaseResult] = None
        self.last_resource_decision: Optional[Mapping[str, Any]] = None
        goal = self._goal_payload()
        state = self._load_campaign_state()
        state["goal"] = goal
        self._save_campaign_state(state)
        self.events.append("goal_declared", {"goal": goal})

    @classmethod
    def from_config_path(cls, path: Path, **kwargs: Any) -> "RemoteCampaign":
        return cls(load_remote_campaign_config(path), **kwargs)

    def _root_state(self, first_child: Optional[ExperienceRecord] = None) -> ModelState:
        child_state = dict((first_child.provenance.get("output_state") if first_child else {}) or {})
        checkpoint = str(
            self.config.remote.model_root
            + "/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
        )
        source_checkpoint = child_state.get("provenance", {}).get("parent_checkpoint_path") if isinstance(child_state.get("provenance"), Mapping) else None
        root_checkpoint = str(source_checkpoint or checkpoint)
        size_bytes = self._remote_file_size_bytes(root_checkpoint)
        provenance = {"remote_root_inferred": True, "source_checkpoint": root_checkpoint}
        if size_bytes is not None:
            provenance["size_bytes"] = size_bytes
        state = ModelState(
            model_id="M0000",
            parent_model_id=None,
            checkpoint_path=root_checkpoint,
            architecture_name=str(child_state.get("architecture_name") or "MiniMax-H3-FL2VA"),
            parameter_count=child_state.get("parameter_count"),
            dtype=str(child_state.get("dtype") or "bfloat16"),
            sampling_steps=int(child_state.get("algorithm_state", {}).get("source_steps", 32)) if isinstance(child_state.get("algorithm_state"), Mapping) else 32,
            components=copy.deepcopy(dict(child_state.get("components") or {})),
            algorithm_state={"imported_root": True},
            runtime_state={"remote_host": self.config.remote.host},
            measured_metrics={},
            provenance=provenance,
        )
        return state

    def _remote_file_size_bytes(self, path: str) -> Optional[int]:
        """Read the size of a trusted checkpoint without loading its payload."""

        try:
            result = self.ssh.run(("stat", "-c", "%s", "--", str(path)), check=False)
            if int(getattr(result, "returncode", 1)) != 0:
                return None
            raw = str(getattr(result, "stdout", "")).strip()
            value = int(raw)
            return value if value >= 0 else None
        except (OSError, TypeError, ValueError, RemoteError):
            return None

    def _load_evaluations(self) -> Dict[str, Dict[str, Any]]:
        if not self.evaluations_path.exists():
            return {}
        raw = json.loads(self.evaluations_path.read_text(encoding="utf-8"))
        return {str(key): dict(value) for key, value in raw.items()} if isinstance(raw, Mapping) else {}

    def _save_evaluations(self, values: Mapping[str, Mapping[str, Any]]) -> None:
        _atomic_json(self.evaluations_path, values)

    def _evaluation_signature(self, split: Optional[str]) -> str:
        payload = {
            "target": self.target.to_dict(),
            "split": split or "all",
            "workflow": str(self.config.workflow.template),
            "quality_scope": self.config.quality_scope,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _evaluation_is_current(self, summary: Mapping[str, Any], split: Optional[str]) -> bool:
        return summary.get("evaluation_signature") == self._evaluation_signature(split)

    def _load_campaign_state(self) -> Dict[str, Any]:
        if not self.campaign_state_path.exists():
            return {
                "evaluated_ids": [],
                "training_calls": 0,
                "current_model_id": "M0000",
                "current_system_id": "S0000",
            }
        raw = json.loads(self.campaign_state_path.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, Mapping) else {
            "evaluated_ids": [],
            "training_calls": 0,
            "current_model_id": "M0000",
            "current_system_id": "S0000",
        }

    def _save_campaign_state(self, value: Mapping[str, Any]) -> None:
        _atomic_json(self.campaign_state_path, value)

    def _active_round_policy(self) -> Optional[RoundPolicy]:
        """Load and runtime-validate the active policy without crashing planning."""

        raw: Any = None
        source = "campaign_state"
        for path in (self.round_policy_path, self.campaign_state_path):
            if not path.exists():
                continue
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.events.append(
                    "round_policy_unavailable",
                    {"source": str(path), "reason": "invalid_json:%s" % str(exc)[:400]},
                )
                continue
            if isinstance(candidate, Mapping) and (
                path == self.round_policy_path or isinstance(candidate.get("active_round_policy"), Mapping)
            ):
                raw = candidate if path == self.round_policy_path else candidate.get("active_round_policy")
                source = "sidecar" if path == self.round_policy_path else "campaign_state"
                break
        if not isinstance(raw, Mapping):
            return None
        try:
            policy = RoundPolicy.from_dict(raw)
            self._validate_round_policy_runtime(policy)
            return policy
        except (TypeError, ValueError) as exc:
            self.events.append(
                "round_policy_unavailable",
                {"source": source, "reason": str(exc)[:500]},
            )
            return None

    def _round_policy_substrate_digest(self) -> str:
        """Digest the immutable execution substrate visible to the Controller."""

        payload = {
            "target": self.target.to_dict(),
            "workflow_template": str(self.config.workflow.template),
            "quality_scope": str(self.config.quality_scope),
            "worker_allowed_operators": sorted(str(item) for item in self.config.worker.allowed_operators),
            "operator_launchers": {
                str(key): str(value)
                for key, value in sorted(self.config.worker.operator_launchers.items())
            },
            "operator_entrypoints": {
                str(key): str(value)
                for key, value in sorted(self.config.worker.operator_entrypoints.items())
            },
            "quantized_bits": [int(item) for item in self.config.worker.quantized_bits],
        }
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _validate_round_policy_runtime(self, policy: RoundPolicy) -> None:
        registered = {
            str(item.get("name"))
            for item in self.registry.visible()
            if isinstance(item, Mapping) and item.get("name")
        }
        registered &= {str(item) for item in self.config.worker.allowed_operators}
        validate_round_policy(
            policy,
            registered_operators=registered,
            substrate_digest=self._round_policy_substrate_digest(),
            evaluation_digest=self._evaluation_signature("heldout"),
            gpu_count=int(getattr(self.scheduler, "gpu_count", 4)),
        )

    def _persist_active_round_policy(self, policy: RoundPolicy, state: Dict[str, Any]) -> None:
        """Persist a normalized primary policy in both recovery locations."""

        normalized = dict(policy.to_dict())
        _atomic_json(self.round_policy_path, normalized)
        prior_progress = state.get("round_policy_progress")
        prior_round_id = (
            str(prior_progress.get("round_id"))
            if isinstance(prior_progress, Mapping) and prior_progress.get("round_id")
            else ""
        )
        # A new policy starts a new bounded search round.  Retrying the same
        # policy after a restart preserves its completed-trial cursor.
        if prior_round_id != policy.round_id:
            state["round_policy_progress"] = normalize_round_policy_progress(policy)
        else:
            state["round_policy_progress"] = normalize_round_policy_progress(policy, prior_progress)
        state["active_round_policy"] = normalized
        state["active_round_policy_path"] = str(self.round_policy_path)

    def _round_policy_progress(
        self,
        policy: Optional[RoundPolicy] = None,
        state: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read the bounded mutable cursor paired with the active policy."""

        active = policy or self._active_round_policy()
        if active is None:
            return None
        snapshot = state if isinstance(state, Mapping) else self._load_campaign_state()
        return normalize_round_policy_progress(active, snapshot.get("round_policy_progress"))

    def _round_policy_budget_status(
        self,
        policy: Optional[RoundPolicy] = None,
        state: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        active = policy or self._active_round_policy()
        if active is None:
            return None
        return round_policy_budget_status(active, self._round_policy_progress(active, state))

    def _mark_round_policy_stop(
        self,
        policy: RoundPolicy,
        state: Dict[str, Any],
        reason: str,
        *,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        progress = mark_round_policy_stop(
            policy,
            state.get("round_policy_progress"),
            reason,
        )
        state["round_policy_progress"] = progress
        self._save_campaign_state(state)
        payload = {
            "round_id": policy.round_id,
            "reason": str(reason),
            "progress": dict(progress),
        }
        if experiment_id:
            payload["experiment_id"] = str(experiment_id)
        self.events.append("round_policy_stopped", payload)
        return progress

    def _build_discovery_digest(
        self,
        *,
        telemetry: Optional[Mapping[str, Any]] = None,
    ) -> DiscoveryDigest:
        """Build the bounded experience view used by the next Controller request."""

        experiments = [record.to_dict() for record in self.experience.read()]
        observations = [record.to_dict() for record in self.observations.read()]
        pareto = [entry.to_dict() for entry in self.pareto.front()]
        telemetry_value = dict(telemetry) if isinstance(telemetry, Mapping) else self._pipeline_telemetry()
        digest = DiscoveryDigest.build(
            experiments,
            observations=observations,
            pareto=pareto,
            telemetry=[telemetry_value],
        )
        self.events.append(
            "discovery_digest_built",
            {
                "source_digest": digest.source_digest,
                "source_observation_ids": list(digest.source_observation_ids),
                "recent_experiment_count": len(digest.recent_experiments),
            },
        )
        return digest

    def _set_pipeline_stage(
        self,
        stage: PipelineStage,
        *,
        evaluation_model_id: Optional[str] = None,
        training_model_id: Optional[str] = None,
        overlap_with_model_id: Optional[str] = None,
    ) -> None:
        """Persist the small pipeline cursor used by the remote supervisor.

        This is deliberately separate from the model/experience stores: a
        restart must be able to explain whether a card was held by evaluation,
        training, or plan prefetch without treating an in-flight worker as a
        completed model lineage event.
        """

        state = self._load_campaign_state()
        state.pop("pipeline_waiting_on", None)
        raw = state.get("pipeline", {})
        try:
            cursor = PipelineState.from_dict(raw) if isinstance(raw, Mapping) else PipelineState()
        except (TypeError, ValueError):
            cursor = PipelineState()
        state["pipeline"] = cursor.advance(
            stage,
            evaluation_model_id=evaluation_model_id,
            training_model_id=training_model_id,
            overlap_with_model_id=overlap_with_model_id,
        ).to_dict()
        self._save_campaign_state(state)
        self.events.append(
            "pipeline_stage",
            {
                "stage": PipelineStage(stage).value,
                "evaluation_model_id": evaluation_model_id,
                "training_model_id": training_model_id,
                "overlap_with_model_id": overlap_with_model_id,
            },
        )

    def _set_pipeline_waiting(
        self,
        dependency: str,
        *,
        current_model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        """Persist a non-optimization dependency wait without advancing the pipeline.

        A Controller/ComfyUI/resource outage is a pause in the control plane,
        not another optimization stage. Keep the last lane identifiers and
        pipeline iteration intact so a status reader can distinguish an idle
        wait from active training without inflating the durable iteration
        counter on every polling attempt.
        """

        dependency_name = str(dependency or "unknown")
        state = self._load_campaign_state()
        raw = state.get("pipeline", {})
        try:
            cursor = PipelineState.from_dict(raw) if isinstance(raw, Mapping) else PipelineState()
        except (TypeError, ValueError):
            cursor = PipelineState()
        same_wait = (
            cursor.stage == PipelineStage.WAITING
            and state.get("pipeline_waiting_on") == dependency_name
            and (
                not current_model_id
                or cursor.overlap_with_model_id == str(current_model_id)
            )
        )
        if same_wait:
            return
        state["pipeline"] = replace(
            cursor,
            stage=PipelineStage.WAITING,
            overlap_with_model_id=str(current_model_id) if current_model_id else cursor.overlap_with_model_id,
            updated_at=time.time(),
        ).to_dict()
        state["pipeline_waiting_on"] = dependency_name
        self._save_campaign_state(state)
        self.events.append(
            "pipeline_waiting",
            {
                "dependency": dependency_name,
                "current_model_id": current_model_id,
                "experiment_id": experiment_id,
                "pipeline_iteration": state["pipeline"]["iteration"],
            },
        )

    def _record_lane_allocation(
        self,
        stage: str,
        *,
        worker_gpus: Sequence[int] = (),
        comfyui_workers: Sequence[Any] = (),
        evaluator_gpus: Sequence[int] = (),
        planned_request: Optional[Mapping[str, Any]] = None,
        effective_request: Optional[Mapping[str, Any]] = None,
        reason: str = "",
    ) -> None:
        """Publish an auditable snapshot of the campaign-owned lane layout."""

        lanes: List[Mapping[str, Any]] = []
        for worker in comfyui_workers:
            lanes.append(
                {
                    "lane": "comfyui",
                    "gpu_index": int(worker.gpu_index),
                    "port": int(worker.port),
                    "lease_file": self._comfyui_lease_for(worker).lease_path,
                }
            )
        if worker_gpus:
            lanes.append(
                {
                    "lane": "worker",
                    "allocated_gpus": [int(index) for index in worker_gpus],
                    "lease_file": getattr(self.scheduler, "lease_path", None),
                }
            )
        controller_file = getattr(self.scheduler, "controller_lease_path", None)
        controller_indices: Optional[Sequence[int]] = None
        try:
            controller_indices = getattr(self.scheduler, "controller_reserved_gpu_indices", None)
            if controller_indices is not None:
                controller_indices = [int(index) for index in controller_indices]
        except (OSError, RemoteError, TypeError, ValueError):
            controller_indices = None
        lanes.append(
            {
                "lane": "controller",
                "allocated_gpus": list(controller_indices or ()),
                "lease_file": controller_file,
                "lease_state": "leased" if controller_indices else "external_or_unleased",
            }
        )
        payload: Dict[str, Any] = {
            "stage": str(stage),
            "lanes": lanes,
            "reason": str(reason),
        }
        if planned_request is not None:
            payload["planned_resource_request"] = dict(planned_request)
        if effective_request is not None:
            payload["effective_resource_request"] = dict(effective_request)
        evaluator_indices = tuple(
            int(worker.gpu_index) for worker in comfyui_workers
        ) if comfyui_workers else tuple(int(index) for index in evaluator_gpus)
        payload["lane_evidence"] = self._lane_evidence(
            evaluator_gpus=evaluator_indices,
            worker_gpus=worker_gpus,
            status="reserved",
            reason=reason,
        )
        self.events.append("lane_allocation", payload)

    def _lane_evidence(
        self,
        *,
        evaluator_gpus: Sequence[int] = (),
        worker_gpus: Sequence[int] = (),
        status: str = "ready",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Return exact lane ownership evidence for one overlap boundary.

        The scheduler is the authority for the Controller lease.  This
        helper only reports the card indices observed at the boundary; it
        never allocates, releases, or guesses around an unmappable process.
        """

        def normalize(values: Sequence[int]) -> Tuple[int, ...]:
            return tuple(sorted(set(int(index) for index in values)))

        evaluator = normalize(evaluator_gpus)
        worker = normalize(worker_gpus)
        controller_error = ""
        try:
            controller_value = getattr(self.scheduler, "controller_reserved_gpu_indices", ())
            controller_value = controller_value() if callable(controller_value) else controller_value
            controller = normalize(controller_value or ())
        except (OSError, RemoteError, TypeError, ValueError) as exc:
            # An unavailable lease read is not proof of an empty Controller
            # lane.  Preserve the evidence record, but fail the overlap gate.
            controller = ()
            controller_error = str(exc)[:500]
        total_gpu_count = int(getattr(self.scheduler, "gpu_count", 4))
        conflicts = {
            "evaluator_controller": sorted(set(evaluator).intersection(controller)),
            "evaluator_worker": sorted(set(evaluator).intersection(worker)),
            "controller_worker": sorted(set(controller).intersection(worker)),
        }
        disjoint = not any(conflicts.values())
        if controller_error:
            disjoint = False
        proposal: Mapping[str, Any] = {}
        proposal_error = ""
        try:
            proposal = pack_evaluation_overlap(
                total_gpu_count,
                evaluator_gpus=evaluator,
                controller_gpus=controller,
                minimum_training_gpus=2,
            ).to_dict()
        except (TypeError, ValueError) as exc:
            proposal_error = str(exc)[:500]
            disjoint = False
        evidence: Dict[str, Any] = {
            "status": str(status),
            "reason": str(reason),
            "evaluator_gpus": list(evaluator),
            "controller_gpus": list(controller),
            "worker_gpus": list(worker),
            "disjoint": bool(disjoint),
            "conflicts": conflicts,
            "gpu_count": total_gpu_count,
        }
        if proposal:
            evidence["planned_layout"] = dict(proposal)
        if proposal_error:
            evidence["planned_layout_error"] = proposal_error
        if controller_error:
            evidence["controller_observation_error"] = controller_error
        return evidence

    def _speculative_state_path(self, slot: str = "primary") -> Path:
        """Return a campaign-local state file for one isolated worker slot."""

        value = str(slot or "primary")
        if value == "primary":
            return self.speculative_state_path
        if value == "parallel":
            return self.output_root / "speculative-worker-parallel.json"
        raise ValueError("unsupported speculative worker slot: %s" % value)

    def _load_speculative_state(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Read the isolated in-flight worker record, if one exists."""

        state_path = Path(path or self.speculative_state_path)
        if not state_path.exists():
            return {}
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(raw) if isinstance(raw, Mapping) else {}

    def _save_speculative_state(self, value: Mapping[str, Any], path: Optional[Path] = None) -> None:
        _atomic_json(Path(path or self.speculative_state_path), value)

    def _reserve_next_child_id(self) -> str:
        """Reserve a unique child id across concurrent in-flight workers."""

        with self._child_id_lock:
            used: set[int] = set()
            try:
                model_ids = [item.id for item in self.models.lineage()]
            except (ModelStoreError, OSError, TypeError, ValueError):
                model_ids = []
            try:
                experience_ids = [item.child_model_id for item in self.experience.read()]
            except (OSError, TypeError, ValueError):
                experience_ids = []
            durable_states = (
                self._load_speculative_state(),
                self._load_speculative_state(self._speculative_state_path("parallel")),
            )
            values = list(model_ids) + list(experience_ids) + list(self._reserved_child_ids)
            for state in durable_states:
                if str(state.get("status") or "") in {"launching", "running", "completed"}:
                    values.append(state.get("child_model_id"))
            for value in values:
                match = re.fullmatch(r"M(\d+)", str(value or ""))
                if match:
                    used.add(int(match.group(1)))
            number = max(used) + 1 if used else 0
            child_id = "M%04d" % number
            while number in used:
                number += 1
                child_id = "M%04d" % number
            self._reserved_child_ids.add(child_id)
            return child_id

    def _release_child_id_reservation(self, child_id: Optional[str]) -> None:
        if child_id:
            with self._child_id_lock:
                self._reserved_child_ids.discard(str(child_id))

    def _clear_speculative_state(self, path: Optional[Path] = None) -> None:
        state_path = Path(path or self.speculative_state_path)
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass

    def _recover_speculative_worker(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Refresh an isolated worker record after a supervisor restart.

        The worker is launched through a background SSH thread, so a
        supervisor restart can lose the in-memory ``Thread`` object while the
        remote process continues to run.  The result JSON is the durable
        completion signal; seeing it is enough to turn a running record into
        a recoverable completed record without guessing from GPU utilization.
        """

        state_path = Path(path or self.speculative_state_path)
        state = self._load_speculative_state(state_path)
        status = str(state.get("status") or "")
        if status not in {"launching", "running"}:
            return state
        result_path = str(state.get("result_path") or "")
        if not result_path:
            return state
        try:
            result = self.ssh.read_json(result_path)
        except (RemoteError, OSError, TypeError, ValueError):
            return state
        if not isinstance(result, Mapping):
            return state
        state = dict(state)
        state.update(
            {
                "status": "completed",
                "recovered_at": time.time(),
                "recovered_result_status": result.get("status"),
            }
        )
        self._save_speculative_state(state, state_path)
        self.events.append(
            "speculative_worker_recovered",
            {
                "experiment_id": state.get("experiment_id"),
                "child_model_id": state.get("child_model_id"),
                "previous_status": status,
                "status": "completed",
                "result_path": result_path,
            },
        )
        return state

    def _speculative_plan_for_candidate(
        self,
        candidate: ModelCandidate,
        training_calls: int,
    ) -> Optional[ExperimentPlan]:
        """Rebase one prefetched plan onto the child currently being evaluated."""

        state = self._load_campaign_state()
        try:
            plan = self._prefetched_plan(state)
        except (TypeError, ValueError, KeyError) as exc:
            self.events.append(
                "controller_plan_prefetch_discarded",
                {"reason": "invalid_speculative_plan", "error": str(exc)[:1000]},
            )
            return None
        if plan is None:
            return None
        expected_calls = state.get("prefetched_training_calls")
        try:
            calls_match = expected_calls is not None and int(expected_calls) == int(training_calls)
        except (TypeError, ValueError):
            calls_match = False
        source_child_id = str(state.get("prefetched_source_child_model_id") or "")
        source_parent_id = str(state.get("prefetched_parent_model_id") or "")
        if (
            not calls_match
            or not source_child_id
            or source_child_id != candidate.id
            or (candidate.parent_id and source_parent_id != candidate.parent_id)
        ):
            self.events.append(
                "controller_plan_prefetch_discarded",
                {
                    "experiment_id": plan.experiment_id,
                    "parent_model_id": plan.parent_model_id,
                    "source_parent_model_id": source_parent_id,
                    "source_child_model_id": source_child_id,
                    "candidate_model_id": candidate.id,
                    "reason": "speculative_lineage_mismatch",
                },
            )
            return None
        current_system = self._system_for_model(candidate.id)
        rebased = replace(
            plan,
            parent_model_id=candidate.id,
            parent_system_id=current_system.id if current_system is not None else plan.parent_system_id,
        )
        self.validation.schema.validate(rebased)
        self._validate_worker_resource_match(rebased)
        return rebased

    def _goal_payload(self) -> Dict[str, Any]:
        return {
            "goal_id": self.target.id,
            "objective": "find a Pareto candidate that satisfies the target profile",
            "success_criteria": {
                key: value
                for key, value in {
                    "max_model_size_gb": self.target.max_model_size_gb,
                    "max_peak_memory_gb": self.target.max_peak_memory_gb,
                    "max_latency_s": self.target.max_latency_s,
                    "max_energy_j": self.target.max_energy_j,
                    "min_quality_score": self.target.min_quality_score,
                    "max_quality_drop": self.target.max_quality_drop,
                }.items()
                if value is not None
            },
            "priority": list(self.target.priority),
            "stop_conditions": ["target_satisfied", "budget_exhausted", "critical_regression"],
        }

    def _goal_satisfied(self, split: Optional[str]) -> bool:
        try:
            active_id = self.models.active_id
        except ModelStoreError:
            # A freshly queued campaign can be waiting on remote services
            # before the baseline has been materialized locally.
            return False
        summary = self._load_evaluations().get(active_id)
        return bool(summary and self._evaluation_is_current(summary, split) and summary.get("feasible") is True)

    def _round_policy_terminal_reason(self) -> Optional[str]:
        """Return a durable policy stop reason, including exhausted budget."""

        policy = self._active_round_policy()
        if policy is None:
            return None
        state = self._load_campaign_state()
        progress = self._round_policy_progress(policy, state)
        reason = round_policy_budget_status(policy, progress)
        if reason is not None and not progress.get("stop_reason"):
            self._mark_round_policy_stop(policy, state, reason)
        return reason

    def _mark_round_policy_target_if_satisfied(self, split: Optional[str]) -> None:
        """Persist target satisfaction when the active policy requests it."""

        policy = self._active_round_policy()
        if policy is None or "target_satisfied" not in set(policy.stop_conditions):
            return
        if not self._goal_satisfied(split):
            return
        state = self._load_campaign_state()
        progress = self._round_policy_progress(policy, state) or {}
        if progress.get("stop_reason") is None:
            self._mark_round_policy_stop(policy, state, "target_satisfied")

    def _append_observation(self, record: Any) -> None:
        if self.observations.append(record):
            self.events.append(
                "observation_appended",
                {
                    "observation_id": record.observation_id,
                    "kind": record.kind,
                    "experiment_id": record.experiment_id,
                    "model_id": record.model_id,
                    "source_uri": record.source_uri,
                    "source_sha256": record.source_sha256,
                },
            )

    def _collect_worker_telemetry(self, experiment_id: str) -> Mapping[str, Any]:
        """Read bounded remote telemetry using fixed, non-Controller commands."""

        telemetry: Dict[str, Any] = {"experiment_id": str(experiment_id), "gpu": [], "worker_processes": []}
        try:
            gpu_result = self.ssh.run(
                (
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ),
                check=False,
            )
            for line in str(getattr(gpu_result, "stdout", "")).splitlines():
                parts = [item.strip() for item in line.split(",")]
                if len(parts) != 4:
                    continue
                try:
                    telemetry["gpu"].append(
                        {
                            "index": int(parts[0]),
                            "utilization_gpu_pct": float(parts[1]),
                            "memory_used_mb": float(parts[2]),
                            "memory_total_mb": float(parts[3]),
                        }
                    )
                except (TypeError, ValueError):
                    continue
            telemetry["gpu_query_returncode"] = int(getattr(gpu_result, "returncode", 0))
        except Exception as exc:
            telemetry["gpu_query_error"] = str(exc)[:500]
        try:
            process_result = self.ssh.run(("pgrep", "-af", "h3_real_train_worker.py"), check=False)
            telemetry["worker_processes"] = [
                line[:500]
                for line in str(getattr(process_result, "stdout", "")).splitlines()[-8:]
                if line.strip()
            ]
            telemetry["process_query_returncode"] = int(getattr(process_result, "returncode", 0))
        except Exception as exc:
            telemetry["process_query_error"] = str(exc)[:500]
        return telemetry

    def _review_request(self, phase: str, trigger: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        state = self._load_campaign_state()
        try:
            active_model_id = str(self.models.active_id)
        except (ModelStoreError, AttributeError):
            active_model_id = str(state.get("current_model_id") or "M0000")
        active_system = self._system_for_model(active_model_id)
        active_system_id = active_system.id if active_system is not None else state.get("current_system_id")
        # Review calls happen during training/benchmarking, before the next
        # loop boundary persists its state.  The reviewer must see the same
        # active lineage as the planner, not a stale bootstrap M0000 value.
        if state.get("current_model_id") != active_model_id or state.get("current_system_id") != active_system_id:
            state["current_model_id"] = active_model_id
            state["current_system_id"] = active_system_id
            self._save_campaign_state(state)
        recent_events: List[Mapping[str, Any]] = []
        for event in list(self.events.read())[-10:]:
            recent_events.append(
                {
                    key: value
                    for key, value in event.items()
                    if key not in {"command_template", "worker_log_tail", "worker_error_tail"}
                }
            )
        controller_reserved: List[int] = []
        controller_lease_error: Optional[str] = None
        try:
            memory, processes = self.scheduler.snapshot()
            controller_reader = getattr(self.scheduler, "_controller_reserved_gpu_indices", None)
            if callable(controller_reader):
                try:
                    controller_reserved = sorted(int(index) for index in controller_reader())
                except (OSError, TypeError, ValueError) as exc:
                    controller_lease_error = str(exc)[:500]
            gpu_snapshot = {
                str(index): {
                    "memory_used_mb": int(values[0]),
                    "memory_total_mb": int(values[1]) if values[1] is not None else None,
                    "free_above_waterline": self.scheduler.meets_memory_waterline(index, memory),
                }
                for index, values in memory.items()
            }
            process_present = bool(processes.strip())
        except Exception as exc:
            gpu_snapshot = {"error": str(exc)[:500]}
            process_present = None
        request: Dict[str, Any] = {
            "phase": str(phase),
            "trigger": str(trigger),
            "current_model_id": active_model_id,
            "current_system_id": active_system_id,
            "training_calls": state.get("training_calls", 0),
            "target": self._goal_payload(),
            "gpu_isolation": {
                "comfyui_reserved_gpu": self.comfyui_lease.gpu_index if self.comfyui_lease.lease_active else None,
                "comfyui_reserved_gpu_indices": sorted(
                    {
                        int(lease.gpu_index)
                        for lease in self.comfyui_leases.values()
                        if lease.lease_active
                    }
                ),
                "scheduler_reserved_gpu_indices": list(getattr(self.scheduler, "reserved_gpu_indices", ())),
                "controller_reserved_gpu_indices": controller_reserved,
                "controller_lease_error": controller_lease_error,
                "gpu_snapshot": gpu_snapshot,
                "compute_processes_present": process_present,
                "allocation_policy": "pack Controller, ComfyUI, and elastic worker lanes from exact leases; never stop existing jobs",
            },
            "recent_events": recent_events,
        }
        request.update(dict(payload))
        request.setdefault("evidence_ids", [str(item.get("event_id")) for item in recent_events if item.get("event_id")])
        return request

    def _invoke_controller_review(self, request: Mapping[str, Any]) -> ReviewDecision:
        reviewer = getattr(self.controller, "review", None)
        if not callable(reviewer):
            raise ControllerProviderError("configured Controller has no structured review method")
        if str(getattr(self.controller, "provider_name", "")) != "vllm":
            result = reviewer(request)
        else:
            self._remote_controller_preflight()
            port = int(getattr(self.controller, "remote_port", 8000))
            with RemotePortForward(self.ssh, port) as tunnel:
                previous_endpoint = str(getattr(self.controller, "endpoint", ""))
                self.controller.endpoint = tunnel.base_url + "/v1/chat/completions"
                try:
                    result = reviewer(request)
                finally:
                    self.controller.endpoint = previous_endpoint
        if isinstance(result, ReviewDecision):
            return result
        if isinstance(result, Mapping):
            return ReviewDecision.from_dict(result)
        raise ControllerProviderError("Controller review returned an invalid decision object")

    def _review_now(self, phase: str, trigger: str, payload: Mapping[str, Any]) -> Optional[ReviewDecision]:
        """Run one serialized review call and apply only safe boundary flags."""

        with self._review_lock:
            if self.review_calls >= int(self.config.max_review_calls):
                self.events.append(
                    "controller_review_skipped",
                    {"phase": phase, "trigger": trigger, "reason": "max_review_calls"},
                )
                return None
            self.review_calls += 1
            request = self._review_request(phase, trigger, payload)
            context_digest = hashlib.sha256(
                json.dumps(request, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            metadata = self._controller_provider_metadata()
            started = time.monotonic()
            self.events.append(
                "controller_review_input",
                {
                    **metadata,
                    "phase": phase,
                    "trigger": trigger,
                    "context_sha256": context_digest,
                    "current_model_id": request.get("current_model_id"),
                    "experiment_id": request.get("experiment_id"),
                    "evidence_ids": list(request.get("evidence_ids") or []),
                },
            )
            try:
                decision = self._invoke_controller_review(request)
            except ControllerUnavailableError as exc:
                self.events.append(
                    "controller_review_unavailable",
                    {**metadata, "phase": phase, "trigger": trigger, "error": str(exc)[:1000], "elapsed_s": time.monotonic() - started},
                )
                self.review_trace.append({"controller_review_call": True, **metadata, "phase": phase, "trigger": trigger, "status": "unavailable"})
                return None
            except (ControllerProviderError, ValueError, TypeError) as exc:
                self.events.append(
                    "controller_review_rejected",
                    {**metadata, "phase": phase, "trigger": trigger, "error": str(exc)[:1000], "elapsed_s": time.monotonic() - started},
                )
                self.review_trace.append({"controller_review_call": True, **metadata, "phase": phase, "trigger": trigger, "status": "rejected"})
                return None
            except Exception as exc:
                self.events.append(
                    "controller_review_unavailable",
                    {**metadata, "phase": phase, "trigger": trigger, "error": str(exc)[:1000], "elapsed_s": time.monotonic() - started},
                )
                self.review_trace.append({"controller_review_call": True, **metadata, "phase": phase, "trigger": trigger, "status": "unavailable"})
                return None
            elapsed = time.monotonic() - started
            applied_action = decision.action
            review_override = None
            if decision.action == "stop" and self._worker_failure_is_recoverable(phase, trigger, payload):
                # A trusted worker failure is experiment evidence, not a
                # campaign-level safety failure.  Keep the full review
                # decision in the audit trail, but route the next cycle back
                # through the Controller so it can change operators.
                applied_action = "replan"
                review_override = "recoverable_worker_failure_replan"
            elif (
                applied_action == "replan"
                and phase == "training"
                and trigger == "worker_completed"
                and str(payload.get("status", "")).strip().lower()
                in {"success", "succeeded", "ok", "completed"}
            ):
                # The review model sometimes calls the normal transition to
                # the next experiment a "replan" even when the trusted worker
                # succeeded and a validated speculative plan is already
                # waiting. Preserve that plan: evaluation is still the next
                # evidence boundary, while an explicit replan from evaluation
                # or a failed worker still invalidates it.
                try:
                    review_state = self._load_campaign_state()
                except Exception:
                    review_state = {}
                if isinstance(review_state.get("prefetched_plan"), Mapping):
                    applied_action = "continue"
                    review_override = "preserve_validated_prefetch_after_success"
            if applied_action == "stop":
                self.review_stop_requested = True
            elif applied_action == "replan":
                self.review_replan_requested = True
            event = self.events.append(
                "controller_review_completed",
                {
                    **metadata,
                    "phase": phase,
                    "trigger": trigger,
                    "action": decision.action,
                    "applied_action": applied_action,
                    "override": review_override,
                    "reason": decision.reason,
                    "evidence_ids": list(decision.evidence_ids),
                    "confidence": decision.confidence,
                    "next_review_after_s": decision.next_review_after_s,
                    "risks": list(decision.risks),
                    "request_id": getattr(self.controller, "last_request_id", None),
                    "elapsed_s": elapsed,
                },
            )
            self.review_trace.append(
                {
                    "controller_review_call": True,
                    **metadata,
                    "phase": phase,
                    "trigger": trigger,
                    "status": "completed",
                    "event_id": event.get("event_id"),
                    "action": decision.action,
                    "applied_action": applied_action,
                    "override": review_override,
                    "elapsed_s": elapsed,
                }
            )
            return decision

    @staticmethod
    def _worker_failure_is_recoverable(
        phase: str,
        trigger: str,
        payload: Mapping[str, Any],
    ) -> bool:
        """Return whether a failed experiment should feed a new plan.

        The long-running campaign must not stop after the first ordinary
        worker/load/timeout failure.  Parent/source tampering and explicit
        safety failures remain terminal review decisions; all other trusted
        worker failures are useful negative experience for the next plan.
        """

        if phase != "training" or trigger != "worker_completed":
            return False
        status = str(payload.get("status", "")).strip().lower()
        if status in {"success", "succeeded", "ok", "completed"}:
            return False
        if status not in {"failed", "error", "timeout", "cancelled"}:
            return False
        failure_type = str(payload.get("failure_type", "")).strip().lower()
        return failure_type not in {
            "parent_modified",
            "source_modified",
            "unsafe_worker",
            "security_violation",
            "data_corruption",
        }

    def _start_review_heartbeat(
        self,
        experiment_id: str,
        phase: str,
        context_factory: Callable[[], Mapping[str, Any]],
    ) -> Tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()
        interval = max(0.01, float(self.config.review_interval_s))

        def monitor() -> None:
            while not stop_event.wait(interval):
                if self.review_stop_requested:
                    break
                try:
                    payload = dict(context_factory())
                    payload["experiment_id"] = str(experiment_id)
                    self._review_now(phase, "heartbeat", payload)
                except Exception as exc:
                    self.events.append(
                        "controller_review_monitor_error",
                        {"experiment_id": str(experiment_id), "phase": phase, "error": str(exc)[:1000]},
                    )

        thread = threading.Thread(
            target=monitor,
            name="harness4h3-review-heartbeat-%s" % experiment_id,
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    @staticmethod
    def _stop_review_heartbeat(handle: Optional[Tuple[threading.Event, threading.Thread]]) -> None:
        if handle is None:
            return
        stop_event, thread = handle
        stop_event.set()
        thread.join(timeout=5.0)

    def _run_with_review_heartbeat(
        self,
        experiment_id: str,
        phase: str,
        context_factory: Callable[[], Mapping[str, Any]],
        operation: Callable[[], Any],
    ) -> Any:
        handle = self._start_review_heartbeat(experiment_id, phase, context_factory)
        try:
            return operation()
        finally:
            self._stop_review_heartbeat(handle)

    def _experience_observation(self, record: ExperienceRecord) -> Any:
        output_state = record.provenance.get("output_state") if isinstance(record.provenance, Mapping) else {}
        output_state = output_state if isinstance(output_state, Mapping) else {}
        checkpoint = output_state.get("checkpoint_path") or record.provenance.get("remote_checkpoint_path")
        checkpoint_hash = record.training.get("child_sha256") if isinstance(record.training, Mapping) else None
        artifacts = []
        checkpoint_value = str(checkpoint) if checkpoint else ""
        if checkpoint_value and not Path(checkpoint_value).is_file() and "://" not in checkpoint_value:
            checkpoint_value = "ssh://%s%s" % (self.config.remote.host, checkpoint_value)
        checkpoint_ref = artifact_reference(
            checkpoint_value,
            sha256=str(checkpoint_hash) if checkpoint_hash else None,
            kind="checkpoint",
            summary={"model_id": record.child_model_id, "immutable_parent": True},
        )
        if checkpoint_ref:
            artifacts.append(checkpoint_ref)
        return make_observation(
            "obs-experience-%s" % record.experience_id,
            "experience",
            record.source_uri,
            record.source_sha256,
            experiment_id=record.experiment_id,
            model_id=record.child_model_id,
            parent_model_id=record.parent_model_id,
            summary={
                "operator": record.operator,
                "operator_args": dict(record.operator_args),
                "status": record.status,
                "training": dict(record.training),
                "evaluation": dict(record.evaluation or {}),
                "decision": dict(record.decision),
                "provenance": {
                    key: value
                    for key, value in dict(record.provenance).items()
                    if key in {"real_worker", "offline_simulation", "quality_scope", "remote_checkpoint_path", "evidence_path"}
                },
            },
            artifacts=artifacts,
        )

    def _replay_experience_observations(self) -> None:
        """Replay all imported M5/M6/M7 results into the common evidence stream."""
        for record in self.experience.read():
            self._append_observation(self._experience_observation(record))

    def _evaluation_observation(self, model_id: str, summary: Mapping[str, Any]) -> Any:
        # The benchmark measurement is immutable, but the surrounding
        # campaign metadata is deliberately enriched after evaluation.  In
        # particular, checkpoint retention and the ComfyUI lease release
        # result are written after the first observation is appended.  Do not
        # let those bookkeeping fields create a second observation for the
        # same measurement: a queued next-training plan would otherwise look
        # stale on resume and trigger an unnecessary LLM call at the boundary.
        digest_summary = dict(summary)
        digest_summary.pop("checkpoint_retention", None)
        recipe = digest_summary.get("benchmark_recipe")
        if isinstance(recipe, Mapping):
            stable_recipe = dict(recipe)
            stable_recipe.pop("comfyui_lease_release", None)
            digest_summary["benchmark_recipe"] = stable_recipe
        encoded = json.dumps(digest_summary, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        artifacts = []
        for run in summary.get("runs", []) if isinstance(summary.get("runs"), list) else []:
            if not isinstance(run, Mapping):
                continue
            for value in run.get("artifacts", []) if isinstance(run.get("artifacts"), (list, tuple)) else []:
                reference = artifact_reference(value, kind="benchmark_artifact", summary={"task_id": run.get("task_id")})
                if reference:
                    artifacts.append(reference)
        hardware = _hardware(summary.get("hardware", {}))
        return make_observation(
            "obs-evaluation-%s-%s" % (model_id, digest[:12]),
            "evaluation",
            "file://%s#%s" % (self.evaluations_path.resolve(), model_id),
            digest,
            model_id=model_id,
            summary={
                "evaluation_id": summary.get("evaluation_id"),
                "system_id": summary.get("system_id"),
                "device_id": summary.get("device_id"),
                "task_split": summary.get("task_split"),
                "benchmark_recipe": dict(summary.get("benchmark_recipe") or {}),
                "evaluator_version": summary.get("evaluator_version"),
                "quality_score": summary.get("quality_score"),
                "quality_metrics": dict(summary.get("quality_metrics") or {}),
                "hardware": {key: hardware.get(key) for key in ("latency_s", "peak_memory_gb", "model_size_gb", "energy_j", "throughput")},
                "hard_gates": dict(summary.get("hard_gates") or {}),
                "feasible": summary.get("feasible"),
                "violations": list(summary.get("violations") or []),
                "task_count": summary.get("task_count"),
                "tasks": [
                    {
                        "task_id": run.get("task_id"),
                        "quality_score": run.get("quality_score"),
                        "failure_type": run.get("failure_type"),
                        "critical_regression": run.get("critical_regression"),
                        "quality_metrics": {
                            key: run.get("quality_metrics", {}).get(key)
                            for key in (
                                "decodable",
                                "semantic_generation_valid",
                                "black_frame_ratio",
                                "motion_score",
                                "stability_score",
                            )
                            if isinstance(run.get("quality_metrics"), Mapping) and key in run.get("quality_metrics", {})
                        },
                    }
                    for run in summary.get("runs", [])
                    if isinstance(run, Mapping)
                ],
            },
            artifacts=artifacts,
        )

    def _replay_evaluation_observations(self) -> None:
        for model_id, summary in self._load_evaluations().items():
            if isinstance(summary, Mapping):
                self._append_observation(self._evaluation_observation(str(model_id), summary))

    def _ensure_goal_observation(self) -> None:
        goal = self._goal_payload()
        encoded = json.dumps(goal, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        self._append_observation(
            make_observation(
                "obs-goal-%s" % self.target.id,
                "goal",
                "controller://goal/%s" % self.target.id,
                digest,
                summary=goal,
            )
        )

    def _import(self) -> ImportSummary:
        importer = RemoteResultImporter(self.experience, self.ssh)
        summary = importer.discover_remote(root=self.config.remote.results_root)
        self._ensure_goal_observation()
        self._replay_experience_observations()
        self._replay_evaluation_observations()
        return summary

    def _register_candidates(self, records: Sequence[ExperienceRecord]) -> Dict[str, ModelCandidate]:
        first = next((item for item in records if item.child_model_id), None)
        root_state = self._root_state(first)
        root_candidate = self.models.initialize(
            ModelCandidate("M0000", None, 0, root_state.checkpoint_path, root_state, None, "baseline", {"remote": self.config.remote.host})
        )
        # ``initialize`` intentionally preserves an existing lineage on
        # resume.  Enrich that immutable root with newly available static
        # checkpoint metadata so a campaign resumed after this code was
        # deployed does not keep reporting model_size_gb=null forever.
        size_bytes = root_state.provenance.get("size_bytes")
        if size_bytes is not None and root_candidate.state.provenance.get("size_bytes") != size_bytes:
            root_candidate = replace(
                root_candidate,
                state=replace(
                    root_candidate.state,
                    provenance={**dict(root_candidate.state.provenance), "size_bytes": int(size_bytes)},
                ),
            )
            self.models.update(root_candidate)
        pending = [item for item in records if item.child_model_id and item.status != "failed"]
        candidates: Dict[str, ModelCandidate] = {item.id: self.models.get(item.id) for item in self.models.lineage()}
        while pending:
            progressed = False
            for record in list(pending):
                child_id = str(record.child_model_id)
                parent_id = record.parent_model_id or "M0000"
                if parent_id not in candidates:
                    continue
                if child_id in candidates:
                    existing = candidates[child_id]
                    parent_candidate = candidates[parent_id]
                    repaired_state = _inherit_quantization_metadata(existing.state, parent_candidate)
                    # Pruning and quantization do not change the sampling
                    # trajectory. Older workers emitted their default
                    # 32-step value when the parent had already been
                    # step-distilled; repair that lineage on resume.
                    if (
                        record.operator != "step_distill"
                        and parent_candidate.state.sampling_steps is not None
                        and existing.state.sampling_steps != parent_candidate.state.sampling_steps
                    ):
                        repaired_state = replace(
                            repaired_state,
                            sampling_steps=parent_candidate.state.sampling_steps,
                        )
                    if repaired_state is not existing.state:
                        existing = replace(existing, state=repaired_state)
                        candidates[child_id] = existing
                        self.models.update(existing)
                    pending.remove(record)
                    progressed = True
                    continue
                state_raw = dict(record.provenance.get("output_state") or {})
                state_raw.update({"model_id": child_id, "parent_model_id": parent_id})
                state_raw.setdefault("checkpoint_path", record.provenance.get("remote_checkpoint_path"))
                state_raw.setdefault("architecture_name", "MiniMax-H3-FL2VA")
                state_raw.setdefault("dtype", "bfloat16")
                if record.operator == "step_distill":
                    state_raw.setdefault("sampling_steps", record.operator_args.get("target_steps"))
                elif candidates[parent_id].state.sampling_steps is not None:
                    state_raw["sampling_steps"] = candidates[parent_id].state.sampling_steps
                state_raw.setdefault("provenance", {})
                state_raw["provenance"] = {**dict(state_raw.get("provenance") or {}), "experience_id": record.experience_id, "remote_source_uri": record.source_uri}
                runtime_state = dict(state_raw.get("runtime_state") or {})
                recipe_runtime = _h3_runtime_optimizations(record.operator, record.operator_args)
                if recipe_runtime:
                    merged_optimizations = dict(runtime_state.get("h3_optimizations") or {})
                    merged_optimizations.update(dict(recipe_runtime["h3_optimizations"]))
                    runtime_state["h3_optimizations"] = merged_optimizations
                if runtime_state:
                    state_raw["runtime_state"] = runtime_state
                if not state_raw.get("checkpoint_path"):
                    pending.remove(record)
                    progressed = True
                    continue
                state = ModelState.from_dict(state_raw)
                state = _inherit_quantization_metadata(state, candidates[parent_id])
                candidate = ModelCandidate(
                    child_id,
                    parent_id,
                    candidates[parent_id].generation + 1,
                    state.checkpoint_path,
                    state,
                    record.experiment_id,
                    "candidate",
                    {"operator": record.operator, "experience_id": record.experience_id},
                )
                try:
                    self.models.create(candidate)
                except ModelCandidateExists:
                    pass
                candidates[child_id] = candidate
                pending.remove(record)
                progressed = True
            if not progressed:
                break
        self._ensure_system_lineage(candidates)
        return candidates

    def _ensure_system_lineage(self, candidates: Mapping[str, ModelCandidate]) -> None:
        """Persist one runtime composition for every imported model candidate.

        The remote worker changes model weights, but the benchmark still
        evaluates a ``ModelCandidate + SystemCandidate`` pair.  Keeping this
        lineage explicit prevents a runtime configuration from masquerading
        as a new model checkpoint and gives the Controller a stable system
        identity on resumed campaigns.
        """

        root = candidates.get("M0000")
        if root is None:
            return
        try:
            self.systems.get("S0000")
        except SystemStoreError:
            self.systems.initialize(
                SystemCandidate.from_model_candidate(
                    "S0000",
                    root,
                    runtime_state={
                        "remote_host": self.config.remote.host,
                        "device_id": self.target.hardware_name,
                        "sampling_steps": root.state.sampling_steps or 32,
                    },
                    status="baseline",
                    metadata={"source": "remote_campaign"},
                )
            )
        for candidate in sorted(candidates.values(), key=lambda item: (item.generation, item.id)):
            if candidate.id == "M0000":
                continue
            if any(item.model_ref == candidate.id for item in self.systems.lineage()):
                continue
            parent_system = self._system_for_model(candidate.parent_id or "M0000")
            if parent_system is None:
                parent_system = self.systems.get("S0000")
            self.systems.create(
                SystemCandidate.from_model_candidate(
                    self.systems.next_id(),
                    candidate,
                    parent_id=parent_system.id,
                    generation=parent_system.generation + 1,
                    runtime_state={
                        "remote_host": self.config.remote.host,
                        "device_id": self.target.hardware_name,
                        **dict(candidate.state.runtime_state),
                        "sampling_steps": candidate.state.sampling_steps or 32,
                    },
                    created_by_experiment_id=candidate.created_by_experiment_id,
                    status="candidate",
                    metadata={"operator": candidate.metadata.get("operator", "unknown")},
                )
            )

    def _system_for_model(self, model_id: Optional[str]) -> Optional[SystemCandidate]:
        if not model_id:
            return None
        matches = [item for item in self.systems.lineage() if item.model_ref == str(model_id)]
        return max(matches, key=lambda item: (item.generation, item.id)) if matches else None

    def _tasks(self, split: Optional[str]) -> List[Task]:
        tasks = load_tasks(self.config.runtime.tasks_path)
        selected = split or "all"
        allowed = set(self.config.benchmark_splits)
        if selected != "all" and selected not in allowed:
            raise ValueError("split %s is not configured for remote campaign" % selected)
        return [task for task in tasks if selected == "all" and task.split in allowed or selected != "all" and task.split == selected]

    def _make_runner(self, base_url: str) -> Any:
        if self.benchmark_factory is not None:
            return self.benchmark_factory(base_url, self.evaluator_factory)
        backend = MiniMaxH3Adapter(
            base_url=base_url,
            task_timeout_s=float(self.config.benchmark_task_timeout_s),
        )
        evaluator = SubprocessEvaluator(self.config.evaluator_command, self.config.evaluator_timeout_s)
        return H3BenchmarkRunner(
            backend,
            evaluator,
            json.loads(self.config.workflow.template.read_text(encoding="utf-8")),
            self.config.workflow,
            self.output_root / "benchmark",
            system_sample_interval_s=self.config.sampling_interval_s,
        )

    def _evaluation_worker_specs(self, tasks: Sequence[Task]) -> Tuple[Any, ...]:
        """Select ComfyUI workers while preserving a training overlap slot.

        In a continuous campaign, fan-out is not the only useful form of
        parallelism.  Three evaluator daemons plus the Controller can occupy
        all four cards while the next H3 worker waits, which is exactly the
        idle gap the pipeline is meant to remove.  For one task, keep one
        evaluator card and leave the remaining cards for the speculative
        successor; for multiple independent tasks, fan the evaluators out
        and let them own the available fill lanes. CPU-only campaigns retain
        the normal task-based fan-out.
        """

        configured = tuple(self.config.comfyui_workers)
        reserved = set(getattr(self.scheduler, "reserved_gpu_indices", ()))
        # The primary ComfyUI API is a configured service lane and remains
        # selectable even when its small resident process is visible.  Extra
        # evaluator daemons, however, must not be placed on a live Controller,
        # worker, or unrelated compute process.  This makes multi-task fan-out
        # follow the same live waterline as training instead of trusting the
        # static ``comfyui_workers`` order.
        eligible: List[Any] = []
        live_compute: Optional[set[int]] = set()
        controller_reserved: set[int] = set()
        live_memory: Mapping[int, Tuple[int, Optional[int]]] = {}
        try:
            live_memory, live_processes = self.scheduler.snapshot()
            process_indices = getattr(self.scheduler, "last_compute_gpu_indices", None)
            if live_processes.strip() and process_indices is None:
                live_compute = set(range(int(getattr(self.scheduler, "gpu_count", 4))))
            else:
                live_compute = set(int(index) for index in (process_indices or ()))
            controller_reader = getattr(self.scheduler, "controller_reserved_gpu_indices", ())
            controller_reserved = set(int(index) for index in controller_reader)
        except (OSError, RemoteError, TypeError, ValueError):
            # Evaluation can still use its established primary daemon when a
            # diagnostic snapshot is unavailable; fail closed for all extra
            # daemons rather than guessing which secondary card is safe.
            live_compute = set(range(int(getattr(self.scheduler, "gpu_count", 4))))
            live_memory = {}
            controller_reserved = set(range(int(getattr(self.scheduler, "gpu_count", 4))))
        if configured:
            primary = configured[0]
            if self._comfyui_primary_gpu_is_safe(primary, live_compute):
                eligible.append(primary)
            else:
                self.events.append(
                    "evaluation_worker_skipped",
                    {
                        "gpu_index": int(primary.gpu_index),
                        "port": int(primary.port),
                        "reason": "foreign_compute_process_on_primary_comfyui_gpu",
                    },
                )
        waterline = getattr(self.scheduler, "meets_memory_waterline", None)
        for worker in configured[1:]:
            index = int(worker.gpu_index)
            if index in reserved or index in live_compute or index in controller_reserved:
                continue
            if live_memory and callable(waterline) and not waterline(index, live_memory):
                continue
            eligible.append(worker)
        # The resident GPU0 service may predate the installed H3 extension.
        # When a safe secondary lane is available, prefer it once so the live
        # registry can be verified by the exact ComfyUI process that evaluates
        # the candidate.  If no secondary is safe, retain the primary for a
        # normal baseline evaluation; optional LPL/TDTM planning remains
        # fail-closed until a fresh worker proves the nodes are registered.
        optional_caps = (
            self.optimization_capabilities
            if isinstance(self.optimization_capabilities, Mapping)
            else {}
        )
        optional_verified = all(
            isinstance(optional_caps.get(name), Mapping)
            and optional_caps[name].get("safe_to_plan") is True
            for name in ("lpl", "tdtm")
        )
        secondary_eligible = [worker for worker in eligible if int(worker.gpu_index) != 0]
        if (
            isinstance(self.ssh, SSHClient)
            and not optional_verified
            and secondary_eligible
            and any(int(worker.gpu_index) == 0 for worker in eligible)
        ):
            primary_workers = [worker for worker in eligible if int(worker.gpu_index) == 0]
            eligible = secondary_eligible + primary_workers
            self.events.append(
                "evaluation_worker_preferred",
                {
                    "gpu_index": int(secondary_eligible[0].gpu_index),
                    "port": int(secondary_eligible[0].port),
                    "reason": "primary_comfyui_predates_h3_extension",
                    "optional_h3_capabilities_verified": False,
                },
            )
        configured = tuple(eligible)
        free_gpu_count = len(configured)
        configured_count = len(configured)
        gpu_training_available = any(
            self.config.worker.operator_launchers.get(operator, "torchrun") == "torchrun"
            for operator in self.config.worker.allowed_operators
        )
        overlap_cap_applied = bool(
            self.config.pipeline_enabled
            and self.config.worker.enabled
            and gpu_training_available
        )
        # A single benchmark task benefits most from leaving two cards for a
        # speculative distributed worker.  With multiple independent tasks,
        # the evaluators themselves are the useful fill work; fan them out
        # instead of reserving cards for a worker that cannot meet its
        # two-GPU minimum alongside the evaluator set.
        single_task_overlap = overlap_cap_applied and len(tasks) <= 1
        if single_task_overlap:
            # The trusted worker requires at least two GPUs and the launcher
            # can elastically use all remaining cards.  One evaluator card is
            # therefore the largest safe default for a 4-GPU overlap.
            configured_count = min(configured_count, 1)
        count = evaluation_gpu_count(len(tasks), configured_count, free_gpu_count)
        selected_evaluation_gpus = {
            int(worker.gpu_index) for worker in configured[:count]
        }
        try:
            training_candidate_gpu_indices = [
                index
                for index in range(int(getattr(self.scheduler, "gpu_count", 4)))
                if (
                    index not in reserved
                    and index not in live_compute
                    and index not in controller_reserved
                    and index not in selected_evaluation_gpus
                    and (
                        not live_memory
                        or not callable(waterline)
                        or waterline(index, live_memory)
                    )
                )
            ]
        except (OSError, RemoteError, TypeError, ValueError):
            training_candidate_gpu_indices = []
        if overlap_cap_applied:
            self.events.append(
                "evaluation_overlap_budget",
                {
                    "evaluation_workers": count,
                    "configured_workers": len(configured),
                    "free_gpu_count": free_gpu_count,
                    "reserved_gpu_indices": sorted(reserved),
                    "training_overlap_reserved": (
                        len(training_candidate_gpu_indices) if single_task_overlap else 0
                    ),
                    "training_candidate_gpu_indices": training_candidate_gpu_indices[:4],
                    "training_candidate_gpu_count": len(training_candidate_gpu_indices),
                    "idle_reason": (
                        "ready_for_two_gpu_speculative_worker"
                        if len(training_candidate_gpu_indices) >= 2
                        else "fewer_than_two_safe_training_gpus"
                    ),
                    "reason": (
                        "preserve_cards_for_speculative_distributed_worker"
                        if single_task_overlap
                        else "fan_out_independent_evaluation_tasks"
                    ),
                },
            )
        return configured[:count]

    def _comfyui_primary_gpu_is_safe(
        self,
        worker: Any,
        live_compute: Sequence[int],
    ) -> bool:
        """Allow the resident primary ComfyUI only when no foreign PID shares it.

        GPU0 is a long-lived service lane, so its own small CUDA context is
        expected.  A different compute PID is not shareable: H3 evaluation
        can approach the full L40 memory budget even when the foreign job
        reports low instantaneous utilization.  Test transports do not have
        a process namespace and retain the historical primary-lane behavior.
        """

        if not isinstance(self.ssh, SSHClient):
            return True
        if int(getattr(worker, "gpu_index", -1)) != 0:
            return True
        if 0 not in {int(index) for index in live_compute}:
            return True
        process_query = self.ssh.run(
            (
                "nvidia-smi",
                "--id=0",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ),
            check=False,
        )
        if int(getattr(process_query, "returncode", 1)) != 0:
            return False
        pids = []
        for raw in str(getattr(process_query, "stdout", "")).splitlines():
            value = raw.strip()
            if value.isdigit() and int(value) > 0:
                pids.append(int(value))
        if not pids:
            # The scheduler saw a process during its snapshot, but the
            # per-GPU PID query has already gone quiet.  A short-lived race
            # is safe to re-check at lease/worker launch; do not reject it.
            return True
        expected_port = int(getattr(self.config.remote, "comfyui_port", worker.port))
        for pid in pids:
            probe = self.ssh.run(("ps", "-p", str(pid), "-o", "args="), check=False)
            if int(getattr(probe, "returncode", 1)) != 0:
                return False
            command = str(getattr(probe, "stdout", "")).strip()
            has_port = (
                "--port %d" % expected_port in command
                or "--port=%d" % expected_port in command
            )
            if "main.py" not in command or not has_port:
                return False
        return True

    def _comfyui_lease_for(self, worker: Any) -> ComfyUILeaseManager:
        key = (int(worker.gpu_index), int(worker.port))
        lease = self.comfyui_leases.get(key)
        if lease is not None:
            return lease
        campaign_root = str(self.config.remote.resolved_campaign_root)
        # Keep GPU 0's marker compatible with controller-wait-launch.sh's
        # historical primary path; secondary workers use a suffixed marker.
        lease_name = (
            ".comfyui-gpu-lease.json"
            if int(worker.gpu_index) == 0
            else ".comfyui-gpu-lease-%d.json" % int(worker.gpu_index)
        )
        lease_path = str(Path(campaign_root) / lease_name)
        lease = ComfyUILeaseManager(
            self.ssh,
            self.scheduler,
            port=int(worker.port),
            gpu_index=int(worker.gpu_index),
            release_wait_s=float(self.config.comfyui_idle_shutdown_s),
            lease_path=lease_path,
        )
        self.comfyui_leases[key] = lease
        return lease

    def _ensure_comfyui_worker(self, worker: Any) -> None:
        """Start an on-demand evaluator daemon for a non-primary port."""

        if not isinstance(self.ssh, SSHClient):
            return
        port = int(worker.port)
        endpoint = "http://127.0.0.1:%d/system_stats" % port
        ready = self.ssh.run(("curl", "-fsS", "--max-time", "5", endpoint), check=False)
        if int(getattr(ready, "returncode", 1)) == 0:
            return
        lease = self._comfyui_lease_for(worker)
        remote_root = str(self.config.remote.harness_root)
        launcher = str(Path(remote_root) / "tools" / "comfyui-wait-launch.sh")
        log_path = str(
            Path(self.config.remote.resolved_campaign_root)
            / ("comfyui-gpu%d-port%d.log" % (int(worker.gpu_index), port))
        )
        env = {
            "COMFY_ROOT": str(self.config.remote.comfyui_root),
            "COMFY_PYTHON": str(self.config.remote.python),
            "COMFY_GPU_INDEX": str(int(worker.gpu_index)),
            "COMFY_PORT": str(port),
            "COMFY_LOG": log_path,
            "COMFY_LEASE_MAX_AGE_SECONDS": str(
                max(1, int(float(getattr(self.config, "comfyui_lease_max_age_s", 21600.0))))
            ),
        }
        if self.config.comfyui_process_policy == "on_demand" and lease.lease_path:
            # The launcher must not turn into an always-on GPU occupant after
            # the benchmark lease has been released.  It watches this exact
            # campaign-owned marker and exits when the lease disappears.
            env["COMFY_LEASE_FILE"] = str(lease.lease_path)
        assignments = " ".join("%s=%s" % (key, shlex.quote(value)) for key, value in env.items())
        command = "nohup env %s bash %s >/dev/null 2>&1 </dev/null & printf '%%s\\n' $!" % (
            assignments,
            shlex.quote(launcher),
        )
        launch_result = self.ssh.run(("bash", "-lc", command), check=False)
        launch_lines = [line.strip() for line in str(getattr(launch_result, "stdout", "")).splitlines() if line.strip()]
        try:
            owned_pid = int(launch_lines[-1]) if launch_lines else 0
        except ValueError:
            owned_pid = 0
        if owned_pid > 0:
            lease.set_owned_process(owned_pid)
            self.events.append(
                "comfyui_worker_owned",
                {"gpu_index": int(worker.gpu_index), "port": port, "pid": owned_pid},
            )
        elif self.config.comfyui_process_policy == "on_demand":
            self.events.append(
                "comfyui_worker_ownership_unavailable",
                {
                    "gpu_index": int(worker.gpu_index),
                    "port": port,
                    "reason": "launcher_pid_not_returned",
                },
            )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            probe = self.ssh.run(("curl", "-fsS", "--max-time", "5", endpoint), check=False)
            if int(getattr(probe, "returncode", 1)) == 0:
                self.events.append(
                    "comfyui_worker_ready",
                    {"gpu_index": int(worker.gpu_index), "port": port, "endpoint": endpoint},
                )
                return
            time.sleep(1.0)
        raise RemoteError("ComfyUI worker did not become ready at %s" % endpoint)

    def _benchmark_tunnel(self, port: int) -> Any:
        if self.tunnel_factory is not None:
            return self.tunnel_factory(self.ssh)
        return RemotePortForward(self.ssh, int(port))

    def _evaluate(
        self,
        candidate: ModelCandidate,
        parent_summary: Optional[Mapping[str, Any]],
        split: Optional[str],
        system: Optional[SystemCandidate] = None,
        on_benchmark_reserved: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        tasks = self._tasks(split)
        if not tasks:
            raise ValueError("no tasks selected for remote campaign")
        self._set_pipeline_stage(
            PipelineStage.EVALUATING,
            evaluation_model_id=candidate.id,
            overlap_with_model_id=candidate.id if on_benchmark_reserved is not None else None,
        )
        # A worker result and its append-only experience record can outlive a
        # rejected checkpoint.  Never send a dangling model link to ComfyUI:
        # that turns an artifact-lifecycle problem into a misleading quality
        # failure after spending a full evaluation window.
        if candidate.id != "M0000":
            checkpoint_probe = self.ssh.run(("test", "-e", candidate.checkpoint_path), check=False)
            if int(getattr(checkpoint_probe, "returncode", 1)) != 0:
                self.events.append(
                    "evaluation_skipped_missing_checkpoint",
                    {
                        "model_id": candidate.id,
                        "checkpoint_path": candidate.checkpoint_path,
                        "split": split or "all",
                        "reason": "candidate_record_survived_checkpoint_cleanup",
                    },
                )
                summary = {
                    "model_id": candidate.id,
                    "task_count": len(tasks),
                    "quality_score": None,
                    "quality_metrics": {
                        "backend": "comfyui",
                        "successful_tasks": 0,
                        "failed_tasks": len(tasks),
                        "failure_types": {task.id: "checkpoint_missing" for task in tasks},
                        "task_scores": {task.id: None for task in tasks},
                    },
                    "hardware": {
                        "latency_s": None,
                        "peak_memory_gb": None,
                        "model_size_gb": None,
                        "energy_j": None,
                        "throughput": None,
                        "thermal": None,
                    },
                    "feasible": False,
                    "violations": ["checkpoint_missing"],
                    "runs": [
                        {
                            "task_id": task.id,
                            "status": "failed",
                            "quality_score": None,
                            "failure_type": "checkpoint_missing",
                            "message": "candidate checkpoint does not exist",
                            "wall_time_s": 0.0,
                        }
                        for task in tasks
                    ],
                    "hard_gates": {
                        "generation_valid": False,
                        "decode_success": False,
                        "no_critical_temporal_collapse": False,
                        "quality_gate": False,
                        "efficiency_gate": False,
                        "target_feasible": False,
                    },
                    "system_id": system.id if system is not None else None,
                    "device_id": system.device_id if system is not None else self.target.hardware_name,
                    "task_split": split or "all",
                    "benchmark_workers": [],
                    "benchmark_recipe": {
                        "target_profile_id": self.target.id,
                        "split": split or "all",
                        "workflow_template": str(self.config.workflow.template),
                        "quality_scope": self.config.quality_scope,
                        "power_target_w": self.config.power_target_w,
                        "comfyui_cache_policy": self.config.comfyui_cache_policy,
                        "comfyui_lease_state": "not_reserved_checkpoint_missing",
                    },
                    "quality_scope": self.config.quality_scope,
                }
                summary["evaluation_signature"] = self._evaluation_signature(split)
                return summary
        # The next training boundary uses this candidate as its parent in the
        # normal closed loop.  Start local staging before ComfyUI allocates a
        # GPU and before benchmark generation, then join it before acquiring
        # the training lease in ``_train_one``.
        self._start_checkpoint_prefetch(candidate.checkpoint_path, reason="candidate_evaluation")
        grouped: List[Tuple[str, List[Task]]] = []
        for name in self.config.benchmark_splits if split in (None, "all") else (split,):
            group = [task for task in tasks if task.split == name]
            if group:
                grouped.append((name, group))
        max_group_size = max((len(group) for _, group in grouped), default=len(tasks))
        worker_specs = self._evaluation_worker_specs(tasks[:max_group_size])
        if not worker_specs:
            # A full-card or otherwise overlapping campaign-owned Controller
            # can make the first evaluator selection appear empty.  Do not
            # wait forever in that state: no ComfyUI lease exists yet, so the
            # Controller launcher would never observe an evaluation window and
            # yield its cards.  Release only the exact launcher-owned process,
            # then take a fresh scheduler snapshot.  Foreign jobs remain
            # fail-closed and still produce the normal resource wait below.
            controller_reserved: Tuple[int, ...] = ()
            try:
                value = getattr(self.scheduler, "controller_reserved_gpu_indices", ())
                value = value() if callable(value) else value
                controller_reserved = tuple(
                    sorted(set(int(index) for index in (value or ())))
                )
            except (OSError, RemoteError, TypeError, ValueError):
                controller_reserved = ()
            if controller_reserved:
                release = self._request_controller_release(
                    "evaluation_gpu_allocation",
                    wait_s=30.0,
                )
                release_status = str(release.get("status", "")).strip().lower()
                self.events.append(
                    "evaluation_controller_release",
                    {
                        "controller_gpus": list(controller_reserved),
                        "status": release_status or "unknown",
                        "release": dict(release),
                        "reason": "no_evaluator_gpu_before_lease",
                    },
                )
                if release_status == "released":
                    worker_specs = self._evaluation_worker_specs(tasks[:max_group_size])
        if not worker_specs:
            self.events.append(
                "evaluation_waiting_for_resources",
                {
                    "model_id": candidate.id,
                    "split": split or "all",
                    "task_count": len(tasks),
                    "reason": "no_safe_comfyui_evaluator_gpu",
                },
            )
            raise RemoteResourceWaitError(
                "no safe ComfyUI evaluator GPU is currently available"
            )
        self._prepare_comfyui_leases(worker_specs)
        self._active_evaluation_gpu_indices = tuple(
            int(worker.gpu_index) for worker in worker_specs
        )
        self._record_lane_allocation(
            "evaluating",
            comfyui_workers=worker_specs,
            reason="benchmark_lanes_reserved",
        )
        self.events.append(
            "evaluation_started",
            {
                "model_id": candidate.id,
                "split": split or "all",
                "task_count": len(tasks),
                "evaluation_workers": len(worker_specs),
                "reason": "independent benchmark phase",
            },
        )
        # Start every selected non-primary daemon.  The first selected worker
        # may itself be a secondary when the resident GPU0 service predates
        # the H3 extension, so iterating from index 1 is not sufficient after
        # the live-capability preference reorder.
        for worker in worker_specs:
            if int(worker.gpu_index) != 0:
                self._ensure_comfyui_worker(worker)
        # Refresh the capability gate against the exact evaluator process
        # that will execute this benchmark.  This turns an installed
        # secondary extension into a safe next-round LLM capability without
        # pretending that the older resident GPU0 process has reloaded it.
        live_capabilities = self._remote_comfyui_worker_capabilities(worker_specs[0])
        if live_capabilities is not None:
            self.optimization_capabilities = live_capabilities
        # Start the overlap callback only after the selected evaluator is
        # ready and its live node registry has been measured.  The next plan
        # therefore sees the real LPL/TDTM capability instead of racing a
        # secondary ComfyUI startup with the Controller request.  The
        # evaluation_started event is intentionally emitted before this
        # callback so a worker launched by the callback is measured as an
        # evaluation overlap rather than being misclassified as a missing
        # successor.
        if on_benchmark_reserved is not None:
            on_benchmark_reserved()
        parent_quality = parent_summary.get("quality_score") if parent_summary else None
        parent_hardware = HardwareMetrics(**{key: value for key, value in _hardware(parent_summary.get("hardware", {})).items() if key in HardwareMetrics.__dataclass_fields__}) if parent_summary else None
        self._review_now(
            "benchmarking",
            "evaluation_started",
            {"model_id": candidate.id, "split": split or "all", "task_count": len(tasks)},
        )
        with ExitStack() as tunnel_stack:
            tunnels = [tunnel_stack.enter_context(self._benchmark_tunnel(int(worker.port))) for worker in worker_specs]
            runners = [self._make_runner(tunnel.base_url) for tunnel in tunnels]
            summaries: List[Dict[str, Any]] = []
            for name, group in grouped:
                if candidate.id != "M0000":
                    deployment_link = self.ssh.ensure_model_link(candidate.checkpoint_path, candidate.id)
                    if getattr(self.ssh, "last_model_link_action", None) == "replaced_stale_symlink":
                        self.events.append(
                            "model_link_replaced",
                            {
                                "model_id": candidate.id,
                                "checkpoint_path": candidate.checkpoint_path,
                                "deployment_link": deployment_link,
                                "reason": "stale_symlink_for_resumed_candidate_id",
                            },
                        )
                def run_group(runner: Any, group_tasks: Sequence[Task]) -> Any:
                    power = RemotePowerSampler(self.ssh, self.config.sampling_interval_s)
                    return runner.run(
                        candidate.state,
                        group_tasks,
                        baseline_quality=float(parent_quality) if parent_quality is not None else None,
                        target=self.target,
                        baseline_hardware=parent_hardware,
                        efficiency_thresholds=self.config.efficiency_thresholds,
                        black_frame_rate_threshold=self.config.black_frame_rate_threshold,
                        reset_backend_before_run=self.config.reset_backend_before_run,
                        power_sampler=power,
                        system=system,
                        device_id=system.device_id if system is not None else self.target.hardware_name,
                        task_split=name,
                        benchmark_recipe={
                            "target_profile_id": self.target.id,
                            "split": name,
                            "workflow_template": str(self.config.workflow.template),
                            "quality_scope": self.config.quality_scope,
                            "power_target_w": self.config.power_target_w,
                            "comfyui_cache_policy": self.config.comfyui_cache_policy,
                            "comfyui_lease_state": "reserved_for_benchmark",
                            "optimization_capabilities": copy.deepcopy(self.optimization_capabilities),
                        },
                    )
                batches = [list(group[index::len(runners)]) for index in range(len(runners))]
                batches = [batch for batch in batches if batch]
                if len(batches) == 1:
                    group_summaries = [
                        _summary_dict(
                            self._run_with_review_heartbeat(
                                candidate.id,
                                "benchmarking",
                                lambda: {
                                    "model_id": candidate.id,
                                    "split": name,
                                    "task_count": len(group),
                                    "telemetry": self._collect_worker_telemetry("benchmark-%s" % candidate.id),
                                },
                                lambda: run_group(runners[0], batches[0]),
                            )
                        )
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=len(batches), thread_name_prefix="h3-eval") as pool:
                        futures = [pool.submit(run_group, runners[index], batch) for index, batch in enumerate(batches)]
                        group_summaries = [_summary_dict(future.result()) for future in futures]
                summary = self._aggregate_summaries(candidate.id, group_summaries)
                summary["task_split"] = name
                summary["benchmark_workers"] = [
                    {"gpu_index": int(worker_specs[index].gpu_index), "port": int(worker_specs[index].port), "task_ids": [task.id for task in batches[index]]}
                    for index in range(len(batches))
                ]
                summaries.append(summary)
                self.events.append(
                    "evaluation_group_completed",
                    {
                        "model_id": candidate.id,
                        "split": name,
                        "task_count": len(group),
                        "summary": {
                            "quality_score": summary.get("quality_score"),
                            "hardware": _hardware(summary.get("hardware", {})),
                            "hard_gates": dict(summary.get("hard_gates") or {}),
                        },
                    },
                )
                self._review_now(
                    "benchmarking",
                    "evaluation_group_completed",
                    {
                        "model_id": candidate.id,
                        "split": name,
                        "task_count": len(group),
                        "summary": {
                            "quality_score": summary.get("quality_score"),
                            "hardware": _hardware(summary.get("hardware", {})),
                            "hard_gates": dict(summary.get("hard_gates") or {}),
                        },
                    },
                )
                hard_gates = summary.get("hard_gates") or {}
                if name == "sanity" and not all(hard_gates.get(key) is True for key in ("generation_valid", "decode_success", "no_critical_temporal_collapse")):
                    break
        aggregated = self._aggregate_summaries(candidate.id, summaries)
        aggregated["benchmark_workers"] = [
            dict(worker)
            for summary in summaries
            for worker in (summary.get("benchmark_workers") or [])
            if isinstance(worker, Mapping)
        ]
        if system is not None:
            aggregated["system_id"] = system.id
            aggregated["device_id"] = system.device_id or self.target.hardware_name
        aggregated["task_split"] = split or "all"
        aggregated["benchmark_recipe"] = {
            "target_profile_id": self.target.id,
            "split": split or "all",
            "workflow_template": str(self.config.workflow.template),
            "quality_scope": self.config.quality_scope,
            "power_target_w": self.config.power_target_w,
            "comfyui_cache_policy": self.config.comfyui_cache_policy,
            "comfyui_lease_state": "reserved_for_benchmark",
            "optimization_capabilities": copy.deepcopy(self.optimization_capabilities),
        }
        aggregated["quality_scope"] = self.config.quality_scope
        evaluation_record = _evaluation_result(
            aggregated,
            model_id=candidate.id,
            system_id=system.id if system is not None else aggregated.get("system_id"),
            device_id=system.device_id if system is not None else self.target.hardware_name,
            task_split=split or "all",
        )
        if evaluation_record is not None:
            aggregated["evaluation_record"] = evaluation_record.to_dict()
            aggregated["evaluation_id"] = evaluation_record.evaluation_id
        aggregated["evaluation_signature"] = self._evaluation_signature(split)
        return aggregated

    @staticmethod
    def _aggregate_summaries(model_id: str, summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not summaries:
            return {"model_id": model_id, "task_count": 0, "quality_score": None, "hardware": {}, "hard_gates": {}}
        runs: List[Mapping[str, Any]] = []
        scores: List[float] = []
        hardware_values = []
        quality_metrics = []
        hard_gate_values: Dict[str, List[Any]] = {}
        system_ids = []
        device_ids = []
        task_splits = []
        evaluator_versions = []
        for summary in summaries:
            runs.extend(summary.get("runs") or [])
            if isinstance(summary.get("quality_score"), (int, float)):
                scores.append(float(summary["quality_score"]))
            hardware_values.append(_hardware(summary.get("hardware", {})))
            quality_metrics.append(summary.get("quality_metrics") or {})
            if summary.get("system_id"):
                system_ids.append(str(summary["system_id"]))
            if summary.get("device_id"):
                device_ids.append(str(summary["device_id"]))
            if summary.get("task_split"):
                task_splits.append(str(summary["task_split"]))
            if summary.get("evaluator_version"):
                evaluator_versions.append(str(summary["evaluator_version"]))
            for key, value in (summary.get("hard_gates") or {}).items():
                hard_gate_values.setdefault(str(key), []).append(value)
        def mean_value(name: str) -> Optional[float]:
            values = [float(item[name]) for item in hardware_values if isinstance(item.get(name), (int, float))]
            return sum(values) / len(values) if values else None
        energy_values = [float(item["energy_j"]) for item in hardware_values if isinstance(item.get("energy_j"), (int, float))]
        hardware = {
            "latency_s": mean_value("latency_s"),
            "peak_memory_gb": max((float(item["peak_memory_gb"]) for item in hardware_values if isinstance(item.get("peak_memory_gb"), (int, float))), default=None),
            "model_size_gb": mean_value("model_size_gb"),
            "energy_j": sum(energy_values) if energy_values and len(energy_values) == len(hardware_values) else None,
            "throughput": mean_value("throughput"),
            "thermal": None,
        }
        hard_gates = {
            key: (all(value is True for value in values) if values and all(isinstance(value, bool) for value in values) else values[-1])
            for key, values in hard_gate_values.items()
        }
        return {
            "model_id": model_id,
            "task_count": len(runs),
            "quality_score": sum(scores) / len(scores) if scores else None,
            "quality_metrics": {"splits": quality_metrics, "successful_tasks": sum(1 for run in runs if run.get("quality_score") is not None)},
            "hardware": hardware,
            "feasible": all(bool(summary.get("feasible")) for summary in summaries) if summaries else False,
            "violations": [item for summary in summaries for item in summary.get("violations", [])],
            "runs": runs,
            "hard_gates": hard_gates,
            "system_id": system_ids[0] if system_ids else None,
            "device_id": device_ids[0] if device_ids else None,
            "task_split": task_splits[0] if task_splits else None,
            "evaluator_version": evaluator_versions[0] if evaluator_versions else "unknown",
        }

    def _append_evaluated_experience(self, record: ExperienceRecord, summary: Mapping[str, Any], decision: DecisionResult) -> ExperienceRecord:
        payload = json.dumps({"summary": summary, "decision": decision.to_dict()}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        updated = ExperienceRecord(
            experience_id="xp-eval-%s-%s" % (record.child_model_id or record.experiment_id, digest[:12]),
            source_uri=record.source_uri + "#benchmark/" + self.target.id,
            source_sha256=digest,
            source_kind="benchmark_result",
            experiment_id=record.experiment_id,
            parent_model_id=record.parent_model_id,
            child_model_id=record.child_model_id,
            operator=record.operator,
            operator_args=record.operator_args,
            training=record.training,
            evaluation=summary,
            decision=decision.to_dict(),
            reward=decision.reward,
            status=decision.status,
            provenance={**dict(record.provenance), "benchmark_target": self.target.id, "quality_scope": self.config.quality_scope},
            created_at=record.created_at,
        )
        self.experience.append(updated)
        return updated

    def _evaluate_round_gate(
        self,
        record: ExperienceRecord,
        summary: Mapping[str, Any],
        *,
        retention: Optional[Mapping[str, Any]] = None,
    ) -> RoundGateResult:
        """Evaluate fixed round evidence and persist its bounded audit result."""

        training = dict(record.training)
        evaluation = dict(summary)
        hard_gates = evaluation.get("hard_gates")
        if "critical_regression" not in evaluation and isinstance(hard_gates, Mapping):
            evaluation["critical_regression"] = (
                hard_gates.get("no_critical_temporal_collapse") is False
            )
        raw_lane = training.get("lane_evidence")
        lane = dict(raw_lane) if isinstance(raw_lane, Mapping) else {
            "status": "unverified",
            "disjoint": False,
            "reason": "worker_result_did_not_publish_lane_evidence",
        }
        worker_status = training.get("status")
        if not worker_status:
            worker_status = "failed" if record.status == "failed" else "success"
        worker = {
            "status": worker_status,
            "experiment_id": record.experiment_id,
            "operator": record.operator,
            "operator_args": dict(record.operator_args),
            "metrics": training,
        }
        result = evaluate_round_gate(
            worker_result=worker,
            evaluation=evaluation,
            target={"min_quality_score": self.target.min_quality_score},
            capability_evidence=(
                self.optimization_capabilities
                if isinstance(self.optimization_capabilities, Mapping)
                else {}
            ),
            lane_evidence=lane,
            parent_digest=(
                training.get("parent_sha256_before")
                or training.get("parent_sha256")
            ),
            child_digest=training.get("child_sha256"),
            retention=dict(retention or {}),
        )
        state = self._load_campaign_state()
        state["round_gate"] = {
            "experiment_id": record.experiment_id,
            "model_id": record.child_model_id,
            "result": result.to_dict(),
            "authoritative": bool(
                training.get("real_worker") is True
                and isinstance(raw_lane, Mapping)
                and bool(raw_lane.get("worker_gpus"))
            ),
            "updated_at": time.time(),
        }
        self._save_campaign_state(state)
        self.events.append(
            "round_gate_evaluated",
            {
                "experiment_id": record.experiment_id,
                "model_id": record.child_model_id,
                "status": result.status,
                "reasons": list(result.reasons),
                "authoritative": state["round_gate"]["authoritative"],
                "retention": dict(retention or {}),
            },
        )
        return result

    @staticmethod
    def _block_decision_for_round_gate(
        decision: DecisionResult,
        gate: RoundGateResult,
    ) -> DecisionResult:
        """Keep legacy decision fields while making a failed fixed gate final."""

        violations = tuple(dict.fromkeys(tuple(decision.violations) + tuple(gate.reasons)))
        return replace(
            decision,
            status="rejected",
            accepted=False,
            pareto_eligible=False,
            violations=violations,
            continuation_status="reject",
            advance=False,
        )

    def _record_evaluation(
        self,
        candidate: ModelCandidate,
        parent: ModelCandidate,
        record: ExperienceRecord,
        summary: Mapping[str, Any],
        decision: DecisionResult,
        fingerprint: str = "",
        repeat_for_statistics: bool = False,
    ) -> None:
        evaluation = _evaluation_result(summary)
        front_ids = [entry.candidate_id for entry in self.pareto.front()]
        candidate_system = self._system_for_model(candidate.id)
        parent_system = self._system_for_model(parent.id)
        # Pareto identity is the evaluated composition, not merely the
        # checkpoint.  This lets the archive retain multiple runtime systems
        # for one model while keeping model-changing candidates distinct via
        # their generated SystemCandidate.
        pareto_id = candidate_system.id if candidate_system is not None else candidate.id
        if evaluation is not None and decision.pareto_eligible:
            if pareto_id not in {entry.candidate_id for entry in self.pareto.entries()}:
                front_ids = [entry.candidate_id for entry in self.pareto.update(pareto_id, evaluation)]
            else:
                front_ids = [entry.candidate_id for entry in self.pareto.front()]
            self.models.set_active(candidate.id)
            if candidate_system is not None:
                self.systems.set_active(candidate_system.id)
        self.experiments.append(
            ExperimentRecord(
                experiment_id="remote-eval-%s" % candidate.id,
                session_id="remote-h3",
                target_profile_id=self.target.id,
                controller={"provider": getattr(self.controller, "provider_name", "fixed"), "model": getattr(self.controller, "model_name", "fixed")},
                parent_model_id=parent.id,
                child_model_id=candidate.id,
                state_digest=_state_digest(parent.state),
                diagnosis={"source_experience_id": record.experience_id},
                plan={"operator": record.operator, "operator_args": dict(record.operator_args)},
                execution={"status": "benchmark_complete", "quality_scope": self.config.quality_scope},
                training_logs=[str(record.provenance.get("remote_checkpoint_path", ""))],
                cost={"wall_time_s": sum(float(run.get("wall_time_s", 0.0)) for run in summary.get("runs", [])), "gpu_hours": 0.0, "controller_calls": 0},
                evaluation=dict(summary),
                failure_type=None if decision.accepted else (decision.violations[0] if decision.violations else "acceptance_rejected"),
                decision=decision.to_dict(),
                pareto_update={"front": front_ids},
                created_at=record.created_at,
                parent_system_id=parent_system.id if parent_system else None,
                child_system_id=candidate_system.id if candidate_system else None,
                system_state_digest=_state_digest(
                    parent_system.to_dict()
                    if parent_system is not None
                    else {}
                ),
                fingerprint=fingerprint,
                repeat_for_statistics=repeat_for_statistics,
            )
        )

    def _persist_evaluation_state(self, candidate: ModelCandidate, summary: Mapping[str, Any]) -> ModelCandidate:
        """Make the latest measured evidence available to the next plan."""

        hardware = _hardware(summary.get("hardware", {}))
        measured = dict(candidate.state.measured_metrics)
        if isinstance(summary.get("quality_score"), (int, float)):
            measured["quality_score"] = float(summary["quality_score"])
        for key in ("latency_s", "peak_memory_gb", "model_size_gb", "energy_j", "throughput"):
            if isinstance(hardware.get(key), (int, float)):
                measured[key] = float(hardware[key])
        state = replace(
            candidate.state,
            measured_metrics=measured,
            runtime_state={
                **dict(candidate.state.runtime_state),
                "last_evaluation_signature": summary.get("evaluation_signature"),
                "quality_scope": self.config.quality_scope,
            },
        )
        updated = replace(candidate, state=state)
        self.models.update(updated)
        return updated

    @staticmethod
    def _worker_operator_args(operator: str, args: Mapping[str, Any], parent: ModelCandidate) -> Dict[str, Any]:
        """Check the worker contract without changing Controller arguments."""

        normalized = dict(args)
        if operator == "step_distill":
            source_steps = int(parent.state.sampling_steps or 32)
            target_steps = int(normalized.get("target_steps", 0))
            if source_steps <= 1 or source_steps % 2 or target_steps <= 0 or source_steps != 2 * target_steps:
                raise ValueError("Controller must set step_distill target_steps so source_steps=2*target_steps")
        return normalized

    def _validate_h3_runtime_recipe(
        self,
        operator: str,
        args: Mapping[str, Any],
        parent: ModelCandidate,
    ) -> None:
        """Fail closed when a recipe asks for an uninstalled ComfyUI hook."""

        if operator != "step_distill":
            return
        capabilities = self.optimization_capabilities if isinstance(self.optimization_capabilities, Mapping) else {}
        for argument, capability_name in (
            ("lpl_target_steps", "lpl"),
            ("tdtm_merge_steps", "tdtm"),
        ):
            if argument not in args:
                continue
            capability = capabilities.get(capability_name)
            if not isinstance(capability, Mapping) or capability.get("safe_to_plan") is not True:
                raise ValueError(
                    "%s requested but remote ComfyUI capability %s is not verified"
                    % (argument, capability_name)
                )
        lpl_target = args.get("lpl_target_steps")
        target_steps = int(args.get("target_steps", 0))
        if lpl_target is not None and int(lpl_target) > target_steps:
            raise ValueError("lpl_target_steps must be <= step_distill target_steps")
        if "tdtm_merge_steps" in args and int(args.get("tdtm_merge_steps", 0)) > target_steps:
            raise ValueError("tdtm_merge_steps must be <= step_distill target_steps")

    def _clear_worker_result_path(self, path: str, experiment_id: str) -> None:
        """Remove only a stale result file before reusing a child id.

        A failed attempt can leave ``trainer_result_<child>.json`` behind.
        The next retry may legitimately reuse that child id, so reading the
        old JSON after a hard worker crash would make the campaign attribute
        an earlier result to the new experiment.  The SSH client validates
        the path and refuses symlinks/directories; this method treats a
        missing path as the normal idempotent case and records the cleanup.
        """

        remover = getattr(self.ssh, "remove_file", None)
        if not isinstance(self.ssh, SSHClient) or not callable(remover):
            # Lightweight test transports and offline doubles do not own a
            # remote result namespace.  There is no stale remote file to
            # clear in that mode, and they must not be asked to emulate SSH
            # deletion just to exercise the campaign state machine.
            return
        response = dict(remover(path))
        status = str(response.get("status") or "")
        if status not in {"missing", "deleted"}:
            raise RemoteError(
                "cannot clear stale worker result for %s: %s" % (experiment_id, response)
            )
        self.events.append(
            "worker_result_path_cleared",
            {"experiment_id": experiment_id, "result_path": path, "status": status},
        )

    def _worker_command(
        self,
        operator: str,
        config_path: str,
        request_path: str,
        result_path: str,
        allocated_gpu_count: Optional[int] = None,
    ) -> Tuple[str, ...]:
        """Resolve a trusted runner for the Controller-selected operator."""
        entrypoint = self.config.worker.operator_entrypoints.get(operator, self.config.worker.entrypoint)
        launcher = self.config.worker.operator_launchers.get(operator, "torchrun")
        if not entrypoint:
            raise ValueError("no trusted worker entrypoint for operator %s" % operator)
        if launcher == "python":
            prefix = (self.config.remote.python, entrypoint)
        elif launcher == "torchrun":
            process_count = 4 if allocated_gpu_count is None else int(allocated_gpu_count)
            if process_count < 2 or process_count > 4:
                raise ValueError("torchrun worker requires an allocated GPU count in [2, 4]")
            prefix = (
                self.config.worker.python,
                "--standalone",
                "--nproc_per_node=%d" % process_count,
                entrypoint,
            )
        else:
            raise ValueError("unsupported trusted worker launcher: %s" % launcher)
        return prefix + (
            "--config",
            config_path,
            "--request",
            request_path,
            "--result",
            result_path,
        )

    def _start_speculative_worker(
        self,
        parent: ModelCandidate,
        plan: ExperimentPlan,
        *,
        slot: str = "primary",
        preserve_controller_lane: bool = False,
        launch_status: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Launch a validated next child while ``parent`` is benchmarked.

        The result is deliberately not imported into the model archive until
        ``_finish_speculative_worker`` sees the current benchmark decision.
        This keeps the expensive worker busy without allowing an unvalidated
        branch to become active or Pareto-eligible.
        """

        if launch_status is not None:
            launch_status.clear()
            launch_status.update({"status": "starting", "retryable": False})

        state_path = self._speculative_state_path(slot)
        existing = self._load_speculative_state(state_path)
        if str(existing.get("status")) in {"launching", "running", "completed"}:
            if str(existing.get("parent_model_id")) == parent.id:
                if launch_status is not None:
                    launch_status.update({"status": "already_running", "retryable": False})
                return existing
            self.events.append(
                "speculative_worker_discarded",
                {
                    "reason": "another_speculative_worker_is_active",
                    "existing_experiment_id": existing.get("experiment_id"),
                },
            )
            if launch_status is not None:
                launch_status.update({"status": "validation_failed", "retryable": False})
            return None
        try:
            operator_args = self._worker_operator_args(plan.operator, plan.operator_args, parent)
            self._validate_h3_runtime_recipe(plan.operator, operator_args, parent)
            execution_plan = (
                self._parallel_gpu_fill_execution_plan(plan)
                if slot == "parallel"
                else plan
            )
            planned_request, effective_request = self._effective_worker_request(execution_plan)
            normalized_request = effective_request
            if (
                self.config.pipeline_enabled
                and bool(normalized_request.get("distributed"))
                and int(normalized_request.get("min_gpu_count", 0)) >= 2
                and not preserve_controller_lane
            ):
                # Evaluation has already reserved ComfyUI's card.  Yield the
                # controller before asking the scheduler for the successor;
                # otherwise vLLM can leave cards 1-3 occupied while the
                # benchmark callback appears to have started overlap.
                release_result = self._request_controller_release(
                    "evaluation_plus_training_overlap",
                    handoff_hold=True,
                )
                if str(release_result.get("status", "")).lower() not in {"released", "not_configured"}:
                    raise RemoteError("Controller release blocked speculative worker: %s" % release_result)
            resource_decision = self._acquire_worker_gpu_lease(
                execution_plan.experiment_id,
                effective_request,
            )
            if not preserve_controller_lane:
                self._clear_controller_handoff_hold()
        except (RemoteError, OSError, TypeError, ValueError) as exc:
            if not preserve_controller_lane:
                self._clear_controller_handoff_hold()
            self.events.append(
                "speculative_worker_discarded",
                {"experiment_id": plan.experiment_id, "reason": "launch_validation_failed", "error": str(exc)[:1200]},
            )
            return None
        self.events.append(
            "resource_scheduled",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "speculative": True,
                "resource_request": dict(execution_plan.resource_request),
                "planned_resource_request": dict(planned_request),
                "effective_resource_request": dict(effective_request),
                "resource_decision": resource_decision.to_dict(),
            },
        )
        if resource_decision.status != "ready":
            if launch_status is not None:
                launch_status.update(
                    {
                        "status": "resource_wait",
                        "retryable": True,
                        "resource_decision": resource_decision.to_dict(),
                    }
                )
            self.events.append(
                "speculative_worker_deferred",
                {
                    "experiment_id": plan.experiment_id,
                    "reason": resource_decision.reason,
                    "resource_decision": resource_decision.to_dict(),
                },
            )
            return None

        if launch_status is not None:
            launch_status.update(
                {
                    "status": "ready",
                    "retryable": False,
                    "resource_decision": resource_decision.to_dict(),
                }
            )

        child_id = self._reserve_next_child_id()
        campaign_root = Path(self.config.remote.resolved_campaign_root)
        results_root = Path(self.config.remote.results_root or self.config.remote.model_root)
        request_path = str(campaign_root / (plan.experiment_id + "-speculative-request.json"))
        config_path = str(campaign_root / (plan.experiment_id + "-speculative-config.json"))
        result_path = str(campaign_root / ("trainer_result_%s.json" % child_id.lower()))
        output_dir = str(results_root / "continuous" / child_id)
        try:
            self._clear_worker_result_path(result_path, plan.experiment_id)
            template = dict(self.ssh.read_json(self.config.worker.config_template))
            dynamic = dict(template)
            dynamic.update(
                {
                    "model_checkpoint": parent.checkpoint_path,
                    "output_dir": output_dir,
                    "source_steps": parent.state.sampling_steps or 32,
                }
            )
            if plan.operator == "step_distill":
                dynamic["target_steps"] = int(operator_args["target_steps"])
            if plan.operator in {"recovery_finetune", "distill", "dmd2"}:
                dynamic["max_steps"] = min(
                    int(plan.operator_args.get("training_steps", dynamic.get("max_steps", 1))),
                    self.config.worker.max_steps,
                )
            if plan.operator == "distill":
                dynamic["dataset_fraction"] = float(operator_args.get("dataset_fraction", 1.0))
            if plan.operator == "dmd2":
                dynamic["generator_update_interval"] = int(operator_args.get("generator_update_interval", 2))
            launcher = self.config.worker.operator_launchers.get(plan.operator, "torchrun")
            if launcher == "torchrun":
                dynamic["world_size"] = resource_decision.actual_gpu_count
            request = {
                "experiment_id": plan.experiment_id,
                "child_model_id": child_id,
                "operator": plan.operator,
                "operator_args": operator_args,
                "parent": {
                    "model_id": parent.id,
                    "checkpoint_path": parent.checkpoint_path,
                    "state": parent.state.to_dict(),
                },
                "artifacts_dir": output_dir,
                "speculative": True,
            }
            self.ssh.write_json(config_path, dynamic)
            self.ssh.write_json(request_path, request)
            command = self._worker_command(
                plan.operator,
                config_path,
                request_path,
                result_path,
                allocated_gpu_count=resource_decision.actual_gpu_count,
            )
            if resource_decision.allocated_gpus:
                visible = ",".join(str(index) for index in resource_decision.allocated_gpus)
                command = ("env", "CUDA_VISIBLE_DEVICES=" + visible) + command
        except (RemoteError, OSError, TypeError, ValueError) as exc:
            self._release_child_id_reservation(child_id)
            self._release_worker_gpu_lease(plan.experiment_id, "speculative_launch_failed")
            self.events.append(
                "speculative_worker_discarded",
                {"experiment_id": plan.experiment_id, "reason": "launch_setup_failed", "error": str(exc)[:1200]},
            )
            return None

        handle: Dict[str, Any] = {
            "status": "launching",
            "speculative": True,
            "experiment_id": plan.experiment_id,
            "operator": plan.operator,
            "operator_args": operator_args,
            "plan": execution_plan.to_dict(),
            "parent_model_id": parent.id,
            "parent_checkpoint_path": parent.checkpoint_path,
            "child_model_id": child_id,
            "request": request,
            "request_path": request_path,
            "config_path": config_path,
            "result_path": result_path,
            "output_dir": output_dir,
            "command": list(command),
            "resource_decision": resource_decision.to_dict(),
            "lane_evidence": self._lane_evidence(
                evaluator_gpus=self._active_evaluation_gpu_indices,
                worker_gpus=resource_decision.allocated_gpus,
                status="ready",
                reason="speculative_worker_started",
            ),
            "started_at": time.time(),
            "worker_timeout_s": max(12.0 * 3600.0, float(dynamic.get("trainer_timeout_s", 0.0) or 0.0)),
            "speculative_slot": str(slot),
            "state_path": str(state_path),
        }
        self._save_speculative_state(handle, state_path)
        self.events.append(
            "speculative_worker_started",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "parent_model_id": parent.id,
                "child_model_id": child_id,
                "allocated_gpus": list(resource_decision.allocated_gpus),
                "resource_decision": resource_decision.to_dict(),
                "lane_evidence": dict(handle["lane_evidence"]),
            },
        )
        self._record_lane_allocation(
            "training",
            worker_gpus=resource_decision.allocated_gpus,
            evaluator_gpus=self._active_evaluation_gpu_indices,
            planned_request=planned_request,
            effective_request=effective_request,
            reason="speculative_worker",
        )
        self._set_pipeline_stage(
            PipelineStage.TRAINING,
            evaluation_model_id=parent.id,
            training_model_id=child_id,
            overlap_with_model_id=parent.id,
        )

        def run_remote() -> None:
            try:
                result = self.ssh.run(tuple(command), check=False, timeout_s=handle["worker_timeout_s"])
                handle.update(
                    {
                        "status": "completed",
                        "returncode": int(getattr(result, "returncode", 0)),
                        "stdout_tail": str(getattr(result, "stdout", ""))[-4000:],
                        "stderr_tail": str(getattr(result, "stderr", ""))[-4000:],
                    }
                )
                self.events.append(
                    "speculative_worker_completed",
                    {
                        "experiment_id": plan.experiment_id,
                        "child_model_id": child_id,
                        "returncode": handle["returncode"],
                        "result_path": result_path,
                    },
                )
            except Exception as exc:
                handle.update({"status": "failed", "error": str(exc)[:2000]})
                self.events.append(
                    "speculative_worker_completed",
                    {
                        "experiment_id": plan.experiment_id,
                        "child_model_id": child_id,
                        "returncode": None,
                        "error": handle["error"],
                    },
                )
            finally:
                self._release_worker_gpu_lease(plan.experiment_id, "speculative_worker_completed")
                self._save_speculative_state(
                    {key: value for key, value in handle.items() if key != "thread"},
                    state_path,
                )

        thread = threading.Thread(
            target=run_remote,
            name="harness4h3-speculative-worker-%s" % plan.experiment_id,
            daemon=True,
        )
        handle["thread"] = thread
        handle["status"] = "running"
        self._save_speculative_state(
            {key: value for key, value in handle.items() if key != "thread"},
            state_path,
        )
        thread.start()
        return handle

    def _finish_speculative_worker(
        self,
        handle: Optional[Mapping[str, Any]],
        *,
        keep: bool,
        reason: str,
        wait_s: float = 0.0,
    ) -> Optional[ExperienceRecord]:
        """Finalize a speculative result after the current benchmark decision.

        The benchmark boundary must not wait for a long checkpoint write.  A
        decision is durable in ``speculative-worker.json`` first; a later
        campaign iteration promotes or cleans up the result after the remote
        worker publishes its JSON.  ``wait_s`` is intentionally opt-in for
        callers that want a short best-effort join.
        """

        if not handle:
            return None
        # If the benchmark decision arrives before training completes, keep
        # the decision durable and return immediately.  The next campaign
        # iteration will finalize the result once its JSON appears; this is
        # what lets a long speculative run remain useful instead of blocking
        # the controller at the evaluation boundary.
        if isinstance(handle, dict):
            handle["finalization_keep"] = bool(keep)
            handle["finalization_reason"] = str(reason)
        thread = handle.get("thread")
        if isinstance(thread, threading.Thread):
            thread.join(timeout=max(0.0, float(wait_s)))
            if thread.is_alive():
                self._save_speculative_state(
                    {key: value for key, value in handle.items() if key != "thread"},
                    Path(str(handle.get("state_path")))
                    if handle.get("state_path")
                    else self.speculative_state_path,
                )
                self.events.append(
                    "speculative_worker_deferred",
                    {
                        "experiment_id": handle.get("experiment_id"),
                        "reason": "worker_still_running",
                        "decision": reason,
                    },
                )
                return None
        state_path = (
            Path(str(handle.get("state_path")))
            if handle.get("state_path")
            else self.speculative_state_path
        )
        state = self._load_speculative_state(state_path)
        status = str(state.get("status") or handle.get("status") or "failed")
        child_id = str(handle.get("child_model_id") or state.get("child_model_id") or "")
        experiment_id = str(handle.get("experiment_id") or state.get("experiment_id") or "")
        result_path = str(handle.get("result_path") or state.get("result_path") or "")
        output_dir = str(handle.get("output_dir") or state.get("output_dir") or "")
        parent_checkpoint = str(handle.get("parent_checkpoint_path") or state.get("parent_checkpoint_path") or "")
        request = handle.get("request") or state.get("request")
        if not isinstance(request, Mapping):
            request = {}
        if status not in {"completed", "failed"}:
            if isinstance(handle, dict):
                self._save_speculative_state(
                    {key: value for key, value in handle.items() if key != "thread"},
                    state_path,
                )
            self.events.append(
                "speculative_worker_discarded",
                {"experiment_id": experiment_id, "reason": "worker_not_complete", "status": status},
            )
            return None
        try:
            result = self.ssh.read_json(result_path)
            digest = self.ssh.sha256(result_path)
        except (RemoteError, OSError, TypeError, ValueError) as exc:
            result = {
                "status": "failed",
                "experiment_id": experiment_id,
                "operator": handle.get("operator"),
                "parent_model_id": handle.get("parent_model_id"),
                "failure_type": "worker_result_missing",
                "message": str(exc)[:2000],
                "metrics": {"real_worker": True, "offline_simulation": False},
            }
            digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
        successful = str(result.get("status", "")).lower() in {"success", "succeeded", "ok", "completed"}
        if not keep or not successful:
            target = str(Path(output_dir) / (child_id + ".safetensors"))
            if successful:
                retention = self.checkpoint_retention.apply(
                    target,
                    child_id,
                    "rejected_candidate",
                    parent_checkpoint_path=parent_checkpoint,
                )
            else:
                retention = self.checkpoint_retention.cleanup_failed(output_dir, child_id)
            self.events.append(
                "speculative_worker_discarded",
                {
                    "experiment_id": experiment_id,
                    "child_model_id": child_id,
                    "reason": reason if keep else "current_candidate_not_eligible",
                    "checkpoint_retention": retention.to_dict(),
                },
            )
            # A discarded speculative result must not be rediscovered by the
            # next campaign import pass.  Keep durable experience only for a
            # promoted branch; its result JSON is otherwise just a stale
            # pointer to a checkpoint that retention has intentionally
            # removed.
            self._clear_worker_result_path(result_path, experiment_id)
            self._clear_speculative_state(state_path)
            self._release_child_id_reservation(child_id)
            return None
        result_uri = "ssh://%s%s" % (self.config.remote.host, result_path)
        imported = RemoteResultImporter(self.experience, self.ssh).import_results(
            [{"source_uri": result_uri, "source_sha256": digest, "result": result, "request": request}]
        )
        for record in imported.records:
            self._append_observation(self._experience_observation(record))
        self.events.append(
            "speculative_worker_promoted",
            {
                "experiment_id": experiment_id,
                "child_model_id": child_id,
                "reason": reason,
                "status": "unevaluated_candidate",
            },
        )
        self._clear_speculative_state(state_path)
        self._release_child_id_reservation(child_id)
        return imported.records[0] if imported.records else None

    def _controller_context(
        self,
        current: ModelCandidate,
        training_calls: int,
        *,
        experiment_cursor: Optional[int] = None,
        planning_intent: str = "primary",
        operator_filter: Optional[Sequence[str]] = None,
    ) -> ControllerContext:
        # A direct Controller call (for example, a resumed process or a test)
        # must see the same persisted goal observation as the normal import
        # path.  This keeps evidence accounting independent of call order.
        self._ensure_goal_observation()
        all_experiences = list(self.experience.read())
        recent = [item.to_dict() for item in all_experiences[-16:]]
        failures = [item for item in recent if item.get("status") in {"failed", "rejected"}]
        # The persisted campaign cursor counts controller calls, while a
        # manually recovered worker can already have produced an experience
        # with a higher experiment id.  Use the durable experiment stream as
        # the id cursor too; otherwise the provider is allowed to propose a
        # duplicate id and overwrite the next worker/result paths.
        experiment_numbers = []
        for item in all_experiences:
            match = re.fullmatch(r"exp_(\d+)", str(item.experiment_id))
            if match is not None:
                experiment_numbers.append(int(match.group(1)))
        cursor_values = [int(training_calls)] + experiment_numbers
        if experiment_cursor is not None:
            cursor_values.append(int(experiment_cursor))
        id_cursor = max(cursor_values)
        budget = BudgetState(
            max_iterations=max(1, self._controller_iteration_budget),
            max_failed_experiments=max(1, self._controller_iteration_budget),
            used_iterations=id_cursor,
            used_controller_calls=training_calls,
        )
        allowed = set(self.config.worker.allowed_operators)
        if operator_filter is not None:
            allowed &= {str(item) for item in operator_filter}
        visible_items = []
        for item in self.registry.visible():
            if item["name"] not in allowed:
                continue
            visible_item = dict(item)
            if item["name"] == "quantize" and self.config.worker.quantized_bits:
                input_schema = dict(item.get("input_schema") or {})
                input_schema["bits"] = {"type": "int", "enum": list(self.config.worker.quantized_bits)}
                visible_item["input_schema"] = input_schema
                visible_item["description"] = "%s; configured variants: %s" % (
                    item.get("description", ""),
                    ", ".join("int%d" % bits for bits in self.config.worker.quantized_bits),
                )
            visible_items.append(visible_item)
        visible = tuple(visible_items)
        campaign_state = self._load_campaign_state()
        consumed = campaign_state.get("consumed_observation_ids", [])
        all_observation_records = tuple(self.observations.read())
        observation_records = self._context_observation_window(
            all_observation_records,
            consumed,
            current.id,
        )
        consumed_ids = {str(item) for item in consumed}
        unconsumed = tuple(
            item
            for item in observation_records
            if str(item.observation_id) not in consumed_ids
        )
        current_system = self._system_for_model(current.id)
        current_evaluation = self._load_evaluations().get(current.id, {})
        retrieved = self._retrieve_relevant_experiences(all_experiences, current)
        operator_stats = self._experience_operator_stats(all_experiences)
        gpu_capacity = self._planning_gpu_capacity()
        active_round_policy = self._active_round_policy()
        campaign_state = self._load_campaign_state()
        round_policy_progress = self._round_policy_progress(active_round_policy, campaign_state)
        pipeline_telemetry = self._pipeline_telemetry()
        discovery_digest = self._build_discovery_digest(telemetry=pipeline_telemetry)
        return ControllerContext(
            self.target,
            current.state,
            budget,
            visible,
            recent,
            failures,
            [entry.to_dict() for entry in self.pareto.front()],
            validated_evaluation=current_evaluation if isinstance(current_evaluation, Mapping) else {},
            goal=self._goal_payload(),
            observations=[item.to_dict() for item in observation_records],
            unconsumed_observation_ids=[item.observation_id for item in unconsumed],
            current_system=current_system.to_dict() if current_system is not None else {
                "id": "S0000",
                "model_ref": current.id,
                "device_id": self.target.hardware_name,
                "runtime_state": {"sampling_steps": current.state.sampling_steps or 32},
            },
            campaign_summary={
                "current_model_id": current.id,
                "current_system_id": current_system.id if current_system is not None else "S0000",
                "evaluated_model_ids": sorted(self._load_evaluations()),
                "training_calls": training_calls,
                "operator_stats": operator_stats,
                "gpu_capacity": gpu_capacity,
                "pipeline_telemetry": pipeline_telemetry,
                "planning_intent": str(planning_intent or "primary"),
                "round_policy_progress": round_policy_progress or {},
                "round_policy_budget_status": (
                    self._round_policy_budget_status(active_round_policy, campaign_state)
                    if active_round_policy is not None
                    else None
                ),
            },
            retrieved_relevant_experiments=retrieved,
            planning_intent=str(planning_intent or "primary"),
            optimization_capabilities=self.optimization_capabilities,
            round_policy=active_round_policy.to_dict() if active_round_policy is not None else {},
            discovery_digest=discovery_digest.to_context(),
        )

    def _planning_gpu_capacity(self) -> Dict[str, Any]:
        """Return a small live GPU summary for the next remote LLM plan.

        This is advisory context only; ``RemoteResourceScheduler.acquire``
        takes a fresh snapshot immediately before launching a worker. Keeping
        just indices, counts, the memory waterline, and one per-card
        power/utilization sample avoids putting raw process output or long
        telemetry traces into the model prompt.
        """

        try:
            memory, processes = self.scheduler.snapshot()
            reserved = sorted(int(index) for index in self.scheduler.reserved_gpu_indices)
            controller_reserved = []
            controller_reader = getattr(self.scheduler, "_controller_reserved_gpu_indices", None)
            if callable(controller_reader):
                controller_reserved = sorted(int(index) for index in controller_reader())
            process_indices = getattr(self.scheduler, "last_compute_gpu_indices", None)
            process_present = bool(str(processes).strip())
            if process_present and process_indices is None:
                # The scheduler uses the same fail-closed rule at launch. Keep
                # the LLM's advisory view consistent with that rule instead of
                # presenting unmapped cards as available capacity.
                compute_process_indices = set(range(int(self.scheduler.gpu_count)))
            else:
                compute_process_indices = set(int(index) for index in (process_indices or ()))
            reserved_for_capacity = (
                set(reserved)
                | set(controller_reserved)
                | compute_process_indices
            )
            free = [
                int(index)
                for index in range(int(self.scheduler.gpu_count))
                if index not in reserved_for_capacity and self.scheduler.meets_memory_waterline(index, memory)
            ]
            per_gpu = {
                str(index): {
                    "memory_used_mb": int(values[0]),
                    "memory_total_mb": int(values[1]) if values[1] is not None else None,
                    "free_above_waterline": bool(self.scheduler.meets_memory_waterline(index, memory)),
                }
                for index, values in sorted(memory.items())
                if 0 <= int(index) < int(self.scheduler.gpu_count)
            }
            live_telemetry = getattr(self.scheduler, "last_gpu_telemetry", {})
            if isinstance(live_telemetry, Mapping):
                for index, values in live_telemetry.items():
                    row = per_gpu.setdefault(str(index), {})
                    if isinstance(values, Mapping):
                        for key in ("power_w", "utilization_gpu_pct"):
                            if key in values:
                                row[key] = values[key]
            return {
                "gpu_count": int(self.scheduler.gpu_count),
                "memory_waterline_mb": int(self.scheduler.min_free_memory_mb),
                "reserved_gpu_indices": reserved,
                "controller_reserved_gpu_indices": controller_reserved,
                "free_above_waterline_indices": free,
                "free_above_waterline_count": len(free),
                "compute_processes_present": process_present,
                "compute_process_gpu_indices": sorted(compute_process_indices),
                "compute_process_mapping_unknown": bool(process_present and process_indices is None),
                "comfyui_reserved_gpu_indices": sorted(
                    int(lease.gpu_index)
                    for lease in self.comfyui_leases.values()
                    if lease.lease_active
                ),
                "per_gpu": per_gpu,
            }
        except Exception as exc:
            return {"error": str(exc)[:500]}

    def _pipeline_telemetry(self) -> Dict[str, Any]:
        """Return bounded recent evidence about overlap efficiency.

        The Controller needs more than a static free-card count: it should
        know whether a previous plan was ready before evaluation, whether a
        parallel candidate was reused, and which cards were underutilized.
        Read only the event tail so this diagnostic remains cheap after a
        long autonomous run.
        """

        try:
            events = self.events.tail(512)
        except (OSError, TypeError, ValueError) as exc:
            return {"error": str(exc)[:500]}
        tracked = {
            "controller_plan_prefetch_started",
            "controller_plan_prefetch_ready",
            "controller_plan_prefetch_unavailable",
            "controller_plan_parallel_prefetch_skipped",
            "controller_plan_parallel_prefetch_discarded",
            "controller_plan_parallel_prefetch_armed",
            "controller_plan_parallel_system_rebased",
            "controller_plan_overlap_waiting",
            "controller_plan_overlap_ready",
            "controller_plan_parallel_waiting",
            "controller_plan_parallel_reused",
            "controller_plan_parallel_ready",
            "controller_plan_evaluation_refill_started",
            "controller_plan_evaluation_refill_ready",
            "controller_plan_evaluation_refill_discarded",
            "evaluation_refill_skipped",
            "evaluation_started",
            "evaluation_waiting_for_resources",
            "evaluation_worker_skipped",
            "evaluation_overlap_budget",
            "parallel_gpu_fill_skipped",
            "speculative_worker_started",
            "worker_gpu_lease_release_skipped",
            "lane_allocation",
            "training_power_sampled",
            "evaluation_completed",
            "pipeline_stage",
        }
        counts: Dict[str, int] = {}
        plan_latencies: List[float] = []
        overlap_waits: List[float] = []
        underutilized: set[str] = set()
        worker_gpu_sets: List[List[int]] = []
        evaluation_gpu_fill_waits: List[float] = []
        evaluation_gpu_fill_ready_count = 0
        evaluation_gpu_fill_missing_count = 0
        evaluation_refill_started_count = 0
        evaluation_refill_ready_count = 0
        active_evaluation_started_at: Optional[float] = None
        active_evaluation_gpu_fill_recorded = False
        last_stage: Optional[str] = None
        last_power_feedback: Optional[Mapping[str, Any]] = None
        last_overlap_budget: Optional[Mapping[str, Any]] = None
        target_power_w = float(self.config.power_target_w)

        def add_float(target: List[float], value: Any) -> None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if number >= 0 and number == number and number != float("inf"):
                target.append(round(number, 3))

        def collect_power(value: Any) -> None:
            nonlocal last_power_feedback
            if not isinstance(value, Mapping):
                return
            per_gpu = value.get("per_gpu")
            if not isinstance(per_gpu, Mapping):
                return
            compact_rows: Dict[str, Mapping[str, Any]] = {}
            under_target: List[str] = []
            for gpu_id, row in per_gpu.items():
                if not isinstance(row, Mapping):
                    continue
                try:
                    utilization = float(
                        row.get("utilization_gpu_pct_avg", row.get("utilization_mean"))
                    )
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(utilization):
                    continue
                if utilization < 70.0:
                    underutilized.add(str(gpu_id))
                compact_row: Dict[str, Any] = {
                    "utilization_gpu_pct_avg": round(utilization, 2),
                }
                try:
                    power = float(row.get("power_w_avg", row.get("power_mean_w")))
                except (TypeError, ValueError):
                    power = None
                if power is not None and math.isfinite(power):
                    compact_row["power_w_avg"] = round(power, 2)
                    if power < target_power_w * 0.8:
                        under_target.append(str(gpu_id))
                lane = row.get("lane")
                if isinstance(lane, str) and lane:
                    compact_row["lane"] = lane[:64]
                compact_rows[str(gpu_id)] = compact_row
            if compact_rows:
                last_power_feedback = {
                    "power_w_avg": value.get("power_w_avg"),
                    "power_w_peak": value.get("power_w_peak"),
                    "target_power_w": target_power_w,
                    "under_target_gpu_indices": sorted(set(under_target))[:4],
                    "underutilized_gpu_indices": sorted(
                        gpu_id
                        for gpu_id, row in compact_rows.items()
                        if float(row.get("utilization_gpu_pct_avg", 100.0)) < 70.0
                    )[:4],
                    "per_gpu": {key: compact_rows[key] for key in sorted(compact_rows)[:4]},
                }

        def event_timestamp(event: Mapping[str, Any]) -> Optional[float]:
            value = event.get("created_at")
            if not value:
                return None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timestamp = float(value)
                return timestamp if math.isfinite(timestamp) else None
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                timestamp = parsed.timestamp()
            except (TypeError, ValueError, OverflowError):
                return None
            return timestamp if math.isfinite(timestamp) else None

        for event in events:
            event_type = str(event.get("event_type") or "")
            if event_type not in tracked:
                continue
            counts[event_type] = counts.get(event_type, 0) + 1
            if event_type == "pipeline_stage":
                stage = event.get("stage")
                if stage:
                    last_stage = str(stage)
            if event_type == "evaluation_started":
                started_at = event_timestamp(event)
                if started_at is not None:
                    active_evaluation_started_at = started_at
                    active_evaluation_gpu_fill_recorded = False
            if event_type == "evaluation_overlap_budget":
                candidate_indices = event.get("training_candidate_gpu_indices")
                if isinstance(candidate_indices, (list, tuple)):
                    try:
                        normalized_candidates = sorted({int(index) for index in candidate_indices})[:4]
                    except (TypeError, ValueError):
                        normalized_candidates = []
                    last_overlap_budget = {
                        "evaluation_workers": int(event.get("evaluation_workers", 0) or 0),
                        "configured_workers": int(event.get("configured_workers", 0) or 0),
                        "training_candidate_gpu_indices": normalized_candidates,
                        "training_candidate_gpu_count": len(normalized_candidates),
                        "idle_reason": str(event.get("idle_reason") or "")[:120],
                    }
            if event_type == "controller_plan_prefetch_ready":
                add_float(plan_latencies, event.get("elapsed_s"))
            if event_type == "controller_plan_evaluation_refill_started":
                evaluation_refill_started_count += 1
            if event_type == "controller_plan_evaluation_refill_ready":
                evaluation_refill_ready_count += 1
            if event_type in {
                "controller_plan_overlap_ready",
                "controller_plan_parallel_ready",
                "controller_plan_parallel_waiting",
            }:
                add_float(overlap_waits, event.get("wait_s"))
            if event_type == "speculative_worker_started":
                values = event.get("allocated_gpus")
                if isinstance(values, (list, tuple)):
                    try:
                        normalized_values = [int(item) for item in values][-4:]
                        worker_gpu_sets.append(normalized_values)
                        if (
                            normalized_values
                            and active_evaluation_started_at is not None
                            and not active_evaluation_gpu_fill_recorded
                        ):
                            started_at = event_timestamp(event)
                            if started_at is not None:
                                add_float(
                                    evaluation_gpu_fill_waits,
                                    max(0.0, started_at - active_evaluation_started_at),
                                )
                                evaluation_gpu_fill_ready_count += 1
                                active_evaluation_gpu_fill_recorded = True
                    except (TypeError, ValueError):
                        pass
            if event_type == "training_power_sampled":
                collect_power((event.get("summary") or {}))
            if event_type == "evaluation_completed":
                if active_evaluation_started_at is not None and not active_evaluation_gpu_fill_recorded:
                    evaluation_gpu_fill_missing_count += 1
                active_evaluation_started_at = None
                active_evaluation_gpu_fill_recorded = False
                summary = event.get("summary")
                if isinstance(summary, Mapping):
                    collect_power(summary.get("power_sampling"))
                    hardware = summary.get("hardware")
                    if isinstance(hardware, Mapping):
                        collect_power(hardware.get("power_sampling"))

        def stats(values: Sequence[float]) -> Mapping[str, Any]:
            if not values:
                return {"samples": 0}
            return {
                "samples": len(values),
                "last_s": values[-1],
                "avg_s": round(sum(values) / len(values), 3),
                "max_s": max(values),
            }

        return {
            "window_events": len(events),
            "event_counts": counts,
            "last_pipeline_stage": last_stage,
            "prefetch_latency_s": stats(plan_latencies[-8:]),
            "overlap_wait_s": stats(overlap_waits[-8:]),
            "evaluation_gpu_fill_wait_s": stats(evaluation_gpu_fill_waits[-8:]),
            "evaluation_gpu_fill_ready_count": evaluation_gpu_fill_ready_count,
            "evaluation_gpu_fill_missing_count": evaluation_gpu_fill_missing_count,
            "evaluation_refill_started_count": evaluation_refill_started_count,
            "evaluation_refill_ready_count": evaluation_refill_ready_count,
            "underutilized_gpu_indices": sorted(underutilized)[:4],
            "recent_speculative_worker_gpu_sets": worker_gpu_sets[-8:],
            "last_evaluation_overlap_budget": last_overlap_budget or {"samples": 0},
            "power_target_w": target_power_w,
            "last_power_feedback": last_power_feedback
            or {"samples": 0, "target_power_w": target_power_w},
        }

    def _context_observation_window(
        self,
        records: Sequence[Any],
        consumed_ids: Iterable[str],
        current_model_id: str,
    ) -> Tuple[Any, ...]:
        """Select bounded evidence while retaining the append-only history.

        Observation JSONL is the durable memory of the campaign, but copying
        every historical record into each Controller context makes prompt size
        grow without bound. Unconsumed evidence and human directives are
        prioritized first; current-lineage evidence and recency fill the
        remaining slots. Records outside the window remain on disk and can be
        selected by a later call.
        """

        limit = max(1, int(self.config.max_context_observations))
        consumed = {str(item) for item in consumed_ids}
        total = max(1, len(records))
        ranked: List[Tuple[float, int, Any]] = []
        for index, record in enumerate(records):
            observation_id = str(getattr(record, "observation_id", ""))
            kind = str(getattr(record, "kind", ""))
            score = 0.0
            if observation_id and observation_id not in consumed:
                score += 2000.0
            if kind == "human_directive" and observation_id not in consumed:
                score += 1000.0
            if str(getattr(record, "model_id", "") or "") == str(current_model_id):
                score += 240.0
            if str(getattr(record, "parent_model_id", "") or "") == str(current_model_id):
                score += 180.0
            if kind in {"experience", "evaluation"}:
                score += 20.0
            score += 10.0 * (float(index) / float(total))
            ranked.append((score, index, record))
        ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
        # The goal observation is a durable control-plane fact. Always keep
        # the current unconsumed goal visible even when a long run has more
        # than ``limit`` unconsumed experiment observations. Human directives
        # receive the same treatment, with the newest directives winning if
        # the directive queue itself exceeds the cap.
        goal_indices = [
            index
            for index, record in enumerate(records)
            if str(getattr(record, "kind", "")) == "goal"
            and str(getattr(record, "observation_id", "")) not in consumed
        ]
        directive_indices = [
            index
            for index, record in enumerate(records)
            if str(getattr(record, "kind", "")) == "human_directive"
            and str(getattr(record, "observation_id", "")) not in consumed
        ]
        selected_indices = set(goal_indices[:limit])
        for index in reversed(directive_indices):
            if len(selected_indices) >= limit:
                break
            selected_indices.add(index)
        if len(selected_indices) < limit:
            for _, index, _ in ranked:
                if index in selected_indices:
                    continue
                selected_indices.add(index)
                if len(selected_indices) >= limit:
                    break
        return tuple(record for index, record in enumerate(records) if index in selected_indices)

    def _retrieve_relevant_experiences(
        self,
        records: Sequence[ExperienceRecord],
        current: ModelCandidate,
    ) -> List[Mapping[str, Any]]:
        """Return a small relevance-ranked slice of the long-term memory.

        ``recent_experiments`` is intentionally short for prompt size, but a
        long RSI campaign must still remember that an operator or failure was
        already tried. Rank by current-lineage relevance first, then by
        constraint/failure signal and recency. The full JSONL stream remains
        the source of truth; only the top bounded records enter the prompt.
        """

        metrics = current.state.measured_metrics
        blocked_terms = {
            name
            for name, limit in (
                ("latency", self.target.max_latency_s),
                ("memory", self.target.max_peak_memory_gb),
                ("size", self.target.max_model_size_gb),
                ("quality", self.target.min_quality_score),
            )
            if limit is not None
            and (
                (name == "quality" and _numeric(metrics.get("quality_score"), 0.0) < float(limit))
                or (name != "quality" and _numeric(metrics.get({"latency": "latency_s", "memory": "peak_memory_gb", "size": "model_size_gb"}[name]), float("inf")) > float(limit))
            )
        }
        ranked: List[Tuple[float, int, ExperienceRecord]] = []
        for index, record in enumerate(records):
            score = 0.0
            if record.parent_model_id == current.id:
                score += 100.0
            if record.child_model_id == current.id:
                score += 120.0
            if record.status in {"failed", "rejected"}:
                score += 20.0
            if record.status == "accepted":
                score += 8.0
            text = " ".join(
                str(value)
                for value in (
                    record.operator,
                    record.decision,
                    record.evaluation,
                    record.training,
                )
            ).lower()
            score += 7.0 * sum(1 for term in blocked_terms if term in text)
            ranked.append((score, index, record))
        ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)
        return [record.to_dict() for _, _, record in ranked[:8]]

    @staticmethod
    def _experience_operator_stats(records: Sequence[ExperienceRecord]) -> Mapping[str, Mapping[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for record in records:
            operator = str(record.operator or "unknown")
            item = stats.setdefault(
                operator,
                {"attempts": 0, "failed": 0, "accepted": 0, "rejected": 0, "unvalidated": 0, "last_status": None},
            )
            item["attempts"] += 1
            if record.status == "failed":
                item["failed"] += 1
            elif record.status == "accepted":
                item["accepted"] += 1
            elif record.status == "rejected":
                item["rejected"] += 1
            else:
                item["unvalidated"] += 1
            item["last_status"] = record.status
        return stats

    def _controller_plan(
        self,
        parent: ModelCandidate,
        training_calls: int,
        *,
        controller: Optional[Any] = None,
        prefetch: bool = False,
        experiment_cursor: Optional[int] = None,
        planning_intent: str = "primary",
        operator_filter: Optional[Sequence[str]] = None,
    ) -> Optional[Any]:
        """Ask the Controller for a validated plan and retain an audit trace.

        ``prefetch`` is used only by the worker overlap path. It runs against a
        cloned provider and records its trace separately, because the plan is
        speculative until the current worker reaches a safe boundary.
        """

        if not prefetch and controller is None and str(planning_intent or "primary") == "primary":
            self._last_primary_controller_batch = None

        active_policy_before_call = self._active_round_policy()
        if active_policy_before_call is not None:
            policy_state = self._load_campaign_state()
            policy_stop = self._round_policy_budget_status(
                active_policy_before_call,
                policy_state,
            )
            if policy_stop is not None:
                provider_metadata = self._controller_provider_metadata(controller or self.controller)
                trace = {
                    "controller_call": False,
                    "prefetch": bool(prefetch),
                    "planning_intent": str(planning_intent or "primary"),
                    **provider_metadata,
                    "parent_model_id": parent.id,
                    "training_calls": training_calls,
                    "status": "round_policy_stopped",
                    "execution_status": "round_policy_stopped",
                    "round_id": active_policy_before_call.round_id,
                    "reason": str(policy_stop),
                    "round_policy_progress": self._round_policy_progress(
                        active_policy_before_call,
                        policy_state,
                    ),
                }
                (self.controller_prefetch_trace if prefetch else self.controller_trace).append(trace)
                self.events.append(
                    "round_policy_plan_blocked",
                    {
                        "planning_intent": str(planning_intent or "primary"),
                        "prefetch": bool(prefetch),
                        "parent_model_id": parent.id,
                        "round_id": active_policy_before_call.round_id,
                        "reason": str(policy_stop),
                        "progress": trace["round_policy_progress"],
                    },
                )
                return None

        context = self._controller_context(
            parent,
            training_calls,
            experiment_cursor=experiment_cursor,
            planning_intent=planning_intent,
            operator_filter=operator_filter,
        )
        planner = controller or self.controller
        provider_metadata = self._controller_provider_metadata(planner)
        candidate_policy: Optional[RoundPolicy] = None
        trace: Dict[str, Any] = {
            "controller_call": True,
            "prefetch": bool(prefetch),
            "planning_intent": str(planning_intent or "primary"),
            **provider_metadata,
            "parent_model_id": parent.id,
            "training_calls": training_calls,
            "goal": self._goal_payload(),
            "input_metrics": dict(parent.state.measured_metrics),
            "context_experience_ids": [
                str(item.get("experience_id"))
                for item in context.recent_experiments
                if item.get("experience_id")
            ],
            # Keep the audit trace aligned with the bounded context actually
            # offered to the Controller.  The append-only observation store
            # remains the source of truth; replaying every historical ID here
            # would make long autonomous runs grow each request linearly.
            "input_observation_ids": [
                str(item.get("observation_id"))
                for item in context.observations
                if item.get("observation_id")
            ],
            "unconsumed_observation_ids": list(context.unconsumed_observation_ids),
        }
        self.events.append(
            "controller_input",
            {
                **provider_metadata,
                "prefetch": bool(prefetch),
                "planning_intent": str(planning_intent or "primary"),
                "goal": context.goal,
                "model_id": parent.id,
                "input_metrics": dict(parent.state.measured_metrics),
                "unconsumed_observation_ids": list(context.unconsumed_observation_ids),
                "optimization_capabilities": copy.deepcopy(context.optimization_capabilities),
                "observation_summaries": [
                    _compact_controller_input_observation(item)
                    for item in context.observations
                ],
            },
        )
        try:
            raw_plan = self._invoke_controller(context, planner)
            candidate_generation = {
                "requested_candidate_count": int(getattr(planner, "candidate_count", 1) or 1),
                "candidate_count": int(
                    getattr(planner, "last_candidate_count", 1)
                    if getattr(planner, "last_candidate_count", None) is not None
                    else 1
                ),
                "eligible_candidate_count": int(
                    getattr(planner, "last_candidate_eligible_count", 1)
                    if getattr(planner, "last_candidate_eligible_count", None) is not None
                    else 1
                ),
                "filter_rejections": list(getattr(planner, "last_candidate_filter_rejections", []) or []),
                "selected_index": getattr(planner, "last_selected_index", None),
                "candidate_request_id": getattr(planner, "last_candidate_request_id", None),
                "selection_request_id": getattr(planner, "last_selection_request_id", None),
                "selection_fallback": getattr(planner, "last_selection_fallback", None),
            }
            trace["candidate_generation"] = candidate_generation
            trace["raw_plan"] = raw_plan.to_dict() if hasattr(raw_plan, "to_dict") else raw_plan
            self.events.append(
                "controller_plan_proposed",
                {
                    **provider_metadata,
                    "prefetch": bool(prefetch),
                    "request_id": getattr(planner, "last_request_id", None),
                    "candidate_generation": candidate_generation,
                    "input_observation_ids": list(context.unconsumed_observation_ids),
                    "plan": trace["raw_plan"],
                    "model_id": parent.id,
                },
            )
            plan = self.validation.schema.validate(raw_plan)
            if plan.round_policy is not None:
                if prefetch or str(planning_intent or "primary") != "primary":
                    raise ValueError("round_policy may only be activated by a primary Controller plan")
                candidate_policy = RoundPolicy.from_dict(plan.round_policy)
                self._validate_round_policy_runtime(candidate_policy)
                plan = replace(plan, round_policy=candidate_policy.to_dict())
            active_policy = candidate_policy or self._active_round_policy()
            if active_policy is not None:
                if plan.operator not in set(active_policy.allowed_operators):
                    raise ValueError(
                        "round policy does not allow operator %s" % plan.operator
                    )
                resource = plan.resource_request if isinstance(plan.resource_request, Mapping) else {}
                if bool(resource.get("distributed")):
                    try:
                        minimum_training_gpus = int(resource.get("min_gpu_count", 0))
                    except (TypeError, ValueError):
                        raise ValueError("round policy requires a valid distributed GPU request")
                    policy_minimum = int(active_policy.resource_policy["min_training_gpus"])
                    if minimum_training_gpus < policy_minimum:
                        raise ValueError(
                            "plan min_gpu_count=%s is below round policy minimum=%s"
                            % (minimum_training_gpus, policy_minimum)
                        )
                try:
                    declared_gpu_hours = float(plan.required_budget.get("gpu_hours", 0.0))
                except (TypeError, ValueError):
                    raise ValueError("plan required_budget.gpu_hours must be numeric")
                policy_gpu_hours = float(active_policy.axis_budget["max_gpu_hours"])
                if declared_gpu_hours > policy_gpu_hours:
                    raise ValueError(
                        "plan gpu_hours=%s exceeds round policy budget=%s"
                        % (declared_gpu_hours, policy_gpu_hours)
                    )
            self.validation.policy.validate(plan, parent.state, context.budget_state)
            trace["declared_budget"] = asdict(self.validation.budget.validate_declared(plan, context.budget_state))
            self._validate_plan_evidence(plan, context)
            current_system = self._system_for_model(parent.id)
            if current_system is not None and plan.parent_system_id != current_system.id:
                # Prefetch candidates are sampled while the current worker is
                # still publishing its child.  Local vLLM responses can copy
                # the root system id even when they name the correct current
                # model.  Rebase only same-parent prefetches; non-prefetch
                # plans remain strict so bad lineage can never become the
                # active successor silently.
                if prefetch and plan.parent_model_id == parent.id:
                    parallel_rebase = str(planning_intent or "") == "parallel_gpu_fill"
                    trace["parent_system_rebased"] = {
                        "from": plan.parent_system_id,
                        "to": current_system.id,
                        "reason": (
                            "same_parent_parallel_prefetch_stale_system_id"
                            if parallel_rebase
                            else "same_parent_prefetch_stale_system_id"
                        ),
                    }
                    plan = replace(plan, parent_system_id=current_system.id)
                    self.events.append(
                        (
                            "controller_plan_parallel_system_rebased"
                            if parallel_rebase
                            else "controller_plan_prefetch_system_rebased"
                        ),
                        {
                            "experiment_id": plan.experiment_id,
                            "parent_model_id": parent.id,
                            "from_parent_system_id": trace["parent_system_rebased"]["from"],
                            "parent_system_id": current_system.id,
                            "reason": trace["parent_system_rebased"]["reason"],
                        },
                    )
                else:
                    raise ValueError(
                        "plan parent_system_id %s does not match current system %s"
                        % (plan.parent_system_id, current_system.id)
                    )
            fingerprint = experiment_fingerprint(
                parent.id,
                current_system.id if current_system is not None else str(plan.parent_system_id or "S0000"),
                plan.operator,
                plan.operator_args,
                self.target.id,
                current_system.device_id if current_system is not None else self.target.hardware_name,
                {
                    "split": "heldout",
                    "workflow_template": str(self.config.workflow.template),
                    "quality_scope": self.config.quality_scope,
                },
            )
            trace["fingerprint"] = fingerprint
            if not plan.repeat_for_statistics and any(
                record.fingerprint == fingerprint for record in self.experiments.read() if record.fingerprint
            ):
                raise ValueError(
                    "duplicate_experiment_fingerprint; set repeat_for_statistics=true for a statistical repeat"
                )
            resource_request = self._validate_resource_request(plan.resource_request)
            plan = replace(plan, resource_request=resource_request)
            self._validate_worker_resource_match(plan)
        except Exception as exc:
            unavailable = isinstance(exc, ControllerUnavailableError) or (
                provider_metadata["provider"] == "vllm" and isinstance(exc, ControllerProviderError)
            )
            trace["status"] = "controller_unavailable" if unavailable else "rejected"
            trace["error"] = str(exc)
            (self.controller_prefetch_trace if prefetch else self.controller_trace).append(trace)
            self.events.append(
                "controller_unavailable" if unavailable else "controller_plan_rejected",
                {
                    **provider_metadata,
                    "prefetch": bool(prefetch),
                    "planning_intent": str(planning_intent or "primary"),
                    "model_id": parent.id,
                    "error": str(exc),
                    "trace": trace,
                },
            )
            return None
        if plan.operator not in set(self.config.worker.allowed_operators) or (
            operator_filter is not None and plan.operator not in {str(item) for item in operator_filter}
        ):
            trace["status"] = "rejected"
            trace["error"] = "operator_not_allowed_for_planning_intent"
            (self.controller_prefetch_trace if prefetch else self.controller_trace).append(trace)
            self.events.append(
                "controller_plan_rejected",
                {
                    "prefetch": bool(prefetch),
                    "planning_intent": str(planning_intent or "primary"),
                    "model_id": parent.id,
                    "error": "operator_not_allowed_for_planning_intent",
                    "trace": trace,
                },
            )
            return None
        try:
            self.registry.validate(plan.operator, parent.state, plan.operator_args, self.target)
            # Validate the trusted worker contract before marking the plan as
            # executable.  In particular, the real step-distill worker is a
            # single binary-halving stage; allowing the LLM to return 32->8
            # here used to produce a validated plan that silently returned
            # before resource scheduling.
            self._worker_operator_args(plan.operator, plan.operator_args, parent)
            self._validate_h3_runtime_recipe(plan.operator, plan.operator_args, parent)
            if plan.operator == "quantize" and self.config.worker.quantized_bits:
                bits = plan.operator_args.get("bits")
                if bits not in set(self.config.worker.quantized_bits):
                    raise ValueError(
                        "quantize.bits=%s is not configured; available variants: %s"
                        % (bits, ", ".join(str(value) for value in self.config.worker.quantized_bits))
                    )
        except Exception as exc:
            trace["status"] = "rejected"
            trace["error"] = str(exc)
            (self.controller_prefetch_trace if prefetch else self.controller_trace).append(trace)
            self.events.append(
                "controller_plan_rejected",
                {
                    "prefetch": bool(prefetch),
                    "planning_intent": str(planning_intent or "primary"),
                    "model_id": parent.id,
                    "error": str(exc),
                    "trace": trace,
                },
            )
            return None
        trace["status"] = "validated"
        trace["plan"] = plan.to_dict()
        (self.controller_prefetch_trace if prefetch else self.controller_trace).append(trace)
        state = self._load_campaign_state()
        consumed = set(str(item) for item in state.get("consumed_observation_ids", []))
        consumed.update(plan.consumed_observation_ids)
        state["consumed_observation_ids"] = sorted(consumed)
        state["goal"] = self._goal_payload()
        if candidate_policy is not None:
            self._persist_active_round_policy(candidate_policy, state)
        self._save_campaign_state(state)
        if candidate_policy is not None:
            self.events.append(
                "round_policy_activated",
                {
                    "round_id": candidate_policy.round_id,
                    "experiment_id": plan.experiment_id,
                    "source_observation_ids": list(candidate_policy.source_observation_ids),
                    "substrate_digest": candidate_policy.substrate_digest,
                    "evaluation_digest": candidate_policy.fixed_evaluation.get("recipe_digest"),
                    "allowed_operators": list(candidate_policy.allowed_operators),
                },
            )
        self.events.append(
            "controller_plan_validated",
            {
                **provider_metadata,
                "prefetch": bool(prefetch),
                "planning_intent": str(planning_intent or "primary"),
                "request_id": getattr(planner, "last_request_id", None),
                "model_id": parent.id,
                "plan": plan.to_dict(),
                "consumed_observation_ids": list(plan.consumed_observation_ids),
                "diagnosis_evidence": list(plan.diagnosis_evidence),
                "resource_request": dict(plan.resource_request),
            },
        )
        if not prefetch and controller is None and str(planning_intent or "primary") == "primary":
            candidates = getattr(planner, "last_eligible_candidates", ())
            if isinstance(candidates, (list, tuple)):
                self._last_primary_controller_batch = {
                    "experiment_id": str(plan.experiment_id),
                    "parent_model_id": str(plan.parent_model_id),
                    "training_calls": int(training_calls),
                    "candidates": tuple(
                        item for item in candidates if isinstance(item, ExperimentPlan)
                    ),
                }
        return plan

    def _consume_primary_batch_parallel_candidate(
        self,
        primary_plan: ExperimentPlan,
        training_calls: int,
    ) -> Optional[ExperimentPlan]:
        """Take one GPU candidate from the just-completed primary n-way call.

        This is a one-shot handoff for the exact primary experiment/cursor.
        The returned plan still passes the normal worker and scheduler gates
        in ``_start_speculative_worker``; stale batches are discarded rather
        than replayed after a restart, review, or lineage change.
        """

        batch = self._last_primary_controller_batch
        self._last_primary_controller_batch = None
        if not isinstance(batch, Mapping):
            return None
        if (
            str(batch.get("experiment_id")) != str(primary_plan.experiment_id)
            or str(batch.get("parent_model_id")) != str(primary_plan.parent_model_id)
            or int(batch.get("training_calls", -1)) != int(training_calls)
            or str(primary_plan.operator) not in {"prune_blocks", "quantize"}
        ):
            return None
        candidates = batch.get("candidates")
        if not isinstance(candidates, (list, tuple)):
            return None
        handle = {
            "prefetch_state_key": "primary",
            "training_calls": int(training_calls) + 1,
            "result": {
                "plan": primary_plan,
                "parallel_candidates": list(candidates),
            },
        }
        parallel = self._parallel_candidate_from_prefetch_handle(
            handle,
            primary_plan,
            int(training_calls) + 1,
        )
        if parallel is not None:
            self.events.append(
                "controller_plan_parallel_reused",
                {
                    "experiment_id": parallel.experiment_id,
                    "parent_model_id": parallel.parent_model_id,
                    "training_calls": int(training_calls) + 1,
                    "primary_experiment_id": primary_plan.experiment_id,
                    "operator": parallel.operator,
                    "reason": "primary_batched_candidate_before_cpu_worker",
                },
            )
        return parallel

    def _validate_resource_request(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("Controller plan must declare resource_request")
        required = {"gpu_count", "distributed", "exclusive", "evaluation_workers", "on_unavailable"}
        if not required.issubset(value):
            raise ValueError("resource_request is missing required fields")
        normalized = dict(self.scheduler.normalize_request(value))
        required = {"gpu_count", "min_gpu_count", "max_gpu_count", "elastic", "distributed", "exclusive", "evaluation_workers", "on_unavailable"}
        if not required.issubset(normalized):
            raise ValueError("resource_request is missing required fields")
        workers = normalized["evaluation_workers"]
        if isinstance(workers, bool) or not isinstance(workers, int) or not 0 <= workers <= 4:
            raise ValueError("resource_request.evaluation_workers must be an integer in [0, 4]")
        if normalized["on_unavailable"] not in {"wait", "replan"}:
            raise ValueError("resource_request.on_unavailable must be wait or replan")
        return normalized

    def _effective_worker_request(
        self,
        plan: ExperimentPlan,
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Shape a validated plan around the live Controller lane.

        Normal elastic plans use the configured Controller overlap lane.  A
        plan that explicitly requests the full scheduler GPU count is a
        different execution mode: ``_train_one`` will finish the successor
        plan first and hand off the Controller before acquiring the worker
        lease, so capping that request here would make the full-card path
        unreachable.
        """

        planned = dict(self._validate_resource_request(plan.resource_request))
        effective = dict(planned)
        total_gpu_count = int(getattr(self.scheduler, "gpu_count", 4))
        full_card_requested = bool(
            planned.get("distributed")
            and (
                int(planned.get("gpu_count", 0)) >= total_gpu_count
                or int(planned.get("min_gpu_count", 0)) >= total_gpu_count
            )
        )
        if full_card_requested:
            self.events.append(
                "worker_resource_request_full_card",
                {
                    "experiment_id": plan.experiment_id,
                    "operator": plan.operator,
                    "planned_resource_request": dict(planned),
                    "effective_resource_request": dict(effective),
                    "reason": "plan_requests_full_card_before_controller_handoff",
                },
            )
        elif self.config.pipeline_enabled and int(self.config.controller_overlap_gpus) > 0:
            effective = dict(
                pack_worker_request(
                    planned,
                    total_gpu_count=total_gpu_count,
                    controller_overlap_gpus=int(self.config.controller_overlap_gpus),
                )
            )
        if effective != planned:
            self.events.append(
                "worker_resource_request_packed",
                {
                    "experiment_id": plan.experiment_id,
                    "operator": plan.operator,
                    "planned_resource_request": dict(planned),
                    "effective_resource_request": dict(effective),
                    "reason": "reserve_controller_overlap_lane",
                },
            )
        return planned, effective

    def _preserve_controller_lane_for_speculative_worker(self) -> bool:
        """Keep the Controller only when its live lease fits the overlap lane.

        During evaluation the launcher may temporarily use TP2 because no
        training lease exists yet.  Once a validated successor is ready,
        retaining both Controller cards would strand the worker below its
        two-GPU minimum.  Ask the owned launcher to hand off in that case;
        the scheduler then allocates the largest safe worker group.  A
        benchmark evaluator is already an independent live lane, so once its
        successor plan is ready, release even a TP1 Controller and let the
        worker use all three non-evaluator cards.  The next plan is durable at
        this boundary; keeping an idle Controller card would only lower the
        measured training power without improving continuity.
        """

        try:
            value = getattr(self.scheduler, "controller_reserved_gpu_indices", ())
            value = value() if callable(value) else value
            controller_gpus = tuple(sorted(set(int(index) for index in (value or ()))))
        except (OSError, RemoteError, TypeError, ValueError) as exc:
            self.events.append(
                "controller_lane_handoff_deferred",
                {
                    "reason": "controller_lease_unavailable",
                    "error": str(exc)[:500],
                },
            )
            return True
        evaluation_lease_active = False
        if self._active_evaluation_gpu_indices:
            try:
                evaluation_lease_active = bool(self._comfyui_lease_active())
            except (OSError, RemoteError, TypeError, ValueError):
                evaluation_lease_active = False
        if controller_gpus and evaluation_lease_active:
            self.events.append(
                "controller_lane_handoff_requested",
                {
                    "controller_gpus": list(controller_gpus),
                    "evaluator_gpus": list(self._active_evaluation_gpu_indices),
                    "configured_overlap_gpus": int(self.config.controller_overlap_gpus),
                    "reason": "evaluation_successor_ready_use_all_non_evaluator_gpus",
                },
            )
            return False
        overlap = max(0, int(self.config.controller_overlap_gpus))
        if len(controller_gpus) <= overlap:
            return True
        self.events.append(
            "controller_lane_handoff_requested",
            {
                "controller_gpus": list(controller_gpus),
                "configured_overlap_gpus": overlap,
                "reason": "validated_speculative_worker_needs_distributed_capacity",
            },
        )
        return False

    def _validate_worker_resource_match(self, plan: Any) -> None:
        launcher = self.config.worker.operator_launchers.get(plan.operator, "torchrun")
        request = self._validate_resource_request(plan.resource_request)
        if launcher == "python" and plan.operator in {"prune_blocks", "quantize"}:
            if any(request[key] != 0 for key in ("gpu_count", "min_gpu_count", "max_gpu_count")):
                raise ValueError(
                    "%s is a CPU worker and must request zero GPUs" % plan.operator
                )
        if launcher != "torchrun":
            return
        if not request["distributed"] or request["min_gpu_count"] < 2:
            raise ValueError("the trusted torchrun worker requires distributed allocation of at least two GPUs")
        if request["max_gpu_count"] > 4:
            raise ValueError("the trusted torchrun worker supports at most four GPUs")

    def _validate_plan_evidence(self, plan: Any, context: ControllerContext) -> None:
        known = {item.get("observation_id") for item in context.observations}
        consumed = list(plan.consumed_observation_ids)
        if len(consumed) != len(set(consumed)):
            raise ValueError("consumed_observation_ids must be unique")
        unknown = sorted(set(consumed) - known)
        if unknown:
            raise ValueError("plan consumes unknown observation(s): %s" % ", ".join(unknown))
        missing = sorted(set(context.unconsumed_observation_ids) - set(consumed))
        if missing:
            raise ValueError("plan must consume every new observation: %s" % ", ".join(missing))
        evidence = list(plan.diagnosis_evidence)
        if not evidence:
            raise ValueError("plan must declare diagnosis_evidence")
        unknown_evidence = sorted(set(evidence) - known)
        if unknown_evidence:
            raise ValueError(
                "diagnosis_evidence must reference known observations: %s"
                % ", ".join(unknown_evidence)
            )
        # ``consumed_observation_ids`` is a cursor for newly appended evidence,
        # not a requirement to repeat every historical observation used by the
        # diagnosis.  Once a cycle has consumed all new observations, an LLM
        # may quite reasonably cite an older evaluation while leaving this
        # cursor empty.  New observations are still forced through ``missing``
        # above, so this relaxation cannot skip fresh evidence.
        historical = known - set(context.unconsumed_observation_ids)
        if not set(evidence).issubset(set(consumed) | historical):
            raise ValueError("diagnosis_evidence must reference known observations")

    def _prepare_comfyui_leases(self, workers: Sequence[Any]) -> Tuple[Mapping[str, Any], ...]:
        payloads = []
        for worker in workers:
            lease = self._comfyui_lease_for(worker)
            payload = lease.prepare_for_benchmark()
            payload = {
                **dict(payload),
                "port": int(worker.port),
                "gpu_index": int(worker.gpu_index),
            }
            payloads.append(payload)
            self.events.append(
                "comfyui_lease_reserved",
                {"cache_policy": self.config.comfyui_cache_policy, **payload},
            )
        return tuple(payloads)

    def _prepare_comfyui_lease(self) -> Mapping[str, Any]:
        """Backward-compatible primary-worker reservation helper."""

        worker = self.config.comfyui_workers[0]
        return self._prepare_comfyui_leases((worker,))[0]

    def _comfyui_lease_active(self) -> bool:
        return any(
            lease.lease_active or lease.gpu_index in self.scheduler.reserved_gpu_indices
            for lease in self.comfyui_leases.values()
        )

    def _release_comfyui_if_idle(self, reason: str) -> ComfyUILeaseResult:
        results: List[ComfyUILeaseResult] = []
        for (gpu_index, _port), lease in sorted(self.comfyui_leases.items()):
            if not (lease.lease_active or gpu_index in self.scheduler.reserved_gpu_indices):
                continue
            self.events.append(
                "comfyui_cache_release_requested",
                {
                    "reason": reason,
                    "cache_policy": self.config.comfyui_cache_policy,
                    "gpu_index": lease.gpu_index,
                    "port": lease.port,
                },
            )
            result = lease.release_if_idle(
                stop_process=self.config.comfyui_process_policy == "on_demand"
            )
            results.append(result)
            running = result.queue.get("queue_running", []) if isinstance(result.queue, Mapping) else []
            pending = result.queue.get("queue_pending", []) if isinstance(result.queue, Mapping) else []
            payload = {
                "reason": reason,
                "cache_policy": self.config.comfyui_cache_policy,
                "state": result.state,
                "success": result.success,
                "release_reason": result.reason,
                "gpu_index": result.gpu_index,
                "port": lease.port,
                "reserved_gpu_indices": list(
                    getattr(
                        self.scheduler,
                        "reserved_gpu_indices",
                        self.comfyui_lease.scheduler.reserved_gpu_indices,
                    )
                ),
                "elapsed_s": result.elapsed_s,
                "queue": {
                    "running": len(running) if isinstance(running, (list, tuple, dict)) else None,
                    "pending": len(pending) if isinstance(pending, (list, tuple, dict)) else None,
                },
                "memory_snapshot": {str(key): list(value) for key, value in result.memory_snapshot.items()},
            }
            self.events.append(
                "comfyui_cache_released" if result.success else "comfyui_cache_release_failed",
                payload,
            )
        primary = next((item for item in results if item.gpu_index == self.comfyui_lease.gpu_index), None)
        if primary is None and results:
            primary = results[0]
        if primary is None:
            primary = ComfyUILeaseResult(
                "released_for_other_work",
                True,
                "no_active_comfyui_lease",
                {},
                {},
                {},
                0.0,
                self.comfyui_lease.gpu_index,
            )
        # The evaluator GPU tuple is telemetry state, not ownership.  Clear
        # it as soon as every ComfyUI lease has actually left the scheduler so
        # a later training lane cannot report an old evaluator card as active
        # (or use it as a handoff hint after a restart).  Keep it on any
        # failed/queued release because the evaluator may still own memory.
        if not any(
            lease.lease_active or lease.gpu_index in self.scheduler.reserved_gpu_indices
            for lease in self.comfyui_leases.values()
        ):
            self._active_evaluation_gpu_indices = ()
        self.last_comfyui_lease = primary
        return primary

    def _release_worker_gpu_lease(self, experiment_id: str, reason: str) -> None:
        """Release the shared worker reservation without masking a worker result."""

        release = getattr(self.scheduler, "release", None)
        if not callable(release):
            return
        requested_owner = str(experiment_id)
        with self._worker_lease_lock:
            lease_owner = self._worker_lease_experiment_id
            # ``run-finally`` is the only cleanup path that is allowed to
            # release whichever lease this process still owns.  A normal
            # worker completion must match its own experiment exactly: the
            # CPU primary can finish while its GPU sibling is still running,
            # and releasing here would remove the sibling's shared lease.
            if (
                lease_owner is not None
                and requested_owner not in {lease_owner, "run-finally"}
            ):
                self.events.append(
                    "worker_gpu_lease_release_skipped",
                    {
                        "experiment_id": requested_owner,
                        "lease_owner_experiment_id": lease_owner,
                        "reason": reason,
                        "release_reason": "lease_owned_by_concurrent_worker",
                    },
                )
                return
            if not bool(getattr(self.scheduler, "lease_active", False)):
                if requested_owner == lease_owner or requested_owner == "run-finally":
                    self._worker_lease_experiment_id = None
                return
            try:
                result = dict(release())
            except Exception as exc:
                # The launcher applies a bounded lease TTL, so a cleanup
                # transport failure cannot strand GPUs forever. Keep the
                # owner marker until a later matching cleanup can retry.
                self.events.append(
                    "worker_gpu_lease_release_failed",
                    {
                        "experiment_id": requested_owner,
                        "reason": reason,
                        "error": str(exc)[:1000],
                    },
                )
                return
            self._worker_lease_experiment_id = None
            self.events.append(
                "worker_gpu_lease_released",
                {
                    "experiment_id": requested_owner,
                    "reason": reason,
                    "result": result,
                },
            )

    def _acquire_worker_gpu_lease(
        self,
        experiment_id: str,
        request: Mapping[str, Any],
    ) -> ResourceDecision:
        """Acquire the single shared worker lease for one logical worker.

        The scheduler lease is intentionally one host-wide union reservation.
        Do not let a second in-process speculative launcher overwrite it while
        the first worker is still running; the caller will record a bounded
        resource wait and keep the plan durable for a later boundary.
        """

        requested_owner = str(experiment_id)
        with self._worker_lease_lock:
            lease_owner = self._worker_lease_experiment_id
            if lease_owner is not None and lease_owner != requested_owner:
                normalized = self.scheduler.normalize_request(request)
                return ResourceDecision(
                    status="wait",
                    requested_gpu_count=int(normalized["gpu_count"]),
                    reason="campaign worker lease is owned by %s" % lease_owner,
                    min_gpu_count=int(normalized["min_gpu_count"]),
                    max_gpu_count=int(normalized["max_gpu_count"]),
                    elastic=bool(normalized["elastic"]),
                    available_gpu_count=0,
                )
            decision = self.scheduler.acquire(request)
            if decision.status == "ready" and decision.actual_gpu_count > 0:
                self._worker_lease_experiment_id = requested_owner
            return decision

    def _retain_checkpoint(
        self,
        checkpoint_path: Any,
        model_id: Any,
        outcome: str,
        parent_checkpoint_path: Any = None,
    ) -> Mapping[str, Any]:
        """Apply the configured remote weight-retention policy at a boundary."""

        result = self.checkpoint_retention.apply(
            checkpoint_path,
            model_id,
            outcome,
            parent_checkpoint_path=parent_checkpoint_path,
        )
        payload = result.to_dict()
        self.events.append("checkpoint_retention", payload)
        return payload

    def _demote_rejected_active(
        self,
        candidates: Mapping[str, ModelCandidate],
        evaluations: Mapping[str, Mapping[str, Any]],
        records: Sequence[ExperienceRecord],
        split: Optional[str],
    ) -> Optional[str]:
        """Move the active pointer off a fully evaluated rejected child.

        Retention is allowed to delete rejected weights, so an active child
        with a rejected decision must be demoted before reconciliation.  This
        also repairs campaigns interrupted after evaluation but before the
        next loop persisted the parent pointer.
        """

        try:
            active_id = self.models.active_id
        except ModelStoreError:
            return None
        if active_id == "M0000" or active_id not in candidates:
            return None
        summary = evaluations.get(active_id, {})
        if not self._evaluation_is_current(summary, split):
            return None
        checkpoint_present = True
        try:
            checkpoint_present = int(
                getattr(self.ssh.run(("test", "-e", candidates[active_id].checkpoint_path), check=False), "returncode", 1)
            ) == 0
        except (RemoteError, OSError):
            checkpoint_present = False
        latest = [
            record
            for record in records
            if str(record.child_model_id or "") == active_id
            and record.status in {"rejected", "evaluated_candidate", "accepted"}
        ]
        if not latest:
            return None
        record = latest[-1]
        decision = record.decision if isinstance(record.decision, Mapping) else {}
        if checkpoint_present and (
            record.status == "accepted"
            or decision.get("accepted") is True
            or decision.get("pareto_eligible") is True
        ):
            return None
        parent_id = candidates[active_id].parent_id
        if not parent_id or parent_id not in candidates:
            return None
        reason = (
            "active_candidate_checkpoint_missing"
            if not checkpoint_present
            else "rejected_active_candidate_checkpoint_cleanup"
        )
        self.models.set_active(parent_id)
        parent_system = self._system_for_model(parent_id)
        if parent_system is not None:
            self.systems.set_active(parent_system.id)
        self.events.append(
            "active_model_demoted",
            {
                "model_id": active_id,
                "parent_model_id": parent_id,
                "reason": reason,
            },
        )
        return parent_id

    def _reconcile_checkpoint_retention(
        self,
        candidates: Mapping[str, ModelCandidate],
        evaluations: Mapping[str, Mapping[str, Any]],
        records: Sequence[ExperienceRecord],
        split: Optional[str],
    ) -> List[Mapping[str, Any]]:
        """Clean rejected children left by an interrupted earlier campaign."""

        if self.config.checkpoint_retention_policy == "keep_all":
            return []
        latest: Dict[str, ExperienceRecord] = {}
        # A restart may re-import a result JSON after an already evaluated
        # record (for example while recovering a retained quantized child).
        # Prefer the latest evaluated decision for retention reconciliation;
        # an unvalidated import must not hide a legacy evaluated decision and
        # cause the checkpoint to be deleted again.
        for record in records:
            if record.child_model_id:
                model_id = str(record.child_model_id)
                current = latest.get(model_id)
                if current is None:
                    latest[model_id] = record
                elif record.status in {"rejected", "evaluated_candidate"}:
                    latest[model_id] = record
                elif current.status not in {"rejected", "evaluated_candidate"}:
                    latest[model_id] = record
        reconciled: List[Mapping[str, Any]] = []
        evaluations_changed = False
        for model_id, candidate in candidates.items():
            if model_id == "M0000" or not self._evaluation_is_current(evaluations.get(model_id, {}), split):
                continue
            record = latest.get(model_id)
            if record is None:
                continue
            decision = record.decision if isinstance(record.decision, Mapping) else {}
            # Older campaigns evaluated CPU quantization with the generic
            # training gates (optimizer steps/gradients), then deleted a
            # valid Pareto-improving checkpoint during retention.  Recompute
            # that legacy decision with the operator-aware evidence gates
            # before reconciling the artifact, so a restart can recover the
            # useful lineage instead of silently losing it.
            if (
                record.operator == "quantize"
                and record.status == "rejected"
                and any(str(item).startswith("training_") for item in decision.get("violations", []))
            ):
                legacy_summary = record.evaluation if isinstance(record.evaluation, Mapping) else evaluations.get(model_id, {})
                parent_summary = evaluations.get(str(record.parent_model_id or candidate.parent_id), {})
                corrected = decide(
                    AcceptanceInput(
                        training_metrics=record.training,
                        benchmark_summary=legacy_summary,
                        parent_summary=parent_summary,
                        target=self.target,
                        efficiency_thresholds=self.config.efficiency_thresholds,
                        reward_weights=self.config.reward,
                        research_grade=self.config.research_grade,
                    )
                )
                if corrected.pareto_eligible:
                    parent_candidate = candidates.get(candidate.parent_id)
                    retention = self._retain_checkpoint(
                        candidate.checkpoint_path,
                        candidate.id,
                        "accepted_candidate",
                        parent_checkpoint_path=parent_candidate.checkpoint_path if parent_candidate is not None else None,
                    )
                    saved = dict(evaluations.get(model_id) or {})
                    saved["checkpoint_retention"] = retention
                    evaluations[model_id] = saved
                    self._save_evaluations(evaluations)
                    candidate_system = self._system_for_model(candidate.id)
                    evaluation = _evaluation_result(
                        legacy_summary,
                        model_id=candidate.id,
                        system_id=candidate_system.id if candidate_system is not None else None,
                        device_id=candidate_system.device_id if candidate_system is not None else self.target.hardware_name,
                        task_split=split,
                    )
                    if evaluation is not None:
                        pareto_id = candidate_system.id if candidate_system is not None else candidate.id
                        self.pareto.update(pareto_id, evaluation)
                    self.models.set_active(candidate.id)
                    if candidate_system is not None:
                        self.systems.set_active(candidate_system.id)
                    self._append_evaluated_experience(record, legacy_summary, corrected)
                    self.events.append(
                        "legacy_decision_migrated",
                        {
                            "model_id": candidate.id,
                            "operator": record.operator,
                            "old_violations": list(decision.get("violations", [])),
                            "new_violations": list(corrected.violations),
                            "pareto_eligible": corrected.pareto_eligible,
                            "checkpoint_retention": retention,
                        },
                    )
                    continue
            if record.status not in {"rejected", "evaluated_candidate"}:
                continue
            if decision.get("accepted") is True or decision.get("pareto_eligible") is True:
                continue
            prior_retention = evaluations.get(model_id, {}).get("checkpoint_retention") if isinstance(evaluations.get(model_id), Mapping) else None
            if (
                isinstance(prior_retention, Mapping)
                and prior_retention.get("retained") is False
                and prior_retention.get("outcome") in {"rejected_candidate", "superseded_candidate"}
            ):
                # The checkpoint decision is durable in evaluations.json.
                # Do not re-issue an SSH unlink (or append the same absent
                # event) on every restart of a long campaign.
                continue
            retention = self._retain_checkpoint(
                candidate.checkpoint_path,
                candidate.id,
                "rejected_candidate",
                parent_checkpoint_path=(candidates.get(candidate.parent_id).checkpoint_path if candidate.parent_id in candidates else None),
            )
            reconciled.append(retention)
            if isinstance(evaluations, dict):
                saved = dict(evaluations.get(model_id) or {})
                saved["checkpoint_retention"] = retention
                evaluations[model_id] = saved
                evaluations_changed = True
        if evaluations_changed:
            self._save_evaluations(evaluations)
        return reconciled

    def _reclaim_superseded_checkpoints(
        self,
        candidates: Mapping[str, ModelCandidate],
        evaluations: Dict[str, Dict[str, Any]],
        split: Optional[str],
    ) -> List[Mapping[str, Any]]:
        """Keep a bounded rollback set of completed lineage weights.

        The append-only model/evaluation/experience records remain the full
        experiment memory. Only checkpoint payloads are capped: the active
        child, its evaluated parent, then Pareto/recent candidates get the
        limited slots. This prevents a long RSI campaign from turning every
        accepted candidate into another 26--66 GB permanent copy while still
        leaving enough weights for a safe rollback and future comparison.
        """

        if self.config.checkpoint_retention_policy == "keep_all":
            return []
        try:
            active_id = self.models.active_id
        except ModelStoreError:
            active_id = "M0000"

        evaluated_ids = {
            str(model_id)
            for model_id in candidates
            if model_id != "M0000" and self._evaluation_is_current(evaluations.get(model_id, {}), split)
        }
        keep_limit = int(self.config.max_retained_checkpoints)
        keep_order: List[str] = []

        def add_keep(model_id: Any, *, require_evaluation: bool = True) -> None:
            model = str(model_id or "")
            if not model or model == "M0000" or model not in candidates or model in keep_order:
                return
            if require_evaluation and model not in evaluated_ids:
                return
            keep_order.append(model)

        # The active model is never reclaimed, even if a process was
        # interrupted between worker completion and evaluation persistence.
        add_keep(active_id, require_evaluation=False)
        active_candidate = candidates.get(str(active_id))
        if active_candidate is not None:
            # Keep one direct rollback parent when the cap allows it.
            add_keep(active_candidate.parent_id)

        front_system_ids = {entry.candidate_id for entry in self.pareto.front()}
        for item in self.systems.lineage():
            if item.id in front_system_ids:
                add_keep(item.model_ref)

        # If the Pareto front has more entries than the cap, prefer recent
        # evaluated candidates after the active/parent slots. Generation is
        # stable across restarts and the ID makes ties deterministic.
        for item in sorted(
            (candidate for candidate in candidates.values() if candidate.id in evaluated_ids),
            key=lambda candidate: (candidate.generation, candidate.id),
            reverse=True,
        ):
            add_keep(item.id)
        keep_ids = set(keep_order[:keep_limit])

        reclaimed: List[Mapping[str, Any]] = []
        for model_id, candidate in candidates.items():
            if model_id == "M0000" or model_id in keep_ids or model_id not in evaluated_ids:
                continue
            saved = evaluations.get(model_id, {})
            prior_retention = saved.get("checkpoint_retention") if isinstance(saved, Mapping) else None
            if isinstance(prior_retention, Mapping) and prior_retention.get("retained") is False:
                continue
            result = self._retain_checkpoint(
                candidate.checkpoint_path,
                candidate.id,
                "superseded_candidate",
                parent_checkpoint_path=(
                    candidates.get(candidate.parent_id).checkpoint_path
                    if candidate.parent_id in candidates
                    else None
                ),
            )
            reclaimed.append(result)
            updated = dict(saved)
            updated["checkpoint_retention"] = result
            evaluations[model_id] = updated
        if reclaimed:
            self._save_evaluations(evaluations)
        self.events.append(
            "checkpoint_retention_cap",
            {
                "max_retained_checkpoints": keep_limit,
                "active_model_id": str(active_id),
                "kept_model_ids": list(keep_order[:keep_limit]),
                "evaluated_model_count": len(evaluated_ids),
                "reclaimed_model_ids": [
                    str(item.get("model_id")) for item in reclaimed if item.get("model_id")
                ],
                "metadata_preserved": True,
            },
        )
        return reclaimed

    @staticmethod
    def _clear_pending_state(state: Dict[str, Any]) -> None:
        for key in (
            "pending_plan",
            "pending_parent_model_id",
            "pending_training_calls",
            "pending_attempts",
            "pending_last_resource_decision",
            "pending_last_resource_review_signature",
            "prefetched_plan",
            "prefetched_parent_model_id",
            "prefetched_training_calls",
            "prefetched_source_experiment_id",
            "prefetched_source_child_model_id",
            "prefetched_observation_ids",
            "prefetched_created_at",
        ):
            state.pop(key, None)

    @staticmethod
    def _clear_parallel_prefetch_state(state: Dict[str, Any]) -> None:
        """Clear only the isolated GPU-fill plan cursor."""

        for key in (
            "parallel_prefetched_plan",
            "parallel_prefetched_parent_model_id",
            "parallel_prefetched_training_calls",
            "parallel_prefetched_source_experiment_id",
            "parallel_prefetched_source_child_model_id",
            "parallel_prefetched_observation_ids",
            "parallel_prefetched_created_at",
        ):
            state.pop(key, None)

    @classmethod
    def _clear_prefetch_state_for_child(
        cls,
        state: Dict[str, Any],
        child_model_id: str,
    ) -> Tuple[bool, bool]:
        """Drop only overlap cursors rooted at a failed child.

        A failed CPU primary can have already published both its ordinary and
        GPU-fill successor plans.  Keep cursors for other lineages intact so
        recovery never turns a worker failure into a global queue flush.
        """

        child = str(child_model_id)
        primary_stale = str(state.get("prefetched_source_child_model_id") or "") == child
        parallel_stale = str(state.get("parallel_prefetched_source_child_model_id") or "") == child
        if primary_stale:
            cls._clear_pending_state(state)
        if parallel_stale:
            cls._clear_parallel_prefetch_state(state)
        return primary_stale, parallel_stale

    def _persist_pending_plan(
        self,
        state: Dict[str, Any],
        plan: ExperimentPlan,
        training_calls: int,
        resource_decision: Mapping[str, Any],
    ) -> int:
        attempts = int(state.get("pending_attempts", 0)) + 1
        state.update(
            {
                "pending_plan": plan.to_dict(),
                "pending_parent_model_id": plan.parent_model_id,
                "pending_training_calls": int(training_calls),
                "pending_attempts": attempts,
                "pending_last_resource_decision": dict(resource_decision),
            }
        )
        self._save_campaign_state(state)
        self.events.append(
            "resource_queued" if attempts == 1 else "resource_retry",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "attempt": attempts,
                "resource_request": dict(plan.resource_request),
                "resource_decision": dict(resource_decision),
            },
        )
        return attempts

    def _pending_plan(self, state: Mapping[str, Any]) -> Optional[ExperimentPlan]:
        raw = state.get("pending_plan")
        if not isinstance(raw, Mapping):
            return None
        plan = self.validation.schema.validate(ExperimentPlan.from_dict(raw))
        plan = replace(plan, resource_request=self._validate_resource_request(plan.resource_request))
        self._validate_worker_resource_match(plan)
        return plan

    def _prefetched_plan(self, state: Mapping[str, Any]) -> Optional[ExperimentPlan]:
        """Decode a speculative plan produced while the previous worker ran."""

        raw = state.get("prefetched_plan")
        if not isinstance(raw, Mapping):
            return None
        # Keep the persisted shape extensible: the plan is nested so metadata
        # such as the context generation and creation time cannot be mistaken
        # for ExperimentPlan fields on resume.
        plan_raw = raw.get("plan", raw)
        if not isinstance(plan_raw, Mapping):
            return None
        plan = self.validation.schema.validate(ExperimentPlan.from_dict(plan_raw))
        plan = replace(plan, resource_request=self._validate_resource_request(plan.resource_request))
        self._validate_worker_resource_match(plan)
        return plan

    def _parallel_prefetched_plan(self, state: Mapping[str, Any]) -> Optional[ExperimentPlan]:
        """Decode the isolated GPU-fill plan produced during the prior worker.

        This cursor is intentionally separate from ``prefetched_plan``.  The
        primary successor may be a CPU-only prune/quantize operation while a
        second, independent GPU branch is prepared for the evaluation
        window.  Neither branch is imported into the active model archive
        until its own worker/evaluation path decides what to do.
        """

        raw = state.get("parallel_prefetched_plan")
        if not isinstance(raw, Mapping):
            return None
        plan_raw = raw.get("plan", raw)
        if not isinstance(plan_raw, Mapping):
            return None
        plan = self.validation.schema.validate(ExperimentPlan.from_dict(plan_raw))
        plan = replace(plan, resource_request=self._validate_resource_request(plan.resource_request))
        self._validate_worker_resource_match(plan)
        return plan

    @staticmethod
    def _parallel_prefetch_is_current(
        state: Mapping[str, Any],
        plan: Optional[ExperimentPlan],
        candidate_model_id: str,
        training_call_cursor: int,
    ) -> bool:
        """Check the durable GPU-fill cursor against its exact boundary.

        A persisted sibling is useful only for the candidate and Controller
        cursor that produced it.  The optional source-child check keeps older
        state files compatible while rejecting a cursor left by another
        evaluation lineage.
        """

        if plan is None or plan.parent_model_id != str(candidate_model_id):
            return False
        try:
            if int(state.get("parallel_prefetched_training_calls")) != int(training_call_cursor):
                return False
        except (TypeError, ValueError):
            return False
        source_child_id = str(state.get("parallel_prefetched_source_child_model_id") or "")
        return not source_child_id or source_child_id == str(candidate_model_id)

    def _ready_parallel_prefetched_plan(
        self,
        state: Mapping[str, Any],
        candidate_model_id: str,
        training_call_cursor: int,
    ) -> Optional[ExperimentPlan]:
        """Return a parallel cursor that is safe to launch at this boundary.

        The primary successor and the GPU-fill successor are deliberately
        persisted in separate state slots.  Evaluation must be able to
        consume the latter even when the primary slot was rejected, expired,
        or was never written after a restart.
        """

        try:
            plan = self._parallel_prefetched_plan(state)
        except (TypeError, ValueError, KeyError):
            return None
        if self._parallel_prefetch_is_current(
            state,
            plan,
            candidate_model_id,
            training_call_cursor,
        ):
            return plan
        return None

    def _parallel_gpu_fill_execution_plan(
        self,
        plan: ExperimentPlan,
    ) -> ExperimentPlan:
        """Cap only the overlap worker to the cards not owned by evaluation.

        A parallel fill plan is an elastic overlap job, not the campaign's
        full-card handoff path.  Keeping a stale ``max_gpu_count=4`` here
        makes downstream code classify it as full-card training and can leave
        the evaluator/worker boundary waiting on an impossible allocation.
        """

        request = dict(self._validate_resource_request(plan.resource_request))
        if not bool(request.get("distributed")) or not bool(request.get("elastic")):
            return plan
        total_gpu_count = int(getattr(self.scheduler, "gpu_count", 4))
        evaluator_count = len(tuple(self._active_evaluation_gpu_indices or ()))
        overlap_cap = total_gpu_count - max(1, evaluator_count)
        minimum = int(request.get("min_gpu_count", 0))
        if overlap_cap < minimum:
            return plan
        packed = dict(request)
        packed["max_gpu_count"] = min(int(packed["max_gpu_count"]), overlap_cap)
        packed["gpu_count"] = min(int(packed["gpu_count"]), packed["max_gpu_count"])
        if packed == request:
            return plan
        self.events.append(
            "parallel_gpu_fill_request_packed",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "evaluator_gpu_indices": list(self._active_evaluation_gpu_indices or ()),
                "planned_resource_request": dict(request),
                "effective_resource_request": dict(packed),
                "reason": "parallel_overlap_excludes_evaluation_cards",
            },
        )
        return replace(plan, resource_request=packed)

    @staticmethod
    def _parallel_candidate_from_prefetch_handle(
        handle: Optional[Mapping[str, Any]],
        primary_plan: ExperimentPlan,
        training_calls: int,
    ) -> Optional[ExperimentPlan]:
        """Reuse a safe GPU candidate from the primary n-way response.

        The primary prefetch already asks the Controller for multiple
        candidates.  When its selected plan is CPU-only, throwing the other
        locally-eligible candidates away forces a second LLM round-trip and
        leaves the evaluation GPUs idle.  Only a narrow, same-parent,
        distributed-training candidate is reusable here; the normal worker
        validation and scheduler gates still run after this method returns.
        """

        if not isinstance(handle, Mapping):
            return None
        if str(handle.get("prefetch_state_key") or "primary") != "primary":
            return None
        try:
            if int(handle.get("training_calls")) != int(training_calls):
                return None
        except (TypeError, ValueError):
            return None
        result = handle.get("result")
        if not isinstance(result, Mapping):
            return None
        selected = result.get("plan")
        if not isinstance(selected, ExperimentPlan):
            return None
        if (
            selected.experiment_id != primary_plan.experiment_id
            or selected.parent_model_id != primary_plan.parent_model_id
        ):
            return None
        match = re.fullmatch(r"exp_(\d+)", str(primary_plan.experiment_id))
        if match is None:
            return None
        next_experiment_id = "exp_%04d" % (int(match.group(1)) + 1)
        allowed_operators = {"recovery_finetune", "distill", "step_distill", "dmd2"}
        alternatives = result.get("parallel_candidates")
        if not isinstance(alternatives, (list, tuple)):
            return None
        eligible: List[Tuple[int, int, int, ExperimentPlan]] = []
        for position, candidate in enumerate(alternatives):
            if not isinstance(candidate, ExperimentPlan):
                continue
            # All candidates sampled from one schema call share the primary
            # experiment id; identity/equality, rather than that id, marks
            # the selected candidate.
            if candidate is selected or candidate == selected:
                continue
            if candidate.parent_model_id != primary_plan.parent_model_id:
                continue
            if candidate.parent_system_id not in (None, primary_plan.parent_system_id):
                # All alternatives came from the same Controller response.
                # A stale root system id is safe to rebase when the model
                # lineage is identical; the selected primary plan has
                # already passed the authoritative system check above.
                candidate = replace(
                    candidate,
                    parent_system_id=primary_plan.parent_system_id,
                )
            if candidate.operator not in allowed_operators:
                continue
            request = candidate.resource_request
            if not isinstance(request, Mapping):
                continue
            if (
                request.get("elastic") is not True
                or request.get("distributed") is not True
                or request.get("exclusive") is not False
                or request.get("evaluation_workers") != 1
                or request.get("min_gpu_count") != 2
                or request.get("max_gpu_count") not in {2, 3, 4}
                or request.get("gpu_count") not in {2, 3, 4}
            ):
                continue
            eligible.append(
                (
                    int(request.get("max_gpu_count", 0)),
                    int(request.get("gpu_count", 0)),
                    -int(position),
                    candidate,
                )
            )
        if not eligible:
            return None
        # Prefer the widest elastic ceiling so a live scheduler can use every
        # safe card left after evaluator/Controller/foreign-process leases.
        # Preserve response order for equal-capacity candidates.
        candidate = max(eligible, key=lambda item: item[:3])[3]
        return replace(
            candidate,
            experiment_id=next_experiment_id,
            parent_model_id=primary_plan.parent_model_id,
            parent_system_id=primary_plan.parent_system_id,
        )

    @staticmethod
    def _prefetch_parent_candidate(
        parent: ModelCandidate,
        source_plan: ExperimentPlan,
        child_model_id: str,
    ) -> ModelCandidate:
        """Build the child-state view used by the overlapping next plan.

        The next Controller call runs before the current worker has published
        its result.  Passing the old parent state makes a binary step-distill
        plan repeat the same target (for example 32 -> 16, then another 16)
        and the trusted worker correctly rejects it.  Predict only the
        deterministic state transitions known from the selected operator;
        measured quality and hardware evidence remain those of the real
        parent until the worker and benchmark publish new evidence.
        """

        state = parent.state
        algorithm_state = {
            **dict(state.algorithm_state),
            "operator": source_plan.operator,
            "prefetch_source_experiment_id": source_plan.experiment_id,
        }
        changes: Dict[str, Any] = {
            "model_id": child_model_id,
            "parent_model_id": parent.id,
            "checkpoint_path": "pending://%s.safetensors" % child_model_id,
            "algorithm_state": algorithm_state,
            "provenance": {
                **dict(state.provenance),
                "prefetch_prediction": True,
                "prefetch_source_experiment_id": source_plan.experiment_id,
            },
        }
        if source_plan.operator == "step_distill":
            changes["sampling_steps"] = int(source_plan.operator_args["target_steps"])
            changes["algorithm_state"] = {
                **algorithm_state,
                "source_steps": int(state.sampling_steps or 32),
                "target_steps": int(source_plan.operator_args["target_steps"]),
            }
        elif source_plan.operator == "quantize":
            changes["quantization"] = {
                **dict(state.quantization),
                "bits": int(source_plan.operator_args["bits"]),
                "scheme": "prefetch_prediction",
            }
        predicted_state = replace(state, **changes)
        return ModelCandidate(
            child_model_id,
            parent.id,
            parent.generation + 1,
            predicted_state.checkpoint_path,
            predicted_state,
            source_plan.experiment_id,
            "speculative_parent",
            {
                **dict(parent.metadata),
                "prefetch_prediction": True,
                "source_experiment_id": source_plan.experiment_id,
            },
        )

    def _start_controller_prefetch(
        self,
        parent: ModelCandidate,
        training_calls: int,
        *,
        source_plan: Optional[ExperimentPlan] = None,
        source_experiment_id: Optional[str] = None,
        source_child_model_id: Optional[str] = None,
        planning_intent: str = "primary",
        operator_filter: Optional[Sequence[str]] = None,
        prefetch_state_key: str = "primary",
        experiment_cursor: Optional[int] = None,
    ) -> Optional[Mapping[str, Any]]:
        """Start one cloned-Controller plan call while the worker is busy.

        The clone is important: heartbeat reviews and the main Controller
        call may update provider request metadata at the same time. A
        speculative plan is never allowed to mutate the live provider object.
        """

        if not callable(getattr(self.controller, "plan", None)):
            return None
        try:
            planner = copy.deepcopy(self.controller)
        except Exception as exc:
            self.events.append(
                "controller_plan_prefetch_skipped",
                {"reason": "provider_not_cloneable", "error": str(exc)[:500]},
            )
            return None
        # Preserve the configured n-way candidate generation for the cloned
        # Controller.  The overlap coordinator can reuse a locally-eligible
        # GPU candidate when the selected successor is CPU-only, avoiding a
        # second full planning request while the evaluator is running.  The
        # normal validation pipeline still owns safety and worker eligibility.
        if str(getattr(planner, "provider_name", "")) == "vllm":
            if hasattr(planner, "candidate_count"):
                if str(planning_intent or "") in {"parallel_gpu_fill", "evaluation_refill"}:
                    # This branch has a narrow operator filter and only needs
                    # one independently validated distributed candidate.  A
                    # four-way sample followed by a second selector call
                    # needlessly holds the evaluation overlap window open.
                    planner.candidate_count = 1
                    if hasattr(planner, "candidate_temperature"):
                        planner.candidate_temperature = 0.0
                    if hasattr(planner, "timeout_s"):
                        try:
                            planner.timeout_s = min(
                                180.0,
                                max(1.0, float(getattr(planner, "timeout_s", 180.0))),
                            )
                        except (TypeError, ValueError):
                            planner.timeout_s = 180.0
                else:
                    planner.candidate_count = max(1, min(8, int(getattr(planner, "candidate_count", 1) or 1)))
        next_training_calls = int(training_calls) + 1
        prefetch_experiment_cursor: Optional[int] = (
            int(experiment_cursor) if experiment_cursor is not None else None
        )
        if prefetch_experiment_cursor is None and source_plan is not None:
            match = re.fullmatch(r"exp_(\d+)", str(source_plan.experiment_id))
            if match is not None:
                # The source plan has not entered the append-only experiment
                # stream yet.  Make the successor's schema const advance past
                # that in-flight id, while preserving the boundary's actual
                # training-call cursor in durable state.
                prefetch_experiment_cursor = int(match.group(1))
        try:
            planning_parent = parent
            if source_plan is not None and source_child_model_id:
                planning_parent = self._prefetch_parent_candidate(
                    parent,
                    source_plan,
                    str(source_child_model_id),
                )
            prefetch_context = self._controller_context(
                planning_parent,
                next_training_calls,
                experiment_cursor=prefetch_experiment_cursor,
            )
            observation_ids = list(prefetch_context.unconsumed_observation_ids)
        except Exception as exc:
            self.events.append(
                "controller_plan_prefetch_skipped",
                {"reason": "context_unavailable", "error": str(exc)[:1000]},
            )
            return None
        result: Dict[str, Any] = {}
        started = time.monotonic()

        # The prefetch thread may finish long before the worker.  Persist the
        # validated plan as soon as it does so a supervisor restart or a
        # heartbeat review can observe and reuse it without waiting for the
        # training boundary.  The per-handle lock makes the boundary join
        # idempotent when it races with this background finalizer.
        handle: Dict[str, Any] = {
            "thread": None,
            "result": result,
            "parent_model_id": parent.id,
            "planning_parent_model_id": planning_parent.id if "planning_parent" in locals() else parent.id,
            "training_calls": next_training_calls,
            "source_experiment_id": source_experiment_id,
            "source_child_model_id": source_child_model_id,
            "observation_ids": observation_ids,
            "started": started,
            "persist_lock": threading.Lock(),
            "prefetch_persisted": False,
            "planning_intent": str(planning_intent or "primary"),
            "operator_filter": list(operator_filter) if operator_filter is not None else None,
            "prefetch_state_key": str(prefetch_state_key or "primary"),
        }

        def run() -> None:
            try:
                # A benchmark lease can stop the current vLLM child just
                # before this fallback starts. Retry only that explicit
                # availability window while the benchmark keeps the other
                # GPU(s) busy; malformed/rejected plans are not retried.
                retry_deadline = time.monotonic() + max(
                    120.0,
                    float(getattr(planner, "timeout_s", 180.0) or 180.0) + 120.0,
                )
                attempt = 0
                while True:
                    attempt += 1
                    result["plan"] = self._controller_plan(
                        planning_parent,
                        next_training_calls,
                        controller=planner,
                        prefetch=True,
                        experiment_cursor=prefetch_experiment_cursor,
                        planning_intent=str(planning_intent or "primary"),
                        operator_filter=operator_filter,
                    )
                    # The selected plan remains the only normal successor,
                    # but an isolated GPU-fill branch may reuse another
                    # candidate from the same structured n-way Controller
                    # response.  This avoids a second LLM request while the
                    # evaluator and Controller already occupy two cards.
                    alternatives = getattr(planner, "last_eligible_candidates", ())
                    if isinstance(alternatives, (list, tuple)):
                        result["parallel_candidates"] = [
                            item for item in alternatives
                            if isinstance(item, ExperimentPlan)
                        ]
                    if result.get("plan") is not None:
                        break
                    last_trace = (
                        self.controller_prefetch_trace[-1]
                        if self.controller_prefetch_trace
                        else {}
                    )
                    unavailable = str(last_trace.get("status", "")) == "controller_unavailable"
                    if not unavailable or time.monotonic() >= retry_deadline:
                        break
                    self.events.append(
                        "controller_plan_prefetch_retry",
                        {
                            **self._controller_provider_metadata(planner),
                            "parent_model_id": planning_parent.id,
                            "training_calls": next_training_calls,
                            "attempt": attempt + 1,
                            "reason": "remote_controller_restart_window",
                            "remaining_s": max(0.0, retry_deadline - time.monotonic()),
                        },
                    )
                    time.sleep(min(15.0, max(2.0, retry_deadline - time.monotonic())))
                plan = result.get("plan")
                if isinstance(plan, ExperimentPlan):
                    self._persist_controller_prefetch(handle, plan)
            except Exception as exc:
                result["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:1500])

        thread = threading.Thread(
            target=run,
            name="harness4h3-controller-plan-prefetch-%s" % parent.id,
            daemon=True,
        )
        handle["thread"] = thread
        thread.start()
        self.events.append(
            "controller_plan_prefetch_started",
            {
                **self._controller_provider_metadata(planner),
                "parent_model_id": parent.id,
                "planning_parent_model_id": handle.get("planning_parent_model_id"),
                "training_calls": next_training_calls,
                "source_experiment_id": source_experiment_id,
                "source_child_model_id": source_child_model_id,
                "planning_intent": str(planning_intent or "primary"),
                "prefetch_state_key": str(prefetch_state_key or "primary"),
                "candidate_count": int(getattr(planner, "candidate_count", 1) or 1),
                "timeout_s": float(getattr(planner, "timeout_s", 0.0) or 0.0),
                "reason": "overlap_with_trusted_worker",
            },
        )
        return handle

    def _early_parallel_prefetch_job(
        self,
        parent_model_id: str,
        training_calls: int,
    ) -> Optional[Mapping[str, Any]]:
        """Return an in-memory early GPU-fill job for one boundary."""

        key = (str(parent_model_id), int(training_calls))
        with self._early_parallel_prefetch_lock:
            return self._early_parallel_prefetch_jobs.get(key)

    def _arm_eager_parallel_gpu_prefetch(
        self,
        parent: ModelCandidate,
        current_plan: ExperimentPlan,
        current_child_id: str,
        training_calls: int,
    ) -> Optional[Mapping[str, Any]]:
        """Start the GPU-fill request while a CPU-only worker is running.

        A CPU-only primary successor can be valid and useful while still
        leaving the two evaluation overlap cards empty.  Do not wait for that
        primary successor to finish before asking for the independent GPU
        branch: the existing bounded in-memory job and durable parallel cursor
        let the evaluation callback consume this exact request without a
        duplicate Controller call.
        """

        if (
            not self.config.pipeline_enabled
            or int(self.config.pipeline_max_inflight) < 2
            or str(current_plan.operator) not in {"prune_blocks", "quantize"}
        ):
            return None
        match = re.fullmatch(r"exp_(\d+)", str(current_plan.experiment_id))
        if match is None:
            self.events.append(
                "controller_plan_parallel_prefetch_skipped",
                {
                    "parent_model_id": str(current_child_id),
                    "training_calls": int(training_calls) + 1,
                    "reason": "source_experiment_id_not_numeric",
                },
            )
            return None
        next_training_calls = int(training_calls) + 1
        key = (str(current_child_id), next_training_calls)
        with self._early_parallel_prefetch_lock:
            existing = self._early_parallel_prefetch_jobs.get(key)
            if existing is not None:
                return existing
        handle = self._start_controller_prefetch(
            parent,
            training_calls,
            source_plan=current_plan,
            source_experiment_id=current_plan.experiment_id,
            source_child_model_id=str(current_child_id),
            planning_intent="parallel_gpu_fill",
            operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
            prefetch_state_key="parallel",
            experiment_cursor=int(match.group(1)) + 1,
        )
        if handle is None:
            self.events.append(
                "controller_plan_parallel_prefetch_skipped",
                {
                    "parent_model_id": str(current_child_id),
                    "training_calls": next_training_calls,
                    "reason": "eager_parallel_prefetch_unavailable",
                },
            )
            return None
        job: Dict[str, Any] = {
            "parent_model_id": str(current_child_id),
            "training_calls": next_training_calls,
            "primary_handle": None,
            "parallel_handle": handle,
            "done": threading.Event(),
            "eager": True,
        }
        with self._early_parallel_prefetch_lock:
            existing = self._early_parallel_prefetch_jobs.get(key)
            if existing is not None:
                return existing
            self._early_parallel_prefetch_jobs[key] = job
        if isinstance(handle, dict) and handle not in self._deferred_prefetch_handles:
            self._deferred_prefetch_handles.append(handle)

        def mark_done() -> None:
            thread = handle.get("thread") if isinstance(handle, Mapping) else None
            if isinstance(thread, threading.Thread):
                thread.join()
            job["done"].set()

        threading.Thread(
            target=mark_done,
            name="harness4h3-eager-parallel-prefetch-%s" % current_child_id,
            daemon=True,
        ).start()
        self.events.append(
            "controller_plan_parallel_prefetch_armed",
            {
                "parent_model_id": str(current_child_id),
                "training_calls": next_training_calls,
                "source_experiment_id": current_plan.experiment_id,
                "reason": "cpu_only_worker_eager_parallel_prefetch",
            },
        )
        return job

    def _arm_evaluation_parallel_prefetch(
        self,
        candidate: ModelCandidate,
        training_calls: int,
        child_model_id: str,
    ) -> Optional[Mapping[str, Any]]:
        """Prepare a GPU branch before a single-task evaluation starts.

        The candidate's normal successor is often still waiting on the
        Controller when ComfyUI acquires its evaluator card. Starting the
        filtered GPU request at the same boundary as the normal prefetch
        gives the evaluator overlap path an independent, validated branch to
        consume without inventing a plan locally or waiting for the primary
        request to finish first.
        """

        if (
            not self.config.pipeline_enabled
            or int(self.config.pipeline_max_inflight) < 2
            or not self.config.worker.enabled
        ):
            return None
        training_call_cursor = int(training_calls) + 1
        state = self._load_campaign_state()
        try:
            if self._ready_parallel_prefetched_plan(
                state,
                candidate.id,
                training_call_cursor,
            ) is not None:
                return None
        except (TypeError, ValueError, KeyError):
            self._clear_parallel_prefetch_state(state)
            self._save_campaign_state(state)

        matching = [
            handle
            for handle in self._deferred_prefetch_handles
            if str(handle.get("source_child_model_id") or "") == str(child_model_id)
            and str(handle.get("prefetch_state_key") or "primary") == "parallel"
            and not bool(handle.get("prefetch_cancelled"))
            and int(handle.get("training_calls", -1)) == training_call_cursor
        ]
        if matching:
            return matching[-1]

        # The candidate was created by the preceding experiment. Reserve one
        # additional experiment-id slot so the concurrently generated primary
        # request and this GPU request cannot receive the same id.
        experiment_cursor: Optional[int] = None
        source_experiment_id = str(candidate.created_by_experiment_id or "")
        match = re.fullmatch(r"exp_(\d+)", source_experiment_id)
        if match is not None:
            experiment_cursor = int(match.group(1)) + 1
        handle = self._start_controller_prefetch(
            candidate,
            training_calls,
            source_experiment_id=source_experiment_id or None,
            source_child_model_id=str(child_model_id),
            planning_intent="parallel_gpu_fill",
            operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
            prefetch_state_key="parallel",
            experiment_cursor=experiment_cursor,
        )
        if handle is None:
            self.events.append(
                "controller_plan_parallel_prefetch_skipped",
                {
                    "candidate_model_id": candidate.id,
                    "training_calls": training_call_cursor,
                    "reason": "evaluation_parallel_prefetch_unavailable",
                },
            )
            return None
        if isinstance(handle, dict) and handle not in self._deferred_prefetch_handles:
            self._deferred_prefetch_handles.append(handle)
        self.events.append(
            "controller_plan_parallel_prefetch_armed",
            {
                "candidate_model_id": candidate.id,
                "source_child_model_id": str(child_model_id),
                "source_experiment_id": source_experiment_id or None,
                "training_calls": training_call_cursor,
                "reason": "evaluation_setup_before_benchmark_callback",
            },
        )
        return handle

    def _arm_parallel_prefetch_after_primary(
        self,
        parent: ModelCandidate,
        current_plan: ExperimentPlan,
        current_child_id: str,
        training_calls: int,
        primary_handle: Mapping[str, Any],
    ) -> Optional[threading.Thread]:
        """Prepare an independent GPU branch during the current worker.

        The primary successor is allowed to be a CPU-only structural
        operation. In that case the evaluation window would otherwise leave
        its non-ComfyUI cards idle while a second LLM request is generated.
        Wait for the primary plan, then ask the cloned Controller for a
        separate GPU operator rooted at the child that the current worker is
        already producing. The plan is persisted under the isolated
        ``parallel_prefetched_*`` cursor and is launched only by the next
        evaluation callback, so it never becomes an unvalidated active
        model by itself.
        """

        if (
            not self.config.pipeline_enabled
            or int(self.config.pipeline_max_inflight) < 2
            or not isinstance(primary_handle, Mapping)
        ):
            return None
        next_training_calls = int(training_calls) + 1
        key = (str(current_child_id), next_training_calls)
        job: Dict[str, Any] = {
            "parent_model_id": str(current_child_id),
            "training_calls": next_training_calls,
            "primary_handle": primary_handle,
            "parallel_handle": None,
            "done": threading.Event(),
        }
        with self._early_parallel_prefetch_lock:
            existing = self._early_parallel_prefetch_jobs.get(key)
            if existing is not None:
                thread = existing.get("thread")
                return thread if isinstance(thread, threading.Thread) else None
            self._early_parallel_prefetch_jobs[key] = job

        def run() -> None:
            try:
                if bool(primary_handle.get("prefetch_cancelled")) or self.review_replan_requested:
                    return
                primary_plan = self._finish_controller_prefetch(primary_handle)
                if primary_plan is None or str(primary_plan.operator) not in {"prune_blocks", "quantize"}:
                    return
                if bool(primary_handle.get("prefetch_cancelled")) or self.review_replan_requested:
                    return
                predicted_child = self._prefetch_parent_candidate(
                    parent,
                    current_plan,
                    str(current_child_id),
                )
                batched_parallel = self._parallel_candidate_from_prefetch_handle(
                    primary_handle,
                    primary_plan,
                    next_training_calls,
                )
                # The planning prompt asks the Controller to reason about the
                # deterministic child that the current worker is producing.
                # Some valid structured responses still echo the old parent
                # ID.  The normal speculative path already rebases that
                # identity at the evaluation boundary; apply the same
                # lineage-safe rebase here so a CPU primary plan can arm its
                # GPU-fill successor during training instead of waiting until
                # evaluation starts.  An unrelated ID remains a hard skip.
                if primary_plan.parent_model_id not in {parent.id, predicted_child.id}:
                    self.events.append(
                        "controller_plan_parallel_prefetch_skipped",
                        {
                            "parent_model_id": predicted_child.id,
                            "primary_experiment_id": primary_plan.experiment_id,
                            "reason": "primary_plan_parent_mismatch",
                        },
                    )
                    return
                if primary_plan.parent_model_id != predicted_child.id:
                    predicted_system = self._system_for_model(predicted_child.id)
                    primary_plan = replace(
                        primary_plan,
                        parent_model_id=predicted_child.id,
                        parent_system_id=(
                            predicted_system.id
                            if predicted_system is not None
                            else primary_plan.parent_system_id
                        ),
                    )
                    self.events.append(
                        "controller_plan_prefetch_rebased_for_parallel",
                        {
                            "experiment_id": primary_plan.experiment_id,
                            "old_parent_model_id": parent.id,
                            "parent_model_id": predicted_child.id,
                            "source_child_model_id": current_child_id,
                            "reason": "validated_primary_prefetch_echoed_old_parent",
                        },
                    )
                state = self._load_campaign_state()
                try:
                    existing_plan = self._parallel_prefetched_plan(state)
                except (TypeError, ValueError, KeyError) as exc:
                    existing_plan = None
                    self._clear_parallel_prefetch_state(state)
                    self._save_campaign_state(state)
                try:
                    existing_calls_match = int(state.get("parallel_prefetched_training_calls")) == next_training_calls
                except (TypeError, ValueError):
                    existing_calls_match = False
                if (
                    existing_plan is not None
                    and existing_plan.parent_model_id == predicted_child.id
                    and existing_calls_match
                ):
                    self.events.append(
                        "controller_plan_parallel_prefetch_already_ready",
                        {
                            "experiment_id": existing_plan.experiment_id,
                            "parent_model_id": predicted_child.id,
                            "training_calls": next_training_calls,
                            "source_experiment_id": primary_plan.experiment_id,
                        },
                    )
                    return
                if batched_parallel is not None:
                    predicted_system = self._system_for_model(predicted_child.id)
                    batched_parallel = replace(
                        batched_parallel,
                        parent_model_id=predicted_child.id,
                        parent_system_id=(
                            predicted_system.id
                            if predicted_system is not None
                            else batched_parallel.parent_system_id
                        ),
                    )
                    self._persist_batched_parallel_candidate(
                        primary_handle,
                        batched_parallel,
                        primary_plan.experiment_id,
                    )
                    self.events.append(
                        "controller_plan_parallel_prefetch_armed",
                        {
                            "experiment_id": batched_parallel.experiment_id,
                            "parent_model_id": predicted_child.id,
                            "primary_experiment_id": primary_plan.experiment_id,
                            "training_calls": next_training_calls,
                            "reason": "batched_controller_candidate",
                        },
                    )
                    return
                match = re.fullmatch(r"exp_(\d+)", str(primary_plan.experiment_id))
                if match is None:
                    self.events.append(
                        "controller_plan_parallel_prefetch_skipped",
                        {
                            "parent_model_id": predicted_child.id,
                            "primary_experiment_id": primary_plan.experiment_id,
                            "reason": "primary_experiment_id_not_numeric",
                        },
                    )
                    return
                handle = self._start_controller_prefetch(
                    predicted_child,
                    int(training_calls),
                    source_experiment_id=primary_plan.experiment_id,
                    planning_intent="parallel_gpu_fill",
                    operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
                    prefetch_state_key="parallel",
                    experiment_cursor=int(match.group(1)),
                )
                if handle is None:
                    return
                job["parallel_handle"] = handle
                if isinstance(handle, dict) and handle not in self._deferred_prefetch_handles:
                    self._deferred_prefetch_handles.append(handle)
                self.events.append(
                    "controller_plan_parallel_prefetch_armed",
                    {
                        "parent_model_id": predicted_child.id,
                        "primary_experiment_id": primary_plan.experiment_id,
                        "training_calls": next_training_calls,
                        "reason": "primary_cpu_plan_ready_during_training",
                    },
                )
            except (RemoteError, OSError, TypeError, ValueError) as exc:
                self.events.append(
                    "controller_plan_parallel_prefetch_skipped",
                    {
                        "parent_model_id": str(current_child_id),
                        "training_calls": next_training_calls,
                        "reason": "early_parallel_prefetch_failed",
                        "error": str(exc)[:1200],
                    },
                )
            finally:
                job["done"].set()
                with self._early_parallel_prefetch_lock:
                    if job.get("parallel_handle") is None:
                        self._early_parallel_prefetch_jobs.pop(key, None)

        thread = threading.Thread(
            target=run,
            name="harness4h3-early-parallel-prefetch-%s" % current_child_id,
            daemon=True,
        )
        job["thread"] = thread
        thread.start()
        return thread

    def _start_checkpoint_prefetch(
        self,
        checkpoint_path: str,
        *,
        reason: str,
        stage_dir_override: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> Optional[Mapping[str, Any]]:
        """Stage one parent checkpoint while ComfyUI evaluates the candidate.

        The helper is a fixed repository script and receives only paths from
        trusted campaign state. It shares an exclusive lock and one-entry
        cache with ``h3_real_train_worker.py``.
        """

        if not self.config.pipeline_enabled or not self.config.worker.enabled:
            return None
        if not isinstance(self.ssh, SSHClient):
            return None
        source = str(checkpoint_path or "").strip()
        if not source.startswith("/") or not source.endswith(".safetensors"):
            return None
        handle_key = str(cache_key or source)
        existing = self._checkpoint_prefetch_handles.get(handle_key)
        if existing is not None:
            return existing
        try:
            template = self.ssh.read_json(self.config.worker.config_template)
            stage_dir = template.get("checkpoint_stage_dir") if isinstance(template, Mapping) else None
            stage_dir = str(stage_dir_override or stage_dir or "").strip()
            if not stage_dir.startswith("/"):
                raise ValueError("worker checkpoint_stage_dir must be absolute")
            helper = str(Path(self.config.remote.harness_root) / "tools" / "h3_checkpoint_stage.py")
            python = str(self.config.remote.python)
            timeout_s = max(
                12.0 * 3600.0,
                float(template.get("trainer_timeout_s", 0.0) or 0.0),
            )
        except (RemoteError, OSError, TypeError, ValueError) as exc:
            self.events.append(
                "checkpoint_prefetch_skipped",
                {"checkpoint_path": source, "reason": "template_unavailable", "error": str(exc)[:1000]},
            )
            return None

        handle: Dict[str, Any] = {
            "checkpoint_path": source,
            "stage_dir": stage_dir,
            "helper": helper,
            "started_at": time.time(),
            "status": "running",
            "result": {},
            "thread": None,
            "deferred_reported": False,
        }
        self._checkpoint_prefetch_handles[handle_key] = handle

        def run() -> None:
            try:
                command_result = self.ssh.run(
                    (python, helper, "--source", source, "--stage-dir", stage_dir),
                    check=False,
                    timeout_s=timeout_s,
                )
                stdout_lines = [
                    line.strip()
                    for line in str(getattr(command_result, "stdout", "")).splitlines()
                    if line.strip()
                ]
                payload: Mapping[str, Any] = {}
                if stdout_lines:
                    try:
                        parsed = json.loads(stdout_lines[-1])
                        if isinstance(parsed, Mapping):
                            payload = parsed
                    except (TypeError, ValueError, json.JSONDecodeError):
                        payload = {}
                return_code = int(getattr(command_result, "returncode", 1))
                handle["result"] = dict(payload)
                handle["status"] = "ready" if return_code == 0 and payload.get("status") == "ready" else "failed"
                if handle["status"] != "ready":
                    handle["error"] = str(getattr(command_result, "stderr", "")).strip()[-2000:]
            except Exception as exc:
                handle["status"] = "failed"
                handle["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:1500])
            finally:
                handle["finished_at"] = time.time()
                self.events.append(
                    "checkpoint_prefetch_completed",
                    {
                        "checkpoint_path": source,
                        "stage_dir": stage_dir,
                        "status": handle.get("status"),
                        "result": dict(handle.get("result") or {}),
                        "error": handle.get("error"),
                        "elapsed_s": max(0.0, float(handle["finished_at"]) - float(handle["started_at"])),
                        "reason": reason,
                    },
                )

        thread = threading.Thread(
            target=run,
            name="harness4h3-checkpoint-prefetch-%s" % Path(source).stem,
            daemon=True,
        )
        handle["thread"] = thread
        thread.start()
        self.events.append(
            "checkpoint_prefetch_started",
            {
                "checkpoint_path": source,
                "stage_dir": stage_dir,
                "helper": helper,
                "reason": reason,
            },
        )
        return handle

    def _finish_checkpoint_prefetch(
        self,
        handle: Optional[Mapping[str, Any]],
        *,
        wait_s: Optional[float] = None,
    ) -> Optional[Mapping[str, Any]]:
        """Join checkpoint staging before allocating the training GPUs."""

        if not handle:
            return None
        thread = handle.get("thread")
        if not isinstance(thread, threading.Thread):
            return None
        timeout_s = 12.0 * 3600.0 if wait_s is None else max(0.0, float(wait_s))
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            if isinstance(handle, dict) and not bool(handle.get("deferred_reported")):
                handle["deferred_reported"] = True
                self.events.append(
                    "checkpoint_prefetch_deferred",
                    {
                        "checkpoint_path": handle.get("checkpoint_path"),
                        "stage_dir": handle.get("stage_dir"),
                        "reason": "staging_not_complete_before_training_boundary",
                    },
                )
            return None
        return handle

    def _persist_controller_prefetch(self, handle: Mapping[str, Any], plan: ExperimentPlan) -> bool:
        """Persist one validated overlap plan exactly once."""

        lock = handle.get("persist_lock")
        if lock is not None and not callable(getattr(lock, "__enter__", None)):
            # Accept manually constructed test handles without requiring a
            # concrete ``threading.Lock`` type (it is a factory on some
            # supported Python versions).
            lock = None

        def persist() -> bool:
            if bool(handle.get("prefetch_persisted")):
                return False
            if bool(handle.get("prefetch_cancelled")):
                return False
            state = self._load_campaign_state()
            state_key = str(handle.get("prefetch_state_key") or "primary")
            prefix = "parallel_prefetched" if state_key == "parallel" else "prefetched"
            state.update(
                {
                    prefix + "_plan": {"plan": plan.to_dict()},
                    prefix + "_parent_model_id": str(handle.get("parent_model_id")),
                    prefix + "_training_calls": int(handle.get("training_calls", 0)),
                    prefix + "_source_experiment_id": handle.get("source_experiment_id"),
                    prefix + "_source_child_model_id": handle.get("source_child_model_id"),
                    prefix + "_observation_ids": list(handle.get("observation_ids") or []),
                    prefix + "_created_at": time.time(),
                }
            )
            self._save_campaign_state(state)
            # A primary n-way response may already contain a legal distributed
            # GPU sibling while the selected successor is CPU-only. Persist
            # exactly one such sibling with the isolated parallel cursor so a
            # restart or evaluation boundary does not discard the useful
            # candidate and issue a second LLM request. The existing helper
            # performs the strict same-parent/operator/resource filtering.
            if state_key == "primary" and plan.operator in {"prune_blocks", "quantize"}:
                sibling = self._parallel_candidate_from_prefetch_handle(
                    handle,
                    plan,
                    int(handle.get("training_calls", 0)),
                )
                if sibling is not None:
                    self._persist_batched_parallel_candidate(
                        handle,
                        sibling,
                        plan.experiment_id,
                    )
                    self.events.append(
                        "controller_plan_parallel_prefetch_armed",
                        {
                            "experiment_id": sibling.experiment_id,
                            "parent_model_id": sibling.parent_model_id,
                            "primary_experiment_id": plan.experiment_id,
                            "training_calls": int(handle.get("training_calls", 0)),
                            "reason": "primary_prefetch_persisted_gpu_candidate",
                        },
                    )
            # Mapping is mutable by design: the boundary join can see that
            # the background thread already published this exact plan.
            handle["prefetch_persisted"] = True
            self.events.append(
                "controller_plan_prefetch_ready",
                {
                    **self._controller_provider_metadata(),
                    "experiment_id": plan.experiment_id,
                    "parent_model_id": plan.parent_model_id,
                    "training_calls": handle.get("training_calls"),
                    "source_experiment_id": handle.get("source_experiment_id"),
                    "source_child_model_id": handle.get("source_child_model_id"),
                    "planning_intent": handle.get("planning_intent", "primary"),
                    "prefetch_state_key": state_key,
                    "operator": plan.operator,
                    "resource_request": dict(plan.resource_request),
                    "elapsed_s": time.monotonic() - float(handle.get("started", time.monotonic())),
                },
            )
            return True

        if lock is None:
            return persist()
        with lock:
            return persist()

    def _persist_batched_parallel_candidate(
        self,
        handle: Optional[Mapping[str, Any]],
        plan: ExperimentPlan,
        source_experiment_id: str,
    ) -> None:
        """Persist an alternate candidate without consuming another observation.

        This is deliberately separate from ``_persist_controller_prefetch``:
        the primary handle may already have published its selected plan, while
        this alternate must occupy the isolated parallel cursor.
        """

        state = self._load_campaign_state()
        state.update(
            {
                "parallel_prefetched_plan": {"plan": plan.to_dict()},
                "parallel_prefetched_parent_model_id": plan.parent_model_id,
                "parallel_prefetched_training_calls": int(handle.get("training_calls", 0)) if isinstance(handle, Mapping) else 0,
                "parallel_prefetched_source_experiment_id": str(source_experiment_id),
                "parallel_prefetched_source_child_model_id": (
                    handle.get("source_child_model_id") if isinstance(handle, Mapping) else None
                ),
                "parallel_prefetched_observation_ids": (
                    list(handle.get("observation_ids") or []) if isinstance(handle, Mapping) else []
                ),
                "parallel_prefetched_created_at": time.time(),
            }
        )
        self._save_campaign_state(state)

    def _finish_controller_prefetch(
        self,
        handle: Optional[Mapping[str, Any]],
        *,
        wait_s: Optional[float] = None,
    ) -> Optional[ExperimentPlan]:
        """Join the overlap call; persistence is idempotent at the boundary."""

        if not handle:
            return None
        thread = handle.get("thread")
        if not isinstance(thread, threading.Thread):
            return None
        timeout_s = (
            max(30.0, float(getattr(self.controller, "timeout_s", 180.0) or 180.0) + 60.0)
            if wait_s is None
            else max(0.0, float(wait_s))
        )
        thread.join(timeout=timeout_s)
        result = handle.get("result")
        if thread.is_alive():
            if isinstance(handle, dict) and handle not in self._deferred_prefetch_handles:
                handle["prefetch_deferred"] = True
                self._deferred_prefetch_handles.append(handle)
            self.events.append(
                "controller_plan_prefetch_deferred",
                {
                    "parent_model_id": handle.get("parent_model_id"),
                    "planning_parent_model_id": handle.get("planning_parent_model_id"),
                    "training_calls": handle.get("training_calls"),
                    "reason": "worker_boundary_reached_before_controller",
                    "elapsed_s": time.monotonic() - float(handle.get("started", time.monotonic())),
                },
            )
            return None
        if not isinstance(result, Mapping):
            return None
        error = result.get("error")
        plan = result.get("plan")
        if not isinstance(plan, ExperimentPlan):
            self.events.append(
                "controller_plan_prefetch_unavailable",
                {
                    "parent_model_id": handle.get("parent_model_id"),
                    "training_calls": handle.get("training_calls"),
                    "reason": "controller_call_failed",
                    "error": str(error or "no validated plan")[:1500],
                    "elapsed_s": time.monotonic() - float(handle.get("started", time.monotonic())),
                },
            )
            return None
        try:
            self._persist_controller_prefetch(handle, plan)
        except Exception as exc:
            self.events.append(
                "controller_plan_prefetch_unavailable",
                {
                    "experiment_id": plan.experiment_id,
                    "reason": "persist_failed",
                    "error": str(exc)[:1000],
                },
            )
            return None
        return plan

    def _drain_deferred_prefetches(self, wait_s: float = 0.0) -> None:
        """Publish completed overlap calls without creating duplicate plans."""

        for handle in list(self._deferred_prefetch_handles):
            thread = handle.get("thread")
            if not isinstance(thread, threading.Thread):
                self._deferred_prefetch_handles.remove(handle)
                continue
            if thread.is_alive() and wait_s > 0:
                self._finish_controller_prefetch(handle, wait_s=wait_s)
            if thread.is_alive():
                continue
            self._finish_controller_prefetch(handle, wait_s=0.0)
            self._deferred_prefetch_handles.remove(handle)

    def _cancel_deferred_prefetches(self, reason: str) -> None:
        """Prevent a stale in-flight plan from being published after a replan."""

        for handle in list(self._deferred_prefetch_handles):
            if isinstance(handle, dict):
                handle["prefetch_cancelled"] = True
            self.events.append(
                "controller_plan_prefetch_discarded",
                {
                    "experiment_id": handle.get("source_experiment_id"),
                    "source_child_model_id": handle.get("source_child_model_id"),
                    "reason": reason,
                },
            )
        self._drain_deferred_prefetches(wait_s=0.0)

    def _controller_provider_metadata(self, controller: Optional[Any] = None) -> Dict[str, str]:
        provider = controller or self.controller
        return {
            "provider": str(getattr(provider, "provider_name", "unknown")),
            "model": str(getattr(provider, "model_name", "unknown")),
        }

    @staticmethod
    def _controller_port_candidates(provider: Any) -> Tuple[int, ...]:
        """Return the preferred and explicitly configured loopback ports."""

        ports: List[int] = []

        def add(value: Any) -> None:
            if isinstance(value, bool):
                return
            try:
                port = int(value)
            except (TypeError, ValueError):
                return
            if 1 <= port <= 65535 and port not in ports:
                ports.append(port)

        add(getattr(provider, "preferred_remote_port", None))
        add(getattr(provider, "remote_port", 8000))
        fallback = getattr(provider, "fallback_remote_ports", ())
        if isinstance(fallback, str):
            fallback = fallback.split(",")
        if isinstance(fallback, Sequence):
            for value in fallback:
                add(value)
        return tuple(ports) or (8000,)

    def _remote_controller_preflight(self, controller: Optional[Any] = None) -> None:
        """Verify a local vLLM endpoint, using only explicit port fallbacks."""

        provider = controller or self.controller
        configured_port = int(getattr(provider, "preferred_remote_port", getattr(provider, "remote_port", 8000)))
        model = str(getattr(provider, "model_name", "")).strip()
        metadata = self._controller_provider_metadata(provider)
        ports = self._controller_port_candidates(provider)
        failures: List[str] = []
        for port in ports:
            endpoint = "http://127.0.0.1:%d/v1/models" % port
            try:
                result = self.ssh.run(("curl", "-fsS", "--max-time", "5", endpoint), check=False)
            except Exception as exc:
                failures.append("remote Controller preflight failed at %s: %s" % (endpoint, exc))
                continue
            if int(getattr(result, "returncode", 1)) != 0:
                detail = str(getattr(result, "stderr", "")).strip()[-500:]
                failures.append("remote Controller is unavailable at %s%s" % (endpoint, (": " + detail) if detail else ""))
                continue
            try:
                payload = json.loads(str(getattr(result, "stdout", "")))
                available = [
                    str(item.get("id"))
                    for item in payload.get("data", [])
                    if isinstance(item, Mapping) and item.get("id")
                ]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                failures.append("remote Controller /v1/models at %s returned invalid JSON: %s" % (endpoint, exc))
                continue
            if model not in available:
                failures.append("remote Controller model %s is not served at %s; available=%s" % (model, endpoint, available))
                continue
            previous_port = int(getattr(provider, "remote_port", configured_port))
            provider.remote_port = port
            self.events.append(
                "controller_preflight",
                {
                    **metadata,
                    "status": "ready",
                    "endpoint": endpoint,
                    "available_models": available,
                    "configured_port": configured_port,
                    "selected_port": port,
                    "fallback_selected": port != configured_port,
                    "previous_port": previous_port,
                    "attempted_ports": list(ports),
                },
            )
            return
        if len(failures) == 1:
            raise ControllerUnavailableError(failures[0])
        raise ControllerUnavailableError(
            "remote Controller is unavailable on ports %s: %s"
            % (list(ports), " | ".join(failures)[-1800:])
        )

    def _controller_handoff_hold_path(self) -> str:
        return str(Path(self.config.remote.resolved_campaign_root) / ".controller-handoff-hold.json")

    def _write_controller_handoff_hold(self, reason: str) -> Optional[str]:
        """Prevent the watcher from relaunching vLLM during worker handoff."""

        writer = getattr(self.ssh, "write_json", None)
        if not isinstance(self.ssh, SSHClient) or not callable(writer):
            return None
        path = self._controller_handoff_hold_path()
        payload = {
            "state": "holding_for_worker_handoff",
            "owner_pid": os.getpid(),
            "created_at": time.time(),
            "reason": str(reason),
        }
        writer(path, payload)
        self.events.append("controller_handoff_hold_created", {**payload, "path": path})
        return path

    def _clear_controller_handoff_hold(self) -> None:
        remover = getattr(self.ssh, "remove_file", None)
        if not isinstance(self.ssh, SSHClient) or not callable(remover):
            return
        path = self._controller_handoff_hold_path()
        try:
            remover(path)
        except Exception as exc:
            self.events.append(
                "controller_handoff_hold_cleanup_failed",
                {"path": path, "error": str(exc)[:500]},
            )

    def _request_controller_release(
        self,
        reason: str,
        wait_s: float = 30.0,
        handoff_hold: bool = False,
        full_card_training: bool = False,
    ) -> Mapping[str, Any]:
        """Ask the campaign-owned vLLM launcher to yield its GPU lease.

        The launcher owns the process tree and is the only component allowed
        to stop it.  A marker avoids broad ``pkill`` behavior on a shared SSH
        host; the launcher consumes this exact marker and then its existing
        worker-lease guard prevents an immediate restart while training owns
        the cards.
        """

        # Test transports and offline campaign doubles do not own the remote
        # controller launcher.  Only the real SSH/local command transport may
        # publish a release marker that another process will consume.
        writer = getattr(self.ssh, "write_json", None)
        if not isinstance(self.ssh, SSHClient) or not callable(writer):
            return {"status": "not_configured", "reason": "transport_has_no_remote_json_writer"}
        marker = str(Path(self.config.remote.resolved_campaign_root) / ".controller-release.json")
        pid_file = str(Path(self.config.remote.resolved_campaign_root) / ".controller-vllm.pid")
        remover = getattr(self.ssh, "remove_file", None)
        hold_path: Optional[str] = None
        payload = {
            "reason": str(reason),
            "owner_pid": os.getpid(),
            "created_at": time.time(),
            "full_card_training": bool(full_card_training),
        }
        if handoff_hold:
            try:
                hold_path = self._write_controller_handoff_hold(reason)
            except Exception as exc:
                result = {"status": "handoff_hold_failed", "reason": str(exc)[:1000]}
                self.events.append("controller_release_requested", {**payload, **result})
                return result
        try:
            writer(marker, payload)
        except Exception as exc:
            if hold_path:
                self._clear_controller_handoff_hold()
            result = {"status": "request_failed", "reason": str(exc)[:1000]}
            self.events.append("controller_release_requested", {**payload, **result})
            return result
        self.events.append("controller_release_requested", {**payload, "marker": marker})
        deadline = time.monotonic() + max(0.0, float(wait_s))
        while True:
            try:
                pid_exists = self.ssh.run(("test", "-e", pid_file), check=False)
                if int(getattr(pid_exists, "returncode", 1)) != 0:
                    owned_pid = None
                else:
                    raw_pid = self.ssh.run(("cat", "--", pid_file), check=False)
                    owned_pid = int(str(getattr(raw_pid, "stdout", "")).strip())
                running = False
                if owned_pid is not None and owned_pid > 0:
                    probe = self.ssh.run(("ps", "-p", str(owned_pid), "-o", "args="), check=False)
                    command = str(getattr(probe, "stdout", "")).strip()
                    running = int(getattr(probe, "returncode", 1)) == 0 and "vllm serve" in command
            except Exception as exc:
                result = {"status": "probe_failed", "reason": str(exc)[:1000], "marker": marker}
                self.events.append("controller_release_timeout", result)
                return result
            if not running:
                if callable(remover):
                    try:
                        remover(marker)
                    except Exception as exc:
                        result = {
                            "status": "cleanup_failed",
                            "marker": marker,
                            "reason": str(exc)[:1000],
                        }
                        self.events.append("controller_release_timeout", result)
                        return result
                result = {"status": "released", "marker": marker, "reason": str(reason)}
                if hold_path:
                    result["handoff_hold_file"] = hold_path
                self.events.append("controller_released", result)
                return result
            if time.monotonic() >= deadline:
                result = {"status": "timeout", "marker": marker, "reason": str(reason)}
                if hold_path:
                    result["handoff_hold_file"] = hold_path
                self.events.append("controller_release_timeout", result)
                return result
            time.sleep(0.5)

    def _invoke_controller(self, context: ControllerContext, controller: Optional[Any] = None) -> Any:
        """Invoke the configured Controller, using SSH forwarding for vLLM."""

        provider = controller or self.controller
        if str(getattr(provider, "provider_name", "")) != "vllm":
            return provider.plan(context)
        self._remote_controller_preflight(provider)
        port = int(getattr(provider, "remote_port", 8000))
        try:
            with RemotePortForward(self.ssh, port) as tunnel:
                previous_endpoint = str(getattr(provider, "endpoint", ""))
                provider.endpoint = tunnel.base_url + "/v1/chat/completions"
                try:
                    if not bool(getattr(provider, "probed", False)):
                        # The remote /v1/models preflight already proves that
                        # the selected vLLM service is reachable.  Do not
                        # spend another long completion on the full
                        # ExperimentPlan schema: some vLLM structured-output
                        # paths truncate the diagnostic probe even when the
                        # real plan endpoint is healthy.  The actual plan
                        # remains strictly parsed and validated below.
                        self.events.append(
                            "controller_probe_skipped",
                            {
                                **self._controller_provider_metadata(provider),
                                "status": "skipped_remote_api_preflight",
                                "reason": "remote_api_preflight_is_the_connectivity_gate",
                                "input_observation_ids": list(context.unconsumed_observation_ids),
                            },
                        )
                        provider.probed = True
                    return provider.plan(context)
                finally:
                    provider.endpoint = previous_endpoint
        except (RemoteError, OSError) as exc:
            raise ControllerUnavailableError("remote Controller SSH tunnel failed: %s" % exc)

    def _train_one(self, parent: ModelCandidate, training_calls: int) -> Optional[ExperienceRecord]:
        if not self.config.worker.enabled:
            return None
        self._drain_deferred_prefetches()
        if self.review_replan_requested:
            self._cancel_deferred_prefetches("review_requested_replan")
        state = self._load_campaign_state()
        active_round_policy = self._active_round_policy()
        if active_round_policy is not None:
            policy_progress = self._round_policy_progress(active_round_policy, state) or {}
            policy_stop = self._round_policy_budget_status(active_round_policy, policy_progress)
            if policy_stop is not None:
                if not policy_progress.get("stop_reason"):
                    policy_progress = self._mark_round_policy_stop(
                        active_round_policy,
                        state,
                        policy_stop,
                    )
                state["round_policy_progress"] = policy_progress
                self._clear_pending_state(state)
                self._clear_parallel_prefetch_state(state)
                self._save_campaign_state(state)
                self.events.append(
                    "round_policy_execution_blocked",
                    {
                        "round_id": active_round_policy.round_id,
                        "reason": str(policy_stop),
                        "progress": dict(policy_progress),
                    },
                )
                return None
        resource_recovery = state.get("resource_replan_intent")
        if isinstance(resource_recovery, Mapping):
            source_experiment_id = str(resource_recovery.get("source_experiment_id") or "")
            source_completed = any(
                str(record.experiment_id) == source_experiment_id
                and str(record.status) not in {"failed", "error", "cancelled"}
                for record in self.experience.read()
            )
            if source_experiment_id and source_completed:
                # A worker can finish between the plan-selection save and the
                # next boundary. Clear a stale recovery request before it is
                # allowed to steer an unrelated successor plan.
                state.pop("resource_replan_intent", None)
                self._save_campaign_state(state)
                self.events.append(
                    "controller_resource_recovery_consumed",
                    {
                        "source_experiment_id": source_experiment_id,
                        "reason": "source_worker_already_recorded",
                    },
                )
        plan = self._pending_plan(state)
        # If a short worker reached its boundary before the overlap request
        # finished, wait for that same request here rather than issuing a
        # duplicate LLM call.  In the normal GPU-training case this list is
        # already empty because the long worker hid the request latency.
        if plan is None and not self.review_replan_requested:
            matching = [
                handle
                for handle in self._deferred_prefetch_handles
                if str(handle.get("source_child_model_id") or "") == parent.id
            ]
            for handle in matching:
                self._finish_controller_prefetch(handle)
            self._drain_deferred_prefetches()
            state = self._load_campaign_state()
            plan = self._pending_plan(state)
        prefetched = False
        if plan is None:
            # A reviewer can invalidate a speculative plan while the previous
            # worker is running. Discard it before considering reuse so an
            # explicit LLM replan always wins.
            if self.review_replan_requested:
                self._clear_pending_state(state)
                self._save_campaign_state(state)
                self.events.append(
                    "controller_plan_prefetch_discarded",
                    {"reason": "review_requested_replan"},
                )
            else:
                try:
                    candidate_prefetch = self._prefetched_plan(state)
                except (TypeError, ValueError, KeyError) as exc:
                    candidate_prefetch = None
                    self._clear_pending_state(state)
                    self._save_campaign_state(state)
                    self.events.append(
                        "controller_plan_prefetch_discarded",
                        {"reason": "invalid_persisted_plan", "error": str(exc)[:1000]},
                    )
                expected_calls = state.get("prefetched_training_calls")
                try:
                    expected_calls_match = expected_calls is not None and int(expected_calls) == int(training_calls)
                except (TypeError, ValueError):
                    expected_calls_match = False
                human_directive_ids = {
                    str(item.observation_id)
                    for item in self.observations.read()
                    if item.kind == "human_directive"
                }
                unconsumed_human_directives = sorted(
                    human_directive_ids
                    - set(str(item) for item in state.get("consumed_observation_ids", []))
                )
                prefetched_parent_id = str(state.get("prefetched_parent_model_id") or "")
                prefetched_source_child_id = str(state.get("prefetched_source_child_model_id") or "")
                prefetched_parent_matches = (
                    candidate_prefetch is not None
                    and candidate_prefetch.parent_model_id == parent.id
                )
                # The previous worker may have become the new active Pareto
                # child during evaluation. The speculative plan was generated
                # against the old active parent, but it is still a safe
                # continuation candidate when the reviewer did not request a
                # replan and the active child is exactly the worker child that
                # produced the prefetch. Rebase only the lineage identity;
                # reviewer-requested replans and unrelated lineage changes
                # still discard the speculative plan.
                prefetched_rebased = False
                if (
                    candidate_prefetch is not None
                    and not prefetched_parent_matches
                    and parent.id == prefetched_source_child_id
                    and parent.parent_id == prefetched_parent_id
                    and prefetched_source_child_id
                ):
                    current_system = self._system_for_model(parent.id)
                    candidate_prefetch = replace(
                        candidate_prefetch,
                        parent_model_id=parent.id,
                        parent_system_id=(
                            current_system.id
                            if current_system is not None
                            else candidate_prefetch.parent_system_id
                        ),
                    )
                    prefetched_rebased = True
                elif candidate_prefetch is not None and prefetched_parent_matches:
                    # A prefetch planned against a predicted child has the
                    # right model lineage already, but its synthetic context
                    # may not have had the durable SystemCandidate yet.
                    # Refresh only that runtime identity before execution.
                    current_system = self._system_for_model(parent.id)
                    if current_system is not None and candidate_prefetch.parent_system_id != current_system.id:
                        candidate_prefetch = replace(
                            candidate_prefetch,
                            parent_system_id=current_system.id,
                        )
                if (
                    candidate_prefetch is not None
                    and (prefetched_parent_matches or prefetched_rebased)
                    and expected_calls_match
                    and not unconsumed_human_directives
                ):
                    plan = candidate_prefetch
                    prefetched = True
                    current_system = self._system_for_model(parent.id)
                    prefetched_fingerprint = experiment_fingerprint(
                        parent.id,
                        current_system.id if current_system is not None else str(plan.parent_system_id or "S0000"),
                        plan.operator,
                        plan.operator_args,
                        self.target.id,
                        current_system.device_id if current_system is not None else self.target.hardware_name,
                        {
                            "split": "heldout",
                            "workflow_template": str(self.config.workflow.template),
                            "quality_scope": self.config.quality_scope,
                        },
                    )
                    self.controller_trace.append(
                        {
                            "controller_call": False,
                            "prefetch_reuse": True,
                            "parent_model_id": parent.id,
                            "training_calls": training_calls,
                            "goal": self._goal_payload(),
                            "input_metrics": dict(parent.state.measured_metrics),
                            "status": "validated",
                            "prefetch_rebased": prefetched_rebased,
                            "fingerprint": prefetched_fingerprint,
                            "plan": plan.to_dict(),
                        }
                    )
                    self.events.append(
                        "controller_plan_prefetch_reused",
                        {
                            "experiment_id": plan.experiment_id,
                            "parent_model_id": plan.parent_model_id,
                            "training_calls": training_calls,
                            "reason": (
                                "worker_child_promoted_without_replan_request"
                                if prefetched_rebased
                                else "worker_completed_without_replan_request"
                            ),
                            "source_experiment_id": state.get("prefetched_source_experiment_id"),
                            "source_child_model_id": state.get("prefetched_source_child_model_id"),
                        },
                    )
                elif candidate_prefetch is not None:
                    self._clear_pending_state(state)
                    self._save_campaign_state(state)
                    reason = "parent_or_training_call_changed"
                    if unconsumed_human_directives:
                        reason = "new_human_directive"
                    self.events.append(
                        "controller_plan_prefetch_discarded",
                        {
                        "experiment_id": candidate_prefetch.experiment_id,
                        "parent_model_id": candidate_prefetch.parent_model_id,
                        "source_experiment_id": state.get("prefetched_source_experiment_id"),
                        "source_child_model_id": state.get("prefetched_source_child_model_id"),
                        "reason": reason,
                            "unconsumed_human_directive_ids": unconsumed_human_directives,
                        },
                    )
        if plan is not None:
            if plan.parent_model_id != parent.id:
                parent = self.models.get(plan.parent_model_id)
            pending_context = self._controller_context(parent, training_calls)
            new_observations = sorted(
                set(pending_context.unconsumed_observation_ids) - set(plan.consumed_observation_ids)
            )
            if new_observations and not prefetched:
                self._clear_pending_state(state)
                self._save_campaign_state(state)
                self.events.append(
                    "resource_pending_plan_invalidated",
                    {
                        "experiment_id": plan.experiment_id,
                        "reason": "new_observations_require_controller_consumption",
                        "new_observation_ids": new_observations,
                    },
                )
                plan = self._controller_plan(parent, training_calls)
            elif not prefetched:
                pending_trace = {
                    "controller_call": False,
                    "pending_retry": True,
                    "parent_model_id": parent.id,
                    "training_calls": training_calls,
                    "goal": self._goal_payload(),
                    "input_metrics": dict(parent.state.measured_metrics),
                    "status": "validated",
                    "plan": plan.to_dict(),
                }
                self.controller_trace.append(pending_trace)
                self.events.append(
                    "resource_retry_started",
                    {
                        "experiment_id": plan.experiment_id,
                        "operator": plan.operator,
                        "attempt": int(state.get("pending_attempts", 0)) + 1,
                        "resource_request": dict(plan.resource_request),
                    },
                )
        else:
            resource_recovery = state.get("resource_replan_intent")
            recovery_operators = []
            if "prune_blocks" in self.config.worker.allowed_operators:
                recovery_operators.append("prune_blocks")
            parent_bits = None
            if isinstance(parent.state.quantization, Mapping):
                raw_parent_bits = parent.state.quantization.get("bits")
                if isinstance(raw_parent_bits, int) and not isinstance(raw_parent_bits, bool):
                    parent_bits = int(raw_parent_bits)
            configured_bits = tuple(int(bits) for bits in self.config.worker.quantized_bits)
            if (
                "quantize" in self.config.worker.allowed_operators
                and configured_bits
                and (parent_bits is None or any(bits < parent_bits for bits in configured_bits))
            ):
                # Do not ask the LLM to select an already-applied precision
                # variant. That is a valid safety rejection in a normal
                # plan, but it would make a resource-recovery loop retry the
                # same impossible CPU branch forever.
                recovery_operators.append("quantize")
            recovery_operators = tuple(recovery_operators)
            if (
                isinstance(resource_recovery, Mapping)
                and not self.review_replan_requested
                and recovery_operators
            ):
                # A bounded GPU wait is a scheduling decision, not a reason
                # to keep asking the remote model for the same distributed
                # worker. Persisted intent makes this survive a detached
                # service restart; the operator filter remains advisory to
                # the Controller but is enforced again by _controller_plan.
                self.events.append(
                    "controller_resource_recovery_requested",
                    {
                        "parent_model_id": parent.id,
                        "training_calls": int(training_calls),
                        "planning_intent": "resource_recovery_cpu",
                        "operator_filter": list(recovery_operators),
                        "source_experiment_id": resource_recovery.get("source_experiment_id"),
                        "reason": resource_recovery.get("reason"),
                    },
                )
                plan = self._controller_plan(
                    parent,
                    training_calls,
                    planning_intent="resource_recovery_cpu",
                    operator_filter=recovery_operators,
                )
                if plan is not None:
                    # Keep the in-memory snapshot in sync with the durable
                    # copy. _train_one writes this snapshot again at the
                    # worker boundary; leaving the marker here would revive
                    # an already-consumed CPU-recovery intent on the next
                    # iteration.
                    state.pop("resource_replan_intent", None)
                    latest_state = self._load_campaign_state()
                    latest_state.pop("resource_replan_intent", None)
                    self._save_campaign_state(latest_state)
                    self.events.append(
                        "controller_resource_recovery_selected",
                        {
                            "experiment_id": plan.experiment_id,
                            "operator": plan.operator,
                            "resource_request": dict(plan.resource_request),
                            "reason": "bounded_gpu_wait_replanned_to_cpu_operator",
                        },
                    )
            else:
                plan = self._controller_plan(parent, training_calls)
        if plan is None:
            return None
        primary_batch_parallel_plan: Optional[ExperimentPlan] = None
        if (
            not prefetched
            and self.config.pipeline_enabled
            and int(self.config.pipeline_max_inflight) >= 2
        ):
            primary_batch_parallel_plan = self._consume_primary_batch_parallel_candidate(
                plan,
                training_calls,
            )
        try:
            operator_args = self._worker_operator_args(plan.operator, plan.operator_args, parent)
            self._validate_h3_runtime_recipe(plan.operator, operator_args, parent)
        except ValueError as exc:
            if self.controller_trace:
                self.controller_trace[-1]["execution_status"] = "invalid_worker_arguments"
                self.controller_trace[-1]["status"] = "rejected"
                self.controller_trace[-1]["error"] = str(exc)
            self._clear_pending_state(state)
            self._save_campaign_state(state)
            self.events.append(
                "controller_plan_rejected",
                {
                    "experiment_id": plan.experiment_id,
                    "model_id": parent.id,
                    "error": str(exc),
                    "reason": "trusted_worker_contract",
                },
            )
            return None
        checkpoint_prefetch_handle = self._checkpoint_prefetch_handles.get(str(parent.checkpoint_path))
        if checkpoint_prefetch_handle is not None:
            # Do not acquire or release GPU leases while a multi-gigabyte
            # staging copy is still running. If it is not ready, the worker
            # still has the same lock-protected fallback path, but it will be
            # launched only after this CPU/I/O boundary has been observed.
            self._finish_checkpoint_prefetch(checkpoint_prefetch_handle)
        if plan.operator == "distill":
            # Distillation needs the frozen source teacher as well as the
            # student parent.  Stage it on CPU/I/O before taking the worker
            # lease; otherwise rank 0 performs a multi-GB NFS copy after the
            # other ranks have started, making three GPUs spin at a barrier
            # while no useful training work is possible.
            try:
                template = self.ssh.read_json(self.config.worker.config_template)
                teacher_source = str(
                    template.get("source_parent_checkpoint")
                    if isinstance(template, Mapping)
                    else ""
                ).strip()
                base_stage_dir = str(
                    template.get("checkpoint_stage_dir")
                    if isinstance(template, Mapping)
                    else ""
                ).strip()
                if teacher_source and base_stage_dir:
                    teacher_handle = self._start_checkpoint_prefetch(
                        teacher_source,
                        reason="teacher_for_%s" % plan.experiment_id,
                        stage_dir_override=str(Path(base_stage_dir) / "teacher"),
                        cache_key="teacher:%s" % teacher_source,
                    )
                    if teacher_handle is not None:
                        self._finish_checkpoint_prefetch(teacher_handle)
            except (RemoteError, OSError, TypeError, ValueError) as exc:
                self.events.append(
                    "checkpoint_prefetch_skipped",
                    {
                        "reason": "teacher_template_unavailable",
                        "experiment_id": plan.experiment_id,
                        "error": str(exc)[:1000],
                    },
                )
        already_released = (
            self.last_comfyui_lease is not None
            and self.last_comfyui_lease.success
            and 0
            not in getattr(
                self.scheduler,
                "reserved_gpu_indices",
                self.comfyui_lease.scheduler.reserved_gpu_indices,
            )
        )
        needs_comfyui_release = (
            self.comfyui_lease.lease_active
            or self.comfyui_lease.gpu_index in getattr(
                self.scheduler,
                "reserved_gpu_indices",
                self.comfyui_lease.scheduler.reserved_gpu_indices,
            )
        )
        if (
            self.config.comfyui_cache_policy in {"idle_release", "cold_cache"}
            and not already_released
            and needs_comfyui_release
        ):
            self._release_comfyui_if_idle("training_boundary")
        # If the next worker may consume all four cards, finish its successor
        # plan while the Controller still has a card group.  For smaller or
        # elastic workers, yielding vLLM first lets the scheduler take the
        # largest safe group and leaves any remainder for the overlap call.
        prefetch_handle: Optional[Mapping[str, Any]] = None
        prefetch_started_before_training = False
        successor_plan: Optional[ExperimentPlan] = None
        planned_resource_request, request_for_pipeline = self._effective_worker_request(plan)
        distributed_training = bool(request_for_pipeline.get("distributed")) and int(
            request_for_pipeline.get("min_gpu_count", 0)
        ) >= 2
        total_gpu_count = int(getattr(self.scheduler, "gpu_count", 4))
        full_gpu_training = distributed_training and int(
            request_for_pipeline.get("max_gpu_count", 0)
        ) >= total_gpu_count
        if (
            self.config.pipeline_enabled
            and self.config.prefetch_before_full_training
            and full_gpu_training
        ):
            next_child_id = self.models.next_id()
            self._set_pipeline_stage(
                PipelineStage.PREFETCHING,
                training_model_id=next_child_id,
                overlap_with_model_id=parent.id,
            )
            prefetch_handle = self._start_controller_prefetch(
                parent,
                training_calls,
                source_plan=plan,
                source_experiment_id=plan.experiment_id,
                source_child_model_id=next_child_id,
            )
            if prefetch_handle is not None:
                # A four-card worker leaves no legal Controller placement.
                # Join here so the next plan is durable before the lease is
                # taken; shorter workers continue to overlap asynchronously.
                successor_plan = self._finish_controller_prefetch(prefetch_handle)
            prefetch_started_before_training = successor_plan is not None
            if successor_plan is None:
                # Full-card training has no safe overlap lane.  Do not start
                # it without a validated successor: the next boundary would
                # otherwise have to stop four ranks just to ask the LLM what
                # to do next, defeating the no-idle handoff contract.
                blocked_decision = {
                    "status": "successor_plan_unavailable",
                    "reason": "full_card_requires_prefetched_successor",
                }
                self._persist_pending_plan(state, plan, training_calls, blocked_decision)
                self._set_pipeline_waiting(
                    "controller",
                    current_model_id=parent.id,
                    experiment_id=plan.experiment_id,
                )
                self.events.append(
                    "full_card_training_blocked",
                    {
                        "experiment_id": plan.experiment_id,
                        "operator": plan.operator,
                        "planned_resource_request": dict(planned_resource_request),
                        "effective_resource_request": dict(request_for_pipeline),
                        "reason": blocked_decision["reason"],
                    },
                )
                return None
        # Keep a small Controller lane alive when it cannot prevent the
        # worker's minimum allocation.  Releasing a TP1 Controller on GPU3
        # while GPUs0-2 are already busy only creates an avoidable model-load
        # gap; release is needed only when the live Controller reservation
        # itself consumes too many cards for the worker minimum.
        controller_reserved: Optional[Tuple[int, ...]] = None
        try:
            controller_indices = getattr(self.scheduler, "controller_reserved_gpu_indices", None)
        except (OSError, RemoteError, TypeError, ValueError):
            controller_indices = None
        if controller_indices is not None:
            try:
                controller_reserved = tuple(int(index) for index in controller_indices)
            except (TypeError, ValueError):
                controller_reserved = None
        can_keep_controller_lane = bool(
            self.config.pipeline_enabled
            and distributed_training
            and not full_gpu_training
            and int(self.config.controller_overlap_gpus) > 0
            and not bool(request_for_pipeline.get("exclusive"))
            and controller_reserved is not None
            and len(controller_reserved)
            <= total_gpu_count - int(request_for_pipeline.get("min_gpu_count", 0))
        )
        handoff_hold = bool(
            self.config.pipeline_enabled
            and distributed_training
            and int(self.config.controller_overlap_gpus) > 0
            and not can_keep_controller_lane
        )
        if self.config.pipeline_enabled and distributed_training and not can_keep_controller_lane:
            release_result = self._request_controller_release(
                "training_gpu_allocation",
                handoff_hold=handoff_hold,
                full_card_training=full_gpu_training,
            )
            release_status = str(release_result.get("status", "")).strip().lower()
            if release_status not in {"released", "not_configured"}:
                # Never allocate a worker behind an unconfirmed Controller
                # release.  A stale vLLM process can otherwise race the
                # worker during weight loading and produce a misleading
                # low-utilization failure.  Keep the validated plan durable
                # so the next loop retries the same plan after the launcher
                # has repaired its process tree.
                if self.controller_trace:
                    self.controller_trace[-1]["execution_status"] = "controller_release_blocked"
                blocked_decision = {
                    "status": "controller_release_blocked",
                    "reason": release_status or "unknown",
                    **dict(release_result),
                }
                self._persist_pending_plan(state, plan, training_calls, blocked_decision)
                self._cancel_deferred_prefetches("controller_release_blocked")
                if handoff_hold:
                    self._clear_controller_handoff_hold()
                self.events.append(
                    "controller_release_blocked",
                    {
                        "experiment_id": plan.experiment_id,
                        "operator": plan.operator,
                        "planned_resource_request": dict(planned_resource_request),
                        "effective_resource_request": dict(request_for_pipeline),
                        "full_card_training": bool(full_gpu_training),
                        "release": dict(release_result),
                    },
                )
                return None
        elif can_keep_controller_lane:
            self.events.append(
                "controller_release_skipped",
                {
                    "experiment_id": plan.experiment_id,
                    "operator": plan.operator,
                    "reason": "live_controller_lane_fits_worker_minimum",
                    "controller_reserved_gpus": list(controller_reserved or ()),
                    "effective_resource_request": dict(request_for_pipeline),
                    "full_card_training": False,
                },
            )
        resource_decision = self._acquire_worker_gpu_lease(
            plan.experiment_id,
            request_for_pipeline,
        )
        if handoff_hold and resource_decision.status == "ready":
            # The worker lease is published inside acquire(), so the watcher
            # can safely resume Controller placement after this point.
            self._clear_controller_handoff_hold()
        self.last_resource_decision = resource_decision.to_dict()
        # Keep the status file truthful even when a bounded wait is later
        # replanned and the pending plan is cleared.  Otherwise a historical
        # ready allocation can remain visible while the live scheduler is
        # correctly refusing to claim a foreign card.
        state["last_resource_decision"] = dict(self.last_resource_decision)
        self._save_campaign_state(state)
        if self.controller_trace:
            self.controller_trace[-1]["resource_decision"] = resource_decision.to_dict()
        self.events.append(
            "resource_scheduled",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "planned_resource_request": dict(planned_resource_request),
                "effective_resource_request": dict(request_for_pipeline),
                "full_card_training": bool(full_gpu_training),
                "resource_decision": resource_decision.to_dict(),
            },
        )
        if resource_decision.status != "ready":
            # A waiting decision is useful Controller evidence: the next
            # cycle may need to keep the plan or replan.  Once resources are
            # ready, however, the plan has already passed structured LLM
            # validation and the worker should launch immediately.  A second
            # Do not spend a Controller request every polling minute when the
            # same foreign job still owns the same cards.  Re-review on the
            # first wait, after a resource-state change, and at the bounded
            # replan threshold; the validated plan remains durable in between.
            decision_payload = resource_decision.to_dict()
            review_attempt = int(state.get("pending_attempts", 0)) + 1
            review_signature = hashlib.sha256(
                json.dumps(
                    {
                        "experiment_id": plan.experiment_id,
                        "status": decision_payload.get("status"),
                        "reason": decision_payload.get("reason"),
                        "available_gpu_count": decision_payload.get("available_gpu_count"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            replan_after = int(getattr(self.config, "resource_wait_replan_after", 12))
            should_review = (
                review_signature
                != str(state.get("pending_last_resource_review_signature") or "")
                or review_attempt >= replan_after
            )
            if should_review:
                self._resource_wait_review_signature = review_signature
                # Persist before the remote call so a supervisor restart
                # cannot repeat the same expensive review for an unchanged
                # resource state.
                state["pending_last_resource_review_signature"] = review_signature
                self._save_campaign_state(state)
                self._review_now(
                    "waiting_for_resource",
                    "resource_scheduled",
                    {
                        "experiment_id": plan.experiment_id,
                        "operator": plan.operator,
                        "resource_request": dict(request_for_pipeline),
                        "planned_resource_request": dict(planned_resource_request),
                        "effective_resource_request": dict(request_for_pipeline),
                        "resource_decision": decision_payload,
                        "wait_attempt": review_attempt,
                    },
                )
            else:
                self.events.append(
                    "controller_review_skipped",
                    {
                        "phase": "waiting_for_resource",
                        "trigger": "resource_scheduled",
                        "experiment_id": plan.experiment_id,
                        "reason": "unchanged_resource_wait",
                        "wait_attempt": review_attempt,
                        "next_review_at_attempt": replan_after,
                        "resource_decision": decision_payload,
                    },
                )
        else:
            self._resource_wait_review_signature = None
            self.events.append(
                "controller_review_skipped",
                {
                    "phase": "planning",
                    "trigger": "resource_scheduled",
                    "experiment_id": plan.experiment_id,
                    "reason": "validated_plan_launches_immediately",
                    "resource_decision": resource_decision.to_dict(),
                },
            )
        if self.review_stop_requested and resource_decision.status == "ready":
            self._release_worker_gpu_lease(plan.experiment_id, "controller_review_stop")
            self.events.append(
                "worker_not_started",
                {"experiment_id": plan.experiment_id, "reason": "controller_review_stop"},
            )
            return None
        if resource_decision.status != "ready":
            if self.controller_trace:
                self.controller_trace[-1]["execution_status"] = (
                    "resource_unavailable_replan"
                    if plan.resource_request.get("on_unavailable") == "replan"
                    else "waiting_for_resources"
                )
            if plan.resource_request.get("on_unavailable") == "replan":
                if handoff_hold:
                    # There will be no retry of this exact worker while the
                    # plan is being replanned; do not strand the Controller
                    # watcher behind a handoff hold that no longer protects a
                    # worker allocation.
                    self._clear_controller_handoff_hold()
                self._clear_pending_state(state)
                state["resource_replan_intent"] = {
                    "source_experiment_id": plan.experiment_id,
                    "source_operator": plan.operator,
                    "reason": str(resource_decision.reason),
                    "requested_at": time.time(),
                    "resource_decision": resource_decision.to_dict(),
                }
                self._save_campaign_state(state)
                self.events.append(
                    "resource_replan_requested",
                    {
                        "experiment_id": plan.experiment_id,
                        "operator": plan.operator,
                        "reason": resource_decision.reason,
                        "resource_decision": resource_decision.to_dict(),
                        "next_planning_intent": "resource_recovery_cpu",
                    },
                )
            else:
                if handoff_hold:
                    self.events.append(
                        "controller_handoff_hold_kept",
                        {
                            "experiment_id": plan.experiment_id,
                            "reason": "worker_resource_wait",
                            "resource_decision": resource_decision.to_dict(),
                        },
                    )
                if prefetched:
                    self._clear_pending_state(state)
                wait_attempts = self._persist_pending_plan(
                    state,
                    plan,
                    training_calls,
                    resource_decision.to_dict(),
                )
                replan_after = int(getattr(self.config, "resource_wait_replan_after", 12))
                if wait_attempts >= replan_after:
                    # A safe wait must not become an infinite deadlock when
                    # the host is shared with an unrelated long-running job.
                    # Clear only the pending decision and hand the next choice
                    # back to the remote Controller; no operator or foreign
                    # process is changed here.
                    if handoff_hold:
                        self._clear_controller_handoff_hold()
                    self._clear_pending_state(state)
                    state["resource_replan_intent"] = {
                        "source_experiment_id": plan.experiment_id,
                        "source_operator": plan.operator,
                        "reason": "bounded_resource_wait_exhausted",
                        "requested_at": time.time(),
                        "attempts": wait_attempts,
                        "threshold": replan_after,
                        "resource_decision": resource_decision.to_dict(),
                    }
                    self._save_campaign_state(state)
                    if self.controller_trace:
                        self.controller_trace[-1]["execution_status"] = "resource_unavailable_replan"
                    self.events.append(
                        "resource_wait_replan_requested",
                        {
                            "experiment_id": plan.experiment_id,
                            "operator": plan.operator,
                            "attempts": wait_attempts,
                            "threshold": replan_after,
                            "reason": "bounded_resource_wait_exhausted",
                            "resource_decision": resource_decision.to_dict(),
                            "next_planning_intent": "resource_recovery_cpu",
                        },
                    )
            return None
        self._clear_pending_state(state)
        state["last_resource_decision"] = resource_decision.to_dict()
        self._save_campaign_state(state)
        child_id = self._reserve_next_child_id()
        request_path = str(Path(self.config.remote.resolved_campaign_root) / (plan.experiment_id + "-request.json"))
        result_path = str(Path(self.config.remote.resolved_campaign_root) / ("trainer_result_%s.json" % child_id.lower()))
        output_dir = str(Path(self.config.remote.results_root or self.config.remote.model_root) / "continuous" / child_id)
        try:
            self._clear_worker_result_path(result_path, plan.experiment_id)
            template = dict(self.ssh.read_json(self.config.worker.config_template))
            dynamic = dict(template)
            dynamic.update({"model_checkpoint": parent.checkpoint_path, "output_dir": output_dir, "source_steps": parent.state.sampling_steps or 32})
            if plan.operator == "step_distill":
                dynamic["target_steps"] = int(operator_args["target_steps"])
            if plan.operator == "recovery_finetune":
                dynamic["max_steps"] = min(int(plan.operator_args.get("training_steps", dynamic.get("max_steps", 1))), self.config.worker.max_steps)
            if plan.operator == "distill":
                dynamic["max_steps"] = min(int(plan.operator_args.get("training_steps", dynamic.get("max_steps", 1))), self.config.worker.max_steps)
                dynamic["dataset_fraction"] = float(operator_args.get("dataset_fraction", 1.0))
            if plan.operator == "dmd2":
                dynamic["max_steps"] = min(int(plan.operator_args.get("training_steps", dynamic.get("max_steps", 1))), self.config.worker.max_steps)
                dynamic["generator_update_interval"] = int(operator_args.get("generator_update_interval", 2))
            launcher = self.config.worker.operator_launchers.get(plan.operator, "torchrun")
            if launcher == "torchrun":
                dynamic["world_size"] = resource_decision.actual_gpu_count
            self.ssh.write_json(str(Path(self.config.remote.resolved_campaign_root) / (plan.experiment_id + "-config.json")), dynamic)
            request = {
                "experiment_id": plan.experiment_id,
                "child_model_id": child_id,
                "operator": plan.operator,
                "operator_args": operator_args,
                "planned_resource_request": dict(planned_resource_request),
                "effective_resource_request": dict(request_for_pipeline),
                "parent": {
                    "model_id": parent.id,
                    "checkpoint_path": parent.checkpoint_path,
                    "state": parent.state.to_dict(),
                },
                "artifacts_dir": output_dir,
            }
            self.ssh.write_json(request_path, request)
            config_path = str(Path(self.config.remote.resolved_campaign_root) / (plan.experiment_id + "-config.json"))
            command = self._worker_command(
                plan.operator,
                config_path,
                request_path,
                result_path,
                allocated_gpu_count=resource_decision.actual_gpu_count,
            )
            if resource_decision.allocated_gpus:
                visible = ",".join(str(index) for index in resource_decision.allocated_gpus)
                command = ("env", "CUDA_VISIBLE_DEVICES=" + visible) + command
        except (RemoteError, OSError, TypeError, ValueError):
            self._release_child_id_reservation(child_id)
            self._release_worker_gpu_lease(plan.experiment_id, "worker_setup_failed")
            raise
        self.events.append(
            "worker_command_selected",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "launcher": launcher,
                "allocated_gpus": list(resource_decision.allocated_gpus),
                "planned_resource_request": dict(planned_resource_request),
                "effective_resource_request": dict(request_for_pipeline),
                "full_card_training": bool(full_gpu_training),
                "nproc_per_node": resource_decision.actual_gpu_count if launcher == "torchrun" else 1,
                "command_template": list(command),
            },
        )
        lane_evidence = self._lane_evidence(
            worker_gpus=resource_decision.allocated_gpus,
            status="ready",
            reason="trusted_worker_started",
        )
        self.events.append(
            "worker_started",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "allocated_gpus": list(resource_decision.allocated_gpus),
                "planned_resource_request": dict(planned_resource_request),
                "effective_resource_request": dict(request_for_pipeline),
                "full_card_training": bool(full_gpu_training),
                "nproc_per_node": resource_decision.actual_gpu_count if launcher == "torchrun" else 1,
                "command_template": list(command),
                "progress_source": "trusted worker stdout and result JSON",
                "lane_evidence": dict(lane_evidence),
            },
        )
        self._record_lane_allocation(
            "training",
            worker_gpus=resource_decision.allocated_gpus,
            planned_request=planned_resource_request,
            effective_request=request_for_pipeline,
            reason="trusted_worker",
        )
        if primary_batch_parallel_plan is not None:
            # The primary worker is CPU-only, so its exact primary batch can
            # immediately occupy the otherwise idle distributed lane. This
            # sibling is speculative and remains isolated until its own
            # result/evaluation gate completes; it does not mutate the active
            # model or bypass checkpoint/lineage validation.
            parallel_handle = self._start_speculative_worker(
                parent,
                primary_batch_parallel_plan,
                slot="parallel",
                preserve_controller_lane=self._preserve_controller_lane_for_speculative_worker(),
            )
            if parallel_handle is None:
                self.events.append(
                    "parallel_gpu_fill_skipped",
                    {
                        "candidate_model_id": parent.id,
                        "primary_experiment_id": plan.experiment_id,
                        "reason": "primary_batched_candidate_resource_unavailable",
                    },
                )
        if self.config.pipeline_enabled and not prefetch_started_before_training and not full_gpu_training:
            # Plan one safe alternative while the current trusted worker
            # occupies its allocated GPUs. The worker lease blocks those
            # cards from the Controller launcher, while an unused remainder
            # remains available for a TP=1/2 Controller placement.
            prefetch_handle = self._start_controller_prefetch(
                parent,
                training_calls,
                source_plan=plan,
                source_experiment_id=plan.experiment_id,
                source_child_model_id=child_id,
            )
            if prefetch_handle is not None:
                if str(plan.operator) in {"prune_blocks", "quantize"}:
                    # This worker itself uses no CUDA cards.  Start the
                    # filtered GPU branch concurrently with the ordinary
                    # successor request so evaluation can consume it without
                    # waiting for a second LLM round-trip while GPU1/2 sit
                    # idle.
                    eager_job = self._arm_eager_parallel_gpu_prefetch(
                        parent,
                        plan,
                        child_id,
                        training_calls,
                    )
                    if eager_job is None:
                        # Retain the older dependent fallback for transports
                        # that cannot launch the eager request immediately.
                        self._arm_parallel_prefetch_after_primary(
                            parent,
                            plan,
                            child_id,
                            training_calls,
                            prefetch_handle,
                        )
                else:
                    # For a GPU worker, the primary successor can be decided
                    # after its normal overlap request completes; the worker
                    # itself already occupies the useful training lane.
                    self._arm_parallel_prefetch_after_primary(
                        parent,
                        plan,
                        child_id,
                        training_calls,
                        prefetch_handle,
                    )
        heartbeat = self._start_review_heartbeat(
            plan.experiment_id,
            "training",
            lambda: {
                "operator": plan.operator,
                "allocated_gpus": list(resource_decision.allocated_gpus),
                "telemetry": self._collect_worker_telemetry(plan.experiment_id),
            },
        )
        training_power_sampler: Optional[RemotePowerSampler] = None
        training_power_summary: Mapping[str, Any] = {}
        try:
            # The sampler is telemetry only: it never changes CUDA clocks or
            # power limits.  In --on-server mode it uses the local transport,
            # so the measurement does not create a nested SSH bottleneck.
            training_power_sampler = RemotePowerSampler(
                self.ssh,
                self.config.sampling_interval_s,
            )
            training_power_sampler.start()
        except Exception as exc:
            training_power_sampler = None
            self.events.append(
                "training_power_sampling_unavailable",
                {"experiment_id": plan.experiment_id, "error": str(exc)[:500]},
            )
        # A real H3 checkpoint is ~66 GB on the remote NFS volume.  The first
        # load plus child write/reload can legitimately exceed three hours;
        # an overnight campaign must not turn transport timeout into a fake
        # training failure.
        worker_timeout_s = max(12.0 * 3600.0, float(dynamic.get("trainer_timeout_s", 0.0) or 0.0))
        try:
            # A trusted worker is required to publish a result JSON on both
            # success and failure.  Keep the transport result even when the
            # process exits non-zero so the failure can be imported into
            # experience memory instead of aborting the autonomous loop.
            worker_result = self.ssh.run(command, check=False, timeout_s=worker_timeout_s)
        finally:
            if training_power_sampler is not None:
                try:
                    training_power_sampler.stop()
                    training_power_summary = dict(training_power_sampler.summary())
                    training_power_summary["target_power_w"] = self.config.power_target_w
                    errors = list(getattr(training_power_sampler, "errors", ()))
                    if errors:
                        training_power_summary["errors"] = errors[-8:]
                    self.events.append(
                        "training_power_sampled",
                        {
                            "experiment_id": plan.experiment_id,
                            "summary": dict(training_power_summary),
                        },
                    )
                except Exception as exc:
                    self.events.append(
                        "training_power_sampling_unavailable",
                        {"experiment_id": plan.experiment_id, "error": str(exc)[:500]},
                    )
            self._stop_review_heartbeat(heartbeat)
            self._release_worker_gpu_lease(plan.experiment_id, "worker_completed")
            self._finish_controller_prefetch(prefetch_handle, wait_s=0.5)
        result_error = ""
        result_uri = "ssh://%s%s" % (self.config.remote.host, result_path)
        try:
            result = self.ssh.read_json(result_path)
            digest = self.ssh.sha256(result_path)
        except (RemoteError, OSError, TypeError, ValueError) as exc:
            # A hard crash can happen before rank 0 writes result JSON. Keep a
            # bounded synthetic failure record so the next Controller turn
            # sees the failure type and can choose a different operator. A
            # fragment prevents a later real result at the same path from
            # being treated as the same source event.
            result_error = str(exc)[:2000]
            result = {
                "status": "failed",
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "parent_model_id": parent.id,
                "failure_type": "worker_result_missing",
                "message": result_error,
                "metrics": {"real_worker": True, "offline_simulation": False},
                "cost": {
                    "wall_time_s": float(getattr(worker_result, "wall_time_s", 0.0) or 0.0),
                    "gpu_hours": 0.0,
                    "controller_calls": 0,
                },
            }
            result_uri += "#synthetic-failure"
            digest = hashlib.sha256(
                json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self.events.append(
                "worker_result_unavailable",
                {
                    "experiment_id": plan.experiment_id,
                    "operator": plan.operator,
                    "result_uri": result_uri,
                    "error": result_error,
                },
            )
        if isinstance(result, Mapping) and training_power_summary:
            # Keep the worker's result contract intact while making the
            # measured training hardware visible to the durable experience
            # record consumed by the next Controller prompt.
            result = dict(result)
            metrics = dict(result.get("metrics") or {})
            metrics["training_power_sampling"] = dict(training_power_summary)
            result["metrics"] = metrics
        if isinstance(result, Mapping):
            result = dict(result)
            metrics = dict(result.get("metrics") or {})
            metrics.setdefault("lane_evidence", dict(lane_evidence))
            result["metrics"] = metrics
        worker_succeeded = (
            isinstance(result, Mapping)
            and str(result.get("status", "")).strip().lower() in {"success", "succeeded", "ok", "completed"}
        )
        if not worker_succeeded:
            failed_retention = self.checkpoint_retention.cleanup_failed(output_dir, child_id)
            self.events.append("checkpoint_retention", failed_retention.to_dict())
            # The overlap request was planned against the child that this
            # worker was supposed to publish.  A hard worker failure means
            # that child has no valid lineage/checkpoint, so publishing that
            # prefetched plan would either reference a missing model or make
            # the next boundary spend time validating a dead branch.
            self._cancel_deferred_prefetches("worker_failed")
            latest_state = self._load_campaign_state()
            primary_prefetch_stale, parallel_prefetch_stale = self._clear_prefetch_state_for_child(
                latest_state,
                child_id,
            )
            if primary_prefetch_stale or parallel_prefetch_stale:
                self._save_campaign_state(latest_state)
            if primary_prefetch_stale:
                self.events.append(
                    "controller_plan_prefetch_discarded",
                    {
                        "experiment_id": plan.experiment_id,
                        "source_child_model_id": child_id,
                        "reason": "worker_failed_before_child_publication",
                    },
                )
            if parallel_prefetch_stale:
                self.events.append(
                    "controller_plan_parallel_prefetch_discarded",
                    {
                        "source_child_model_id": child_id,
                        "reason": "worker_failed_before_child_publication",
                    },
                )
        self.events.append(
            "worker_completed",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "result_uri": result_uri,
                "result_sha256": digest,
                "result_error": result_error or None,
                "returncode": int(getattr(worker_result, "returncode", 0)),
                "status": result.get("status") if isinstance(result, Mapping) else None,
                "failure_type": result.get("failure_type") if isinstance(result, Mapping) else None,
                "worker_log_tail": str(getattr(worker_result, "stdout", ""))[-4000:],
                "worker_error_tail": str(getattr(worker_result, "stderr", ""))[-4000:],
                "worker_log_head": str(getattr(worker_result, "stdout", ""))[:4000],
                "worker_error_head": str(getattr(worker_result, "stderr", ""))[:4000],
            },
        )
        self._review_now(
            "training",
            "worker_completed",
            {
                "experiment_id": plan.experiment_id,
                "operator": plan.operator,
                "status": result.get("status") if isinstance(result, Mapping) else None,
                "failure_type": result.get("failure_type") if isinstance(result, Mapping) else None,
                "returncode": int(getattr(worker_result, "returncode", 0)),
                "result_uri": result_uri,
            },
        )
        importer = RemoteResultImporter(self.experience, self.ssh)
        imported = importer.import_results([{"source_uri": result_uri, "source_sha256": digest, "result": result, "request": request}])
        for record in imported.records:
            self._append_observation(self._experience_observation(record))
        self._release_child_id_reservation(child_id)
        return imported.records[0] if imported.records else None

    def run(self, resume: bool = True, max_experiments: int = 1, split: Optional[str] = None) -> CampaignResult:
        """Run one cycle and release only an active ComfyUI benchmark lease.

        The benchmark service is a temporary evaluator, not a permanent
        owner of GPU 0.  Keeping this boundary in a ``finally`` block also
        covers controller/benchmark/decision exceptions, where the normal
        end-of-cycle release code would otherwise never run.  Release remains
        fail-closed: an active queue or a failed ``/free`` call keeps the GPU
        reserved instead of racing another workload.  When no benchmark lease
        is active, the scheduler and launcher use the live GPU waterline and
        no unnecessary ComfyUI ``/free`` request or 30-second wait is made.
        """

        try:
            return self._run_once(resume=resume, max_experiments=max_experiments, split=split)
        finally:
            speculative_states = (
                self._load_speculative_state(),
                self._load_speculative_state(self._speculative_state_path("parallel")),
            )
            if not any(
                str(item.get("status")) in {"launching", "running"}
                for item in speculative_states
            ):
                self._release_worker_gpu_lease("run-finally", "campaign_boundary")
            if self.config.comfyui_cache_policy in {"idle_release", "cold_cache"}:
                if self._comfyui_lease_active():
                    self._release_comfyui_if_idle("run_finally")

    def _run_once(self, resume: bool = True, max_experiments: int = 1, split: Optional[str] = None) -> CampaignResult:
        if max_experiments <= 0:
            raise ValueError("max_experiments must be positive")
        self.controller_trace = []
        self.controller_prefetch_trace = []
        self.last_comfyui_lease = None
        imported = self._import()
        all_records = list(self.experience.read())
        candidates = self._register_candidates(all_records)
        evaluations = self._load_evaluations()
        evaluations_changed = False
        for model_id, summary in list(evaluations.items()):
            if not isinstance(summary, Mapping) or isinstance(summary.get("evaluation_record"), Mapping):
                continue
            system = self._system_for_model(str(model_id))
            record = _evaluation_result(
                summary,
                model_id=str(model_id),
                system_id=system.id if system is not None else summary.get("system_id"),
                device_id=system.device_id if system is not None else summary.get("device_id"),
                task_split=summary.get("task_split") or split,
            )
            if record is not None:
                summary = dict(summary)
                summary["evaluation_record"] = record.to_dict()
                summary["evaluation_id"] = record.evaluation_id
                evaluations[str(model_id)] = summary
                evaluations_changed = True
        if evaluations_changed:
            self._save_evaluations(evaluations)
        self._demote_rejected_active(candidates, evaluations, all_records, split)
        retention_reconciled = self._reconcile_checkpoint_retention(
            candidates,
            evaluations,
            all_records,
            split,
        )
        evaluation_task_count = len(self._tasks(split))
        state = self._load_campaign_state() if resume else {
            "evaluated_ids": [],
            "training_calls": 0,
            "current_model_id": self.models.active_id,
            "current_system_id": "S0000",
            "goal": self._goal_payload(),
        }
        # The model store is the source of truth for the active lineage.  A
        # state file written by an earlier pre-lineage run may still contain
        # M0000 even though M0004 (or a later accepted/Pareto child) is active.
        # Reconcile before any planner or reviewer sees the state.
        try:
            active_model_id = str(self.models.active_id)
        except ModelStoreError:
            active_model_id = str(state.get("current_model_id", "M0000"))
        state["current_model_id"] = active_model_id
        active_system = self._system_for_model(active_model_id)
        if active_system is not None:
            state["current_system_id"] = active_system.id
        resource_recovery = state.get("resource_replan_intent")
        if isinstance(resource_recovery, Mapping):
            source_experiment_id = str(resource_recovery.get("source_experiment_id") or "")
            source_completed = any(
                str(record.experiment_id) == source_experiment_id
                and str(record.status) not in {"failed", "error", "cancelled"}
                for record in all_records
            )
            if source_experiment_id and source_completed:
                # Resume may enter evaluation before _train_one is reached.
                # Consume a marker whose source worker is already recorded so
                # an old recovery decision cannot steer a later branch.
                state.pop("resource_replan_intent", None)
                self.events.append(
                    "controller_resource_recovery_consumed",
                    {
                        "source_experiment_id": source_experiment_id,
                        "reason": "source_worker_already_recorded_at_cycle_start",
                    },
                )
        self._save_campaign_state(state)
        recovered_speculative = self._recover_speculative_worker()
        recovered_status = str(recovered_speculative.get("status") or "")
        if recovered_status in {"completed", "failed"} and "finalization_keep" in recovered_speculative:
            # A prior supervisor may have returned while the remote worker
            # was still running.  Once the durable result is visible, finish
            # the already-recorded keep/discard decision before planning a
            # new branch.
            self._finish_speculative_worker(
                recovered_speculative,
                keep=bool(recovered_speculative.get("finalization_keep")),
                reason=str(recovered_speculative.get("finalization_reason") or "recovered_finalization"),
            )
            recovered_speculative = self._load_speculative_state()
        recovered_parallel = self._recover_speculative_worker(
            self._speculative_state_path("parallel")
        )
        recovered_parallel_status = str(recovered_parallel.get("status") or "")
        if recovered_parallel_status in {"completed", "failed"} and "finalization_keep" in recovered_parallel:
            self._finish_speculative_worker(
                recovered_parallel,
                keep=bool(recovered_parallel.get("finalization_keep")),
                reason=str(recovered_parallel.get("finalization_reason") or "recovered_finalization"),
            )
            recovered_parallel = self._load_speculative_state(
                self._speculative_state_path("parallel")
            )
        evaluated_count = 0
        phase_evaluated_ids: List[str] = []
        candidate_records = [item for item in all_records if item.child_model_id and item.child_model_id in candidates]
        # A newly produced child is the loop's immediate feedback target. Do
        # not make it wait behind an old imported lineage when a bounded run
        # allows only one benchmark. Older unvalidated history remains in the
        # stream and is replayed on later resumes.
        pending_records = [
            item for item in candidate_records
            if not self._evaluation_is_current(evaluations.get(str(item.child_model_id), {}), split)
        ]
        evaluated_records = [
            item for item in candidate_records
            if self._evaluation_is_current(evaluations.get(str(item.child_model_id), {}), split)
        ]
        ordered = sorted(pending_records, key=lambda item: str(item.created_at), reverse=True) + sorted(
            evaluated_records,
            key=lambda item: (candidates[str(item.child_model_id)].generation, str(item.child_model_id)),
        )
        latest_records = {str(item.child_model_id): item for item in all_records if item.child_model_id}
        tunnel_work = False
        speculative_launch_lock = threading.Lock()
        speculative_launch_state: Dict[str, Any] = {
            "started": False,
            "cancelled": False,
            "thread": None,
            "prefetch": None,
            "handle": None,
        }
        for record in ordered:
            child_id = str(record.child_model_id)
            if self._evaluation_is_current(evaluations.get(child_id, {}), split) or evaluated_count >= max_experiments:
                continue
            candidate = candidates[child_id]
            parent = candidates.get(record.parent_model_id or "M0000")
            if parent is None:
                continue
            parent_summary = evaluations.get(parent.id)
            if parent_summary is None or not self._evaluation_is_current(parent_summary, split):
                parent_summary = self._evaluate(parent, None, split, self._system_for_model(parent.id))
                evaluations[parent.id] = parent_summary
                self._save_evaluations(evaluations)
                phase_evaluated_ids.append(parent.id)
                parent_observation = self._evaluation_observation(parent.id, parent_summary)
                self._append_observation(parent_observation)
                self.events.append("evaluation_completed", {"model_id": parent.id, "summary": parent_observation.summary})
                self._review_now(
                    "evaluating",
                    "evaluation_completed",
                    {"model_id": parent.id, "summary": parent_observation.summary},
                )
            parent = self._persist_evaluation_state(parent, parent_summary)
            candidates[parent.id] = parent
            speculative_handle: Optional[Mapping[str, Any]] = None
            parallel_speculative_handle: Optional[Mapping[str, Any]] = None
            if (
                recovered_speculative
                and str(recovered_speculative.get("parent_model_id")) == candidate.id
                and str(recovered_speculative.get("status") or "") in {"launching", "running", "completed", "failed"}
            ):
                speculative_handle = recovered_speculative
            if (
                recovered_parallel
                and str(recovered_parallel.get("parent_model_id")) == candidate.id
                and str(recovered_parallel.get("status") or "")
                in {"launching", "running", "completed", "failed"}
            ):
                parallel_speculative_handle = recovered_parallel
            speculative_launch_lock = threading.Lock()
            speculative_launch_state: Dict[str, Any] = {
                "started": False,
                "cancelled": False,
                "thread": None,
                "prefetch": None,
                "handle": None,
                "parallel_started": False,
                "parallel_prefetch": None,
                "parallel_handle": None,
                "evaluation_refill_started": False,
                "evaluation_refill_prefetch": None,
                "evaluation_refill_thread": None,
                "evaluation_refill_owned_prefetch": False,
            }
            evaluation_done = threading.Event()
            # Do not wait for the benchmark callback to discover that the
            # next plan is missing.  Start the one allowed primary prefetch
            # while the candidate's ComfyUI lease/checkpoint is preparing;
            # the callback reuses this exact handle and never opens a second
            # LLM request for the same boundary.
            evaluation_training_calls = int(state.get("training_calls", 0))
            initial_prefetch: Optional[Mapping[str, Any]] = None
            if self.config.pipeline_enabled and self.config.worker.enabled:
                try:
                    reusable_plan = self._speculative_plan_for_candidate(
                        candidate,
                        evaluation_training_calls,
                    )
                except (RemoteError, OSError, TypeError, ValueError):
                    reusable_plan = None
                if reusable_plan is None:
                    matching_prefetches = [
                        handle
                        for handle in self._deferred_prefetch_handles
                        if str(handle.get("source_child_model_id") or "") == candidate.id
                        and str(handle.get("prefetch_state_key") or "primary") == "primary"
                        and not bool(handle.get("prefetch_cancelled"))
                    ]
                    if matching_prefetches:
                        initial_prefetch = matching_prefetches[-1]
                    else:
                        initial_prefetch = self._start_controller_prefetch(
                            candidate,
                            evaluation_training_calls,
                            source_child_model_id=child_id,
                            planning_intent="evaluation_overlap",
                        )
                        if (
                            isinstance(initial_prefetch, dict)
                            and initial_prefetch not in self._deferred_prefetch_handles
                        ):
                            self._deferred_prefetch_handles.append(initial_prefetch)
                    if initial_prefetch is not None:
                        speculative_launch_state["prefetch"] = initial_prefetch
                        self.events.append(
                            "controller_plan_prefetch_advanced",
                            {
                                "parent_model_id": candidate.id,
                                "training_calls": evaluation_training_calls,
                                "reason": "evaluation_setup_before_benchmark_callback",
                            },
                        )
                if evaluation_task_count == 1:
                    self._arm_evaluation_parallel_prefetch(
                        candidate,
                        evaluation_training_calls,
                        child_id,
                    )

            def start_parallel_gpu_fill(primary_plan: ExperimentPlan) -> None:
                """Ask for one isolated GPU plan when the primary is CPU-only."""

                nonlocal parallel_speculative_handle
                if (
                    not self.config.pipeline_enabled
                    or int(self.config.pipeline_max_inflight) < 2
                    or str(primary_plan.operator) not in {"prune_blocks", "quantize"}
                ):
                    self.events.append(
                        "parallel_gpu_fill_skipped",
                        {
                            "candidate_model_id": candidate.id,
                            "primary_experiment_id": primary_plan.experiment_id,
                            "reason": "primary_plan_is_not_cpu_only_or_pipeline_disabled",
                        },
                    )
                    return
                with speculative_launch_lock:
                    if (
                        parallel_speculative_handle is not None
                        or speculative_launch_state["parallel_started"]
                        or speculative_launch_state["cancelled"]
                    ):
                        return
                    speculative_launch_state["parallel_started"] = True
                fill_prefetch: Optional[Mapping[str, Any]] = None
                ready_parallel_plan: Optional[ExperimentPlan] = None
                planning_training_calls = int(state.get("training_calls", 0))
                # A prefetch handle stores the cursor of the successor it
                # generates (current + 1).  Keep the base cursor separate so
                # evaluation can reuse that exact request instead of issuing
                # a second LLM call and leaving the two overlap GPUs idle.
                training_call_cursor = planning_training_calls + 1
                early_job = self._early_parallel_prefetch_job(candidate.id, training_call_cursor)
                if early_job is not None:
                    early_handle = early_job.get("parallel_handle")
                    if isinstance(early_handle, Mapping):
                        fill_prefetch = early_handle
                if fill_prefetch is None:
                    try:
                        persisted_parallel = self._parallel_prefetched_plan(state)
                    except (TypeError, ValueError, KeyError) as exc:
                        persisted_parallel = None
                        self._clear_parallel_prefetch_state(state)
                        self._save_campaign_state(state)
                        self.events.append(
                            "parallel_gpu_fill_skipped",
                            {
                                "candidate_model_id": candidate.id,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "reason": "invalid_persisted_parallel_plan",
                                "error": str(exc)[:800],
                            },
                        )
                    if self._parallel_prefetch_is_current(
                        state,
                        persisted_parallel,
                        candidate.id,
                        training_call_cursor,
                    ):
                        ready_parallel_plan = persisted_parallel
                        self.events.append(
                            "controller_plan_parallel_reused",
                            {
                                "experiment_id": persisted_parallel.experiment_id,
                                "parent_model_id": candidate.id,
                                "training_calls": training_call_cursor,
                                "reason": "early_parallel_prefetch_ready_before_evaluation",
                            },
                        )
                if fill_prefetch is None and ready_parallel_plan is None and early_job is None:
                    # The primary evaluation-overlap request is already an
                    # n-way Controller call.  Reuse one of its eligible GPU
                    # candidates before opening a second request for the same
                    # boundary; this is the common CPU-primary idle window.
                    with speculative_launch_lock:
                        primary_prefetch = speculative_launch_state.get("prefetch")
                    batched_parallel = self._parallel_candidate_from_prefetch_handle(
                        primary_prefetch,
                        primary_plan,
                        training_call_cursor,
                    )
                    if batched_parallel is not None:
                        ready_parallel_plan = batched_parallel
                        self._persist_batched_parallel_candidate(
                            primary_prefetch,
                            batched_parallel,
                            primary_plan.experiment_id,
                        )
                        self.events.append(
                            "controller_plan_parallel_reused",
                            {
                                "experiment_id": batched_parallel.experiment_id,
                                "parent_model_id": batched_parallel.parent_model_id,
                                "training_calls": training_call_cursor,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "operator": batched_parallel.operator,
                                "reason": "batched_controller_candidate",
                            },
                        )
                if fill_prefetch is None and ready_parallel_plan is None and early_job is None:
                    try:
                        memory, processes = self.scheduler.snapshot()
                        reserved = set(int(index) for index in self.scheduler.reserved_gpu_indices)
                        # Custom test/transport schedulers may not expose the
                        # process mapping. If they report live processes,
                        # treat the missing mapping as unknown and fail closed.
                        process_indices = getattr(self.scheduler, "last_compute_gpu_indices", None)
                        if processes.strip() and process_indices is None:
                            process_busy = set(range(int(self.scheduler.gpu_count)))
                        else:
                            process_busy = set(int(index) for index in (process_indices or ()))
                        available = [
                            index
                            for index in range(int(self.scheduler.gpu_count))
                            if (
                                index not in reserved
                                and index not in process_busy
                                and self.scheduler.meets_memory_waterline(index, memory)
                            )
                        ]
                    except Exception as exc:
                        self.events.append(
                            "parallel_gpu_fill_skipped",
                            {
                                "candidate_model_id": candidate.id,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "reason": "capacity_probe_failed",
                                "error": str(exc)[:800],
                            },
                        )
                        return
                    if len(available) < 2:
                        self.events.append(
                            "parallel_gpu_fill_skipped",
                            {
                                "candidate_model_id": candidate.id,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "available_gpu_indices": available,
                                "blocked_compute_gpu_indices": sorted(process_busy),
                                "reason": "fewer_than_two_gpu_waterline_slots",
                            },
                        )
                        return
                    self.events.append(
                        "controller_plan_parallel_waiting",
                        {
                            "candidate_model_id": candidate.id,
                            "primary_experiment_id": primary_plan.experiment_id,
                            "available_gpu_indices": available,
                            "reason": "no_ready_batched_gpu_candidate",
                        },
                    )
                    fill_prefetch = self._start_controller_prefetch(
                        candidate,
                        planning_training_calls,
                        source_plan=primary_plan,
                        source_experiment_id=primary_plan.experiment_id,
                        planning_intent="parallel_gpu_fill",
                        operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
                        prefetch_state_key="parallel",
                    )
                    if fill_prefetch is None:
                        self.events.append(
                            "parallel_gpu_fill_skipped",
                            {
                                "candidate_model_id": candidate.id,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "reason": "parallel_controller_prefetch_unavailable",
                                "available_gpu_indices": available,
                            },
                        )
                        return
                with speculative_launch_lock:
                    speculative_launch_state["parallel_prefetch"] = fill_prefetch
                if isinstance(fill_prefetch, dict) and fill_prefetch not in self._deferred_prefetch_handles:
                    self._deferred_prefetch_handles.append(fill_prefetch)

                def launch_parallel_after_prefetch() -> None:
                    nonlocal parallel_speculative_handle
                    try:
                        parallel_handle = fill_prefetch
                        fill_plan = ready_parallel_plan
                        if fill_plan is None and parallel_handle is None and early_job is not None:
                            done = early_job.get("done")
                            if isinstance(done, threading.Event):
                                done.wait(
                                    timeout=max(
                                        30.0,
                                        float(getattr(self.controller, "timeout_s", 180.0) or 180.0) + 60.0,
                                    )
                                )
                            parallel_handle = early_job.get("parallel_handle")
                            if parallel_handle is None:
                                latest_state = self._load_campaign_state()
                                try:
                                    fill_plan = self._parallel_prefetched_plan(latest_state)
                                except (TypeError, ValueError, KeyError):
                                    fill_plan = None
                        if fill_plan is None and parallel_handle is None:
                            # The early arm may have lost a Controller
                            # restart race. Fall back to exactly one request
                            # after its marker is finished; the primary plan
                            # remains the source of the experiment cursor.
                            parallel_handle = self._start_controller_prefetch(
                                candidate,
                                planning_training_calls,
                                source_plan=primary_plan,
                                source_experiment_id=primary_plan.experiment_id,
                                planning_intent="parallel_gpu_fill",
                                operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
                                prefetch_state_key="parallel",
                            )
                        if fill_plan is None:
                            fill_plan = self._finish_controller_prefetch(parallel_handle)
                        if fill_plan is None:
                            # The durable cursor is useful only while this
                            # exact overlap request is still launchable.  A
                            # failed/filtered prefetch must not be replayed
                            # after resume as if it were a fresh successor.
                            if isinstance(fill_prefetch, dict):
                                fill_prefetch["prefetch_cancelled"] = True
                            latest_state = self._load_campaign_state()
                            self._clear_parallel_prefetch_state(latest_state)
                            self._save_campaign_state(latest_state)
                            return
                        with speculative_launch_lock:
                            if speculative_launch_state["cancelled"]:
                                if isinstance(fill_prefetch, dict):
                                    fill_prefetch["prefetch_cancelled"] = True
                                latest_state = self._load_campaign_state()
                                self._clear_parallel_prefetch_state(latest_state)
                                self._save_campaign_state(latest_state)
                                return
                        parallel_wait_s = None
                        if isinstance(parallel_handle, Mapping) and parallel_handle.get("started") is not None:
                            try:
                                parallel_wait_s = max(
                                    0.0,
                                    time.monotonic() - float(parallel_handle.get("started")),
                                )
                            except (TypeError, ValueError):
                                parallel_wait_s = None
                        self.events.append(
                            "controller_plan_parallel_ready",
                            {
                                "experiment_id": fill_plan.experiment_id,
                                "parent_model_id": fill_plan.parent_model_id,
                                "planning_intent": "parallel_gpu_fill",
                                "primary_experiment_id": primary_plan.experiment_id,
                                "wait_s": parallel_wait_s,
                            },
                        )
                        launch_ready_speculative_worker(
                            candidate,
                            fill_plan,
                            slot="parallel",
                        )
                    except Exception as exc:
                        self.events.append(
                            "parallel_gpu_fill_skipped",
                            {
                                "candidate_model_id": candidate.id,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "reason": "parallel_worker_launch_failed",
                                "error": str(exc)[:1200],
                            },
                        )

                launcher_thread = threading.Thread(
                    target=launch_parallel_after_prefetch,
                    name="harness4h3-parallel-gpu-fill-%s" % candidate.id,
                    daemon=True,
                )
                with speculative_launch_lock:
                    speculative_launch_state["parallel_thread"] = launcher_thread
                launcher_thread.start()

            def start_evaluation_refill(
                primary_plan: ExperimentPlan,
                primary_handle: Optional[Mapping[str, Any]],
            ) -> None:
                """Prepare one bounded sibling to cover a long evaluation tail.

                The normal primary worker is already the next hypothesis.  If
                it finishes before ComfyUI, ask the local Controller for one
                additional GPU-capable sibling rooted at the *same* candidate
                and launch it only after the primary lease is released.  This
                keeps the overlap queue bounded while eliminating the common
                ``worker_done -> evaluation_done`` idle gap.
                """

                if str(primary_plan.operator) in {"prune_blocks", "quantize"}:
                    # CPU-only primaries already use start_parallel_gpu_fill.
                    return
                if (
                    not self.config.pipeline_enabled
                    or int(self.config.pipeline_max_inflight) < 2
                    or not isinstance(primary_handle, Mapping)
                ):
                    self.events.append(
                        "evaluation_refill_skipped",
                        {
                            "candidate_model_id": candidate.id,
                            "primary_experiment_id": primary_plan.experiment_id,
                            "reason": "pipeline_disabled_or_no_primary_worker",
                        },
                    )
                    return
                with speculative_launch_lock:
                    if (
                        speculative_launch_state["evaluation_refill_started"]
                        or speculative_launch_state["parallel_started"]
                        or parallel_speculative_handle is not None
                        or speculative_launch_state["cancelled"]
                    ):
                        return
                    speculative_launch_state["evaluation_refill_started"] = True

                planning_training_calls = int(state.get("training_calls", 0)) + 1
                ready_plan: Optional[ExperimentPlan] = None
                refill_prefetch: Optional[Mapping[str, Any]] = None
                owns_prefetch = False
                latest_state = self._load_campaign_state()
                ready_plan = self._ready_parallel_prefetched_plan(
                    latest_state,
                    candidate.id,
                    planning_training_calls,
                )
                if ready_plan is None:
                    early_job = self._early_parallel_prefetch_job(
                        candidate.id,
                        planning_training_calls,
                    )
                    if isinstance(early_job, Mapping):
                        candidate_handle = early_job.get("parallel_handle")
                        if isinstance(candidate_handle, Mapping):
                            refill_prefetch = candidate_handle
                    if refill_prefetch is None:
                        match = re.fullmatch(r"exp_(\d+)", str(primary_plan.experiment_id))
                        if match is None:
                            self.events.append(
                                "evaluation_refill_skipped",
                                {
                                    "candidate_model_id": candidate.id,
                                    "primary_experiment_id": primary_plan.experiment_id,
                                    "reason": "primary_experiment_id_not_numeric",
                                },
                            )
                            return
                        refill_prefetch = self._start_controller_prefetch(
                            candidate,
                            int(state.get("training_calls", 0)),
                            source_experiment_id=primary_plan.experiment_id,
                            source_child_model_id=child_id,
                            planning_intent="evaluation_refill",
                            operator_filter=("recovery_finetune", "distill", "step_distill", "dmd2"),
                            prefetch_state_key="parallel",
                            experiment_cursor=int(match.group(1)) + 1,
                        )
                        owns_prefetch = refill_prefetch is not None
                        if (
                            isinstance(refill_prefetch, dict)
                            and refill_prefetch not in self._deferred_prefetch_handles
                        ):
                            self._deferred_prefetch_handles.append(refill_prefetch)
                if ready_plan is None and refill_prefetch is None:
                    self.events.append(
                        "evaluation_refill_skipped",
                        {
                            "candidate_model_id": candidate.id,
                            "primary_experiment_id": primary_plan.experiment_id,
                            "reason": "refill_prefetch_unavailable",
                        },
                    )
                    return
                with speculative_launch_lock:
                    speculative_launch_state["evaluation_refill_prefetch"] = refill_prefetch
                    speculative_launch_state["evaluation_refill_owned_prefetch"] = owns_prefetch

                def launch_after_primary() -> None:
                    try:
                        primary_thread = primary_handle.get("thread")
                        if isinstance(primary_thread, threading.Thread):
                            primary_thread.join()
                        if evaluation_done.is_set() or speculative_launch_state["cancelled"]:
                            return
                        fill_plan = ready_plan
                        if fill_plan is None:
                            fill_plan = self._finish_controller_prefetch(refill_prefetch)
                        if fill_plan is None or evaluation_done.is_set():
                            return
                        with speculative_launch_lock:
                            if (
                                speculative_launch_state["cancelled"]
                                or evaluation_done.is_set()
                                or speculative_launch_state["parallel_started"]
                                or parallel_speculative_handle is not None
                            ):
                                return
                            speculative_launch_state["parallel_started"] = True
                        self.events.append(
                            "controller_plan_evaluation_refill_ready",
                            {
                                "experiment_id": fill_plan.experiment_id,
                                "parent_model_id": fill_plan.parent_model_id,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "training_calls": planning_training_calls,
                                "reason": "primary_speculative_worker_finished_before_evaluation",
                            },
                        )
                        launch_ready_speculative_worker(
                            candidate,
                            fill_plan,
                            slot="parallel",
                        )
                    except Exception as exc:
                        self.events.append(
                            "evaluation_refill_skipped",
                            {
                                "candidate_model_id": candidate.id,
                                "primary_experiment_id": primary_plan.experiment_id,
                                "reason": "refill_launch_failed",
                                "error": str(exc)[:1200],
                            },
                        )

                refill_thread = threading.Thread(
                    target=launch_after_primary,
                    name="harness4h3-evaluation-refill-%s" % candidate.id,
                    daemon=True,
                )
                with speculative_launch_lock:
                    speculative_launch_state["evaluation_refill_thread"] = refill_thread
                refill_thread.start()
                self.events.append(
                    "controller_plan_evaluation_refill_started",
                    {
                        "candidate_model_id": candidate.id,
                        "primary_experiment_id": primary_plan.experiment_id,
                        "training_calls": planning_training_calls,
                        "prefetch_ready": ready_plan is not None,
                        "reason": "preplan_sibling_during_primary_speculative_worker",
                    },
                )

            def launch_ready_speculative_worker(
                worker_parent: ModelCandidate,
                ready_plan: ExperimentPlan,
                *,
                slot: str = "primary",
            ) -> Optional[Mapping[str, Any]]:
                """Launch a ready successor, retrying only a transient lease wait.

                The evaluation callback runs after the evaluator lease is
                reserved, so the first scheduler sample can still see a
                Controller process or a just-released worker lease.  A single
                ``wait`` must not turn into an idle evaluation window.  Keep
                the validated plan and retry for a short bounded interval;
                malformed plans and Controller handoff failures remain
                fail-closed.
                """

                nonlocal speculative_handle, parallel_speculative_handle

                def remember(handle: Mapping[str, Any]) -> None:
                    nonlocal speculative_handle, parallel_speculative_handle
                    with speculative_launch_lock:
                        if slot == "parallel":
                            parallel_speculative_handle = handle
                            speculative_launch_state["parallel_handle"] = handle
                        else:
                            speculative_handle = handle
                            speculative_launch_state["handle"] = handle
                    latest_state = self._load_campaign_state()
                    if slot == "parallel":
                        self._clear_parallel_prefetch_state(latest_state)
                    else:
                        self._clear_pending_state(latest_state)
                    self._save_campaign_state(latest_state)

                def attempt() -> Tuple[Optional[Mapping[str, Any]], Dict[str, Any]]:
                    status: Dict[str, Any] = {}
                    handle = self._start_speculative_worker(
                        worker_parent,
                        ready_plan,
                        slot=slot,
                        preserve_controller_lane=(
                            self._preserve_controller_lane_for_speculative_worker()
                        ),
                        launch_status=status,
                    )
                    if handle is not None:
                        remember(handle)
                    return handle, status

                handle, launch_status = attempt()
                if handle is not None:
                    if slot == "primary":
                        if str(ready_plan.operator) in {"prune_blocks", "quantize"}:
                            start_parallel_gpu_fill(ready_plan)
                        else:
                            start_evaluation_refill(ready_plan, handle)
                    return handle
                if not bool(launch_status.get("retryable")):
                    if slot == "parallel":
                        latest_state = self._load_campaign_state()
                        self._clear_parallel_prefetch_state(latest_state)
                        self._save_campaign_state(latest_state)
                    return None

                retry_window_s = max(
                    5.0,
                    min(
                        300.0,
                        float(getattr(self.config, "benchmark_task_timeout_s", 300.0)),
                    ),
                )
                retry_state: Dict[str, Any] = {"scheduled": True, "started": False}
                retry_key = "parallel_retry_thread" if slot == "parallel" else "retry_thread"
                with speculative_launch_lock:
                    speculative_launch_state[retry_key] = retry_state
                self.events.append(
                    "speculative_worker_retry_scheduled",
                    {
                        "experiment_id": ready_plan.experiment_id,
                        "parent_model_id": worker_parent.id,
                        "slot": slot,
                        "retry_window_s": retry_window_s,
                        "reason": "transient_resource_wait_during_evaluation",
                    },
                )

                def retry() -> None:
                    deadline = time.monotonic() + retry_window_s
                    attempt_number = 0
                    try:
                        while time.monotonic() < deadline:
                            with speculative_launch_lock:
                                cancelled = bool(speculative_launch_state["cancelled"])
                            if cancelled or evaluation_done.is_set():
                                break
                            evaluation_done.wait(timeout=1.0)
                            if evaluation_done.is_set():
                                break
                            attempt_number += 1
                            candidate_handle, candidate_status = attempt()
                            if candidate_handle is not None:
                                retry_state["started"] = True
                                self.events.append(
                                    "speculative_worker_retry_started",
                                    {
                                        "experiment_id": ready_plan.experiment_id,
                                        "parent_model_id": worker_parent.id,
                                        "slot": slot,
                                        "attempt": attempt_number,
                                        "reason": "resource_available_after_transient_wait",
                                    },
                                )
                                if slot == "primary":
                                    if str(ready_plan.operator) in {"prune_blocks", "quantize"}:
                                        start_parallel_gpu_fill(ready_plan)
                                    else:
                                        start_evaluation_refill(ready_plan, candidate_handle)
                                return
                            if not bool(candidate_status.get("retryable")):
                                break
                        self.events.append(
                            "speculative_worker_retry_exhausted",
                            {
                                "experiment_id": ready_plan.experiment_id,
                                "parent_model_id": worker_parent.id,
                                "slot": slot,
                                "attempts": attempt_number,
                                "reason": (
                                    "evaluation_completed"
                                    if evaluation_done.is_set()
                                    else "resource_wait_window_expired"
                                ),
                            },
                        )
                    finally:
                        if slot == "parallel" and not retry_state["started"]:
                            latest_state = self._load_campaign_state()
                            self._clear_parallel_prefetch_state(latest_state)
                            self._save_campaign_state(latest_state)

                retry_thread = threading.Thread(
                    target=retry,
                    name="harness4h3-speculative-retry-%s" % ready_plan.experiment_id,
                    daemon=True,
                )
                retry_state["thread"] = retry_thread
                retry_thread.start()
                return None

            def start_speculative_worker() -> None:
                nonlocal speculative_handle
                with speculative_launch_lock:
                    if speculative_handle is not None or speculative_launch_state["started"]:
                        return
                    speculative_launch_state["started"] = True
                if evaluation_task_count > 1:
                    # Multiple independent evaluator tasks already occupy the
                    # available overlap lanes.  A distributed speculative
                    # worker would only wait for two cards and add no useful
                    # throughput during this benchmark window.
                    self.events.append(
                        "speculative_worker_skipped",
                        {
                            "candidate_model_id": candidate.id,
                            "reason": "evaluation_task_fanout_owns_available_gpu_lanes",
                            "evaluation_tasks": evaluation_task_count,
                        },
                    )
                    return
                plan = self._speculative_plan_for_candidate(
                    candidate,
                    int(state.get("training_calls", 0)),
                )
                speculative_parent = candidate
                # The GPU-fill cursor has its own exact boundary.  Consume it
                # before falling back to a new primary request; otherwise a
                # ready sibling can sit in durable state while evaluation
                # leaves its overlap cards unused.
                planning_training_calls = int(state.get("training_calls", 0))
                latest_state = self._load_campaign_state()
                ready_parallel_plan = self._ready_parallel_prefetched_plan(
                    latest_state,
                    candidate.id,
                    planning_training_calls + 1,
                )
                if ready_parallel_plan is not None and (
                    plan is None or str(plan.operator) in {"prune_blocks", "quantize"}
                ):
                    launch_parallel = False
                    with speculative_launch_lock:
                        if (
                            parallel_speculative_handle is None
                            and not speculative_launch_state["parallel_started"]
                            and not speculative_launch_state["cancelled"]
                        ):
                            speculative_launch_state["parallel_started"] = True
                            launch_parallel = True
                    if launch_parallel:
                        self.events.append(
                            "controller_plan_parallel_reused",
                            {
                                "experiment_id": ready_parallel_plan.experiment_id,
                                "parent_model_id": candidate.id,
                                "training_calls": planning_training_calls + 1,
                                "reason": "ready_parallel_cursor_without_primary_plan",
                            },
                        )
                        parallel_handle = launch_ready_speculative_worker(
                            candidate,
                            ready_parallel_plan,
                            slot="parallel",
                        )
                        # The dedicated GPU branch already occupies the
                        # evaluator overlap cards. Do not open a second
                        # primary request when the normal successor is still
                        # unknown; it would only queue another worker behind
                        # the same two cards and recreate the idle boundary.
                        if plan is None and (
                            parallel_handle is not None
                            or isinstance(
                                speculative_launch_state.get("parallel_retry_thread"),
                                Mapping,
                            )
                        ):
                            return
                if plan is None:
                    # After a rejected candidate, the next validated plan is
                    # intentionally rooted at the active parent rather than
                    # at the candidate currently being benchmarked.  It is
                    # still a useful independent branch: launch it on the
                    # GPUs freed by the single ComfyUI task instead of
                    # leaving those cards idle until the benchmark ends.
                    latest_state = self._load_campaign_state()
                    try:
                        pending = self._pending_plan(latest_state)
                    except (TypeError, ValueError, KeyError):
                        pending = None
                    try:
                        pending_calls_match = int(latest_state.get("training_calls", -1)) == int(
                            state.get("training_calls", 0)
                        )
                    except (TypeError, ValueError):
                        pending_calls_match = False
                    if (
                        pending is not None
                        and pending_calls_match
                        and pending.parent_model_id == parent.id
                    ):
                        plan = pending
                        speculative_parent = parent
                        self.events.append(
                            "speculative_worker_plan_selected",
                            {
                                "experiment_id": plan.experiment_id,
                                "plan_parent_model_id": plan.parent_model_id,
                                "candidate_model_id": candidate.id,
                                "reason": "pending_active_parent_branch_during_candidate_evaluation",
                            },
                        )
                if plan is None:
                    # A restart or a just-released Controller may leave no
                    # persisted prefetch. Start exactly one fallback plan in
                    # the background at the benchmark boundary. The
                    # benchmark can proceed while the LLM plans; the
                    # successor worker is launched as soon as that plan is
                    # durable, so a long evaluation does not serialize GPU
                    # allocation behind a Controller round trip.
                    if self.config.pipeline_enabled and self.config.pipeline_max_inflight >= 1:
                        overlap_wait_started = time.monotonic()
                        self.events.append(
                            "controller_plan_overlap_waiting",
                            {
                                "candidate_model_id": candidate.id,
                                "training_calls": int(state.get("training_calls", 0)),
                                "reason": "no_ready_speculative_plan_at_evaluation_boundary",
                            },
                        )
                        fallback = speculative_launch_state.get("prefetch")
                        if fallback is None:
                            fallback = self._start_controller_prefetch(
                                candidate,
                                int(state.get("training_calls", 0)),
                                source_child_model_id=child_id,
                                planning_intent="evaluation_overlap",
                            )
                        if fallback is not None:
                            with speculative_launch_lock:
                                speculative_launch_state["prefetch"] = fallback
                            if (
                                isinstance(fallback, dict)
                                and fallback not in self._deferred_prefetch_handles
                            ):
                                self._deferred_prefetch_handles.append(fallback)
                            def launch_after_prefetch() -> None:
                                try:
                                    # The training-side prefetch may publish a
                                    # valid plan just after this evaluation
                                    # callback starts.  Do not wait only on the
                                    # fallback handle created above: that used
                                    # to miss the already-running prefetch and
                                    # leave the spare evaluator cards idle.
                                    # Reuse only the exact parent/cursor plan;
                                    # no second LLM request is created here.
                                    ready_plan: Optional[ExperimentPlan] = None
                                    deadline = time.monotonic() + max(
                                        30.0,
                                        float(self.config.benchmark_task_timeout_s),
                                    )
                                    while time.monotonic() < deadline:
                                        latest_state = self._load_campaign_state()
                                        try:
                                            persisted = self._prefetched_plan(latest_state)
                                        except (TypeError, ValueError, KeyError):
                                            persisted = None
                                        try:
                                            persisted_calls_match = int(
                                                latest_state.get("prefetched_training_calls", -1)
                                            ) == int(state.get("training_calls", 0))
                                        except (TypeError, ValueError):
                                            persisted_calls_match = False
                                        if (
                                            persisted is not None
                                            and persisted.parent_model_id == candidate.id
                                            and persisted_calls_match
                                        ):
                                            ready_plan = persisted
                                            self.events.append(
                                                "controller_plan_overlap_reused",
                                                {
                                                    "experiment_id": persisted.experiment_id,
                                                    "parent_model_id": persisted.parent_model_id,
                                                    "reason": "persisted_training_prefetch_available_during_evaluation",
                                                },
                                            )
                                            break
                                        # Finish the callback-owned request
                                        # only after its thread exits.  Calling
                                        # the join helper every polling tick
                                        # would emit a deferred event for each
                                        # tick; meanwhile another prefetch can
                                        # still win the race and publish the
                                        # durable plan above.
                                        fallback_thread = (
                                            fallback.get("thread")
                                            if isinstance(fallback, Mapping)
                                            else None
                                        )
                                        if not isinstance(fallback_thread, threading.Thread):
                                            break
                                        if not fallback_thread.is_alive():
                                            candidate_plan = self._finish_controller_prefetch(
                                                fallback,
                                                wait_s=0.0,
                                            )
                                            if candidate_plan is not None:
                                                ready_plan = candidate_plan
                                            break
                                        time.sleep(1.0)
                                    if ready_plan is None:
                                        return
                                    if speculative_launch_state["cancelled"]:
                                        return
                                    self.events.append(
                                        "controller_plan_overlap_ready",
                                        {
                                            "experiment_id": ready_plan.experiment_id,
                                            "parent_model_id": ready_plan.parent_model_id,
                                            "wait_s": max(0.0, time.monotonic() - overlap_wait_started),
                                            "reason": "speculative_plan_ready_after_evaluation_wait",
                                        },
                                    )
                                    self.events.append(
                                        "controller_plan_overlap_generated",
                                        {
                                            "experiment_id": ready_plan.experiment_id,
                                            "parent_model_id": ready_plan.parent_model_id,
                                            "reason": "benchmark_boundary_fallback_async",
                                        },
                                    )
                                    launch_ready_speculative_worker(
                                        speculative_parent,
                                        ready_plan,
                                        slot="primary",
                                    )
                                except Exception as exc:
                                    self.events.append(
                                        "speculative_worker_discarded",
                                        {
                                            "reason": "async_launch_failed",
                                            "error": str(exc)[:1200],
                                        },
                                    )

                            launcher_thread = threading.Thread(
                                target=launch_after_prefetch,
                                name="harness4h3-speculative-launch-%s" % candidate.id,
                                daemon=True,
                            )
                            with speculative_launch_lock:
                                speculative_launch_state["thread"] = launcher_thread
                            launcher_thread.start()
                            return
                    if plan is None:
                        return
                launch_ready_speculative_worker(
                    speculative_parent,
                    plan,
                    slot="primary",
                )

            def current_speculative_handle() -> Optional[Mapping[str, Any]]:
                with speculative_launch_lock:
                    return speculative_handle or speculative_launch_state.get("handle")

            def current_speculative_handles() -> List[Mapping[str, Any]]:
                with speculative_launch_lock:
                    values = []
                    for item in (
                        speculative_handle,
                        speculative_launch_state.get("handle"),
                        parallel_speculative_handle,
                        speculative_launch_state.get("parallel_handle"),
                    ):
                        if item is not None and item not in values:
                            values.append(item)
                    return values

            try:
                summary = self._evaluate(
                    candidate,
                    parent_summary,
                    split,
                    self._system_for_model(candidate.id),
                    on_benchmark_reserved=start_speculative_worker,
                )
            except Exception:
                with speculative_launch_lock:
                    speculative_launch_state["cancelled"] = True
                    fallback = speculative_launch_state.get("prefetch")
                    if isinstance(fallback, dict):
                        fallback["prefetch_cancelled"] = True
                active_speculative_handles = current_speculative_handles()
                for active_speculative_handle in active_speculative_handles:
                    self._finish_speculative_worker(
                        active_speculative_handle,
                        keep=False,
                        reason="current_evaluation_failed",
                    )
                raise
            finally:
                # Retry threads are only useful while the benchmark still
                # owns its overlap window.  Once evaluation returns, leave
                # the durable plan for the next boundary but never start a
                # new worker behind the completed decision.
                evaluation_done.set()
                with speculative_launch_lock:
                    refill_prefetch = speculative_launch_state.get("evaluation_refill_prefetch")
                    refill_owned = bool(speculative_launch_state.get("evaluation_refill_owned_prefetch"))
                    refill_started = bool(speculative_launch_state.get("parallel_started"))
                if refill_owned and not refill_started:
                    if isinstance(refill_prefetch, dict):
                        refill_prefetch["prefetch_cancelled"] = True
                    latest_state = self._load_campaign_state()
                    source_matches = (
                        isinstance(refill_prefetch, Mapping)
                        and str(latest_state.get("parallel_prefetched_source_experiment_id") or "")
                        == str(refill_prefetch.get("source_experiment_id") or "")
                        and str(latest_state.get("parallel_prefetched_parent_model_id") or "") == candidate.id
                    )
                    if source_matches:
                        self._clear_parallel_prefetch_state(latest_state)
                        self._save_campaign_state(latest_state)
                        self.events.append(
                            "controller_plan_evaluation_refill_discarded",
                            {
                                "candidate_model_id": candidate.id,
                                "reason": "evaluation_completed_before_refill_launch",
                            },
                        )
            evaluations[child_id] = summary
            self._save_evaluations(evaluations)
            phase_evaluated_ids.append(child_id)
            candidate_observation = self._evaluation_observation(child_id, summary)
            self._append_observation(candidate_observation)
            self.events.append("evaluation_completed", {"model_id": child_id, "summary": candidate_observation.summary})
            self._review_now(
                "evaluating",
                "evaluation_completed",
                {"model_id": child_id, "summary": candidate_observation.summary},
            )
            candidate = self._persist_evaluation_state(candidate, summary)
            candidates[child_id] = candidate
            training = dict(record.training)
            decision = decide(AcceptanceInput(training, summary, parent_summary, self.target, self.config.efficiency_thresholds, self.config.reward, self.config.research_grade))
            round_gate = self._evaluate_round_gate(record, summary)
            gate_authoritative = bool(
                training.get("real_worker") is True
                and isinstance(training.get("lane_evidence"), Mapping)
                and bool(training.get("lane_evidence", {}).get("worker_gpus"))
            )
            if gate_authoritative and round_gate.status != "accepted":
                decision = self._block_decision_for_round_gate(decision, round_gate)
            # Pareto-eligible exploratory candidates may become the active
            # parent even when research-grade acceptance is not enabled; keep
            # those weights for the next loop just like an accepted child.
            if not decision.accepted and not decision.pareto_eligible:
                try:
                    active_id = self.models.active_id
                except ModelStoreError:
                    active_id = "M0000"
                if active_id == candidate.id and parent.id in candidates:
                    self.models.set_active(parent.id)
                    parent_system = self._system_for_model(parent.id)
                    if parent_system is not None:
                        self.systems.set_active(parent_system.id)
                    self.events.append(
                        "active_model_demoted",
                        {
                            "model_id": candidate.id,
                            "parent_model_id": parent.id,
                            "reason": "rejected_candidate_before_retention",
                        },
                    )
            retention_outcome = "accepted_candidate" if (decision.accepted or decision.pareto_eligible) else "rejected_candidate"
            retention = self._retain_checkpoint(
                candidate.checkpoint_path,
                candidate.id,
                retention_outcome,
                parent_checkpoint_path=parent.checkpoint_path,
            )
            summary = dict(summary)
            summary["checkpoint_retention"] = retention
            evaluations[child_id] = summary
            self._save_evaluations(evaluations)
            round_gate = self._evaluate_round_gate(record, summary, retention=retention)
            evaluated_experience = self._append_evaluated_experience(record, summary, decision)
            latest_records[child_id] = evaluated_experience
            # Make the newly persisted recipe/evaluation experience visible
            # in this same cycle.  If it waited for the next ``_import()``, a
            # pending boundary plan would see an apparently new observation,
            # discard itself, and spend another remote LLM round-trip before
            # launching the already validated next experiment.
            self._append_observation(self._experience_observation(evaluated_experience))
            plan_trace = self.controller_trace[-1] if self.controller_trace else {}
            self._record_evaluation(
                candidate,
                parent,
                record,
                summary,
                decision,
                fingerprint=str(plan_trace.get("fingerprint", "")),
                repeat_for_statistics=bool(plan_trace.get("plan", {}).get("repeat_for_statistics", False)),
            )
            self.events.append(
                "evaluation_decision",
                {
                    "model_id": child_id,
                    "parent_model_id": parent.id,
                    "status": decision.status,
                    "accepted": decision.accepted,
                    "pareto_eligible": decision.pareto_eligible,
                    "metrics": {
                        "Q": summary.get("quality_score"),
                        "L": _hardware(summary.get("hardware", {})).get("latency_s"),
                        "M": _hardware(summary.get("hardware", {})).get("peak_memory_gb"),
                        "E": _hardware(summary.get("hardware", {})).get("energy_j"),
                    },
                    "violations": list(decision.violations),
                    "checkpoint_retention": retention,
                    "next_loop_reason": "promote_candidate" if decision.accepted else "controller_diagnose_rejected_candidate",
                },
            )
            active_speculative_handles = current_speculative_handles()
            if active_speculative_handles:
                # A speculative child is the next experiment, not a second
                # copy of the candidate just evaluated.  It must remain
                # available for its own benchmark even when its parent is
                # rejected or a reviewer requests a replan.  Deleting it at
                # this boundary leaves a registered candidate with a
                # dangling checkpoint path; the following evaluation then
                # reports a misleading ComfyUI "model not found" failure.
                # The next Controller turn still sees the parent's decision
                # and can reject the child after it has real evidence.
                keep_current_candidate = bool(decision.accepted or decision.pareto_eligible)
                for active_speculative_handle in active_speculative_handles:
                    rooted_at_current = str(active_speculative_handle.get("parent_model_id")) == candidate.id
                    self._finish_speculative_worker(
                        active_speculative_handle,
                        keep=(keep_current_candidate or not rooted_at_current),
                        reason=(
                            "preserve_speculative_child_for_next_evaluation"
                            if keep_current_candidate or not rooted_at_current
                            else "discard_branch_from_rejected_candidate"
                        ),
                    )
            evaluated_count += 1
            # The speculative worker owns a separate state file, but the
            # prefetch keys are cleared concurrently when it starts. Reload
            # before saving this evaluation cursor so a stale in-memory
            # snapshot cannot resurrect a consumed plan.
            state = self._load_campaign_state()
            state["evaluated_ids"] = sorted(set(state.get("evaluated_ids", [])) | {child_id})
            state["current_model_id"] = self.models.active_id
            active_system = self._system_for_model(state["current_model_id"])
            if active_system is not None:
                state["current_system_id"] = active_system.id
            active_round_policy = self._active_round_policy()
            if active_round_policy is not None:
                training_cost = record.training.get("cost") if isinstance(record.training, Mapping) else {}
                if not isinstance(training_cost, Mapping):
                    training_cost = {}
                gpu_hours = training_cost.get(
                    "gpu_hours",
                    record.training.get("gpu_hours", 0.0),
                )
                progress, counted = record_round_policy_trial(
                    active_round_policy,
                    state.get("round_policy_progress"),
                    record.experiment_id,
                    gpu_hours,
                )
                state["round_policy_progress"] = progress
                if counted:
                    self.events.append(
                        "round_policy_trial_recorded",
                        {
                            "round_id": active_round_policy.round_id,
                            "experiment_id": record.experiment_id,
                            "progress": dict(progress),
                        },
                    )
                critical = bool(summary.get("critical_regression"))
                hard_gates = summary.get("hard_gates")
                if isinstance(hard_gates, Mapping):
                    critical = critical or hard_gates.get("no_critical_temporal_collapse") is False
                if critical and "critical_regression" in set(active_round_policy.stop_conditions):
                    progress = self._mark_round_policy_stop(
                        active_round_policy,
                        state,
                        "critical_regression",
                        experiment_id=record.experiment_id,
                    )
                elif round_policy_budget_status(active_round_policy, progress) is not None:
                    progress = self._mark_round_policy_stop(
                        active_round_policy,
                        state,
                        "budget_exhausted",
                        experiment_id=record.experiment_id,
                    )
                state["round_policy_progress"] = progress
            self._save_campaign_state(state)
        retention_reclaimed = self._reclaim_superseded_checkpoints(candidates, evaluations, split)
        benchmark_cache_lease: Dict[str, Any]
        if phase_evaluated_ids and self.config.comfyui_cache_policy in {"idle_release", "cold_cache"}:
            lease_result = self._release_comfyui_if_idle("evaluation_phase_completed")
            benchmark_cache_lease = {
                "state": lease_result.state,
                "success": lease_result.success,
                "reason": lease_result.reason,
                "elapsed_s": lease_result.elapsed_s,
            }
            release_metadata = dict(benchmark_cache_lease)
            for model_id in dict.fromkeys(phase_evaluated_ids):
                saved = dict(evaluations.get(model_id) or {})
                recipe = dict(saved.get("benchmark_recipe") or {})
                recipe["comfyui_lease_release"] = release_metadata
                saved["benchmark_recipe"] = recipe
                evaluations[model_id] = saved
            self._save_evaluations(evaluations)
        elif phase_evaluated_ids:
            benchmark_cache_lease = {
                "state": "reserved_for_benchmark",
                "success": False,
                "reason": "warm_cache_policy",
                "elapsed_s": 0.0,
            }
        elif self.last_comfyui_lease is not None:
            benchmark_cache_lease = {
                "state": self.last_comfyui_lease.state,
                "success": self.last_comfyui_lease.success,
                "reason": self.last_comfyui_lease.reason,
                "elapsed_s": self.last_comfyui_lease.elapsed_s,
            }
        else:
            benchmark_cache_lease = {
                "state": "reserved_for_benchmark",
                "success": False,
                "reason": "no_evaluation_phase",
                "elapsed_s": 0.0,
            }
        if evaluated_count < max_experiments and self.config.worker.enabled:
            parent = self.models.active()
            trained = self._train_one(parent, int(state.get("training_calls", 0)))
            if trained is not None:
                # ``_train_one`` may finish a speculative Controller call in
                # its worker-cleanup ``finally`` block.  That call persists
                # ``prefetched_plan`` after this cycle's state snapshot was
                # loaded, so do not write the stale snapshot back over it.
                latest_state = self._load_campaign_state()
                latest_state["training_calls"] = int(state.get("training_calls", 0)) + 1
                self._save_campaign_state(latest_state)
        elif self.config.worker.enabled:
            # A bounded run still lets the Controller inspect the newly
            # measured evidence, but the validated result must survive until
            # the next training boundary.  Previously this call only updated
            # the audit trace, so the next loop asked the LLM again and could
            # spend several minutes regenerating an equivalent plan while all
            # worker GPUs were idle.
            self._drain_deferred_prefetches()
            latest_state = self._load_campaign_state()
            if self.review_replan_requested:
                self._cancel_deferred_prefetches("review_requested_replan_before_training_boundary")
                self._clear_pending_state(latest_state)
                self._save_campaign_state(latest_state)
                self.events.append(
                    "controller_plan_prefetch_discarded",
                    {"reason": "review_requested_replan_before_training_boundary"},
                )
                # The fresh plan below is already generated after the review;
                # do not let run_loop discard it a second time.
                self.review_replan_requested = False
                latest_state = self._load_campaign_state()
            elif self._deferred_prefetch_handles and not isinstance(latest_state.get("prefetched_plan"), Mapping):
                # The worker/evaluation boundary is the last useful place to
                # wait for the already-issued LLM request.  Waiting here
                # avoids a duplicate main request, while a long GPU worker
                # normally means this path is instantaneous.
                for handle in list(self._deferred_prefetch_handles):
                    self._finish_controller_prefetch(handle)
                self._drain_deferred_prefetches()
                latest_state = self._load_campaign_state()
            has_deferred_plan = isinstance(latest_state.get("pending_plan"), Mapping)
            has_prefetched_plan = isinstance(latest_state.get("prefetched_plan"), Mapping)
            async_launch_thread = speculative_launch_state.get("thread")
            async_launch_inflight = isinstance(async_launch_thread, threading.Thread) and async_launch_thread.is_alive()
            speculative_status = str(self._load_speculative_state().get("status") or "")
            speculative_worker_inflight = speculative_status in {"launching", "running"}
            if has_deferred_plan or has_prefetched_plan or async_launch_inflight or speculative_worker_inflight:
                self.events.append(
                    "controller_plan_boundary_reused",
                    {
                        "reason": "validated_plan_already_waiting_for_training_boundary",
                        "pending_plan": has_deferred_plan,
                        "prefetched_plan": has_prefetched_plan,
                        "async_launch_inflight": async_launch_inflight,
                        "speculative_worker_inflight": speculative_worker_inflight,
                    },
                )
            elif not self.review_stop_requested:
                # A plan produced after evaluation is not speculative: no
                # worker is running yet. Persist it as a pending boundary plan
                # so the next iteration can launch immediately without another
                # Controller round-trip.
                next_plan = self._controller_plan(
                    self.models.active(), int(state.get("training_calls", 0))
                )
                if next_plan is not None:
                    latest_state = self._load_campaign_state()
                    latest_state.update(
                        {
                            "pending_plan": next_plan.to_dict(),
                            "pending_parent_model_id": next_plan.parent_model_id,
                            "pending_training_calls": int(state.get("training_calls", 0)),
                            "pending_attempts": 0,
                            "pending_last_resource_decision": {
                                "status": "deferred",
                                "reason": "bounded_evaluation_window",
                            },
                        }
                    )
                    self._save_campaign_state(latest_state)
                    self.events.append(
                        "controller_plan_queued_for_training_boundary",
                        {
                            "experiment_id": next_plan.experiment_id,
                            "parent_model_id": next_plan.parent_model_id,
                            "training_calls": int(state.get("training_calls", 0)),
                            "reason": "bounded_evaluation_window",
                        },
                    )
        report_metrics = {}
        for child_id, summary in evaluations.items():
            if child_id == "M0000":
                continue
            hardware = _hardware(summary.get("hardware", {}))
            report_metrics[child_id] = {
                "Q": summary.get("quality_score"),
                "L": hardware.get("latency_s"),
                "M": hardware.get("peak_memory_gb"),
                "E": hardware.get("energy_j"),
                "status": latest_records.get(child_id).status if child_id in latest_records else "evaluated_candidate",
            }
        accepted = [item for item in latest_records.values() if item.status == "accepted"]
        report = {
            "imported": imported.imported,
            "imported_duplicates": imported.skipped_duplicates,
            "corrupt_imports": list(imported.corrupt),
            "metrics": report_metrics,
            "goal": self._goal_payload(),
            "quality_scope": self.config.quality_scope,
            "research_grade": self.config.research_grade,
            "benchmark_cache_policy": self.config.comfyui_cache_policy,
            "benchmark_cache_lease": benchmark_cache_lease,
            "checkpoint_retention_policy": self.config.checkpoint_retention_policy,
            "max_retained_checkpoints": self.config.max_retained_checkpoints,
            "checkpoint_retention_reconciled": list(retention_reconciled),
            "checkpoint_retention_reclaimed": list(retention_reclaimed),
            "pipeline": self._pipeline_telemetry(),
            "controller": {
                "calls": sum(1 for item in self.controller_trace if item.get("controller_call", True))
                + sum(1 for item in self.controller_prefetch_trace if item.get("controller_call", True)),
                "trace": list(self.controller_trace),
                "prefetch_trace": list(self.controller_prefetch_trace),
            },
            "pareto_front": [entry.to_dict() for entry in self.pareto.front()],
            "system_lineage": [item.to_dict() for item in self.systems.lineage()],
            "claim_boundary": "structural_proxy is not semantic video quality",
        }
        status = "accepted" if accepted else "completed"
        return CampaignResult(status, self.models.active_id, latest_records, report)

    def _remote_comfyui_runtime_evidence(self, worker: Optional[Any] = None) -> Mapping[str, Any]:
        """Read bounded startup evidence for ComfyUI's DynamicVRAM mode."""

        if not isinstance(self.ssh, SSHClient):
            return {}
        port = int(
            getattr(worker, "port", self.config.remote.comfyui_port)
            if worker is not None
            else self.config.remote.comfyui_port
        )
        process = self.ssh.run(
            ("pgrep", "-f", "main.py.*--port[ =]%d" % port),
            check=False,
        )
        pid = next(
            (
                int(value.strip())
                for value in str(getattr(process, "stdout", "")).splitlines()
                if value.strip().isdigit() and int(value.strip()) > 0
            ),
            None,
        )
        if pid is None:
            return {"dynamic_vram_enabled": False, "runtime_probe": "comfyui_pid_missing"}
        command_probe = self.ssh.run(("ps", "-p", str(pid), "-o", "args="), check=False)
        command = str(getattr(command_probe, "stdout", "")).strip()
        disabled_flags = (
            "--disable_dynamic_vram",
            "--highvram",
            "--gpu-only",
            "--novram",
            "--cpu",
        )
        cli_enabled = not any(flag in command for flag in disabled_flags)
        log_detected = False
        output_link = self.ssh.run(("readlink", "/proc/%d/fd/1" % pid), check=False)
        log_path = str(getattr(output_link, "stdout", "")).strip()
        if log_path.startswith("/") and "\n" not in log_path and "\x00" not in log_path:
            log_probe = self.ssh.run(
                (
                    "grep",
                    "-F",
                    "-m",
                    "1",
                    "DynamicVRAM support detected and enabled",
                    log_path,
                ),
                check=False,
            )
            log_detected = int(getattr(log_probe, "returncode", 1)) == 0
        return {
            "port": port,
            "dynamic_vram_enabled": bool(cli_enabled and log_detected),
            "cli_enabled": bool(cli_enabled),
            "startup_log_detected": bool(log_detected),
            "runtime_probe": "process_args_and_startup_log",
        }

    def _remote_comfyui_worker_capabilities(
        self,
        worker: Any,
    ) -> Optional[Mapping[str, Any]]:
        """Probe the live node registry for one exact ComfyUI evaluator lane."""

        if not isinstance(self.ssh, SSHClient):
            return None
        port = int(worker.port)
        object_info_endpoint = "http://127.0.0.1:%d/object_info" % port
        object_info = self.ssh.run(
            ("curl", "-fsS", "--max-time", "5", object_info_endpoint),
            check=False,
        )
        live_node_classes: Optional[Mapping[str, Any]] = None
        object_info_status = "unavailable"
        if int(getattr(object_info, "returncode", 1)) == 0:
            try:
                parsed_object_info = json.loads(str(getattr(object_info, "stdout", "")))
                if isinstance(parsed_object_info, Mapping):
                    live_node_classes = parsed_object_info
                    object_info_status = "ready"
            except (TypeError, ValueError, json.JSONDecodeError):
                object_info_status = "invalid_json"
        capabilities = probe_minimax_h3_capabilities(
            self.config.workflow.template,
            self.config.remote.comfyui_root,
            live_node_classes=live_node_classes if live_node_classes is not None else (),
            runtime_evidence=self._remote_comfyui_runtime_evidence(worker),
        )
        self._last_comfyui_object_info_status = object_info_status
        self.events.append(
            "comfyui_worker_capabilities_probed",
            {
                "gpu_index": int(worker.gpu_index),
                "port": port,
                "object_info_endpoint": object_info_endpoint,
                "object_info_status": object_info_status,
                "h3_optimization_capabilities": {
                    name: {
                        key: value
                        for key, value in capability.items()
                        if key in {
                            "status",
                            "safe_to_plan",
                            "live_node_registered",
                            "extension_installed",
                            "runtime_confirmed",
                            "execution_contract",
                            "workflow_hook_ready",
                        }
                    }
                    for name, capability in capabilities.items()
                    if isinstance(capability, Mapping)
                },
            },
        )
        return capabilities

    def _remote_comfyui_preflight(self) -> None:
        """Check the benchmark backend before touching a pending candidate.

        A real remote campaign may be queued while either the Controller or
        ComfyUI is occupying/waiting for a GPU.  Check the backend before
        ``run()`` so a stopped service does not turn a resumable queue item
        into a failed evaluation attempt.  The actual ComfyUI lease is only
        written by ``_evaluate`` immediately before benchmark work; planning
        and training must remain free to use GPU 0.  Test doubles
        intentionally skip this transport-only check.
        """

        if not isinstance(self.ssh, SSHClient):
            return
        endpoint = "http://127.0.0.1:%d/system_stats" % int(self.config.remote.comfyui_port)
        result = self.ssh.run(("curl", "-fsS", "--max-time", "5", endpoint), check=False)
        if int(getattr(result, "returncode", 1)) != 0:
            detail = str(getattr(result, "stderr", "")).strip()[-500:]
            raise RemoteError("remote ComfyUI is unavailable at %s%s" % (endpoint, (": " + detail) if detail else ""))
        # Disk installation is not enough: an already-running external
        # ComfyUI process may not have loaded the custom node yet.  Query the
        # live registry and gate LPL/TDTM on the exact process that will
        # execute the benchmark workflow.  If this auxiliary endpoint is
        # unavailable, keep normal benchmarking available but fail closed for
        # the optional optimization nodes.
        object_info_endpoint = "http://127.0.0.1:%d/object_info" % int(self.config.remote.comfyui_port)
        self.optimization_capabilities = self._remote_comfyui_worker_capabilities(
            self.config.comfyui_workers[0]
        ) or probe_minimax_h3_capabilities(
            self.config.workflow.template,
            self.config.remote.comfyui_root,
            live_node_classes=(),
            runtime_evidence=self._remote_comfyui_runtime_evidence(),
        )
        object_info_status = getattr(
            self,
            "_last_comfyui_object_info_status",
            "unavailable",
        )
        self.events.append(
            "benchmark_preflight",
            {
                "provider": "comfyui",
                "endpoint": endpoint,
                "status": "ready",
                "object_info_endpoint": object_info_endpoint,
                "object_info_status": object_info_status,
                "h3_optimization_capabilities": {
                    name: {
                        key: value
                        for key, value in capability.items()
                        if key in {
                            "status",
                            "safe_to_plan",
                            "live_node_registered",
                            "extension_installed",
                            "runtime_confirmed",
                            "execution_contract",
                            "workflow_hook_ready",
                        }
                    }
                    for name, capability in self.optimization_capabilities.items()
                    if isinstance(capability, Mapping)
                },
            },
        )

    def run_loop(
        self,
        resume: bool = True,
        max_iterations: int = 4,
        split: Optional[str] = None,
        resource_poll_interval_s: float = 5.0,
        stop_file: Optional[Path] = None,
    ) -> CampaignResult:
        """Run repeated Controller -> worker -> evaluation cycles.

        ``run`` is one idempotent import/evaluate/train cycle.  This method is
        the continuous loop: every completed optimization cycle re-reads
        measured evidence and asks the Controller for the next operation.
        Temporary Controller, ComfyUI, or worker-resource outages are wait
        states and do not consume the caller's optimization-iteration bound.
        The loop stops only when the explicit target is satisfied, a
        non-retryable Controller decision is reached, or the caller's bound
        of completed optimization cycles is reached.
        """
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if resource_poll_interval_s < 0:
            raise ValueError("resource_poll_interval_s must be non-negative")
        self._controller_iteration_budget = max(
            int(self._controller_iteration_budget), int(max_iterations)
        )
        cycles: List[Mapping[str, Any]] = []
        dependency_waits: List[Mapping[str, Any]] = []
        controller_traces: List[Mapping[str, Any]] = []
        controller_prefetch_traces: List[Mapping[str, Any]] = []
        result: Optional[CampaignResult] = None
        stop_requested = False
        completed_iterations = 0
        # A controller/resource outage is a wait state, not an optimization
        # iteration. Keep polling until the explicit stop request, goal, or
        # max_iterations completed optimization cycles is reached.
        for index in itertools.count():
            if stop_file is not None and Path(stop_file).exists():
                speculative_state = self._recover_speculative_worker()
                speculative_status = str(speculative_state.get("status") or "")
                if speculative_status in {"launching", "running"}:
                    # A clean service handoff must not release the campaign
                    # lock while a remote successor still owns its GPU lease.
                    # The durable result JSON is the completion boundary;
                    # resume the same check without starting another cycle.
                    self.events.append(
                        "campaign_stop_deferred",
                        {
                            "iteration": index + 1,
                            "reason": "speculative_worker_still_running",
                            "experiment_id": speculative_state.get("experiment_id"),
                            "child_model_id": speculative_state.get("child_model_id"),
                        },
                    )
                    time.sleep(max(1.0, min(60.0, float(resource_poll_interval_s or 5.0))))
                    continue
                stop_requested = True
                self.events.append(
                    "campaign_stop_requested",
                    {
                        "iteration": index + 1,
                        "reason": "stop_file_present_before_iteration",
                        "stop_file": str(stop_file),
                    },
                )
                break
            policy_stop = self._round_policy_terminal_reason()
            if policy_stop is not None:
                try:
                    policy_model_id = self.models.active_id
                except ModelStoreError:
                    policy_model_id = "M0000"
                result = CampaignResult(
                    "round_policy_stopped",
                    policy_model_id,
                    {},
                    {
                        "controller": {"calls": 0, "trace": []},
                        "round_policy_stop_reason": str(policy_stop),
                    },
                )
                self.events.append(
                    "loop_round_policy_stopped",
                    {
                        "iteration": index + 1,
                        "model_id": policy_model_id,
                        "reason": str(policy_stop),
                    },
                )
                break
            try:
                current_model_id = self.models.active_id
            except ModelStoreError:
                # A dry loop/test double may provide ``run`` without first
                # materializing the model store; the real campaign initializes
                # it during the first cycle.
                current_model_id = "M0000"
            retry_state = self._load_campaign_state()
            retrying_held_worker = bool(
                isinstance(retry_state.get("pending_plan"), Mapping)
                and Path(self._controller_handoff_hold_path()).exists()
            )
            if str(getattr(self.controller, "provider_name", "")) == "vllm":
                try:
                    # Do this before the full cycle.  A remote campaign may
                    # wait for a dynamically selected GPU group for hours;
                    # importing all history and reconciling checkpoints while
                    # vLLM is known to be offline only creates redundant IO.
                    if retrying_held_worker:
                        # A pending distributed worker deliberately holds the
                        # Controller lane while the previous vLLM process
                        # drains.  The next cycle must retry the scheduler,
                        # not fail early because the Controller is correctly
                        # offline during this handoff.
                        self.events.append(
                            "loop_controller_preflight_bypassed",
                            {
                                "reason": "pending_worker_resource_retry_holds_controller_lane",
                                "pending_experiment_id": retry_state.get("pending_plan", {}).get("experiment_id"),
                            },
                        )
                    else:
                        self._remote_controller_preflight()
                except ControllerUnavailableError as exc:
                    trace = {
                        "controller_call": False,
                        **self._controller_provider_metadata(),
                        "status": "controller_unavailable",
                        "preflight": True,
                        "error": str(exc)[:2000],
                        "retry_after_s": float(resource_poll_interval_s),
                    }
                    result = CampaignResult(
                        "controller_unavailable",
                        current_model_id,
                        {},
                        {"controller": {"calls": 0, "trace": [trace]}},
                    )
                    self.events.append(
                        "loop_dependency_unavailable",
                        {
                            "dependency": "controller",
                            "iteration": index + 1,
                            "endpoint": "http://127.0.0.1:%d/v1/models"
                            % int(getattr(self.controller, "remote_port", 8000)),
                            "model": str(getattr(self.controller, "model_name", "")),
                            "error": trace["error"],
                            "retry_after_s": float(resource_poll_interval_s),
                        },
                    )
                    dependency_waits.append(
                        {
                            "attempt": index + 1,
                            "status": result.status,
                            "current_model_id": current_model_id,
                            "dependency": "controller",
                            "error": trace["error"],
                        }
                    )
                    controller_traces.append(trace)
                    self._set_pipeline_waiting(
                        "controller",
                        current_model_id=current_model_id,
                    )
                    if resource_poll_interval_s:
                        time.sleep(resource_poll_interval_s)
                    continue
            try:
                self._remote_comfyui_preflight()
            except RemoteError as exc:
                result = CampaignResult(
                    "remote_unavailable",
                    current_model_id,
                    {},
                    {"controller": {"calls": 0, "trace": list(self.controller_trace)}},
                )
                self.events.append(
                    "loop_dependency_unavailable",
                    {
                        "dependency": "comfyui",
                        "iteration": index + 1,
                        "endpoint": "http://127.0.0.1:%d/system_stats" % int(self.config.remote.comfyui_port),
                        "error": str(exc),
                    },
                )
                dependency_waits.append(
                    {
                        "attempt": index + 1,
                        "status": result.status,
                        "current_model_id": current_model_id,
                        "dependency": "comfyui",
                        "error": str(exc),
                    }
                )
                self._set_pipeline_waiting(
                    "comfyui",
                    current_model_id=current_model_id,
                )
                if resource_poll_interval_s:
                    time.sleep(resource_poll_interval_s)
                continue
            self.events.append(
                "loop_iteration_started",
                {
                    "iteration": index + 1,
                    "max_iterations": max_iterations,
                    "goal": self._goal_payload(),
                    "current_model_id": current_model_id,
                },
            )
            try:
                result = self.run(
                    resume=resume if index == 0 else True,
                    max_experiments=1,
                    split=split,
                )
            except RemoteResourceWaitError as exc:
                # An evaluator GPU can become unsafe after ComfyUI preflight
                # (for example, an external Geneval process may enter GPU0).
                # Keep the candidate pending and classify this as a resource
                # dependency so it does not consume an optimization cycle.
                error = str(exc)[:2000]
                trace = {
                    "controller_call": False,
                    "status": "remote_resource_wait",
                    "execution_status": "waiting_for_resources",
                    "error": error,
                }
                result = CampaignResult(
                    "iteration_failed",
                    current_model_id,
                    {},
                    {
                        "controller": {"calls": 0, "trace": [trace]},
                        "iteration_failure": error,
                    },
                )
                self.events.append(
                    "loop_iteration_waiting",
                    {
                        "iteration": index + 1,
                        "current_model_id": current_model_id,
                        "dependency": "worker_resources",
                        "failure_type": "no_safe_comfyui_evaluator_gpu",
                        "error": error,
                        "retry_after_s": float(resource_poll_interval_s),
                    },
                )
            except (RemoteError, OSError, TimeoutError, subprocess.TimeoutExpired) as exc:
                # A detached overnight process must survive a transient
                # evaluator/transport failure. ``run()`` already releases
                # any ComfyUI lease in its finally block; the next cycle
                # re-imports the immutable result stream and retries the
                # pending evaluation or asks the Controller for a new plan.
                error = str(exc)[:2000]
                self.events.append(
                    "loop_iteration_failed",
                    {
                        "iteration": index + 1,
                        "current_model_id": current_model_id,
                        "failure_type": "recoverable_remote_error",
                        "error": error,
                        "retry_after_s": float(resource_poll_interval_s),
                    },
                )
                self._review_now(
                    "iteration",
                    "iteration_failed",
                    {
                        "status": "failed",
                        "failure_type": "recoverable_remote_error",
                        "error": error,
                    },
                )
                result = CampaignResult(
                    "iteration_failed",
                    current_model_id,
                    {},
                    {
                        "controller": {"calls": 0, "trace": list(self.controller_trace)},
                        "iteration_failure": error,
                    },
                )
            cycle_trace = result.report.get("controller", {}).get("trace", [])
            controller_traces.extend(item for item in cycle_trace if isinstance(item, Mapping))
            cycle_prefetch_trace = result.report.get("controller", {}).get("prefetch_trace", [])
            controller_prefetch_traces.extend(item for item in cycle_prefetch_trace if isinstance(item, Mapping))
            last_trace = cycle_trace[-1] if cycle_trace and isinstance(cycle_trace[-1], Mapping) else {}
            execution_status = last_trace.get("execution_status")
            dependency_name = (
                "controller"
                if result.status == "controller_unavailable"
                or last_trace.get("status") == "controller_unavailable"
                else "comfyui"
                if result.status == "remote_unavailable"
                or last_trace.get("status") == "remote_unavailable"
                else "worker_resources"
                if execution_status in {"waiting_for_resources", "resource_unavailable_replan"}
                else None
            )
            if dependency_name is not None:
                self._set_pipeline_waiting(
                    dependency_name,
                    current_model_id=result.current_model_id,
                    experiment_id=last_trace.get("experiment_id"),
                )
                dependency_waits.append(
                    {
                        "attempt": index + 1,
                        "status": result.status,
                        "current_model_id": result.current_model_id,
                        "dependency": dependency_name,
                        "error": last_trace.get("error")
                        or result.report.get("iteration_failure"),
                        "resource_decision": last_trace.get("resource_decision"),
                    }
                )
            else:
                completed_iterations += 1
                cycles.append(
                    {
                        "iteration": completed_iterations,
                        "status": result.status,
                        "current_model_id": result.current_model_id,
                        "controller_calls": result.report.get("controller", {}).get("calls", 0),
                        "goal_satisfied": self._goal_satisfied(split),
                    }
                )
            self.events.append(
                "loop_iteration_completed",
                {
                    "iteration": index + 1,
                    "optimization_iteration": completed_iterations,
                    "status": result.status,
                    "current_model_id": result.current_model_id,
                    "goal_satisfied": self._goal_satisfied(split),
                    "controller_calls": result.report.get("controller", {}).get("calls", 0),
                },
            )
            if stop_file is not None and Path(stop_file).exists():
                speculative_state = self._recover_speculative_worker()
                speculative_status = str(speculative_state.get("status") or "")
                if speculative_status in {"launching", "running"}:
                    self.events.append(
                        "campaign_stop_deferred",
                        {
                            "iteration": index + 1,
                            "reason": "speculative_worker_still_running",
                            "experiment_id": speculative_state.get("experiment_id"),
                            "child_model_id": speculative_state.get("child_model_id"),
                        },
                    )
                    time.sleep(max(1.0, min(60.0, float(resource_poll_interval_s or 5.0))))
                    continue
                stop_requested = True
                self.events.append(
                    "campaign_stop_boundary_reached",
                    {
                        "iteration": index + 1,
                        "reason": "stop_file_present_after_iteration",
                        "stop_file": str(stop_file),
                    },
                )
                break
            self._mark_round_policy_target_if_satisfied(split)
            if self._round_policy_terminal_reason() is not None:
                break
            if result.status == "iteration_failed":
                if self.review_stop_requested:
                    break
                if completed_iterations < max_iterations:
                    if resource_poll_interval_s:
                        time.sleep(resource_poll_interval_s)
                    continue
                break
            if self.review_stop_requested:
                break
            if self._goal_satisfied(split):
                break
            if not self.config.worker.enabled:
                break
            trace = result.report.get("controller", {}).get("trace", [])
            if completed_iterations >= max_iterations:
                break
            if result.status in {"controller_unavailable", "remote_unavailable"} or (
                trace and trace[-1].get("status") == "controller_unavailable"
            ):
                if completed_iterations < max_iterations:
                    if resource_poll_interval_s:
                        time.sleep(resource_poll_interval_s)
                    continue
                break
            if trace and trace[-1].get("execution_status") in {
                "waiting_for_resources",
                "resource_unavailable_replan",
            }:
                unavailable_policy = trace[-1].get("plan", {}).get("resource_request", {}).get("on_unavailable")
                if completed_iterations < max_iterations and unavailable_policy in {"wait", "replan"}:
                    if resource_poll_interval_s:
                        time.sleep(resource_poll_interval_s)
                    continue
                break
            if self.review_replan_requested:
                state = self._load_campaign_state()
                self._clear_pending_state(state)
                self._save_campaign_state(state)
                self.review_replan_requested = False
                if completed_iterations < max_iterations:
                    continue
                break
            if (
                str(getattr(self.controller, "provider_name", "")) == "vllm"
                and trace
                and trace[-1].get("status") == "rejected"
            ):
                # A real LLM can return a schema-valid plan that fails a
                # harness safety/evidence rule (for example, omitting one
                # newly appended observation ID).  Treat that as a bounded
                # replan opportunity instead of terminating a long campaign
                # after its first malformed decision.
                self.events.append(
                    "controller_plan_rejected_retry",
                    {
                        "iteration": index + 1,
                        "next_iteration": index + 2,
                        "error": str(trace[-1].get("error", ""))[:2000],
                    },
                )
                if completed_iterations < max_iterations:
                    if resource_poll_interval_s:
                        time.sleep(resource_poll_interval_s)
                    continue
                break
            if not trace or trace[-1].get("status") != "validated":
                # ``run(max_experiments=1)`` can finish an evaluation while
                # the next plan is already persisted by the training-time
                # prefetch thread.  That cycle legitimately has no current
                # synchronous Controller trace, but it must not terminate a
                # long campaign before the following boundary consumes the
                # validated plan.  The next iteration will enter
                # ``_train_one`` and launch it without another LLM roundtrip.
                state = self._load_campaign_state()
                has_next_plan = isinstance(state.get("pending_plan"), Mapping) or isinstance(
                    state.get("prefetched_plan"), Mapping
                )
                if has_next_plan and completed_iterations < max_iterations:
                    self.events.append(
                        "loop_continuation_pending_plan",
                        {
                            "iteration": index + 1,
                            "next_iteration": index + 2,
                            "pending_plan": isinstance(state.get("pending_plan"), Mapping),
                            "prefetched_plan": isinstance(state.get("prefetched_plan"), Mapping),
                            "reason": "validated_plan_waiting_for_next_training_boundary",
                        },
                    )
                    if resource_poll_interval_s:
                        time.sleep(resource_poll_interval_s)
                    continue
                break
        if result is None:
            # A stop request may already be present when a detached service is
            # restarted.  Return a durable terminal result instead of
            # asserting after the loop performed no cycle.
            try:
                stopped_model_id = self.models.active_id
            except ModelStoreError:
                stopped_model_id = "M0000"
            result = CampaignResult(
                "stop_requested",
                stopped_model_id,
                {},
                {"controller": {"calls": 0, "trace": []}},
            )
        report = dict(result.report)
        report["goal"] = self._goal_payload()
        report["pipeline"] = self._pipeline_telemetry()
        report["controller"] = {
            "calls": sum(1 for item in controller_traces if item.get("controller_call", True))
            + sum(1 for item in controller_prefetch_traces if item.get("controller_call", True)),
            "trace": controller_traces,
            "prefetch_trace": controller_prefetch_traces,
        }
        active_round_policy = self._active_round_policy()
        if active_round_policy is not None:
            policy_state = self._load_campaign_state()
            report["round_policy"] = {
                "round_id": active_round_policy.round_id,
                "progress": self._round_policy_progress(active_round_policy, policy_state),
                "budget_status": self._round_policy_budget_status(active_round_policy, policy_state),
            }
        last_cycle = cycles[-1] if cycles else {}
        policy_stop_reason = self._round_policy_terminal_reason()
        stop_reason = (
            "stop_requested"
            if stop_requested
            else
            "round_policy_stopped"
            if policy_stop_reason is not None
            else
            "review_stop"
            if self.review_stop_requested
            else "target_satisfied"
            if self._goal_satisfied(split)
            else "remote_dependency_unavailable"
            if last_cycle.get("dependency") == "comfyui"
            or (result is not None and result.status == "remote_unavailable")
            else "controller_unavailable"
            if last_cycle.get("dependency") == "controller"
            or (controller_traces and controller_traces[-1].get("status") == "controller_unavailable")
            else "resources_unavailable"
            if controller_traces and controller_traces[-1].get("execution_status") == "waiting_for_resources"
            else "iteration_bound_or_no_valid_plan"
        )
        report["loop"] = {
            "enabled": True,
            "iterations": len(cycles),
            "max_iterations": max_iterations,
            "cycles": cycles,
            "dependency_waits": dependency_waits,
            "stop_reason": stop_reason,
        }
        report["controller_reviews"] = {
            "calls": self.review_calls,
            "trace": list(self.review_trace),
            "stop_requested": self.review_stop_requested,
            "replan_requested": self.review_replan_requested,
        }
        self.events.append(
            "campaign_completed",
            {
                "status": report.get("loop", {}).get("stop_reason"),
                "current_model_id": result.current_model_id,
                "iterations": len(cycles),
                "goal_satisfied": self._goal_satisfied(split),
            },
        )
        return CampaignResult(
            "target_satisfied"
            if stop_reason == "target_satisfied"
            else stop_reason
            if stop_reason in {"controller_unavailable", "resources_unavailable", "stop_requested"}
            else result.status,
            result.current_model_id,
            result.records,
            report,
        )


def build_campaign_from_config(path: Path, **kwargs: Any) -> RemoteCampaign:
    return RemoteCampaign.from_config_path(path, **kwargs)
