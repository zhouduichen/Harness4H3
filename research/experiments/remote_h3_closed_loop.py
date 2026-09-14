"""SSH-backed MiniMax-H3 experience import, evaluation, and promotion loop.

This module is deliberately a coordinator. The Controller only proposes an
operator; trusted remote configuration owns every executable, path, threshold,
and benchmark setting.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.model_store import ModelCandidateExists, ModelStore
from harness4h3.archive.pareto import ParetoArchive
from harness4h3.backends.comfyui import MiniMaxH3Adapter
from harness4h3.benchmark.h3 import BenchmarkSummary, H3BenchmarkRunner
from harness4h3.config import Target, WorkflowConfig
from harness4h3.controller.context import ControllerContext
from harness4h3.controller.policy import ValidationPipeline
from harness4h3.controller.provider import ControllerProvider, RuleBasedMockController
from harness4h3.controller.schemas import BudgetState, EvaluationResult, HardwareMetrics
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.h3.state import ModelState
from harness4h3.harness.state import Task, load_tasks
from harness4h3.memory.experience import ExperienceRecord, ExperienceStore
from harness4h3.memory.experiment_store import ExperimentRecord, ExperimentStore
from harness4h3.remote.config import RemoteCampaignConfig, load_remote_campaign_config
from harness4h3.remote.decision import AcceptanceInput, DecisionResult, decide
from harness4h3.remote.importer import ImportSummary, RemoteResultImporter
from harness4h3.remote.power import RemotePowerSampler
from harness4h3.remote.ssh import ComfyUITunnel, RemoteError, SSHClient
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


def _summary_dict(summary: Any) -> Dict[str, Any]:
    if isinstance(summary, BenchmarkSummary):
        return summary.to_dict()
    if hasattr(summary, "to_dict"):
        raw = summary.to_dict()
        return dict(raw) if isinstance(raw, Mapping) else {}
    return dict(summary) if isinstance(summary, Mapping) else {}


def _hardware(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, HardwareMetrics):
        return asdict(raw)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        name: getattr(raw, name, None)
        for name in ("latency_s", "peak_memory_gb", "model_size_gb", "energy_j", "throughput", "thermal")
    }


def _evaluation_result(summary: Mapping[str, Any]) -> Optional[EvaluationResult]:
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
    )


def _state_digest(state: ModelState) -> str:
    return hashlib.sha256(json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


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
        self.controller = controller or RuleBasedMockController()
        self.ssh = ssh or SSHClient(config.remote)
        self.target = target or load_target_profile(config.runtime.target_path)
        self.output_root = Path(output_root or config.runtime.output_root).resolve()
        self.experience_path = config.runtime.experience_path if output_root is None else self.output_root / "experience.jsonl"
        self.experience = ExperienceStore(self.experience_path)
        self.experiments = ExperimentStore(self.output_root / "experiments.jsonl")
        self.models = ModelStore(self.output_root / "models")
        self.pareto = ParetoArchive(self.output_root / "pareto")
        self.evaluations_path = self.output_root / "evaluations.json"
        self.campaign_state_path = self.output_root / "campaign_state.json"
        self.evaluator_factory = evaluator_factory
        self.benchmark_factory = benchmark_factory
        self.tunnel_factory = tunnel_factory or (lambda client: ComfyUITunnel(client))
        self.validation = ValidationPipeline()
        self.registry = build_model_evolution_registry()
        self.controller_trace: List[Dict[str, Any]] = []

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
        state = ModelState(
            model_id="M0000",
            parent_model_id=None,
            checkpoint_path=str(source_checkpoint or checkpoint),
            architecture_name=str(child_state.get("architecture_name") or "MiniMax-H3-FL2VA"),
            parameter_count=child_state.get("parameter_count"),
            dtype=str(child_state.get("dtype") or "bfloat16"),
            sampling_steps=int(child_state.get("algorithm_state", {}).get("source_steps", 32)) if isinstance(child_state.get("algorithm_state"), Mapping) else 32,
            components=copy.deepcopy(dict(child_state.get("components") or {})),
            algorithm_state={"imported_root": True},
            runtime_state={"remote_host": self.config.remote.host},
            measured_metrics={},
            provenance={"remote_root_inferred": True, "source_checkpoint": str(source_checkpoint or checkpoint)},
        )
        return state

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
            return {"evaluated_ids": [], "training_calls": 0, "current_model_id": "M0000"}
        raw = json.loads(self.campaign_state_path.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, Mapping) else {"evaluated_ids": [], "training_calls": 0, "current_model_id": "M0000"}

    def _save_campaign_state(self, value: Mapping[str, Any]) -> None:
        _atomic_json(self.campaign_state_path, value)

    def _import(self) -> ImportSummary:
        importer = RemoteResultImporter(self.experience, self.ssh)
        return importer.discover_remote(root=self.config.remote.results_root)

    def _register_candidates(self, records: Sequence[ExperienceRecord]) -> Dict[str, ModelCandidate]:
        first = next((item for item in records if item.child_model_id), None)
        root_state = self._root_state(first)
        self.models.initialize(
            ModelCandidate("M0000", None, 0, root_state.checkpoint_path, root_state, None, "baseline", {"remote": self.config.remote.host})
        )
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
                    pending.remove(record)
                    progressed = True
                    continue
                state_raw = dict(record.provenance.get("output_state") or {})
                state_raw.update({"model_id": child_id, "parent_model_id": parent_id})
                state_raw.setdefault("checkpoint_path", record.provenance.get("remote_checkpoint_path"))
                state_raw.setdefault("architecture_name", "MiniMax-H3-FL2VA")
                state_raw.setdefault("dtype", "bfloat16")
                state_raw.setdefault("sampling_steps", record.operator_args.get("target_steps"))
                state_raw.setdefault("provenance", {})
                state_raw["provenance"] = {**dict(state_raw.get("provenance") or {}), "experience_id": record.experience_id, "remote_source_uri": record.source_uri}
                if not state_raw.get("checkpoint_path"):
                    pending.remove(record)
                    progressed = True
                    continue
                state = ModelState.from_dict(state_raw)
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
        return candidates

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
        backend = MiniMaxH3Adapter(base_url=base_url, task_timeout_s=3600.0)
        evaluator = SubprocessEvaluator(self.config.evaluator_command, self.config.evaluator_timeout_s)
        return H3BenchmarkRunner(
            backend,
            evaluator,
            json.loads(self.config.workflow.template.read_text(encoding="utf-8")),
            self.config.workflow,
            self.output_root / "benchmark",
            system_sample_interval_s=self.config.sampling_interval_s,
        )

    def _evaluate(self, candidate: ModelCandidate, parent_summary: Optional[Mapping[str, Any]], split: Optional[str]) -> Dict[str, Any]:
        tasks = self._tasks(split)
        if not tasks:
            raise ValueError("no tasks selected for remote campaign")
        grouped: List[Tuple[str, List[Task]]] = []
        for name in self.config.benchmark_splits if split in (None, "all") else (split,):
            group = [task for task in tasks if task.split == name]
            if group:
                grouped.append((name, group))
        parent_quality = parent_summary.get("quality_score") if parent_summary else None
        parent_hardware = HardwareMetrics(**{key: value for key, value in _hardware(parent_summary.get("hardware", {})).items() if key in HardwareMetrics.__dataclass_fields__}) if parent_summary else None
        with self.tunnel_factory(self.ssh) as tunnel:
            runner = self._make_runner(tunnel.base_url)
            summaries: List[Dict[str, Any]] = []
            for name, group in grouped:
                self.ssh.ensure_model_link(candidate.checkpoint_path, candidate.id) if candidate.id != "M0000" else None
                power = RemotePowerSampler(self.ssh, self.config.sampling_interval_s)
                summary = _summary_dict(
                    runner.run(
                        candidate.state,
                        group,
                        baseline_quality=float(parent_quality) if parent_quality is not None else None,
                        target=self.target,
                        baseline_hardware=parent_hardware,
                        efficiency_thresholds=self.config.efficiency_thresholds,
                        black_frame_rate_threshold=self.config.black_frame_rate_threshold,
                        reset_backend_before_run=self.config.reset_backend_before_run,
                        power_sampler=power,
                    )
                )
                summaries.append(summary)
                hard_gates = summary.get("hard_gates") or {}
                if name == "sanity" and not all(hard_gates.get(key) is True for key in ("generation_valid", "decode_success", "no_critical_temporal_collapse")):
                    break
        aggregated = self._aggregate_summaries(candidate.id, summaries)
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
        for summary in summaries:
            runs.extend(summary.get("runs") or [])
            if isinstance(summary.get("quality_score"), (int, float)):
                scores.append(float(summary["quality_score"]))
            hardware_values.append(_hardware(summary.get("hardware", {})))
            quality_metrics.append(summary.get("quality_metrics") or {})
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

    def _record_evaluation(
        self,
        candidate: ModelCandidate,
        parent: ModelCandidate,
        record: ExperienceRecord,
        summary: Mapping[str, Any],
        decision: DecisionResult,
    ) -> None:
        evaluation = _evaluation_result(summary)
        front_ids = [entry.candidate_id for entry in self.pareto.front()]
        if evaluation is not None and decision.pareto_eligible:
            if candidate.id not in {entry.candidate_id for entry in self.pareto.entries()}:
                front_ids = [entry.candidate_id for entry in self.pareto.update(candidate.id, evaluation)]
            else:
                front_ids = [entry.candidate_id for entry in self.pareto.front()]
            self.models.set_active(candidate.id)
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
        """Adapt a fixed plan to the real worker's step-distill contract."""

        normalized = dict(args)
        if operator == "step_distill":
            source_steps = int(parent.state.sampling_steps or 32)
            target_steps = int(normalized.get("target_steps", 0))
            if source_steps <= 1 or source_steps % 2:
                raise ValueError("step_distill requires an even positive parent sampling_steps")
            if source_steps != 2 * target_steps:
                normalized["target_steps"] = source_steps // 2
        return normalized

    def _controller_context(self, current: ModelCandidate, training_calls: int) -> ControllerContext:
        recent = [item.to_dict() for item in list(self.experience.read())[-16:]]
        failures = [item for item in recent if item.get("status") in {"failed", "rejected"}]
        budget = BudgetState(
            max_iterations=max(1, self.config.worker.max_steps),
            max_failed_experiments=max(1, self.config.worker.max_steps),
            used_iterations=training_calls,
            used_controller_calls=training_calls,
        )
        visible = tuple(item for item in self.registry.visible() if item["name"] in set(self.config.worker.allowed_operators))
        return ControllerContext(self.target, current.state, budget, visible, recent, failures, [entry.to_dict() for entry in self.pareto.front()])

    def _controller_plan(self, parent: ModelCandidate, training_calls: int) -> Optional[Any]:
        """Ask the Controller for the next plan and retain an auditable trace."""

        context = self._controller_context(parent, training_calls)
        trace: Dict[str, Any] = {
            "parent_model_id": parent.id,
            "training_calls": training_calls,
            "input_metrics": dict(parent.state.measured_metrics),
            "context_experience_ids": [
                str(item.get("experience_id"))
                for item in context.recent_experiments
                if item.get("experience_id")
            ],
        }
        try:
            raw_plan = self.controller.plan(context)
            trace["raw_plan"] = raw_plan.to_dict() if hasattr(raw_plan, "to_dict") else raw_plan
            plan = self.validation.schema.validate(raw_plan)
            self.validation.policy.validate(plan, parent.state, context.budget_state)
        except Exception as exc:
            trace["status"] = "rejected"
            trace["error"] = str(exc)
            self.controller_trace.append(trace)
            return None
        if plan.operator not in set(self.config.worker.allowed_operators):
            trace["status"] = "rejected"
            trace["error"] = "operator_not_allowed"
            self.controller_trace.append(trace)
            return None
        try:
            self.registry.validate(plan.operator, parent.state, plan.operator_args, self.target)
        except Exception as exc:
            trace["status"] = "rejected"
            trace["error"] = str(exc)
            self.controller_trace.append(trace)
            return None
        trace["status"] = "validated"
        trace["plan"] = plan.to_dict()
        self.controller_trace.append(trace)
        return plan

    def _train_one(self, parent: ModelCandidate, training_calls: int) -> Optional[ExperienceRecord]:
        if not self.config.worker.enabled:
            return None
        plan = self._controller_plan(parent, training_calls)
        if plan is None:
            return None
        try:
            operator_args = self._worker_operator_args(plan.operator, plan.operator_args, parent)
        except ValueError:
            return None
        child_id = self.models.next_id()
        request_path = str(Path(self.config.remote.resolved_campaign_root) / (plan.experiment_id + "-request.json"))
        result_path = str(Path(self.config.remote.resolved_campaign_root) / ("trainer_result_%s.json" % child_id.lower()))
        output_dir = str(Path(self.config.remote.results_root or self.config.remote.model_root) / "continuous" / child_id)
        template = dict(self.ssh.read_json(self.config.worker.config_template))
        dynamic = dict(template)
        dynamic.update({"model_checkpoint": parent.checkpoint_path, "output_dir": output_dir, "source_steps": parent.state.sampling_steps or 32})
        if plan.operator == "step_distill":
            dynamic["target_steps"] = int(operator_args["target_steps"])
        if plan.operator == "recovery_finetune":
            dynamic["max_steps"] = min(int(plan.operator_args.get("training_steps", dynamic.get("max_steps", 1))), self.config.worker.max_steps)
        self.ssh.write_json(str(Path(self.config.remote.resolved_campaign_root) / (plan.experiment_id + "-config.json")), dynamic)
        request = {
            "experiment_id": plan.experiment_id,
            "child_model_id": child_id,
            "operator": plan.operator,
            "operator_args": operator_args,
            "parent": {"model_id": parent.id, "checkpoint_path": parent.checkpoint_path},
            "artifacts_dir": output_dir,
        }
        self.ssh.write_json(request_path, request)
        config_path = str(Path(self.config.remote.resolved_campaign_root) / (plan.experiment_id + "-config.json"))
        command = (
            self.config.worker.python,
            "--standalone",
            "--nproc_per_node=4",
            self.config.worker.entrypoint,
            "--config",
            config_path,
            "--request",
            request_path,
            "--result",
            result_path,
        )
        self.ssh.run(command, timeout_s=10800.0)
        result = self.ssh.read_json(result_path)
        digest = self.ssh.sha256(result_path)
        importer = RemoteResultImporter(self.experience, self.ssh)
        imported = importer.import_results([{"source_uri": "ssh://%s%s" % (self.config.remote.host, result_path), "source_sha256": digest, "result": result, "request": request}])
        return imported.records[0] if imported.records else None

    def run(self, resume: bool = True, max_experiments: int = 1, split: Optional[str] = None) -> CampaignResult:
        if max_experiments <= 0:
            raise ValueError("max_experiments must be positive")
        self.controller_trace = []
        imported = self._import()
        all_records = list(self.experience.read())
        candidates = self._register_candidates(all_records)
        evaluations = self._load_evaluations()
        state = self._load_campaign_state() if resume else {"evaluated_ids": [], "training_calls": 0, "current_model_id": self.models.active_id}
        evaluated_count = 0
        ordered = sorted((item for item in all_records if item.child_model_id and item.child_model_id in candidates), key=lambda item: (candidates[str(item.child_model_id)].generation, str(item.child_model_id)))
        latest_records = {str(item.child_model_id): item for item in all_records if item.child_model_id}
        tunnel_work = False
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
                parent_summary = self._evaluate(parent, None, split)
                evaluations[parent.id] = parent_summary
                self._save_evaluations(evaluations)
            parent = self._persist_evaluation_state(parent, parent_summary)
            candidates[parent.id] = parent
            summary = self._evaluate(candidate, parent_summary, split)
            evaluations[child_id] = summary
            self._save_evaluations(evaluations)
            candidate = self._persist_evaluation_state(candidate, summary)
            candidates[child_id] = candidate
            training = dict(record.training)
            decision = decide(AcceptanceInput(training, summary, parent_summary, self.target, self.config.efficiency_thresholds, self.config.reward, self.config.research_grade))
            latest_records[child_id] = self._append_evaluated_experience(record, summary, decision)
            self._record_evaluation(candidate, parent, record, summary, decision)
            evaluated_count += 1
            state["evaluated_ids"] = sorted(set(state.get("evaluated_ids", [])) | {child_id})
            state["current_model_id"] = self.models.active_id
            self._save_campaign_state(state)
        if evaluated_count < max_experiments and self.config.worker.enabled:
            parent = self.models.active()
            trained = self._train_one(parent, int(state.get("training_calls", 0)))
            if trained is not None:
                state["training_calls"] = int(state.get("training_calls", 0)) + 1
                self._save_campaign_state(state)
        elif self.config.worker.enabled:
            # A bounded run still lets the Controller inspect the newly
            # measured evidence; execution waits for the next budget window.
            self._controller_plan(self.models.active(), int(state.get("training_calls", 0)))
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
            "quality_scope": self.config.quality_scope,
            "research_grade": self.config.research_grade,
            "controller": {"calls": len(self.controller_trace), "trace": list(self.controller_trace)},
            "pareto_front": [entry.to_dict() for entry in self.pareto.front()],
            "claim_boundary": "structural_proxy is not semantic video quality",
        }
        status = "accepted" if accepted else "completed"
        return CampaignResult(status, self.models.active_id, latest_records, report)


def build_campaign_from_config(path: Path, **kwargs: Any) -> RemoteCampaign:
    return RemoteCampaign.from_config_path(path, **kwargs)
