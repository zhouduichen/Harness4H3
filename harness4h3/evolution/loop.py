from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact, CompressionResult, ModelStructure, WanModelAdapter
from ..archive.model_candidate import ModelCandidate
from ..archive.model_store import ModelStore
from ..h3.state import ModelState
from ..memory.experience import ExperienceRecord, ExperienceStore
from ..profiler import ProfileReport, profile_model
from ..sandbox.runner import AlgorithmSandbox, SandboxResult
from .agent import AlgorithmAgent
from .candidate import AlgorithmCandidate
from .policy import SearchPolicy
from .skill_hub import SkillHub
from .store import AlgorithmStore
from ..operators.wan import WanCompressionRegistry, build_wan_compression_registry


EvaluationFn = Callable[[CheckpointArtifact, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class EvolutionRound:
    round_id: str
    parent_model_id: str
    algorithm_id: str
    status: str
    child_model_id: Optional[str]
    decision: Mapping[str, Any]
    paths: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvolutionLoop:
    """One bounded, restartable algorithm-generation/compression round."""

    def __init__(
        self,
        *,
        adapter: WanModelAdapter,
        parent: CheckpointArtifact,
        output_root: Path,
        model_store: Optional[ModelStore] = None,
        algorithm_store: Optional[AlgorithmStore] = None,
        experience_store: Optional[ExperienceStore] = None,
        skill_hub: Optional[SkillHub] = None,
        agent: Optional[AlgorithmAgent] = None,
        sandbox: Optional[AlgorithmSandbox] = None,
        registry: Optional[WanCompressionRegistry] = None,
        evaluator: Optional[EvaluationFn] = None,
        search_policy: Optional[SearchPolicy] = None,
    ) -> None:
        self.adapter = adapter
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rounds_root = self.output_root / "rounds"
        self.rounds_root.mkdir(parents=True, exist_ok=True)
        self.model_store = model_store or ModelStore(self.output_root / "models")
        self.algorithm_store = algorithm_store or AlgorithmStore(self.output_root / "algorithms")
        self.experience_store = experience_store or ExperienceStore(self.output_root / "experiences.jsonl")
        self.skill_hub = skill_hub or SkillHub(self.experience_store.path, self.output_root / "design-genes.jsonl")
        self.agent = agent or AlgorithmAgent()
        self.sandbox = sandbox or AlgorithmSandbox(self.output_root / "sandbox")
        self.registry = registry or build_wan_compression_registry()
        self.evaluator = evaluator or self._default_evaluator
        self.search_policy = search_policy or self._load_search_policy()
        self.parent = parent
        self._initialize_parent()

    def _initialize_parent(self) -> None:
        if self.model_store.lineage():
            return
        structure = self.parent.structure
        state = ModelState(
            model_id="M0000",
            parent_model_id=None,
            checkpoint_path=self.parent.path,
            architecture_name=structure.architecture,
            parameter_count=structure.parameter_count,
            num_blocks=structure.block_count,
            hidden_size=structure.hidden_size,
            ffn_width=structure.ffn_width,
            components={"adapter": "wan"},
            measured_metrics={"offline_simulation": bool(self.parent.metadata.get("offline_simulation", False))},
            provenance={"checkpoint_sha256": self.parent.sha256, "offline_simulation": bool(self.parent.metadata.get("offline_simulation", False))},
        )
        self.model_store.initialize(ModelCandidate("M0000", None, 0, self.parent.path, state, None, "active", {"adapter": "wan"}))

    def run_round(self) -> EvolutionRound:
        round_id = self._next_round_id()
        round_dir = self.rounds_root / round_id
        round_dir.mkdir(parents=True, exist_ok=False)
        parent_candidate = self.model_store.active()
        parent_artifact = self._artifact_for_candidate(parent_candidate)
        profile = profile_model(self.adapter, self.adapter.reload_model(parent_artifact), seed=7)
        profile_path = round_dir / "profile.json"
        profile.write_json(profile_path)
        architecture = str(profile.model["architecture"])
        skills = self.skill_hub.search(architecture=architecture, limit=8)
        algorithm_id = self.algorithm_store.next_id()
        algorithm = self.agent.generate(algorithm_id, parent_id=parent_candidate.id, profile=profile.to_dict(), history=skills, skills=skills)
        self.algorithm_store.create(algorithm)
        algorithm_path = round_dir / "algorithm.json"
        _write_json(algorithm_path, algorithm.to_dict())

        sandbox_result = self.sandbox.run(algorithm.source, profile.to_dict(), skills)
        sandbox_path = round_dir / "sandbox.json"
        _write_json(sandbox_path, _sandbox_dict(sandbox_result))
        if sandbox_result.status != "success" or sandbox_result.plan is None:
            return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, "sandbox", sandbox_result.failure_type or "sandbox_failed", sandbox_result.message, {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path)})

        compression_dir = round_dir / "checkpoint"
        compression = self.registry.execute(algorithm_plan_operator(sandbox_result), self.adapter, parent_artifact, sandbox_result.plan.args, compression_dir)
        compression_path = round_dir / "compression.json"
        _write_json(compression_path, compression.to_dict())
        if compression.status != "success" or compression.child_checkpoint is None:
            return self._finish_failure(round_id, round_dir, parent_candidate, algorithm, "compression", compression.failure_type or "compression_failed", compression.message, {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "compression": str(compression_path)})

        evaluation = dict(self.evaluator(compression.child_checkpoint, profile.to_dict()))
        evaluation_path = round_dir / "evaluation.json"
        _write_json(evaluation_path, evaluation)
        child_candidate = self._create_child(parent_candidate, compression.child_checkpoint, algorithm, evaluation)
        feasible = bool(evaluation.get("feasible", False)) and not bool(evaluation.get("critical_regression", False))
        decision_status = "promoted" if feasible else "rejected"
        policy_choice = self.search_policy.choose(has_best=feasible)
        self._persist_search_policy()
        decision = {
            "status": decision_status,
            "parent_model_id": parent_candidate.id,
            "child_model_id": child_candidate.id,
            "algorithm_id": algorithm.algorithm_id,
            "operator": sandbox_result.plan.operator,
            "operator_args": dict(sandbox_result.plan.args),
            "feasible": feasible,
            "evaluation": evaluation,
            "search_policy": {"choice": policy_choice, **self.search_policy.to_dict()},
        }
        decision_path = round_dir / "decision.json"
        _write_json(decision_path, decision)
        status = "accepted" if feasible else "rejected"
        self._record_experience(round_id, parent_candidate, child_candidate, algorithm, evaluation, decision, status, decision_path)
        self.algorithm_store.update(replace(algorithm, status=status, experiment_ids=(round_id,)))
        if feasible:
            self.model_store.set_active(child_candidate.id)
        return EvolutionRound(round_id, parent_candidate.id, algorithm.algorithm_id, decision_status, child_candidate.id, decision, {"profile": str(profile_path), "algorithm": str(algorithm_path), "sandbox": str(sandbox_path), "compression": str(compression_path), "evaluation": str(evaluation_path), "decision": str(decision_path)})

    def _finish_failure(self, round_id: str, round_dir: Path, parent: ModelCandidate, algorithm: AlgorithmCandidate, stage: str, failure_type: str, message: str, paths: Mapping[str, str]) -> EvolutionRound:
        decision = {"status": "failed", "stage": stage, "failure_type": failure_type, "message": message, "parent_model_id": parent.id, "algorithm_id": algorithm.algorithm_id}
        decision_path = round_dir / "decision.json"
        _write_json(decision_path, decision)
        self._record_experience(round_id, parent, None, algorithm, None, decision, "failed", decision_path)
        self.algorithm_store.update(replace(algorithm, status="failed", experiment_ids=(round_id,), failure_reasons=(failure_type,)))
        all_paths = dict(paths)
        all_paths["decision"] = str(decision_path)
        return EvolutionRound(round_id, parent.id, algorithm.algorithm_id, "failed", None, decision, all_paths)

    def _create_child(self, parent: ModelCandidate, artifact: CheckpointArtifact, algorithm: AlgorithmCandidate, evaluation: Mapping[str, Any]) -> ModelCandidate:
        structure = artifact.structure
        model_id = self.model_store.next_id()
        state = ModelState(
            model_id=model_id,
            parent_model_id=parent.id,
            checkpoint_path=artifact.path,
            architecture_name=structure.architecture,
            parameter_count=structure.parameter_count,
            num_blocks=structure.block_count,
            hidden_size=structure.hidden_size,
            ffn_width=structure.ffn_width,
            measured_metrics=dict(evaluation),
            provenance={"checkpoint_sha256": artifact.sha256, "algorithm_id": algorithm.algorithm_id, "offline_simulation": bool(artifact.metadata.get("offline_simulation", False))},
        )
        child = ModelCandidate(model_id, parent.id, parent.generation + 1, artifact.path, state, algorithm.algorithm_id, "evaluated", {"algorithm_id": algorithm.algorithm_id})
        self.model_store.create(child)
        return child

    def _record_experience(self, round_id: str, parent: ModelCandidate, child: Optional[ModelCandidate], algorithm: AlgorithmCandidate, evaluation: Optional[Mapping[str, Any]], decision: Mapping[str, Any], status: str, decision_path: Path) -> None:
        source = decision_path.read_bytes()
        record = ExperienceRecord(
            experience_id=round_id,
            source_uri=str(decision_path.resolve()),
            source_sha256=hashlib.sha256(source).hexdigest(),
            source_kind="compression_round",
            experiment_id=round_id,
            parent_model_id=parent.id,
            child_model_id=child.id if child else None,
            operator=str(decision.get("operator")) if decision.get("operator") else None,
            operator_args=dict(decision.get("operator_args") or {}),
            training={"architecture": parent.state.architecture_name, "algorithm_id": algorithm.algorithm_id},
            evaluation=dict(evaluation) if evaluation is not None else None,
            decision=dict(decision),
            reward=float(evaluation.get("search_score", evaluation.get("quality_score", 0.0))) if evaluation else None,
            status=status,
            provenance={"architecture": parent.state.architecture_name, "offline_simulation": bool(parent.state.provenance.get("offline_simulation", False))},
            created_at="",
        )
        self.experience_store.append(record)

    def _artifact_for_candidate(self, candidate: ModelCandidate) -> CheckpointArtifact:
        model = self.adapter.load_model(candidate.checkpoint_path)
        structure = self.adapter.inspect(model)
        digest = str(candidate.state.provenance.get("checkpoint_sha256", structure.checkpoint_sha256))
        structure = ModelStructure(structure.architecture, structure.block_count, structure.hidden_size, structure.ffn_width, structure.parameter_count, digest, structure.metadata)
        return CheckpointArtifact(candidate.checkpoint_path, digest, structure.architecture, structure, {"offline_simulation": bool(candidate.state.provenance.get("offline_simulation", False))})

    def _next_round_id(self) -> str:
        numbers = []
        for path in self.rounds_root.glob("R*"):
            try:
                numbers.append(int(path.name[1:]))
            except ValueError:
                continue
        return "R%04d" % (max(numbers, default=0) + 1)

    def _load_search_policy(self) -> SearchPolicy:
        path = self.output_root / "search-policy.json"
        if not path.exists():
            return SearchPolicy()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return SearchPolicy.from_dict(dict(raw))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return SearchPolicy()

    def _persist_search_policy(self) -> None:
        _write_json(self.output_root / "search-policy.json", self.search_policy.to_dict())

    @staticmethod
    def _default_evaluator(child: CheckpointArtifact, profile: Mapping[str, Any]) -> Mapping[str, Any]:
        before = int(profile["model"]["block_count"])
        after = int(child.structure.block_count)
        ratio = float(after) / float(max(before, 1))
        return {
            "quality_score": max(0.0, 0.98 - (1.0 - ratio) * 0.06),
            "latency_s": 1.0 * ratio,
            "peak_memory_gb": 1.0 * ratio,
            "model_size_gb": max(0.01, 1.0 * ratio),
            "feasible": True,
            "critical_regression": False,
            "offline_simulation": bool(child.metadata.get("offline_simulation", False)),
            "search_score": max(0.0, 0.98 - (1.0 - ratio) * 0.06),
        }


def algorithm_plan_operator(result: SandboxResult) -> str:
    if result.plan is None:
        raise ValueError("algorithm plan is unavailable")
    return result.plan.operator


def _sandbox_dict(result: SandboxResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "source_sha256": result.source_sha256,
        "plan": result.plan.to_dict() if result.plan else None,
        "failure_type": result.failure_type,
        "message": result.message,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
        "metadata": dict(result.metadata),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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
