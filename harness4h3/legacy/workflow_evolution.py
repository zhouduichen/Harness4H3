from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..archive.store import Candidate, CandidateStore
from ..config import EvolutionConfig
from ..harness.state import Task
from ..memory.trajectory import Trajectory


@dataclass(frozen=True)
class Diagnosis:
    failure_type: str
    evidence_task_ids: List[str]
    reason: str


@dataclass(frozen=True)
class EvolutionOutcome:
    status: str
    parent_id: str
    candidate_id: Optional[str]
    parent_score: Optional[float]
    candidate_score: Optional[float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "parent_id": self.parent_id,
            "candidate_id": self.candidate_id,
            "parent_score": self.parent_score,
            "candidate_score": self.candidate_score,
            "reason": self.reason,
        }


def diagnose(
    trajectories: Sequence[Trajectory],
    recurrence_threshold: int = 2,
    low_score_threshold: float = 0.65,
) -> Optional[Diagnosis]:
    eligible = [item for item in trajectories if item.split != "heldout"]
    labels = [item.failure_type or "low_score" for item in eligible if item.failure_type or (item.score is not None and item.score < low_score_threshold)]
    if not labels:
        return None
    failure_type, count = Counter(labels).most_common(1)[0]
    if count < recurrence_threshold:
        return None
    evidence = [
        item.task_id
        for item in eligible
        if item.failure_type == failure_type
        or (failure_type == "low_score" and not item.failure_type and item.score is not None and item.score < low_score_threshold)
    ]
    return Diagnosis(
        failure_type=failure_type,
        evidence_task_ids=evidence,
        reason="%s recurred in %d trajectories" % (failure_type, count),
    )


def _append_suffix(policy: Dict[str, Any], sentence: str) -> Optional[Mapping[str, Any]]:
    prompt = policy.setdefault("prompt", {})
    existing = str(prompt.get("suffix", "")).strip()
    if sentence in existing:
        return None
    prompt["suffix"] = (existing + " " + sentence).strip()
    return {"op": "append", "path": "prompt.suffix", "value": sentence}


def propose_mutation(parent: Candidate, diagnosis: Diagnosis, candidate_id: str) -> Optional[Candidate]:
    policy = copy.deepcopy(dict(parent.policy))
    failure = diagnosis.failure_type
    if failure == "low_luma":
        mutation_type = "prompt"
        patch = _append_suffix(policy, "Use balanced exposure with clearly visible subjects and avoid dark or underexposed frames.")
    elif failure == "temporal_instability":
        mutation_type = "prompt"
        patch = _append_suffix(policy, "Keep subject identity, geometry, background, and motion temporally consistent across all frames.")
    elif failure == "low_score":
        prompt_options = (
            "Keep subject identity, geometry, background, and motion temporally consistent across all frames.",
            "Stage the requested actions in a clear chronological sequence with smooth transitions and stable composition.",
            "Use balanced exposure, readable silhouettes, and natural motion without frozen or flickering frames.",
        )
        patch = None
        mutation_type = "prompt"
        for sentence in prompt_options:
            patch = _append_suffix(policy, sentence)
            if patch is not None:
                break
        if patch is None:
            workflow = policy.setdefault("workflow", {})
            current_steps = int(workflow.get("steps", 4))
            if current_steps < 8:
                workflow["steps"] = current_steps + 1
                mutation_type = "workflow"
                patch = {"op": "replace", "path": "workflow.steps", "value": current_steps + 1}
            else:
                return None
    elif failure == "backend_timeout":
        workflow = policy.setdefault("workflow", {})
        current = int(workflow.get("steps", 8))
        if current <= 1:
            return None
        workflow["steps"] = current - 1
        mutation_type = "workflow"
        patch = {"op": "replace", "path": "workflow.steps", "value": current - 1}
    elif failure in {"artifact_missing", "artifact_download", "decode_failed", "backend_execution", "backend_submission", "backend_request", "evaluator_failure"}:
        context = policy.setdefault("context", {})
        current = int(context.get("max_prompt_chars", 4000))
        new_value = max(512, min(current, 3000))
        if new_value == current:
            return None
        context["max_prompt_chars"] = new_value
        mutation_type = "context"
        patch = {"op": "replace", "path": "context.max_prompt_chars", "value": new_value}
    else:
        return None
    if patch is None:
        return None
    changed_categories = [
        key for key in ("prompt", "context", "workflow")
        if parent.policy.get(key) != policy.get(key)
    ]
    if len(changed_categories) != 1 or changed_categories[0] != mutation_type:
        return None
    return Candidate(
        id=candidate_id,
        parent=parent.id,
        generation=parent.generation + 1,
        mutation_type=mutation_type,
        patch=patch,
        reason=diagnosis.reason,
        policy=policy,
        evidence_task_ids=list(diagnosis.evidence_task_ids),
        metadata={"diagnosis": diagnosis.failure_type},
    )


def should_promote(
    parent_score: float,
    child_score: float,
    sanity_passed: bool,
    critical_regression: bool,
    regression_count: int,
    config: EvolutionConfig,
) -> bool:
    return (
        sanity_passed
        and not critical_regression
        and regression_count <= config.max_regressions
        and child_score > parent_score + config.min_delta
    )


class EvolutionController:
    def __init__(
        self,
        store: CandidateStore,
        config: EvolutionConfig,
        run_batch: Callable[[Sequence[Task], Candidate], List[Trajectory]],
    ):
        self.store = store
        self.config = config
        self.run_batch = run_batch

    @staticmethod
    def _latest(trajectories: Sequence[Trajectory], candidate_id: str) -> Dict[str, Trajectory]:
        result: Dict[str, Trajectory] = {}
        for item in trajectories:
            if item.harness_version == candidate_id and item.split != "heldout":
                result[item.task_id] = item
        return result

    def evolve(self, tasks: Sequence[Task], history: Sequence[Trajectory]) -> EvolutionOutcome:
        parent = self.store.active()
        parent_history = [item for item in history if item.harness_version == parent.id]
        diagnosis = diagnose(parent_history, self.config.recurrence_threshold, self.config.low_score_threshold)
        if diagnosis is None:
            return EvolutionOutcome("no_mutation", parent.id, None, None, None, "no recurring failure")
        candidate = propose_mutation(parent, diagnosis, self.store.next_id())
        if candidate is None:
            return EvolutionOutcome("no_mutation", parent.id, None, None, None, "no safe catalog mutation")
        self.store.create(candidate)

        sanity_tasks = [task for task in tasks if task.split == "sanity"]
        sanity_results = self.run_batch(sanity_tasks, candidate) if sanity_tasks else []
        sanity_passed = all(item.score is not None and item.score > 0 and not item.critical_regression for item in sanity_results)

        evidence_ids = set(candidate.evidence_task_ids)
        compare_tasks = [task for task in tasks if task.split == "dev" or task.id in evidence_ids]
        unique_tasks = {task.id: task for task in compare_tasks}
        compare_tasks = list(unique_tasks.values())
        latest_parent = self._latest(parent_history, parent.id)
        missing_parent = [task for task in compare_tasks if task.id not in latest_parent]
        if missing_parent:
            for item in self.run_batch(missing_parent, parent):
                latest_parent[item.task_id] = item
        child_results = self.run_batch(compare_tasks, candidate) if sanity_passed else []
        child_by_task = {item.task_id: item for item in child_results}

        paired = [(latest_parent[task.id], child_by_task[task.id]) for task in compare_tasks if task.id in latest_parent and task.id in child_by_task]
        parent_score = mean([float(parent_item.score or 0) for parent_item, _ in paired]) if paired else 0.0
        child_score = mean([float(child_item.score or 0) for _, child_item in paired]) if paired else 0.0
        critical = any(item.critical_regression for item in child_results)
        regressions = sum(1 for parent_item, child_item in paired if float(child_item.score or 0) < float(parent_item.score or 0))
        promoted = should_promote(parent_score, child_score, sanity_passed, critical, regressions, self.config)
        status = "promoted" if promoted else "dropped"
        reason = "candidate passed all gates" if promoted else "candidate failed promotion gates"
        outcome = EvolutionOutcome(status, parent.id, candidate.id, parent_score, child_score, reason)
        self.store.record_outcome(
            candidate.id,
            {
                **outcome.to_dict(),
                "sanity_passed": sanity_passed,
                "critical_regression": critical,
                "regression_count": regressions,
                "evidence_task_ids": candidate.evidence_task_ids,
            },
        )
        if promoted:
            self.store.promote(candidate.id)
        return outcome
