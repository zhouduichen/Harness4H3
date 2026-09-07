from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..archive.store import Candidate
from ..config import WorkflowConfig
from ..evaluator.evaluator import EvaluatorError, SubprocessEvaluator, make_request
from ..memory.trajectory import Trajectory, TrajectoryStore
from ..model.minimax_h3 import BackendError, MiniMaxH3Adapter
from .context import ExecutionRequest, build_execution_request
from .state import Task, TaskState


class HarnessRunner:
    def __init__(
        self,
        workflow_template: Mapping[str, Any],
        workflow_config: WorkflowConfig,
        backend: MiniMaxH3Adapter,
        evaluator: SubprocessEvaluator,
        trajectories: TrajectoryStore,
        output_root: Path,
    ):
        self.workflow_template = workflow_template
        self.workflow_config = workflow_config
        self.backend = backend
        self.evaluator = evaluator
        self.trajectories = trajectories
        self.output_root = Path(output_root)

    def run_task(self, task: Task, candidate: Candidate) -> Trajectory:
        state = TaskState(task_id=task.id, goal=task.prompt)
        started = time.monotonic()
        request: ExecutionRequest
        try:
            request = build_execution_request(self.workflow_template, task, candidate, self.workflow_config)
        except (ValueError, TypeError, KeyError) as exc:
            trajectory = self._failure(task, candidate, None, "context_invalid", str(exc), started)
            self.trajectories.append(trajectory)
            return trajectory

        state.observe({"action": "render_workflow", "context_digest": request.context_digest})
        try:
            result = self.backend.run(request.workflow, self.output_root / candidate.id / task.id)
            artifact_paths = [str(path.resolve()) for path in result.artifacts]
            state.artifacts.extend(artifact_paths)
            state.observe({"action": "backend_complete", "prompt_id": result.prompt_id, "artifacts": artifact_paths})
            evaluation = self.evaluator.evaluate(
                make_request(task.id, task.expected, artifact_paths, result.wall_time_s, backend_success=True)
            )
            state.observe({"action": "external_evaluation", "score": evaluation.score, "metrics": dict(evaluation.metrics)})
            state.done = True
            trajectory = Trajectory(
                task_id=task.id,
                harness_version=candidate.id,
                split=task.split,
                inputs={"prompt": task.prompt, "seed": task.seed, "constraints": dict(task.constraints)},
                steps=list(state.recent_history),
                final_result={"prompt_id": result.prompt_id, "artifacts": artifact_paths},
                score=evaluation.score,
                failure_type=evaluation.failure_type,
                cost={"tokens": 0.0, "wall_time": time.monotonic() - started},
                evaluation=dict(evaluation.metrics),
                critical_regression=evaluation.critical_regression,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        except BackendError as exc:
            trajectory = self._failure(task, candidate, request, exc.failure_type, str(exc), started)
        except EvaluatorError as exc:
            trajectory = self._failure(task, candidate, request, "evaluator_failure", str(exc), started)
        self.trajectories.append(trajectory)
        return trajectory

    def _failure(
        self,
        task: Task,
        candidate: Candidate,
        request: Any,
        failure_type: str,
        message: str,
        started: float,
    ) -> Trajectory:
        steps: List[Mapping[str, Any]] = []
        if request is not None:
            steps.append({"action": "render_workflow", "context_digest": request.context_digest})
        steps.append({"action": "failure", "failure_type": failure_type, "message": message})
        return Trajectory(
            task_id=task.id,
            harness_version=candidate.id,
            split=task.split,
            inputs={"prompt": task.prompt, "seed": task.seed, "constraints": dict(task.constraints)},
            steps=steps,
            final_result=None,
            score=0.0,
            failure_type=failure_type,
            cost={"tokens": 0.0, "wall_time": time.monotonic() - started},
            evaluation={},
            critical_regression=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def run_batch(self, tasks: Sequence[Task], candidate: Candidate) -> List[Trajectory]:
        return [self.run_task(task, candidate) for task in tasks]


def load_workflow(path: Path) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("workflow must be an object")
    return raw

