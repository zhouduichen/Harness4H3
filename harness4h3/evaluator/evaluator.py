from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .. import Harness4H3Error


class EvaluatorError(Harness4H3Error):
    pass


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    metrics: Mapping[str, Any]
    critical_regression: bool = False
    failure_type: Optional[str] = None


def validate_result(raw: Any) -> EvaluationResult:
    if not isinstance(raw, Mapping):
        raise EvaluatorError("evaluator result must be a JSON object")
    try:
        score = float(raw["score"])
    except (KeyError, TypeError, ValueError):
        raise EvaluatorError("evaluator result requires numeric score")
    if not 0 <= score <= 1:
        raise EvaluatorError("evaluator score must be between 0 and 1")
    metrics = raw.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvaluatorError("evaluator result requires metrics object")
    return EvaluationResult(
        score=score,
        metrics=dict(metrics),
        critical_regression=bool(raw.get("critical_regression", False)),
        failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
    )


class SubprocessEvaluator:
    def __init__(self, command: Sequence[str], timeout_s: float = 120):
        self.command = tuple(command) or (sys.executable, "-m", "harness4h3.evaluator.worker")
        self.timeout_s = timeout_s

    def evaluate(self, request: Mapping[str, Any]) -> EvaluationResult:
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvaluatorError("evaluator process failed: %s" % exc)
        if completed.returncode != 0:
            raise EvaluatorError("evaluator exited %d: %s" % (completed.returncode, completed.stderr.strip()[:500]))
        try:
            raw = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EvaluatorError("evaluator did not return valid JSON: %s" % exc)
        return validate_result(raw)


def make_request(
    task_id: str,
    expected: Mapping[str, Any],
    artifacts: Sequence[str],
    wall_time_s: float,
    backend_success: bool = True,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "expected": dict(expected),
        "artifacts": list(artifacts),
        "wall_time_s": float(wall_time_s),
        "backend_success": bool(backend_success),
    }

