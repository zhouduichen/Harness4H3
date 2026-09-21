from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .. import Harness4H3Error
from ..controller.schemas import EvaluationRecord, HardwareMetrics


class EvaluatorError(Harness4H3Error):
    pass


def legacy_evaluator_result(
    score: float,
    metrics: Mapping[str, Any],
    critical_regression: bool = False,
    failure_type: Optional[str] = None,
) -> EvaluationRecord:
    """Build canonical evidence from the old score/metrics subprocess shape."""

    return EvaluationRecord(
        quality_score=float(score),
        quality_metrics=dict(metrics),
        hardware=HardwareMetrics(),
        feasible=False,
        critical_regression=bool(critical_regression),
        failure_type=str(failure_type) if failure_type else None,
    )


# Existing benchmark integrations construct ``EvaluationResult(...)`` with
# the legacy positional shape. Keep that spelling as a factory while using a
# single canonical record internally.
EvaluationResult = legacy_evaluator_result


def validate_result(raw: Any) -> EvaluationRecord:
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
    return legacy_evaluator_result(
        score=score,
        metrics=dict(metrics),
        critical_regression=bool(raw.get("critical_regression", False)),
        failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
    )


class SubprocessEvaluator:
    def __init__(self, command: Sequence[str], timeout_s: float = 120):
        self.command = tuple(command) or (sys.executable, "-m", "harness4h3.evaluator.worker")
        self.timeout_s = timeout_s

    @staticmethod
    def _child_environment() -> Mapping[str, str]:
        """Make the bundled evaluator importable from any working directory.

        Remote campaigns launch from a tools directory and may use a system
        Python for the controller while the evaluator package lives only in
        the checked-out harness.  The evaluator is a trusted local child, so
        add this repository root to the inherited import path instead of
        relying on the caller's current directory or shell setup.
        """

        environment = os.environ.copy()
        repository_root = str(Path(__file__).resolve().parents[2])
        existing = environment.get("PYTHONPATH", "")
        entries = [item for item in existing.split(os.pathsep) if item]
        if repository_root not in entries:
            entries.insert(0, repository_root)
        environment["PYTHONPATH"] = os.pathsep.join(entries)
        return environment

    def evaluate(self, request: Mapping[str, Any]) -> EvaluationRecord:
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
                env=self._child_environment(),
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
