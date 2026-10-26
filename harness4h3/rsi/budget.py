from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class BudgetLedger:
    """Cross-process, fail-closed reservations for a bounded research run."""

    def __init__(self, path: Path, limits: Mapping[str, int | float]) -> None:
        self.path = Path(path)
        self.limits = self._normalize(limits)
        if not self.limits:
            raise ValueError("budget_limits_empty")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with self._locked():
            if not self.path.exists():
                self._write({"schema_version": 1, "limits": self.limits, "jobs": {}})
            self._read()

    @staticmethod
    def _normalize(values: Mapping[str, int | float]) -> dict[str, float]:
        result = {}
        for key, raw in values.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
                raise ValueError("budget_value_invalid:%s" % key)
            result[str(key)] = float(raw)
        return result

    def reserve(self, job_id: str, costs: Mapping[str, int | float]) -> bool:
        identifier = str(job_id).strip()
        if not identifier:
            raise ValueError("budget_job_id_required")
        request = self._normalize(costs)
        if set(request) - set(self.limits):
            raise ValueError("budget_metric_not_configured")
        with self._locked():
            state = self._read()
            existing = state["jobs"].get(identifier)
            if existing is not None:
                if existing["reserved"] != request:
                    raise ValueError("budget_job_reservation_conflict")
                return True
            reserved = {key: 0.0 for key in self.limits}
            for job in state["jobs"].values():
                for key, amount in job["reserved"].items():
                    reserved[key] += amount
                for key, amount in (job.get("actual") or {}).items():
                    reserved[key] += max(0.0, amount - job["reserved"].get(key, 0.0))
            if any(reserved[key] + request.get(key, 0.0) > limit for key, limit in self.limits.items()):
                return False
            state["jobs"][identifier] = {"reserved": request, "actual": None}
            self._write(state)
            return True

    def finalize(self, job_id: str, actual: Mapping[str, int | float]) -> None:
        identifier = str(job_id).strip()
        measured = self._normalize(actual)
        if set(measured) - set(self.limits):
            raise ValueError("budget_metric_not_configured")
        with self._locked():
            state = self._read()
            job = state["jobs"].get(identifier)
            if job is None:
                raise ValueError("budget_job_not_reserved")
            if job["actual"] is not None:
                if job["actual"] != measured:
                    raise ValueError("budget_job_settlement_conflict")
                return
            reservation = job["reserved"]
            # Actual work above the reservation is recorded; it cannot be hidden
            # by a retry or deducted from already completed work.
            for key, value in measured.items():
                if value > self.limits[key]:
                    raise ValueError("budget_job_actual_exceeds_total_limit:%s" % key)
            job["actual"] = measured
            self._write(state)

    def snapshot(self) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            spent = {key: 0.0 for key in self.limits}
            for job in state["jobs"].values():
                usage = job["actual"] if job["actual"] is not None else job["reserved"]
                for key, value in usage.items():
                    spent[key] += value
            return {"limits": dict(self.limits), "spent_or_reserved": spent, "jobs": state["jobs"]}

    def _read(self) -> dict[str, Any]:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("limits") != self.limits or not isinstance(state.get("jobs"), dict):
            raise ValueError("budget_ledger_protocol_mismatch")
        return state

    def _write(self, value: Mapping[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".tmp-", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, str(self.path))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    class _Lock:
        def __init__(self, path: Path) -> None:
            self.path, self.handle = path, None
        def __enter__(self):
            self.handle = self.path.open("a+")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            return self
        def __exit__(self, exc_type, exc, tb):
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def _locked(self):
        return self._Lock(self.lock_path)
