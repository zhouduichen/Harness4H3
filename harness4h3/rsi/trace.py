from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FAILURE_CLASSES = frozenset({"none", "technical", "resource_limited", "method_negative", "protocol_violation"})


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    task_id: str
    bundle_id: str
    stage: str
    event: str
    status: str
    started_ns: int
    ended_ns: int
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...] = ()
    failure_class: str = "none"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for key in ("run_id", "task_id", "bundle_id", "stage", "event", "status"):
            if not str(getattr(self, key)).strip():
                raise ValueError("trace_%s_required" % key)
        if self.started_ns < 0 or self.ended_ns < self.started_ns:
            raise ValueError("trace_timestamps_invalid")
        if self.failure_class not in FAILURE_CLASSES:
            raise ValueError("trace_failure_class_invalid")

    @classmethod
    def create(cls, *, run_id: str, task_id: str, bundle_id: str, stage: str, event: str, status: str, started_ns: int, ended_ns: int, payload: Mapping[str, Any], evidence_refs: Sequence[str] = (), failure_class: str = "none") -> "TraceEvent":
        return cls(run_id, task_id, bundle_id, stage, event, status, int(started_ns), int(ended_ns), dict(payload), tuple(str(x) for x in evidence_refs), failure_class)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TraceEvent":
        if int(raw.get("schema_version", 0)) != 1:
            raise ValueError("trace_schema_unsupported")
        return cls(str(raw["run_id"]), str(raw["task_id"]), str(raw["bundle_id"]), str(raw["stage"]), str(raw["event"]), str(raw["status"]), int(raw["started_ns"]), int(raw["ended_ns"]), dict(raw["payload"]), tuple(raw.get("evidence_refs", ())), str(raw.get("failure_class", "none")), 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def append(self, event: TraceEvent) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (event.run_id + ".jsonl")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self, run_id: str) -> list[TraceEvent]:
        path = self.root / (str(run_id) + ".jsonl")
        if not path.exists():
            return []
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                records.append(TraceEvent.from_dict(json.loads(line)))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError("trace_record_invalid:%d" % line_number) from exc
        return records


class CostLedger:
    @staticmethod
    def summary(events: Sequence[TraceEvent]) -> dict[str, Any]:
        durations = sum(item.ended_ns - item.started_ns for item in events)
        totals = {"input_tokens": 0, "output_tokens": 0, "optimizer_updates": 0, "gpu_lease_seconds": 0.0}
        known = {key: True for key in totals}
        for item in events:
            for key in totals:
                value = item.payload.get(key)
                if value is None:
                    known[key] = False
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError("trace_cost_invalid:%s" % key)
                else:
                    totals[key] += value
        return {
            "event_count": len(events),
            "stage_elapsed_seconds_sum": durations / 1_000_000_000,
            "metrics": {key: {"value": totals[key] if known[key] else None, "status": "measured" if known[key] else "unavailable"} for key in totals},
            "estimated_usd": {"value": None, "status": "unavailable", "reason": "no_pinned_price_table"},
        }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, str(path))
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
