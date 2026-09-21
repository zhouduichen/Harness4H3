from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .trajectory import redact


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    session_id: str
    target_profile_id: str
    controller: Mapping[str, Any]
    parent_model_id: str
    child_model_id: Optional[str]
    state_digest: str
    diagnosis: Mapping[str, Any]
    plan: Mapping[str, Any]
    execution: Mapping[str, Any]
    training_logs: List[str]
    cost: Mapping[str, Any]
    evaluation: Optional[Mapping[str, Any]]
    failure_type: Optional[str]
    decision: Mapping[str, Any]
    pareto_update: Mapping[str, Any]
    created_at: str
    parent_system_id: Optional[str] = None
    child_system_id: Optional[str] = None
    system_state_digest: str = ""
    fingerprint: str = ""
    repeat_for_statistics: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return redact(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=str(raw["experiment_id"]),
            session_id=str(raw["session_id"]),
            target_profile_id=str(raw["target_profile_id"]),
            controller=dict(raw.get("controller") or {}),
            parent_model_id=str(raw["parent_model_id"]),
            child_model_id=str(raw["child_model_id"]) if raw.get("child_model_id") else None,
            state_digest=str(raw.get("state_digest", "")),
            diagnosis=dict(raw.get("diagnosis") or {}),
            plan=dict(raw.get("plan") or {}),
            execution=dict(raw.get("execution") or {}),
            training_logs=[str(item) for item in raw.get("training_logs", [])],
            cost=dict(raw.get("cost") or {}),
            evaluation=dict(raw["evaluation"]) if raw.get("evaluation") is not None else None,
            failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
            decision=dict(raw.get("decision") or {}),
            pareto_update=dict(raw.get("pareto_update") or {}),
            created_at=str(raw.get("created_at", "")),
            parent_system_id=str(raw["parent_system_id"]) if raw.get("parent_system_id") else None,
            child_system_id=str(raw["child_system_id"]) if raw.get("child_system_id") else None,
            system_state_digest=str(raw.get("system_state_digest", "")),
            fingerprint=str(raw.get("fingerprint", "")),
            repeat_for_statistics=bool(raw.get("repeat_for_statistics", False)),
        )


class ExperimentStore:
    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def read(self) -> Iterator[ExperimentRecord]:
        if not self.path.exists():
            return iter(())

        def records() -> Iterator[ExperimentRecord]:
            with self.path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield ExperimentRecord.from_dict(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise ValueError("invalid experiment record line %d: %s" % (number, exc))

        return records()
