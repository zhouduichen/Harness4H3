from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional


SENSITIVE_KEYS = ("token", "password", "secret", "authorization", "api_key")


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if any(mark in str(key).lower() for mark in SENSITIVE_KEYS) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    harness_version: str
    split: str
    inputs: Mapping[str, Any]
    steps: List[Mapping[str, Any]]
    final_result: Any
    score: Optional[float]
    failure_type: Optional[str]
    cost: Mapping[str, float]
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    critical_regression: bool = False
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return redact(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Trajectory":
        return cls(
            task_id=str(raw["task_id"]),
            harness_version=str(raw["harness_version"]),
            split=str(raw.get("split", "dev")),
            inputs=dict(raw.get("inputs") or {}),
            steps=list(raw.get("steps") or []),
            final_result=raw.get("final_result"),
            score=float(raw["score"]) if raw.get("score") is not None else None,
            failure_type=str(raw["failure_type"]) if raw.get("failure_type") else None,
            cost=dict(raw.get("cost") or {}),
            evaluation=dict(raw.get("evaluation") or {}),
            critical_regression=bool(raw.get("critical_regression", False)),
            created_at=str(raw.get("created_at", "")),
        )


class TrajectoryStore:
    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, trajectory: Trajectory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trajectory.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def read(self) -> Iterator[Trajectory]:
        if not self.path.exists():
            return iter(())

        def records() -> Iterator[Trajectory]:
            with self.path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield Trajectory.from_dict(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise ValueError("invalid trajectory line %d: %s" % (number, exc))

        return records()

