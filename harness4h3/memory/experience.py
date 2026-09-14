"""Normalized, append-only memory for remote training and evaluation evidence."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Set, Tuple

from .trajectory import redact


EXPERIENCE_SCHEMA_VERSION = 1
EXPERIENCE_STATUSES = frozenset(
    {
        "training_only_unvalidated",
        "evaluated_candidate",
        "accepted",
        "rejected",
        "failed",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def experience_status(value: str) -> str:
    """Validate and return a normalized experience status."""

    status = str(value).strip()
    if status not in EXPERIENCE_STATUSES:
        raise ValueError("unsupported experience status: %s" % status)
    return status


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    source_uri: str
    source_sha256: str
    source_kind: str
    experiment_id: str
    parent_model_id: Optional[str]
    child_model_id: Optional[str]
    operator: Optional[str]
    operator_args: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Optional[Mapping[str, Any]]
    decision: Mapping[str, Any]
    reward: Optional[float]
    status: str
    provenance: Mapping[str, Any]
    created_at: str
    schema_version: int = EXPERIENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("experience_id", "source_uri", "source_kind", "experiment_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError("%s must not be empty" % name)
        if not _SHA256_RE.fullmatch(str(self.source_sha256)):
            raise ValueError("source_sha256 must be a 64-character hexadecimal SHA-256")
        if self.schema_version != EXPERIENCE_SCHEMA_VERSION:
            raise ValueError("unsupported experience schema version: %s" % self.schema_version)
        experience_status(self.status)
        if self.reward is not None:
            try:
                reward = float(self.reward)
            except (TypeError, ValueError):
                raise ValueError("reward must be numeric or None")
            if reward != reward or reward in (float("inf"), float("-inf")):
                raise ValueError("reward must be finite")

    @classmethod
    def minimal(cls, experience_id: str, source_uri: str, source_sha256: str) -> "ExperienceRecord":
        return cls(
            experience_id=experience_id,
            source_uri=source_uri,
            source_sha256=source_sha256,
            source_kind="unknown",
            experiment_id=experience_id,
            parent_model_id=None,
            child_model_id=None,
            operator=None,
            operator_args={},
            training={},
            evaluation=None,
            decision={"status": "not_evaluated"},
            reward=None,
            status="training_only_unvalidated",
            provenance={},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return redact(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperienceRecord":
        if not isinstance(raw, Mapping):
            raise TypeError("experience record must be a mapping")
        return cls(
            experience_id=str(raw["experience_id"]),
            source_uri=str(raw["source_uri"]),
            source_sha256=str(raw["source_sha256"]),
            source_kind=str(raw.get("source_kind", "unknown")),
            experiment_id=str(raw.get("experiment_id", raw["experience_id"])),
            parent_model_id=str(raw["parent_model_id"]) if raw.get("parent_model_id") else None,
            child_model_id=str(raw["child_model_id"]) if raw.get("child_model_id") else None,
            operator=str(raw["operator"]) if raw.get("operator") else None,
            operator_args=dict(raw.get("operator_args") or {}),
            training=dict(raw.get("training") or {}),
            evaluation=dict(raw["evaluation"]) if raw.get("evaluation") is not None else None,
            decision=dict(raw.get("decision") or {}),
            reward=float(raw["reward"]) if raw.get("reward") is not None else None,
            status=str(raw["status"]),
            provenance=dict(raw.get("provenance") or {}),
            created_at=str(raw.get("created_at", "")),
            schema_version=int(raw.get("schema_version", EXPERIENCE_SCHEMA_VERSION)),
        )


class ExperienceStore:
    """An fsynced JSONL store with source URI/hash idempotency."""

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)

    def _source_keys(self) -> Set[Tuple[str, str]]:
        return {(item.source_uri, item.source_sha256) for item in self.read()}

    def append(self, record: ExperienceRecord) -> bool:
        """Append a record, returning False when its source was already imported."""

        if not isinstance(record, ExperienceRecord):
            raise TypeError("record must be an ExperienceRecord")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            if (record.source_uri, record.source_sha256) in self._source_keys():
                return False
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return True

    def read(self) -> Iterator[ExperienceRecord]:
        if not self.path.exists():
            return iter(())

        def records() -> Iterator[ExperienceRecord]:
            with self.path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield ExperienceRecord.from_dict(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise ValueError("invalid experience record line %d: %s" % (number, exc))

        return records()

    def source_hashes(self) -> Dict[str, str]:
        return {item.source_uri: item.source_sha256 for item in self.read()}

