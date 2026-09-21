"""Append-only observation events shared by the Controller loop.

Observations carry references and bounded summaries, never large checkpoints or
videos. The Controller consumes evidence by ID without copying model bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Set, Tuple

from .trajectory import redact


OBSERVATION_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    kind: str
    source_uri: str
    source_sha256: str
    experiment_id: Optional[str]
    model_id: Optional[str]
    parent_model_id: Optional[str]
    summary: Mapping[str, Any]
    artifacts: Tuple[Mapping[str, Any], ...] = ()
    created_at: str = ""
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("observation_id", "kind", "source_uri"):
            if not str(getattr(self, name)).strip():
                raise ValueError("%s must not be empty" % name)
        if not _SHA256_RE.fullmatch(str(self.source_sha256)):
            raise ValueError("source_sha256 must be a 64-character hexadecimal SHA-256")
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported observation schema version: %s" % self.schema_version)

    def to_dict(self) -> Dict[str, Any]:
        return redact(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservationRecord":
        if not isinstance(raw, Mapping):
            raise TypeError("observation record must be a mapping")
        return cls(
            observation_id=str(raw["observation_id"]),
            kind=str(raw["kind"]),
            source_uri=str(raw["source_uri"]),
            source_sha256=str(raw["source_sha256"]),
            experiment_id=str(raw["experiment_id"]) if raw.get("experiment_id") else None,
            model_id=str(raw["model_id"]) if raw.get("model_id") else None,
            parent_model_id=str(raw["parent_model_id"]) if raw.get("parent_model_id") else None,
            summary=dict(raw.get("summary") or {}),
            artifacts=tuple(dict(item) for item in raw.get("artifacts", ()) if isinstance(item, Mapping)),
            created_at=str(raw.get("created_at") or _now()),
            schema_version=int(raw.get("schema_version", OBSERVATION_SCHEMA_VERSION)),
        )


class ObservationStore:
    """Fsync-backed JSONL store with idempotent source events."""

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> Iterator[ObservationRecord]:
        if not self.path.exists():
            return iter(())

        def records() -> Iterator[ObservationRecord]:
            with self.path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        if isinstance(raw, Mapping):
                            yield ObservationRecord.from_dict(raw)
                    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                        raise ValueError("invalid observation line %d: %s" % (number, exc))

        return records()

    def append(self, record: ObservationRecord) -> bool:
        if not isinstance(record, ObservationRecord):
            raise TypeError("record must be an ObservationRecord")
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            existing = {
                item.observation_id: (item.source_uri, item.source_sha256, item.kind)
                for item in self.read()
            }
            key = (record.observation_id, record.source_uri, record.source_sha256, record.kind)
            prior = existing.get(record.observation_id)
            if prior is not None and prior != key[1:]:
                raise ValueError("observation_id is already bound to different source evidence: %s" % record.observation_id)
            if prior is not None:
                return False
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return True

    def ids(self) -> Set[str]:
        return {item.observation_id for item in self.read()}

    def unconsumed(self, consumed_ids: Iterable[str]) -> Tuple[ObservationRecord, ...]:
        consumed = {str(item) for item in consumed_ids}
        return tuple(item for item in self.read() if item.observation_id not in consumed)


def artifact_reference(
    value: Any,
    *,
    sha256: Optional[str] = None,
    kind: str = "artifact",
    summary: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a bounded URI/hash/summary reference for a large artifact."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    digest = str(sha256 or "")
    size = None
    uri = value
    if "://" not in value and path.is_file():
        path = path.resolve()
        uri = "file://%s" % path
        digest = digest or _sha256_file(path)
        size = path.stat().st_size
    if not _SHA256_RE.fullmatch(digest):
        return None
    result: Dict[str, Any] = {"uri": uri, "sha256": digest, "kind": kind}
    if size is not None:
        result["size_bytes"] = int(size)
    if summary:
        result["summary"] = dict(summary)
    return result


def make_observation(
    observation_id: str,
    kind: str,
    source_uri: str,
    source_sha256: str,
    *,
    experiment_id: Optional[str] = None,
    model_id: Optional[str] = None,
    parent_model_id: Optional[str] = None,
    summary: Optional[Mapping[str, Any]] = None,
    artifacts: Iterable[Mapping[str, Any]] = (),
) -> ObservationRecord:
    return ObservationRecord(
        observation_id=str(observation_id),
        kind=str(kind),
        source_uri=str(source_uri),
        source_sha256=str(source_sha256),
        experiment_id=str(experiment_id) if experiment_id else None,
        model_id=str(model_id) if model_id else None,
        parent_model_id=str(parent_model_id) if parent_model_id else None,
        summary=dict(summary or {}),
        artifacts=tuple(dict(item) for item in artifacts if isinstance(item, Mapping)),
        created_at=_now(),
    )


class ControllerEventStore:
    """Append-only lifecycle audit stream for status/follow consumers."""

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, event_type: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        event = {
            "event_id": "evt-" + hashlib.sha256(
                json.dumps({"type": event_type, "payload": payload, "time": _now()}, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
            "event_type": str(event_type),
            "created_at": _now(),
            **dict(payload),
        }
        line = json.dumps(redact(event), ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return event

    def read(self) -> Iterator[Mapping[str, Any]]:
        if not self.path.exists():
            return iter(())

        def events() -> Iterator[Mapping[str, Any]]:
            with self.path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError("invalid controller event line %d: %s" % (number, exc))
                    if isinstance(value, Mapping):
                        yield dict(value)

        return events()

    def tail(self, limit: int = 256) -> Tuple[Mapping[str, Any], ...]:
        """Read only the newest bounded events without scanning the whole log.

        Controller event logs are intentionally append-only and can become
        large over an overnight campaign.  Planning telemetry needs recent
        lane evidence, but must not reread a multi-hundred-megabyte JSONL file
        on every LLM call.
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("event tail limit must be a positive integer")
        if not self.path.exists():
            return ()
        chunk_size = 64 * 1024
        data = b""
        with self.path.open("rb") as handle:
            position = handle.seek(0, os.SEEK_END)
            while position > 0 and data.count(b"\n") <= limit:
                size = min(chunk_size, position)
                position -= size
                handle.seek(position)
                data = handle.read(size) + data
        lines = data.splitlines()[-limit:]
        values = []
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid controller event in tail: %s" % exc)
            if isinstance(value, Mapping):
                values.append(dict(value))
        return tuple(values)


__all__ = [
    "ControllerEventStore",
    "ObservationRecord",
    "ObservationStore",
    "artifact_reference",
    "make_observation",
]
