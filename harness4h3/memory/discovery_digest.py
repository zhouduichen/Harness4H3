"""Bounded, provenance-carrying experiment memory for the next Controller turn."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Tuple


_DROP_KEY_FRAGMENTS = ("checkpoint", "stdout", "stderr", "video", "payload")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _drop_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return any(fragment in normalized for fragment in _DROP_KEY_FRAGMENTS)


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    """Remove large/raw fields before the value enters Controller context."""

    if depth > 5:
        return "<depth-limited>"
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, depth=depth + 1)
            for key, item in value.items()
            if not _drop_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth=depth + 1) for item in list(value)[:32]]
    if isinstance(value, str):
        return value if len(value) <= 512 else value[:509] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:512]


def _fit(value: Any, max_chars: int) -> Any:
    """Fit one JSON object under the per-item budget deterministically."""

    compact = _sanitize(value)
    if _json_size(compact) <= max_chars:
        return compact
    if isinstance(compact, Mapping):
        # Preserve the high-value identity/decision fields before optional detail.
        priority = (
            "experiment_id",
            "operator",
            "plan",
            "execution",
            "evaluation",
            "failure_type",
            "decision",
            "cost",
            "created_at",
        )
        ordered = {}
        for key in priority:
            if key in compact:
                ordered[key] = compact[key]
        for key in sorted(compact):
            if key not in ordered:
                ordered[key] = compact[key]
        while ordered and _json_size(ordered) > max_chars:
            removable = next((key for key in reversed(tuple(ordered)) if key not in priority), None)
            if removable is None:
                removable = next(iter(ordered))
            ordered.pop(removable)
        if _json_size(ordered) <= max_chars:
            return ordered
    elif isinstance(compact, list):
        values = list(compact)
        while values and _json_size(values) > max_chars:
            values.pop()
        if _json_size(values) <= max_chars:
            return values
    digest = hashlib.sha256(
        json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {"truncated": True, "value_sha256": digest}


def _record_key(record: Mapping[str, Any], index: int) -> Tuple[str, int]:
    created = record.get("created_at") or record.get("updated_at") or ""
    return str(created), index


def _operator(record: Mapping[str, Any]) -> str:
    plan = record.get("plan")
    if isinstance(plan, Mapping) and plan.get("operator"):
        return str(plan["operator"])
    return str(record.get("operator") or "unknown")


def _failure_type(record: Mapping[str, Any]) -> str:
    if record.get("failure_type"):
        return str(record["failure_type"])
    execution = record.get("execution")
    if isinstance(execution, Mapping) and execution.get("failure_type"):
        return str(execution["failure_type"])
    status = execution.get("status") if isinstance(execution, Mapping) else record.get("status")
    return "execution_failed" if str(status).lower() in {"failed", "failure", "error"} else ""


@dataclass(frozen=True)
class DigestLimits:
    max_recent_experiments: int = 8
    max_operator_findings: int = 2
    max_frontier: int = 4
    max_failures: int = 16
    max_item_chars: int = 2048

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values.values()):
            raise ValueError("discovery digest limits must be positive integers")


@dataclass(frozen=True)
class DiscoveryDigest:
    """Small, deterministic memory view used by the remote Controller."""

    schema_version: int
    source_observation_ids: Tuple[str, ...]
    source_digest: str
    recent_experiments: Tuple[Mapping[str, Any], ...]
    operator_findings: Mapping[str, Tuple[Mapping[str, Any], ...]]
    pareto_frontier: Tuple[Mapping[str, Any], ...]
    failure_counts: Mapping[str, int]
    telemetry: Mapping[str, Any]
    created_at: str

    @classmethod
    def build(
        cls,
        experiments: Iterable[Mapping[str, Any]],
        observations: Iterable[Mapping[str, Any]] = (),
        pareto: Iterable[Mapping[str, Any]] = (),
        telemetry: Iterable[Mapping[str, Any]] = (),
        limits: DigestLimits = DigestLimits(),
    ) -> "DiscoveryDigest":
        if not isinstance(limits, DigestLimits):
            raise TypeError("limits must be a DigestLimits")
        records = [dict(item) for item in experiments if isinstance(item, Mapping)]
        ordered = [
            item
            for _, item in sorted(enumerate(records), key=lambda pair: _record_key(pair[1], pair[0]))
        ]
        sanitized = [_fit(item, limits.max_item_chars) for item in ordered]
        recent = tuple(sanitized[-limits.max_recent_experiments :][::-1])

        grouped: Dict[str, List[Tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
        for item, raw in zip(sanitized, ordered):
            grouped.setdefault(_operator(raw), []).append((item, raw))
        findings: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
        for operator, values in sorted(grouped.items()):
            failures = [item for item, raw in reversed(values) if _failure_type(raw)]
            successes = [item for item, raw in reversed(values) if not _failure_type(raw)]
            selected = (failures[:1] + successes)[: limits.max_operator_findings]
            if len(selected) < limits.max_operator_findings:
                selected.extend(failures[1 : limits.max_operator_findings - len(selected) + 1])
            findings[operator] = tuple(selected[: limits.max_operator_findings])

        failures: Dict[str, int] = {}
        for item in ordered:
            failure = _failure_type(item)
            if failure:
                failures[failure] = failures.get(failure, 0) + 1
        failure_counts = dict(sorted(failures.items(), key=lambda pair: (-pair[1], pair[0]))[: limits.max_failures])

        observation_ids = []
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            value = item.get("observation_id")
            if value and str(value) not in observation_ids:
                observation_ids.append(str(value))
        frontier = tuple(_fit(item, limits.max_item_chars) for item in list(pareto)[-limits.max_frontier :])
        telemetry_values = list(telemetry)
        bounded_telemetry = _fit(telemetry_values[-limits.max_frontier :], limits.max_item_chars)
        if not isinstance(bounded_telemetry, list):
            bounded_telemetry = [bounded_telemetry]

        content = {
            "schema_version": 1,
            "source_observation_ids": observation_ids,
            "recent_experiments": list(recent),
            "operator_findings": {key: list(value) for key, value in findings.items()},
            "pareto_frontier": list(frontier),
            "failure_counts": failure_counts,
            "telemetry": bounded_telemetry,
        }
        source_digest = "sha256:" + hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            schema_version=1,
            source_observation_ids=tuple(observation_ids),
            source_digest=source_digest,
            recent_experiments=tuple(recent),
            operator_findings=findings,
            pareto_frontier=frontier,
            failure_counts=failure_counts,
            telemetry={"items": tuple(bounded_telemetry)},
            created_at=_now(),
        )

    def to_context(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_observation_ids": list(self.source_observation_ids),
            "source_digest": self.source_digest,
            "recent_experiments": [dict(item) for item in self.recent_experiments],
            "operator_findings": {
                str(key): [dict(item) for item in values]
                for key, values in self.operator_findings.items()
            },
            "pareto_frontier": [dict(item) for item in self.pareto_frontier],
            "failure_counts": dict(self.failure_counts),
            "telemetry": {"items": [dict(item) if isinstance(item, Mapping) else item for item in self.telemetry["items"]]},
            "created_at": self.created_at,
        }


__all__ = ["DigestLimits", "DiscoveryDigest"]
