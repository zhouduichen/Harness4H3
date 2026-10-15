from __future__ import annotations

import hashlib
import difflib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CompressionPlan:
    operator: str
    args: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"operator": self.operator, "args": dict(self.args)}


@dataclass(frozen=True)
class AlgorithmCandidate:
    algorithm_id: str
    parent_id: Optional[str]
    source: str
    source_sha256: str
    parent_algorithm_id: Optional[str] = None
    entrypoint: str = "propose"
    status: str = "proposed"
    hypothesis: str = ""
    experiment_ids: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    created_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parent_source_sha256: Optional[str] = None
    source_diff: str = ""
    change_summary: str = ""
    change_rationale: str = ""
    evidence_ids: tuple[str, ...] = ()
    validation_status: str = "unvalidated"
    validation_record_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.algorithm_id.startswith("A") or not self.source.strip():
            raise ValueError("algorithm candidate requires an id and source")
        if self.parent_algorithm_id is not None and not str(self.parent_algorithm_id).startswith("A"):
            raise ValueError("algorithm parent must be an algorithm id")
        actual = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        if actual != self.source_sha256:
            raise ValueError("algorithm source digest mismatch")
        if self.parent_source_sha256 is not None and len(str(self.parent_source_sha256)) != 64:
            raise ValueError("parent algorithm source digest must be SHA-256")
        if self.entrypoint != "propose":
            raise ValueError("algorithm entrypoint must be propose")
        if self.status not in {"proposed", "executed", "accepted", "rejected", "failed"}:
            raise ValueError("unsupported algorithm candidate status")
        if self.validation_status not in {"unvalidated", "pending", "validated", "rejected", "failed"}:
            raise ValueError("unsupported algorithm validation status")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_source(
        cls,
        algorithm_id: str,
        source: str,
        *,
        parent_id: Optional[str] = None,
        parent_algorithm_id: Optional[str] = None,
        hypothesis: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        parent_source_sha256: Optional[str] = None,
        source_diff: str = "",
        change_summary: str = "",
        change_rationale: str = "",
        evidence_ids: tuple[str, ...] = (),
        validation_status: str = "unvalidated",
        validation_record_path: Optional[str] = None,
    ) -> "AlgorithmCandidate":
        return cls(
            algorithm_id,
            parent_id,
            source,
            hashlib.sha256(source.encode("utf-8")).hexdigest(),
            parent_algorithm_id=parent_algorithm_id,
            hypothesis=hypothesis,
            metadata=dict(metadata or {}),
            parent_source_sha256=parent_source_sha256,
            source_diff=str(source_diff),
            change_summary=str(change_summary),
            change_rationale=str(change_rationale),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            validation_status=str(validation_status),
            validation_record_path=validation_record_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlgorithmCandidate":
        return cls(
            algorithm_id=str(raw["algorithm_id"]),
            parent_id=str(raw["parent_id"]) if raw.get("parent_id") else None,
            source=str(raw["source"]),
            source_sha256=str(raw["source_sha256"]),
            parent_algorithm_id=str(raw["parent_algorithm_id"]) if raw.get("parent_algorithm_id") else None,
            entrypoint=str(raw.get("entrypoint", "propose")),
            status=str(raw.get("status", "proposed")),
            hypothesis=str(raw.get("hypothesis", "")),
            experiment_ids=tuple(str(item) for item in raw.get("experiment_ids", [])),
            failure_reasons=tuple(str(item) for item in raw.get("failure_reasons", [])),
            created_at=str(raw.get("created_at", "")),
            metadata=dict(raw.get("metadata") or {}),
            parent_source_sha256=str(raw["parent_source_sha256"]) if raw.get("parent_source_sha256") else None,
            source_diff=str(raw.get("source_diff", "")),
            change_summary=str(raw.get("change_summary", "")),
            change_rationale=str(raw.get("change_rationale", "")),
            evidence_ids=tuple(str(item) for item in raw.get("evidence_ids", [])),
            validation_status=str(raw.get("validation_status", "unvalidated")),
            validation_record_path=str(raw["validation_record_path"]) if raw.get("validation_record_path") else None,
        )


def build_source_diff(parent_source: str, child_source: str, *, parent_id: str, child_id: str) -> str:
    """Return the auditable source-level change between two algorithm versions."""

    return "".join(
        difflib.unified_diff(
            str(parent_source).splitlines(keepends=True),
            str(child_source).splitlines(keepends=True),
            fromfile="%s.py" % parent_id,
            tofile="%s.py" % child_id,
        )
    )
