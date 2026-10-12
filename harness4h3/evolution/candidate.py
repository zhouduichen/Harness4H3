from __future__ import annotations

import hashlib
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

    def __post_init__(self) -> None:
        if not self.algorithm_id.startswith("A") or not self.source.strip():
            raise ValueError("algorithm candidate requires an id and source")
        if self.parent_algorithm_id is not None and not str(self.parent_algorithm_id).startswith("A"):
            raise ValueError("algorithm parent must be an algorithm id")
        actual = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        if actual != self.source_sha256:
            raise ValueError("algorithm source digest mismatch")
        if self.entrypoint != "propose":
            raise ValueError("algorithm entrypoint must be propose")
        if self.status not in {"proposed", "executed", "accepted", "rejected", "failed"}:
            raise ValueError("unsupported algorithm candidate status")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_source(cls, algorithm_id: str, source: str, *, parent_id: Optional[str] = None, parent_algorithm_id: Optional[str] = None, hypothesis: str = "", metadata: Optional[Mapping[str, Any]] = None) -> "AlgorithmCandidate":
        return cls(algorithm_id, parent_id, source, hashlib.sha256(source.encode("utf-8")).hexdigest(), parent_algorithm_id=parent_algorithm_id, hypothesis=hypothesis, metadata=dict(metadata or {}))

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
        )
