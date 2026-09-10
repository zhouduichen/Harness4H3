from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional

from ..h3.state import ModelState
from .model_candidate import ModelCandidate, MODEL_ID_PATTERN


SYSTEM_ID_PATTERN = re.compile(r"C[0-9]{4,}")


@dataclass(frozen=True)
class SystemCandidate:
    """Immutable composition of a model candidate and runtime/algorithm state.

    ``model_ref`` points at the immutable ``ModelCandidate`` checkpoint. A
    runtime-only experiment therefore creates a new ``C…`` identity without
    copying or pretending to change model weights.
    """

    id: str
    parent_id: Optional[str]
    generation: int
    model_ref: str
    algorithm_state: Mapping[str, Any] = field(default_factory=dict)
    runtime_state: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    created_by_experiment_id: Optional[str] = None
    status: str = "candidate"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not SYSTEM_ID_PATTERN.fullmatch(self.id):
            raise ValueError("invalid system candidate id %r" % self.id)
        if self.parent_id is not None and not SYSTEM_ID_PATTERN.fullmatch(self.parent_id):
            raise ValueError("invalid parent system candidate id %r" % self.parent_id)
        if not MODEL_ID_PATTERN.fullmatch(self.model_ref):
            raise ValueError("invalid model reference %r" % self.model_ref)
        if self.generation < 0:
            raise ValueError("system candidate generation must be non-negative")
        if not self.status.strip():
            raise ValueError("system candidate status must be non-empty")

    @classmethod
    def from_model_candidate(
        cls,
        system_id: str,
        model: ModelCandidate,
        *,
        parent_id: Optional[str] = None,
        generation: int = 0,
        algorithm_state: Optional[Mapping[str, Any]] = None,
        runtime_state: Optional[Mapping[str, Any]] = None,
        evaluation: Optional[Mapping[str, Any]] = None,
        created_by_experiment_id: Optional[str] = None,
        status: str = "candidate",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SystemCandidate":
        return cls(
            id=system_id,
            parent_id=parent_id,
            generation=generation,
            model_ref=model.id,
            algorithm_state=copy.deepcopy(dict(algorithm_state or model.state.algorithm_state)),
            runtime_state=copy.deepcopy(dict(runtime_state or model.state.runtime_state)),
            evaluation=copy.deepcopy(dict(evaluation or {})),
            created_by_experiment_id=created_by_experiment_id,
            status=status,
            metadata=copy.deepcopy(dict(metadata or {})),
        )

    def with_evaluation(self, evaluation: Mapping[str, Any], status: Optional[str] = None) -> "SystemCandidate":
        return replace(self, evaluation=copy.deepcopy(dict(evaluation)), status=status or self.status)

    def evaluation_state(self, model: ModelState) -> ModelState:
        """Return a benchmark view that keeps the referenced model identity."""
        return replace(
            model,
            algorithm_state=copy.deepcopy(dict(self.algorithm_state)),
            runtime_state=copy.deepcopy(dict(self.runtime_state)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SystemCandidate":
        return cls(
            id=str(raw["id"]),
            parent_id=str(raw["parent_id"]) if raw.get("parent_id") is not None else None,
            generation=int(raw["generation"]),
            model_ref=str(raw["model_ref"]),
            algorithm_state=copy.deepcopy(dict(raw.get("algorithm_state") or {})),
            runtime_state=copy.deepcopy(dict(raw.get("runtime_state") or {})),
            evaluation=copy.deepcopy(dict(raw.get("evaluation") or {})),
            created_by_experiment_id=(
                str(raw["created_by_experiment_id"]) if raw.get("created_by_experiment_id") else None
            ),
            status=str(raw.get("status", "candidate")),
            metadata=copy.deepcopy(dict(raw.get("metadata") or {})),
        )
