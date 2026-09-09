from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..h3.state import ModelState


MODEL_ID_PATTERN = re.compile(r"M[0-9]{4,}")


@dataclass(frozen=True)
class ModelCandidate:
    id: str
    parent_id: Optional[str]
    generation: int
    checkpoint_path: str
    state: ModelState
    created_by_experiment_id: Optional[str]
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not MODEL_ID_PATTERN.fullmatch(self.id):
            raise ValueError("invalid model candidate id %r" % self.id)
        if self.parent_id is not None and not MODEL_ID_PATTERN.fullmatch(self.parent_id):
            raise ValueError("invalid parent model candidate id %r" % self.parent_id)
        if self.generation < 0:
            raise ValueError("candidate generation must be non-negative")
        if self.state.model_id != self.id or self.state.parent_model_id != self.parent_id:
            raise ValueError("candidate identity must match embedded model state")
        if self.checkpoint_path != self.state.checkpoint_path:
            raise ValueError("candidate checkpoint must match embedded model state")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelCandidate":
        return cls(
            id=str(raw["id"]),
            parent_id=str(raw["parent_id"]) if raw.get("parent_id") is not None else None,
            generation=int(raw["generation"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            state=ModelState.from_dict(raw["state"]),
            created_by_experiment_id=str(raw["created_by_experiment_id"]) if raw.get("created_by_experiment_id") else None,
            status=str(raw["status"]),
            metadata=copy.deepcopy(dict(raw.get("metadata") or {})),
        )
