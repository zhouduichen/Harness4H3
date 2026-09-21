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

    @property
    def model_id(self) -> str:
        return self.id

    @property
    def parent_model_id(self) -> Optional[str]:
        return self.parent_id

    @property
    def architecture(self) -> str:
        return self.state.architecture_name

    @property
    def parameter_count(self) -> Optional[int]:
        return self.state.parameter_count

    @property
    def quantization(self) -> Mapping[str, Any]:
        return self.state.quantization

    @property
    def training_method(self) -> Optional[str]:
        value = self.metadata.get("training_method") if isinstance(self.metadata, Mapping) else None
        value = value or self.state.provenance.get("training_method")
        return str(value) if value else None

    @property
    def algorithm_state(self) -> Mapping[str, Any]:
        return self.state.algorithm_state

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.state.provenance

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
