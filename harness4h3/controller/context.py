from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, Mapping, Tuple

from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .schemas import BudgetState


@dataclass(frozen=True)
class ControllerContext:
    target_profile: TargetProfile
    current_model_state: ModelState
    budget_state: BudgetState
    available_operators: Tuple[Mapping[str, Any], ...]
    recent_experiments: List[Mapping[str, Any]] = field(default_factory=list)
    relevant_failures: List[Mapping[str, Any]] = field(default_factory=list)
    pareto_front: List[Mapping[str, Any]] = field(default_factory=list)
    validated_design_genes: List[Mapping[str, Any]] = field(default_factory=list)
    validated_evaluation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)
