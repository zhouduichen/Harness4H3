from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Type

from ..archive.model_candidate import ModelCandidate
from ..controller.schemas import CostEstimate, OperatorResult
from ..h3.state import ModelState
from ..target.profile import TargetProfile


class OperatorValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionContext:
    experiment_dir: Path
    child_model_id: str
    model_store: Optional[Any] = None


class Operator(Protocol):
    name: str
    description: str

    def schema(self) -> Mapping[str, Any]:
        ...

    def validate(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> None:
        ...

    def estimate_cost(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        ...

    def execute(self, parent: ModelCandidate, args: Mapping[str, Any], runtime: ExecutionContext) -> OperatorResult:
        ...


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: Dict[str, Operator] = {}

    def register(self, operator: Operator) -> None:
        if not operator.name or operator.name in self._operators:
            raise OperatorValidationError("operator name must be non-empty and unique")
        self._operators[operator.name] = operator

    def names(self) -> Tuple[str, ...]:
        return tuple(self._operators)

    def visible(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(
            {"name": item.name, "description": item.description, "input_schema": dict(item.schema())}
            for item in self._operators.values()
        )

    def get(self, name: str) -> Operator:
        try:
            return self._operators[name]
        except KeyError:
            raise OperatorValidationError("operator %r is not registered" % name)

    def validate(self, name: str, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> None:
        if not isinstance(args, Mapping):
            raise OperatorValidationError("operator_args must be a mapping")
        self.get(name).validate(parent, args, target)

    def estimate_cost(self, name: str, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        self.validate(name, parent, args, target)
        return self.get(name).estimate_cost(parent, args, target)

    def execute(
        self,
        name: str,
        parent: ModelCandidate,
        args: Mapping[str, Any],
        target: TargetProfile,
        runtime: ExecutionContext,
    ) -> OperatorResult:
        self.validate(name, parent.state, args, target)
        return self.get(name).execute(parent, args, runtime)
