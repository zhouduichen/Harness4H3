from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..h3.state import ModelState
from ..operators.base import OperatorRegistry, OperatorValidationError
from ..target.profile import TargetProfile
from .schemas import BudgetState, CostEstimate, ExperimentPlan


class PlanValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SchemaValidator:
    def validate(self, raw: Any) -> ExperimentPlan:
        try:
            plan = raw if isinstance(raw, ExperimentPlan) else ExperimentPlan.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanValidationError("schema_invalid", str(exc))
        if not re.fullmatch(r"exp_[A-Za-z0-9_.-]+", plan.experiment_id):
            raise PlanValidationError("schema_invalid", "invalid experiment_id")
        required_text = (plan.diagnosis, plan.objective, plan.hypothesis, plan.operator, plan.rationale)
        if any(not value.strip() for value in required_text):
            raise PlanValidationError("schema_invalid", "plan text fields must be non-empty")
        if not plan.expected_effects or not plan.acceptance:
            raise PlanValidationError("schema_invalid", "expected_effects and acceptance must be non-empty")
        return plan


class PolicyValidator:
    FORBIDDEN_ARGUMENT_KEYS = {
        "argv",
        "benchmark",
        "code",
        "command",
        "cwd",
        "env",
        "environment",
        "evaluator",
        "hard_threshold",
        "script",
        "shell",
        "source",
        "target_profile",
    }

    def validate(self, plan: ExperimentPlan, parent: ModelState) -> None:
        if plan.parent_model_id != parent.model_id:
            raise PlanValidationError("policy_invalid", "plan parent does not match current model")

        def inspect(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if str(key).lower() in self.FORBIDDEN_ARGUMENT_KEYS:
                        raise PlanValidationError("policy_invalid", "forbidden operator argument %s" % key)
                    inspect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    inspect(item)

        inspect(plan.operator_args)


class BudgetValidator:
    def validate_declared(self, plan: ExperimentPlan, budget: BudgetState) -> CostEstimate:
        try:
            cost = CostEstimate.from_mapping(plan.required_budget)
        except (TypeError, ValueError) as exc:
            raise PlanValidationError("budget_invalid", str(exc))
        if not budget.can_afford(cost, controller_calls=1):
            raise PlanValidationError("budget_exhausted", "declared experiment cost exceeds remaining budget")
        return cost

    def validate_estimate(self, declared: CostEstimate, estimate: CostEstimate, budget: BudgetState) -> CostEstimate:
        if declared.wall_time_s < estimate.wall_time_s or declared.gpu_hours < estimate.gpu_hours:
            raise PlanValidationError("budget_invalid", "plan underestimates registered operator cost")
        if not budget.can_afford(estimate, controller_calls=1):
            raise PlanValidationError("budget_exhausted", "operator estimate exceeds remaining budget")
        return estimate


class OperatorValidator:
    def validate(self, plan: ExperimentPlan, parent: ModelState, target: TargetProfile, registry: OperatorRegistry) -> CostEstimate:
        try:
            registry.validate(plan.operator, parent, plan.operator_args, target)
            return registry.estimate_cost(plan.operator, parent, plan.operator_args, target)
        except OperatorValidationError as exc:
            raise PlanValidationError("operator_invalid", str(exc))


@dataclass(frozen=True)
class ValidatedPlan:
    plan: ExperimentPlan
    estimated_cost: CostEstimate


class ValidationPipeline:
    def __init__(self) -> None:
        self.schema = SchemaValidator()
        self.policy = PolicyValidator()
        self.budget = BudgetValidator()
        self.operator = OperatorValidator()

    def validate(
        self,
        raw: Any,
        parent: ModelState,
        target: TargetProfile,
        budget: BudgetState,
        registry: OperatorRegistry,
    ) -> ValidatedPlan:
        plan = self.schema.validate(raw)
        self.policy.validate(plan, parent)
        declared = self.budget.validate_declared(plan, budget)
        estimate = self.operator.validate(plan, parent, target, registry)
        self.budget.validate_estimate(declared, estimate, budget)
        return ValidatedPlan(plan, estimate)
