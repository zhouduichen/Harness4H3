"""Conservative evidence taxonomy for algorithm and search-space changes."""

from __future__ import annotations

import ast
import hashlib
from typing import Any, Mapping, Optional

from ..adapters.wan.contracts import stable_digest


CHANGE_CLASSES = {"parameter_search", "algorithm_rule_evolution", "compression_space_extension"}
RULE_FACTORIES = {"prune_ffn_by_weight_threshold", "generated_channel_rule"}


def _logic_digest(source: str) -> str:
    try:
        tree = ast.parse(str(source), mode="exec")
        target = next((item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "propose"), tree)
        return hashlib.sha256(ast.dump(target, include_attributes=False).encode("utf-8")).hexdigest()
    except (SyntaxError, TypeError, ValueError):
        return hashlib.sha256(str(source).encode("utf-8")).hexdigest()


def _operator_expressible(plan: Mapping[str, Any], active_names: set[str]) -> bool:
    operator = str(plan.get("operator", ""))
    if operator in active_names:
        return True
    if operator == "tica_prune":
        # TICA is a bounded orchestration rule over the existing structural
        # closure, not a newly reachable primitive transformation.
        args = plan.get("args") if isinstance(plan.get("args"), Mapping) else {}
        mode = str(args.get("mode", "block_plus_ffn"))
        return mode in {"block", "ffn", "block_plus_ffn"} and {"prune_blocks", "prune_ffn_channels"}.issubset(active_names)
    if operator == "compose":
        steps = plan.get("args", {}).get("steps", []) if isinstance(plan.get("args"), Mapping) else []
        return isinstance(steps, list) and all(
            isinstance(step, Mapping) and str(step.get("operator", "")) in active_names
            for step in steps
        )
    return False


def classify_algorithm_change(
    parent_source: str,
    child_source: str,
    parent_plan: Mapping[str, Any],
    child_plan: Mapping[str, Any],
    active_search_space: Mapping[str, Any],
    *,
    structural_change: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Classify a source/plan delta without treating a new name as novelty."""

    active_names = {
        str(item.get("name"))
        for item in active_search_space.get("operators", [])
        if isinstance(item, Mapping) and item.get("name")
    }
    parent_expressible = _operator_expressible(parent_plan, active_names)
    child_expressible = _operator_expressible(child_plan, active_names)
    source_changed = str(parent_source) != str(child_source)
    plan_changed = stable_digest(parent_plan) != stable_digest(child_plan)
    child_operator = str(child_plan.get("operator", ""))
    if not child_expressible:
        change_class = "compression_space_extension"
        reachable: bool | None = False
        counterexample = {
            "kind": "operator_outside_active_closure",
            "child_plan": dict(child_plan),
            "active_operator_names": sorted(active_names),
        }
        evidence_status = "counterexample_pending_structural_validation"
    elif not source_changed and plan_changed:
        change_class = "parameter_search"
        reachable = True
        counterexample = None
        evidence_status = "expressible_by_active_operator_closure"
    elif source_changed and (child_operator in RULE_FACTORIES or child_operator == "tica_prune"):
        change_class = "algorithm_rule_evolution"
        reachable = True
        counterexample = None
        evidence_status = (
            "rule_delegates_to_existing_structural_operator"
            if child_operator != "tica_prune"
            else "tica_orchestrates_existing_structural_closure"
        )
    elif source_changed and parent_expressible and child_expressible:
        change_class = "algorithm_rule_evolution"
        reachable = True
        counterexample = None
        evidence_status = "expressible_by_active_operator_closure"
    else:
        change_class = "parameter_search"
        reachable = child_expressible
        counterexample = None
        evidence_status = "source_and_plan_delta_not_structurally_novel"
    return {
        "change_class": change_class,
        "source_changed": source_changed,
        "source_logic_parent_sha256": _logic_digest(parent_source),
        "source_logic_child_sha256": _logic_digest(child_source),
        "plan_changed": plan_changed,
        "parent_reachable_by_existing_closure": parent_expressible,
        "reachable_by_existing_closure": reachable,
        "counterexample": counterexample,
        "evidence_status": evidence_status,
        "evidence_level": "cpu_fake_structure" if structural_change is not None else "sandbox_only",
        "structural_change": dict(structural_change or {}),
    }


def classify_search_space_proposal(
    spec: Mapping[str, Any],
    active_search_space: Mapping[str, Any],
    validation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Classify a proposal while withholding novelty claims until validation."""

    active_names = {
        str(item.get("name"))
        for item in active_search_space.get("operators", [])
        if isinstance(item, Mapping) and item.get("name")
    }
    name = str(spec.get("name", ""))
    factory = str(spec.get("trusted_factory_key", ""))
    if name in active_names:
        return {
            "change_class": "parameter_search",
            "reachable_by_existing_closure": True,
            "evidence_status": "duplicate_active_capability",
            "counterexample": None,
        }
    if factory in RULE_FACTORIES:
        return {
            "change_class": "algorithm_rule_evolution",
            "reachable_by_existing_closure": True,
            "evidence_status": "delegates_to_existing_structural_operator",
            "counterexample": None,
        }
    validation = dict(validation or {})
    counterexample = validation.get("structural_counterexample")
    proven = isinstance(counterexample, Mapping) and bool(validation.get("structure_outside_active_closure"))
    return {
        "change_class": "compression_space_extension",
        "reachable_by_existing_closure": False if proven else None,
        "evidence_status": "validated_structural_counterexample" if proven else "unproven_pending_adapter_validation",
        "counterexample": dict(counterexample) if isinstance(counterexample, Mapping) else None,
        "limitations": ["new trusted factory requires adapter capability and independent structure validation"],
    }


__all__ = ["CHANGE_CLASSES", "classify_algorithm_change", "classify_search_space_proposal"]
