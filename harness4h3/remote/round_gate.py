"""Fixed, non-LLM acceptance gate for one remote H3 round."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Mapping, Tuple


RoundGateStatus = Literal["accepted", "rejected", "replan", "waiting"]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _success(value: Any) -> bool:
    return str(value or "").strip().lower() in {"success", "succeeded", "ok", "completed"}


@dataclass(frozen=True)
class RoundGateResult:
    """Durable gate result; it contains metadata, never model bytes."""

    status: RoundGateStatus
    reasons: Tuple[str, ...]
    evidence: Mapping[str, Any]
    retention: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return copy.deepcopy(value)


def _capability_reasons(worker: Mapping[str, Any], capabilities: Mapping[str, Any]) -> Tuple[str, ...]:
    operator = str(worker.get("operator") or "")
    args = _mapping(worker.get("operator_args"))
    requested = []
    if "lpl_target_steps" in args:
        requested.append("lpl")
    if "tdtm_merge_steps" in args or "tdtm_similarity_threshold" in args:
        requested.append("tdtm")
    if operator in {"lpl", "tdtm", "ci_dl"}:
        requested.append(operator)
    reasons = []
    for name in sorted(set(requested)):
        capability = _mapping(capabilities.get(name))
        if capability.get("safe_to_plan") is not True:
            reasons.append("capability_unverified:%s" % name)
            continue
        contract = _mapping(capability.get("execution_contract"))
        if not contract.get("workflow_hook") or contract.get("creates_checkpoint") is not False:
            reasons.append("capability_contract_missing:%s" % name)
    return tuple(reasons)


def evaluate_round_gate(
    *,
    worker_result: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    target: Mapping[str, Any],
    capability_evidence: Mapping[str, Any],
    lane_evidence: Mapping[str, Any],
    parent_digest: Any,
    child_digest: Any,
    retention: Mapping[str, Any],
) -> RoundGateResult:
    """Apply fixed evidence and target rules without asking the Controller."""

    worker = _mapping(worker_result)
    measured = _mapping(worker.get("metrics"))
    eval_value = _mapping(evaluation)
    target_value = _mapping(target)
    lane = _mapping(lane_evidence)
    reasons = []

    if str(lane.get("status") or "").lower() in {"waiting", "resource_wait", "dependency_wait"}:
        reasons.append("resource_dependency_wait")
        return RoundGateResult("waiting", tuple(reasons), {"lane": dict(lane)}, dict(retention or {}))

    if not _success(worker.get("status")) or measured.get("real_worker") is not True:
        reasons.append("real_worker_evidence_missing")
    if measured.get("offline_simulation") is True:
        reasons.append("offline_simulation_not_allowed")
    if not parent_digest:
        reasons.append("parent_digest_missing")
    if not child_digest:
        reasons.append("child_digest_missing")
    if lane.get("disjoint") is not True:
        reasons.append("lane_disjointness_unproven")
    reasons.extend(_capability_reasons(worker, _mapping(capability_evidence)))
    if reasons:
        status: RoundGateStatus = "rejected" if any(
            reason in {"real_worker_evidence_missing", "offline_simulation_not_allowed", "child_digest_missing"}
            for reason in reasons
        ) else "replan"
        return RoundGateResult(
            status,
            tuple(dict.fromkeys(reasons)),
            {"worker": dict(worker), "lane": dict(lane)},
            dict(retention or {}),
        )

    if eval_value.get("failure_type"):
        return RoundGateResult(
            "replan",
            ("evaluation_failure:%s" % eval_value["failure_type"],),
            {"evaluation": dict(eval_value), "lane": dict(lane)},
            dict(retention or {}),
        )
    quality = eval_value.get("quality_score")
    if isinstance(quality, bool) or not isinstance(quality, (int, float)) or not math.isfinite(float(quality)):
        return RoundGateResult(
            "replan",
            ("quality_score_missing",),
            {"evaluation": dict(eval_value), "lane": dict(lane)},
            dict(retention or {}),
        )
    if eval_value.get("critical_regression") is True:
        return RoundGateResult(
            "rejected",
            ("critical_quality_regression",),
            {"evaluation": dict(eval_value), "lane": dict(lane)},
            dict(retention or {}),
        )
    if eval_value.get("feasible") is not True:
        return RoundGateResult(
            "rejected",
            tuple(str(item) for item in eval_value.get("violations", [])) or ("evaluation_infeasible",),
            {"evaluation": dict(eval_value), "lane": dict(lane)},
            dict(retention or {}),
        )
    minimum_quality = target_value.get("min_quality_score")
    if minimum_quality is not None and float(quality) < float(minimum_quality):
        return RoundGateResult(
            "rejected",
            ("quality_floor",),
            {"evaluation": dict(eval_value), "lane": dict(lane)},
            dict(retention or {}),
        )
    return RoundGateResult(
        "accepted",
        ("fixed_gates_satisfied",),
        {
            "worker": {key: worker[key] for key in ("experiment_id", "operator", "status") if key in worker},
            "evaluation": {
                key: eval_value[key]
                for key in ("model_id", "system_id", "quality_score", "feasible", "violations")
                if key in eval_value
            },
            "lane": dict(lane),
            "parent_digest": str(parent_digest),
            "child_digest": str(child_digest),
        },
        dict(retention or {}),
    )


__all__ = ["RoundGateResult", "RoundGateStatus", "evaluate_round_gate"]
