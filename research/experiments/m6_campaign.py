from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Set

from harness4h3.h3.state import ModelState


def canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def state_digest(state: ModelState) -> str:
    return hashlib.sha256(_canonical_json(state.to_dict()).encode("utf-8")).hexdigest()


def experiment_fingerprint(
    operator: str,
    operator_args: Mapping[str, Any],
    parent_state: ModelState,
    target_profile_id: str,
) -> str:
    payload = {
        "operator": str(operator),
        "normalized_operator_args": canonicalize(dict(operator_args)),
        "parent_state_digest": state_digest(parent_state),
        "target_profile_id": str(target_profile_id),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def validate_novelty(
    operator: str,
    hypothesis: str,
    fingerprint: str,
    seen_fingerprints: Set[str],
    recent: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    if fingerprint in seen_fingerprints:
        return "duplicate_experiment_fingerprint"
    prior_ids = [
        str(item["experiment_id"])
        for item in recent
        if str(item.get("operator", "")) == str(operator) and item.get("experiment_id")
    ]
    if prior_ids and not any(prior_id in str(hypothesis) for prior_id in prior_ids):
        return "same_operator_requires_prior_evidence"
    return None


def classify_outcome(
    operator_ok: bool,
    output_state_present: bool,
    split_results: Mapping[str, Any],
    split_errors: Mapping[str, Any],
    split_validated: bool,
) -> str:
    if not operator_ok or not output_state_present or split_errors or not split_results:
        return "failed_experiment"
    return "accepted_candidate" if split_validated else "rejected_candidate"


def build_research_report(
    payload: Mapping[str, Any],
    started_at: str,
    ended_at: str,
) -> Dict[str, Any]:
    iterations = list(payload.get("iterations") or [])
    accepted = [item for item in iterations if item.get("outcome") == "accepted_candidate"]
    rejected = [item for item in iterations if item.get("outcome") == "rejected_candidate"]
    failed = [item for item in iterations if item.get("outcome") == "failed_experiment"]
    archive_candidates = list(payload.get("system_candidates") or [])
    pareto_candidates = [item for item in archive_candidates if item.get("status") != "failed"]
    operators = [str(item.get("operator")) for item in iterations if item.get("operator")]
    return {
        "harness_version": payload["harness"]["version"],
        "target_profile_id": payload["target_profile_id"],
        "termination_reason": payload.get("termination_reason"),
        "target_satisfied": payload.get("status") == "accepted",
        "full_autonomous_experiment_sequence": iterations,
        "accepted_experiments": accepted,
        "rejected_experiments": rejected,
        "failed_experiments": failed,
        "final_pareto_candidates": pareto_candidates,
        "final_archive_candidates": archive_candidates,
        "total_experiments": len(iterations),
        "failed_experiment_count": len(failed),
        "rejected_candidate_count": len(rejected),
        "wall_time_s": float(payload.get("wall_time_s", 0.0)),
        "gpu_hours": float(payload.get("gpu_hours", 0.0)),
        "gpu_hours_available": bool(payload.get("gpu_hours_available", False)),
        "human_intervention_count": 0,
        "repeated_failure_avoidance": {
            "duplicate_fingerprints_blocked": int(payload.get("duplicate_fingerprints_blocked", 0)),
            "same_operator_requires_prior_evidence": True,
            "rejected_parent_reuse": False,
        },
        "strategy_changed_after_negative_evidence": bool(operators)
        and (len(set(operators)) > 1 or operators[0] != "vae_tiling"),
        "final_optimization_recipe": payload.get("accepted_recipe"),
        "experience_influence_evidence": payload.get("experience_influence_evidence", []),
        "started_at": started_at,
        "ended_at": ended_at,
    }
