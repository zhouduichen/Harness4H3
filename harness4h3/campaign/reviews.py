"""Bounded, structured pre-training review by independent agents."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from .base import ActorIdentity, CampaignBase, canonical_json
from .proposals import CandidateEnvelope, MUTATION_FIELDS


OBJECTION_CATEGORIES = frozenset(
    {
        "unsupported_assumption",
        "proxy_gaming",
        "goal_drift",
        "credit_assignment_error",
        "repeated_failed_design",
        "evaluator_blind_spot",
        "resource_mismatch",
        "architecture_algorithm_incompatibility",
        "target_device_mismatch",
    }
)


class ReviewContractError(ValueError):
    """Raised when an agent emits malformed or unsafe review data."""


class ReviewIdentityError(ReviewContractError):
    """Raised when review actors violate the trust boundary."""


class ReviewAgent(Protocol):
    identity: ActorIdentity

    def review(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewContractError("%s must be a non-empty string" % name)
    return value.strip()


def _string_array(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ReviewContractError("%s must be an array of non-empty strings" % name)
    return tuple(dict.fromkeys(item.strip() for item in value))


def _number_mapping(value: Any, name: str) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ReviewContractError("%s must be a mapping" % name)
    result = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ReviewContractError("%s must contain finite numeric values" % name)
        result[str(key)] = float(item)
    return result


def _strict_keys(raw: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    unknown = sorted(set(raw) - set(expected))
    missing = sorted(set(expected) - set(raw))
    if unknown:
        raise ReviewContractError("%s has unknown field(s): %s" % (name, ", ".join(map(str, unknown))))
    if missing:
        raise ReviewContractError("%s is missing field(s): %s" % (name, ", ".join(missing)))


@dataclass(frozen=True)
class AdvocateReport:
    bottleneck: str
    changed_fields: Tuple[str, ...]
    expected_metric_delta: Mapping[str, float]
    supporting_evidence_ids: Tuple[str, ...]
    falsification_experiment: str
    resource_assumptions: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AdvocateReport":
        if not isinstance(raw, Mapping):
            raise ReviewContractError("advocate report must be a mapping")
        fields = (
            "bottleneck", "changed_fields", "expected_metric_delta", "supporting_evidence_ids",
            "falsification_experiment", "resource_assumptions",
        )
        _strict_keys(raw, fields, "advocate report")
        if not isinstance(raw["resource_assumptions"], Mapping):
            raise ReviewContractError("resource_assumptions must be a mapping")
        canonical_json(raw["resource_assumptions"])
        return cls(
            bottleneck=_required_string(raw["bottleneck"], "bottleneck"),
            changed_fields=_string_array(raw["changed_fields"], "changed_fields"),
            expected_metric_delta=_number_mapping(raw["expected_metric_delta"], "expected_metric_delta"),
            supporting_evidence_ids=_string_array(raw["supporting_evidence_ids"], "supporting_evidence_ids"),
            falsification_experiment=_required_string(raw["falsification_experiment"], "falsification_experiment"),
            resource_assumptions=copy.deepcopy(dict(raw["resource_assumptions"])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bottleneck": self.bottleneck,
            "changed_fields": list(self.changed_fields),
            "expected_metric_delta": dict(self.expected_metric_delta),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "falsification_experiment": self.falsification_experiment,
            "resource_assumptions": copy.deepcopy(dict(self.resource_assumptions)),
        }


@dataclass(frozen=True)
class CriticalReport:
    objections: Tuple[str, ...]
    objection_categories: Tuple[str, ...]
    missing_evidence_ids: Tuple[str, ...]
    proxy_gaming_risks: Tuple[str, ...]
    target_device_risks: Tuple[str, ...]
    required_revisions: Tuple[str, ...]
    hard_objection: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CriticalReport":
        if not isinstance(raw, Mapping):
            raise ReviewContractError("critical report must be a mapping")
        fields = (
            "objections", "objection_categories", "missing_evidence_ids", "proxy_gaming_risks",
            "target_device_risks", "required_revisions", "hard_objection",
        )
        _strict_keys(raw, fields, "critical report")
        categories = _string_array(raw["objection_categories"], "objection_categories")
        unknown = sorted(set(categories) - OBJECTION_CATEGORIES)
        if unknown:
            raise ReviewContractError("unknown objection category: %s" % ", ".join(unknown))
        if not isinstance(raw["hard_objection"], bool):
            raise ReviewContractError("hard_objection must be boolean")
        return cls(
            objections=_string_array(raw["objections"], "objections"),
            objection_categories=categories,
            missing_evidence_ids=_string_array(raw["missing_evidence_ids"], "missing_evidence_ids"),
            proxy_gaming_risks=_string_array(raw["proxy_gaming_risks"], "proxy_gaming_risks"),
            target_device_risks=_string_array(raw["target_device_risks"], "target_device_risks"),
            required_revisions=_string_array(raw["required_revisions"], "required_revisions"),
            hard_objection=raw["hard_objection"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objections": list(self.objections),
            "objection_categories": list(self.objection_categories),
            "missing_evidence_ids": list(self.missing_evidence_ids),
            "proxy_gaming_risks": list(self.proxy_gaming_risks),
            "target_device_risks": list(self.target_device_risks),
            "required_revisions": list(self.required_revisions),
            "hard_objection": self.hard_objection,
        }


@dataclass(frozen=True)
class RevisionRecord:
    candidate: CandidateEnvelope
    base_digest: str
    changed_fields: Tuple[str, ...]
    resolved_objection_ids: Tuple[str, ...]
    reason: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RevisionRecord":
        if not isinstance(raw, Mapping):
            raise ReviewContractError("revision must be a mapping")
        fields = {"candidate", "base_digest", "changed_fields", "resolved_objection_ids", "reason"}
        _strict_keys(raw, fields, "revision")
        changed = _string_array(raw["changed_fields"], "changed_fields")
        invalid = sorted(set(changed) - MUTATION_FIELDS)
        if invalid:
            raise ReviewContractError("revision changed field is not registered: %s" % ", ".join(invalid))
        return cls(
            candidate=CandidateEnvelope.from_dict(raw["candidate"]),
            base_digest=_required_string(raw["base_digest"], "revision.base_digest"),
            changed_fields=changed,
            resolved_objection_ids=_string_array(raw["resolved_objection_ids"], "resolved_objection_ids"),
            reason=_required_string(raw["reason"], "revision.reason"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "base_digest": self.base_digest,
            "changed_fields": list(self.changed_fields),
            "resolved_objection_ids": list(self.resolved_objection_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CandidateReview:
    candidate_id: str
    advocate: AdvocateReport
    critical_rounds: Tuple[CriticalReport, ...]
    revision: Optional[RevisionRecord]
    final_critical: CriticalReport
    approved: bool
    rejection_reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "advocate": self.advocate.to_dict(),
            "critical_rounds": [item.to_dict() for item in self.critical_rounds],
            "revision": self.revision.to_dict() if self.revision else None,
            "final_critical": self.final_critical.to_dict(),
            "approved": self.approved,
            "rejection_reasons": list(self.rejection_reasons),
        }


class ReviewPipeline:
    def __init__(
        self,
        advocate: ReviewAgent,
        critical: ReviewAgent,
        modifier: ReviewAgent,
        base: CampaignBase,
        max_rounds: int = 2,
    ):
        if isinstance(max_rounds, bool) or int(max_rounds) <= 0:
            raise ReviewContractError("max_rounds must be positive")
        self.advocate = advocate
        self.critical = critical
        self.modifier = modifier
        self.base = base
        self.max_rounds = int(max_rounds)
        self._validate_identities()

    def _validate_identities(self) -> None:
        agents = (self.advocate, self.critical, self.modifier)
        identities = []
        for agent in agents:
            identity = getattr(agent, "identity", None)
            if not isinstance(identity, ActorIdentity):
                raise ReviewIdentityError("review agent identity must be ActorIdentity")
            identities.append(identity)
        if len(set(identities)) != len(identities):
            raise ReviewIdentityError("review agent identities must be distinct")
        if self.base.controller_identity in identities or self.base.evaluator_identity in identities:
            raise ReviewIdentityError("review agent identities must be distinct from controller and evaluator")

    def _request(self, candidate: CandidateEnvelope, context: Mapping[str, Any], *, phase: str, round_index: int) -> Dict[str, Any]:
        request = copy.deepcopy(dict(context)) if isinstance(context, Mapping) else {}
        request.update(
            {
                "phase": phase,
                "review_round": round_index,
                "base_digest": self.base.digest,
                "candidate": candidate.to_dict(),
            }
        )
        return request

    def review(self, candidate: CandidateEnvelope, context: Mapping[str, Any]) -> CandidateReview:
        advocate = AdvocateReport.from_dict(
            self.advocate.review(self._request(candidate, context, phase="advocate", round_index=0))
        )
        current = candidate
        critical_rounds = []
        revision = None
        final_critical = None
        for round_index in range(self.max_rounds + 1):
            critical = CriticalReport.from_dict(
                self.critical.review(
                    self._request(current, context, phase="critical", round_index=round_index)
                )
            )
            critical_rounds.append(critical)
            if not critical.hard_objection and not critical.required_revisions:
                final_critical = CriticalReport.from_dict(
                    self.critical.review(
                        self._request(current, context, phase="final_critical", round_index=round_index)
                    )
                )
                break
            if round_index >= self.max_rounds:
                break
            revision = RevisionRecord.from_dict(
                self.modifier.review(
                    self._request(current, context, phase="revision", round_index=round_index)
                )
            )
            if revision.base_digest != self.base.digest:
                raise ReviewContractError("revision cannot change the immutable campaign base")
            if revision.candidate.candidate_id != candidate.candidate_id:
                raise ReviewContractError("revision cannot change candidate_id")
            current = revision.candidate
        if final_critical is None:
            final_critical = critical_rounds[-1]
        reasons = []
        if final_critical.hard_objection:
            reasons.append("unresolved_hard_objection")
        elif final_critical.required_revisions:
            reasons.append("unresolved_required_revision")
        return CandidateReview(
            candidate_id=candidate.candidate_id,
            advocate=advocate,
            critical_rounds=tuple(critical_rounds),
            revision=revision,
            final_critical=final_critical,
            approved=not reasons,
            rejection_reasons=tuple(reasons),
        )


__all__ = [
    "AdvocateReport",
    "CandidateReview",
    "CriticalReport",
    "OBJECTION_CATEGORIES",
    "ReviewAgent",
    "ReviewContractError",
    "ReviewIdentityError",
    "ReviewPipeline",
    "RevisionRecord",
]
