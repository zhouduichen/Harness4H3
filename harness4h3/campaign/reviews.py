"""Bounded, structured pre-training review by independent agents."""

from __future__ import annotations

import copy
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from .base import ActorIdentity, CampaignBase, canonical_json
from .proposals import CandidateEnvelope, MUTATION_FIELDS
from .revision import (
    RevisionPatch,
    RevisionPatchError,
    apply_revision_patch,
    canonical_candidate_from_patch,
    canonical_candidate_from_proposal,
    proposal_from_candidate,
)


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

_OBJECTION_CATEGORY_ALIASES = {
    "architecture/algorithm_incompatibility": "architecture_algorithm_incompatibility",
}


class ReviewContractError(ValueError):
    """Raised when an agent emits malformed or unsafe review data."""


class ReviewIdentityError(ReviewContractError):
    """Raised when review actors violate the trust boundary."""


class ReviewLLMError(ReviewContractError):
    """Raised when a structured LLM review cannot be obtained or decoded."""


class ReviewAgent(Protocol):
    identity: ActorIdentity

    def review(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


def review_json_schema(role: str) -> Mapping[str, Any]:
    """Return the closed JSON schema for one independent review role."""

    role = str(role).strip().lower()
    string_array = {"type": "array", "items": {"type": "string"}}
    if role == "advocate":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bottleneck": {"type": "string", "minLength": 1},
                "changed_fields": string_array,
                "expected_metric_delta": {"type": "object", "additionalProperties": {"type": "number"}},
                "supporting_evidence_ids": string_array,
                "falsification_experiment": {"type": "string", "minLength": 1},
                "resource_assumptions": {"type": "object"},
            },
            "required": [
                "bottleneck", "changed_fields", "expected_metric_delta", "supporting_evidence_ids",
                "falsification_experiment", "resource_assumptions",
            ],
        }
    if role == "critical":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "objections": string_array,
                "objection_categories": string_array,
                "missing_evidence_ids": string_array,
                "proxy_gaming_risks": string_array,
                "target_device_risks": string_array,
                "required_revisions": string_array,
                "hard_objection": {"type": "boolean"},
            },
            "required": [
                "objections", "objection_categories", "missing_evidence_ids", "proxy_gaming_risks",
                "target_device_risks", "required_revisions", "hard_objection",
            ],
        }
    if role == "revision":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {"type": "string", "minLength": 1},
                "base_digest": {"type": "string", "minLength": 1},
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "op": {"enum": ["add", "replace"]},
                            "path": {"type": "string", "pattern": "^/(architecture|training|deployment)/[^/]+$"},
                            "value": {},
                        },
                        "required": ["op", "path", "value"],
                    },
                },
                "changed_fields": string_array,
                "resolved_objection_ids": string_array,
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["candidate_id", "base_digest", "operations", "changed_fields", "resolved_objection_ids", "reason"],
        }
    raise ReviewContractError("unsupported review role: %s" % role)


def _review_prompt(role: str, request: Mapping[str, Any]) -> str:
    candidate = request.get("candidate")
    context_request = dict(request)
    context_request.pop("candidate", None)
    context = json.dumps(context_request, ensure_ascii=False, sort_keys=True)
    if role == "advocate":
        instruction = (
            "Act as the Advocate for this candidate. Build the strongest evidence-grounded case "
            "for the proposed design, identify its actual bottleneck, and predict numeric metric "
            "deltas. Do not invent evidence or alter the target profile/verifier bank. "
            "Return exactly these keys: bottleneck (string), changed_fields (string array), "
            "expected_metric_delta (numeric object), supporting_evidence_ids (string array), "
            "falsification_experiment (string), resource_assumptions (object)."
        )
    elif role == "critical":
        instruction = (
            "Act as an independent Critical reviewer. Assume the Controller candidate is wrong "
            "until evidence proves otherwise. Actively search for unsupported assumption, proxy "
            "gaming, goal drift, repeated failed design, resource mismatch, evaluator blind spot, "
            "architecture/algorithm incompatibility, and target-device mismatch. You have no right "
            "to modify TargetProfile, VerifierBank, or the final Gate; report objections only. "
            "Return exactly these keys: objections (string array), objection_categories (string "
            "array using only unsupported_assumption, proxy_gaming, goal_drift, "
            "credit_assignment_error, repeated_failed_design, evaluator_blind_spot, "
            "resource_mismatch, architecture_algorithm_incompatibility, target_device_mismatch), "
            "missing_evidence_ids (string array), proxy_gaming_risks (string array), "
            "target_device_risks (string array), required_revisions (string array), "
            "hard_objection (boolean)."
        )
    else:
        instruction = (
            "Act as an independent Revision agent. Return only a RevisionPatch with JSON-Pointer "
            "operations over the executable proposal roots /architecture, /training, or "
            "/deployment. The Candidate display uses architecture, training_recipe, and "
            "deployment_recipe names, but patch paths MUST use /architecture, /training, and "
            "/deployment; never use *_recipe roots. Preserve candidate_id "
            "and immutable campaign base digest. Never return a complete CandidateEnvelope, edit "
            "parent/generation/Teacher identity, TargetProfile, VerifierBank, or any final gate. "
            "Return exactly these keys: candidate_id (string), base_digest (string), operations "
            "(array of objects with op/path/value), changed_fields (string array), "
            "resolved_objection_ids (string array), reason (string)."
        )
    return (
        instruction
        + " Return exactly one JSON object matching the supplied schema, with no markdown or prose outside JSON.\n"
        + "CANDIDATE=" + json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        + "\nREQUEST=" + context
    )


class StructuredLLMReviewAgent:
    """One role-specific structured-output LLM reviewer.

    The three campaign roles use separate instances, identities, and prompts.
    The response is parsed and then validated by the typed report classes and
    CandidateEnvelope parser; the LLM cannot mutate trusted campaign objects.
    """

    def __init__(
        self,
        identity: ActorIdentity,
        role: str,
        *,
        model_name: str,
        base_url: str,
        provider: str = "openai_compatible",
        timeout_s: float = 180.0,
    ) -> None:
        self.identity = identity
        self.role = str(role).strip().lower()
        review_json_schema(self.role)
        self.model_name = str(model_name)
        self.base_url = str(base_url).rstrip("/")
        self.provider = str(provider).strip().lower().replace("-", "_")
        self.timeout_s = float(timeout_s)
        if not self.model_name or not self.base_url or self.timeout_s <= 0:
            raise ValueError("structured review agent configuration is invalid")

    def _endpoint_and_payload(self, request: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
        schema = review_json_schema(self.role)
        prompt = _review_prompt(self.role, request)
        if self.provider == "ollama":
            endpoint = self.base_url + "/api/chat"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "format": schema,
                "options": {"temperature": 0.0},
            }
        else:
            endpoint = self.base_url + ("/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions")
            # Critical reports contain several independent arrays and are
            # routinely longer than an Advocate/Revision report.  A shared
            # 512-token cap lets vLLM truncate a valid JSON object halfway
            # through (finish_reason=length), which then looks like a parser
            # failure and needlessly replans the experiment.
            max_tokens = 1024 if self.role == "critical" else 512
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "Return JSON only. Do not emit markdown, tools, or executable code."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                # vLLM's nested guided-json compiler is slow on this remote
                # controller. The typed report parser below remains the
                # authoritative review contract.
                "response_format": {"type": "json_object"},
            }
        return endpoint, payload

    @staticmethod
    def _content(raw: Mapping[str, Any]) -> Any:
        message: Mapping[str, Any] = {}
        try:
            message = raw["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            try:
                content = raw["message"]["content"]
            except (KeyError, TypeError) as nested:
                raise ReviewLLMError("review response has no structured message content") from nested
        if content is None and isinstance(message, Mapping):
            content = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                value = item.get("text")
                if value is None:
                    value = item.get("content")
                if value is None:
                    value = item.get("json")
                if isinstance(value, Mapping):
                    value = json.dumps(dict(value), ensure_ascii=False)
                if value is not None:
                    parts.append(str(value))
            content = "".join(parts)
        if isinstance(content, str):
            try:
                return json.loads(content)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                # Some local checkpoints still wrap the object in a short
                # prose/preamble despite the JSON-only instruction. Extract
                # only the outermost object; typed role validation remains
                # authoritative after this compatibility step.
                start = content.find("{")
                end = content.rfind("}")
                if start >= 0 and end > start:
                    try:
                        return json.loads(content[start : end + 1])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                raise ReviewLLMError("review response content is not JSON") from exc
        return content

    def review(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error: Optional[ReviewLLMError] = None
        required = set(review_json_schema(self.role).get("required", ()))
        for attempt in range(2):
            if attempt:
                attempt_request = {
                    key: request[key]
                    for key in ("phase", "review_round", "base_digest", "candidate")
                    if key in request
                }
                attempt_request["_retry_instruction"] = (
                    "The previous review response was unusable. Use only the candidate core below. "
                    "Repeat exactly the required keys and return one JSON object with no prose or extra fields."
                )
            else:
                attempt_request = dict(request)
            endpoint, payload = self._endpoint_and_payload(attempt_request)
            http_request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                parsed = self._content(raw if isinstance(raw, Mapping) else {})
                if not isinstance(parsed, Mapping):
                    raise ReviewLLMError("%s review response must be a JSON object" % self.role)
                normalized = dict(parsed)
                # Older controller prompts called the advocate's bottleneck
                # narrative "advocacy_case" in addition to the typed field.
                if self.role == "advocate":
                    normalized.pop("advocacy_case", None)
                if attempt == 0 and (not required.issubset(normalized) or set(normalized) - required):
                    last_error = ReviewLLMError("%s review response did not match its required keys" % self.role)
                    continue
                return normalized
            except ReviewLLMError as exc:
                last_error = exc
                if attempt == 0:
                    continue
                raise
            except (OSError, urllib.error.URLError, urllib.error.HTTPError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = ReviewLLMError("%s review request failed: %s" % (self.role, exc))
                if attempt == 0:
                    continue
                raise last_error from exc
        if last_error is not None:
            raise last_error
        raise ReviewLLMError("%s review request failed without a response" % self.role)


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
        categories = tuple(
            _OBJECTION_CATEGORY_ALIASES.get(category, category)
            for category in _string_array(raw["objection_categories"], "objection_categories")
        )
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
    candidate: Optional[CandidateEnvelope]
    patch: RevisionPatch
    base_digest: str
    changed_fields: Tuple[str, ...]
    resolved_objection_ids: Tuple[str, ...]
    reason: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RevisionRecord":
        if not isinstance(raw, Mapping):
            raise ReviewContractError("revision must be a mapping")
        fields = {"candidate_id", "base_digest", "operations", "changed_fields", "resolved_objection_ids", "reason"}
        _strict_keys(raw, fields, "revision")
        try:
            patch = RevisionPatch.from_dict(raw)
        except RevisionPatchError as exc:
            raise ReviewContractError(str(exc)) from exc
        return cls(
            candidate=None,
            patch=patch,
            base_digest=patch.base_digest,
            changed_fields=patch.changed_fields,
            resolved_objection_ids=patch.resolved_objection_ids,
            reason=patch.reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.patch.to_dict(),
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
    final_candidate: Optional[CandidateEnvelope] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "advocate": self.advocate.to_dict(),
            "critical_rounds": [item.to_dict() for item in self.critical_rounds],
            "revision": self.revision.to_dict() if self.revision else None,
            "final_critical": self.final_critical.to_dict(),
            "approved": self.approved,
            "rejection_reasons": list(self.rejection_reasons),
            "final_candidate": self.final_candidate.to_dict() if self.final_candidate else None,
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
        candidate_payload = candidate.to_dict()
        provenance = candidate_payload.get("provenance")
        if isinstance(provenance, Mapping) and "student_proposal" in provenance:
            provenance = dict(provenance)
            provenance.pop("student_proposal", None)
            candidate_payload["provenance"] = provenance
        request.update(
            {
                "phase": phase,
                "review_round": round_index,
                "base_digest": self.base.digest,
                "candidate": candidate_payload,
            }
        )
        return request

    def review(self, candidate: CandidateEnvelope, context: Mapping[str, Any]) -> CandidateReview:
        advocate = AdvocateReport.from_dict(
            self.advocate.review(self._request(candidate, context, phase="advocate", round_index=0))
        )
        # The Advocate owns the prediction; persist it in the immutable
        # candidate envelope before Critical/Revision see the candidate.
        current = copy.copy(candidate)
        current = CandidateEnvelope(
            candidate_id=current.candidate_id,
            parent_candidate_id=current.parent_candidate_id,
            generation=current.generation,
            experiment_id=current.experiment_id,
            proposal_digest=current.proposal_digest,
            mutation_fields=current.mutation_fields,
            architecture=current.architecture,
            training_recipe=current.training_recipe,
            deployment_recipe=current.deployment_recipe,
            provenance=current.provenance,
            predicted_metric_delta=dict(advocate.expected_metric_delta),
        )
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
            patch_raw = self.modifier.review(
                self._request(current, context, phase="revision", round_index=round_index)
            )
            parsed_patch = RevisionPatch.from_dict(patch_raw)
            if parsed_patch.base_digest != self.base.digest:
                raise ReviewContractError("revision cannot change the immutable campaign base")
            if parsed_patch.candidate_id != candidate.candidate_id:
                raise ReviewContractError("revision cannot change candidate_id")
            try:
                source_proposal = proposal_from_candidate(current)
                revised_proposal = apply_revision_patch(source_proposal, parsed_patch)
                revised = canonical_candidate_from_proposal(current, revised_proposal, parsed_patch)
            except RevisionPatchError as exc:
                if current.provenance.get("student_proposal") is not None:
                    raise ReviewContractError("revision patch could not produce a canonical Student Candidate: %s" % exc) from exc
                try:
                    revised = canonical_candidate_from_patch(current, parsed_patch)
                except (TypeError, ValueError) as fallback_exc:
                    raise ReviewContractError("revision patch could not produce a canonical Candidate: %s" % fallback_exc) from fallback_exc
            except (TypeError, ValueError) as exc:
                raise ReviewContractError("revision patch could not produce a canonical Candidate: %s" % exc) from exc
            if dict(revised.predicted_metric_delta) != dict(advocate.expected_metric_delta):
                revised = CandidateEnvelope(
                    candidate_id=revised.candidate_id,
                    parent_candidate_id=revised.parent_candidate_id,
                    generation=revised.generation,
                    experiment_id=revised.experiment_id,
                    proposal_digest=revised.proposal_digest,
                    mutation_fields=revised.mutation_fields,
                    architecture=revised.architecture,
                    training_recipe=revised.training_recipe,
                    deployment_recipe=revised.deployment_recipe,
                    provenance=revised.provenance,
                    predicted_metric_delta=dict(advocate.expected_metric_delta),
                )
            revision = RevisionRecord(
                candidate=revised,
                patch=parsed_patch,
                base_digest=parsed_patch.base_digest,
                changed_fields=parsed_patch.changed_fields,
                resolved_objection_ids=parsed_patch.resolved_objection_ids,
                reason=parsed_patch.reason,
            )
            current = revised
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
            final_candidate=current,
        )


__all__ = [
    "AdvocateReport",
    "CandidateReview",
    "CriticalReport",
    "OBJECTION_CATEGORIES",
    "ReviewAgent",
    "ReviewContractError",
    "ReviewIdentityError",
    "ReviewLLMError",
    "ReviewPipeline",
    "RevisionRecord",
    "StructuredLLMReviewAgent",
    "review_json_schema",
]
