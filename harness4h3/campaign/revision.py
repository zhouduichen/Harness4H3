"""Trusted application of LLM review patches to executable Student proposals."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .base import CampaignBase, canonical_json
from .proposals import CandidateEnvelope, MUTATION_FIELDS
from ..student.proposal import ProposalValidationError, StudentProposal


class RevisionPatchError(ValueError):
    """Raised when a revision patch crosses the trusted proposal boundary."""


_PATCH_FIELDS = {
    "candidate_id",
    "base_digest",
    "operations",
    "changed_fields",
    "resolved_objection_ids",
    "reason",
}
_OP_FIELDS = {"op", "path", "value"}
_ROOTS = {"architecture", "training", "deployment"}


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RevisionPatchError("%s must be a non-empty string" % name)
    return value.strip()


def _strings(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RevisionPatchError("%s must be an array" % name)
    result = tuple(_string(item, name + "[]") for item in value)
    if len(result) != len(set(result)):
        raise RevisionPatchError("%s must contain unique values" % name)
    return result


@dataclass(frozen=True)
class RevisionPatch:
    candidate_id: str
    base_digest: str
    operations: Tuple[Mapping[str, Any], ...]
    changed_fields: Tuple[str, ...]
    resolved_objection_ids: Tuple[str, ...]
    reason: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RevisionPatch":
        if not isinstance(raw, Mapping):
            raise RevisionPatchError("revision patch must be an object")
        unknown = sorted(set(raw) - _PATCH_FIELDS)
        missing = sorted(_PATCH_FIELDS - set(raw))
        if unknown:
            raise RevisionPatchError("revision patch has unknown field(s): %s" % ", ".join(unknown))
        if missing:
            raise RevisionPatchError("revision patch is missing field(s): %s" % ", ".join(missing))
        operations = []
        if not isinstance(raw["operations"], (list, tuple)):
            raise RevisionPatchError("revision patch operations must be an array")
        for index, operation in enumerate(raw["operations"]):
            if not isinstance(operation, Mapping):
                raise RevisionPatchError("revision patch operation %d must be an object" % index)
            unknown_operation = sorted(set(operation) - _OP_FIELDS)
            if unknown_operation or set(operation) != _OP_FIELDS:
                raise RevisionPatchError("revision patch operation %d must contain only op, path, and value" % index)
            op = _string(operation["op"], "revision.operations[%d].op" % index)
            path = _string(operation["path"], "revision.operations[%d].path" % index)
            if op not in {"add", "replace"}:
                raise RevisionPatchError("unsupported revision operation: %s" % op)
            if not path.startswith("/") or len(path.split("/")) != 3 or path.split("/")[1] not in _ROOTS:
                raise RevisionPatchError("revision path is outside executable proposal fields: %s" % path)
            try:
                canonical_json(operation["value"])
            except (TypeError, ValueError) as exc:
                raise RevisionPatchError("revision operation value is not JSON-compatible") from exc
            operations.append({"op": op, "path": path, "value": copy.deepcopy(operation["value"])})
        changed_fields = _strings(raw["changed_fields"], "revision.changed_fields")
        invalid = sorted(set(changed_fields) - MUTATION_FIELDS)
        if invalid:
            raise RevisionPatchError("revision changed field is not registered: %s" % ", ".join(invalid))
        return cls(
            candidate_id=_string(raw["candidate_id"], "revision.candidate_id"),
            base_digest=_string(raw["base_digest"], "revision.base_digest"),
            operations=tuple(operations),
            changed_fields=changed_fields,
            resolved_objection_ids=_strings(raw["resolved_objection_ids"], "revision.resolved_objection_ids"),
            reason=_string(raw["reason"], "revision.reason"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "base_digest": self.base_digest,
            "operations": [copy.deepcopy(dict(item)) for item in self.operations],
            "changed_fields": list(self.changed_fields),
            "resolved_objection_ids": list(self.resolved_objection_ids),
            "reason": self.reason,
        }


def _proposal_path(operation_path: str) -> tuple[str, str]:
    parts = operation_path.split("/")
    return parts[1], parts[2]


def apply_revision_patch(proposal: StudentProposal, patch: RevisionPatch) -> StudentProposal:
    """Apply an allowlisted patch, then parse the complete proposal canonically."""

    if not isinstance(proposal, StudentProposal):
        raise RevisionPatchError("revision source must be a StudentProposal")
    raw = copy.deepcopy(proposal.to_dict())
    for operation in patch.operations:
        root, field = _proposal_path(str(operation["path"]))
        if field in {"proposal_id", "parent_proposal_id", "teacher", "schema_version"}:
            raise RevisionPatchError("revision cannot mutate immutable proposal identity")
        raw[root][field] = copy.deepcopy(operation["value"])
    try:
        revised = StudentProposal.from_dict(raw)
    except (ProposalValidationError, TypeError, ValueError) as exc:
        raise RevisionPatchError("revision produced an invalid StudentProposal: %s" % exc) from exc
    return revised


def canonical_candidate_from_proposal(
    original: CandidateEnvelope,
    proposal: StudentProposal,
    patch: RevisionPatch,
) -> CandidateEnvelope:
    """Rebuild every executable Candidate field from trusted proposal values."""

    if patch.candidate_id != original.candidate_id:
        raise RevisionPatchError("revision cannot change candidate_id")
    provenance = copy.deepcopy(dict(original.provenance))
    provenance["student_proposal"] = proposal.to_dict()
    provenance["proposal_digest"] = proposal.digest
    provenance["revision_patch"] = patch.to_dict()
    return CandidateEnvelope(
        candidate_id=original.candidate_id,
        parent_candidate_id=original.parent_candidate_id,
        generation=original.generation,
        experiment_id=original.experiment_id,
        proposal_digest=proposal.digest,
        mutation_fields=patch.changed_fields or original.mutation_fields,
        architecture=proposal.architecture.to_dict(),
        training_recipe=proposal.training.to_dict(),
        deployment_recipe=proposal.deployment.to_dict(),
        provenance=provenance,
        predicted_metric_delta=copy.deepcopy(dict(original.predicted_metric_delta)),
    )


def proposal_from_candidate(candidate: CandidateEnvelope) -> StudentProposal:
    raw = candidate.provenance.get("student_proposal")
    if not isinstance(raw, Mapping):
        raise RevisionPatchError("Candidate provenance lacks canonical student_proposal")
    proposal = StudentProposal.from_dict(raw)
    if candidate.proposal_digest != proposal.digest:
        raise RevisionPatchError("Candidate proposal_digest does not match canonical StudentProposal")
    return proposal


def canonical_candidate_from_patch(original: CandidateEnvelope, patch: RevisionPatch) -> CandidateEnvelope:
    """Apply a trusted patch to a non-Student legacy Candidate.

    Student Campaign candidates must use canonical_candidate_from_proposal;
    this narrow fallback keeps the older generic review contract executable
    without allowing the LLM to supply a complete CandidateEnvelope.
    """

    raw = copy.deepcopy(original.to_dict())
    roots = {"architecture": "architecture", "training": "training_recipe", "deployment": "deployment_recipe"}
    for operation in patch.operations:
        root, field = _proposal_path(str(operation["path"]))
        raw[roots[root]][field] = copy.deepcopy(operation["value"])
    provenance = copy.deepcopy(dict(original.provenance))
    provenance["revision_patch"] = patch.to_dict()
    raw["provenance"] = provenance
    raw["mutation_fields"] = list(patch.changed_fields or original.mutation_fields)
    return CandidateEnvelope.from_dict(raw)


__all__ = [
    "RevisionPatch",
    "RevisionPatchError",
    "apply_revision_patch",
    "canonical_candidate_from_proposal",
    "canonical_candidate_from_patch",
    "proposal_from_candidate",
]
