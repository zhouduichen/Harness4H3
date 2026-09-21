from __future__ import annotations

import pytest

from harness4h3.campaign.proposals import CandidateEnvelope
from harness4h3.campaign.revision import (
    RevisionPatch,
    RevisionPatchError,
    apply_revision_patch,
    canonical_candidate_from_proposal,
)
from harness4h3.student.proposal import StudentProposal
from tests.unit.test_student_proposal import valid_payload


def _proposal() -> StudentProposal:
    return StudentProposal.from_dict(valid_payload())


def _candidate(proposal: StudentProposal) -> CandidateEnvelope:
    return CandidateEnvelope(
        candidate_id=proposal.proposal_id,
        parent_candidate_id="M0000",
        generation=1,
        experiment_id="exp-1",
        proposal_digest=proposal.digest,
        mutation_fields=("training.learning_rate",),
        architecture=proposal.architecture.to_dict(),
        training_recipe=proposal.training.to_dict(),
        deployment_recipe=proposal.deployment.to_dict(),
        provenance={"student_proposal": proposal.to_dict()},
        predicted_metric_delta={"quality": 0.01},
    )


def _patch(candidate: CandidateEnvelope) -> RevisionPatch:
    return RevisionPatch.from_dict(
        {
            "candidate_id": candidate.candidate_id,
            "base_digest": "sha256:base",
            "operations": [{"op": "replace", "path": "/training/learning_rate", "value": 0.0005}],
            "changed_fields": ["training.learning_rate"],
            "resolved_objection_ids": ["resource-1"],
            "reason": "reduce optimizer pressure",
        }
    )


def test_revision_rebuilds_digest_and_candidate_from_proposal():
    proposal = _proposal()
    candidate = _candidate(proposal)
    patch = _patch(candidate)
    revised = apply_revision_patch(proposal, patch)
    rebuilt = canonical_candidate_from_proposal(candidate, revised, patch)
    assert revised.training.learning_rate == 0.0005
    assert rebuilt.training_recipe == revised.training.to_dict()
    assert rebuilt.architecture == revised.architecture.to_dict()
    assert rebuilt.deployment_recipe == revised.deployment.to_dict()
    assert rebuilt.proposal_digest == revised.digest
    assert rebuilt.provenance["student_proposal"] == revised.to_dict()


def test_revision_cannot_return_parent_or_unknown_fields():
    candidate = _candidate(_proposal())
    with pytest.raises(RevisionPatchError, match="outside"):
        RevisionPatch.from_dict(
            {
                "candidate_id": candidate.candidate_id,
                "base_digest": "sha256:base",
                "operations": [{"op": "replace", "path": "/parent_candidate_id", "value": "M0009"}],
                "changed_fields": ["training.method"],
                "resolved_objection_ids": [],
                "reason": "bad",
            }
        )


def test_revision_rejects_candidate_id_change():
    proposal = _proposal()
    candidate = _candidate(proposal)
    patch = RevisionPatch.from_dict(
        {
            "candidate_id": "other",
            "base_digest": "sha256:base",
            "operations": [],
            "changed_fields": [],
            "resolved_objection_ids": [],
            "reason": "bad",
        }
    )
    with pytest.raises(RevisionPatchError, match="candidate_id"):
        canonical_candidate_from_proposal(candidate, proposal, patch)
