from __future__ import annotations

import pytest

from harness4h3.campaign.capabilities import Capability, CapabilitySnapshot
from harness4h3.campaign.proposals import (
    CandidateEnvelope,
    ProposalBatch,
    ProposalValidationError,
    validate_batch,
)
from tests.unit.campaign_fixtures import make_base


def valid_snapshot():
    return CapabilitySnapshot(
        (
            Capability("velocity_distill", "training", "test", {}, "V2", True, ""),
            Capability("dmd2", "training", "test", {}, "V2", True, ""),
        )
    )


def valid_candidate(candidate_id="C0001", parent_candidate_id="M0000", mutation_fields=("training.method",)):
    return CandidateEnvelope(
        candidate_id=candidate_id,
        parent_candidate_id=parent_candidate_id,
        generation=1,
        experiment_id="exp-" + candidate_id,
        proposal_digest="sha256:" + candidate_id,
        mutation_fields=tuple(mutation_fields),
        architecture={"family": "video_latent_dit"},
        training_recipe={"method": "velocity_distill"},
        deployment_recipe={"precision": "bf16"},
        provenance={"source": "test"},
        predicted_metric_delta={"quality": 0.01},
    )


def valid_batch(count=3, candidates=None):
    items = tuple(candidates or (valid_candidate("C%04d" % (index + 1)) for index in range(count)))
    return ProposalBatch("batch-test", "R0001", "latency bottleneck", ("obs-1",), items)


def test_batch_rejects_duplicate_ids_unknown_mutations_and_invalid_parent():
    batch = valid_batch(
        candidates=(
            valid_candidate("C0001", "M0000", ("training.method",)),
            valid_candidate("C0001", "M9999", ("repository.file",)),
            valid_candidate("C0003"),
        )
    )
    report = validate_batch(
        batch,
        base=make_base(),
        snapshot=valid_snapshot(),
        parent_ids={"M0000"},
        max_candidates=5,
    )
    assert "duplicate candidate_id" in report.errors
    assert "parent_candidate_id is unknown" in report.candidate_errors["C0001"]
    assert "mutation field is not registered" in report.candidate_errors["C0001"]


def test_batch_of_three_candidates_is_accepted_before_review():
    report = validate_batch(
        valid_batch(count=3),
        base=make_base(),
        snapshot=valid_snapshot(),
        parent_ids={"M0000"},
        max_candidates=5,
    )
    assert report.ok


def test_batch_size_and_unavailable_capability_are_rejected():
    one = validate_batch(
        valid_batch(count=1),
        base=make_base(),
        snapshot=valid_snapshot(),
        parent_ids={"M0000"},
        max_candidates=5,
    )
    assert "candidate count must be between 3 and 5" in one.errors

    unavailable = valid_candidate("C0001")
    unavailable = CandidateEnvelope(
        **{**unavailable.to_dict(), "training_recipe": {"method": "progressive_distill"}}
    )
    report = validate_batch(
        valid_batch(candidates=(unavailable, valid_candidate("C0002"), valid_candidate("C0003"))),
        base=make_base(),
        snapshot=valid_snapshot(),
        parent_ids={"M0000"},
        max_candidates=5,
    )
    assert any(item.startswith("capability is unavailable") for item in report.candidate_errors["C0001"])


def test_candidate_rejects_non_json_prediction():
    with pytest.raises(ProposalValidationError, match="JSON"):
        CandidateEnvelope(
            candidate_id="C0001",
            parent_candidate_id="M0000",
            generation=1,
            experiment_id="exp-C0001",
            proposal_digest="sha256:C0001",
            mutation_fields=("training.method",),
            architecture={},
            training_recipe={"method": "velocity_distill"},
            deployment_recipe={},
            provenance={},
            predicted_metric_delta={"quality": float("nan")},
        )
