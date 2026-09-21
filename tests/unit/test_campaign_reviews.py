from __future__ import annotations

import pytest

from harness4h3.campaign.base import ActorIdentity
from harness4h3.campaign.proposals import CandidateEnvelope
from harness4h3.campaign.reviews import (
    AdvocateReport,
    CandidateReview,
    CriticalReport,
    ReviewContractError,
    ReviewIdentityError,
    ReviewPipeline,
    RevisionRecord,
)
from tests.unit.campaign_fixtures import make_base


def valid_candidate():
    return CandidateEnvelope(
        candidate_id="C0001",
        parent_candidate_id="M0000",
        generation=1,
        experiment_id="exp-C0001",
        proposal_digest="sha256:C0001",
        mutation_fields=("training.method",),
        architecture={"family": "video_latent_dit"},
        training_recipe={"method": "velocity_distill"},
        deployment_recipe={"precision": "bf16"},
        provenance={"source": "test"},
        predicted_metric_delta={"quality": 0.01},
    )


def valid_advocate():
    return {
        "bottleneck": "latency",
        "changed_fields": ["training.method"],
        "expected_metric_delta": {"latency_s": -0.2},
        "supporting_evidence_ids": ["obs-1"],
        "falsification_experiment": "F1 smoke run",
        "resource_assumptions": {"gpu_hours": 1.0},
    }


def critical_raw(*, hard=False, required=None):
    return {
        "objections": ["check target runtime"] if hard else [],
        "objection_categories": ["target_device_mismatch"] if hard else [],
        "missing_evidence_ids": ["edge-1"] if hard else [],
        "proxy_gaming_risks": [],
        "target_device_risks": ["server proxy"] if hard else [],
        "required_revisions": list(required or []),
        "hard_objection": hard,
    }


class ScriptedAdvocate:
    identity = ActorIdentity("advocate", "model-a", "1")

    def __init__(self, payload):
        self.payload = payload

    def review(self, request):
        return self.payload


class ScriptedCritical:
    def __init__(self, payloads):
        self.identity = make_base().critic_identity
        self.payloads = list(payloads)

    def review(self, request):
        return self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]


class ScriptedModifier:
    identity = ActorIdentity("modifier", "model-m", "1")

    def __init__(self, candidate):
        self.candidate = candidate

    def review(self, request):
        return {
            "candidate": self.candidate.to_dict(),
            "base_digest": request["base_digest"],
            "changed_fields": ["training.method"],
            "resolved_objection_ids": ["objection-1"],
            "reason": "use the registered distillation backend",
        }


def test_critical_agent_cannot_change_base_or_hard_constraints():
    base = make_base()
    pipeline = ReviewPipeline(
        advocate=ScriptedAdvocate(valid_advocate()),
        critical=ScriptedCritical([critical_raw(hard=True, required=["training.method"]), critical_raw()]),
        modifier=ScriptedModifier(valid_candidate()),
        base=base,
        max_rounds=1,
    )
    result = pipeline.review(valid_candidate(), {"hard_constraints": {"max_latency_s": 3}})
    assert result.approved is True
    assert result.revision.base_digest == base.digest
    assert result.revision.changed_fields == ("training.method",)


def test_unresolved_hard_objection_is_rejected_at_review_limit():
    pipeline = ReviewPipeline(
        advocate=ScriptedAdvocate(valid_advocate()),
        critical=ScriptedCritical([critical_raw(hard=True, required=["training.method"])]),
        modifier=ScriptedModifier(valid_candidate()),
        base=make_base(),
        max_rounds=1,
    )
    result = pipeline.review(valid_candidate(), {})
    assert result.approved is False
    assert result.rejection_reasons == ("unresolved_hard_objection",)


def test_review_identity_must_differ_from_controller_and_evaluator():
    base = make_base()
    advocate = ScriptedAdvocate(valid_advocate())
    advocate.identity = base.controller_identity
    with pytest.raises(ReviewIdentityError, match="distinct"):
        ReviewPipeline(
            advocate=advocate,
            critical=ScriptedCritical([critical_raw()]),
            modifier=ScriptedModifier(valid_candidate()),
            base=base,
            max_rounds=1,
        )


def test_report_parsers_reject_unknown_fields_and_invalid_categories():
    raw = valid_advocate()
    raw["unknown"] = True
    with pytest.raises(ReviewContractError, match="unknown"):
        AdvocateReport.from_dict(raw)
    invalid = critical_raw()
    invalid["objection_categories"] = ["invented_category"]
    with pytest.raises(ReviewContractError, match="category"):
        CriticalReport.from_dict(invalid)
