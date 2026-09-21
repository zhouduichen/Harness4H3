"""Shared contracts for verifier-grounded autonomous campaigns."""

from .base import ActorIdentity, CampaignBase, CampaignBaseError, canonical_digest, canonical_json
from .adapters import CandidateExecutor, LegacyH3CampaignAdapter, StudentCampaignAdapter
from .capabilities import Capability, CapabilityRegistry, CapabilitySnapshot
from .events import DecisionEvent, DecisionTrace, TraceIntegrityError
from .failures import FailureAttributor, FailureReport
from .gates import AcceptanceGate, GateDecision, MetricEvidence, pareto_dominates
from .proposals import CandidateEnvelope, ProposalBatch, ProposalValidationError, ProposalValidationReport, validate_batch
from .revision import RevisionPatch, RevisionPatchError, apply_revision_patch, canonical_candidate_from_proposal
from .reviews import (
    AdvocateReport,
    CandidateReview,
    CriticalReport,
    ReviewContractError,
    ReviewIdentityError,
    ReviewPipeline,
    RevisionRecord,
)

__all__ = [
    "ActorIdentity",
    "CandidateExecutor",
    "CampaignBase",
    "CampaignBaseError",
    "Capability",
    "CapabilityRegistry",
    "CapabilitySnapshot",
    "CandidateEnvelope",
    "AdvocateReport",
    "CandidateReview",
    "CriticalReport",
    "DecisionEvent",
    "DecisionTrace",
    "FailureAttributor",
    "FailureReport",
    "AcceptanceGate",
    "GateDecision",
    "MetricEvidence",
    "LegacyH3CampaignAdapter",
    "pareto_dominates",
    "TraceIntegrityError",
    "ProposalBatch",
    "ProposalValidationError",
    "ProposalValidationReport",
    "ReviewContractError",
    "ReviewIdentityError",
    "ReviewPipeline",
    "RevisionRecord",
    "StudentCampaignAdapter",
    "canonical_digest",
    "canonical_json",
    "validate_batch",
    "RevisionPatch",
    "RevisionPatchError",
    "apply_revision_patch",
    "canonical_candidate_from_proposal",
]
