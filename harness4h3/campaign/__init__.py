"""Shared contracts for verifier-grounded autonomous campaigns."""

from .base import ActorIdentity, CampaignBase, CampaignBaseError, canonical_digest, canonical_json
from .capabilities import Capability, CapabilityRegistry, CapabilitySnapshot
from .events import DecisionEvent, DecisionTrace, TraceIntegrityError
from .proposals import CandidateEnvelope, ProposalBatch, ProposalValidationError, ProposalValidationReport, validate_batch

__all__ = [
    "ActorIdentity",
    "CampaignBase",
    "CampaignBaseError",
    "Capability",
    "CapabilityRegistry",
    "CapabilitySnapshot",
    "CandidateEnvelope",
    "DecisionEvent",
    "DecisionTrace",
    "TraceIntegrityError",
    "ProposalBatch",
    "ProposalValidationError",
    "ProposalValidationReport",
    "canonical_digest",
    "canonical_json",
    "validate_batch",
]
