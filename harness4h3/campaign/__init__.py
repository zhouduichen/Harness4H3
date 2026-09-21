"""Shared contracts for verifier-grounded autonomous campaigns."""

from .base import ActorIdentity, CampaignBase, CampaignBaseError, canonical_digest, canonical_json
from .events import DecisionEvent, DecisionTrace, TraceIntegrityError

__all__ = [
    "ActorIdentity",
    "CampaignBase",
    "CampaignBaseError",
    "DecisionEvent",
    "DecisionTrace",
    "TraceIntegrityError",
    "canonical_digest",
    "canonical_json",
]
