"""Declarative student-model contracts and autonomous campaign primitives."""

from .proposal import (
    ArchitectureSpec,
    DeploymentSpec,
    ProposalValidationError,
    StudentProposal,
    StudentTarget,
    TrainingSpec,
    ValidationReport,
    canonical_digest,
)

__all__ = [
    "ArchitectureSpec",
    "DeploymentSpec",
    "ProposalValidationError",
    "StudentProposal",
    "StudentTarget",
    "TrainingSpec",
    "ValidationReport",
    "canonical_digest",
]
