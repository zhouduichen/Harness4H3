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
from .campaign import StudentProposalBatchProvider, build_student_control_plane, student_proposal_batch_json_schema

__all__ = [
    "ArchitectureSpec",
    "DeploymentSpec",
    "ProposalValidationError",
    "StudentProposal",
    "StudentTarget",
    "TrainingSpec",
    "ValidationReport",
    "canonical_digest",
    "StudentProposalBatchProvider",
    "build_student_control_plane",
    "student_proposal_batch_json_schema",
]
