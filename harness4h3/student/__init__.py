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
from .fidelity import FidelityGate, FidelityGateDecision, stage_spec
from .worker import FidelitySpec, fidelity_spec
from .target import TargetDeviceProfile
from .wan import WanStudentError, WanStudentSpec, generate_wan_clip, load_pretrained_wan, load_wan_spec
from .wan_experiment import (
    RemoteWanExperiment,
    RemoteWanTrainingExperiment,
    WanExperimentConfigError,
    WanRealExperimentConfig,
    WanTrainingExperimentConfig,
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
    "StudentProposalBatchProvider",
    "build_student_control_plane",
    "student_proposal_batch_json_schema",
    "FidelitySpec",
    "fidelity_spec",
    "FidelityGate",
    "FidelityGateDecision",
    "stage_spec",
    "TargetDeviceProfile",
    "WanStudentError",
    "WanStudentSpec",
    "generate_wan_clip",
    "load_pretrained_wan",
    "load_wan_spec",
    "RemoteWanExperiment",
    "RemoteWanTrainingExperiment",
    "WanExperimentConfigError",
    "WanRealExperimentConfig",
    "WanTrainingExperimentConfig",
]
