"""Declarative student-model contracts and autonomous campaign primitives.

The optional campaign/runtime modules are loaded lazily.  Compression and
quality evaluation must be importable in a clean checkout without pulling in
the unrelated remote training worker dependency tree.
"""

from importlib import import_module

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

_LAZY_EXPORTS = {
    "StudentProposalBatchProvider": ("campaign", "StudentProposalBatchProvider"),
    "build_student_control_plane": ("campaign", "build_student_control_plane"),
    "student_proposal_batch_json_schema": ("campaign", "student_proposal_batch_json_schema"),
    "FidelityGate": ("fidelity", "FidelityGate"),
    "FidelityGateDecision": ("fidelity", "FidelityGateDecision"),
    "stage_spec": ("fidelity", "stage_spec"),
    "FidelitySpec": ("worker", "FidelitySpec"),
    "fidelity_spec": ("worker", "fidelity_spec"),
    "TargetDeviceProfile": ("target", "TargetDeviceProfile"),
    "WanStudentError": ("wan", "WanStudentError"),
    "WanStudentSpec": ("wan", "WanStudentSpec"),
    "generate_wan_clip": ("wan", "generate_wan_clip"),
    "load_pretrained_wan": ("wan", "load_pretrained_wan"),
    "load_wan_spec": ("wan", "load_wan_spec"),
    "WanLongVideoError": ("wan_long_video", "WanLongVideoError"),
    "WanLongVideoSpec": ("wan_long_video", "WanLongVideoSpec"),
    "generate_wan_long_video": ("wan_long_video", "generate_wan_long_video"),
    "validate_wan_long_video": ("wan_long_video", "validate_wan_long_video"),
    "RemoteWanExperiment": ("wan_experiment", "RemoteWanExperiment"),
    "RemoteWanTrainingExperiment": ("wan_experiment", "RemoteWanTrainingExperiment"),
    "WanExperimentConfigError": ("wan_experiment", "WanExperimentConfigError"),
    "WanRealExperimentConfig": ("wan_experiment", "WanRealExperimentConfig"),
    "WanTrainingExperimentConfig": ("wan_experiment", "WanTrainingExperimentConfig"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(".%s" % module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "ArchitectureSpec",
    "DeploymentSpec",
    "ProposalValidationError",
    "StudentProposal",
    "StudentTarget",
    "TrainingSpec",
    "ValidationReport",
    "canonical_digest",
    *_LAZY_EXPORTS,
]
