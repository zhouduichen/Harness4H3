"""Trusted transport and measurement helpers for remote H3 experiments."""

from .ssh import (
    ComfyUITunnel,
    RemoteCommandError,
    RemoteConfig,
    RemoteError,
    RemoteLinkConflict,
    RemotePathError,
    SSHClient,
    LocalCommandClient,
    RemotePortForward,
    link_action,
)
from .scheduler import RemoteResourceScheduler, ResourceDecision
from .comfyui_lease import ComfyUILeaseManager, ComfyUILeaseResult
from .checkpoint_retention import CheckpointRetentionResult, RemoteCheckpointRetention
from .pipeline import (
    EvaluationOverlap,
    PipelineAllocation,
    PipelineStage,
    PipelineState,
    evaluation_gpu_count,
    pack_evaluation_overlap,
    pack_overlap_resources,
)
from .round_gate import RoundGateResult, RoundGateStatus, evaluate_round_gate

__all__ = [
    "ComfyUITunnel",
    "RemoteCommandError",
    "RemoteConfig",
    "RemoteError",
    "RemoteLinkConflict",
    "RemotePathError",
    "SSHClient",
    "LocalCommandClient",
    "RemotePortForward",
    "link_action",
    "RemoteResourceScheduler",
    "ResourceDecision",
    "ComfyUILeaseManager",
    "ComfyUILeaseResult",
    "CheckpointRetentionResult",
    "RemoteCheckpointRetention",
    "EvaluationOverlap",
    "PipelineAllocation",
    "PipelineStage",
    "PipelineState",
    "evaluation_gpu_count",
    "pack_evaluation_overlap",
    "pack_overlap_resources",
    "RoundGateResult",
    "RoundGateStatus",
    "evaluate_round_gate",
]
