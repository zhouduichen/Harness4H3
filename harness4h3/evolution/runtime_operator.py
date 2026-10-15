"""Capability-gated runtime compression interfaces.

The current phase only defines the evidence contract for sparse-attention
runtime experiments.  It deliberately does not patch Diffusers Wan kernels or
claim a weight/parameter reduction.  Activation remains disabled until an
isolated implementation supplies a trusted runtime and independent hardware
measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class RuntimeCompressionRequest:
    operator_name: str
    model_checkpoint_sha256: str
    runtime_version: str
    calibration_digest: str
    enabled: bool = False
    changes_weights: bool = False

    def __post_init__(self) -> None:
        if not self.operator_name.strip() or not self.runtime_version.strip():
            raise ValueError("runtime compression request requires operator and version")
        if len(str(self.model_checkpoint_sha256)) != 64 or len(str(self.calibration_digest)) != 64:
            raise ValueError("runtime compression request requires SHA-256 evidence identifiers")
        if self.changes_weights:
            raise ValueError("runtime compression operators cannot mutate model weights")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeCompressionOperator(Protocol):
    name: str
    enabled: bool

    def prepare(self, request: RuntimeCompressionRequest) -> Mapping[str, Any]:
        ...


class SparseAttentionRuntimeOperator:
    """Disabled SVOO/SparseVideo-style runtime operator boundary."""

    name = "sparse_attention_runtime"
    enabled = False

    def prepare(self, request: RuntimeCompressionRequest) -> Mapping[str, Any]:
        if request.operator_name != self.name:
            raise ValueError("runtime_operator_name_mismatch")
        if not self.enabled or not request.enabled:
            return {
                "status": "disabled",
                "operator": self.name,
                "evidence_level": "interface_only",
                "changes_weights": False,
                "hardware_metrics": {"available": False, "unavailable_reason": "runtime operator is not activated"},
                "activation_boundary": "requires_isolated_runtime_and_independent_gpu_measurement",
            }
        raise RuntimeError("sparse_attention_runtime_activation_requires_trusted_implementation")


__all__ = ["RuntimeCompressionOperator", "RuntimeCompressionRequest", "SparseAttentionRuntimeOperator"]
