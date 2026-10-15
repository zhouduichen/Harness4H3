"""Capability-gated runtime compression interfaces.

The legacy boundary in this module now delegates to the Wan adapter's real
top-k key/value runtime path.  It intentionally does not claim an SVOO,
SparseVideo, or custom CUDA-kernel implementation: those remain separate
backends and require their own evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from ..operators.wan.runtime_policy import SparseRuntimePolicy


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

    def execute(
        self,
        adapter: Any,
        model: Any,
        request: RuntimeCompressionRequest,
        policy: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


class SparseAttentionRuntimeOperator:
    """Execute the supported top-k KV runtime backend through a Wan adapter."""

    name = "sparse_attention_runtime"
    enabled = True
    backend = "topk_kv"

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
        return {
            "status": "ready",
            "operator": self.name,
            "backend": self.backend,
            "evidence_level": "executable",
            "changes_weights": False,
            "dynamic_compute": True,
            "parameter_reduction_ratio": 0.0,
            "execution_method": "execute",
            "hardware_metrics": {"available": False, "unavailable_reason": "independent GPU measurement not supplied"},
        }

    def execute(
        self,
        adapter: Any,
        model: Any,
        request: RuntimeCompressionRequest,
        policy: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Apply the real adapter implementation and return auditable evidence."""

        prepared = self.prepare(request)
        if prepared["status"] == "disabled":
            return prepared
        parsed = SparseRuntimePolicy.from_mapping(policy)
        method = getattr(adapter, "apply_sparse_runtime", None)
        if not callable(method):
            raise RuntimeError("sparse_attention_runtime_adapter_method_missing")
        change = method(model, parsed.to_dict())
        details = dict(getattr(change, "details", {}) or {})
        return {
            **prepared,
            "status": "executed",
            "policy": parsed.to_dict(),
            "execution": details,
            "model_checkpoint_sha256": request.model_checkpoint_sha256,
            "calibration_digest": request.calibration_digest,
            "runtime_version": request.runtime_version,
        }


__all__ = ["RuntimeCompressionOperator", "RuntimeCompressionRequest", "SparseAttentionRuntimeOperator"]
