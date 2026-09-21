"""Cumulative multi-fidelity budgets and fail-closed stage gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from ..campaign.gates import MetricEvidence
from .worker import FidelitySpec, fidelity_spec


@dataclass(frozen=True)
class FidelityGateDecision:
    fidelity: str
    passed: bool
    verifier_strength: str
    required_evidence: Tuple[str, ...]
    violations: Tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity": self.fidelity,
            "passed": self.passed,
            "verifier_strength": self.verifier_strength,
            "required_evidence": list(self.required_evidence),
            "violations": list(self.violations),
        }


class FidelityGate:
    """Apply the independent verifier required before entering the next tier."""

    _REQUIRED = {
        "cheap": ("video_decodable", "algorithm_dispatch", "parent_checkpoint_bound", "fidelity_executed"),
        "semantic": ("video_decodable", "algorithm_dispatch", "parent_checkpoint_bound", "fidelity_executed", "semantic_verified"),
        "full": (
            "video_decodable", "algorithm_dispatch", "parent_checkpoint_bound", "fidelity_executed",
            "semantic_verified", "quality_verified", "latency_verified", "memory_verified", "model_size_verified",
        ),
    }

    def evaluate(
        self,
        spec: FidelitySpec,
        evidence: Sequence[MetricEvidence] | Mapping[str, MetricEvidence],
    ) -> FidelityGateDecision:
        evidence_map = dict(evidence) if isinstance(evidence, Mapping) else {item.metric_name: item for item in evidence}
        required = self._REQUIRED[spec.verifier_strength]
        violations = []
        for name in required:
            item = evidence_map.get(name)
            if item is None or item.value is None or not item.valid or float(item.value) <= 0:
                violations.append(name)
        return FidelityGateDecision(
            fidelity=spec.name,
            passed=not violations,
            verifier_strength=spec.verifier_strength,
            required_evidence=tuple(required),
            violations=tuple(violations),
        )


def stage_spec(max_steps: int, fidelity: str) -> FidelitySpec:
    """Stable import point for callers that must not duplicate tier math."""

    return fidelity_spec(max_steps, fidelity)


__all__ = ["FidelityGate", "FidelityGateDecision", "stage_spec"]
