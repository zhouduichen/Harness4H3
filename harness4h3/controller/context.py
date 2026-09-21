from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, Mapping, Tuple

from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .schemas import BudgetState


@dataclass(frozen=True)
class ControllerContext:
    target_profile: TargetProfile
    current_model_state: ModelState
    budget_state: BudgetState
    available_operators: Tuple[Mapping[str, Any], ...]
    recent_experiments: List[Mapping[str, Any]] = field(default_factory=list)
    relevant_failures: List[Mapping[str, Any]] = field(default_factory=list)
    pareto_front: List[Mapping[str, Any]] = field(default_factory=list)
    validated_design_genes: List[Mapping[str, Any]] = field(default_factory=list)
    validated_evaluation: Mapping[str, Any] = field(default_factory=dict)
    # Explicit goal payload kept separate from the target schema so a
    # Controller can reason about the objective and stop conditions without
    # inventing them from raw metrics.
    goal: Mapping[str, Any] = field(default_factory=dict)
    observations: List[Mapping[str, Any]] = field(default_factory=list)
    unconsumed_observation_ids: List[str] = field(default_factory=list)
    current_system: Mapping[str, Any] = field(default_factory=dict)
    campaign_summary: Mapping[str, Any] = field(default_factory=dict)
    # Bounded retrieval over append-only experiment memory.  This is kept
    # separate from recent_experiments so a long campaign does not silently
    # lose older evidence when the recent window advances.
    retrieved_relevant_experiments: List[Mapping[str, Any]] = field(default_factory=list)
    # ``primary`` is the normal lineage plan. ``parallel_gpu_fill`` is a
    # second, isolated plan requested only to use spare GPUs during a
    # CPU-only primary plan; it must never alter the primary decision.
    planning_intent: str = "primary"
    # Capability evidence is separate from the operator registry. A method
    # named in the goal is not executable unless the real runtime exposes it.
    optimization_capabilities: Mapping[str, Any] = field(default_factory=dict)
    # The round policy limits the next search space; the digest is a bounded
    # view of append-only experience and never contains model payloads.
    round_policy: Mapping[str, Any] = field(default_factory=dict)
    discovery_digest: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)
