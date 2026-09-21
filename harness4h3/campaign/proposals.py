"""Bounded candidate proposal contracts and deterministic validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .base import CampaignBase, canonical_digest, canonical_json
from .capabilities import CapabilitySnapshot


MUTATION_FIELDS = frozenset(
    {
        "architecture",
        "architecture.family",
        "architecture.hidden_size",
        "architecture.depth",
        "architecture.num_heads",
        "architecture.temporal_layers",
        "training.method",
        "training_recipe",
        "distillation_strategy",
        "quantization",
        "deployment.precision",
        "runtime",
        "data_recipe",
    }
)


_CAPABILITY_ALIASES = {
    "velocity_distill": "progressive_distillation",
    "progressive_distill": "progressive_distillation",
}


class ProposalValidationError(ValueError):
    """Raised for malformed proposal data, before semantic report validation."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalValidationError("%s must be a non-empty string" % name)
    return value.strip()


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposalValidationError("%s must be a mapping" % name)
    try:
        copied = copy.deepcopy(dict(value))
        canonical_json(copied)
    except (TypeError, ValueError) as exc:
        raise ProposalValidationError("%s must contain finite JSON-compatible values" % name) from exc
    return copied


@dataclass(frozen=True)
class CandidateEnvelope:
    candidate_id: str
    parent_candidate_id: Optional[str]
    generation: int
    experiment_id: str
    proposal_digest: str
    mutation_fields: Tuple[str, ...]
    architecture: Mapping[str, Any]
    training_recipe: Mapping[str, Any]
    deployment_recipe: Mapping[str, Any]
    provenance: Mapping[str, Any]
    predicted_metric_delta: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.candidate_id, "candidate_id")
        if self.parent_candidate_id is not None:
            _nonempty(self.parent_candidate_id, "parent_candidate_id")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ProposalValidationError("generation must be a non-negative integer")
        _nonempty(self.experiment_id, "experiment_id")
        _nonempty(self.proposal_digest, "proposal_digest")
        fields = tuple(_nonempty(item, "mutation_fields[]") for item in self.mutation_fields)
        if len(fields) != len(set(fields)):
            raise ProposalValidationError("mutation_fields must be unique")
        object.__setattr__(self, "mutation_fields", fields)
        for name in ("architecture", "training_recipe", "deployment_recipe", "provenance", "predicted_metric_delta"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "generation": self.generation,
            "experiment_id": self.experiment_id,
            "proposal_digest": self.proposal_digest,
            "mutation_fields": list(self.mutation_fields),
            "architecture": copy.deepcopy(dict(self.architecture)),
            "training_recipe": copy.deepcopy(dict(self.training_recipe)),
            "deployment_recipe": copy.deepcopy(dict(self.deployment_recipe)),
            "provenance": copy.deepcopy(dict(self.provenance)),
            "predicted_metric_delta": copy.deepcopy(dict(self.predicted_metric_delta)),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateEnvelope":
        if not isinstance(raw, Mapping):
            raise ProposalValidationError("candidate must be a mapping")
        required = {
            "candidate_id", "parent_candidate_id", "generation", "experiment_id", "proposal_digest",
            "mutation_fields", "architecture", "training_recipe", "deployment_recipe", "provenance",
            "predicted_metric_delta",
        }
        unknown = sorted(set(raw) - required)
        missing = sorted(required - set(raw))
        if unknown:
            raise ProposalValidationError("candidate has unknown field(s): %s" % ", ".join(map(str, unknown)))
        if missing:
            raise ProposalValidationError("candidate is missing field(s): %s" % ", ".join(sorted(missing)))
        fields = raw["mutation_fields"]
        if not isinstance(fields, (list, tuple)):
            raise ProposalValidationError("mutation_fields must be an array")
        return cls(
            candidate_id=str(raw["candidate_id"]),
            parent_candidate_id=str(raw["parent_candidate_id"]) if raw["parent_candidate_id"] is not None else None,
            generation=int(raw["generation"]),
            experiment_id=str(raw["experiment_id"]),
            proposal_digest=str(raw["proposal_digest"]),
            mutation_fields=tuple(str(item) for item in fields),
            architecture=raw["architecture"],
            training_recipe=raw["training_recipe"],
            deployment_recipe=raw["deployment_recipe"],
            provenance=raw["provenance"],
            predicted_metric_delta=raw["predicted_metric_delta"],
        )

    def digest(self, base: CampaignBase) -> str:
        return canonical_digest(
            {
                "base_digest": base.digest,
                "capability_snapshot": base.capability_snapshot,
                "candidate": self.to_dict(),
            }
        )


@dataclass(frozen=True)
class ProposalBatch:
    batch_id: str
    round_id: str
    diagnosis: str
    parent_selection_evidence_ids: Tuple[str, ...]
    candidates: Tuple[CandidateEnvelope, ...]

    def __post_init__(self) -> None:
        _nonempty(self.batch_id, "batch_id")
        _nonempty(self.round_id, "round_id")
        _nonempty(self.diagnosis, "diagnosis")
        if any(not isinstance(item, str) or not item for item in self.parent_selection_evidence_ids):
            raise ProposalValidationError("parent_selection_evidence_ids must contain non-empty strings")
        if not self.candidates or not all(isinstance(item, CandidateEnvelope) for item in self.candidates):
            raise ProposalValidationError("candidates must contain at least one CandidateEnvelope")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "round_id": self.round_id,
            "diagnosis": self.diagnosis,
            "parent_selection_evidence_ids": list(self.parent_selection_evidence_ids),
            "candidates": [item.to_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class ProposalValidationReport:
    errors: Tuple[str, ...] = ()
    candidate_errors: Mapping[str, Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "candidate_errors", dict(self.candidate_errors or {}))

    @property
    def ok(self) -> bool:
        return not self.errors and not any(self.candidate_errors.values())


def _parent_info(parent_ids: Any) -> Mapping[str, Optional[int]]:
    if isinstance(parent_ids, Mapping):
        return {str(key): (int(value) if value is not None else None) for key, value in parent_ids.items()}
    return {str(key): None for key in parent_ids}


def validate_batch(
    batch: ProposalBatch,
    *,
    base: CampaignBase,
    snapshot: CapabilitySnapshot,
    parent_ids: Union[Iterable[str], Mapping[str, Optional[int]]],
    max_candidates: int = 5,
    min_candidates: int = 3,
) -> ProposalValidationReport:
    errors = []
    candidate_errors: Dict[str, list[str]] = {}
    if len(batch.candidates) < min_candidates or len(batch.candidates) > max_candidates:
        errors.append("candidate count must be between %d and %d" % (min_candidates, max_candidates))
    parents = _parent_info(parent_ids)
    seen_candidates = set()
    seen_experiments = set()
    seen_proposals = set()
    for candidate in batch.candidates:
        current = candidate_errors.setdefault(candidate.candidate_id, [])
        if candidate.candidate_id in seen_candidates:
            errors.append("duplicate candidate_id")
        seen_candidates.add(candidate.candidate_id)
        if candidate.experiment_id in seen_experiments:
            current.append("experiment_id is duplicated")
        seen_experiments.add(candidate.experiment_id)
        if candidate.proposal_digest in seen_proposals:
            current.append("proposal_digest is duplicated")
        seen_proposals.add(candidate.proposal_digest)
        if candidate.parent_candidate_id is None:
            if candidate.generation != 0:
                current.append("root candidate must have generation zero")
        elif candidate.parent_candidate_id not in parents:
            current.append("parent_candidate_id is unknown")
        elif parents[candidate.parent_candidate_id] is not None and candidate.generation != parents[candidate.parent_candidate_id] + 1:
            current.append("generation must be parent generation plus one")
        for field_name in candidate.mutation_fields:
            if field_name not in MUTATION_FIELDS:
                current.append("mutation field is not registered")
        method = candidate.training_recipe.get("method") or candidate.provenance.get("operator")
        if method is not None:
            capability_name = _CAPABILITY_ALIASES.get(str(method), str(method))
            if not snapshot.is_available(capability_name) and not snapshot.is_available(str(method)):
                current.append("capability is unavailable: %s" % method)
        try:
            candidate.digest(base)
        except (TypeError, ValueError):
            current.append("candidate digest could not be computed")
    return ProposalValidationReport(tuple(dict.fromkeys(errors)), {
        key: tuple(dict.fromkeys(value)) for key, value in candidate_errors.items() if value
    })


__all__ = [
    "CandidateEnvelope",
    "MUTATION_FIELDS",
    "ProposalBatch",
    "ProposalValidationError",
    "ProposalValidationReport",
    "validate_batch",
]
