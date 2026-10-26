from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ..memory.design_gene import DesignGene, DesignGeneStore
from ..memory.experience import ExperienceRecord, ExperienceStore


class SkillHub:
    """Small architecture-aware retrieval layer over append-only evidence."""

    def __init__(self, experience_path: Path, gene_path: Optional[Path] = None) -> None:
        self.experiences = ExperienceStore(Path(experience_path))
        self.genes = DesignGeneStore(Path(gene_path)) if gene_path is not None else None

    def record(self, item: ExperienceRecord | DesignGene) -> bool:
        if isinstance(item, ExperienceRecord):
            return self.experiences.append(item)
        if isinstance(item, DesignGene):
            if self.genes is None:
                raise ValueError("design gene store is not configured")
            self.genes.append(item)
            return True
        raise TypeError("SkillHub accepts ExperienceRecord or DesignGene")

    def search(self, *, architecture: str, operator: Optional[str] = None, limit: int = 8) -> list[Mapping[str, Any]]:
        if limit <= 0:
            return []
        architecture_text = str(architecture).strip().lower()
        operator_text = str(operator).strip() if operator else None
        matches: list[tuple[tuple[int, int, int, float, str], Mapping[str, Any]]] = []
        for record in self.experiences.read():
            training = dict(record.training)
            record_architecture = str(training.get("architecture", record.provenance.get("architecture", ""))).strip().lower()
            record_operator = str(record.operator or training.get("operator", ""))
            if record_architecture and record_architecture != architecture_text:
                continue
            if operator_text and record_operator and record_operator != operator_text:
                continue
            if operator_text and not record_operator:
                continue
            exact_architecture = 1 if record_architecture == architecture_text else 0
            exact_operator = 1 if record_operator == operator_text else 0 if operator_text else 1
            reward = float(record.reward) if record.reward is not None else float("-inf")
            item = {
                "experience_id": record.experience_id,
                "memory_kind": "experience",
                "architecture": record_architecture or architecture_text,
                "operator": record_operator or operator_text,
                "status": record.status,
                "reward": record.reward,
                "failure_reason": (
                    record.decision.get("failure_reason", record.decision.get("failure_type"))
                    if isinstance(record.decision, Mapping)
                    else None
                ),
                "training": training,
                "evaluation": dict(record.evaluation or {}),
                "decision": dict(record.decision),
                "provenance": dict(record.provenance),
            }
            status_rank = 1 if record.status == "accepted" else 0
            matches.append(((exact_architecture, exact_operator, status_rank, reward, record.created_at), item))
        if self.genes is not None:
            for gene in self.genes.read():
                state = dict(gene.state)
                intervention = dict(gene.intervention)
                gene_architecture = str(state.get("architecture", gene.evidence.get("architecture", ""))).strip().lower()
                gene_operator = str(intervention.get("operator", ""))
                if gene_architecture and gene_architecture != architecture_text:
                    continue
                if operator_text and gene_operator != operator_text:
                    continue
                exact_architecture = 1 if gene_architecture == architecture_text else 0
                exact_operator = 1 if gene_operator == operator_text else 0 if operator_text else 1
                benefit = dict(gene.benefit)
                raw_reward = benefit.get("search_score", benefit.get("quality_score"))
                try:
                    reward = float(raw_reward) if raw_reward is not None else float("-inf")
                except (TypeError, ValueError):
                    reward = float("-inf")
                item = {
                    "experience_id": gene.gene_id,
                    "memory_kind": "design_gene",
                    "architecture": gene_architecture or architecture_text,
                    "operator": gene_operator or operator_text,
                    "status": gene.status,
                    "reward": reward if reward != float("-inf") else None,
                    "failure_reason": dict(gene.remaining_limitation).get("reason"),
                    "training": {
                        "architecture": gene_architecture or architecture_text,
                        "design_strategy": intervention.get("strategy", gene_operator),
                        "design": gene.to_dict(),
                        "design_fingerprint": gene.evidence.get("design_fingerprint"),
                    },
                    "evaluation": benefit,
                    "decision": {"status": gene.status, "failure_reason": dict(gene.remaining_limitation).get("reason")},
                    "provenance": dict(gene.evidence),
                }
                confidence_rank = 1 if gene.confidence == "measured" else 0
                status_rank = 1 if gene.status == "accepted" else 0
                matches.append(((exact_architecture, exact_operator, status_rank + confidence_rank, reward, gene.created_at), item))
        matches.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in matches[:limit]]
