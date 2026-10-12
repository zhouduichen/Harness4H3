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
        matches: list[tuple[tuple[int, int, float, str], Mapping[str, Any]]] = []
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
                "provenance": dict(record.provenance),
            }
            matches.append(((exact_architecture, exact_operator, reward, record.created_at), item))
        matches.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in matches[:limit]]
