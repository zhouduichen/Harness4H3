from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from ..memory.design_gene import DesignGene, DesignGeneStore
from ..memory.experience import ExperienceRecord, ExperienceStore


class SkillHub:
    """Small architecture-aware retrieval layer over append-only evidence."""

    _algorithm_lock = threading.Lock()

    def __init__(self, experience_path: Path, gene_path: Optional[Path] = None, algorithm_path: Optional[Path] = None) -> None:
        self.experiences = ExperienceStore(Path(experience_path))
        self.genes = DesignGeneStore(Path(gene_path)) if gene_path is not None else None
        self.algorithms_path = Path(algorithm_path) if algorithm_path is not None else None

    def record(self, item: ExperienceRecord | DesignGene) -> bool:
        if isinstance(item, ExperienceRecord):
            return self.experiences.append(item)
        if isinstance(item, DesignGene):
            if self.genes is None:
                raise ValueError("design gene store is not configured")
            self.genes.append(item)
            return True
        raise TypeError("SkillHub accepts ExperienceRecord or DesignGene")

    def record_algorithm(
        self,
        candidate: Any,
        *,
        architecture: str,
        validation_status: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Persist a verified algorithm skill without replacing AlgorithmStore."""

        if self.algorithms_path is None:
            return False
        status = str(validation_status or getattr(candidate, "validation_status", ""))
        if status not in {"validated", "accepted", "model_promoted"}:
            return False
        self.algorithms_path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids = set()
        if self.algorithms_path.exists():
            for line in self.algorithms_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping) and value.get("algorithm_id"):
                    existing_ids.add(str(value["algorithm_id"]))
        algorithm_id = str(getattr(candidate, "algorithm_id", ""))
        if not algorithm_id or algorithm_id in existing_ids:
            return False
        payload = {
            "memory_kind": "algorithm_skill",
            "algorithm_id": algorithm_id,
            "parent_algorithm_id": getattr(candidate, "parent_algorithm_id", None),
            "parent_source_sha256": getattr(candidate, "parent_source_sha256", None),
            "source_sha256": getattr(candidate, "source_sha256", None),
            "source": getattr(candidate, "source", ""),
            "change_summary": getattr(candidate, "change_summary", ""),
            "change_rationale": getattr(candidate, "change_rationale", ""),
            "validation_status": status,
            "architecture": str(architecture),
            "evidence": dict(evidence or {}),
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._algorithm_lock:
            with self.algorithms_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return True

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
        if self.algorithms_path is not None and self.algorithms_path.exists():
            for line in self.algorithms_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, Mapping):
                    continue
                record_architecture = str(record.get("architecture", "")).strip().lower()
                if record_architecture and record_architecture != architecture_text:
                    continue
                item = {
                    "experience_id": record.get("algorithm_id"),
                    "memory_kind": "algorithm_skill",
                    "architecture": record_architecture or architecture_text,
                    "operator": None,
                    "status": record.get("validation_status", "validated"),
                    "reward": None,
                    "failure_reason": None,
                    "algorithm_id": record.get("algorithm_id"),
                    "algorithm_parent_id": record.get("parent_algorithm_id"),
                    "algorithm_source": record.get("source", ""),
                    "change_summary": record.get("change_summary", ""),
                    "change_rationale": record.get("change_rationale", ""),
                    "evidence": dict(record.get("evidence") or {}),
                }
                matches.append(((1, 1, 2, 0.0, str(record.get("algorithm_id", ""))), item))
        matches.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in matches[:limit]]
