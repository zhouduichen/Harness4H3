from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping


@dataclass(frozen=True)
class DesignGene:
    """A compact, evidence-grounded optimization experience."""

    gene_id: str
    status: str
    state: Mapping[str, Any]
    bottleneck: Mapping[str, Any]
    intervention: Mapping[str, Any]
    controlled_conditions: Mapping[str, Any]
    benefit: Mapping[str, Any]
    remaining_limitation: Mapping[str, Any]
    risks_lessons: List[str] = field(default_factory=list)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    confidence: str = "partial"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.gene_id.strip() or not self.status.strip():
            raise ValueError("design gene requires gene_id and status")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DesignGene":
        return cls(
            gene_id=str(raw["gene_id"]),
            status=str(raw["status"]),
            state=dict(raw.get("state") or {}),
            bottleneck=dict(raw.get("bottleneck") or {}),
            intervention=dict(raw.get("intervention") or {}),
            controlled_conditions=dict(raw.get("controlled_conditions") or {}),
            benefit=dict(raw.get("benefit") or {}),
            remaining_limitation=dict(raw.get("remaining_limitation") or {}),
            risks_lessons=[str(item) for item in raw.get("risks_lessons", [])],
            evidence=dict(raw.get("evidence") or {}),
            confidence=str(raw.get("confidence", "partial")),
            created_at=str(raw.get("created_at", "")),
        )


class DesignGeneStore:
    """Append-only JSONL storage for validated optimization experience."""

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, gene: DesignGene) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(gene.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def read(self) -> Iterator[DesignGene]:
        if not self.path.exists():
            return iter(())

        def records() -> Iterator[DesignGene]:
            with self.path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield DesignGene.from_dict(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise ValueError("invalid design gene line %d: %s" % (number, exc))

        return records()
