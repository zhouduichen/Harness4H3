from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..controller.schemas import EvaluationResult
from .model_candidate import MODEL_ID_PATTERN


@dataclass(frozen=True)
class ParetoEntry:
    candidate_id: str
    evaluation: EvaluationResult

    def to_dict(self) -> Dict[str, Any]:
        return {"candidate_id": self.candidate_id, "evaluation": self.evaluation.to_dict()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ParetoEntry":
        return cls(str(raw["candidate_id"]), EvaluationResult.from_dict(raw["evaluation"]))


def _objectives(result: EvaluationResult) -> Tuple[Tuple[Optional[float], bool], ...]:
    hardware = result.hardware
    return (
        (result.quality_score, True),
        (hardware.latency_s, False),
        (hardware.peak_memory_gb, False),
        (hardware.model_size_gb, False),
        (hardware.energy_j, False),
    )


def dominates(a: EvaluationResult, b: EvaluationResult) -> bool:
    if a.feasible != b.feasible:
        return a.feasible
    strictly_better = False
    compared = False
    for (a_value, maximize), (b_value, _) in zip(_objectives(a), _objectives(b)):
        if b_value is None:
            continue
        if a_value is None:
            return False
        compared = True
        if maximize:
            if a_value < b_value:
                return False
            strictly_better = strictly_better or a_value > b_value
        else:
            if a_value > b_value:
                return False
            strictly_better = strictly_better or a_value < b_value
    return compared and strictly_better


class ParetoArchive:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.entries_dir = self.root / "entries"
        self.front_path = self.root / "front.json"

    def entries(self) -> List[ParetoEntry]:
        if not self.entries_dir.exists():
            return []
        result = [ParetoEntry.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in self.entries_dir.glob("M*.json")]
        return sorted(result, key=lambda item: item.candidate_id)

    def update(self, candidate_id: str, evaluation: EvaluationResult) -> List[ParetoEntry]:
        if not MODEL_ID_PATTERN.fullmatch(candidate_id):
            raise ValueError("invalid model candidate id %r" % candidate_id)
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        entry = ParetoEntry(candidate_id, evaluation)
        path = self.entries_dir / (candidate_id + ".json")
        with path.open("x", encoding="utf-8") as handle:
            json.dump(entry.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        all_entries = self.entries()
        front = [item for item in all_entries if not any(other.candidate_id != item.candidate_id and dominates(other.evaluation, item.evaluation) for other in all_entries)]
        self._write_front([item.candidate_id for item in front])
        return front

    def front(self) -> List[ParetoEntry]:
        if not self.front_path.exists():
            return []
        ids = json.loads(self.front_path.read_text(encoding="utf-8")).get("candidate_ids", [])
        by_id = {entry.candidate_id: entry for entry in self.entries()}
        return [by_id[candidate_id] for candidate_id in ids if candidate_id in by_id]

    def _write_front(self, candidate_ids: Sequence[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="pareto-", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"candidate_ids": list(candidate_ids)}, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.front_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
