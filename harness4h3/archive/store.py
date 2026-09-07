from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .. import Harness4H3Error


class ArchiveError(Harness4H3Error):
    pass


class CandidateExists(ArchiveError):
    pass


@dataclass(frozen=True)
class Candidate:
    id: str
    parent: Optional[str]
    generation: int
    mutation_type: str
    patch: Mapping[str, Any]
    reason: str
    policy: Mapping[str, Any]
    evidence_task_ids: List[str] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent,
            "generation": self.generation,
            "mutation_type": self.mutation_type,
            "patch": dict(self.patch),
            "reason": self.reason,
            "policy": json.loads(json.dumps(self.policy)),
            "evidence_task_ids": list(self.evidence_task_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Candidate":
        return cls(
            id=str(raw["id"]),
            parent=str(raw["parent"]) if raw.get("parent") is not None else None,
            generation=int(raw["generation"]),
            mutation_type=str(raw.get("mutation_type", "baseline")),
            patch=dict(raw.get("patch") or {}),
            reason=str(raw.get("reason", "")),
            policy=dict(raw.get("policy") or {}),
            evidence_task_ids=[str(item) for item in raw.get("evidence_task_ids", [])],
            metadata=dict(raw.get("metadata") or {}),
        )


class CandidateStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.candidates_dir = self.root / "candidates"
        self.outcomes_dir = self.root / "outcomes"
        self.active_path = self.root / "active.json"

    def _ensure(self) -> None:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)

    def create(self, candidate: Candidate) -> Candidate:
        self._ensure()
        self._validate_id(candidate.id)
        if candidate.parent is not None:
            self._validate_id(candidate.parent)
            parent = self.get(candidate.parent)
            if candidate.generation != parent.generation + 1:
                raise ArchiveError("candidate generation must be parent generation plus one")
        elif candidate.generation != 0:
            raise ArchiveError("root candidate generation must be zero")
        path = self.candidates_dir / (candidate.id + ".json")
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(candidate.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError:
            raise CandidateExists("candidate %s already exists" % candidate.id)
        return candidate

    def get(self, candidate_id: str) -> Candidate:
        self._validate_id(candidate_id)
        path = self.candidates_dir / (candidate_id + ".json")
        try:
            return Candidate.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise ArchiveError("candidate %s not found: %s" % (candidate_id, exc))

    def initialize(self, policy: Optional[Mapping[str, Any]] = None) -> Candidate:
        self._ensure()
        files = list(self.candidates_dir.glob("H*.json"))
        if files:
            return self.get(self.active_id)
        candidate = Candidate(
            id="H0",
            parent=None,
            generation=0,
            mutation_type="baseline",
            patch={},
            reason="initial harness policy",
            policy=dict(policy or default_policy()),
        )
        self.create(candidate)
        self.promote(candidate.id)
        return candidate

    @property
    def active_id(self) -> str:
        try:
            raw = json.loads(self.active_path.read_text(encoding="utf-8"))
            return str(raw["id"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ArchiveError("active candidate is unavailable: %s" % exc)

    def active(self) -> Candidate:
        return self.get(self.active_id)

    def promote(self, candidate_id: str) -> None:
        self.get(candidate_id)
        self.root.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(prefix="active-", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"id": candidate_id}, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw_path, self.active_path)
        finally:
            if os.path.exists(raw_path):
                os.unlink(raw_path)

    def record_outcome(self, candidate_id: str, outcome: Mapping[str, Any]) -> None:
        self.get(candidate_id)
        self._ensure()
        path = self.outcomes_dir / (candidate_id + ".json")
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(dict(outcome), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        except FileExistsError:
            raise ArchiveError("outcome for %s already exists" % candidate_id)

    def outcome(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        path = self.outcomes_dir / (candidate_id + ".json")
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def lineage(self) -> List[Candidate]:
        self._ensure()
        items = [Candidate.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in self.candidates_dir.glob("*.json")]
        return sorted(items, key=lambda item: (item.generation, item.id))

    def next_id(self) -> str:
        numbers = []
        for candidate in self.lineage():
            if candidate.id.startswith("H") and candidate.id[1:].isdigit():
                numbers.append(int(candidate.id[1:]))
        return "H%d" % ((max(numbers) + 1) if numbers else 0)

    @staticmethod
    def _validate_id(candidate_id: str) -> None:
        if not re.fullmatch(r"H[0-9]+", candidate_id):
            raise ArchiveError("invalid candidate id %r" % candidate_id)


def default_policy() -> Dict[str, Any]:
    return {
        "prompt": {"prefix": "", "suffix": ""},
        "context": {"include_constraints": True, "max_prompt_chars": 4000},
        "workflow": {},
    }
