from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from .. import Harness4H3Error
from .model_candidate import MODEL_ID_PATTERN, ModelCandidate


class ModelStoreError(Harness4H3Error):
    pass


class ModelCandidateExists(ModelStoreError):
    pass


class ModelStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.candidates_dir = self.root / "candidates"
        self.active_path = self.root / "active.json"

    def _ensure(self) -> None:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, candidate: ModelCandidate) -> ModelCandidate:
        self._ensure()
        existing = self.lineage()
        if existing:
            return self.active()
        if candidate.id != "M0000" or candidate.parent_id is not None or candidate.generation != 0:
            raise ModelStoreError("initial model candidate must be root M0000")
        self.create(candidate)
        self.set_active(candidate.id)
        return candidate

    def create(self, candidate: ModelCandidate) -> ModelCandidate:
        self._ensure()
        if candidate.parent_id is None:
            if candidate.generation != 0:
                raise ModelStoreError("root generation must be zero")
        else:
            parent = self.get(candidate.parent_id)
            if candidate.generation != parent.generation + 1:
                raise ModelStoreError("candidate generation must be parent generation plus one")
        path = self.candidates_dir / (candidate.id + ".json")
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(candidate.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise ModelCandidateExists("model candidate %s already exists" % candidate.id)
        return candidate

    def get(self, candidate_id: str) -> ModelCandidate:
        if not MODEL_ID_PATTERN.fullmatch(candidate_id):
            raise ModelStoreError("invalid model candidate id %r" % candidate_id)
        path = self.candidates_dir / (candidate_id + ".json")
        try:
            return ModelCandidate.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise ModelStoreError("model candidate %s not found: %s" % (candidate_id, exc))

    def lineage(self) -> List[ModelCandidate]:
        if not self.candidates_dir.exists():
            return []
        items = [ModelCandidate.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in self.candidates_dir.glob("M*.json")]
        return sorted(items, key=lambda item: (item.generation, item.id))

    def children(self, candidate_id: str) -> List[ModelCandidate]:
        self.get(candidate_id)
        return [item for item in self.lineage() if item.parent_id == candidate_id]

    def next_id(self) -> str:
        numbers = [int(item.id[1:]) for item in self.lineage()]
        return "M%04d" % ((max(numbers) + 1) if numbers else 0)

    @property
    def active_id(self) -> str:
        try:
            return str(json.loads(self.active_path.read_text(encoding="utf-8"))["id"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ModelStoreError("active model candidate is unavailable: %s" % exc)

    def active(self) -> ModelCandidate:
        return self.get(self.active_id)

    def set_active(self, candidate_id: str) -> None:
        self.get(candidate_id)
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="active-model-", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"id": candidate_id}, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.active_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
