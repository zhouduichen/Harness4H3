from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from .. import Harness4H3Error
from .model_candidate import MODEL_ID_PATTERN
from .system_candidate import SYSTEM_ID_PATTERN, SystemCandidate


class SystemStoreError(Harness4H3Error):
    pass


class SystemCandidateExists(SystemStoreError):
    pass


class SystemCandidateStore:
    """Atomic JSON archive for model/runtime compositions."""

    def __init__(self, root: Path, model_store: Optional[object] = None):
        self.root = Path(root)
        self.candidates_dir = self.root / "candidates"
        self.active_path = self.root / "active.json"
        self.model_store = model_store

    def _ensure(self) -> None:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def create(self, candidate: SystemCandidate) -> SystemCandidate:
        self._ensure()
        self._validate_id(candidate.id)
        if candidate.parent_id is None:
            if candidate.generation != 0:
                raise SystemStoreError("root system candidate generation must be zero")
        else:
            parent = self.get(candidate.parent_id)
            if candidate.generation != parent.generation + 1:
                raise SystemStoreError("system candidate generation must be parent generation plus one")
        self._validate_model_ref(candidate.model_ref)
        if self.model_store is not None:
            try:
                self.model_store.get(candidate.model_ref)
            except Exception as exc:
                raise SystemStoreError("model reference %s is unavailable: %s" % (candidate.model_ref, exc))
        path = self.candidates_dir / (candidate.id + ".json")
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(candidate.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise SystemCandidateExists("system candidate %s already exists" % candidate.id)
        return candidate

    def get(self, candidate_id: str) -> SystemCandidate:
        self._validate_id(candidate_id)
        path = self.candidates_dir / (candidate_id + ".json")
        try:
            return SystemCandidate.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SystemStoreError("system candidate %s not found or invalid: %s" % (candidate_id, exc))

    def lineage(self) -> List[SystemCandidate]:
        if not self.candidates_dir.exists():
            return []
        paths = {path for pattern in ("S*.json", "C*.json") for path in self.candidates_dir.glob(pattern)}
        items = [SystemCandidate.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
        return sorted(items, key=lambda item: (item.generation, item.id))

    def children(self, candidate_id: str) -> List[SystemCandidate]:
        self.get(candidate_id)
        return [item for item in self.lineage() if item.parent_id == candidate_id]

    def next_id(self) -> str:
        numbers = [int(item.id[1:]) for item in self.lineage()]
        return "S%04d" % ((max(numbers) + 1) if numbers else 0)

    def initialize(self, candidate: SystemCandidate) -> SystemCandidate:
        self._ensure()
        existing = self.lineage()
        if existing:
            return self.active()
        if candidate.id not in {"S0000", "C0000"} or candidate.parent_id is not None or candidate.generation != 0:
            raise SystemStoreError("initial system candidate must be root S0000")
        self.create(candidate)
        self.set_active(candidate.id)
        return candidate

    @property
    def active_id(self) -> str:
        try:
            return str(json.loads(self.active_path.read_text(encoding="utf-8"))["id"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise SystemStoreError("active system candidate is unavailable: %s" % exc)

    def active(self) -> SystemCandidate:
        return self.get(self.active_id)

    def set_active(self, candidate_id: str) -> None:
        self.get(candidate_id)
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="active-system-", suffix=".json", dir=str(self.root))
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

    @staticmethod
    def _validate_id(candidate_id: str) -> None:
        if not SYSTEM_ID_PATTERN.fullmatch(candidate_id):
            raise SystemStoreError("invalid system candidate id %r" % candidate_id)

    @staticmethod
    def _validate_model_ref(model_ref: str) -> None:
        if not MODEL_ID_PATTERN.fullmatch(model_ref):
            raise SystemStoreError("invalid model reference %r" % model_ref)
