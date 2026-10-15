from __future__ import annotations

import json
import os
import ast
import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from .candidate import AlgorithmCandidate, build_source_diff


class AlgorithmStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.candidates_dir = self.root / "candidates"

    def _ensure(self) -> None:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> str:
        numbers = []
        for path in self.candidates_dir.glob("A*.json"):
            try:
                numbers.append(int(path.stem[1:]))
            except ValueError:
                continue
        return "A%04d" % (max(numbers, default=0) + 1)

    def create(self, candidate: AlgorithmCandidate) -> AlgorithmCandidate:
        self._ensure()
        candidate_logic = _algorithm_logic_digest(candidate.source)
        if any(item.source_sha256 == candidate.source_sha256 or _algorithm_logic_digest(item.source) == candidate_logic for item in self.read()):
            raise ValueError("duplicate_algorithm_source")
        source_path = self.candidates_dir / (candidate.algorithm_id + ".py")
        metadata_path = self.candidates_dir / (candidate.algorithm_id + ".json")
        if source_path.exists() or metadata_path.exists():
            raise ValueError("algorithm_candidate_exists: %s" % candidate.algorithm_id)
        self._atomic_write(source_path, candidate.source)
        self._atomic_write(metadata_path, json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if candidate.source_diff:
            self._atomic_write(self.candidates_dir / (candidate.algorithm_id + ".diff"), candidate.source_diff)
        return candidate

    def create_child(self, candidate: AlgorithmCandidate, parent_source_sha256: str) -> AlgorithmCandidate:
        """Create an immutable algorithm child after checking parent content."""

        parent_id = candidate.parent_algorithm_id
        if not parent_id:
            raise ValueError("algorithm_parent_required")
        parent = self.get(parent_id)
        expected = str(parent_source_sha256)
        if parent.source_sha256 != expected or candidate.parent_source_sha256 != expected:
            raise ValueError("parent_source_hash_mismatch")
        expected_diff = build_source_diff(parent.source, candidate.source, parent_id=parent.algorithm_id, child_id=candidate.algorithm_id)
        if candidate.source_diff and candidate.source_diff != expected_diff:
            raise ValueError("algorithm_source_diff_mismatch")
        prepared = candidate if candidate.source_diff else replace(candidate, source_diff=expected_diff)
        return self.create(prepared)

    def duplicate_of(self, candidate: AlgorithmCandidate) -> AlgorithmCandidate | None:
        logic = _algorithm_logic_digest(candidate.source)
        return next(
            (item for item in self.read() if item.source_sha256 == candidate.source_sha256 or _algorithm_logic_digest(item.source) == logic),
            None,
        )

    def get(self, algorithm_id: str) -> AlgorithmCandidate:
        path = self.candidates_dir / (str(algorithm_id) + ".json")
        try:
            return AlgorithmCandidate.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("algorithm_candidate_unavailable: %s" % algorithm_id) from exc

    def update(self, candidate: AlgorithmCandidate) -> AlgorithmCandidate:
        existing = self.get(candidate.algorithm_id)
        if existing.source_sha256 != candidate.source_sha256 or existing.parent_id != candidate.parent_id:
            raise ValueError("algorithm_candidate_identity_is_immutable")
        path = self.candidates_dir / (candidate.algorithm_id + ".json")
        self._atomic_write(path, json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return candidate

    def read(self) -> Iterator[AlgorithmCandidate]:
        if not self.candidates_dir.exists():
            return iter(())

        def records() -> Iterator[AlgorithmCandidate]:
            for path in sorted(self.candidates_dir.glob("A*.json")):
                yield AlgorithmCandidate.from_dict(json.loads(path.read_text(encoding="utf-8")))

        return records()

    def latest(self) -> AlgorithmCandidate | None:
        """Return the newest algorithm version, including rejected experiments."""

        candidates = list(self.read())
        if not candidates:
            return None
        return max(candidates, key=lambda item: int(item.algorithm_id[1:]))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _algorithm_logic_digest(source: str) -> str:
    """Ignore comments/algorithm IDs while retaining propose/search behavior."""

    try:
        tree = ast.parse(source, mode="exec")
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"propose", "search"}
        ]
        normalized = "\n".join(
            ast.dump(item, annotate_fields=True, include_attributes=False)
            for item in (functions or [tree])
        )
    except SyntaxError:
        normalized = source
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
