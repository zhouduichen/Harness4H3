from __future__ import annotations

import json
import os
import ast
import hashlib
import tempfile
from pathlib import Path
from typing import Iterator

from .candidate import AlgorithmCandidate


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
        return candidate

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
    """Ignore comments/algorithm IDs when detecting repeated selection logic."""

    try:
        tree = ast.parse(source, mode="exec")
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "propose"]
        node = functions[0] if functions else tree
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        normalized = source
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
