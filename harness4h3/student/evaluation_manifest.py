"""Deterministic, immutable evaluation cases for Student quality comparisons."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    cache_path: str
    caption: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.cache_path or not self.caption.strip():
            raise ValueError("evaluation cases require case_id, cache_path, and caption")
        if not self.seeds or any(int(seed) < 0 for seed in self.seeds):
            raise ValueError("evaluation cases require non-negative seeds")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["seeds"] = list(self.seeds)
        return value


@dataclass(frozen=True)
class EvaluationManifest:
    cases: tuple[EvaluationCase, ...]
    unique_cache_items: int
    source_cache_dir: str
    digest: str
    version: int = 1

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("evaluation manifest requires at least one case")
        if self.unique_cache_items <= 0:
            raise ValueError("evaluation manifest requires a positive cache-item count")
        expected = self._digest_for(self.version, self.source_cache_dir, self.unique_cache_items, self.cases)
        if self.digest != expected:
            raise ValueError("evaluation manifest digest mismatch")

    @staticmethod
    def _digest_for(
        version: int,
        source_cache_dir: str,
        unique_cache_items: int,
        cases: Sequence[EvaluationCase],
    ) -> str:
        payload = {
            "version": int(version),
            "source_cache_dir": str(source_cache_dir),
            "unique_cache_items": int(unique_cache_items),
            "cases": [case.to_dict() for case in cases],
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_cache_dir": self.source_cache_dir,
            "unique_cache_items": self.unique_cache_items,
            "cases": [case.to_dict() for case in self.cases],
            "digest": self.digest,
        }

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluationManifest":
        cases = tuple(
            EvaluationCase(
                case_id=str(item["case_id"]),
                cache_path=str(item["cache_path"]),
                caption=str(item["caption"]),
                seeds=tuple(int(seed) for seed in item["seeds"]),
            )
            for item in raw.get("cases", [])
        )
        return cls(
            cases=cases,
            unique_cache_items=int(raw["unique_cache_items"]),
            source_cache_dir=str(raw["source_cache_dir"]),
            digest=str(raw["digest"]),
            version=int(raw.get("version", 1)),
        )

    @classmethod
    def from_path(cls, path: Path) -> "EvaluationManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation manifest must be a JSON object")
        return cls.from_dict(raw)


def _cache_metadata(path: Path) -> str:
    try:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("unable to load evaluation cache item %s: %s" % (path, exc)) from exc
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation cache item must be a mapping: %s" % path)
    caption = str(raw.get("caption", "")).strip()
    if not caption:
        raise ValueError("evaluation cache item requires a caption: %s" % path)
    if "prompt" not in raw or "video" not in raw:
        raise ValueError("evaluation cache item requires prompt and video: %s" % path)
    return caption


def build_manifest(
    cache_dir: Path,
    output_path: Path,
    *,
    case_count: int = 4,
    seeds: Sequence[int] = (20260920, 20260921),
) -> EvaluationManifest:
    """Build a stable manifest, reusing cache items only when the cache is small."""

    if int(case_count) <= 0 or not tuple(seeds):
        raise ValueError("case_count and seeds must be non-empty")
    cache_dir = Path(cache_dir).resolve()
    cache_paths = sorted(cache_dir.glob("*.pt"))
    if not cache_paths:
        raise ValueError("no H3 cache items under %s" % cache_dir)
    captions = {path: _cache_metadata(path) for path in cache_paths}
    cases = tuple(
        EvaluationCase(
            case_id="case_%04d" % (index + 1),
            cache_path=str(cache_paths[index % len(cache_paths)].resolve()),
            caption=captions[cache_paths[index % len(cache_paths)]],
            seeds=tuple(int(seed) for seed in seeds),
        )
        for index in range(int(case_count))
    )
    digest = EvaluationManifest._digest_for(1, str(cache_dir), len(cache_paths), cases)
    manifest = EvaluationManifest(cases, len(cache_paths), str(cache_dir), digest)
    manifest.write(Path(output_path))
    return manifest


__all__ = ["EvaluationCase", "EvaluationManifest", "build_manifest"]
