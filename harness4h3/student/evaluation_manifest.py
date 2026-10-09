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
class CacheContract:
    name: str
    latent_frames: int
    decoded_frames: int
    width: int
    height: int
    require_real_teacher: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("cache contract requires a name")
        for field in ("latent_frames", "decoded_frames", "width", "height"):
            if isinstance(getattr(self, field), bool) or int(getattr(self, field)) <= 0:
                raise ValueError("cache contract %s must be positive" % field)

    @classmethod
    def native_5s(cls) -> "CacheContract":
        return cls(
            name="native_5s",
            latent_frames=24,
            decoded_frames=81,
            width=256,
            height=256,
            require_real_teacher=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationManifest:
    cases: tuple[EvaluationCase, ...]
    unique_cache_items: int
    source_cache_dir: str
    digest: str
    version: int = 1
    cache_contract: CacheContract | None = None

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("evaluation manifest requires at least one case")
        if self.unique_cache_items <= 0:
            raise ValueError("evaluation manifest requires a positive cache-item count")
        expected = self._digest_for(
            self.version,
            self.source_cache_dir,
            self.unique_cache_items,
            self.cases,
            cache_contract=self.cache_contract,
        )
        if self.digest != expected:
            raise ValueError("evaluation manifest digest mismatch")

    @staticmethod
    def _digest_for(
        version: int,
        source_cache_dir: str,
        unique_cache_items: int,
        cases: Sequence[EvaluationCase],
        cache_contract: CacheContract | None = None,
    ) -> str:
        payload = {
            "version": int(version),
            "source_cache_dir": str(source_cache_dir),
            "unique_cache_items": int(unique_cache_items),
            "cases": [case.to_dict() for case in cases],
        }
        if cache_contract is not None:
            payload["cache_contract"] = cache_contract.to_dict()
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "version": self.version,
            "source_cache_dir": self.source_cache_dir,
            "unique_cache_items": self.unique_cache_items,
            "cases": [case.to_dict() for case in self.cases],
            "digest": self.digest,
        }
        if self.cache_contract is not None:
            value["cache_contract"] = self.cache_contract.to_dict()
        return value

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
            cache_contract=(
                CacheContract(**dict(raw["cache_contract"]))
                if raw.get("cache_contract") is not None
                else None
            ),
        )

    @classmethod
    def from_path(cls, path: Path) -> "EvaluationManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation manifest must be a JSON object")
        return cls.from_dict(raw)


def _cache_metadata(path: Path, *, require_real_h3: bool = False) -> str:
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
    if require_real_h3 and (
        raw.get("synthetic_latent") is not False
        or raw.get("teacher_generated") is not True
        or raw.get("offline_simulation") is not False
    ):
        raise ValueError("evaluation cache item is not an authenticated real H3 Teacher sample: %s" % path)
    return caption


def validate_h3_cache_item(path: Path, contract: CacheContract) -> Mapping[str, Any]:
    """Validate a packed H3 cache item against an explicit geometry contract."""

    selected = Path(path)
    prefix = "%s_cache_contract" % contract.name
    try:
        import torch

        raw = torch.load(selected, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("%s: unable to load %s: %s" % (prefix, selected, exc)) from exc
    if not isinstance(raw, Mapping):
        raise ValueError("%s: cache item must be a mapping" % prefix)
    for key in ("prompt", "video", "audio", "caption"):
        if key not in raw:
            raise ValueError("%s: missing %s" % (prefix, key))
    if not str(raw.get("caption", "")).strip():
        raise ValueError("%s: caption is empty" % prefix)
    latent_frames = int(raw.get("latent_frames", -1))
    if latent_frames != contract.latent_frames:
        raise ValueError("%s: latent_frames=%s expected %s" % (prefix, latent_frames, contract.latent_frames))
    if int(raw.get("num_frames", -1)) != contract.decoded_frames:
        raise ValueError("%s: num_frames mismatch" % prefix)
    if int(raw.get("width", -1)) != contract.width or int(raw.get("height", -1)) != contract.height:
        raise ValueError("%s: geometry mismatch" % prefix)
    try:
        video = torch.as_tensor(raw["video"])
        prompt = torch.as_tensor(raw["prompt"])
        audio = torch.as_tensor(raw["audio"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("%s: tensor field is invalid" % prefix) from exc
    if video.ndim != 2 or int(video.shape[0]) != contract.latent_frames * 64:
        raise ValueError("%s: packed video rows mismatch" % prefix)
    if prompt.ndim != 2 or not torch.isfinite(prompt.float()).all():
        raise ValueError("%s: prompt tensor is invalid" % prefix)
    if audio.ndim < 1 or not torch.isfinite(audio.float()).all() or not torch.isfinite(video.float()).all():
        raise ValueError("%s: latent tensor is invalid" % prefix)
    if contract.require_real_teacher and (
        raw.get("teacher_generated") is not True
        or raw.get("synthetic_latent") is not False
        or raw.get("offline_simulation") is not False
    ):
        raise ValueError("%s: cache item is not an authenticated real H3 Teacher sample" % prefix)
    return raw


def _sequence_anchor(path: Path) -> bool:
    """Use one case per authenticated long sequence, not one per window."""

    try:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("unable to load sequence metadata %s: %s" % (path, exc)) from exc
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation cache item must be a mapping: %s" % path)
    return bool(raw.get("sequence_anchor", True))


def _prompt_group(path: Path) -> str | None:
    """Group long-cache anchors by prompt, not by the seed in sequence_id."""

    try:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("unable to load prompt grouping metadata %s: %s" % (path, exc)) from exc
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation cache item must be a mapping: %s" % path)
    sequence_id = str(raw.get("sequence_id", ""))
    if not sequence_id or int(raw.get("window_count", 1) or 1) <= 1:
        return None
    if sequence_id.startswith("prompt_") and "_seed_" in sequence_id:
        return sequence_id.split("_seed_", 1)[0]
    return sequence_id


def build_manifest(
    cache_dir: Path,
    output_path: Path,
    *,
    case_count: int = 4,
    seeds: Sequence[int] = (20260920, 20260921),
    require_real_h3: bool = False,
    cache_contract: CacheContract | None = None,
) -> EvaluationManifest:
    """Build a stable manifest, reusing cache items only when the cache is small."""

    if int(case_count) <= 0 or not tuple(seeds):
        raise ValueError("case_count and seeds must be non-empty")
    cache_dir = Path(cache_dir).resolve()
    cache_paths = sorted(cache_dir.glob("*.pt"))
    if not cache_paths:
        raise ValueError("no H3 cache items under %s" % cache_dir)
    if cache_contract is not None:
        metadata = {path: validate_h3_cache_item(path, cache_contract) for path in cache_paths}
        captions = {path: str(item["caption"]).strip() for path, item in metadata.items()}
    else:
        captions = {path: _cache_metadata(path, require_real_h3=require_real_h3) for path in cache_paths}
    anchored_paths = tuple(path for path in cache_paths if _sequence_anchor(path))
    grouped: dict[str, Path] = {}
    case_paths_list: list[Path] = []
    for path in anchored_paths:
        group = _prompt_group(path)
        if group is None:
            case_paths_list.append(path)
        elif group not in grouped:
            grouped[group] = path
            case_paths_list.append(path)
    case_paths = tuple(case_paths_list)
    if not case_paths:
        raise ValueError("no sequence anchor cache items under %s" % cache_dir)
    cases = tuple(
        EvaluationCase(
            case_id="case_%04d" % (index + 1),
            cache_path=str(case_paths[index % len(case_paths)].resolve()),
            caption=captions[case_paths[index % len(case_paths)]],
            seeds=tuple(int(seed) for seed in seeds),
        )
        for index in range(int(case_count))
    )
    digest = EvaluationManifest._digest_for(
        1,
        str(cache_dir),
        len(cache_paths),
        cases,
        cache_contract=cache_contract,
    )
    manifest = EvaluationManifest(cases, len(cache_paths), str(cache_dir), digest, cache_contract=cache_contract)
    manifest.write(Path(output_path))
    return manifest


__all__ = [
    "CacheContract",
    "EvaluationCase",
    "EvaluationManifest",
    "build_manifest",
    "validate_h3_cache_item",
]
