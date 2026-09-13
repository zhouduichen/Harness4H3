"""Content-addressed, verified tensor cache for expensive preprocessing."""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from h3_training.engine.state import TrainingFailure


CACHE_SCHEMA = 1
SUPPORTED_KINDS = frozenset(
    {"text_embedding", "video_latent", "audio_latent", "teacher_trajectory", "teacher_prediction"}
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheKey:
    digest: str
    material: Mapping[str, Any]

    def __str__(self) -> str:
        return self.digest


@dataclass(frozen=True)
class CacheManifest:
    schema_version: int
    key: str
    kind: str
    payload_sha256: str
    tensors: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]


class TrainingCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, kind: str, **material: Any) -> CacheKey:
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported cache kind {kind!r}")
        complete = {"schema_version": CACHE_SCHEMA, "kind": kind, **material}
        return CacheKey(hashlib.sha256(_canonical(complete)).hexdigest(), complete)

    def paths(self, key: CacheKey) -> Tuple[Path, Path, Path]:
        entry = self.root / key.digest[:2] / key.digest
        return entry, entry / "payload.safetensors", entry / "manifest.json"

    def store(
        self,
        key: CacheKey,
        tensors: Mapping[str, torch.Tensor],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CacheManifest:
        if not tensors or any(not name for name in tensors):
            raise ValueError("cache entry requires named tensors")
        entry, payload_path, manifest_path = self.paths(key)
        if entry.exists():
            _, manifest = self.load(key)
            return manifest
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key.digest}.", dir=entry.parent))
        try:
            payload = temporary / payload_path.name
            normalized = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
            save_file(normalized, str(payload))
            with payload.open("rb") as stream:
                os.fsync(stream.fileno())
            tensor_manifest = {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in normalized.items()
            }
            manifest_data = {
                "schema_version": CACHE_SCHEMA,
                "key": key.digest,
                "kind": key.material["kind"],
                "payload_sha256": _sha256_file(payload),
                "tensors": tensor_manifest,
                "metadata": _jsonable(dict(metadata or {})),
                "key_material": _jsonable(key.material),
            }
            manifest = temporary / manifest_path.name
            manifest.write_bytes(_canonical(manifest_data))
            with manifest.open("rb") as stream:
                os.fsync(stream.fileno())
            try:
                os.rename(temporary, entry)
            except FileExistsError:
                pass
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        _, verified = self.load(key)
        return verified

    def load(self, key: CacheKey) -> Tuple[Dict[str, torch.Tensor], CacheManifest]:
        entry, payload_path, manifest_path = self.paths(key)
        if not entry.is_dir():
            raise KeyError(key.digest)
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != CACHE_SCHEMA or raw.get("key") != key.digest:
                raise ValueError("manifest identity mismatch")
            if hashlib.sha256(_canonical(raw.get("key_material", {}))).hexdigest() != key.digest:
                raise ValueError("key material mismatch")
            if _sha256_file(payload_path) != raw.get("payload_sha256"):
                raise ValueError("payload hash mismatch")
            tensors = load_file(str(payload_path), device="cpu")
            declarations = raw.get("tensors", {})
            if set(tensors) != set(declarations):
                raise ValueError("tensor name mismatch")
            for name, tensor in tensors.items():
                declared = declarations[name]
                if list(tensor.shape) != declared.get("shape") or str(tensor.dtype) != declared.get("dtype"):
                    raise ValueError(f"tensor declaration mismatch for {name}")
            manifest = CacheManifest(
                schema_version=raw["schema_version"],
                key=raw["key"],
                kind=raw["kind"],
                payload_sha256=raw["payload_sha256"],
                tensors=raw["tensors"],
                metadata=raw.get("metadata", {}),
            )
            return tensors, manifest
        except TrainingFailure:
            raise
        except Exception as exc:
            raise TrainingFailure("cache_corrupt", str(exc)) from exc
