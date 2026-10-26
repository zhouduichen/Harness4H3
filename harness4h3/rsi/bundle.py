from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MAX_BUNDLE_FILE_BYTES = 256 * 1024


def _safe_relative(raw: str) -> str:
    path = PurePosixPath(str(raw))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("bundle_path_not_relative")
    return path.as_posix()


def _bundle_digest(files: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        encoded_name = name.encode("utf-8")
        encoded = files[name].encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class HarnessBundle:
    bundle_id: str
    parent_id: str | None
    files: Mapping[str, str]
    file_sha256: Mapping[str, str]
    hypothesis: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized = {_safe_relative(name): content for name, content in self.files.items()}
        if not normalized or any(not isinstance(value, str) for value in normalized.values()):
            raise ValueError("bundle_requires_text_files")
        if any(len(value.encode("utf-8")) > MAX_BUNDLE_FILE_BYTES for value in normalized.values()):
            raise ValueError("bundle_file_too_large")
        expected_files = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in normalized.items()}
        if dict(self.file_sha256) != expected_files or self.bundle_id != _bundle_digest(normalized):
            raise ValueError("bundle_digest_mismatch")
        object.__setattr__(self, "files", normalized)
        object.__setattr__(self, "file_sha256", expected_files)

    @classmethod
    def capture(
        cls,
        source: Path,
        store: Path,
        *,
        parent_id: str | None,
        hypothesis: Mapping[str, Any],
        allowed_paths: Sequence[str],
    ) -> "HarnessBundle":
        root, destination = Path(source).resolve(), Path(store).resolve()
        if not root.is_dir():
            raise ValueError("bundle_source_not_directory")
        allowed = tuple(sorted({_safe_relative(item) for item in allowed_paths}))
        if not allowed:
            raise ValueError("bundle_allowlist_empty")
        files: dict[str, str] = {}
        for relative in allowed:
            path = root.joinpath(*PurePosixPath(relative).parts)
            if path.is_symlink() or not path.is_file():
                raise ValueError("bundle_allowlisted_file_missing_or_unsafe:%s" % relative)
            if path.resolve().parent != root and root not in path.resolve().parents:
                raise ValueError("bundle_path_escapes_source")
            files[relative] = path.read_text(encoding="utf-8")
        identifier = _bundle_digest(files)
        hashes = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in files.items()}
        bundle = cls(identifier, parent_id, files, hashes, dict(hypothesis))
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / identifier
        if target.exists():
            existing = cls.load(destination, identifier)
            if dict(existing.files) != dict(bundle.files) or dict(existing.file_sha256) != dict(bundle.file_sha256):
                raise ValueError("bundle_identity_collision")
            return existing
        temp = Path(tempfile.mkdtemp(prefix=".bundle-", dir=str(destination)))
        try:
            for relative, content in files.items():
                path = temp.joinpath(*PurePosixPath(relative).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            metadata = bundle.to_dict()
            with (temp / "bundle.json").open("x", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temp), str(target))
        finally:
            if temp.exists():
                shutil.rmtree(str(temp))
        return bundle

    @classmethod
    def load(cls, store: Path, bundle_id: str) -> "HarnessBundle":
        if len(bundle_id) != 64 or any(ch not in "0123456789abcdef" for ch in bundle_id):
            raise ValueError("invalid_bundle_id")
        root = Path(store).resolve() / bundle_id
        try:
            raw = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            if int(raw["schema_version"]) != SCHEMA_VERSION:
                raise ValueError("unsupported_bundle_schema")
            contents: dict[str, str] = {}
            for name in raw["files"]:
                relative = _safe_relative(name)
                path = root.joinpath(*PurePosixPath(relative).parts)
                if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
                    raise ValueError("bundle_stored_file_missing_or_unsafe:%s" % relative)
                contents[relative] = path.read_text(encoding="utf-8")
            return cls(str(raw["bundle_id"]), raw.get("parent_id"), contents, dict(raw["file_sha256"]), dict(raw.get("hypothesis") or {}))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("bundle_unavailable:%s" % bundle_id) from exc

    def restore(self, store: Path, destination: Path) -> Path:
        verified = self.load(store, self.bundle_id)
        target = Path(destination)
        if target.exists():
            raise ValueError("bundle_restore_destination_exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=".restore-", dir=str(target.parent.resolve())))
        try:
            for relative, content in verified.files.items():
                path = temp.joinpath(*PurePosixPath(relative).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            os.replace(str(temp), str(target))
        finally:
            if temp.exists():
                shutil.rmtree(str(temp))
        return target

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
