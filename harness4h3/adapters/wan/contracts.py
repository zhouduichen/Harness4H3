from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stable_digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_content_sha256(path: str | Path) -> str:
    """Hash checkpoint bytes and relative filenames, never model labels.

    Directory checkpoints (such as Diffusers pipelines) are canonicalized by
    sorted relative paths.  This is deliberately separate from a state-dict
    key digest: a parent/child identity must change when the actual weights or
    config bytes change.
    """

    root = Path(path)
    digest = hashlib.sha256()
    if root.is_file():
        with root.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty: %s" % root)
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelStructure:
    architecture: str
    block_count: int
    hidden_size: int
    ffn_width: int
    parameter_count: int
    checkpoint_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.architecture.strip():
            raise ValueError("architecture must not be empty")
        if self.block_count <= 0 or self.hidden_size <= 0 or self.ffn_width <= 0 or self.parameter_count < 0:
            raise ValueError("model structure dimensions must be positive")
        if len(str(self.checkpoint_sha256)) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    index: int
    parameter_count: int
    module_path: str

    def __post_init__(self) -> None:
        if not self.block_id.strip() or self.index < 0 or self.parameter_count < 0:
            raise ValueError("invalid block reference")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ChannelPruningConfig:
    block_ids: Tuple[str, ...]
    retained_indices: Tuple[int, ...]
    projection_name: str = "ffn"

    def __post_init__(self) -> None:
        if not self.block_ids:
            raise ValueError("block_ids must not be empty")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("block_ids must be unique")
        if not self.retained_indices:
            raise ValueError("retained_indices must not be empty")
        if len(set(self.retained_indices)) != len(self.retained_indices):
            raise ValueError("retained_indices must be unique")
        if any(index < 0 for index in self.retained_indices):
            raise ValueError("retained_indices must be non-negative")
        if not self.projection_name.strip():
            raise ValueError("projection_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class StructuralChange:
    kind: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CheckpointArtifact:
    path: str
    sha256: str
    architecture: str
    structure: ModelStructure
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path.strip() or len(self.sha256) != 64:
            raise ValueError("checkpoint artifact requires path and SHA-256")
        if self.structure.checkpoint_sha256 != self.sha256:
            raise ValueError("checkpoint artifact digest must match structure digest")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CompressionResult:
    status: str
    parent_checkpoint_sha256: str
    child_checkpoint: Optional[CheckpointArtifact]
    structural_change: Mapping[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    message: str = ""
    wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed"}:
            raise ValueError("compression status must be success or failed")
        if self.status == "success" and self.child_checkpoint is None:
            raise ValueError("successful compression requires a child checkpoint")
        if self.status == "failed" and not self.failure_type:
            raise ValueError("failed compression requires failure_type")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class WanModelAdapter(Protocol):
    architecture_name: str

    def load_model(self, checkpoint: str, *, device: str = "cpu", dtype: str = "float32") -> Any:
        ...

    def inspect(self, model: Any) -> ModelStructure:
        ...

    def get_blocks(self, model: Any) -> tuple[BlockRef, ...]:
        ...

    def remove_blocks(self, model: Any, block_ids: Sequence[str]) -> StructuralChange:
        ...

    def prune_channels(self, model: Any, config: ChannelPruningConfig) -> StructuralChange:
        ...

    def save_model(self, model: Any, output_dir: Path) -> CheckpointArtifact:
        ...

    def reload_model(self, artifact: CheckpointArtifact) -> Any:
        ...
