"""Cryptographic and tensor-level evidence for deployable children."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Mapping

import torch

from h3_training.algorithms.base import ModelRole
from .state import TrainingFailure


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParentEvidence:
    path: Path
    sha256: str
    tensors: Mapping[str, torch.Tensor]
    trainable_names: FrozenSet[str]
    frozen_names: FrozenSet[str]


@dataclass(frozen=True)
class ChildEvidence:
    path: Path
    parent_sha256: str
    child_sha256: str
    changed_trainable: int
    unchanged_frozen: int
    reloaded: bool
    manifest_path: Path


def capture_parent(path: Path, role: ModelRole, trainable_names) -> ParentEvidence:
    names = frozenset(trainable_names)
    tensors = {name: value.detach().cpu().clone() for name, value in role.model.state_dict().items()}
    tensor_names = frozenset(tensors)
    if not names.issubset(tensor_names):
        raise TrainingFailure("invalid_training_config", "trainable evidence names do not match model state")
    return ParentEvidence(Path(path), sha256_file(path), tensors, names, tensor_names - names)


def save_verified_child(role: ModelRole, parent: ParentEvidence, path: Path) -> ChildEvidence:
    path = Path(path)
    if path.resolve() == parent.path.resolve():
        raise TrainingFailure("invalid_training_config", "child checkpoint path must differ from parent")
    if sha256_file(parent.path) != parent.sha256:
        raise TrainingFailure("parent_modified", "parent bytes changed during training")
    current = {name: value.detach().cpu() for name, value in role.model.state_dict().items()}
    changed = {name for name in parent.trainable_names if name in current and not torch.equal(parent.tensors[name], current[name])}
    if not changed:
        raise TrainingFailure("unchanged_child", "no trainable tensor changed")
    frozen_changed = {
        name for name in parent.frozen_names if name in current and not torch.equal(parent.tensors[name], current[name])
    }
    if frozen_changed:
        raise TrainingFailure("frozen_tensor_changed", ",".join(sorted(frozen_changed)[:5]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        role.adapter.save_role(role, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(parent.path) != parent.sha256:
        raise TrainingFailure("parent_modified", "parent bytes changed while publishing child")
    child_hash = sha256_file(path)
    if child_hash == parent.sha256:
        raise TrainingFailure("unchanged_child", "child bytes equal parent bytes")
    try:
        reloaded = role.adapter.reload_role(path)
        for tensor in reloaded.model.state_dict().values():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError("non-finite child tensor")
    except Exception as exc:
        raise TrainingFailure("child_reload_failed", str(exc)) from exc
    manifest_path = path.with_suffix(path.suffix + ".evidence.json")
    evidence = ChildEvidence(path, parent.sha256, child_hash, len(changed), len(parent.frozen_names), True, manifest_path)
    manifest = asdict(evidence)
    manifest["path"] = str(path)
    manifest["manifest_path"] = str(manifest_path)
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with temporary_manifest.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_manifest, manifest_path)
    return evidence
