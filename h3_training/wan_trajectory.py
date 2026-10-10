"""Immutable, digest-checked artifacts for native Wan Teacher trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_SCHEMA_VERSION = 1
_TENSOR_ARTIFACT = "trajectory-tensors.pt"
_RECORD_ARTIFACT = "trajectory-record.json"


def tensor_digest(value: torch.Tensor) -> str:
    """Hash dtype, shape, and canonical CPU tensor bytes."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("trajectory digest values must be torch.Tensor instances")
    canonical = value.detach().to(device="cpu").contiguous()
    header = f"{canonical.dtype}|{tuple(canonical.shape)}|".encode("utf-8")
    return hashlib.sha256(header + canonical.numpy().tobytes(order="C")).hexdigest()


def _snapshot_digest(value: Any) -> str:
    """Digest scheduler snapshots without depending on object identity."""

    if isinstance(value, torch.Tensor):
        return "tensor:" + tensor_digest(value)
    if isinstance(value, Mapping):
        payload = {str(key): _snapshot_digest(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "mapping:" + hashlib.sha256(encoded).hexdigest()
    if isinstance(value, (tuple, list)):
        encoded = json.dumps([_snapshot_digest(item) for item in value], separators=(",", ":")).encode("utf-8")
        return "sequence:" + hashlib.sha256(encoded).hexdigest()
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return "scalar:" + hashlib.sha256(encoded).hexdigest()


def tensors_digest(value: "LoadedWanTrajectory") -> str:
    """Hash every tensor and native scheduler snapshot in a fixed order."""

    parts = [
        "noise=" + tensor_digest(value.noise),
        "clean_latent=" + tensor_digest(value.clean_latent),
    ]
    parts.extend(f"teacher_state[{index}]={tensor_digest(item)}" for index, item in enumerate(value.teacher_states))
    parts.extend(
        f"teacher_snapshot[{index}]={_snapshot_digest(item)}"
        for index, item in enumerate(value.teacher_scheduler_snapshots)
    )
    parts.extend(
        f"teacher_prediction[{index}]={tensor_digest(item)}"
        for index, item in enumerate(value.teacher_predictions)
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("trajectory tensor artifact contains a non-tensor value")
    return value.detach().to(device="cpu").contiguous()


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _cpu_tensor(value)
    if isinstance(value, Mapping):
        return {str(key): _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    return value


@dataclass(frozen=True)
class WanTrajectoryRecord:
    schema_version: int
    cache_item: str
    cache_digest: str
    conditioning_digest: str
    clean_latent_digest: str
    noise_digest: str
    teacher_checkpoint_digest: str
    scheduler_digest: str
    source_nfe: int
    target_nfe: int
    source_sigmas: tuple[float, ...]
    target_sigmas: tuple[float, ...]
    source_timesteps: tuple[float, ...]
    target_timesteps: tuple[float, ...]
    tensor_artifact: str = _TENSOR_ARTIFACT
    tensor_digest: str = ""

    @classmethod
    def load(cls, directory: Path) -> "WanTrajectoryRecord":
        directory = Path(directory)
        payload = json.loads((directory / _RECORD_ARTIFACT).read_text(encoding="utf-8"))
        record = cls(
            schema_version=int(payload["schema_version"]),
            cache_item=str(payload["cache_item"]),
            cache_digest=str(payload["cache_digest"]),
            conditioning_digest=str(payload["conditioning_digest"]),
            clean_latent_digest=str(payload["clean_latent_digest"]),
            noise_digest=str(payload["noise_digest"]),
            teacher_checkpoint_digest=str(payload["teacher_checkpoint_digest"]),
            scheduler_digest=str(payload["scheduler_digest"]),
            source_nfe=int(payload["source_nfe"]),
            target_nfe=int(payload["target_nfe"]),
            source_sigmas=tuple(float(value) for value in payload["source_sigmas"]),
            target_sigmas=tuple(float(value) for value in payload["target_sigmas"]),
            source_timesteps=tuple(float(value) for value in payload["source_timesteps"]),
            target_timesteps=tuple(float(value) for value in payload["target_timesteps"]),
            tensor_artifact=str(payload.get("tensor_artifact", _TENSOR_ARTIFACT)),
            tensor_digest=str(payload["tensor_digest"]),
        )
        load_trajectory_tensors(directory, record).validate()
        return record


@dataclass(frozen=True)
class LoadedWanTrajectory:
    metadata: WanTrajectoryRecord
    noise: torch.Tensor
    clean_latent: torch.Tensor
    teacher_states: tuple[torch.Tensor, ...]
    teacher_scheduler_snapshots: tuple[dict[str, Any], ...]
    teacher_predictions: tuple[torch.Tensor, ...]

    def validate(self) -> None:
        metadata = self.metadata
        if metadata.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema version: {metadata.schema_version}")
        if metadata.source_nfe != 2 * metadata.target_nfe or metadata.target_nfe <= 0:
            raise ValueError("trajectory must contain a binary source/target NFE pair")
        if len(metadata.source_sigmas) != metadata.source_nfe + 1:
            raise ValueError("source sigma boundary count does not match source NFE")
        if len(metadata.target_sigmas) != metadata.target_nfe + 1:
            raise ValueError("target sigma boundary count does not match target NFE")
        if len(metadata.source_timesteps) != metadata.source_nfe:
            raise ValueError("source timestep count does not match source NFE")
        if len(metadata.target_timesteps) != metadata.target_nfe:
            raise ValueError("target timestep count does not match target NFE")
        if len(self.teacher_states) != metadata.source_nfe + 1:
            raise ValueError("teacher state boundary count does not match source NFE")
        if len(self.teacher_scheduler_snapshots) != metadata.source_nfe + 1:
            raise ValueError("teacher scheduler snapshot count does not match source NFE")
        if len(self.teacher_predictions) != metadata.source_nfe:
            raise ValueError("teacher prediction count does not match source NFE")
        for name, values in (
            ("source sigma", metadata.source_sigmas),
            ("target sigma", metadata.target_sigmas),
        ):
            if not values or values[-1] != 0.0 or any(left <= right for left, right in zip(values, values[1:])):
                raise ValueError(f"{name} table must be strictly decreasing and end at zero")
        for name, value in (("noise", self.noise), ("clean latent", self.clean_latent)):
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError(f"{name} must be a detached CPU tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        for collection_name, collection in (
            ("teacher state", self.teacher_states),
            ("teacher prediction", self.teacher_predictions),
        ):
            for value in collection:
                if value.device.type != "cpu" or value.requires_grad:
                    raise ValueError(f"{collection_name} tensors must be detached CPU tensors")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{collection_name} contains non-finite values")
        if tensor_digest(self.noise) != metadata.noise_digest:
            raise ValueError("noise digest does not match trajectory metadata")
        if tensor_digest(self.clean_latent) != metadata.clean_latent_digest:
            raise ValueError("clean latent digest does not match trajectory metadata")
        if tensors_digest(self) != metadata.tensor_digest:
            raise ValueError("trajectory tensor digest does not match trajectory metadata")

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        self.validate()
        payload = {
            "noise": _cpu_tensor(self.noise),
            "clean_latent": _cpu_tensor(self.clean_latent),
            "teacher_states": [_cpu_tensor(item) for item in self.teacher_states],
            "teacher_scheduler_snapshots": [_cpu_snapshot(item) for item in self.teacher_scheduler_snapshots],
            "teacher_predictions": [_cpu_tensor(item) for item in self.teacher_predictions],
        }
        _atomic_torch_save(directory / self.metadata.tensor_artifact, payload)
        metadata = asdict(self.metadata)
        metadata["tensor_digest"] = tensors_digest(self)
        _atomic_bytes(
            directory / _RECORD_ARTIFACT,
            json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8"),
        )


def load_trajectory_tensors(directory: Path, record: WanTrajectoryRecord) -> LoadedWanTrajectory:
    directory = Path(directory)
    payload = torch.load(directory / record.tensor_artifact, map_location="cpu", weights_only=True)
    required = {"noise", "clean_latent", "teacher_states", "teacher_scheduler_snapshots", "teacher_predictions"}
    if set(payload) != required:
        raise ValueError("trajectory tensor artifact keys do not match the schema")
    return LoadedWanTrajectory(
        metadata=record,
        noise=_cpu_tensor(payload["noise"]),
        clean_latent=_cpu_tensor(payload["clean_latent"]),
        teacher_states=tuple(_cpu_tensor(item) for item in payload["teacher_states"]),
        teacher_scheduler_snapshots=tuple(payload["teacher_scheduler_snapshots"]),
        teacher_predictions=tuple(_cpu_tensor(item) for item in payload["teacher_predictions"]),
    )


__all__ = [
    "LoadedWanTrajectory",
    "WanTrajectoryRecord",
    "json_digest",
    "load_trajectory_tensors",
    "tensor_digest",
    "tensors_digest",
]
