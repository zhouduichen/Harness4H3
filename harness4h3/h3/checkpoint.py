from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


class CheckpointInspectionError(ValueError):
    pass


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]

    @property
    def parameter_count(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    format: str
    size_bytes: int
    header_sha256: str
    file_sha256: Optional[str]
    tensors: Tuple[TensorMetadata, ...]
    metadata: Mapping[str, str]

    @property
    def parameter_count(self) -> int:
        return sum(item.parameter_count for item in self.tensors)

    @property
    def dtype_parameter_counts(self) -> Mapping[str, int]:
        result: Dict[str, int] = {}
        for tensor in self.tensors:
            result[tensor.dtype] = result.get(tensor.dtype, 0) + tensor.parameter_count
        return result


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path, include_file_sha256: bool = False) -> CheckpointMetadata:
    path = Path(path).resolve()
    if not path.is_file():
        raise CheckpointInspectionError("checkpoint is not a readable file: %s" % path)
    size_bytes = path.stat().st_size
    if size_bytes < 10:
        raise CheckpointInspectionError("safetensors checkpoint is truncated")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 1 or header_length > 128 * 1024 * 1024 or header_length > size_bytes - 8:
            raise CheckpointInspectionError("invalid safetensors header length")
        header = handle.read(header_length)
    try:
        raw = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointInspectionError("invalid safetensors JSON header: %s" % exc)
    if not isinstance(raw, Mapping):
        raise CheckpointInspectionError("safetensors header must be a JSON object")

    user_metadata = raw.get("__metadata__") or {}
    if not isinstance(user_metadata, Mapping):
        raise CheckpointInspectionError("safetensors __metadata__ must be an object")
    tensors = []
    payload_size = size_bytes - 8 - header_length
    for name, value in raw.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, Mapping):
            raise CheckpointInspectionError("invalid tensor metadata for %s" % name)
        dtype = str(value.get("dtype", "")).strip()
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not dtype or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise CheckpointInspectionError("incomplete tensor metadata for %s" % name)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
            raise CheckpointInspectionError("invalid tensor shape for %s" % name)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in offsets):
            raise CheckpointInspectionError("invalid tensor offsets for %s" % name)
        start, end = int(offsets[0]), int(offsets[1])
        if end < start or end > payload_size:
            raise CheckpointInspectionError("tensor data lies outside checkpoint for %s" % name)
        tensors.append(TensorMetadata(str(name), dtype, tuple(shape), (start, end)))
    if not tensors:
        raise CheckpointInspectionError("safetensors checkpoint contains no tensors")
    tensors.sort(key=lambda item: item.name)
    return CheckpointMetadata(
        path=path,
        format="safetensors",
        size_bytes=size_bytes,
        header_sha256=hashlib.sha256(header).hexdigest(),
        file_sha256=sha256_file(path) if include_file_sha256 else None,
        tensors=tuple(tensors),
        metadata={str(key): str(value) for key, value in user_metadata.items()},
    )
