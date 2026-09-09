from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any, Dict, List

from .checkpoint import CheckpointInspectionError, CheckpointMetadata, TensorMetadata, sha256_file


GGML_DTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.data):
            raise CheckpointInspectionError("truncated GGUF header")
        result = self.data[self.offset : self.offset + size]
        self.offset += size
        return result

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        length = self.unpack("<Q")
        if length > len(self.data):
            raise CheckpointInspectionError("invalid GGUF string length")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CheckpointInspectionError("invalid UTF-8 in GGUF header: %s" % exc)

    def value(self, value_type: int) -> Any:
        formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = self.unpack("<I")
            count = self.unpack("<Q")
            if count > 10_000_000:
                raise CheckpointInspectionError("GGUF metadata array is too large")
            return [self.value(element_type) for _ in range(count)]
        try:
            return self.unpack(formats[value_type])
        except KeyError:
            raise CheckpointInspectionError("unsupported GGUF metadata type %d" % value_type)


def json_scalar(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(json_scalar(item) for item in value)
    return str(value)


def inspect_gguf(path: Path, include_file_sha256: bool = False) -> CheckpointMetadata:
    path = Path(path).resolve()
    if not path.is_file():
        raise CheckpointInspectionError("checkpoint is not a readable file: %s" % path)
    size_bytes = path.stat().st_size
    if size_bytes < 24:
        raise CheckpointInspectionError("GGUF checkpoint is truncated")
    with path.open("rb") as handle:
        header = handle.read(128 * 1024 * 1024)
    reader = _Reader(header)
    if reader.take(4) != b"GGUF":
        raise CheckpointInspectionError("invalid GGUF magic")
    version = reader.unpack("<I")
    if version not in {2, 3, 4}:
        raise CheckpointInspectionError("unsupported GGUF version %d" % version)
    tensor_count = reader.unpack("<Q")
    metadata_count = reader.unpack("<Q")
    if tensor_count > 10_000_000 or metadata_count > 10_000_000:
        raise CheckpointInspectionError("GGUF header counts are unreasonably large")
    metadata: Dict[str, str] = {
        "GGUF.version": str(version),
        "GGUF.tensor_count": str(tensor_count),
        "GGUF.kv_count": str(metadata_count),
    }
    for _ in range(metadata_count):
        key = reader.string()
        value = reader.value(reader.unpack("<I"))
        metadata[key] = json_scalar(value)
    tensors: List[TensorMetadata] = []
    for _ in range(tensor_count):
        name = reader.string()
        dimensions = reader.unpack("<I")
        if dimensions > 64:
            raise CheckpointInspectionError("invalid GGUF tensor rank")
        shape = tuple(reader.unpack("<Q") for _ in range(dimensions))
        dtype_code = reader.unpack("<I")
        offset = reader.unpack("<Q")
        dtype = GGML_DTYPE_NAMES.get(dtype_code, "GGML_%d" % dtype_code)
        tensors.append(TensorMetadata(name, dtype, shape, (offset, offset)))
    header_bytes = header[: reader.offset]
    return CheckpointMetadata(
        path=path,
        format="gguf",
        size_bytes=size_bytes,
        header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        file_sha256=sha256_file(path) if include_file_sha256 else None,
        tensors=tuple(sorted(tensors, key=lambda item: item.name)),
        metadata=metadata,
    )
