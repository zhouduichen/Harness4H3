from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .checkpoint import CheckpointInspectionError, CheckpointMetadata, inspect_safetensors
from .gguf import inspect_gguf
from .state import ModelState


H3_TENSOR_SIGNATURE = {
    "audio_patch_proj.weight",
    "video_patch_proj.weight",
    "condition_proj.weight",
    "blocks.0.attn.qkv_proj.weight",
    "token_refiner.blocks.0.attn.qkv_proj.weight",
    "final_layer.audio_out.weight",
    "final_layer.video_out.weight",
}

DTYPE_NAMES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def _primary_dtype(metadata: CheckpointMetadata) -> str:
    counts = metadata.dtype_parameter_counts
    if len(counts) > 1:
        return "mixed"
    code = max(counts, key=counts.get)
    return DTYPE_NAMES.get(code, code.lower())


def _number_of_blocks(names: Sequence[str]) -> Optional[int]:
    indices = []
    for name in names:
        match = re.match(r"^blocks\.(\d+)\.", name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else None


def _matrix_dimension(metadata: CheckpointMetadata, names: Sequence[str], axis: int) -> Optional[int]:
    by_name = {item.name: item for item in metadata.tensors}
    for name in names:
        tensor = by_name.get(name)
        if tensor is not None and len(tensor.shape) > axis:
            return tensor.shape[axis]
    return None


def _quantization(path: Path, dtype: str) -> Mapping[str, Any]:
    filename = path.name.lower()
    if "nvfp4" in filename:
        return {"bits": 4, "scheme": "nvfp4", "source": "filename"}
    match = re.search(r"(?:^|[-_])(q[248](?:_[a-z0-9]+)?)(?:[-_.]|$)", filename)
    if match:
        bits = int(match.group(1)[1])
        return {"bits": bits, "scheme": match.group(1).upper(), "source": "filename"}
    if "int8" in filename or dtype == "int8":
        return {"bits": 8, "scheme": "int8", "source": "filename_or_dtype"}
    return {"bits": 16 if dtype in {"float16", "bfloat16"} else None, "scheme": "none"}


class H3Inspector:
    """Build a normalized ModelState from checkpoint metadata without loading weights."""

    def inspect(
        self,
        checkpoint_path: Path,
        model_id: str = "M0000",
        parent_model_id: Optional[str] = None,
        architecture_name: Optional[str] = None,
        sampling_steps: Optional[int] = None,
        components: Optional[Mapping[str, Any]] = None,
        include_file_sha256: bool = False,
    ) -> ModelState:
        path = Path(checkpoint_path).resolve()
        if path.suffix.lower() == ".safetensors":
            checkpoint = inspect_safetensors(path, include_file_sha256)
        elif path.suffix.lower() == ".gguf":
            checkpoint = inspect_gguf(path, include_file_sha256)
        else:
            raise CheckpointInspectionError("unsupported checkpoint format %s; expected .safetensors or .gguf" % path.suffix)
        names = [item.name for item in checkpoint.tensors]
        name_set = set(names)
        warnings = []
        signature_matches = sorted(H3_TENSOR_SIGNATURE & name_set)
        if architecture_name:
            architecture = architecture_name
        elif H3_TENSOR_SIGNATURE.issubset(name_set):
            architecture = "MiniMax-H3"
        elif "minimax_h3" in path.name.lower() or "minimax-h3" in path.name.lower():
            architecture = "MiniMax-H3"
            warnings.append("architecture inferred from checkpoint filename; full H3 tensor signature was not present")
        else:
            architecture = "unknown"
            warnings.append("checkpoint does not contain the complete MiniMax-H3 tensor signature")

        dtype = _primary_dtype(checkpoint)
        qkv = next((item for item in checkpoint.tensors if item.name == "blocks.0.attn.qkv_proj.weight"), None)
        if qkv is None:
            hidden_size = None
        elif checkpoint.format == "gguf" and len(qkv.shape) >= 2:
            hidden_size = qkv.shape[0] // 2
        else:
            hidden_size = _matrix_dimension(checkpoint, ["blocks.0.attn.qkv_proj.weight"], 1)
        ffn_width = _matrix_dimension(
            checkpoint,
            ["blocks.0.ffn.up_proj.weight", "blocks.0.mlp.up_proj.weight", "blocks.0.ffn.fc1.weight", "blocks.0.mlp.fc1.weight"],
            1 if checkpoint.format == "gguf" else 0,
        )
        provenance: Dict[str, Any] = {
            "kind": "checkpoint_inspection",
            "format": checkpoint.format,
            "size_bytes": checkpoint.size_bytes,
            "header_sha256": checkpoint.header_sha256,
            "file_sha256": checkpoint.file_sha256,
            "tensor_count": len(checkpoint.tensors),
            "dtype_parameter_counts": dict(checkpoint.dtype_parameter_counts),
            "h3_signature_matches": signature_matches,
            "weights_loaded": False,
        }
        return ModelState(
            model_id=model_id,
            parent_model_id=parent_model_id,
            checkpoint_path=str(path),
            architecture_name=architecture,
            parameter_count=checkpoint.parameter_count,
            trainable_parameter_count=0,
            num_blocks=_number_of_blocks(names),
            hidden_size=hidden_size,
            num_attention_heads=None,
            ffn_width=ffn_width,
            dtype=dtype,
            quantization=_quantization(path, dtype),
            sampling_steps=sampling_steps,
            components=dict(components or {"diffusion_model": str(path)}),
            algorithm_state={},
            runtime_state={"inspector": checkpoint.format + "_header", "weights_loaded": False},
            measured_metrics={"model_size_gb": checkpoint.size_bytes / 1_000_000_000},
            provenance=provenance,
            warnings=warnings,
        )
