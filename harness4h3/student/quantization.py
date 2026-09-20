"""Portable per-tensor Student quantization with an explicit dequantization path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class StudentQuantizationError(ValueError):
    pass


def quantize_checkpoint(source: Path, destination: Path, *, bits: int, metadata: Mapping[str, str]) -> Tuple[int, int]:
    if int(bits) != 8:
        raise StudentQuantizationError("only int8 Student quantization is supported")
    tensors = load_file(str(source), device="cpu")
    output: Dict[str, torch.Tensor] = {}
    quantized = 0
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            output[name] = value
            continue
        maximum = value.detach().abs().amax().clamp_min(1e-8)
        scale = (maximum / 127.0).to(dtype=torch.float32)
        output[name] = torch.round(value.detach().to(dtype=torch.float32) / scale).clamp(-127, 127).to(torch.int8)
        output[name + ".__scale"] = scale.reshape(1).contiguous()
        quantized += 1
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        output,
        str(destination),
        metadata={**{str(key): str(value) for key, value in metadata.items()}, "quantization": "int8", "quantized_tensor_count": str(quantized)},
    )
    return quantized, destination.stat().st_size


def load_student_state(path: Path) -> Tuple[Dict[str, torch.Tensor], Mapping[str, str]]:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    tensors = load_file(str(path), device="cpu")
    if metadata.get("quantization") != "int8":
        return dict(tensors), metadata
    state: Dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if name.endswith(".__scale"):
            continue
        scale_name = name + ".__scale"
        scale = tensors.get(scale_name)
        if scale is None:
            # Integer/bool buffers are intentionally copied verbatim.  Only
            # floating tensors receive a sidecar scale during quantization.
            state[name] = value
            continue
        state[name] = value.to(dtype=torch.float32) * scale.reshape(1).to(dtype=torch.float32)
    return state, metadata


__all__ = ["StudentQuantizationError", "load_student_state", "quantize_checkpoint"]
