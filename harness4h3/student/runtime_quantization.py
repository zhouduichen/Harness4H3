"""Runtime-only loading of the Student's artifact-backed int8 weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from .model import build_student
from .proposal import StudentProposal, StudentTarget


class QuantizedLinear(nn.Module):
    """A weight-only int8 Linear that never stores a full floating weight."""

    def __init__(
        self,
        linear: nn.Linear,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if weight_int8.dtype != torch.int8:
            raise ValueError("runtime linear weight must be int8")
        if tuple(weight_int8.shape) != tuple(linear.weight.shape):
            raise ValueError(
                "runtime linear weight shape %s does not match %s"
                % (tuple(weight_int8.shape), tuple(linear.weight.shape))
            )
        self.register_buffer("weight_int8", weight_int8.detach().to(dtype=torch.int8), persistent=False)
        self.register_buffer("scale", scale.detach().to(dtype=torch.float32).reshape(1), persistent=False)
        # Some registered Student modules inspect ``projection.weight.dtype``
        # for scheduler input casting.  Keep only a zero-sized dtype anchor.
        self.register_buffer(
            "_dtype_anchor",
            torch.empty(0, dtype=linear.weight.dtype),
            persistent=False,
        )
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().to(dtype=torch.float32), persistent=False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
    ) -> "QuantizedLinear":
        return cls(linear, weight_int8, scale)

    @property
    def weight(self) -> torch.Tensor:
        return self._dtype_anchor

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight_int8.to(dtype=value.dtype).mul(self.scale.to(dtype=value.dtype))
        bias = None if self.bias is None else self.bias.to(dtype=value.dtype)
        return F.linear(value, weight, bias)


def _metadata(path: Path) -> Mapping[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _replace_linear_modules(
    model: nn.Module,
    tensors: Mapping[str, torch.Tensor],
) -> set[str]:
    linear_weight_names: set[str] = set()
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            prefix = "%s." % parent_name if parent_name else ""
            weight_name = "%s%s.weight" % (prefix, child_name)
            scale_name = weight_name + ".__scale"
            if weight_name not in tensors or scale_name not in tensors:
                raise ValueError("int8 artifact is missing %s or %s" % (weight_name, scale_name))
            replacement = QuantizedLinear.from_linear(
                child,
                tensors[weight_name],
                tensors[scale_name],
            )
            bias_name = "%s%s.bias" % (prefix, child_name)
            if child.bias is not None:
                if bias_name not in tensors:
                    raise ValueError("int8 artifact is missing %s" % bias_name)
                bias = tensors[bias_name]
                bias_scale = tensors.get(bias_name + ".__scale")
                if bias_scale is not None:
                    bias = bias.to(dtype=torch.float32).mul(
                        bias_scale.reshape(1).to(dtype=torch.float32)
                    )
                if replacement.bias is not None:
                    replacement.bias.copy_(bias.to(dtype=torch.float32))
            setattr(parent, child_name, replacement)
            linear_weight_names.add(weight_name)
            if child.bias is not None:
                linear_weight_names.add(bias_name)
    if not linear_weight_names:
        raise ValueError("int8 artifact contains no Student Linear weights")
    return linear_weight_names


def load_runtime_quantized_model(
    proposal: StudentProposal,
    checkpoint: Path,
    target: StudentTarget,
    device: torch.device,
) -> nn.Module:
    """Load a compiled int8 artifact without materializing full fp weights."""

    checkpoint = Path(checkpoint)
    metadata = _metadata(checkpoint)
    if str(metadata.get("quantization", "")) != "int8":
        raise ValueError("target runtime requires artifact metadata quantization=int8")
    tensors = load_file(str(checkpoint), device="cpu")
    model = build_student(proposal, device=torch.device("cpu"), target=target)
    linear_weight_names = _replace_linear_modules(model, tensors)

    model_state_names = set(model.state_dict())
    state: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if name.endswith(".__scale") or name in linear_weight_names:
            continue
        if name not in model_state_names:
            raise ValueError("int8 artifact contains unexpected tensor: %s" % name)
        scale = tensors.get(name + ".__scale")
        if scale is not None:
            value = value.to(dtype=torch.float32).mul(scale.reshape(1).to(dtype=torch.float32))
        state[name] = value

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError("int8 Student state mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    dtype = torch.bfloat16 if proposal.deployment.precision == "bf16" else torch.float16
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


__all__ = ["QuantizedLinear", "load_runtime_quantized_model"]
