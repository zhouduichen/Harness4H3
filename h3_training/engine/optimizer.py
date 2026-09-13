"""Optimizer validation and gradient utilities."""

from typing import Iterable

import torch
from torch import nn

from .state import TrainingFailure


def finite_total_loss(loss: torch.Tensor) -> None:
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise TrainingFailure("nonfinite_loss", "total_loss must be a finite scalar")


def gradient_norm(modules: Iterable[nn.Module], max_norm: float) -> float:
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.grad is not None and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    if not parameters:
        raise TrainingFailure("zero_gradient", "scheduled optimizer produced no gradients")
    nonzero = any(torch.count_nonzero(parameter.grad).item() for parameter in parameters)
    if not nonzero:
        raise TrainingFailure("zero_gradient", "scheduled optimizer produced only zero gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm):
        raise TrainingFailure("nonfinite_loss", "gradient norm is not finite")
    return float(norm)
