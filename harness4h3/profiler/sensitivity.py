from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional

from ..adapters.wan.contracts import ChannelPruningConfig, WanModelAdapter


def _distance(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    try:
        import torch

        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.shape != right.shape:
            return None
        return float(torch.mean(torch.abs(left.detach().float() - right.detach().float())).item())
    except ImportError:
        return None


def _forward(model: Any, sample: Any, forward_fn: Optional[Callable[[Any, Any], Any]]) -> Any:
    if forward_fn is not None:
        return forward_fn(model, sample)
    if sample is None or not callable(model):
        return None
    try:
        import torch

        with torch.no_grad():
            return model(sample)
    except (RuntimeError, TypeError, ValueError):
        return None


def measure_block_sensitivity(adapter: WanModelAdapter, model: Any, *, sample: Any = None, forward_fn: Optional[Callable[[Any, Any], Any]] = None) -> tuple[Mapping[str, Any], ...]:
    base = _forward(model, sample, forward_fn)
    records = []
    for block in adapter.get_blocks(model):
        try:
            candidate = copy.deepcopy(model)
            adapter.remove_blocks(candidate, (block.block_id,))
            distance = _distance(base, _forward(candidate, sample, forward_fn))
            records.append({"block_id": block.block_id, "importance_score": distance, "available": distance is not None, "method": "leave_one_block_out"})
        except (RuntimeError, TypeError, ValueError, AttributeError):
            records.append({"block_id": block.block_id, "importance_score": None, "available": False, "method": "leave_one_block_out"})
    return tuple(records)


def _resolve(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


def measure_channel_sensitivity(adapter: WanModelAdapter, model: Any, *, max_probes: int = 32) -> tuple[Mapping[str, Any], ...]:
    records = []
    for block in adapter.get_blocks(model):
        try:
            module = _resolve(model, block.module_path)
            up = getattr(module, "up_proj")
            down = getattr(module, "down_proj")
            width = int(up.out_features)
            limit = min(width, max(1, int(max_probes)))
            values = []
            for channel in range(limit):
                weight = float(up.weight[channel].detach().float().abs().mean().item() + down.weight[:, channel].detach().float().abs().mean().item())
                values.append({"channel": channel, "importance_score": weight, "available": True})
            records.append({"block_id": block.block_id, "width": width, "channels": values, "method": "paired_ffn_weight_contribution", "available": True})
        except (AttributeError, IndexError, TypeError, RuntimeError):
            records.append({"block_id": block.block_id, "width": None, "channels": [], "method": "paired_ffn_weight_contribution", "available": False})
    return tuple(records)


def measure_timestep_sensitivity(model: Any, *, sample: Any = None, forward_fn: Optional[Callable[[Any, str, Any], Any]] = None) -> dict[str, Mapping[str, Any]]:
    buckets = (("high_noise", 0.9), ("middle", 0.5), ("low_noise", 0.1))
    records: dict[str, Mapping[str, Any]] = {}
    for name, value in buckets:
        if forward_fn is None or sample is None:
            records[name] = {"timestep": value, "importance_score": None, "available": False, "method": "timestep_probe"}
            continue
        try:
            output = forward_fn(model, name, sample)
            score = float(output.detach().float().abs().mean().item()) if hasattr(output, "detach") else float(abs(output))
            records[name] = {"timestep": value, "importance_score": score, "available": True, "method": "timestep_probe"}
        except (RuntimeError, TypeError, ValueError, AttributeError):
            records[name] = {"timestep": value, "importance_score": None, "available": False, "method": "timestep_probe"}
    return records

