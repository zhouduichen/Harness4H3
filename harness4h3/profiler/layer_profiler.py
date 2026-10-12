from __future__ import annotations

import copy
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from ..adapters.wan.contracts import WanModelAdapter
from .report import ProfileReport
from .sensitivity import measure_block_sensitivity, measure_channel_sensitivity, measure_timestep_sensitivity


ForwardFn = Callable[[Any, Any], Any]


def _parameter_count(module: Any) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(item.numel()) for item in parameters())


def _default_sample(adapter: WanModelAdapter, model: Any, seed: int) -> Any:
    try:
        import torch
    except ImportError:
        return None
    structure = adapter.inspect(model)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(2, structure.hidden_size, generator=generator)


def _default_forward(model: Any, sample: Any) -> Any:
    if sample is None or not callable(model):
        return None
    try:
        import torch

        with torch.no_grad():
            return model(sample)
    except (RuntimeError, TypeError, ValueError):
        return None


def _resolve(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


def _estimate_flops(model: Any, module_path: str) -> Optional[int]:
    try:
        block = _resolve(model, module_path)
        up = getattr(block, "up_proj")
        down = getattr(block, "down_proj")
        return 2 * (int(up.in_features) * int(up.out_features) + int(down.in_features) * int(down.out_features))
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def profile_layers(adapter: WanModelAdapter, model: Any, *, sample: Any = None, forward_fn: Optional[ForwardFn] = None, measure_latency: bool = True, seed: int = 7) -> tuple[Mapping[str, Any], ...]:
    """Collect structural layer records without claiming unavailable metrics."""

    sample = _default_sample(adapter, model, seed) if sample is None else sample
    forward = forward_fn or _default_forward
    records = []
    for block in adapter.get_blocks(model):
        latency_ms: Optional[float] = None
        latency_available = False
        output_shape = None
        if measure_latency and sample is not None:
            started = time.perf_counter()
            try:
                value = forward(model, sample)
                if value is not None and hasattr(value, "shape"):
                    output_shape = list(value.shape)
                latency_ms = (time.perf_counter() - started) * 1000.0
                latency_available = True
            except (RuntimeError, TypeError, ValueError):
                latency_ms = None
        records.append({
            "block_id": block.block_id,
            "index": block.index,
            "module_path": block.module_path,
            "parameter_count": block.parameter_count,
            "estimated_flops": _estimate_flops(model, block.module_path),
            "latency_ms": latency_ms,
            "memory_delta_bytes": None,
            "measurement": {"latency_available": latency_available, "memory_available": False, "flops_available": False},
            "output_shape": output_shape,
        })
    return tuple(records)


def profile_model(
    adapter: WanModelAdapter,
    model: Any,
    *,
    seed: int = 7,
    measure_latency: bool = True,
    forward_fn: Optional[ForwardFn] = None,
    timestep_fn: Optional[Callable[[Any, str, Any], Any]] = None,
    max_channel_probes: int = 32,
) -> ProfileReport:
    structure = adapter.inspect(model)
    sample = _default_sample(adapter, model, seed)
    layers = profile_layers(adapter, model, sample=sample, forward_fn=forward_fn, measure_latency=measure_latency, seed=seed)
    block_sensitivity = measure_block_sensitivity(adapter, model, sample=sample, forward_fn=forward_fn)
    channel_sensitivity = measure_channel_sensitivity(adapter, model, max_probes=max_channel_probes)
    timestep_sensitivity = measure_timestep_sensitivity(model, sample=sample, forward_fn=timestep_fn or _default_timestep_forward)
    offline = bool(structure.metadata.get("offline_simulation", False))
    return ProfileReport(
        model=structure.to_dict(),
        layers=layers,
        block_sensitivity=block_sensitivity,
        channel_sensitivity=channel_sensitivity,
        timestep_sensitivity=timestep_sensitivity,
        provenance={
            "offline_simulation": offline,
            "seed": int(seed),
            "latency_available": any(item["measurement"]["latency_available"] for item in layers),
            "memory_available": False,
            "flops_available": False,
            "channel_probe_limit": int(max_channel_probes),
        },
    )


def _default_timestep_forward(model: Any, bucket: str, sample: Any) -> Any:
    """Use scaled fake inputs for timestep probes; real Wan layouts may fail closed."""

    scale = {"high_noise": 0.9, "middle": 0.5, "low_noise": 0.1}[bucket]
    return _default_forward(model, sample * scale)
