from __future__ import annotations

import copy
import statistics
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from ..adapters.wan.contracts import WanModelAdapter
from .calibration import CalibrationCorpus
from .report import ProfileReport
from .sensitivity import measure_attention_sensitivity, measure_block_sensitivity, measure_channel_sensitivity, measure_head_sensitivity, measure_sampling_step_sensitivity, measure_timestep_sensitivity, measure_token_sensitivity


ForwardFn = Callable[[Any, Any], Any]


def _parameter_count(module: Any) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(item.numel()) for item in parameters())


def _default_sample(adapter: WanModelAdapter, model: Any, seed: int) -> Any:
    profile_sample = getattr(adapter, "profile_sample", None)
    if callable(profile_sample):
        return profile_sample(model, seed)
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


def _adapter_forward(adapter: WanModelAdapter, model: Any, sample: Any) -> Any:
    forward_profile = getattr(adapter, "forward_profile", None)
    if callable(forward_profile):
        try:
            return forward_profile(model, sample)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return None
    return _default_forward(model, sample)


def _resolve_module(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


def _first_tensor(value: Any) -> Any:
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    elif hasattr(value, "device") and hasattr(value, "shape"):
        return value
    return None


def _runtime_device(model: Any, sample: Any) -> Any:
    tensor = _first_tensor(sample)
    if tensor is not None:
        return getattr(tensor, "device", None)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except (StopIteration, AttributeError):
            pass
    return None


def _timed_layer_forward(
    model: Any,
    sample: Any,
    forward: ForwardFn,
    blocks: Sequence[Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[Mapping[str, float], Mapping[str, Any]]:
    """Measure each block hook, rather than repeating whole-model latency."""

    if warmup < 0 or repeats <= 0:
        raise ValueError("latency warmup must be non-negative and repeats positive")
    try:
        import torch
    except ImportError:
        return {}, {"available": False, "reason": "torch_unavailable"}

    device = _runtime_device(model, sample)
    use_cuda = bool(device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available())
    durations: dict[str, list[float]] = {str(item.block_id): [] for item in blocks}
    starts: dict[str, Any] = {}
    handles = []
    peak_bytes: Optional[int] = None

    def pre_hook(block_id: str):
        def before(_module: Any, _inputs: Any) -> None:
            if use_cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(device=device))
                starts[block_id] = event
            else:
                starts[block_id] = time.perf_counter()
        return before

    def post_hook(block_id: str):
        def after(_module: Any, _inputs: Any, _output: Any) -> None:
            start = starts.pop(block_id, None)
            if start is None:
                return
            if use_cuda:
                end = torch.cuda.Event(enable_timing=True)
                end.record(torch.cuda.current_stream(device=device))
                durations[block_id].append((start, end))
            else:
                durations[block_id].append((time.perf_counter() - start) * 1000.0)
        return after

    try:
        for block in blocks:
            module = _resolve_module(model, str(block.module_path))
            register_pre = getattr(module, "register_forward_pre_hook", None)
            register_post = getattr(module, "register_forward_hook", None)
            if not callable(register_pre) or not callable(register_post):
                continue
            handles.append(register_pre(pre_hook(str(block.block_id))))
            handles.append(register_post(post_hook(str(block.block_id))))
        for _ in range(int(warmup)):
            forward(model, sample)
        if use_cuda:
            torch.cuda.synchronize(device=device)
            try:
                torch.cuda.reset_peak_memory_stats(device=device)
            except (AttributeError, RuntimeError):
                pass
        for _ in range(int(repeats)):
            forward(model, sample)
            if use_cuda:
                torch.cuda.synchronize(device=device)
                try:
                    value = int(torch.cuda.max_memory_allocated(device=device))
                    peak_bytes = max(value, peak_bytes or 0)
                except (AttributeError, RuntimeError):
                    pass
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return {}, {"available": False, "reason": "forward_failed", "device": str(device) if device is not None else None}
    finally:
        for handle in handles:
            handle.remove()

    measured: dict[str, float] = {}
    for block_id, values in durations.items():
        if use_cuda:
            try:
                measured[block_id] = float(statistics.median(start.elapsed_time(end) for start, end in values)) if values else None  # type: ignore[assignment]
            except (RuntimeError, ValueError):
                measured[block_id] = None  # type: ignore[assignment]
        elif values:
            measured[block_id] = float(statistics.median(values))
    return measured, {
        "available": bool(measured),
        "device": str(device) if device is not None else "unknown",
        "method": "per_block_forward_hook_cuda_events" if use_cuda else "per_block_forward_hook_perf_counter",
        "warmup": int(warmup),
        "repeats": int(repeats),
        "peak_memory_bytes": peak_bytes,
        "memory_available": peak_bytes is not None,
    }


def _resolve(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


def _estimate_flops(model: Any, module_path: str) -> Optional[int]:
    try:
        block = _resolve(model, module_path)
        up, down = _paired_ffn(block)
        return 2 * (int(up.in_features) * int(up.out_features) + int(down.in_features) * int(down.out_features))
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _paired_ffn(block: Any) -> tuple[Any, Any]:
    candidates = (("up_proj", "down_proj"), ("ffn.up_proj", "ffn.down_proj"), ("ffn.net.0.proj", "ffn.net.2"))
    for up_path, down_path in candidates:
        try:
            up = _resolve(block, up_path)
            down = _resolve(block, down_path)
        except (AttributeError, IndexError, TypeError):
            continue
        if all(hasattr(module, attribute) for module in (up, down) for attribute in ("in_features", "out_features")):
            return up, down
    raise AttributeError("paired FFN projections unavailable")


def _profile_layers(
    adapter: WanModelAdapter,
    model: Any,
    *,
    sample: Any = None,
    forward_fn: Optional[ForwardFn] = None,
    measure_latency: bool = True,
    seed: int = 7,
    latency_warmup: int = 1,
    latency_repeats: int = 3,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    """Collect structural layer records without claiming unavailable metrics."""

    sample = _default_sample(adapter, model, seed) if sample is None else sample
    forward = forward_fn or (lambda current_model, current_sample: _adapter_forward(adapter, current_model, current_sample))
    blocks = adapter.get_blocks(model)
    measured, timing = ({}, {"available": False, "reason": "disabled"})
    if measure_latency and sample is not None:
        measured, timing = _timed_layer_forward(model, sample, forward, blocks, warmup=latency_warmup, repeats=latency_repeats)
    output_shape = None
    if measure_latency and sample is not None:
        try:
            value = forward(model, sample)
            if value is not None and hasattr(value, "shape"):
                output_shape = list(value.shape)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            output_shape = None
    records = []
    for block in blocks:
        latency_ms: Optional[float] = None
        value = measured.get(str(block.block_id))
        if isinstance(value, (int, float)):
            latency_ms = float(value)
        latency_available = latency_ms is not None
        records.append({
            "block_id": block.block_id,
            "index": block.index,
            "module_path": block.module_path,
            "parameter_count": block.parameter_count,
            "estimated_flops": _estimate_flops(model, block.module_path),
            "latency_ms": latency_ms,
            "memory_delta_bytes": None,
            "measurement": {
                "latency_available": latency_available,
                "memory_available": False,
                "flops_available": False,
                "latency_method": timing.get("method"),
            },
            "output_shape": output_shape,
        })
    return tuple(records), timing


def profile_layers(adapter: WanModelAdapter, model: Any, *, sample: Any = None, forward_fn: Optional[ForwardFn] = None, measure_latency: bool = True, seed: int = 7, latency_warmup: int = 1, latency_repeats: int = 3) -> tuple[Mapping[str, Any], ...]:
    records, _ = _profile_layers(adapter, model, sample=sample, forward_fn=forward_fn, measure_latency=measure_latency, seed=seed, latency_warmup=latency_warmup, latency_repeats=latency_repeats)
    return records


def profile_model(
    adapter: WanModelAdapter,
    model: Any,
    *,
    seed: int = 7,
    measure_latency: bool = True,
    forward_fn: Optional[ForwardFn] = None,
    timestep_fn: Optional[Callable[[Any, str, Any], Any]] = None,
    max_channel_probes: int = 32,
    latency_warmup: int = 1,
    latency_repeats: int = 3,
    calibration: Optional[CalibrationCorpus] = None,
    advanced: bool = False,
    channel_method: str = "weight",
    channel_chunk_size: Optional[int] = None,
    head_method: str = "output_ablation",
) -> ProfileReport:
    structure = adapter.inspect(model)
    calibration_samples = calibration.samples() if calibration is not None else ()
    sample = calibration_samples[0].payload if calibration_samples else _default_sample(adapter, model, seed)
    effective_forward = forward_fn or (lambda current_model, current_sample: _adapter_forward(adapter, current_model, current_sample))
    layers, timing = _profile_layers(adapter, model, sample=sample, forward_fn=effective_forward, measure_latency=measure_latency, seed=seed, latency_warmup=latency_warmup, latency_repeats=latency_repeats)
    block_sensitivity = measure_block_sensitivity(adapter, model, sample=sample, forward_fn=effective_forward)
    channel_samples = tuple(item.payload for item in calibration_samples) or (sample,)
    channel_sensitivity = measure_channel_sensitivity(
        adapter,
        model,
        max_probes=max_channel_probes,
        method=channel_method,
        chunk_size=channel_chunk_size,
        sample_set=channel_samples,
        full=bool(advanced),
        forward_fn=effective_forward,
    )
    default_timestep = getattr(adapter, "forward_timestep_profile", None)
    if callable(default_timestep):
        timestep_forward = default_timestep
    else:
        timestep_forward = timestep_fn or _default_timestep_forward
    timestep_sensitivity = measure_timestep_sensitivity(adapter, model, sample=sample, forward_fn=timestep_forward)
    head_sensitivity = measure_head_sensitivity(
        adapter,
        model,
        sample=sample,
        method=head_method,
        model_forward_fn=effective_forward,
    )
    attention_sensitivity = measure_attention_sensitivity(
        adapter,
        model,
        sample=sample,
        method=head_method,
        model_forward_fn=effective_forward,
    )
    token_sensitivity = measure_token_sensitivity(adapter, model, sample=sample)
    sampling_step_sensitivity = measure_sampling_step_sensitivity(adapter, model)
    offline = bool(structure.metadata.get("offline_simulation", False))
    device = _runtime_device(model, sample)
    first = _first_tensor(sample)
    relationships = {
        str(item["block_id"]): {
            "layer": str(item["block_id"]),
            "block_sensitivity": str(item["block_id"]),
            "channel_sensitivity": str(item["block_id"]),
            "timestep_buckets": ["high_noise", "middle", "low_noise"],
        }
        for item in layers
    }
    calibration_payload = calibration.to_dict() if calibration is not None else {}
    evidence_status = "offline_simulation" if offline else "measured_or_partial"
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
            "memory_available": bool(timing.get("memory_available", False)),
            "flops_available": any(item["estimated_flops"] is not None for item in layers),
            "channel_probe_limit": int(max_channel_probes),
            "model_sha256": structure.checkpoint_sha256,
            "device": str(device) if device is not None else "unknown",
            "input_shape": list(first.shape) if first is not None and hasattr(first, "shape") else None,
            "precision": str(getattr(first, "dtype", getattr(model, "dtype", "unknown"))),
            "seed": int(seed),
            "latency_method": timing.get("method"),
            "latency_warmup": int(latency_warmup),
            "latency_repeats": int(latency_repeats),
            "peak_memory_bytes": timing.get("peak_memory_bytes"),
            "measurement_status": "measured" if timing.get("available") else "partial_or_unavailable",
        },
        head_sensitivity=head_sensitivity,
        attention_sensitivity=attention_sensitivity,
        token_sensitivity=token_sensitivity,
        sampling_step_sensitivity=sampling_step_sensitivity,
        hardware={
            "parameter_count": int(structure.parameter_count),
            "estimated_flops": sum(int(item["estimated_flops"]) for item in layers if item["estimated_flops"] is not None),
            "peak_memory_bytes": timing.get("peak_memory_bytes"),
            "memory_available": bool(timing.get("memory_available", False)),
        },
        calibration=calibration_payload,
        relationships=relationships,
        evidence={
            "status": evidence_status,
            "advanced": bool(advanced),
            "latency": {
                "status": "measured" if timing.get("available") else "unavailable",
                "method": timing.get("method"),
                "reason": timing.get("reason"),
            },
            "memory": {
                "status": "measured" if timing.get("memory_available") else "unavailable",
                "reason": None if timing.get("memory_available") else "runtime_telemetry_unavailable",
            },
        },
    )


def _default_timestep_forward(model: Any, bucket: str, sample: Any) -> Any:
    """Use scaled fake inputs for timestep probes; real Wan layouts may fail closed."""

    scale = {"high_noise": 0.9, "middle": 0.5, "low_noise": 0.1}[bucket]
    return _default_forward(model, sample * scale)
