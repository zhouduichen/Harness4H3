from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

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
            probe = getattr(adapter, "probe_block_removal", None)
            if callable(probe):
                callback = forward_fn or (lambda current_model, current_sample: _forward(current_model, current_sample, None))
                removed = probe(model, block.block_id, sample, callback)
            else:
                candidate = copy.deepcopy(model)
                adapter.remove_blocks(candidate, (block.block_id,))
                removed = _forward(candidate, sample, forward_fn)
            distance = _distance(base, removed)
            records.append({"block_id": block.block_id, "importance_score": distance, "available": distance is not None, "method": "leave_one_block_out", "input_fixed": sample is not None, "score_semantics": "mean_absolute_output_difference"})
        except (RuntimeError, TypeError, ValueError, AttributeError):
            records.append({"block_id": block.block_id, "importance_score": None, "available": False, "method": "leave_one_block_out", "input_fixed": sample is not None, "score_semantics": "mean_absolute_output_difference"})
    return tuple(records)


def _resolve(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part) if not part.isdigit() else value[int(part)]
    return value


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


class ChannelImportanceMethod(Protocol):
    name: str

    def measure(
        self,
        adapter: WanModelAdapter,
        model: Any,
        block: Any,
        channel_indices: Sequence[int],
        samples: Sequence[Any],
    ) -> tuple[Mapping[str, Any], ...]:
        ...


def _offline_simulation(adapter: Any) -> bool:
    try:
        return bool(adapter.inspect(adapter.load_model("fake://baseline")).metadata.get("offline_simulation", False))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _evidence(status: str, method: str, offline_simulation: bool, reason: Optional[str] = None) -> dict[str, Any]:
    return {
        "status": status,
        "method": method,
        "offline_simulation": bool(offline_simulation),
        "reason": reason,
    }


class WeightContributionMethod:
    name = "weight_proxy"

    def measure(
        self,
        adapter: WanModelAdapter,
        model: Any,
        block: Any,
        channel_indices: Sequence[int],
        samples: Sequence[Any],
    ) -> tuple[Mapping[str, Any], ...]:
        del samples
        module = _resolve(model, str(block.module_path))
        up, down = _paired_ffn(module)
        offline = bool(adapter.inspect(model).metadata.get("offline_simulation", False))
        return tuple(
            {
                "channel": int(channel),
                "importance_score": float(
                    up.weight[channel].detach().float().abs().mean().item()
                    + down.weight[:, channel].detach().float().abs().mean().item()
                ),
                "available": True,
                "evidence": _evidence("estimated", self.name, offline),
            }
            for channel in channel_indices
        )


class UnavailableImportanceMethod:
    def __init__(self, name: str, reason: str) -> None:
        self.name = str(name)
        self.reason = str(reason)

    def measure(
        self,
        adapter: WanModelAdapter,
        model: Any,
        block: Any,
        channel_indices: Sequence[int],
        samples: Sequence[Any],
    ) -> tuple[Mapping[str, Any], ...]:
        del model, block, samples
        offline = bool(getattr(adapter, "architecture_name", "").lower().startswith("wanfake"))
        return tuple(
            {
                "channel": int(channel),
                "importance_score": None,
                "available": False,
                "evidence": _evidence("unavailable", self.name, offline, self.reason),
            }
            for channel in channel_indices
        )


def _channel_method(method: str) -> ChannelImportanceMethod:
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized in {"weight", "weight_proxy", "paired_ffn_weight_contribution"}:
        return WeightContributionMethod()
    if normalized in {"activation", "activation_importance"}:
        return UnavailableImportanceMethod("activation", "activation_hook_not_declared_by_adapter")
    if normalized in {"output_perturbation", "perturbation"}:
        return UnavailableImportanceMethod("output_perturbation", "output_probe_not_declared_by_adapter")
    raise ValueError("unsupported_channel_importance_method: %s" % method)


def measure_channel_sensitivity(
    adapter: WanModelAdapter,
    model: Any,
    *,
    max_probes: int = 32,
    method: str = "weight",
    chunk_size: Optional[int] = None,
    sample_set: Sequence[Any] = (),
    full: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if int(max_probes) <= 0:
        raise ValueError("max_probes must be positive")
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    importance_method = _channel_method(method)
    effective_chunk = max(1, int(chunk_size or max_probes))
    records = []
    for block in adapter.get_blocks(model):
        try:
            module = _resolve(model, block.module_path)
            up, down = _paired_ffn(module)
            width = int(up.out_features)
            limit = width if full else min(width, max(1, int(max_probes)))
            values: list[Mapping[str, Any]] = []
            for start in range(0, limit, effective_chunk):
                indices = tuple(range(start, min(start + effective_chunk, limit)))
                values.extend(importance_method.measure(adapter, model, block, indices, sample_set))
            available = bool(values) and all(item.get("available") is True for item in values)
            evidence_status = "estimated" if importance_method.name == "weight_proxy" else "unavailable"
            records.append({
                "block_id": block.block_id,
                "width": width,
                "channels": values,
                "method": importance_method.name,
                "available": available,
                "approximation": importance_method.name == "weight_proxy",
                "full": bool(full),
                "chunk_size": effective_chunk,
                "evidence": _evidence(evidence_status, importance_method.name, bool(adapter.inspect(model).metadata.get("offline_simulation", False))),
                "score_semantics": "paired_weight_absolute_mean_not_activation_perturbation" if importance_method.name == "weight_proxy" else "unavailable",
            })
        except (AttributeError, IndexError, TypeError, RuntimeError, ValueError) as exc:
            records.append({
                "block_id": block.block_id,
                "width": None,
                "channels": [],
                "method": importance_method.name,
                "available": False,
                "approximation": importance_method.name == "weight_proxy",
                "full": bool(full),
                "chunk_size": effective_chunk,
                "evidence": _evidence("unavailable", importance_method.name, False, str(exc)),
                "score_semantics": "unavailable",
            })
    return tuple(records)


def measure_timestep_sensitivity(
    adapter: Any,
    model: Any = None,
    *,
    sample: Any = None,
    forward_fn: Optional[Callable[[Any, str, Any], Any]] = None,
) -> dict[str, Mapping[str, Any]]:
    # Keep the old positional form usable for callers outside the profiler.
    if model is None:
        model = adapter
        adapter = None
    buckets = (("high_noise", 0.9), ("middle", 0.5), ("low_noise", 0.1))
    records: dict[str, Mapping[str, Any]] = {}
    blocks = tuple(adapter.get_blocks(model)) if adapter is not None else ()
    reference_block = blocks[0].block_id if blocks else None
    for name, value in buckets:
        if forward_fn is None or sample is None or adapter is None or reference_block is None:
            records[name] = {"timestep": value, "importance_score": None, "available": False, "method": "structural_timestep_delta", "reference_block_id": reference_block, "score_semantics": "removed_block_output_difference"}
            continue
        try:
            base = forward_fn(model, name, sample)
            probe = getattr(adapter, "probe_block_removal", None)
            callback = lambda current_model, current_sample: forward_fn(current_model, name, current_sample)
            if callable(probe):
                modified = probe(model, reference_block, sample, callback)
            else:
                candidate = copy.deepcopy(model)
                adapter.remove_blocks(candidate, (reference_block,))
                modified = callback(candidate, sample)
            score = _distance(base, modified)
            records[name] = {"timestep": value, "importance_score": score, "available": score is not None, "method": "structural_timestep_delta", "reference_block_id": reference_block, "score_semantics": "removed_block_output_difference"}
        except (RuntimeError, TypeError, ValueError, AttributeError):
            records[name] = {"timestep": value, "importance_score": None, "available": False, "method": "structural_timestep_delta", "reference_block_id": reference_block, "score_semantics": "removed_block_output_difference"}
    return records


def _attention_module(block: Any, name: str) -> Any:
    value = getattr(block, str(name), None)
    if value is None:
        raise AttributeError("attention module unavailable: %s" % name)
    return value


def measure_head_sensitivity(
    adapter: WanModelAdapter,
    model: Any,
    *,
    attention_name: str = "attn1",
) -> tuple[Mapping[str, Any], ...]:
    """Rank heads from their paired Q/K/V contribution, without fake zeros.

    A real Wan checkpoint cannot be deep-copied safely at 1.3B scale.  The
    projection contribution is therefore a measured structural proxy and is
    labelled as such; the execution operator later validates the selected
    heads by real reload and evaluation.
    """

    offline = bool(adapter.inspect(model).metadata.get("offline_simulation", False))
    records: list[Mapping[str, Any]] = []
    for block in adapter.get_blocks(model):
        try:
            module = _resolve(model, block.module_path)
            attention = _attention_module(module, attention_name)
            heads = int(getattr(attention, "heads"))
            inner = int(getattr(attention, "inner_dim"))
            if heads <= 0 or inner % heads:
                raise ValueError("head geometry unavailable")
            width = inner // heads
            q = attention.to_q.weight.detach().float()
            k = attention.to_k.weight.detach().float()
            v = attention.to_v.weight.detach().float()
            for head in range(heads):
                rows = slice(head * width, (head + 1) * width)
                score = float(q[rows].abs().mean().item() + k[rows].abs().mean().item() + v[rows].abs().mean().item())
                records.append({
                    "block_id": block.block_id,
                    "attention": attention_name,
                    "head": head,
                    "importance_score": score,
                    "available": True,
                    "method": "qkv_projection_contribution",
                    "approximation": True,
                    "evidence": _evidence("estimated", "qkv_projection_contribution", offline),
                    "score_semantics": "mean_absolute_qkv_projection_weight",
                })
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            records.append({
                "block_id": block.block_id,
                "attention": attention_name,
                "head": None,
                "importance_score": None,
                "available": False,
                "method": "qkv_projection_contribution",
                "approximation": True,
                "evidence": _evidence("unavailable", "qkv_projection_contribution", offline, str(exc)),
                "score_semantics": "unavailable",
            })
    return tuple(records)


def measure_attention_sensitivity(adapter: WanModelAdapter, model: Any, *, attention_names: Sequence[str] = ("attn1", "attn2")) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for name in attention_names:
        rows = measure_head_sensitivity(adapter, model, attention_name=str(name))
        records.extend(rows)
    return tuple(records)


def measure_token_sensitivity(
    adapter: WanModelAdapter,
    model: Any,
    *,
    sample: Any = None,
    keep_ratios: Sequence[float] = (0.75, 0.5, 0.25),
) -> tuple[Mapping[str, Any], ...]:
    """Measure token energy concentration for dynamic token selection."""

    offline = bool(adapter.inspect(model).metadata.get("offline_simulation", False))
    token_sample = sample
    layout = "sequence"
    if isinstance(sample, Mapping):
        token_sample = sample.get("hidden_states")
        layout = "wan_latent_grid"
    try:
        # Diffusers Wan profiles begin as [B, C, T, H, W].  Flatten the
        # actual spatiotemporal patch grid into [B, N, C] for importance
        # estimation; this is only a profiler view and does not alter the
        # model or claim that the external latent itself is tokenized.
        if token_sample is not None and int(token_sample.ndim) == 5:
            token_sample = token_sample.permute(0, 2, 3, 4, 1).reshape(token_sample.shape[0], -1, token_sample.shape[1])
        elif token_sample is not None and int(token_sample.ndim) == 4:
            token_sample = token_sample.permute(0, 2, 3, 1).reshape(token_sample.shape[0], -1, token_sample.shape[1])
    except (AttributeError, RuntimeError, TypeError, ValueError):
        token_sample = None
    if token_sample is None or not hasattr(token_sample, "ndim") or int(token_sample.ndim) < 3:
        return tuple({"keep_ratio": float(ratio), "available": False, "method": "token_energy_concentration", "evidence": _evidence("unavailable", "token_energy_concentration", offline, "profile sample has no sequence dimension"), "score_semantics": "unavailable"} for ratio in keep_ratios)
    try:
        scores = token_sample.detach().float().pow(2).mean(dim=tuple(index for index in range(token_sample.ndim) if index != 1))
        ordered = scores.sort(dim=1, descending=True).values
        total = ordered.sum(dim=1).clamp_min(1e-12)
        return tuple({
            "keep_ratio": float(ratio),
            "token_count": int(token_sample.shape[1]),
            "selected_count": max(1, int(round(int(token_sample.shape[1]) * float(ratio)))),
            "importance_score": float((ordered[:, :max(1, int(round(int(token_sample.shape[1]) * float(ratio))))].sum(dim=1) / total).mean().item()),
            "available": True,
            "method": "token_energy_concentration",
            "approximation": True,
            "layout": layout,
            "evidence": _evidence("estimated", "token_energy_concentration", offline),
            "score_semantics": "fraction_of_input_energy_retained_by_top_tokens",
        } for ratio in keep_ratios)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return tuple({"keep_ratio": float(ratio), "available": False, "method": "token_energy_concentration", "evidence": _evidence("unavailable", "token_energy_concentration", offline, str(exc)), "score_semantics": "unavailable"} for ratio in keep_ratios)


def measure_sampling_step_sensitivity(
    adapter: WanModelAdapter,
    model: Any,
    *,
    baseline_steps: int = 16,
    candidate_steps: Sequence[int] = (12, 8, 4),
) -> Mapping[str, Any]:
    """Expose the adapter's actual schedule selector as measured capability."""

    selector = getattr(adapter, "sampling_steps", None)
    if not callable(selector):
        return {"available": False, "method": "adapter_sampling_schedule", "reason": "adapter_sampling_steps_unavailable"}
    timesteps = tuple(range(int(baseline_steps), 0, -1))
    records = []
    for count in candidate_steps:
        try:
            selected = selector(model, tuple(timesteps[: int(count)]))
            records.append({"target_steps": int(count), "actual_steps": len(selected), "available": True})
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            records.append({"target_steps": int(count), "actual_steps": None, "available": False, "reason": str(exc)})
    return {"available": all(item["available"] for item in records), "method": "adapter_sampling_schedule", "baseline_steps": int(baseline_steps), "candidates": records, "evidence": _evidence("measured", "adapter_sampling_schedule", bool(adapter.inspect(model).metadata.get("offline_simulation", False)))}
