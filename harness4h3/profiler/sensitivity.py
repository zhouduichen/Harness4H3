from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ..adapters.wan.contracts import ChannelPruningConfig, WanModelAdapter
from ..adapters.wan.attention import clone_head_surgery


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


def _profile_forward(adapter: Any, model: Any, sample: Any, forward_fn: Optional[Callable[[Any, Any], Any]]) -> Any:
    if forward_fn is not None:
        return forward_fn(model, sample)
    adapter_forward = getattr(adapter, "forward_profile", None)
    if callable(adapter_forward):
        return adapter_forward(model, sample)
    return _forward(model, sample, None)


class OutputPerturbationMethod:
    """Measure FFN channel leave-out deltas through the real model forward."""

    name = "output_perturbation"

    def __init__(self, adapter: Any, forward_fn: Optional[Callable[[Any, Any], Any]] = None) -> None:
        self.adapter = adapter
        self.forward_fn = forward_fn

    @staticmethod
    def _zero_channel(channel: int) -> Callable[[Any, Any, Any], Any]:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            try:
                import torch

                if not isinstance(output, torch.Tensor) or output.ndim < 1:
                    raise TypeError("ffn_output_is_not_tensor")
                if int(output.shape[-1]) <= int(channel):
                    raise ValueError("ffn_output_channel_out_of_range")
                changed = output.clone()
                changed[..., int(channel)] = 0
                return changed
            except ImportError as exc:  # pragma: no cover - torch is an adapter dependency
                raise RuntimeError("channel output perturbation requires torch") from exc

        return hook

    def measure(
        self,
        adapter: WanModelAdapter,
        model: Any,
        block: Any,
        channel_indices: Sequence[int],
        samples: Sequence[Any],
    ) -> tuple[Mapping[str, Any], ...]:
        del adapter
        module = _resolve(model, str(block.module_path))
        up, _down = _paired_ffn(module)
        offline = bool(self.adapter.inspect(model).metadata.get("offline_simulation", False))
        if not samples:
            return tuple({
                "channel": int(channel),
                "importance_score": None,
                "available": False,
                "method": self.name,
                "approximation": False,
                "sample_count": 0,
                "evidence": _evidence("unavailable", self.name, offline, "sample_set_unavailable"),
                "score_semantics": "unavailable",
            } for channel in channel_indices)

        records: list[Mapping[str, Any]] = []
        for channel in channel_indices:
            scores: list[float] = []
            for sample in samples:
                try:
                    base = _profile_forward(self.adapter, model, sample, self.forward_fn)
                    if base is None:
                        continue
                    register = getattr(up, "register_forward_hook", None)
                    if not callable(register):
                        raise AttributeError("ffn_output_hook_unavailable")
                    handle = register(self._zero_channel(int(channel)))
                    try:
                        modified = _profile_forward(self.adapter, model, sample, self.forward_fn)
                    finally:
                        handle.remove()
                    score = _distance(base, modified)
                    if score is not None:
                        scores.append(float(score))
                except (RuntimeError, TypeError, ValueError, AttributeError):
                    continue
            if scores:
                records.append({
                    "channel": int(channel),
                    "importance_score": float(sum(scores) / len(scores)),
                    "available": True,
                    "method": self.name,
                    "approximation": False,
                    "sample_count": len(scores),
                    "evidence": _evidence("estimated" if offline else "measured", self.name, offline),
                    "score_semantics": "mean_absolute_model_output_difference_after_zeroing_ffn_channel",
                })
            else:
                records.append({
                    "channel": int(channel),
                    "importance_score": None,
                    "available": False,
                    "method": self.name,
                    "approximation": False,
                    "sample_count": 0,
                    "evidence": _evidence("unavailable", self.name, offline, "forward_or_output_probe_unavailable"),
                    "score_semantics": "unavailable",
                })
        return tuple(records)


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


def _channel_method(
    method: str,
    *,
    adapter: Optional[WanModelAdapter] = None,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> ChannelImportanceMethod:
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized in {"weight", "weight_proxy", "paired_ffn_weight_contribution"}:
        return WeightContributionMethod()
    if normalized in {"output_perturbation", "output_probe", "perturbation"}:
        if adapter is None:
            return UnavailableImportanceMethod("output_perturbation", "adapter_required_for_output_probe")
        return OutputPerturbationMethod(adapter, forward_fn)
    if normalized in {"activation", "activation_importance"}:
        return UnavailableImportanceMethod("activation", "activation_hook_not_declared_by_adapter")
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
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> tuple[Mapping[str, Any], ...]:
    if int(max_probes) <= 0:
        raise ValueError("max_probes must be positive")
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    importance_method = _channel_method(method, adapter=adapter, forward_fn=forward_fn)
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
            if importance_method.name == "output_perturbation":
                evidence_status = "estimated" if bool(adapter.inspect(model).metadata.get("offline_simulation", False)) else "measured"
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
                "score_semantics": "paired_weight_absolute_mean_not_activation_perturbation" if importance_method.name == "weight_proxy" else ("mean_absolute_model_output_difference_after_zeroing_ffn_channel" if importance_method.name == "output_perturbation" else "unavailable"),
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


def measure_block_timestep_sensitivity(
    adapter: WanModelAdapter,
    model: Any,
    *,
    sample: Any,
    forward_fn: Callable[[Any, str, Any], Any],
) -> tuple[Mapping[str, Any], ...]:
    """Measure each Block's output effect at matched denoising timesteps."""

    buckets = (("high_noise", 0.9), ("middle", 0.5), ("low_noise", 0.1))
    rows: list[Mapping[str, Any]] = []
    probe = getattr(adapter, "probe_block_removal", None)
    for block in adapter.get_blocks(model):
        scores: dict[str, Optional[float]] = {}
        for name, _ in buckets:
            try:
                base = forward_fn(model, name, sample)
                callback = lambda current_model, current_sample, bucket=name: forward_fn(current_model, bucket, current_sample)
                if callable(probe):
                    modified = probe(model, block.block_id, sample, callback)
                else:
                    candidate = copy.deepcopy(model)
                    adapter.remove_blocks(candidate, (block.block_id,))
                    modified = callback(candidate, sample)
                scores[name] = _distance(base, modified)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                scores[name] = None
        available = [value for value in scores.values() if value is not None]
        rows.append(
            {
                "block_id": str(block.block_id),
                "per_timestep": scores,
                "importance_score": sum(available) / len(available) if available else None,
                "best_bucket": min(
                    (name for name, value in scores.items() if value is not None),
                    key=lambda name: scores[name],
                    default=None,
                ),
                "worst_bucket": max(
                    (name for name, value in scores.items() if value is not None),
                    key=lambda name: scores[name],
                    default=None,
                ),
                "available": bool(available),
                "method": "leave_one_block_out_per_timestep",
                "score_semantics": "mean_absolute_output_difference",
            }
        )
    return tuple(rows)


def _attention_module(block: Any, name: str) -> Any:
    value = getattr(block, str(name), None)
    if value is None:
        raise AttributeError("attention module unavailable: %s" % name)
    return value


def _sequence_probe(sample: Any) -> Any:
    """Return an attention-shaped tensor, without treating raw Wan latents as one.

    Diffusers Wan profiling samples are mappings containing a 5-D latent.  That
    tensor is valid for the transformer, but it is not a valid direct input to
    an individual attention module.  Only a 2-D/3-D hidden-state tensor is
    accepted here; callers can therefore distinguish a measured attention
    ablation from a structural proxy instead of silently flattening latents.
    """

    candidate = sample
    if isinstance(candidate, Mapping):
        for key in ("hidden_states", "sample", "input", "x"):
            value = candidate.get(key)
            if value is not None:
                candidate = value
                break
    rank = getattr(candidate, "ndim", None)
    if rank not in {2, 3}:
        return None
    return candidate


def _call_attention_probe(attention: Any, sample: Any, forward_fn: Optional[Callable[[Any, Any], Any]]) -> Any:
    if sample is None or not callable(attention):
        return None
    try:
        import torch

        with torch.no_grad():
            if forward_fn is not None:
                return forward_fn(attention, sample)
            return attention(sample)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None


def _call_model_probe(model: Any, sample: Any, forward_fn: Optional[Callable[[Any, Any], Any]]) -> Any:
    if sample is None:
        return None
    try:
        import torch

        with torch.no_grad():
            if forward_fn is not None:
                return forward_fn(model, sample)
            return _forward(model, sample, None)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None


def _attention_output_ablation(
    attention: Any,
    sample: Any,
    heads: int,
    *,
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> tuple[Mapping[str, Any], ...]:
    """Measure one-head leave-out deltas using a copied executable module."""

    was_training = bool(getattr(attention, "training", False))
    evaluation = getattr(attention, "eval", None)
    if callable(evaluation):
        evaluation()
    try:
        base = _call_attention_probe(attention, sample, forward_fn)
        if base is None:
            raise ValueError("attention_forward_unavailable")
        values: list[Mapping[str, Any]] = []
        for head in range(int(heads)):
            candidate, surgery = clone_head_surgery(attention, (head,))
            candidate_eval = getattr(candidate, "eval", None)
            if callable(candidate_eval):
                candidate_eval()
            modified = _call_attention_probe(candidate, sample, forward_fn)
            score = _distance(base, modified)
            if score is None:
                raise ValueError("attention_output_shape_unavailable")
            values.append({
                "head": int(head),
                "importance_score": score,
                "available": True,
                "method": "attention_output_ablation",
                "approximation": False,
                "surgery": surgery,
                "score_semantics": "mean_absolute_attention_output_difference",
            })
        return tuple(values)
    finally:
        if was_training:
            train = getattr(attention, "train", None)
            if callable(train):
                train()


def _captured_attention_output_ablation(
    model: Any,
    attention: Any,
    sample: Any,
    heads: int,
    *,
    model_forward_fn: Callable[[Any, Any], Any],
) -> tuple[Mapping[str, Any], ...]:
    """Run one model forward, then probe copied heads on captured real inputs.

    This keeps raw Wan latent profiling practical: a 1.3B model is executed
    once to capture each Attention's actual hidden/context/RoPE arguments, and
    only the small Attention module is copied for each head probe.
    """

    captured: dict[str, Any] = {}

    def pre_with_kwargs(_module: Any, args: Any, kwargs: Any) -> None:
        captured["args"] = tuple(args)
        captured["kwargs"] = dict(kwargs)

    def pre_legacy(_module: Any, args: Any) -> None:
        captured["args"] = tuple(args)
        captured["kwargs"] = {}

    def post(_module: Any, _inputs: Any, output: Any) -> None:
        captured["output"] = output

    register_pre = getattr(attention, "register_forward_pre_hook", None)
    register_post = getattr(attention, "register_forward_hook", None)
    if not callable(register_pre) or not callable(register_post):
        raise AttributeError("attention_forward_hooks_unavailable")
    try:
        try:
            pre_handle = register_pre(pre_with_kwargs, with_kwargs=True)
        except TypeError:
            pre_handle = register_pre(pre_legacy)
        post_handle = register_post(post)
        try:
            _call_model_probe(model, sample, model_forward_fn)
        finally:
            pre_handle.remove()
            post_handle.remove()
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("attention_input_capture_failed:%s" % type(exc).__name__) from exc
    if "args" not in captured or "output" not in captured:
        raise ValueError("attention_input_capture_empty")

    base = captured["output"]
    args = tuple(captured["args"])
    kwargs = dict(captured.get("kwargs", {}))
    values: list[Mapping[str, Any]] = []
    for head in range(int(heads)):
        candidate, surgery = clone_head_surgery(attention, (head,))
        candidate_eval = getattr(candidate, "eval", None)
        if callable(candidate_eval):
            candidate_eval()
        try:
            import torch

            with torch.no_grad():
                modified = candidate(*args, **kwargs)
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("attention_candidate_forward_failed:%s" % type(exc).__name__) from exc
        score = _distance(base, modified)
        if score is None:
            raise ValueError("attention_output_shape_unavailable")
        values.append({
            "head": int(head),
            "importance_score": score,
            "available": True,
            "method": "attention_output_ablation",
            "approximation": False,
            "input_source": "captured_from_model_forward",
            "surgery": surgery,
            "score_semantics": "mean_absolute_attention_output_difference",
        })
    return tuple(values)


def _attention_qkv_weights(attention: Any) -> tuple[Any, Any, Any]:
    projections = tuple(getattr(attention, name, None) for name in ("to_q", "to_k", "to_v"))
    if all(item is not None and hasattr(item, "weight") for item in projections):
        return tuple(item.weight.detach().float() for item in projections)  # type: ignore[union-attr]
    fused = getattr(attention, "to_qkv", None)
    if fused is None or not hasattr(fused, "weight"):
        raise AttributeError("attention_qkv_projections_unavailable")
    rows = int(fused.weight.shape[0])
    if rows <= 0 or rows % 3:
        raise ValueError("fused_attention_projection_not_qkv_aligned")
    return tuple(fused.weight.detach().float().chunk(3, dim=0))  # type: ignore[return-value]


def measure_head_sensitivity(
    adapter: WanModelAdapter,
    model: Any,
    *,
    attention_name: str = "attn1",
    sample: Any = None,
    method: str = "output_ablation",
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    model_forward_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> tuple[Mapping[str, Any], ...]:
    """Rank heads from executable attention ablation when a sequence is available.

    The copied object is one attention module, not the 1.3B model.  Raw Wan
    latent profiling samples cannot be passed to an individual attention
    module, so those cases capture its real arguments during one full model
    forward and probe copied Attention modules with those arguments when
    ``model_forward_fn`` is provided; only unsupported layouts fall back to
    the paired Q/K/V projection proxy.
    """

    normalized_method = str(method).strip().lower().replace("-", "_")
    if normalized_method in {"output_ablation", "attention_output_ablation", "perturbation"}:
        normalized_method = "output_ablation"
    elif normalized_method in {"projection_proxy", "qkv_projection_contribution", "weight_proxy"}:
        normalized_method = "projection_proxy"
    else:
        raise ValueError("unsupported_head_importance_method: %s" % method)
    sequence = _sequence_probe(sample) if normalized_method == "output_ablation" else None
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
            fallback_reason = None
            if normalized_method == "output_ablation" and sequence is not None:
                try:
                    ablated = _attention_output_ablation(
                        attention,
                        sequence,
                        heads,
                        forward_fn=forward_fn,
                    )
                    records.extend({
                        "block_id": block.block_id,
                        "attention": attention_name,
                        "evidence": _evidence("estimated" if offline else "measured", "attention_output_ablation", offline),
                        "input_fixed": True,
                        **item,
                    } for item in ablated)
                    continue
                except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                    fallback_reason = "%s:%s" % (type(exc).__name__, str(exc))
            elif normalized_method == "output_ablation":
                if model_forward_fn is not None and sample is not None:
                    try:
                        ablated = _captured_attention_output_ablation(
                            model,
                            attention,
                            sample,
                            heads,
                            model_forward_fn=model_forward_fn,
                        )
                        records.extend({
                            "block_id": block.block_id,
                            "attention": attention_name,
                            "evidence": _evidence("estimated" if offline else "measured", "attention_output_ablation", offline),
                            "input_fixed": True,
                            **item,
                        } for item in ablated)
                        continue
                    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                        fallback_reason = "%s:%s" % (type(exc).__name__, str(exc))
                else:
                    fallback_reason = "sequence_sample_unavailable"
            q, k, v = _attention_qkv_weights(attention)
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
                    "fallback_reason": fallback_reason,
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


def measure_attention_sensitivity(
    adapter: WanModelAdapter,
    model: Any,
    *,
    attention_names: Sequence[str] = ("attn1", "attn2"),
    sample: Any = None,
    method: str = "output_ablation",
    forward_fn: Optional[Callable[[Any, Any], Any]] = None,
    model_forward_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for name in attention_names:
        rows = measure_head_sensitivity(
            adapter,
            model,
            attention_name=str(name),
            sample=sample,
            method=method,
            forward_fn=forward_fn,
            model_forward_fn=model_forward_fn,
        )
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
        # Keep the batch axis so each sample retains an ordered token score.
        # Reducing every dimension except the token axis collapses [B, N, C]
        # to [N] for real Wan probes, which makes the later batch-wise sort
        # fail.  Feature axes start at 2 for the normalized [B, N, ...] view.
        feature_axes = tuple(range(2, int(token_sample.ndim)))
        scores = token_sample.detach().float().pow(2).mean(dim=feature_axes)
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
