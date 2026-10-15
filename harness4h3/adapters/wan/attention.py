"""Small, version-tolerant helpers for Wan attention surgery.

The Diffusers Wan implementation exposes the same projection layout as the
CPU fake used by Harness4H3: Q/K/V projections, a head count, and an output
projection.  Keeping the tensor surgery here lets both adapters execute the
same head-level algorithm while the adapter remains responsible for locating
the actual modules.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
        raise RuntimeError("Wan attention surgery requires torch") from exc
    return torch, nn


def _linear_with_rows(module: Any, rows: Sequence[int]) -> Any:
    torch, nn = _torch_modules()
    if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
        raise ValueError("attention projection is not a Linear module")
    result = nn.Linear(
        int(module.in_features),
        len(rows),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    indices = torch.tensor(tuple(int(item) for item in rows), device=module.weight.device, dtype=torch.long)
    with torch.no_grad():
        result.weight.copy_(module.weight.index_select(0, indices))
        if result.bias is not None:
            result.bias.copy_(module.bias.index_select(0, indices))
    return result


def _linear_with_columns(module: Any, columns: Sequence[int]) -> Any:
    torch, nn = _torch_modules()
    if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
        raise ValueError("attention output projection is not a Linear module")
    result = nn.Linear(
        len(columns),
        int(module.out_features),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    indices = torch.tensor(tuple(int(item) for item in columns), device=module.weight.device, dtype=torch.long)
    with torch.no_grad():
        result.weight.copy_(module.weight.index_select(1, indices))
        if result.bias is not None:
            result.bias.copy_(module.bias)
    return result


def _resize_norm(module: Any, indices: Sequence[int]) -> Any:
    """Resize RMSNorm/LayerNorm in place without depending on a class name."""

    torch, _ = _torch_modules()
    if not hasattr(module, "weight") or module.weight is None:
        return module
    index = torch.tensor(tuple(int(item) for item in indices), device=module.weight.device, dtype=torch.long)
    with torch.no_grad():
        module.weight = torch.nn.Parameter(module.weight.index_select(0, index).detach().clone())
        if getattr(module, "bias", None) is not None:
            module.bias = torch.nn.Parameter(module.bias.index_select(0, index).detach().clone())
    if hasattr(module, "normalized_shape"):
        module.normalized_shape = (len(indices),)
    return module


def _head_indices(heads: int, head_dim: int, removed_heads: Iterable[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = int(heads)
    width = int(head_dim)
    removed = tuple(sorted({int(item) for item in removed_heads}))
    if count < 2:
        raise ValueError("attention must have at least two heads before pruning")
    if not removed or any(item < 0 or item >= count for item in removed):
        raise ValueError("invalid_attention_head_selection")
    retained = tuple(item for item in range(count) if item not in set(removed))
    if not retained:
        raise ValueError("invalid_attention_head_selection: cannot remove every head")
    rows = tuple(head * width + offset for head in retained for offset in range(width))
    return retained, rows


def prune_attention_module_heads(attention: Any, removed_heads: Sequence[int]) -> dict[str, Any]:
    """Remove heads from a Wan-style attention module in place.

    The function intentionally fails closed when a future Diffusers layout
    does not expose the paired projections.  A silent mask would make a
    structural search claim parameter reduction without changing execution.
    """

    # WanAttention can be switched to a fused QKV projection for inference.
    # Structural head surgery must operate on the projections used by forward;
    # unfuse first instead of mutating dormant to_q/to_k/to_v modules while
    # leaving to_qkv untouched.
    if bool(getattr(attention, "fused_projections", False)):
        unfuse = getattr(attention, "unfuse_projections", None)
        if not callable(unfuse):
            raise ValueError("unsupported_attention_layout: fused projections cannot be unfused")
        unfuse()
        if bool(getattr(attention, "fused_projections", False)):
            raise ValueError("unsupported_attention_layout: fused projections remained active")
    heads = int(getattr(attention, "heads", 0))
    inner_dim = int(getattr(attention, "inner_dim", 0))
    if heads <= 0 or inner_dim <= 0 or inner_dim % heads:
        raise ValueError("unsupported_attention_layout: head geometry unavailable")
    head_dim = inner_dim // heads
    retained_heads, rows = _head_indices(heads, head_dim, removed_heads)
    for name in ("to_q", "to_k", "to_v"):
        module = getattr(attention, name, None)
        if module is None:
            raise ValueError("unsupported_attention_layout: missing %s" % name)
        if int(module.out_features) != inner_dim:
            # Wan cross-attention uses the same head geometry in current
            # releases.  Refuse a layout where that invariant is false.
            raise ValueError("unsupported_attention_layout: %s width is not head-aligned" % name)
        setattr(attention, name, _linear_with_rows(module, rows))

    output = getattr(attention, "to_out", None)
    if output is None or not hasattr(output, "__getitem__"):
        raise ValueError("unsupported_attention_layout: missing to_out")
    output[0] = _linear_with_columns(output[0], rows)
    for name in ("norm_q", "norm_k"):
        module = getattr(attention, name, None)
        if module is not None:
            setattr(attention, name, _resize_norm(module, rows))
    for name in ("add_k_proj", "add_v_proj"):
        module = getattr(attention, name, None)
        if module is not None:
            setattr(attention, name, _linear_with_rows(module, rows))
    attention.heads = len(retained_heads)
    attention.inner_dim = len(rows)
    if hasattr(attention, "kv_inner_dim"):
        attention.kv_inner_dim = len(rows)
    if hasattr(attention, "cross_attention_dim_head") and getattr(attention, "cross_attention_dim_head", None) is not None:
        attention.cross_attention_dim_head = head_dim
    return {
        "original_heads": heads,
        "removed_heads": list(sorted(int(item) for item in removed_heads)),
        "retained_heads": list(retained_heads),
        "head_dim": head_dim,
    }


def clone_head_surgery(attention: Any, removed_heads: Sequence[int]) -> tuple[Any, dict[str, Any]]:
    """Copy a module before probing a head-level sensitivity delta."""

    candidate = copy.deepcopy(attention)
    return candidate, prune_attention_module_heads(candidate, removed_heads)
