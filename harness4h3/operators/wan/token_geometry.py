"""Deterministic TokenMap helpers shared by Fake and Diffusers Wan paths."""

from __future__ import annotations

from typing import Any, Optional, Sequence


def _validated_grid(grid_shape: Optional[Sequence[int]]) -> tuple[int, int, int] | None:
    if grid_shape is None or tuple(grid_shape) == ():
        return None
    values = tuple(int(item) for item in grid_shape)
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError("token grid_shape must contain three positive integers")
    return values  # type: ignore[return-value]


def _group_shape(grid: tuple[int, int, int], target: int) -> tuple[int, int, int]:
    """Choose balanced grid counts whose product is as close as possible to target."""

    counts = [1, 1, 1]
    while True:
        choices: list[tuple[int, tuple[int, int, int], float]] = []
        for axis, dimension in enumerate(grid):
            if counts[axis] >= dimension:
                continue
            candidate = list(counts)
            candidate[axis] += 1
            product = candidate[0] * candidate[1] * candidate[2]
            if product > target:
                continue
            ratios = [float(grid[index]) / float(candidate[index]) for index in range(3)]
            spread = max(ratios) - min(ratios)
            choices.append((product, (candidate[0], candidate[1], candidate[2]), spread))
        if not choices:
            break
        _product, selected, _spread = max(choices, key=lambda item: (item[0], -item[2]))
        counts[:] = selected
    return counts[0], counts[1], counts[2]


def build_token_groups(
    sequence_length: int,
    keep_ratio: float,
    torch: Any,
    *,
    grid_shape: Optional[Sequence[int]] = None,
) -> tuple[Any, Any, tuple[int, int, int] | None]:
    """Build ``source -> destination`` groups and first-token anchors.

    With no grid this preserves the historical contiguous 1-D grouping.  With
    a grid, flattened tokens are interpreted as ``(t, y, x)`` in row-major
    order and are pooled into balanced 3-D cuboids.  The returned group shape
    records the number of cuboids along each axis; it is part of the runtime
    evidence so a caller cannot mistake this for an arbitrary 1-D merge.
    """

    length = int(sequence_length)
    if length <= 0:
        raise ValueError("token sequence length must be positive")
    ratio = float(keep_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("token keep ratio must be in (0, 1]")
    target = max(1, min(length, int(round(length * ratio))))
    grid = _validated_grid(grid_shape)
    if grid is None:
        group_count = target
        positions = torch.arange(length)
        source_groups = torch.div(positions * group_count, length, rounding_mode="floor")
        group_shape = None
    else:
        if grid[0] * grid[1] * grid[2] != length:
            raise ValueError("token grid_shape product must equal sequence length")
        group_shape = _group_shape(grid, target)
        time = torch.arange(length) // (grid[1] * grid[2])
        y = (torch.arange(length) // grid[2]) % grid[1]
        x = torch.arange(length) % grid[2]
        counts_t, counts_y, counts_x = group_shape
        group_t = torch.div(time * counts_t, grid[0], rounding_mode="floor")
        group_y = torch.div(y * counts_y, grid[1], rounding_mode="floor")
        group_x = torch.div(x * counts_x, grid[2], rounding_mode="floor")
        source_groups = (group_t * counts_y + group_y) * counts_x + group_x
        group_count = counts_t * counts_y * counts_x
    anchors = torch.empty(group_count, device=source_groups.device, dtype=torch.long)
    if hasattr(anchors, "scatter_reduce_"):
        anchors.fill_(length)
        anchors.scatter_reduce_(0, source_groups, torch.arange(length, device=source_groups.device), reduce="amin", include_self=True)
    else:  # pragma: no cover - supported by the target torch runtime
        for group in range(group_count):
            anchors[group] = torch.arange(length, device=source_groups.device)[source_groups == group][0]
    return source_groups, anchors, group_shape


def merge_token_sequence(
    hidden_states: Any,
    keep_ratio: float,
    torch: Any,
    *,
    grid_shape: Optional[Sequence[int]] = None,
) -> tuple[Any, Any, Any, tuple[int, int, int] | None]:
    """Average grouped tokens and return merged tensor plus reversible map."""

    source_groups, anchors, group_shape = build_token_groups(
        int(hidden_states.shape[1]),
        float(keep_ratio),
        torch,
        grid_shape=grid_shape,
    )
    source_groups = source_groups.to(device=hidden_states.device)
    anchors = anchors.to(device=hidden_states.device)
    if int(source_groups.numel()) == int(anchors.numel()) and int(anchors.numel()) == int(hidden_states.shape[1]):
        return hidden_states, None, None, group_shape
    merged = hidden_states.new_zeros((hidden_states.shape[0], int(anchors.numel()), hidden_states.shape[2]))
    merged.scatter_add_(1, source_groups.view(1, -1, 1).expand(hidden_states.shape[0], -1, hidden_states.shape[2]), hidden_states)
    counts = torch.bincount(source_groups, minlength=int(anchors.numel())).clamp_min(1).to(hidden_states.dtype).view(1, -1, 1)
    return merged / counts, source_groups, anchors, group_shape
