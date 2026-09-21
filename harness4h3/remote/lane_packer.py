"""Pure GPU request shaping for the remote campaign lane packer."""

from __future__ import annotations

from typing import Any, Mapping


def pack_worker_request(
    request: Mapping[str, Any],
    *,
    total_gpu_count: int,
    controller_overlap_gpus: int,
) -> Mapping[str, Any]:
    """Reserve a Controller lane without weakening a worker's minimum.

    The Controller plan remains the source of truth.  This function only
    shapes the execution request when the campaign deliberately keeps a
    Controller lane alive.  CPU-only operators and non-distributed requests
    are returned as copies so pruning/quantization do not acquire fake GPUs.
    """

    if not isinstance(request, Mapping):
        raise ValueError("worker resource request must be a mapping")
    if (
        isinstance(total_gpu_count, bool)
        or not isinstance(total_gpu_count, int)
        or total_gpu_count <= 0
    ):
        raise ValueError("total_gpu_count must be a positive integer")
    if (
        isinstance(controller_overlap_gpus, bool)
        or not isinstance(controller_overlap_gpus, int)
        or controller_overlap_gpus < 0
        or controller_overlap_gpus >= total_gpu_count
    ):
        raise ValueError("controller_overlap_gpus must be in [0, total_gpu_count)")

    effective = dict(request)
    distributed = effective.get("distributed")
    if not isinstance(distributed, bool):
        raise ValueError("worker resource request distributed must be boolean")
    if not distributed or controller_overlap_gpus == 0:
        return effective

    def integer(name: str) -> int:
        value = effective.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("worker resource request %s must be an integer" % name)
        if value < 0 or value > total_gpu_count:
            raise ValueError("worker resource request %s is outside GPU range" % name)
        return value

    minimum = integer("min_gpu_count")
    maximum = integer("max_gpu_count")
    requested = integer("gpu_count")
    if minimum > maximum:
        raise ValueError("worker resource request min_gpu_count cannot exceed max_gpu_count")
    if not minimum <= requested <= maximum:
        raise ValueError("worker resource request gpu_count must be within its range")

    cap = total_gpu_count - controller_overlap_gpus
    if cap < minimum:
        raise ValueError("controller overlap leaves fewer GPUs than the distributed minimum")
    effective_maximum = min(maximum, cap)
    effective["max_gpu_count"] = effective_maximum
    effective["gpu_count"] = min(requested, effective_maximum)
    # The lane packer is explicitly an elastic execution policy.  Keeping
    # min_gpu_count unchanged means the worker still fails closed when the
    # evaluator or Controller consumes too much live capacity.
    effective["elastic"] = True
    # ``exclusive`` applies to the selected worker GPUs.  A Controller on
    # its own reserved card is an intentional disjoint lane, so retaining a
    # host-wide exclusive flag would make the scheduler reject safe overlap
    # merely because nvidia-smi reports another process somewhere else.
    effective["exclusive"] = False
    return effective


__all__ = ["pack_worker_request"]
