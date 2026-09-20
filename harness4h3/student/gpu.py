"""Small server-local GPU lease helper for detached Student jobs."""

from __future__ import annotations

import subprocess
import time
from typing import Sequence


class GPUResourceUnavailable(RuntimeError):
    """Raised when no GPU satisfies the requested free-memory budget."""


def select_free_cuda_device(min_free_memory_gb: float, wait_s: int) -> str:
    return select_free_cuda_devices((float(min_free_memory_gb),), wait_s)[0]


def select_free_cuda_devices(min_free_memory_gb: Sequence[float], wait_s: int) -> tuple[str, ...]:
    """Lease distinct GPUs for roles with descending free-memory needs."""

    requirements = tuple(float(item) for item in min_free_memory_gb)
    if not requirements or any(item <= 0 for item in requirements):
        raise ValueError("GPU free-memory requirements must be positive")
    deadline = time.monotonic() + max(0, int(wait_s))
    last_status = "nvidia-smi returned no GPU with enough free memory"
    while True:
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_status = "unable to query nvidia-smi: %s" % exc
        else:
            available = []
            for line in probe.stdout.splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 2:
                    continue
                try:
                    index, free_mib = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
                available.append((free_mib, index))
            selected = []
            used = set()
            # Satisfy the largest role first, then use the most-free remaining GPU.
            for requirement in sorted(requirements, reverse=True):
                minimum_mib = int(requirement * 1024)
                candidates = [item for item in available if item[1] not in used and item[0] >= minimum_mib]
                if not candidates:
                    selected = []
                    break
                free_mib, index = max(candidates)
                used.add(index)
                selected.append((free_mib, index))
            if len(selected) == len(requirements):
                return tuple("cuda:%d" % index for _, index in selected)
            last_status = "fewer than %d GPUs satisfy free-memory requirements %s" % (len(requirements), requirements)
        if time.monotonic() >= deadline:
            raise GPUResourceUnavailable("GPU lease timeout after %ss: %s" % (int(wait_s), last_status))
        time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))


__all__ = ["GPUResourceUnavailable", "select_free_cuda_device", "select_free_cuda_devices"]
