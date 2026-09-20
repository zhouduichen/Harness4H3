"""Small server-local GPU lease helper for detached Student jobs."""

from __future__ import annotations

import subprocess
import time


class GPUResourceUnavailable(RuntimeError):
    """Raised when no GPU satisfies the requested free-memory budget."""


def select_free_cuda_device(min_free_memory_gb: float, wait_s: int) -> str:
    deadline = time.monotonic() + max(0, int(wait_s))
    minimum_mib = int(float(min_free_memory_gb) * 1024)
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
            candidates = []
            for line in probe.stdout.splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 2:
                    continue
                try:
                    index, free_mib = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
                if free_mib >= minimum_mib:
                    candidates.append((free_mib, index))
            if candidates:
                _, index = max(candidates)
                return "cuda:%d" % index
            last_status = "no GPU has at least %.1f GiB free" % float(min_free_memory_gb)
        if time.monotonic() >= deadline:
            raise GPUResourceUnavailable("GPU lease timeout after %ss: %s" % (int(wait_s), last_status))
        time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))


__all__ = ["GPUResourceUnavailable", "select_free_cuda_device"]
