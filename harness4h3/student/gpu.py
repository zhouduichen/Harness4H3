"""Small server-local GPU lease helper for detached Student jobs."""

from __future__ import annotations

import json
import os
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Sequence


class GPUResourceUnavailable(RuntimeError):
    """Raised when no GPU satisfies the requested free-memory budget."""


def _atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def acquire_controller_handoff(
    hold_file: str | None,
    release_file: str | None,
    worker_lease_file: str | None,
) -> None:
    """Ask the server-local Controller launcher to yield all GPUs.

    The marker protocol is intentionally file based because the LLM server and
    the training worker are sibling detached processes on the remote host.
    ``hold_file`` prevents the launcher from starting a replacement while the
    worker owns the cards; the release marker makes an already-running vLLM
    child terminate promptly.
    """

    if hold_file:
        _atomic_json(Path(hold_file), {"owner_pid": os.getpid(), "state": "worker_handoff"})
    if release_file:
        _atomic_json(Path(release_file), {"owner_pid": os.getpid(), "state": "release_controller"})
    if worker_lease_file:
        _atomic_json(
            Path(worker_lease_file),
            {
                "allocated_gpus": [],
                "created_at": time.time(),
                "expires_at": time.time() + 86400.0,
                "owner_pid": os.getpid(),
                "state": "acquiring",
            },
        )


def publish_worker_gpu_lease(worker_lease_file: str | None, devices: Sequence[str]) -> None:
    if not worker_lease_file:
        return
    gpu_ids = []
    for device in devices:
        value = str(device)
        if not value.startswith("cuda:"):
            raise ValueError("worker lease device must be cuda:N: %s" % value)
        gpu_ids.append(int(value.split(":", 1)[1]))
    now = time.time()
    _atomic_json(
        Path(worker_lease_file),
        {
            "allocated_gpus": gpu_ids,
            "created_at": now,
            "expires_at": now + 86400.0,
            "owner_pid": os.getpid(),
            "state": "allocated",
        },
    )


def release_controller_handoff(
    hold_file: str | None,
    release_file: str | None,
    worker_lease_file: str | None,
) -> None:
    for value in (hold_file, release_file, worker_lease_file):
        if value:
            try:
                Path(value).unlink()
            except FileNotFoundError:
                pass


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


__all__ = [
    "GPUResourceUnavailable",
    "acquire_controller_handoff",
    "publish_worker_gpu_lease",
    "release_controller_handoff",
    "select_free_cuda_device",
    "select_free_cuda_devices",
]
