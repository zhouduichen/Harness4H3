"""Small server-local GPU lease helper for detached Student jobs."""

from __future__ import annotations

import json
import os
import tempfile
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


class GPUResourceUnavailable(RuntimeError):
    """Raised when no GPU satisfies the requested free-memory budget."""


@dataclass(frozen=True)
class GPUAllocation:
    """Distinct role devices selected for one Student worker."""

    teacher_devices: tuple[str, ...]
    student_device: str
    worker_min_free_memory_gb: float
    teacher_rank_min_free_memory_gb: float
    student_min_free_memory_gb: float
    free_memory_gb: dict[str, float]

    @property
    def all_devices(self) -> tuple[str, ...]:
        return self.teacher_devices + (self.student_device,)


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


def select_teacher_gpu_devices(
    world_size: int,
    min_free_memory_gb: float,
    wait_s: int,
    *,
    visible_devices: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Select exactly ``world_size`` distinct GPUs for a sharded Teacher."""

    world_size = int(world_size)
    minimum = float(min_free_memory_gb)
    if world_size != 3 or minimum <= 0:
        raise ValueError("online H3 Teacher requires world_size=3 and a positive memory floor")
    allowed = set(str(item) for item in visible_devices) if visible_devices is not None else None
    deadline = time.monotonic() + max(0, int(wait_s))
    last_status = "nvidia-smi returned fewer than three eligible GPUs"
    while True:
        try:
            available = list(_query_free_memory())
        except (OSError, subprocess.SubprocessError) as exc:
            available = []
            last_status = "unable to query nvidia-smi: %s" % exc
        if allowed is not None:
            available = [item for item in available if item[0] in allowed]
        selected = [item for item in sorted(available, key=lambda item: item[1], reverse=True) if item[1] >= minimum]
        if len(selected) >= world_size:
            return tuple(item[0] for item in selected[:world_size])
        last_status = "fewer than three Teacher GPUs satisfy %.3f GiB" % minimum
        if time.monotonic() >= deadline:
            raise GPUResourceUnavailable("Teacher GPU lease timeout after %ss: %s" % (int(wait_s), last_status))
        time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))


def _query_free_memory() -> tuple[tuple[str, float], ...]:
    """Return visible CUDA devices and free memory in GiB."""

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
    values = []
    for line in probe.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            index, free_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        values.append(("cuda:%d" % index, float(free_mib) / 1024.0))
    return tuple(values)


def select_role_gpu_allocation(
    teacher_world_size: int,
    teacher_rank_min_free_memory_gb: float,
    student_min_free_memory_gb: float,
    wait_s: int,
    *,
    worker_min_free_memory_gb: float = 0.0,
    visible_devices: Optional[Sequence[str]] = None,
) -> GPUAllocation:
    """Select distinct Teacher ranks and one Student device.

    ``worker_min_free_memory_gb`` is an aggregate worker admission floor;
    role-specific values are per-device floors.  This keeps all three
    scheduler controls observable and independently effective.
    """

    world_size = int(teacher_world_size)
    teacher_min = float(teacher_rank_min_free_memory_gb)
    student_min = float(student_min_free_memory_gb)
    worker_min = float(worker_min_free_memory_gb)
    if world_size < 1 or teacher_min <= 0 or student_min <= 0 or worker_min < 0:
        raise ValueError("GPU role allocation limits are invalid")
    allowed = tuple(str(item) for item in visible_devices) if visible_devices is not None else None
    deadline = time.monotonic() + max(0, int(wait_s))
    last_status = "no GPU satisfies role-specific memory requirements"
    while True:
        try:
            available = list(_query_free_memory())
        except (OSError, subprocess.SubprocessError) as exc:
            available = []
            last_status = "unable to query nvidia-smi: %s" % exc
        if allowed is not None:
            allowed_set = set(allowed)
            available = [item for item in available if item[0] in allowed_set]
        available.sort(key=lambda item: item[1], reverse=True)
        teachers = [item for item in available if item[1] >= teacher_min]
        if len(teachers) >= world_size:
            teacher_items = teachers[:world_size]
            used = {item[0] for item in teacher_items}
            students = [item for item in available if item[0] not in used and item[1] >= student_min]
            if students:
                student_item = students[0]
                selected = teacher_items + [student_item]
                if sum(item[1] for item in selected) >= worker_min:
                    memory = {device: free for device, free in selected}
                    return GPUAllocation(
                        teacher_devices=tuple(item[0] for item in teacher_items),
                        student_device=student_item[0],
                        worker_min_free_memory_gb=worker_min,
                        teacher_rank_min_free_memory_gb=teacher_min,
                        student_min_free_memory_gb=student_min,
                        free_memory_gb=memory,
                    )
                last_status = "selected GPUs have less than worker aggregate minimum %.3f GiB" % worker_min
            else:
                last_status = "no distinct Student GPU satisfies %.3f GiB" % student_min
        else:
            last_status = "fewer than %d Teacher GPUs satisfy %.3f GiB" % (world_size, teacher_min)
        if time.monotonic() >= deadline:
            raise GPUResourceUnavailable("GPU role lease timeout after %ss: %s" % (int(wait_s), last_status))
        time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))


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
    "GPUAllocation",
    "GPUResourceUnavailable",
    "acquire_controller_handoff",
    "publish_worker_gpu_lease",
    "release_controller_handoff",
    "select_free_cuda_device",
    "select_free_cuda_devices",
    "select_teacher_gpu_devices",
    "select_role_gpu_allocation",
]
