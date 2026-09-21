"""Non-destructive remote resource scheduling for Controller requests."""

from __future__ import annotations

import csv
import io
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional, Tuple

from .ssh import SSHClient


@dataclass(frozen=True)
class ResourceDecision:
    status: str
    requested_gpu_count: int
    allocated_gpus: Tuple[int, ...] = ()
    reason: str = ""
    min_gpu_count: int = 0
    max_gpu_count: int = 0
    elastic: bool = False
    available_gpu_count: int = 0

    @property
    def actual_gpu_count(self) -> int:
        return len(self.allocated_gpus)

    def to_dict(self):
        value = asdict(self)
        value["actual_gpu_count"] = self.actual_gpu_count
        return value


class RemoteResourceScheduler:
    """Map a Controller resource request without killing or selecting work."""

    _GPU_QUERY = (
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )
    _GPU_UUID_QUERY = (
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    )
    _GPU_TELEMETRY_QUERY = (
        "nvidia-smi",
        "--query-gpu=index,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    _PROCESS_QUERY = ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits")

    def __init__(
        self,
        client: SSHClient,
        gpu_count: int = 4,
        min_free_memory_mb: int = 20 * 1024,
        reserved_gpu_indices: Tuple[int, ...] = (),
        lease_path: Optional[str] = None,
        controller_lease_path: Optional[str] = None,
        lease_ttl_s: float = 24 * 3600.0,
        allocation_recheck_s: float = 1.0,
    ):
        self.client = client
        self.gpu_count = int(gpu_count)
        self.min_free_memory_mb = int(min_free_memory_mb)
        if self.min_free_memory_mb <= 0:
            raise ValueError("min_free_memory_mb must be positive")
        self.reserved_gpu_indices = tuple(sorted(set(int(index) for index in reserved_gpu_indices)))
        if any(index < 0 or index >= self.gpu_count for index in self.reserved_gpu_indices):
            raise ValueError("reserved_gpu_indices must be within the GPU range")
        if float(lease_ttl_s) <= 0:
            raise ValueError("lease_ttl_s must be positive")
        if float(allocation_recheck_s) < 0:
            raise ValueError("allocation_recheck_s must be non-negative")
        config = getattr(client, "config", None)
        campaign_root = getattr(config, "resolved_campaign_root", None)
        self.lease_path = str(
            lease_path
            or (PurePosixPath(str(campaign_root)) / ".h3-worker-gpu-lease.json" if campaign_root else "")
        ) or None
        self.controller_lease_path = str(
            controller_lease_path
            or (PurePosixPath(str(campaign_root)) / ".controller-gpu-lease.json" if campaign_root else "")
        ) or None
        self.lease_ttl_s = float(lease_ttl_s)
        # A live foreign job can enter between the first nvidia-smi query and
        # torchrun/vLLM initialization.  A short second sample closes that
        # race for normal scheduler-owned launches without pretending that a
        # shared host can provide an atomic GPU allocation primitive.
        self.allocation_recheck_s = float(allocation_recheck_s)
        self._lease_token: Optional[str] = None
        self._last_process_gpu_indices: Optional[Tuple[int, ...]] = None
        self._last_gpu_telemetry: Mapping[int, Mapping[str, float]] = {}

    @property
    def shared_lease_enabled(self) -> bool:
        """Whether this transport can publish a lease visible to the launcher."""

        return bool(
            self.lease_path
            and callable(getattr(self.client, "write_json", None))
            and callable(getattr(self.client, "read_json", None))
            and callable(getattr(self.client, "remove_file", None))
        )

    @property
    def lease_active(self) -> bool:
        """Whether this scheduler currently owns a shared worker lease."""

        return self._lease_token is not None

    @property
    def last_compute_gpu_indices(self) -> Optional[Tuple[int, ...]]:
        """GPU indices occupied by the last compute-process snapshot.

        ``None`` is the fail-closed marker for a non-empty process listing
        whose UUID could not be mapped. Callers must also inspect the process
        string returned by :meth:`snapshot` so an empty tuple is not confused
        with an unknown mapping.
        """

        return self._last_process_gpu_indices

    @property
    def last_gpu_telemetry(self) -> Mapping[int, Mapping[str, float]]:
        """Return the latest advisory power/utilization sample per GPU.

        This is diagnostic scheduling context, not an allocation gate.  A
        failed telemetry query therefore returns an empty mapping while the
        existing memory/process fail-closed checks remain authoritative.
        """

        return {int(index): dict(values) for index, values in self._last_gpu_telemetry.items()}

    @property
    def controller_reserved_gpu_indices(self) -> Tuple[int, ...]:
        """Return the live launcher-owned Controller GPU reservation."""

        return self._controller_reserved_gpu_indices()

    def _read_lease(self) -> Optional[Mapping[str, Any]]:
        if not self.shared_lease_enabled or self.lease_path is None:
            return None
        exists = self.client.run(("test", "-e", self.lease_path), check=False)
        if int(getattr(exists, "returncode", 1)) != 0:
            return None
        value = self.client.read_json(self.lease_path)
        if not isinstance(value, Mapping):
            raise ValueError("worker GPU lease is not a mapping")
        return value

    def _read_controller_lease(self) -> Optional[Mapping[str, Any]]:
        """Read a live launcher-owned Controller reservation, if present."""

        if not self.controller_lease_path or not callable(getattr(self.client, "read_json", None)):
            return None
        exists = self.client.run(("test", "-e", self.controller_lease_path), check=False)
        if int(getattr(exists, "returncode", 1)) != 0:
            return None
        value = self.client.read_json(self.controller_lease_path)
        if not isinstance(value, Mapping):
            raise ValueError("Controller GPU lease is not a mapping")
        try:
            created = float(value.get("created_at", 0.0))
            expires = float(value.get("expires_at", 0.0))
            owner_pid = int(value.get("owner_pid", 0))
            allocated = value.get("allocated_gpus")
        except (TypeError, ValueError) as exc:
            raise ValueError("Controller GPU lease has invalid metadata") from exc
        if created <= 0.0 or expires <= 0.0 or not isinstance(allocated, list):
            raise ValueError("Controller GPU lease has incomplete metadata")
        now = time.time()
        if now > expires or now - created > self.lease_ttl_s:
            return None
        if owner_pid <= 0:
            raise ValueError("Controller GPU lease owner_pid must be positive")
        owner = self.client.run(("ps", "-p", str(owner_pid), "-o", "args="), check=False)
        if int(getattr(owner, "returncode", 1)) != 0 or not str(getattr(owner, "stdout", "")).strip():
            return None
        indices = []
        for item in allocated:
            if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < self.gpu_count:
                raise ValueError("Controller GPU lease contains an invalid GPU index")
            indices.append(item)
        if len(indices) != len(set(indices)):
            raise ValueError("Controller GPU lease contains duplicate GPU indices")
        return value

    def _controller_reserved_gpu_indices(self) -> Tuple[int, ...]:
        value = self._read_controller_lease()
        if value is None:
            return ()
        return tuple(sorted(int(item) for item in value.get("allocated_gpus", [])))

    def _lease_is_live(self, value: Mapping[str, Any], now: Optional[float] = None) -> bool:
        current = time.time() if now is None else float(now)
        try:
            expires = float(value.get("expires_at", 0.0))
            created = float(value.get("created_at", 0.0))
        except (TypeError, ValueError):
            return True
        if expires <= 0 or created <= 0:
            return True
        return current <= expires and current - created <= self.lease_ttl_s

    def _lease_owner_alive(self, value: Mapping[str, Any]) -> bool:
        """Fail closed for a live lease whose owner process still exists.

        A supervisor restart can leave a lease with a long TTL behind after
        the old process has exited.  Reclaim only that case; an unrelated
        live supervisor must continue to block another worker allocation.
        """

        try:
            owner_pid = int(value.get("owner_pid"))
        except (TypeError, ValueError):
            return True
        result = self.client.run(("ps", "-p", str(owner_pid), "-o", "args="), check=False)
        if int(getattr(result, "returncode", 1)) != 0:
            return False
        command = str(getattr(result, "stdout", "")).strip()
        if not command:
            return False
        # Worker leases are published by the overnight campaign supervisor.
        # Do not reclaim a PID that is alive but cannot be identified as that
        # supervisor; this keeps the scheduler conservative on shared hosts.
        return "run_overnight_controller.py" in command

    def _lease_has_live_compute(self, value: Mapping[str, Any]) -> bool:
        """Return whether a stale lease still overlaps a live compute process.

        The campaign supervisor can die while its torchrun child survives. In
        that case the supervisor PID is gone, but reclaiming the lease would
        allow a replacement campaign to select the child's GPUs. Treat an
        unmappable or unavailable process snapshot as unsafe as well.
        """

        allocated = value.get("allocated_gpus")
        if not isinstance(allocated, list):
            return True
        try:
            allocated_indices = set()
            for item in allocated:
                if isinstance(item, bool):
                    return True
                index = int(item)
                if index < 0 or index >= self.gpu_count:
                    return True
                allocated_indices.add(index)
            if not allocated_indices:
                return False
            self._snapshot()
        except Exception:
            return True
        process_indices = self._last_process_gpu_indices
        if process_indices is None:
            return True
        return bool(allocated_indices.intersection(process_indices))

    def _publish_lease(self, allocated_gpus: Tuple[int, ...]) -> Tuple[bool, str]:
        if not self.shared_lease_enabled or self.lease_path is None:
            return True, "shared lease unavailable on this transport"
        now = time.time()
        try:
            current = self._read_lease()
        except Exception as exc:
            return False, "cannot inspect existing worker GPU lease: %s" % str(exc)[:500]
        reclaimed = False
        if current is not None:
            token = str(current.get("token", ""))
            if self._lease_is_live(current, now) and token != self._lease_token:
                if self._lease_owner_alive(current):
                    return False, "another live worker GPU lease is present"
                if self._lease_has_live_compute(current):
                    return False, "stale worker GPU lease overlaps a live compute process"
                try:
                    self.client.remove_file(self.lease_path)
                except Exception as exc:
                    return False, "cannot reclaim stale worker GPU lease: %s" % str(exc)[:500]
                reclaimed = True
        token = "h3-worker-%d-%s" % (os.getpid(), uuid.uuid4().hex)
        payload = {
            "state": "allocated_for_worker",
            "token": token,
            "owner_pid": os.getpid(),
            "created_at": now,
            "expires_at": now + self.lease_ttl_s,
            "allocated_gpus": list(allocated_gpus),
        }
        try:
            self.client.write_json(self.lease_path, payload)
        except Exception as exc:
            return False, "cannot publish worker GPU lease: %s" % str(exc)[:500]
        self._lease_token = token
        return True, (
            "reclaimed stale worker GPU lease; shared worker GPU lease published"
            if reclaimed
            else "shared worker GPU lease published"
        )

    def release(self) -> Mapping[str, Any]:
        """Release the current worker lease if this scheduler owns it."""

        if not self.shared_lease_enabled or self.lease_path is None or self._lease_token is None:
            self._lease_token = None
            return {"status": "not_configured" if not self.shared_lease_enabled else "missing"}
        try:
            current = self._read_lease()
            if current is None or str(current.get("token", "")) != self._lease_token:
                self._lease_token = None
                return {"status": "not_owner"}
            result = dict(self.client.remove_file(self.lease_path))
        finally:
            self._lease_token = None
        return result

    @staticmethod
    def _parse_gpu_memory(raw: str) -> Mapping[int, Tuple[int, Optional[int]]]:
        values = {}
        for row in csv.reader(io.StringIO(raw), skipinitialspace=True):
            if len(row) < 2:
                continue
            try:
                total = int(float(row[2])) if len(row) >= 3 else None
                values[int(row[0])] = (int(float(row[1])), total)
            except (TypeError, ValueError):
                continue
        return values

    @staticmethod
    def _parse_compute_gpu_indices(
        process_raw: str,
        gpu_uuid_raw: str,
        gpu_count: int,
    ) -> Tuple[Optional[Tuple[int, ...]], bool]:
        """Map live compute processes to GPU indices.

        ``nvidia-smi --query-compute-apps`` reports UUIDs rather than indices.
        Keep the mapping explicit so a process on GPU0 cannot be mistaken for
        harmless memory usage on another card.  ``None`` means a non-empty
        process list could not be mapped and must be treated fail-closed.
        """

        uuid_to_index = {}
        for row in csv.reader(io.StringIO(gpu_uuid_raw), skipinitialspace=True):
            if len(row) < 2:
                continue
            try:
                index = int(row[0])
            except (TypeError, ValueError):
                continue
            uuid = str(row[1]).strip()
            if 0 <= index < gpu_count and uuid:
                uuid_to_index[uuid] = index

        process_rows = []
        for row in csv.reader(io.StringIO(process_raw), skipinitialspace=True):
            if not row or not str(row[0]).strip():
                continue
            first = str(row[0]).strip()
            # Test doubles and older wrappers sometimes expose an index
            # directly. Accept that form without weakening UUID mapping.
            if first.isdigit():
                index = int(first)
                if 0 <= index < gpu_count:
                    process_rows.append(index)
                    continue
            process_rows.append(uuid_to_index.get(first))
        if not process_rows:
            return (), False
        if any(index is None for index in process_rows):
            return None, True
        return tuple(sorted(set(int(index) for index in process_rows))), True

    @staticmethod
    def _parse_gpu_telemetry(raw: str) -> Mapping[int, Mapping[str, float]]:
        values = {}
        for row in csv.reader(io.StringIO(raw), skipinitialspace=True):
            if len(row) < 3:
                continue
            try:
                index = int(row[0])
                power = float(row[1])
                utilization = float(row[2])
            except (TypeError, ValueError):
                continue
            if index < 0 or not math.isfinite(power) or not math.isfinite(utilization):
                continue
            if power < 0 or utilization < 0:
                continue
            values[index] = {
                "power_w": power,
                "utilization_gpu_pct": utilization,
            }
        return values

    def _snapshot(self) -> Tuple[Mapping[int, Tuple[int, Optional[int]]], str]:
        gpu = self.client.run(self._GPU_QUERY)
        try:
            telemetry = self.client.run(self._GPU_TELEMETRY_QUERY, check=False)
            self._last_gpu_telemetry = self._parse_gpu_telemetry(str(getattr(telemetry, "stdout", "")))
        except Exception:
            self._last_gpu_telemetry = {}
        processes = self.client.run(self._PROCESS_QUERY, check=False)
        process_raw = str(getattr(processes, "stdout", ""))
        process_returncode = int(getattr(processes, "returncode", 0) or 0)
        if process_returncode != 0 and not process_raw.strip():
            # An unavailable process query is not evidence that the GPUs are
            # idle. Feed an unmappable sentinel through the existing
            # fail-closed path instead of allowing a worker to race an
            # unobserved CUDA process.
            process_raw = "<compute-process-query-unavailable>"
        try:
            gpu_uuids = self.client.run(self._GPU_UUID_QUERY, check=False)
            uuid_raw = str(getattr(gpu_uuids, "stdout", ""))
        except Exception:
            uuid_raw = ""
        process_indices, process_present = self._parse_compute_gpu_indices(
            process_raw,
            uuid_raw,
            self.gpu_count,
        )
        # Preserve ``None`` as the explicit fail-closed marker for a live
        # process whose GPU UUID could not be mapped.  ``acquire`` then blocks
        # the whole snapshot rather than guessing which card is safe.
        self._last_process_gpu_indices = process_indices
        return self._parse_gpu_memory(str(getattr(gpu, "stdout", ""))), process_raw

    def reserve_gpu(self, index: int) -> None:
        index = int(index)
        if index < 0 or index >= self.gpu_count:
            raise ValueError("reserved GPU index must be within the GPU range")
        self.reserved_gpu_indices = tuple(sorted(set(self.reserved_gpu_indices) | {index}))

    def release_gpu(self, index: int) -> None:
        index = int(index)
        if index < 0 or index >= self.gpu_count:
            raise ValueError("reserved GPU index must be within the GPU range")
        self.reserved_gpu_indices = tuple(item for item in self.reserved_gpu_indices if item != index)

    def snapshot(self) -> Tuple[Mapping[int, Tuple[int, Optional[int]]], str]:
        return self._snapshot()

    def meets_memory_waterline(
        self,
        index: int,
        snapshot: Optional[Mapping[int, Tuple[int, Optional[int]]]] = None,
    ) -> bool:
        values = self.snapshot()[0] if snapshot is None else snapshot
        item = values.get(int(index))
        if item is None or item[1] is None:
            return False
        used_mb, total_mb = item
        return int(total_mb) - int(used_mb) >= self.min_free_memory_mb

    def normalize_request(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate and normalize a Controller resource request.

        Older persisted plans only have ``gpu_count``. They retain exact-count
        semantics rather than gaining elasticity implicitly.
        """

        if not isinstance(request, Mapping):
            raise ValueError("resource request must be a mapping")
        count = request.get("gpu_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.gpu_count:
            raise ValueError("resource request gpu_count is invalid")
        exclusive = request.get("exclusive")
        distributed = request.get("distributed")
        if not isinstance(exclusive, bool) or not isinstance(distributed, bool):
            raise ValueError("resource request distributed/exclusive is invalid")
        elastic = request.get("elastic", False)
        if not isinstance(elastic, bool):
            raise ValueError("resource request elastic is invalid")
        minimum = request.get("min_gpu_count", count)
        maximum = request.get("max_gpu_count", count)
        for name, value in (("min_gpu_count", minimum), ("max_gpu_count", maximum)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= self.gpu_count:
                raise ValueError("resource request %s is invalid" % name)
        if minimum > maximum:
            raise ValueError("resource request min_gpu_count cannot exceed max_gpu_count")
        if not minimum <= count <= maximum:
            raise ValueError("resource request gpu_count must be within min_gpu_count and max_gpu_count")
        if not elastic and (minimum != count or maximum != count):
            raise ValueError("non-elastic resource request must use an exact GPU count")
        if distributed and minimum < 2:
            raise ValueError("distributed resource request must allow at least two GPUs")
        normalized = dict(request)
        normalized.update(
            {
                "gpu_count": count,
                "min_gpu_count": minimum,
                "max_gpu_count": maximum,
                "elastic": elastic,
            }
        )
        return normalized

    def acquire(self, request: Mapping[str, Any]) -> ResourceDecision:
        normalized = self.normalize_request(request)
        count = int(normalized["gpu_count"])
        minimum = int(normalized["min_gpu_count"])
        maximum = int(normalized["max_gpu_count"])
        elastic = bool(normalized["elastic"])
        exclusive = bool(normalized["exclusive"])
        if count == 0:
            return ResourceDecision("ready", 0, (), "CPU-only operator", 0, 0, elastic, 0)
        def classify_snapshot(
            snapshot_memory: Mapping[int, Tuple[int, Optional[int]]],
            snapshot_processes: str,
        ) -> Tuple[Tuple[int, ...], set, bool]:
            try:
                controller_reserved = set(self._controller_reserved_gpu_indices())
            except Exception as exc:
                raise RuntimeError("cannot inspect Controller GPU lease: %s" % str(exc)[:500]) from exc
            # For non-exclusive elastic work, a GPU with an unrelated small
            # allocation can still be safe to use when its measured free
            # memory exceeds the H3 worker's safety waterline. Older test
            # doubles only report used memory; retain their conservative exact
            # semantics.
            busy = set()
            for index, (used_mb, total_mb) in snapshot_memory.items():
                if total_mb is None:
                    if used_mb > 128:
                        busy.add(index)
                elif total_mb - used_mb < self.min_free_memory_mb:
                    busy.add(index)
            busy.update(controller_reserved)
            process_present = bool(snapshot_processes.strip())
            process_gpu_indices = self._last_process_gpu_indices
            if process_present and process_gpu_indices is None:
                # A live but unmappable compute process is not safe to share.
                # Block every card for this attempt instead of guessing its GPU.
                busy.update(range(self.gpu_count))
            elif process_gpu_indices:
                # Never share a card with a live compute process. This avoids
                # the previous ``[0,2,3]`` overlap when external Geneval jobs
                # enter between planning and worker initialization.
                busy.update(process_gpu_indices)
            free = tuple(
                index
                for index in range(self.gpu_count)
                if index not in busy and index not in self.reserved_gpu_indices
            )
            return free, busy, process_present

        memory, processes = self._snapshot()
        try:
            free, busy, process_present = classify_snapshot(memory, processes)
        except RuntimeError as exc:
            return ResourceDecision(
                "wait",
                count,
                (),
                str(exc),
                minimum,
                maximum,
                elastic,
                0,
            )
        if exclusive and (process_present or busy):
            return ResourceDecision(
                "wait",
                count,
                (),
                "exclusive GPU request cannot be honored without stopping existing processes",
                minimum,
                maximum,
                elastic,
                0,
            )
        available = len(free)
        actual = min(maximum, available) if elastic else count
        if actual < minimum or (not elastic and available < count):
            requested_text = "%d-%d" % (minimum, maximum) if elastic else str(count)
            return ResourceDecision(
                "wait",
                count,
                (),
                "fewer than the required %s GPUs are currently free" % requested_text,
                minimum,
                maximum,
                elastic,
                available,
            )

        # Recheck only when the first sample found a feasible allocation. A
        # second sample is intentionally before publishing the lease: a lease
        # cannot make a foreign process disappear, but it can prevent this
        # campaign from launching into a GPU that became busy in the gap.
        if self.allocation_recheck_s > 0:
            time.sleep(self.allocation_recheck_s)
            memory, processes = self._snapshot()
            try:
                free, busy, process_present = classify_snapshot(memory, processes)
            except RuntimeError as exc:
                return ResourceDecision(
                    "wait",
                    count,
                    (),
                    str(exc),
                    minimum,
                    maximum,
                    elastic,
                    0,
                )
            if exclusive and (process_present or busy):
                return ResourceDecision(
                    "wait",
                    count,
                    (),
                    "exclusive GPU request cannot be honored without stopping existing processes",
                    minimum,
                    maximum,
                    elastic,
                    0,
                )
            available = len(free)
            actual = min(maximum, available) if elastic else count
            if actual < minimum or (not elastic and available < count):
                requested_text = "%d-%d" % (minimum, maximum) if elastic else str(count)
                return ResourceDecision(
                    "wait",
                    count,
                    (),
                    "fewer than the required %s GPUs are currently free after allocation recheck" % requested_text,
                    minimum,
                    maximum,
                    elastic,
                    available,
                )
        allocated = free[:actual]
        lease_ok, lease_reason = self._publish_lease(allocated)
        if not lease_ok:
            return ResourceDecision(
                "wait",
                count,
                (),
                lease_reason,
                minimum,
                maximum,
                elastic,
                available,
            )
        # Publishing a lease is not an atomic reservation primitive on a
        # shared SSH host.  A foreign job can still enter during the tiny
        # window between the final pre-lease snapshot and torchrun startup.
        # Sample once more while this exact lease is live; if one of the
        # selected cards is no longer free, release only our own lease and
        # retry instead of turning a scheduler race into a training OOM.
        if self.allocation_recheck_s > 0:
            time.sleep(min(self.allocation_recheck_s, 1.0))
            post_lease_memory, post_lease_processes = self._snapshot()
            try:
                post_lease_free, _, _ = classify_snapshot(
                    post_lease_memory,
                    post_lease_processes,
                )
            except RuntimeError as exc:
                self.release()
                return ResourceDecision(
                    "wait",
                    count,
                    (),
                    str(exc) + " after lease publication",
                    minimum,
                    maximum,
                    elastic,
                    0,
                )
            if not set(allocated).issubset(set(post_lease_free)):
                self.release()
                return ResourceDecision(
                    "wait",
                    count,
                    (),
                    "selected GPU became busy after lease publication",
                    minimum,
                    maximum,
                    elastic,
                    len(post_lease_free),
                )
        return ResourceDecision(
            "ready",
            count,
            allocated,
            "mapped by nvidia-smi snapshot; %s" % lease_reason,
            minimum,
            maximum,
            elastic,
            available,
        )


__all__ = ["RemoteResourceScheduler", "ResourceDecision"]
