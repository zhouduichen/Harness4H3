"""Fail-closed campaign-level lease management for the ComfyUI GPU."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Optional, Tuple

from .scheduler import RemoteResourceScheduler
from .ssh import SSHClient


@dataclass(frozen=True)
class ComfyUILeaseResult:
    state: str
    success: bool
    reason: str
    queue: Mapping[str, Any]
    response: Mapping[str, Any]
    memory_snapshot: Mapping[int, Tuple[int, Optional[int]]]
    elapsed_s: float
    gpu_index: int


class ComfyUILeaseManager:
    """Coordinate ComfyUI cache release with scheduler reservations.

    The manager deliberately reserves the GPU before every release attempt.
    A reservation is removed only after the queue is idle, ``/free`` returns a
    JSON object, and the scheduler's memory waterline is satisfied.
    """

    def __init__(
        self,
        client: SSHClient,
        scheduler: RemoteResourceScheduler,
        port: int,
        gpu_index: int = 0,
        api_timeout_s: float = 10.0,
        release_wait_s: float = 30.0,
        lease_path: Optional[str] = None,
    ):
        self.client = client
        self.scheduler = scheduler
        self.port = int(port)
        self.gpu_index = int(gpu_index)
        self.api_timeout_s = float(api_timeout_s)
        if self.api_timeout_s <= 0:
            raise ValueError("api_timeout_s must be positive")
        self.release_wait_s = float(release_wait_s)
        if self.release_wait_s <= 0:
            raise ValueError("release_wait_s must be positive")
        if self.gpu_index < 0 or self.gpu_index >= self.scheduler.gpu_count:
            raise ValueError("gpu_index must be within the scheduler GPU range")
        self._last_successful_release: Optional[ComfyUILeaseResult] = None
        # Only a PID explicitly recorded by this campaign may be stopped.  A
        # pre-existing ComfyUI service has no owner here and is never killed.
        self._owned_pid: Optional[int] = None
        # A scheduler reservation is created only by prepare_for_benchmark().
        # Keeping lease ownership separate from the scheduler state makes the
        # Controller free to use GPU 0 while no ComfyUI evaluation is active.
        self._lease_active = False
        config = getattr(self.client, "config", None)
        campaign_root = getattr(config, "resolved_campaign_root", None)
        if lease_path is not None and callable(getattr(self.client, "write_json", None)) and callable(
            getattr(self.client, "remove_file", None)
        ):
            self.lease_path = str(lease_path)
        else:
            self.lease_path = (
                str(PurePosixPath(str(campaign_root)) / ".comfyui-gpu-lease.json")
                if campaign_root
                and callable(getattr(self.client, "write_json", None))
                and callable(getattr(self.client, "remove_file", None))
                else None
            )

    @property
    def lease_active(self) -> bool:
        """Whether this manager still owns the evaluator GPU reservation."""

        return bool(self._lease_active)

    def set_owned_process(self, pid: int) -> None:
        """Record the campaign-owned launcher PID for scoped shutdown."""

        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("owned ComfyUI PID must be a positive integer")
        if self._owned_pid is not None and self._owned_pid != pid:
            raise RuntimeError("ComfyUI lease already owns PID %d" % self._owned_pid)
        self._owned_pid = pid

    def _marker_owned_pid(self) -> Optional[int]:
        """Recover a previously recorded PID after a campaign restart."""

        if self._owned_pid is not None or self.lease_path is None:
            return self._owned_pid
        if not callable(getattr(self.client, "read_json", None)):
            return None
        try:
            value = self.client.read_json(self.lease_path)
        except Exception:
            return None
        if not isinstance(value, Mapping):
            return None
        raw = value.get("process_pid")
        if isinstance(raw, bool):
            return None
        try:
            pid = int(raw)
        except (TypeError, ValueError):
            return None
        if pid <= 0:
            return None
        self._owned_pid = pid
        return pid

    def stop_owned_process(self, wait_s: float = 10.0) -> Mapping[str, Any]:
        """Gracefully stop only the launcher PID owned by this campaign.

        The launcher is responsible for forwarding TERM to its ComfyUI child.
        The final KILL is still scoped to that same recorded PID and is used
        only after a bounded graceful wait.  Missing or foreign PIDs are
        treated as no-op states.
        """

        pid = self._marker_owned_pid()
        if pid is None:
            return {"status": "not_owned", "pid": None}
        try:
            probe = self.client.run(("kill", "-0", str(pid)), check=False)
        except Exception as exc:
            return {"status": "stop_failed", "pid": pid, "reason": str(exc)[:500]}
        if int(getattr(probe, "returncode", 1)) != 0:
            self._owned_pid = None
            return {"status": "already_stopped", "pid": pid}
        try:
            term = self.client.run(("kill", "-TERM", str(pid)), check=False)
            if int(getattr(term, "returncode", 1)) != 0:
                return {"status": "stop_failed", "pid": pid, "reason": "TERM was refused"}
            deadline = time.monotonic() + max(0.0, float(wait_s))
            while time.monotonic() < deadline:
                probe = self.client.run(("kill", "-0", str(pid)), check=False)
                if int(getattr(probe, "returncode", 1)) != 0:
                    self._owned_pid = None
                    return {"status": "stopped", "pid": pid}
                time.sleep(0.25)
            kill = self.client.run(("kill", "-KILL", str(pid)), check=False)
            if int(getattr(kill, "returncode", 1)) != 0:
                return {"status": "stop_failed", "pid": pid, "reason": "KILL was refused"}
            probe = self.client.run(("kill", "-0", str(pid)), check=False)
            if int(getattr(probe, "returncode", 1)) == 0:
                return {"status": "stop_failed", "pid": pid, "reason": "process survived KILL"}
            self._owned_pid = None
            return {"status": "stopped", "pid": pid, "forced": True}
        except Exception as exc:
            return {"status": "stop_failed", "pid": pid, "reason": str(exc)[:500]}

    @property
    def _base_url(self) -> str:
        return "http://127.0.0.1:%d" % self.port

    def _request(self, path: str, payload: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        command = ["curl", "-fsS", "--max-time", str(max(1, int(self.api_timeout_s)))]
        if payload is not None:
            command.extend(
                [
                    "-X",
                    "POST",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                ]
            )
        command.append(self._base_url + path)
        result = self.client.run(tuple(command), check=False, timeout_s=self.api_timeout_s + 2.0)
        if int(getattr(result, "returncode", 1)) != 0:
            detail = str(getattr(result, "stderr", "")).strip()
            raise RuntimeError(detail or "ComfyUI request failed")
        raw = str(getattr(result, "stdout", ""))
        if path == "/free" and not raw.strip():
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid JSON from ComfyUI %s: %s" % (path, exc))
        if not isinstance(value, Mapping):
            raise RuntimeError("ComfyUI %s response must be an object" % path)
        return dict(value)

    def prepare_for_benchmark(self) -> Mapping[str, Any]:
        self._last_successful_release = None
        self.scheduler.reserve_gpu(self.gpu_index)
        self._lease_active = True
        if self.lease_path is not None:
            # The launcher watches this marker before selecting a GPU.  It is
            # deliberately written before any benchmark request so a live
            # Controller on GPU 0 can be stopped before ComfyUI loads H3.
            self.client.write_json(
                self.lease_path,
                {
                    "state": "reserved_for_benchmark",
                    "owner_pid": os.getpid(),
                    "created_at": time.time(),
                    "gpu_index": self.gpu_index,
                    "process_pid": self._owned_pid,
                },
            )
        return {
            "state": "reserved_for_benchmark",
            "gpu_index": self.gpu_index,
            "reserved_gpu_indices": list(self.scheduler.reserved_gpu_indices),
            "lease_path": self.lease_path,
        }

    def _clear_lease_marker(self) -> Mapping[str, Any]:
        if self.lease_path is None:
            return {"status": "not_configured"}
        response = dict(self.client.remove_file(self.lease_path))
        status = str(response.get("status", "")).strip().lower()
        if status not in {"deleted", "missing"}:
            raise RuntimeError("ComfyUI lease marker removal was refused: %s" % (response.get("reason") or status))
        return response

    def release_if_idle(self, stop_process: bool = False) -> ComfyUILeaseResult:
        started = time.monotonic()
        if self._last_successful_release is not None and self.gpu_index not in self.scheduler.reserved_gpu_indices:
            return self._last_successful_release
        self.scheduler.reserve_gpu(self.gpu_index)
        queue: Mapping[str, Any] = {}
        response: Mapping[str, Any] = {}
        snapshot: Mapping[int, Tuple[int, Optional[int]]] = {}
        try:
            queue = self._request("/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                return ComfyUILeaseResult(
                    "reserved_for_benchmark",
                    False,
                    "queue_active",
                    queue,
                    response,
                    snapshot,
                    time.monotonic() - started,
                    self.gpu_index,
                )
            response = self._request(
                "/free",
                {"unload_models": True, "free_memory": True},
            )
            # ComfyUI's /free wakes its prompt worker and performs unloading
            # asynchronously.  Poll the same scheduler waterline for a short,
            # bounded interval instead of treating the first post-request
            # sample as proof that unloading failed.
            deadline = time.monotonic() + self.release_wait_s
            while True:
                snapshot, _ = self.scheduler.snapshot()
                if self.scheduler.meets_memory_waterline(self.gpu_index, snapshot):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(1.0, remaining))
            if not self.scheduler.meets_memory_waterline(self.gpu_index, snapshot):
                return ComfyUILeaseResult(
                    "release_failed",
                    False,
                    "post_release_memory_below_waterline",
                    queue,
                    response,
                    snapshot,
                    time.monotonic() - started,
                    self.gpu_index,
                )
            if stop_process:
                process_result = self.stop_owned_process()
                response = {**dict(response), "owned_process": dict(process_result)}
                if process_result.get("status") == "stop_failed":
                    return ComfyUILeaseResult(
                        "release_failed",
                        False,
                        "owned_process_stop_failed",
                        queue,
                        response,
                        snapshot,
                        time.monotonic() - started,
                        self.gpu_index,
                    )
            marker = self._clear_lease_marker()
            response = {**dict(response), "lease_marker": marker}
            self.scheduler.release_gpu(self.gpu_index)
            self._lease_active = False
            result = ComfyUILeaseResult(
                "released_for_other_work",
                True,
                "released_and_verified",
                queue,
                response,
                snapshot,
                time.monotonic() - started,
                self.gpu_index,
            )
            self._last_successful_release = result
            return result
        except Exception as exc:
            return ComfyUILeaseResult(
                "release_failed",
                False,
                str(exc),
                queue,
                response,
                snapshot,
                time.monotonic() - started,
                self.gpu_index,
            )


__all__ = ["ComfyUILeaseManager", "ComfyUILeaseResult"]
