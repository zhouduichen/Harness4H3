from types import SimpleNamespace
import time

import pytest

from harness4h3.remote.scheduler import RemoteResourceScheduler


class FakeSSH:
    def __init__(self, gpu_memory, processes="", gpu_uuids="", process_returncode=0, gpu_telemetry=""):
        self.gpu_memory = gpu_memory
        self.processes = processes
        self.gpu_uuids = gpu_uuids
        self.process_returncode = process_returncode
        self.gpu_telemetry = gpu_telemetry

    def run(self, command, **kwargs):
        if any("query-compute-apps" in str(item) for item in command):
            return SimpleNamespace(stdout=self.processes, returncode=self.process_returncode)
        if any("query-gpu=index,uuid" in str(item) for item in command):
            return SimpleNamespace(stdout=self.gpu_uuids, returncode=0)
        if any("query-gpu=index,power.draw,utilization.gpu" in str(item) for item in command):
            return SimpleNamespace(stdout=self.gpu_telemetry, returncode=0)
        return SimpleNamespace(stdout=self.gpu_memory, returncode=0)


class ChangingSnapshotSSH(FakeSSH):
    """Expose a foreign compute process only in the allocation recheck."""

    def __init__(self):
        super().__init__(
            "0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
            gpu_uuids="0, GPU-uuid-0\n1, GPU-uuid-1\n2, GPU-uuid-2\n3, GPU-uuid-3\n",
        )
        self.snapshot_count = 0

    def run(self, command, **kwargs):
        if any("query-gpu=index,memory.used" in str(item) for item in command):
            self.snapshot_count += 1
        if any("query-compute-apps" in str(item) for item in command):
            process = "GPU-uuid-0, 123, 4096\n" if self.snapshot_count >= 2 else ""
            return SimpleNamespace(stdout=process, returncode=0)
        return super().run(command, **kwargs)


class PostLeaseProcessSSH(FakeSSH):
    """Expose a foreign process only after the worker lease is published."""

    def __init__(self):
        super().__init__(
            "0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
            gpu_uuids="0, GPU-uuid-0\n1, GPU-uuid-1\n2, GPU-uuid-2\n3, GPU-uuid-3\n",
        )
        self.snapshot_count = 0

    def run(self, command, **kwargs):
        if any("query-gpu=index,memory.used" in str(item) for item in command):
            self.snapshot_count += 1
        if any("query-compute-apps" in str(item) for item in command):
            process = "GPU-uuid-0, 123, 4096\\n" if self.snapshot_count >= 3 else ""
            return SimpleNamespace(stdout=process, returncode=0)
        return super().run(command, **kwargs)


class LeaseSSH(FakeSSH):
    def __init__(self, gpu_memory):
        super().__init__(gpu_memory)
        self.config = SimpleNamespace(resolved_campaign_root="/srv/harness/campaign")
        self.files = {}
        self.live_owner = True

    def run(self, command, **kwargs):
        if tuple(command[:2]) == ("test", "-e"):
            return SimpleNamespace(stdout="", returncode=0 if command[2] in self.files else 1)
        if command and command[0] == "ps":
            if self.live_owner:
                return SimpleNamespace(stdout="python tools/run_overnight_controller.py", returncode=0)
            return SimpleNamespace(stdout="", returncode=1)
        return super().run(command, **kwargs)

    def write_json(self, path, value):
        self.files[path] = dict(value)

    def read_json(self, path):
        return dict(self.files[path])

    def remove_file(self, path):
        self.files.pop(path, None)
        return {"status": "deleted"}


def request(**changes):
    value = {
        "gpu_count": 4,
        "distributed": True,
        "exclusive": True,
        "evaluation_workers": 1,
        "on_unavailable": "wait",
    }
    value.update(changes)
    return value


def test_scheduler_maps_all_four_gpus_without_touching_processes():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 1024\n1, 0\n2, 0\n3, 0\n"))
    decision = scheduler.acquire(request(gpu_count=3, distributed=False, exclusive=False))
    assert decision.status == "ready"
    assert decision.allocated_gpus == (1, 2, 3)


def test_scheduler_exposes_advisory_power_and_utilization_sample():
    scheduler = RemoteResourceScheduler(
        FakeSSH(
            "0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
            gpu_telemetry="0, 287.5, 96\n1, 141.0, 42\n2, 299.0, 99\n3, 0.0, 0\n",
        )
    )

    scheduler.snapshot()

    assert scheduler.last_gpu_telemetry[0]["power_w"] == pytest.approx(287.5)
    assert scheduler.last_gpu_telemetry[1]["utilization_gpu_pct"] == pytest.approx(42.0)


def test_scheduler_waits_for_exclusive_request_when_unrelated_job_is_running():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 0\n1, 0\n2, 0\n3, 0\n", "GPU-1, 123, 4096\n"))
    decision = scheduler.acquire(request())
    assert decision.status == "wait"
    assert decision.allocated_gpus == ()
    assert scheduler.last_compute_gpu_indices is None
    assert "without stopping" in decision.reason


def test_scheduler_elastically_allocates_three_of_four_gpus():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 1024\n1, 0\n2, 0\n3, 0\n"))
    decision = scheduler.acquire(
        request(
            gpu_count=4,
            min_gpu_count=2,
            max_gpu_count=4,
            elastic=True,
            distributed=True,
            exclusive=False,
        )
    )
    assert decision.status == "ready"
    assert decision.allocated_gpus == (1, 2, 3)
    assert decision.actual_gpu_count == 3


def test_scheduler_excludes_gpu_with_unrelated_compute_process_for_nonexclusive_worker():
    scheduler = RemoteResourceScheduler(
        FakeSSH(
            "0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
            "GPU-uuid-0, 123, 4096\n",
            "0, GPU-uuid-0\n1, GPU-uuid-1\n2, GPU-uuid-2\n3, GPU-uuid-3\n",
        ),
        min_free_memory_mb=26000,
    )

    decision = scheduler.acquire(request(gpu_count=3, distributed=True, exclusive=False))

    assert decision.status == "ready"
    assert decision.allocated_gpus == (1, 2, 3)
    assert scheduler.last_compute_gpu_indices == (0,)


def test_scheduler_rechecks_before_publishing_lease_when_gpu_becomes_busy():
    scheduler = RemoteResourceScheduler(
        ChangingSnapshotSSH(),
        min_free_memory_mb=26000,
        allocation_recheck_s=0.01,
    )

    decision = scheduler.acquire(
        request(
            gpu_count=4,
            min_gpu_count=4,
            max_gpu_count=4,
            distributed=True,
            exclusive=False,
        )
    )

    assert decision.status == "wait"
    assert decision.allocated_gpus == ()
    assert "after allocation recheck" in decision.reason


def test_scheduler_rechecks_selected_gpu_after_publishing_lease():
    scheduler = RemoteResourceScheduler(
        PostLeaseProcessSSH(),
        min_free_memory_mb=26000,
        allocation_recheck_s=0.01,
    )

    decision = scheduler.acquire(
        request(
            gpu_count=4,
            min_gpu_count=4,
            max_gpu_count=4,
            distributed=True,
            exclusive=False,
        )
    )

    assert decision.status == "wait"
    assert decision.allocated_gpus == ()
    assert "after lease publication" in decision.reason


def test_scheduler_fails_closed_when_compute_process_gpu_cannot_be_mapped():
    scheduler = RemoteResourceScheduler(
        FakeSSH(
            "0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
            "GPU-unknown, 123, 4096\n",
            "0, GPU-uuid-0\n1, GPU-uuid-1\n2, GPU-uuid-2\n3, GPU-uuid-3\n",
        ),
        min_free_memory_mb=26000,
    )

    decision = scheduler.acquire(request(gpu_count=2, distributed=True, exclusive=False))

    assert decision.status == "wait"
    assert decision.allocated_gpus == ()
    assert scheduler.last_compute_gpu_indices is None


def test_scheduler_fails_closed_when_compute_process_query_fails():
    scheduler = RemoteResourceScheduler(
        FakeSSH(
            "0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
            process_returncode=1,
        ),
        min_free_memory_mb=26000,
    )

    decision = scheduler.acquire(request(gpu_count=2, distributed=True, exclusive=False))

    assert decision.status == "wait"
    assert decision.allocated_gpus == ()
    assert scheduler.last_compute_gpu_indices is None


def test_scheduler_waits_when_elastic_capacity_is_below_minimum():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 1024\n1, 1024\n2, 1024\n3, 0\n"))
    decision = scheduler.acquire(
        request(
            gpu_count=4,
            min_gpu_count=2,
            max_gpu_count=4,
            elastic=True,
            distributed=True,
            exclusive=False,
        )
    )
    assert decision.status == "wait"
    assert decision.allocated_gpus == ()
    assert decision.actual_gpu_count == 0


def test_scheduler_keeps_exact_request_exact():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 0\n1, 0\n2, 0\n3, 1024\n"))
    decision = scheduler.acquire(request(gpu_count=2, distributed=True, exclusive=False))
    assert decision.status == "ready"
    assert decision.allocated_gpus == (0, 1)
    assert decision.elastic is False


def test_scheduler_rejects_invalid_elastic_range():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 0\n1, 0\n2, 0\n3, 0\n"))
    with pytest.raises(ValueError, match="min_gpu_count"):
        scheduler.acquire(request(gpu_count=4, min_gpu_count=3, max_gpu_count=2, elastic=True))


def test_scheduler_can_reserve_and_release_one_gpu():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n"))
    scheduler.reserve_gpu(0)
    assert 0 in scheduler.reserved_gpu_indices
    scheduler.release_gpu(0)
    assert 0 not in scheduler.reserved_gpu_indices


def test_scheduler_waterline_uses_total_and_used_memory():
    scheduler = RemoteResourceScheduler(
        FakeSSH("0, 20000, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n"),
        min_free_memory_mb=26000,
    )
    memory, _ = scheduler.snapshot()
    assert scheduler.meets_memory_waterline(0, memory) is True
    assert scheduler.meets_memory_waterline(9, memory) is False


def test_scheduler_reservation_rejects_out_of_range_gpu():
    scheduler = RemoteResourceScheduler(FakeSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n"))
    with pytest.raises(ValueError, match="within the GPU range"):
        scheduler.reserve_gpu(4)


def test_scheduler_publishes_and_releases_shared_worker_lease():
    remote = LeaseSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n")
    scheduler = RemoteResourceScheduler(remote, min_free_memory_mb=26000)
    decision = scheduler.acquire(
        request(
            gpu_count=4,
            min_gpu_count=2,
            max_gpu_count=4,
            elastic=True,
            distributed=True,
            exclusive=False,
        )
    )

    lease_path = "/srv/harness/campaign/.h3-worker-gpu-lease.json"
    assert decision.status == "ready"
    assert decision.allocated_gpus == (0, 1, 2, 3)
    assert scheduler.lease_active is True
    assert remote.files[lease_path]["allocated_gpus"] == [0, 1, 2, 3]

    scheduler.release()
    assert scheduler.lease_active is False
    assert lease_path not in remote.files


def test_scheduler_excludes_live_controller_lease_gpu():
    remote = LeaseSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n")
    controller_path = "/srv/harness/campaign/.controller-gpu-lease.json"
    remote.files[controller_path] = {
        "state": "allocated_for_controller",
        "owner_pid": 123,
        "created_at": time.time(),
        "expires_at": time.time() + 60.0,
        "allocated_gpus": [1],
    }
    scheduler = RemoteResourceScheduler(
        remote,
        min_free_memory_mb=26000,
        controller_lease_path=controller_path,
    )

    decision = scheduler.acquire(
        request(
            gpu_count=2,
            min_gpu_count=2,
            max_gpu_count=2,
            elastic=False,
            exclusive=False,
        )
    )

    assert decision.status == "ready"
    assert decision.allocated_gpus == (0, 2)


def test_scheduler_ignores_controller_lease_after_exact_owner_exits():
    remote = LeaseSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n")
    controller_path = "/srv/harness/campaign/.controller-gpu-lease.json"
    remote.files[controller_path] = {
        "state": "allocated_for_controller",
        "owner_pid": 123,
        "created_at": time.time(),
        "expires_at": time.time() + 60.0,
        "allocated_gpus": [1],
    }
    remote.live_owner = False
    scheduler = RemoteResourceScheduler(
        remote,
        min_free_memory_mb=26000,
        controller_lease_path=controller_path,
    )

    decision = scheduler.acquire(
        request(
            gpu_count=2,
            min_gpu_count=2,
            max_gpu_count=2,
            elastic=False,
            exclusive=False,
        )
    )

    assert decision.status == "ready"
    assert decision.allocated_gpus == (0, 1)


def test_scheduler_waits_for_another_live_shared_worker_lease():
    remote = LeaseSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n")
    first = RemoteResourceScheduler(remote, min_free_memory_mb=26000)
    second = RemoteResourceScheduler(remote, min_free_memory_mb=26000)
    assert first.acquire(request(gpu_count=2, distributed=True, exclusive=False)).status == "ready"

    decision = second.acquire(request(gpu_count=2, distributed=True, exclusive=False))
    assert decision.status == "wait"
    assert decision.allocated_gpus == ()
    assert "live worker GPU lease" in decision.reason


def test_scheduler_reclaims_live_ttl_lease_when_owner_process_is_gone():
    remote = LeaseSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n")
    lease_path = "/srv/harness/campaign/.h3-worker-gpu-lease.json"
    remote.files[lease_path] = {
        "state": "allocated_for_worker",
        "token": "old-token",
        "owner_pid": 987654,
        "created_at": time.time() - 10.0,
        "expires_at": time.time() + 3600.0,
        "allocated_gpus": [0, 2, 3],
    }
    remote.live_owner = False
    scheduler = RemoteResourceScheduler(remote, min_free_memory_mb=26000)

    decision = scheduler.acquire(request(gpu_count=2, distributed=True, exclusive=False))

    assert decision.status == "ready"
    assert "reclaimed stale" in decision.reason
    assert remote.files[lease_path]["token"] != "old-token"


def test_scheduler_does_not_reclaim_dead_owner_lease_over_live_worker_process():
    remote = LeaseSSH("0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n")
    remote.gpu_uuids = "0, GPU-uuid-0\n1, GPU-uuid-1\n2, GPU-uuid-2\n3, GPU-uuid-3\n"
    remote.processes = "GPU-uuid-2, 2468, 4096\n"
    lease_path = "/srv/harness/campaign/.h3-worker-gpu-lease.json"
    remote.files[lease_path] = {
        "state": "allocated_for_worker",
        "token": "old-token",
        "owner_pid": 987654,
        "created_at": time.time() - 10.0,
        "expires_at": time.time() + 3600.0,
        "allocated_gpus": [0, 2, 3],
    }
    remote.live_owner = False
    scheduler = RemoteResourceScheduler(remote, min_free_memory_mb=26000)

    decision = scheduler.acquire(request(gpu_count=2, distributed=True, exclusive=False))

    assert decision.status == "wait"
    assert "live compute process" in decision.reason
    assert remote.files[lease_path]["token"] == "old-token"
