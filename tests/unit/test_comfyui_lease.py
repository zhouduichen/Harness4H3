from types import SimpleNamespace

from harness4h3.remote.comfyui_lease import ComfyUILeaseManager
from harness4h3.remote.scheduler import RemoteResourceScheduler
from harness4h3.remote.ssh import RemoteConfig


class LeaseSSH:
    def __init__(
        self,
        queue,
        free_response='{"ok": true}',
        memory="0, 0, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
    ):
        self.queue = queue
        self.free_response = free_response
        self.memory = memory
        self.commands = []

    def run(self, command, **kwargs):
        command = tuple(command)
        self.commands.append(command)
        if any("/queue" in item for item in command):
            return SimpleNamespace(stdout=self.queue, stderr="", returncode=0)
        if any("/free" in item for item in command):
            return SimpleNamespace(stdout=self.free_response, stderr="", returncode=0)
        if any("query-compute-apps" in item for item in command):
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return SimpleNamespace(stdout=self.memory, stderr="", returncode=0)


class MarkerLeaseSSH(LeaseSSH):
    def __init__(self):
        super().__init__('{"queue_running": [], "queue_pending": []}')
        self.config = RemoteConfig(
            "fake",
            "/srv/harness",
            "/srv/models",
            "/srv/comfy",
            campaign_root="/srv/harness/campaign",
        )
        self.markers = {}

    def write_json(self, path, value):
        self.markers[path] = value

    def remove_file(self, path):
        self.markers.pop(path, None)
        return {"status": "deleted"}


def manager(remote):
    scheduler = RemoteResourceScheduler(remote, min_free_memory_mb=26000, reserved_gpu_indices=(0,))
    return ComfyUILeaseManager(remote, scheduler, port=8188), scheduler


def test_release_if_idle_calls_free_and_releases_gpu0():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}')
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.success is True
    assert result.state == "released_for_other_work"
    assert 0 not in scheduler.reserved_gpu_indices
    assert any(any("/free" in item for item in command) for command in remote.commands)


def test_release_if_idle_is_idempotent_after_success():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}')
    lease, scheduler = manager(remote)
    first = lease.release_if_idle()
    second = lease.release_if_idle()
    assert second == first
    assert sum(any("/free" in item for item in command) for command in remote.commands) == 1
    assert 0 not in scheduler.reserved_gpu_indices


def test_release_refuses_active_queue_without_calling_free():
    remote = LeaseSSH('{"queue_running": [{"prompt": "busy"}], "queue_pending": []}')
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.success is False
    assert result.state == "reserved_for_benchmark"
    assert "queue_active" in result.reason
    assert 0 in scheduler.reserved_gpu_indices
    assert not any(any("/free" in item for item in command) for command in remote.commands)


def test_release_api_failure_is_fail_closed():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}', free_response="not-json")
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.success is False
    assert result.state == "release_failed"
    assert 0 in scheduler.reserved_gpu_indices


def test_release_accepts_empty_success_body_from_comfyui():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}', free_response="")
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.success is True
    assert result.state == "released_for_other_work"
    assert 0 not in scheduler.reserved_gpu_indices


def test_prepare_is_idempotent_and_reserves_gpu0():
    remote = LeaseSSH('{"queue_running": [], "queue_pending": []}')
    lease, scheduler = manager(remote)
    scheduler.release_gpu(0)
    payload = lease.prepare_for_benchmark()
    assert payload["state"] == "reserved_for_benchmark"
    assert 0 in scheduler.reserved_gpu_indices


def test_campaign_lease_marker_blocks_controller_until_release():
    remote = MarkerLeaseSSH()
    scheduler = RemoteResourceScheduler(remote, min_free_memory_mb=26000, reserved_gpu_indices=(0,))
    lease = ComfyUILeaseManager(remote, scheduler, port=8188)

    payload = lease.prepare_for_benchmark()
    assert payload["lease_path"] == "/srv/harness/campaign/.comfyui-gpu-lease.json"
    assert payload["lease_path"] in remote.markers

    result = lease.release_if_idle()

    assert result.success is True
    assert payload["lease_path"] not in remote.markers
    assert result.response["lease_marker"]["status"] == "deleted"


def test_release_malformed_queue_is_fail_closed():
    remote = LeaseSSH("not-json")
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.state == "release_failed"
    assert 0 in scheduler.reserved_gpu_indices


def test_release_keeps_gpu_reserved_when_memory_waterline_fails():
    remote = LeaseSSH(
        '{"queue_running": [], "queue_pending": []}',
        memory="0, 30000, 46080\n1, 0, 46080\n2, 0, 46080\n3, 0, 46080\n",
    )
    lease, scheduler = manager(remote)
    result = lease.release_if_idle()
    assert result.reason == "post_release_memory_below_waterline"
    assert 0 in scheduler.reserved_gpu_indices
