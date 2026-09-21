from types import SimpleNamespace

from harness4h3.remote.comfyui_lease import ComfyUILeaseManager


class ProcessSSH:
    def __init__(self):
        self.commands = []
        self.alive = {4321}

    def run(self, command, **kwargs):
        command = tuple(command)
        self.commands.append(command)
        if command[:2] == ("kill", "-0"):
            return SimpleNamespace(returncode=0 if int(command[2]) in self.alive else 1, stdout="", stderr="")
        if command[:2] in (("kill", "-TERM"), ("kill", "-KILL")):
            self.alive.discard(int(command[2]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout='{"queue_running": [], "queue_pending": []}', stderr="")


class Scheduler:
    gpu_count = 4
    reserved_gpu_indices = ()


def test_stop_owned_comfyui_pid_is_scoped_to_recorded_process(monkeypatch):
    remote = ProcessSSH()
    lease = ComfyUILeaseManager(remote, Scheduler(), port=8188, release_wait_s=1)
    lease.set_owned_process(4321)

    monkeypatch.setattr("harness4h3.remote.comfyui_lease.time.sleep", lambda _seconds: None)
    result = lease.stop_owned_process()

    assert result["status"] == "stopped"
    assert ("kill", "-TERM", "4321") in remote.commands
    assert remote.alive == set()


def test_process_without_owned_pid_is_not_killed():
    remote = ProcessSSH()
    lease = ComfyUILeaseManager(remote, Scheduler(), port=8188)

    result = lease.stop_owned_process()

    assert result["status"] == "not_owned"
    assert not any(command and command[0] == "kill" for command in remote.commands)
