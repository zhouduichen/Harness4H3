import json
from types import SimpleNamespace
from pathlib import Path

from harness4h3.remote.config import load_remote_campaign_config
from harness4h3.remote.ssh import SSHClient
from harness4h3.target.profile import load_target_profile
from research.experiments.remote_h3_closed_loop import RemoteCampaign


class _LauncherSSH(SSHClient):
    def __init__(self, config):
        super().__init__(config)
        self.files = {
            ".controller-vllm.pid": "4321\n",
        }
        self.commands = []
        self.release_payload = None

    def _name(self, path):
        return str(path).rsplit("/", 1)[-1]

    def write_json(self, path, value):
        self.files[self._name(path)] = json.dumps(value)
        if self._name(path) == ".controller-release.json":
            self.release_payload = dict(value)
            # Model the owned launcher consuming the request and stopping its
            # child before the next campaign poll.
            self.files.pop(".controller-vllm.pid", None)

    def remove_file(self, path):
        self.files.pop(self._name(path), None)
        return {"status": "deleted"}

    def run(self, command, **kwargs):
        command = tuple(command)
        self.commands.append(command)
        if command[:2] == ("test", "-e"):
            return SimpleNamespace(returncode=0 if self._name(command[2]) in self.files else 1, stdout="", stderr="")
        if command[:1] == ("cat",):
            return SimpleNamespace(returncode=0, stdout=self.files[self._name(command[2])], stderr="")
        if command[:2] == ("ps", "-p"):
            return SimpleNamespace(returncode=0, stdout="vllm serve qwen3.5-controller\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_release_marker_only_targets_campaign_owned_controller(tmp_path):
    config = load_remote_campaign_config("configs/remote-l40-h3.yaml")
    remote = _LauncherSSH(config.remote)
    campaign = RemoteCampaign(
        config,
        ssh=remote,
        target=load_target_profile(config.runtime.target_path),
        output_root=tmp_path,
    )

    result = campaign._request_controller_release("unit_test", wait_s=0.0, full_card_training=True)

    assert result["status"] == "released"
    assert remote.release_payload["full_card_training"] is True
    assert ".controller-release.json" not in remote.files
    assert not any(command[:1] == ("pgrep",) for command in remote.commands)


def test_controller_launcher_does_not_wait_on_uninterruptible_orphan():
    source = Path("tools/controller-wait-launch.sh").read_text(encoding="utf-8")

    assert "ORPHAN_PIDS" in source
    assert "polling exact orphan without wait" in source
    assert "orphan_gpu_quiescent" in source
    assert "clear_controller_lease_if_owner" in source
    assert "if child_is_alive; then\n        # ``terminate_owned_child``" in source


def test_controller_launcher_publishes_and_honors_lane_handoff_markers():
    source = Path("tools/controller-wait-launch.sh").read_text(encoding="utf-8")

    assert "CONTROLLER_LEASE_FILE" in source
    assert "write_controller_lease \"launching\"" in source
    assert "allocated_for_controller" in source
    assert "CONTROLLER_HOLD_FILE" in source
    assert "handoff_hold_active" in source
    assert "campaign holds Controller lane for worker handoff" in source
    assert "capping Controller TP at" in source


def test_controller_launcher_falls_back_to_quiet_subset_when_larger_group_is_busy():
    source = Path("tools/controller-wait-launch.sh").read_text(encoding="utf-8")

    assert "query-compute-apps=pid" in source
    assert "candidate_busy_gpus" in source
    assert "select_quiet_candidate" in source
    assert "refreshed=$(select_quiet_candidate)" in source


def test_controller_launcher_preserves_two_gpu_evaluation_overlap_slot():
    source = Path("tools/controller-wait-launch.sh").read_text(encoding="utf-8")

    assert "EVALUATION_MAX_TENSOR_PARALLEL=${CONTROLLER_EVALUATION_MAX_TENSOR_PARALLEL:-1}" in source
    assert "capping Controller TP at" in source
    assert 'effective_max_tp" -gt "$EVALUATION_MAX_TENSOR_PARALLEL' in source
    assert "evaluation overlap requires Controller TP" in source
    assert "worker lease is active; restarting owned TP" in source
    assert "deadlock a worker whose minimum is two GPUs" in source


def test_controller_launcher_pauses_own_launch_for_explicit_external_controller():
    source = Path("tools/controller-wait-launch.sh").read_text(encoding="utf-8")

    assert "EXTERNAL_CONTROLLER_PORTS=${CONTROLLER_EXTERNAL_PORTS:-8001}" in source
    assert "external_controller_api_ready" in source
    assert "own vLLM launch is paused" in source
