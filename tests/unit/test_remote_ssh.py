from types import SimpleNamespace

import pytest


def test_remote_path_must_stay_under_configured_root():
    from harness4h3.remote.ssh import RemoteConfig, RemotePathError, SSHClient

    client = SSHClient(
        RemoteConfig("host", "/srv/harness", "/srv/models", "/srv/comfy"),
        runner=lambda *a, **k: None,
    )
    with pytest.raises(RemotePathError):
        client.read_text("/srv/harness/../etc/passwd")


def test_command_uses_argument_quoting_and_never_shell_true():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    from harness4h3.remote.ssh import RemoteConfig, SSHClient

    SSHClient(RemoteConfig("host", "/srv/harness", "/srv/models", "/srv/comfy"), runner=runner).run(
        ("sha256sum", "/srv/harness/a file.json")
    )
    assert calls[0][1]["shell"] is False
    assert "a file.json" in " ".join(calls[0][0])


def test_existing_link_pointing_elsewhere_is_refused():
    from harness4h3.remote.ssh import RemoteLinkConflict, link_action

    assert link_action("/models/M0006.safetensors", "/models/M0006.safetensors") == "keep"
    with pytest.raises(RemoteLinkConflict):
        link_action("/models/other.safetensors", "/models/M0006.safetensors")


def test_remote_config_rejects_campaign_escape():
    from harness4h3.remote.ssh import RemoteConfig

    with pytest.raises(ValueError):
        RemoteConfig("host", "/srv/harness", "/srv/models", "/srv/comfy", campaign_root="/tmp/campaign")
