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


def test_ensure_model_link_replaces_only_a_stale_validated_symlink():
    from harness4h3.remote.ssh import RemoteConfig, SSHClient

    calls = []

    def runner(argv, **kwargs):
        command = " ".join(argv)
        calls.append(command)
        if "readlink -f /srv/harness/new.safetensors" in command:
            return SimpleNamespace(returncode=0, stdout="/srv/harness/new.safetensors\n", stderr="")
        if "test -e /srv/comfy/models/diffusion_models/M0005.safetensors" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "test -L /srv/comfy/models/diffusion_models/M0005.safetensors" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "readlink -f /srv/comfy/models/diffusion_models/M0005.safetensors" in command:
            return SimpleNamespace(returncode=0, stdout="/srv/models/old.safetensors\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    client = SSHClient(
        RemoteConfig(
            "host",
            "/srv/harness",
            "/srv/models",
            "/srv/comfy",
            deployment_dir="/srv/comfy/models/diffusion_models",
        ),
        runner=runner,
    )
    target = client.ensure_model_link("/srv/harness/new.safetensors", "M0005")
    assert target == "/srv/comfy/models/diffusion_models/M0005.safetensors"
    assert client.last_model_link_action == "replaced_stale_symlink"
    assert any("ln -sfn /srv/harness/new.safetensors" in command for command in calls)


def test_ensure_model_link_refuses_a_regular_deployment_file():
    from harness4h3.remote.ssh import RemoteConfig, RemoteLinkConflict, SSHClient

    def runner(argv, **kwargs):
        command = " ".join(argv)
        if "readlink -f /srv/harness/new.safetensors" in command:
            return SimpleNamespace(returncode=0, stdout="/srv/harness/new.safetensors\n", stderr="")
        if "test -e /srv/comfy/models/diffusion_models/M0005.safetensors" in command:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "test -L /srv/comfy/models/diffusion_models/M0005.safetensors" in command:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    client = SSHClient(
        RemoteConfig(
            "host",
            "/srv/harness",
            "/srv/models",
            "/srv/comfy",
            deployment_dir="/srv/comfy/models/diffusion_models",
        ),
        runner=runner,
    )
    with pytest.raises(RemoteLinkConflict):
        client.ensure_model_link("/srv/harness/new.safetensors", "M0005")


def test_remote_config_rejects_campaign_escape():
    from harness4h3.remote.ssh import RemoteConfig

    with pytest.raises(ValueError):
        RemoteConfig("host", "/srv/harness", "/srv/models", "/srv/comfy", campaign_root="/tmp/campaign")


def test_local_command_client_runs_without_ssh_and_preserves_argument_boundaries():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    from harness4h3.remote.ssh import LocalCommandClient, RemoteConfig

    client = LocalCommandClient(
        RemoteConfig("unused", "/srv/harness", "/srv/models", "/srv/comfy"),
        runner=runner,
    )
    result = client.run(("python3", "-c", "print('hello world')", "/srv/models/a file"))
    assert result.stdout == "ok"
    assert calls[0][0] == ("python3", "-c", "print('hello world')", "/srv/models/a file")
    assert calls[0][1]["shell"] is False
    assert client.is_local is True


def test_local_transport_does_not_create_port_forward_processes():
    from harness4h3.remote.ssh import ComfyUITunnel, LocalCommandClient, RemoteConfig, RemotePortForward

    client = LocalCommandClient(RemoteConfig("unused", "/srv/harness", "/srv/models", "/srv/comfy", comfyui_port=8188))
    with RemotePortForward(client, 8000) as controller_tunnel:
        assert controller_tunnel.base_url == "http://127.0.0.1:8000"
        assert controller_tunnel.process is None
    with ComfyUITunnel(client) as comfyui_tunnel:
        assert comfyui_tunnel.base_url == "http://127.0.0.1:8188"
        assert comfyui_tunnel.process is None
