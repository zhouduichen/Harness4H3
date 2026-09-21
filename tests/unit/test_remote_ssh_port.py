from types import SimpleNamespace


def test_ssh_client_uses_configured_nonstandard_port():
    from harness4h3.remote.ssh import RemoteConfig, SSHClient

    calls = []

    def runner(argv, **kwargs):
        calls.append(tuple(argv))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    client = SSHClient(
        RemoteConfig("222.29.98.132", "/srv/harness", "/srv/models", "/srv/comfy", ssh_port=30902),
        runner=runner,
    )
    client.run(("true",))

    assert "-p" in calls[0]
    assert calls[0][calls[0].index("-p") + 1] == "30902"


def test_rsi_remote_config_uses_reachable_server_endpoint():
    from harness4h3.remote.config import load_remote_campaign_config

    config = load_remote_campaign_config("configs/remote-l40-h3-rsi-overnight.yaml")

    assert config.remote.host == "Jiayu-intern"
    assert config.remote.ssh_port == 30902
