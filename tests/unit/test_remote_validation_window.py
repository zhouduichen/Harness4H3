import os
import subprocess
from pathlib import Path


SCRIPT = Path("tools/remote-validation-window.sh")


def test_validation_window_default_is_read_only(tmp_path):
    repo = tmp_path / "repo"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    marker = tmp_path / "start-called"
    service = tools / "remote-campaign-service.sh"
    service.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = status ]; then echo 'campaign=/tmp/output pid=none state=paused'; else touch -- '" + str(marker) + "'; fi\n",
        encoding="utf-8",
    )
    service.chmod(0o755)
    env = os.environ.copy()
    env.update({"REMOTE_CAMPAIGN_REPO_ROOT": str(repo), "REMOTE_CAMPAIGN_PYTHON": "/usr/bin/python3"})

    result = subprocess.run(["bash", str(SCRIPT.resolve())], env=env, capture_output=True, text=True, check=True)

    assert "state=paused" in result.stdout
    assert not marker.exists()


def test_validation_window_has_explicit_start_and_fail_closed_guards():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--start" in source
    assert "operator-paused" in source
    assert "nvidia-smi" in source
    assert "REMOTE_CAMPAIGN_MAX_ITERATIONS" in source
    assert 'service_cmd pause' in source
    assert "pkill" not in source
    assert "killall" not in source


def test_validation_window_help_is_available_without_remote_dependencies():
    result = subprocess.run(["bash", str(SCRIPT.resolve()), "--help"], capture_output=True, text=True, check=True)

    assert "bounded campaign window" in result.stdout
    assert "--max-iterations" in result.stdout


def test_validation_window_refuses_to_override_operator_pause(tmp_path):
    repo = tmp_path / "repo"
    tools = repo / "tools"
    output = repo / "var" / "campaign"
    tools.mkdir(parents=True)
    output.mkdir(parents=True)
    marker = tmp_path / "start-called"
    (output / ".operator-paused").touch()
    service = tools / "remote-campaign-service.sh"
    service.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = start ]; then touch -- '" + str(marker) + "'; fi\n"
        "echo 'campaign=/tmp/output pid=none state=paused'\n",
        encoding="utf-8",
    )
    service.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "REMOTE_CAMPAIGN_REPO_ROOT": str(repo),
            "REMOTE_CAMPAIGN_PYTHON": "/usr/bin/python3",
            "REMOTE_CAMPAIGN_OUTPUT": str(output),
        }
    )

    result = subprocess.run(["bash", str(SCRIPT.resolve()), "--start"], env=env, capture_output=True, text=True)

    assert result.returncode == 3
    assert "operator-paused" in result.stderr
    assert not marker.exists()
