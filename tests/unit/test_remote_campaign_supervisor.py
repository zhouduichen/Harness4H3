from pathlib import Path
import os
import shutil
import subprocess
import textwrap
import time


def test_campaign_supervisor_runs_runner_and_releases_owned_controller():
    source = Path("tools/remote-campaign-supervisor.sh").read_text(encoding="utf-8")

    assert "run_overnight_controller.py" in source
    assert 'bash "$SERVICE" ensure-controller' in source
    assert 'bash "$SERVICE" cleanup-controller' in source
    assert "trap cleanup EXIT INT TERM" in source
    assert "kill_child_tree" in source


def test_campaign_supervisor_has_no_broad_process_matcher():
    source = Path("tools/remote-campaign-supervisor.sh").read_text(encoding="utf-8")

    assert "pkill" not in source
    assert "killall" not in source
    assert "pgrep -f" not in source


def test_campaign_supervisor_passes_optional_iteration_bound():
    source = Path("tools/remote-campaign-supervisor.sh").read_text(encoding="utf-8")

    assert 'MAX_ITERATIONS="${REMOTE_CAMPAIGN_MAX_ITERATIONS:-512}"' in source
    assert '--max-iterations "$MAX_ITERATIONS"' in source
    assert "must be a positive integer" in source


def test_campaign_service_exports_iteration_bound_to_supervisor():
    source = Path("tools/remote-campaign-service.sh").read_text(encoding="utf-8")

    assert 'CAMPAIGN_MAX_ITERATIONS="${REMOTE_CAMPAIGN_MAX_ITERATIONS:-512}"' in source
    assert 'REMOTE_CAMPAIGN_MAX_ITERATIONS="$CAMPAIGN_MAX_ITERATIONS"' in source


def test_campaign_service_reaps_owned_controller_after_runner_finishes(tmp_path):
    repo = tmp_path / "repo"
    tools = repo / "tools"
    output = repo / "var" / "campaign"
    tools.mkdir(parents=True)
    output.mkdir(parents=True)
    for name in ("remote-campaign-service.sh", "remote-campaign-supervisor.sh"):
        shutil.copy2(Path("tools") / name, tools / name)

    marker = tmp_path / "launcher-running"
    launcher = tools / "controller-wait-launch.sh"
    launcher.write_text(
        textwrap.dedent(
            f"""
            #!/usr/bin/env bash
            set -eu
            trap 'rm -f -- {marker}; exit 0' TERM INT EXIT
            touch -- {marker}
            while true; do sleep 0.05; done
            """
        ),
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    runner = tools / "run_overnight_controller.py"
    runner.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            from pathlib import Path
            import time

            parser = argparse.ArgumentParser()
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            time.sleep(0.1)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "overnight-result.json").write_text(
                json.dumps({"status": "target_satisfied"}) + "\\n",
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "REMOTE_CAMPAIGN_REPO_ROOT": str(repo),
            "REMOTE_CAMPAIGN_PYTHON": "/usr/bin/python3",
            "REMOTE_CAMPAIGN_CONFIG": str(repo / "config.yaml"),
            "REMOTE_CAMPAIGN_OUTPUT": str(output),
            "REMOTE_CAMPAIGN_MANAGE_CONTROLLER": "1",
            "REMOTE_CAMPAIGN_RESOURCE_POLL_INTERVAL_S": "0",
        }
    )
    service = tools / "remote-campaign-service.sh"
    subprocess.run(["bash", str(service), "start"], env=env, check=True, capture_output=True, text=True)

    launcher_pid_file = repo / "work" / "remote-h3-controller-20260914" / ".controller-launcher.pid"
    deadline = time.monotonic() + 5.0
    try:
        while time.monotonic() < deadline:
            if (
                (output / "overnight-result.json").exists()
                and not marker.exists()
                and not launcher_pid_file.exists()
            ):
                break
            time.sleep(0.05)
        assert (output / "overnight-result.json").exists()
        assert not marker.exists()
        assert not launcher_pid_file.exists()
    finally:
        subprocess.run(["bash", str(service), "stop"], env=env, check=False, capture_output=True, text=True)
