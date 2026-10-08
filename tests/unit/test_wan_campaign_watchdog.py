import json
import os
import subprocess
import sys
from pathlib import Path


WATCHDOG = Path("tools/wan-vllm-campaign-watchdog.sh")


def _env(tmp_path: Path, runner: Path) -> dict[str, str]:
    output = tmp_path / "campaign"
    output.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("wan_loop: {}\n", encoding="utf-8")
    return {
        **os.environ,
        "WAN_CAMPAIGN_PYTHON": sys.executable,
        "WAN_CAMPAIGN_SCRIPT": str(runner),
        "WAN_CAMPAIGN_CONFIG": str(config),
        "WAN_CAMPAIGN_OUTPUT": str(output),
        "WAN_CAMPAIGN_MAX_ROUNDS": "2",
        "WAN_CAMPAIGN_RESTART_LIMIT": "3",
        "WAN_CAMPAIGN_BACKOFF_SECONDS": "0",
        "WAN_CAMPAIGN_STOP_GRACE_SECONDS": "1",
        "WAN_CAMPAIGN_STOP_FILE": str(output / ".stop"),
        "WAN_CAMPAIGN_LOCK_FILE": str(output / ".lock"),
        "WAN_CAMPAIGN_CHILD_PID_FILE": str(output / ".child.pid"),
        "WAN_CAMPAIGN_LOG": str(output / "watchdog.log"),
    }


def test_watchdog_restarts_only_when_no_final_result(tmp_path: Path):
    counter = tmp_path / "count"
    result = tmp_path / "campaign" / "campaign-result.json"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "from pathlib import Path\n"
        "import json, os\n"
        "counter = Path(os.environ['FAKE_COUNTER'])\n"
        "result = Path(os.environ['FAKE_RESULT'])\n"
        "count = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(count + 1))\n"
        "if count == 0: raise SystemExit(17)\n"
        "result.write_text(json.dumps({'status': 'success'}))\n",
        encoding="utf-8",
    )
    environment = _env(tmp_path, runner)
    environment["FAKE_COUNTER"] = str(counter)
    environment["FAKE_RESULT"] = str(result)
    completed = subprocess.run(["bash", str(WATCHDOG)], env=environment, check=False, timeout=10)
    assert completed.returncode == 0
    assert counter.read_text(encoding="utf-8") == "2"
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "success"


def test_watchdog_does_not_launch_when_operator_stop_exists(tmp_path: Path):
    counter = tmp_path / "count"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "from pathlib import Path; Path(%r).write_text('ran')\n" % str(counter),
        encoding="utf-8",
    )
    environment = _env(tmp_path, runner)
    Path(environment["WAN_CAMPAIGN_STOP_FILE"]).write_text("stop\n", encoding="utf-8")
    completed = subprocess.run(["bash", str(WATCHDOG)], env=environment, check=False, timeout=10)
    assert completed.returncode == 0
    assert not counter.exists()
