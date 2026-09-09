from __future__ import annotations

import json
import sys

import pytest

from harness4h3.executor.local import ExecutorValidationError, LocalProcessExecutor


def test_local_executor_isolates_cwd_environment_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_SECRET", "must-not-leak")
    executor = LocalProcessExecutor(allowed_env=("VISIBLE_VALUE",))
    script = (
        "import json,os; "
        "print(json.dumps({'cwd':os.getcwd(),'visible':os.getenv('VISIBLE_VALUE'),'secret':os.getenv('HARNESS_SECRET')}))"
    )
    result = executor.execute([sys.executable, "-c", script], tmp_path / "exp", {"VISIBLE_VALUE": "yes"})
    assert result.ok
    assert result.pid is not None
    assert result.cwd == str((tmp_path / "exp").resolve())
    payload = json.loads((tmp_path / "exp" / "stdout.log").read_text(encoding="utf-8"))
    assert payload == {"cwd": result.cwd, "visible": "yes", "secret": None}
    assert (tmp_path / "exp" / "stderr.log").read_text(encoding="utf-8") == ""


def test_local_executor_rejects_unallowlisted_environment(tmp_path):
    executor = LocalProcessExecutor()
    with pytest.raises(ExecutorValidationError, match="not allowlisted"):
        executor.execute([sys.executable, "-c", "pass"], tmp_path, {"TOKEN": "secret"})


def test_local_executor_terminates_process_group_on_timeout(tmp_path):
    executor = LocalProcessExecutor(timeout_s=0.05, termination_grace_s=0.2)
    result = executor.execute([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path)
    assert not result.ok
    assert result.failure_type == "timeout"
    assert result.timed_out is True
    assert result.terminated is True
    assert result.wall_time_s < 2
