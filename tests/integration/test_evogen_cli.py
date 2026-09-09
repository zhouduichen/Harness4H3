from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "harness4h3", *args, "--json"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_target_profile_is_offline():
    result = run_cli("validate", "--target", "configs/targets/mobile_example.yaml")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["target"]["id"] == "mobile_h3_v1"


def test_optimize_cli_reaches_fake_target_and_exposes_lineage(tmp_path):
    session = tmp_path / "evogen"
    result = run_cli("optimize", "--session-dir", str(session), "--session-id", "cli-test")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "target_satisfied"
    assert [item["id"] for item in payload["lineage"]] == ["M0000", "M0001", "M0002"]
    lineage = run_cli("lineage", "--session-dir", str(session))
    assert json.loads(lineage.stdout)["active"] == "M0002"
    pareto = run_cli("pareto", "--session-dir", str(session))
    assert json.loads(pareto.stdout)["pareto_front"][0]["candidate_id"] == "M0002"
