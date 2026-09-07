from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validate_config_is_offline():
    result = subprocess.run(
        [sys.executable, "-m", "harness4h3", "--config", "configs/default.yaml", "validate-config", "--json"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["tasks"] == 4

