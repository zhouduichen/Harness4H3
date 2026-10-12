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


def test_profile_wan_fake_cli_writes_json(tmp_path):
    config = tmp_path / "compression.yaml"
    config.write_text(
        "backend: fake\n"
        "checkpoint: fake://baseline\n"
        "output_root: %s\n"
        "device: cpu\n"
        "dtype: float32\n"
        "fake:\n"
        "  block_count: 4\n"
        "  hidden_size: 8\n"
        "  ffn_width: 16\n" % (tmp_path / "run"),
        encoding="utf-8",
    )
    output = tmp_path / "profile.json"
    result = subprocess.run(
        [sys.executable, "-m", "harness4h3", "profile-wan", "--config", str(config), "--output", str(output), "--json"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["model"]["architecture"] == "WanFakeDiT"


def test_evolve_compression_fake_cli_writes_lineage(tmp_path):
    config = tmp_path / "compression.yaml"
    config.write_text(
        "backend: fake\n"
        "checkpoint: fake://baseline\n"
        "output_root: %s\n"
        "device: cpu\n"
        "dtype: float32\n"
        "fake:\n"
        "  block_count: 4\n"
        "  hidden_size: 8\n"
        "  ffn_width: 16\n" % (tmp_path / "run"),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "harness4h3", "evolve-compression", "--config", str(config), "--rounds", "1", "--json"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["rounds"][0]["child_model_id"] == "M0001"
