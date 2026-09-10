from __future__ import annotations

import json

from harness4h3.cli import main


def test_a1_cli_fails_closed_without_real_parent_checkpoint(tmp_path, capsys):
    code = main(
        [
            "a1-evolve",
            "--parent-checkpoint",
            str(tmp_path / "missing.safetensors"),
            "--worker-command",
            "python",
            "tools/h3_model_worker.py",
            "--worker-config",
            "configs/a1-worker.example.json",
            "--baseline-quality",
            "0.991137",
            "--baseline-model-size-gb",
            "12.5286368",
            "--baseline-latency-s",
            "90.280584",
            "--baseline-peak-memory-gb",
            "16.294528",
            "--output-root",
            str(tmp_path / "a1"),
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["type"] == "ValueError"
    assert "parent checkpoint does not exist" in payload["error"]
