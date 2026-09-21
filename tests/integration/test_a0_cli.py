from __future__ import annotations

import json

from harness4h3.cli import main


def test_a0_cli_runs_offline_campaign(tmp_path, capsys):
    output_root = tmp_path / "a0"
    code = main(
        [
            "a0-evolve",
            "--target",
            "configs/targets/rtx5080_example.yaml",
            "--output-root",
            str(output_root),
            "--max-experiments",
            "2",
            "--max-gpu-hours",
            "2",
            "--json",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["campaign"] == "A0"
    assert summary["offline_simulation"] is True
    report = json.loads((output_root / "report.json").read_text(encoding="utf-8"))
    assert report["human_intervention_count"] == 0
    assert report["harness_version"] == "Harness4H3-v0.4"
