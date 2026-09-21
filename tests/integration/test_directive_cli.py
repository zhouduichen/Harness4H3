from __future__ import annotations

import json

from harness4h3.cli import main
from harness4h3.memory.observation import ObservationStore


def test_directive_cli_appends_and_retries_idempotently(tmp_path, capsys):
    output_root = tmp_path / "campaign"
    argv = [
        "directive",
        "--output-root",
        str(output_root),
        "--directive-id",
        "goal-001",
        "--text",
        "下一轮优先降低 peak_memory，质量下降不得超过 2%",
        "--json",
    ]

    assert main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "submitted"
    assert first["directive_id"] == "goal-001"
    assert first["apply_at"] == "next_controller_plan"

    assert main(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["status"] == "already_present"
    assert second["observation_id"] == first["observation_id"]
    assert len(list(ObservationStore(output_root / "observations.jsonl").read())) == 1
    assert not (output_root / "campaign_state.json").exists()
    assert not (output_root / "controller-events.jsonl").exists()


def test_directive_cli_rejects_malformed_text_without_writing(tmp_path, capsys):
    output_root = tmp_path / "campaign"

    assert main(["directive", "--output-root", str(output_root), "--text", "   ", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "blank" in payload["error"]
    assert not (output_root / "observations.jsonl").exists()
