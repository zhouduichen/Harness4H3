import json

from harness4h3.cli import main
from harness4h3.memory.experience import ExperienceRecord
from harness4h3.remote.importer import ImportSummary


def test_import_experience_cli_reports_status_without_remote_write(tmp_path, monkeypatch, capsys):
    record = ExperienceRecord.minimal("xp-smoke", "ssh://host/result.json", "a" * 64)

    def fake_discover(self, root=None):
        return ImportSummary(1, 0, (), (record,))

    monkeypatch.setattr("harness4h3.cli.RemoteResultImporter.discover_remote", fake_discover)
    output = tmp_path / "experience.jsonl"
    code = main(
        [
            "import-experience",
            "--remote-config",
            "configs/remote-l40-h3.yaml",
            "--output",
            str(output),
            "--dry-run",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["records"][0]["status"] == "training_only_unvalidated"
    assert not output.exists()

