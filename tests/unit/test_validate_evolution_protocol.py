from __future__ import annotations

import json

from tests.unit.test_evolution_protocol import _make_configs
from tools.validate_evolution_protocol import main


def _args(configs, *extra):
    args = []
    for name in ("fixed", "coding-agent", "agent-memory", "rsi"):
        if name in configs:
            args.extend(["--run", "%s=%s" % (name, configs[name])])
    args.extend(extra)
    return args


def test_cli_validates_writes_manifests_and_report(tmp_path, capsys):
    configs = _make_configs(tmp_path)
    report_path = tmp_path / "protocol.json"
    assert main(_args(configs, "--write-manifests", "--output", str(report_path), "--json")) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protocol_fingerprint"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_cli_reports_invalid_protocol_without_writing_report(tmp_path, capsys):
    configs = _make_configs(tmp_path)
    report_path = tmp_path / "protocol.json"
    configs.pop("rsi")
    assert main(_args(configs, "--output", str(report_path))) == 1
    assert "protocol_invalid" in capsys.readouterr().out
    assert not report_path.exists()
