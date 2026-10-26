from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness4h3.evolution.protocol import ProtocolValidationError, validate_protocol
from tests.unit.test_evolution_protocol import _make_configs
from tools.build_evolution_protocol_configs import build_protocol_configs, main


def test_build_protocol_configs_preserves_inputs_and_separates_controls(tmp_path):
    source_configs = _make_configs(tmp_path / "source")
    output_dir = tmp_path / "generated"
    run_root = tmp_path / "runs"
    generated = build_protocol_configs(
        source_configs["coding-agent"],
        output_dir,
        run_root,
        source_revision="revision-generated",
    )
    report = validate_protocol(generated, source_revision="revision-generated")
    assert set(generated) == {"fixed", "coding-agent", "agent-memory", "rsi"}
    assert report["protocol_fingerprint"]
    assert report["arms"]["fixed"]["provider"] == "deterministic"
    assert report["arms"]["coding-agent"]["provider"] == "vllm"
    assert report["arms"]["agent-memory"]["memory_enabled"] is True
    assert report["arms"]["rsi"]["adapt_search_policy"] is True
    assert len({item["output_root"] for item in report["arms"].values()}) == 4

    fixed = yaml.safe_load(generated["fixed"].read_text(encoding="utf-8"))
    assert fixed["evaluator"]["source_revision"] == "revision-generated"
    assert Path(fixed["evaluator"]["evaluation_manifest"]).is_absolute()
    assert fixed["output_root"] == str(run_root / "fixed")


def test_build_protocol_configs_rejects_deterministic_base(tmp_path):
    source_configs = _make_configs(tmp_path / "source")
    raw = yaml.safe_load(source_configs["coding-agent"].read_text(encoding="utf-8"))
    raw["controller"]["provider"] = "deterministic"
    source_configs["coding-agent"].write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="non-deterministic Coding Agent"):
        build_protocol_configs(
            source_configs["coding-agent"],
            tmp_path / "generated",
            tmp_path / "runs",
            source_revision="revision-generated",
        )


def test_protocol_rejects_coding_arm_that_is_secretly_deterministic(tmp_path):
    configs = _make_configs(tmp_path)
    raw = yaml.safe_load(configs["coding-agent"].read_text(encoding="utf-8"))
    raw["controller"]["provider"] = "offline"
    configs["coding-agent"].write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="coding-agent must use a non-deterministic"):
        validate_protocol(configs)


def test_build_protocol_configs_does_not_overwrite_without_explicit_flag(tmp_path):
    source_configs = _make_configs(tmp_path / "source")
    output_dir = tmp_path / "generated"
    build_protocol_configs(source_configs["coding-agent"], output_dir, tmp_path / "runs", source_revision="revision-generated")
    with pytest.raises(ProtocolValidationError, match="already exists"):
        build_protocol_configs(source_configs["coding-agent"], output_dir, tmp_path / "runs", source_revision="revision-generated")


def test_build_protocol_configs_cli_writes_report_and_manifests(tmp_path, capsys):
    source_configs = _make_configs(tmp_path / "source")
    report_path = tmp_path / "protocol.json"
    assert main(
        [
            "--base-config", str(source_configs["coding-agent"]),
            "--output-dir", str(tmp_path / "generated"),
            "--run-root", str(tmp_path / "runs"),
            "--source-revision", "revision-generated",
            "--write-manifests",
            "--output", str(report_path),
            "--json",
        ]
    ) == 0
    assert "protocol_configs_valid" not in capsys.readouterr().out
    assert report_path.exists()
    assert (tmp_path / "runs" / "rsi" / "run-manifest.json").exists()
