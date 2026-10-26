from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from harness4h3.evolution.protocol import ProtocolValidationError, validate_protocol, write_manifests


def _make_configs(tmp_path: Path, *, missing_asset: bool = False) -> dict[str, Path]:
    assets = tmp_path / "assets"
    assets.mkdir(parents=True)
    checkpoint = assets / "checkpoint.marker"
    wan_config = assets / "wan.yaml"
    evaluation_manifest = assets / "manifest.json"
    clip = assets / "clip"
    checkpoint.write_text("checkpoint-a", encoding="utf-8")
    wan_config.write_text("model: wan\n", encoding="utf-8")
    evaluation_manifest.write_text('{"cases": []}\n', encoding="utf-8")
    clip.mkdir()
    if missing_asset:
        checkpoint.unlink()

    configs: dict[str, Path] = {}
    controls = {
        "fixed": {"provider": "deterministic", "model": "fixed", "memory_enabled": False, "adapt_search_policy": False, "exploit_ratio": 1.0},
        "coding-agent": {"provider": "vllm", "model": "controller-a", "memory_enabled": False, "adapt_search_policy": False, "exploit_ratio": 0.8},
        "agent-memory": {"provider": "vllm", "model": "controller-a", "memory_enabled": True, "adapt_search_policy": False, "exploit_ratio": 0.8},
        "rsi": {"provider": "vllm", "model": "controller-a", "memory_enabled": True, "adapt_search_policy": True, "exploit_ratio": 0.8},
    }
    for name, control in controls.items():
        path = tmp_path / (name + ".yaml")
        raw = {
            "backend": "diffusers",
            "checkpoint": str(checkpoint),
            "output_root": str(tmp_path / "outputs" / name),
            "device": "cuda:0",
            "dtype": "bfloat16",
            "local_files_only": True,
            "seed": 7,
            **{key: control[key] for key in ("memory_enabled", "adapt_search_policy", "exploit_ratio")},
            "controller": {"provider": control["provider"], "model": control["model"], "endpoint": "http://127.0.0.1:8000/v1/chat/completions"},
            "evaluator": {
                "backend": "wan_video",
                "source_revision": "revision-a",
                "wan_config": str(wan_config),
                "evaluation_manifest": str(evaluation_manifest),
                "clip_model_path": str(clip),
                "device": "cuda:0",
                "quality_device": "cuda:0",
                "dtype": "bfloat16",
                "local_files_only": True,
                "max_cases": 8,
                "seed_count": 2,
                "sample_frames": 4,
                "max_quality_drop": 0.05,
                "latency_warmup": 1,
                "latency_repeats": 3,
                "visual_quality_gate": {"enabled": True, "version": "test-gate"},
            },
        }
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        configs[name] = path
    return configs


def test_validate_protocol_builds_stable_four_arm_manifest(tmp_path):
    configs = _make_configs(tmp_path)
    report = validate_protocol(configs)
    assert report["required_arms"] == ["fixed", "coding-agent", "agent-memory", "rsi"]
    assert len(report["protocol_fingerprint"]) == 64
    assert report["arms"]["coding-agent"]["evaluator"]["max_cases"] == 8
    assert report["run_manifests"]["rsi"]["controls"]["adapt_search_policy"] is True

    write_manifests(report)
    for name in report["required_arms"]:
        path = Path(report["arms"][name]["output_root"]) / "run-manifest.json"
        assert json.loads(path.read_text(encoding="utf-8")) == report["run_manifests"][name]


def test_protocol_fingerprint_does_not_depend_on_host_absolute_paths(tmp_path):
    first = validate_protocol(_make_configs(tmp_path / "host-a"))
    second = validate_protocol(_make_configs(tmp_path / "host-b"))
    assert first["protocol_fingerprint"] == second["protocol_fingerprint"]
    assert first["arms"]["fixed"]["output_root"] != second["arms"]["fixed"]["output_root"]


def test_validate_protocol_rejects_missing_arm_and_bad_control(tmp_path):
    configs = _make_configs(tmp_path)
    configs.pop("rsi")
    with pytest.raises(ProtocolValidationError, match="missing arms"):
        validate_protocol(configs)

    configs = _make_configs(tmp_path / "bad")
    raw = yaml.safe_load(configs["agent-memory"].read_text(encoding="utf-8"))
    raw["adapt_search_policy"] = True
    configs["agent-memory"].write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="agent-memory must enable memory and disable adaptation"):
        validate_protocol(configs)


def test_validate_protocol_rejects_missing_assets_unless_explicitly_allowed(tmp_path):
    configs = _make_configs(tmp_path, missing_asset=True)
    with pytest.raises(ProtocolValidationError, match="assets are not available"):
        validate_protocol(configs)
    report = validate_protocol(configs, allow_missing_assets=True)
    assert report["arms"]["fixed"]["checkpoint_asset"]["exists"] is False


def test_write_manifests_checks_all_conflicts_before_writing(tmp_path):
    configs = _make_configs(tmp_path)
    report = validate_protocol(configs)
    first_target = Path(report["arms"]["fixed"]["output_root"]) / "run-manifest.json"
    first_target.parent.mkdir(parents=True)
    first_target.write_text(json.dumps({"foreign": True}), encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="conflicts"):
        write_manifests(report)
    other_target = Path(report["arms"]["coding-agent"]["output_root"]) / "run-manifest.json"
    assert not other_target.exists()
