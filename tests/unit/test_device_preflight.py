from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import yaml

import tools.device_preflight as device_preflight


class Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def write_profile(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "device.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def valid_profile(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "schema_version": 1,
        "id": "test-device",
        "platform": {"os": "test", "python": "3.12"},
        "hardware": {"gpu_name": "test-gpu", "gpu_count": 1, "vram_gb": 16, "ram_gb": 32},
        "paths": {"workspace": {"path": str(workspace), "required": True}},
        "services": {
            "comfyui": {
                "url": "http://127.0.0.1:8188",
                "health_path": "/system_stats",
                "required_for": ["benchmark"],
            }
        },
        "limits": {"max_peak_vram_gb": 16},
        "formats": ["safetensors"],
        "capabilities": {
            "benchmark": {"enabled": True, "reason": "test service"},
            "recovery_finetune": {"enabled": False, "reason": "no trainer"},
        },
    }


def check(result: dict, name: str) -> dict:
    return next(item for item in result["checks"] if item["name"] == name)


def test_preflight_accepts_complete_profile_with_existing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(device_preflight, "urlopen", lambda request, timeout: Response())
    result = device_preflight.preflight(write_profile(tmp_path, valid_profile(tmp_path)), "benchmark")
    assert result["status"] == "ready"
    assert check(result, "profile.schema")["status"] == "passed"
    assert check(result, "capability.benchmark")["status"] == "passed"
    assert check(result, "path.workspace")["status"] == "passed"
    assert check(result, "service.comfyui")["status"] == "passed"


def test_preflight_reports_missing_required_field(tmp_path):
    raw = valid_profile(tmp_path)
    del raw["hardware"]
    result = device_preflight.preflight(write_profile(tmp_path, raw), None, check_services=False)
    assert result["status"] == "blocked"
    assert check(result, "profile.schema")["status"] == "failed"


def test_preflight_reports_disabled_operator_reason(tmp_path):
    result = device_preflight.preflight(
        write_profile(tmp_path, valid_profile(tmp_path)), "recovery_finetune", check_services=False
    )
    assert result["status"] == "blocked"
    assert check(result, "capability.recovery_finetune")["detail"] == "no trainer"


def test_preflight_reports_missing_path(tmp_path):
    raw = valid_profile(tmp_path)
    raw["paths"]["workspace"]["path"] = str(tmp_path / "missing")
    result = device_preflight.preflight(write_profile(tmp_path, raw), "benchmark", check_services=False)
    assert result["status"] == "blocked"
    assert check(result, "path.workspace")["status"] == "failed"


def test_preflight_reports_unreachable_required_service(tmp_path, monkeypatch):
    def unavailable(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(device_preflight, "urlopen", unavailable)
    result = device_preflight.preflight(write_profile(tmp_path, valid_profile(tmp_path)), "benchmark")
    assert result["status"] == "blocked"
    assert check(result, "service.comfyui")["status"] == "failed"
