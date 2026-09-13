from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest
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


def distributed_profile(tmp_path: Path) -> dict:
    raw = valid_profile(tmp_path)
    raw["hardware"] = {
        "gpu_name_contains": "NVIDIA L40",
        "gpu_count": 4,
        "min_vram_gib_per_gpu": 40,
        "min_system_ram_gib": 128,
        "distributed_backend": "nccl",
    }
    raw["capabilities"]["recovery_finetune"] = {"enabled": True, "reason": "test only"}
    return raw


def install_hardware_observations(tmp_path, monkeypatch, gpu_lines: str, ram_gib: int = 256):
    class Completed:
        returncode = 0
        stdout = gpu_lines
        stderr = ""

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: %d kB\n" % (ram_gib * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(device_preflight, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(device_preflight, "MEMINFO_PATH", meminfo)
    monkeypatch.setattr(
        device_preflight,
        "_torch_observation",
        lambda: {"version": "2.11.0", "cuda": "12.8", "available": True, "device_count": 4},
    )


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


def test_preflight_accepts_four_matching_l40_gpus(tmp_path, monkeypatch):
    install_hardware_observations(
        tmp_path,
        monkeypatch,
        "NVIDIA L40, 46068\nNVIDIA L40, 46068\nNVIDIA L40, 46068\nNVIDIA L40, 46068\n",
    )
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)),
        "recovery_finetune",
        check_services=False,
    )
    assert result["status"] == "ready"
    assert check(result, "hardware.gpu_count")["status"] == "passed"
    assert check(result, "hardware.gpu_name")["status"] == "passed"
    assert check(result, "hardware.gpu_vram")["status"] == "passed"
    assert check(result, "hardware.system_ram")["status"] == "passed"
    assert check(result, "runtime.torch")["status"] == "passed"


@pytest.mark.parametrize(
    ("gpu_lines", "failed_check"),
    [
        ("NVIDIA L40, 46068\n" * 3, "hardware.gpu_count"),
        ("NVIDIA A100, 46068\n" + "NVIDIA L40, 46068\n" * 3, "hardware.gpu_name"),
        ("NVIDIA L40, 39000\n" + "NVIDIA L40, 46068\n" * 3, "hardware.gpu_vram"),
    ],
)
def test_preflight_rejects_gpu_mismatch(tmp_path, monkeypatch, gpu_lines, failed_check):
    install_hardware_observations(tmp_path, monkeypatch, gpu_lines)
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)),
        "recovery_finetune",
        check_services=False,
    )
    assert result["status"] == "blocked"
    assert check(result, failed_check)["status"] == "failed"


def test_preflight_reports_missing_nvidia_smi(tmp_path, monkeypatch):
    raw = distributed_profile(tmp_path)
    monkeypatch.setattr(device_preflight, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))
    monkeypatch.setattr(device_preflight, "_system_ram_gib", lambda: 256.0)
    monkeypatch.setattr(
        device_preflight,
        "_torch_observation",
        lambda: {"version": "2.11.0", "cuda": "12.8", "available": True, "device_count": 4},
    )
    result = device_preflight.preflight(write_profile(tmp_path, raw), "recovery_finetune", check_services=False)
    assert result["status"] == "blocked"
    assert check(result, "hardware.gpu_count")["status"] == "failed"
    assert "nvidia-smi" in check(result, "hardware.gpu_count")["detail"]


def test_preflight_reports_malformed_nvidia_smi(tmp_path, monkeypatch):
    install_hardware_observations(tmp_path, monkeypatch, "not-a-csv-line\n")
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)), "recovery_finetune", check_services=False
    )
    assert result["status"] == "blocked"
    assert check(result, "hardware.gpu_count")["status"] == "failed"


def test_preflight_reports_insufficient_system_ram(tmp_path, monkeypatch):
    install_hardware_observations(tmp_path, monkeypatch, "NVIDIA L40, 46068\n" * 4, ram_gib=64)
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)), "recovery_finetune", check_services=False
    )
    assert result["status"] == "blocked"
    assert check(result, "hardware.system_ram")["status"] == "failed"


def test_preflight_reports_unavailable_torch_cuda(tmp_path, monkeypatch):
    install_hardware_observations(tmp_path, monkeypatch, "NVIDIA L40, 46068\n" * 4)
    monkeypatch.setattr(
        device_preflight,
        "_torch_observation",
        lambda: {"version": "2.11.0", "cuda": "None", "available": False, "device_count": 0},
    )
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)), "recovery_finetune", check_services=False
    )
    assert result["status"] == "blocked"
    assert check(result, "runtime.torch")["status"] == "failed"


def test_preflight_skip_hardware_remains_blocked(tmp_path):
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)),
        "recovery_finetune",
        check_services=False,
        check_hardware=False,
    )
    assert result["status"] == "blocked"
    assert check(result, "hardware.gpu_count")["status"] == "skipped"
    assert check(result, "runtime.torch")["status"] == "skipped"


def test_preflight_blocks_pending_verification(tmp_path, monkeypatch):
    install_hardware_observations(
        tmp_path,
        monkeypatch,
        "NVIDIA L40, 46068\n" * 4,
    )
    raw = distributed_profile(tmp_path)
    raw["verification"] = {"status": "pending"}
    result = device_preflight.preflight(
        write_profile(tmp_path, raw),
        "recovery_finetune",
        check_services=False,
    )
    assert result["status"] == "blocked"
    assert check(result, "profile.verification")["status"] == "failed"
    assert "pending" in check(result, "profile.verification")["detail"]


def test_preflight_accepts_verified_profile(tmp_path, monkeypatch):
    install_hardware_observations(
        tmp_path,
        monkeypatch,
        "NVIDIA L40, 46068\n" * 4,
    )
    raw = distributed_profile(tmp_path)
    raw["verification"] = {"status": "verified"}
    result = device_preflight.preflight(
        write_profile(tmp_path, raw),
        "recovery_finetune",
        check_services=False,
    )
    assert result["status"] == "ready"
    assert check(result, "profile.verification")["status"] == "passed"


def test_write_result_persists_exact_json(tmp_path):
    result = {"status": "blocked", "checks": [{"name": "x", "status": "failed"}]}
    output = device_preflight.write_result(tmp_path / "preflight.json", result)
    assert output == (tmp_path / "preflight.json").resolve()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert not (tmp_path / "preflight.json.tmp").exists()


def test_preflight_rejects_file_kind_for_directory(tmp_path):
    raw = valid_profile(tmp_path)
    raw["paths"]["workspace"]["kind"] = "file"
    result = device_preflight.preflight(write_profile(tmp_path, raw), "benchmark", check_services=False)
    assert result["status"] == "blocked"
    assert check(result, "path.workspace")["status"] == "failed"
    assert "expected file" in check(result, "path.workspace")["detail"]


def test_preflight_rejects_directory_kind_for_file(tmp_path):
    raw = valid_profile(tmp_path)
    file_path = tmp_path / "workspace-file"
    file_path.write_text("not a directory", encoding="utf-8")
    raw["paths"]["workspace"] = {"path": str(file_path), "required": True, "kind": "directory"}
    result = device_preflight.preflight(write_profile(tmp_path, raw), "benchmark", check_services=False)
    assert result["status"] == "blocked"
    assert check(result, "path.workspace")["status"] == "failed"
    assert "expected directory" in check(result, "path.workspace")["detail"]


def test_preflight_filters_paths_by_operation(tmp_path):
    raw = valid_profile(tmp_path)
    benchmark_path = tmp_path / "benchmark"
    benchmark_path.mkdir()
    raw["paths"] = {
        "trainer": {
            "path": str(tmp_path / "missing-trainer"),
            "required": True,
            "required_for": ["recovery_finetune"],
        },
        "benchmark": {
            "path": str(benchmark_path),
            "required": True,
            "required_for": ["benchmark"],
        },
    }
    result = device_preflight.preflight(write_profile(tmp_path, raw), "benchmark", check_services=False)
    assert result["status"] == "ready"
    assert check(result, "path.benchmark")["status"] == "passed"
    assert not any(item["name"] == "path.trainer" for item in result["checks"])
