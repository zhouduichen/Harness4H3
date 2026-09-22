from __future__ import annotations

import pytest

from harness4h3.student.target import TargetDeviceProfile


def _raw():
    return {
        "id": "edge-a",
        "runtime_backend": "runtime-v1",
        "max_latency_s": 1.0,
        "max_memory_gb": 4.0,
        "max_energy_j": 10.0,
        "max_thermal_c": 70.0,
        "max_model_size_gb": 2.0,
        "supported_precision": ["bf16", "fp16"],
        "supported_quantization": ["none", "int8"],
        "resolution": "256x256",
        "frames": 5,
        "sampling_steps": 4,
    }


def test_target_device_profile_is_typed_and_keeps_edge_memory_explicit():
    profile = TargetDeviceProfile.from_mapping(_raw())
    assert profile.resolution == (256, 256)
    assert profile.max_edge_memory_gb == 4.0
    assert profile.to_dict()["supported_quantization"] == ["none", "int8"]


def test_target_device_profile_rejects_unknown_or_invalid_fields():
    raw = _raw()
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="unknown field"):
        TargetDeviceProfile.from_mapping(raw)
    raw = _raw()
    raw["max_energy_j"] = 0
    with pytest.raises(ValueError, match="positive"):
        TargetDeviceProfile.from_mapping(raw)
