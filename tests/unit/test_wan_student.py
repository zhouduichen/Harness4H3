from pathlib import Path

import numpy as np
import pytest
import yaml

from harness4h3.student.wan import WanStudentError, WanStudentSpec, _as_frame_array, load_wan_spec


def test_wan_spec_loads_pretrained_lineage_contract():
    path = Path(__file__).parents[2] / "configs" / "wan-student-1.3b.yaml"
    spec = load_wan_spec(path)
    assert spec.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert spec.lineage_root == "W0000"
    assert spec.native_duration_s == pytest.approx(81 / 16.0)
    assert spec.to_dict()["pretrained"] is True


def test_wan_spec_rejects_h3_or_random_init_shape():
    with pytest.raises(ValueError, match="lineage_root"):
        WanStudentSpec(lineage_root="M0000")
    with pytest.raises(ValueError, match="multiple of 8"):
        WanStudentSpec(width=255)


def test_wan_spec_rejects_unknown_config_fields(tmp_path):
    path = tmp_path / "wan.yaml"
    path.write_text(yaml.safe_dump({"wan_student": {"random_init": True}}), encoding="utf-8")
    with pytest.raises(WanStudentError, match="unknown field"):
        load_wan_spec(path)


def test_wan_frame_contract_normalizes_batch_and_unit_range():
    frames = _as_frame_array(np.zeros((1, 3, 2, 4, 3), dtype=np.float32))
    assert frames.shape == (3, 2, 4, 3)
    assert frames.dtype == np.uint8


def test_wan_frame_contract_rejects_non_rgb():
    with pytest.raises(WanStudentError, match="RGB"):
        _as_frame_array(np.zeros((3, 2, 4, 2), dtype=np.float32))
