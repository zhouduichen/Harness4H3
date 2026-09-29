from pathlib import Path

import pytest

from harness4h3.student.wan_experiment import WanExperimentConfigError, WanRealExperimentConfig


def test_wan_real_experiment_config_validates_real_remote_boundary():
    config = WanRealExperimentConfig.from_path(Path(__file__).parents[2] / "configs" / "student-campaign.wan-pretrained.yaml")
    assert config.remote.host == "ai4mbse-remote"
    assert config.remote_result.endswith("wan-baseline-result.json")
    assert config.wan_config.endswith("wan-student-1.3b.remote.yaml")


def test_wan_real_experiment_rejects_worker_outside_harness_root(tmp_path):
    source = Path(__file__).parents[2] / "configs" / "student-campaign.wan-pretrained.yaml"
    text = source.read_text(encoding="utf-8").replace(
        "/home/intern/huangjiahao/Harness4H3-long-video-30s/tools/wan_student_real_worker.py",
        "/tmp/wan_student_real_worker.py",
    )
    path = tmp_path / "wan.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(WanExperimentConfigError, match="worker_entrypoint"):
        WanRealExperimentConfig.from_path(path)
