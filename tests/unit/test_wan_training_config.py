from pathlib import Path

from harness4h3.student.wan_experiment import WanTrainingExperimentConfig


def test_wan_training_config_builds_isolated_remote_paths(tmp_path: Path):
    path = tmp_path / "student-campaign.wan-5s.yaml"
    path.write_text(
        "wan_training:\n"
        "  train_steps: 1\n"
        "remote:\n"
        "  host: ai4mbse-remote\n"
        "  ssh_port: 30902\n"
        "  harness_root: /srv/harness\n"
        "  model_root: /data/models\n"
        "  comfyui_root: /srv/comfy\n"
        "  output_root: /srv/harness/work/wan-student-5s-w1a\n"
        "  result_path: /srv/harness/work/wan-student-5s-w1a/result.json\n",
        encoding="utf-8",
    )
    config = WanTrainingExperimentConfig.from_path(path)
    assert config.steps == 1
    assert config.training_config == "/srv/harness/configs/student-campaign.wan-5s.yaml"
    assert config.worker_entrypoint.endswith("/tools/wan_student_train_worker.py")
    assert config.remote_result.endswith("wan-student-5s-w1a/result.json")
