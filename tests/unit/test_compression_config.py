from __future__ import annotations

from harness4h3.evolution.config import CompressionConfig


def test_evolution_features_are_default_off(tmp_path):
    path = tmp_path / "compression.yaml"
    path.write_text("backend: fake\ncheckpoint: fake://baseline\n", encoding="utf-8")
    config = CompressionConfig.from_path(path)
    assert config.advanced_profiling is False
    assert config.algorithm_improvement is False
    assert config.search_space_evolution is False
    assert config.frozen_expanded_search_space is False
    assert config.candidate_screening is False
    assert config.channel_method == "weight"


def test_evolution_features_can_be_enabled_independently(tmp_path):
    path = tmp_path / "compression.yaml"
    path.write_text(
        "\n".join(
            [
                "backend: fake",
                "checkpoint: fake://baseline",
                "advanced_profiling: true",
                "algorithm_improvement: true",
                "search_space_evolution: true",
                "candidate_screening: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = CompressionConfig.from_path(path)
    assert config.advanced_profiling is True
    assert config.algorithm_improvement is True
    assert config.search_space_evolution is True
    assert config.candidate_screening is True


def test_frozen_expanded_search_space_is_a_distinct_mode(tmp_path):
    path = tmp_path / "compression.yaml"
    path.write_text(
        "backend: fake\ncheckpoint: fake://baseline\n"
        "frozen_expanded_search_space: true\n",
        encoding="utf-8",
    )
    config = CompressionConfig.from_path(path)
    assert config.frozen_expanded_search_space is True
    assert config.search_space_evolution is False


def test_channel_method_can_enable_real_ffn_output_perturbation(tmp_path):
    path = tmp_path / "compression.yaml"
    path.write_text(
        "backend: fake\ncheckpoint: fake://baseline\nchannel_method: output_perturbation\n",
        encoding="utf-8",
    )

    config = CompressionConfig.from_path(path)

    assert config.channel_method == "output_perturbation"


def test_tica_split_manifest_paths_are_resolved_independently(tmp_path):
    path = tmp_path / "compression.yaml"
    path.write_text(
        "\n".join(
            [
                "backend: diffusers",
                "checkpoint: /models/Wan2.1-T2V-1.3B-Diffusers",
                "tica_prune: true",
                "tica_calibration_manifest: manifests/calibration.json",
                "tica_validation_manifest: manifests/validation.json",
                "tica_test_manifest: manifests/test.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = CompressionConfig.from_path(path)
    assert config.tica_calibration_manifest == (tmp_path / "manifests/calibration.json").resolve()
    assert config.tica_validation_manifest == (tmp_path / "manifests/validation.json").resolve()
    assert config.tica_test_manifest == (tmp_path / "manifests/test.json").resolve()
