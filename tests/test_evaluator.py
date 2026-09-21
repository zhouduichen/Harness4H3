from __future__ import annotations

import sys

import pytest

from harness4h3.evaluator.evaluator import EvaluatorError, SubprocessEvaluator, make_request


def test_subprocess_evaluator_is_score_authority(tmp_path):
    artifact = tmp_path / "video.mp4"
    artifact.write_bytes(b"not-a-real-video")
    result = SubprocessEvaluator([]).evaluate(make_request("t", {}, [str(artifact)], 1.5))
    assert 0 <= result.score <= 1
    assert result.metrics["artifact_exists"] == 1.0
    assert result.metrics["artifact_non_empty"] == 1.0


def test_invalid_evaluator_output_fails_closed(tmp_path):
    evaluator = SubprocessEvaluator([sys.executable, "-c", "print('not-json')"])
    with pytest.raises(EvaluatorError, match="valid JSON"):
        evaluator.evaluate(make_request("t", {}, [], 0))


def test_default_evaluator_imports_from_outside_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    result = SubprocessEvaluator([]).evaluate(make_request("t", {}, [], 0))

    assert result.metrics["artifact_exists"] == 0.0
