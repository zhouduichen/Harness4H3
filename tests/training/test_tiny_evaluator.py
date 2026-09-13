import pytest

torch = pytest.importorskip("torch")

from h3_training.tiny.evaluator import TinyCheckpointEvaluator
from h3_training.tiny.factory import create_tiny_checkpoint, load_tiny_checkpoint


def test_tiny_checkpoint_roundtrip_and_measured_evaluation(tmp_path):
    path = create_tiny_checkpoint(tmp_path / "M0000.pt", model_id="M0000", sampling_nfe=4, seed=9)
    model, metadata = load_tiny_checkpoint(path)
    result = TinyCheckpointEvaluator(dataset_size=4).evaluate_checkpoint(path)
    assert metadata["model_id"] == "M0000"
    assert model.config.hidden_size == 32
    assert 0.0 < result.score <= 1.0
    assert result.metrics["heldout_mse"] > 0
    assert result.metrics["latency_s"] > 0
    assert result.metrics["checkpoint_bytes"] == path.stat().st_size
    assert result.metrics["measured"] is True
