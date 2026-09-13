import pytest

torch = pytest.importorskip("torch")

from h3_training.engine.checkpoint import load_training_checkpoint
from h3_training.engine.state import TrainingFailure
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from tests.training.test_trainer_engine import CountingMethod, batches


def test_checkpoint_rejects_parent_mismatch(tmp_path):
    method = CountingMethod()
    engine = TrainerEngine(TrainerConfig(parent_sha256="a" * 64))
    result = engine.run(method, batches(), max_steps=1)
    checkpoint = engine.save_checkpoint(method, result.loop_state, tmp_path / "resume.pt")
    other = CountingMethod()
    with pytest.raises(TrainingFailure, match="resume_mismatch"):
        load_training_checkpoint(checkpoint, other, "b" * 64, engine.config.digest, torch.Generator())


def test_corrupt_checkpoint_fails_closed(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(TrainingFailure, match="checkpoint_corrupt"):
        load_training_checkpoint(path, CountingMethod(), "", TrainerConfig().digest, torch.Generator())


def test_resume_matches_uninterrupted_cpu(tmp_path):
    torch.manual_seed(3)
    uninterrupted = CountingMethod()
    full_engine = TrainerEngine(TrainerConfig(seed=9))
    full = full_engine.run(uninterrupted, batches(), max_steps=4)

    torch.manual_seed(3)
    interrupted = CountingMethod()
    first_engine = TrainerEngine(TrainerConfig(seed=9))
    first = first_engine.run(interrupted, batches(), max_steps=2)
    checkpoint = first_engine.save_checkpoint(interrupted, first.loop_state, tmp_path / "resume.pt")
    resumed_engine = TrainerEngine(TrainerConfig(seed=9))
    resumed = resumed_engine.run(interrupted, batches(), max_steps=4, resume_from=checkpoint)
    assert torch.equal(uninterrupted.weight.weight, interrupted.weight.weight)
    assert resumed.loop_state == full.loop_state
