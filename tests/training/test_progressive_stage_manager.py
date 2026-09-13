from pathlib import Path

import pytest

from h3_training.algorithms.progressive_distillation import DistillationStage
from h3_training.algorithms.progressive_stage_manager import ProgressiveStageManager


def test_manager_promotes_each_accepted_child_as_the_next_teacher():
    manager = ProgressiveStageManager(4, 1, "M0000", "/tmp/M0000.pt")
    assert manager.next_stage() == DistillationStage(4, 2)
    assert manager.next_stage_request()["teacher_model_id"] == "M0000"

    manager.promote("M0001", Path("/tmp/M0001.pt"), 2, accepted=True)
    assert manager.next_stage() == DistillationStage(2, 1)
    assert manager.next_stage_request()["teacher_model_id"] == "M0001"
    assert manager.next_stage_request()["teacher_nfe"] == 2

    manager.promote("M0002", Path("/tmp/M0002.pt"), 1, accepted=True)
    assert manager.complete
    assert manager.next_stage() is None
    assert manager.teacher_model_id == "M0002"

    restored = ProgressiveStageManager.from_state_dict(manager.state_dict())
    assert restored.state_dict() == manager.state_dict()


def test_manager_does_not_promote_rejected_child():
    manager = ProgressiveStageManager(4, 1, "M0000")
    with pytest.raises(ValueError, match="acceptance"):
        manager.promote("M0001", Path("/tmp/M0001.pt"), 2, accepted=False)
    assert manager.stage_index == 0


def test_manager_rejects_non_binary_promotion():
    manager = ProgressiveStageManager(4, 1, "M0000")
    with pytest.raises(ValueError, match="NFE"):
        manager.promote("M0001", Path("/tmp/M0001.pt"), 1, accepted=True)
