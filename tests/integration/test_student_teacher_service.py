from __future__ import annotations

import torch

from h3_training.data.schema import Conditioning, ModalLatents, ModalPrediction, ModalTimesteps
from harness4h3.student.teacher_service import TeacherService
from harness4h3.student.worker import StudentAlgorithmAdapter


def test_student_algorithm_uses_online_predictor_instead_of_batch_target():
    handle = TeacherService.from_predictors(
        [lambda noisy, timestep, conditioning: ModalPrediction(video=noisy.video + 3.0)]
    )
    adapter = StudentAlgorithmAdapter(teacher_predictor=handle)
    noisy = ModalLatents(video=torch.zeros(1, 2, 1, 2, 2))
    timestep = ModalTimesteps(video=torch.tensor([0.25]))
    conditioning = Conditioning(torch.zeros(1, 1, 3))
    try:
        output = adapter.predict(type("TeacherRole", (), {"name": "teacher"})(), noisy, timestep, conditioning)
        assert torch.allclose(output.video, torch.full_like(noisy.video, 3.0))
        assert handle.ranks_used == {0}
    finally:
        handle.close()
