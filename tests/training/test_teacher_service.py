from __future__ import annotations

import pytest
import torch

from h3_training.data.schema import Conditioning, ModalLatents, ModalPrediction, ModalTimesteps
from harness4h3.student.teacher_service import TeacherService, TeacherServiceError


def _request():
    noisy = ModalLatents(video=torch.zeros(1, 2, 1, 2, 2), audio=torch.zeros(1, 1, 1))
    timestep = ModalTimesteps(video=torch.tensor([0.25]), audio=torch.tensor([0.5]))
    conditioning = Conditioning(torch.zeros(1, 1, 3))
    return noisy, timestep, conditioning


def test_in_process_teacher_service_round_robin_uses_every_rank():
    def predictor(rank):
        def _predict(noisy, timestep, conditioning):
            del timestep, conditioning
            return ModalPrediction(video=noisy.video + float(rank), audio=noisy.audio + float(rank))

        return _predict

    handle = TeacherService.from_predictors([predictor(0), predictor(1), predictor(2)])
    noisy, timestep, conditioning = _request()
    try:
        outputs = [handle.predict(noisy, timestep, conditioning) for _ in range(6)]
        assert handle.world_size == 3
        assert handle.sharded is False
        assert handle.ranks_used == {0, 1, 2}
        assert handle.rank_forward_counts == {0: 2, 1: 2, 2: 2}
        assert [float(output.video.flatten()[0]) for output in outputs] == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    finally:
        handle.close()


def test_real_teacher_service_requires_exactly_three_distinct_cuda_devices(tmp_path):
    checkpoint = tmp_path / "teacher.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="exactly three"):
        TeacherService(checkpoint, tmp_path, ("cuda:0", "cuda:1"))
    with pytest.raises(ValueError, match="distinct"):
        TeacherService(checkpoint, tmp_path, ("cuda:0", "cuda:0", "cuda:1"))


def test_closed_teacher_service_rejects_forward():
    handle = TeacherService.from_predictors([lambda noisy, timestep, conditioning: ModalPrediction(video=noisy.video)])
    handle.close()
    with pytest.raises(TeacherServiceError, match="closed"):
        handle.predict(*_request())
