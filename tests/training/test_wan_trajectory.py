from dataclasses import replace

import pytest
import torch

from h3_training.wan_trajectory import (
    LoadedWanTrajectory,
    WanTrajectoryRecord,
    load_trajectory_tensors,
    tensor_digest,
    tensors_digest,
)


def make_minimal_trajectory(source_nfe=4, target_nfe=2):
    noise = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    clean = noise + 1.0
    states = tuple(clean + float(index) for index in range(source_nfe + 1))
    predictions = tuple(torch.full_like(clean, float(index + 1)) for index in range(source_nfe))
    snapshots = tuple({"index": index, "history": torch.tensor([float(index)])} for index in range(source_nfe + 1))
    placeholder = WanTrajectoryRecord(
        schema_version=1,
        cache_item="cache.pt",
        cache_digest="cache-digest",
        conditioning_digest="conditioning-digest",
        clean_latent_digest=tensor_digest(clean),
        noise_digest=tensor_digest(noise),
        teacher_checkpoint_digest="teacher-digest",
        scheduler_digest="scheduler-digest",
        source_nfe=source_nfe,
        target_nfe=target_nfe,
        source_sigmas=tuple(float(source_nfe - index) / source_nfe for index in range(source_nfe)) + (0.0,),
        target_sigmas=tuple(float(target_nfe - index) / target_nfe for index in range(target_nfe)) + (0.0,),
        source_timesteps=tuple(float(source_nfe - index) for index in range(source_nfe)),
        target_timesteps=tuple(float(target_nfe - index) for index in range(target_nfe)),
    )
    trajectory = LoadedWanTrajectory(
        metadata=placeholder,
        noise=noise,
        clean_latent=clean,
        teacher_states=states,
        teacher_scheduler_snapshots=snapshots,
        teacher_predictions=predictions,
    )
    return replace(trajectory, metadata=replace(placeholder, tensor_digest=tensors_digest(trajectory)))


def test_record_round_trip_preserves_digests(tmp_path):
    trajectory = make_minimal_trajectory()
    trajectory.save(tmp_path)
    loaded = WanTrajectoryRecord.load(tmp_path)
    restored = load_trajectory_tensors(tmp_path, loaded)
    restored.validate()
    assert loaded.tensor_digest == tensors_digest(restored)
    assert loaded.noise_digest == tensor_digest(restored.noise)


def test_record_rejects_mutated_noise(tmp_path):
    trajectory = make_minimal_trajectory()
    trajectory.save(tmp_path)
    record = trajectory.metadata
    payload = torch.load(tmp_path / record.tensor_artifact, map_location="cpu", weights_only=True)
    payload["noise"][0, 0, 0, 0, 0] += 1.0
    torch.save(payload, tmp_path / record.tensor_artifact)
    with pytest.raises(ValueError, match="noise digest"):
        load_trajectory_tensors(tmp_path, record).validate()


def test_record_rejects_wrong_boundary_count():
    trajectory = make_minimal_trajectory(source_nfe=4, target_nfe=2)
    broken = replace(trajectory, teacher_states=trajectory.teacher_states[:-1])
    with pytest.raises(ValueError, match="teacher state boundary count"):
        broken.validate()
