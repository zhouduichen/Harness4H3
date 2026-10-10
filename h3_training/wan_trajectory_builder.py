"""Build deterministic native Wan Teacher trajectory records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from h3_training.data.schema import Conditioning, ModalInterval, ModalLatents, ModalTimesteps
from h3_training.wan_trajectory import (
    LoadedWanTrajectory,
    WanTrajectoryRecord,
    json_digest,
    tensor_digest,
    tensors_digest,
)

if TYPE_CHECKING:
    from h3_training.algorithms.base import ModelRole


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_digest(value: Path | str) -> str:
    path = Path(value)
    if path.is_file():
        return _file_digest(path)
    return str(value)


def _batch_timestep(value: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.full((reference.shape[0],), float(value), device=reference.device, dtype=torch.float32)


def _metadata_timetable(adapter: Any, schedule: Any) -> tuple[float, ...]:
    metadata = adapter.native_schedule_metadata(schedule)
    values = metadata.get("timesteps")
    if not isinstance(values, (tuple, list)) or len(values) != schedule.nfe:
        raise ValueError("native schedule metadata must contain one timestep per model evaluation")
    return tuple(float(value) for value in values)


def _conditioning_digest(conditioning: Conditioning) -> str:
    payload = {
        "text": tensor_digest(conditioning.text),
        "negative_text": tensor_digest(conditioning.negative_text)
        if conditioning.negative_text is not None
        else None,
        "metadata": dict(conditioning.metadata),
    }
    return json_digest(payload)


def _tensor_or_raise(clean: ModalLatents) -> torch.Tensor:
    if clean.video is None:
        raise ValueError("native Wan trajectory requires a video clean latent")
    if clean.video.ndim != 5:
        raise ValueError("native Wan clean latent must have shape [B,C,T,H,W]")
    return clean.video


def build_wan_trajectory_record(
    *,
    adapter: Any,
    teacher_role: ModelRole,
    clean: ModalLatents,
    conditioning: Conditioning,
    cache_item: Path | str,
    teacher_checkpoint: Path | str,
    output_dir: Path | None,
    seed: int,
    source_nfe: int = 32,
    target_nfe: int = 16,
) -> LoadedWanTrajectory:
    """Replay the native Teacher once and persist its complete source trajectory."""

    if int(source_nfe) != 2 * int(target_nfe) or int(target_nfe) <= 0:
        raise ValueError("native trajectory builder requires source_nfe == 2 * target_nfe")
    clean_video = _tensor_or_raise(clean)
    source_schedule = adapter.schedule(int(source_nfe))
    target_schedule = adapter.schedule(int(target_nfe))
    source_sigmas = tuple(float(value) for value in (source_schedule.video_sigmas or ()))
    target_sigmas = tuple(float(value) for value in (target_schedule.video_sigmas or ()))
    if len(source_sigmas) != int(source_nfe) + 1 or len(target_sigmas) != int(target_nfe) + 1:
        raise ValueError("adapter returned an invalid native sigma table")
    source_timesteps = _metadata_timetable(adapter, source_schedule)
    target_timesteps = _metadata_timetable(adapter, target_schedule)
    # Wan's production ``prepare_latents`` samples float32 noise on the
    # inference device.  Use the same device-local RNG here so a fixed seed
    # names the same initial latent in training and in the real 16-step
    # sampler; a CPU generator produces a different Philox stream and makes
    # single-sample overfit evidence invalid.  Keep the persisted artifact on
    # CPU, but do the trajectory replay from the device-local float32 noise.
    generator = torch.Generator(device=clean_video.device).manual_seed(int(seed))
    noise = torch.randn(clean_video.shape, generator=generator, device=clean_video.device, dtype=torch.float32)

    adapter.begin_trajectory(teacher_role, source_schedule, 0, clean_video.device)
    initial_timestep = ModalTimesteps(video=_batch_timestep(source_sigmas[0], clean_video))
    state = adapter.add_noise(clean, ModalLatents(video=noise), initial_timestep)
    if state.video is None:
        raise ValueError("native Wan adapter returned no video trajectory state")
    teacher_states = [state.video.detach().cpu().contiguous()]
    teacher_predictions: list[torch.Tensor] = []
    teacher_snapshots = [adapter.capture_native_scheduler_state(teacher_role)]
    for index, (current_sigma, next_sigma) in enumerate(zip(source_sigmas, source_sigmas[1:])):
        # The shared adapter API receives sigma coordinates and performs the
        # exact native sigma->timestep lookup internally.  source_timesteps
        # are persisted as audit metadata only; passing 999/989/... here would
        # make the adapter interpret them as sigmas and select the wrong model
        # timestep.
        sigma_timestep = ModalTimesteps(video=_batch_timestep(current_sigma, state.video))
        prediction = adapter.predict(teacher_role, state, sigma_timestep, conditioning)
        if prediction.video is None:
            raise ValueError("native Wan Teacher returned no video prediction")
        teacher_predictions.append(prediction.video.detach().cpu().contiguous())
        state = adapter.scheduler_step(
            teacher_role,
            state,
            prediction,
            ModalInterval(video=(float(current_sigma), float(next_sigma))),
        )
        if state.video is None:
            raise ValueError("native Wan adapter returned no video endpoint")
        teacher_states.append(state.video.detach().cpu().contiguous())
        teacher_snapshots.append(adapter.capture_native_scheduler_state(teacher_role))

    scheduler_metadata = {
        "source": dict(adapter.native_schedule_metadata(source_schedule)),
        "target": dict(adapter.native_schedule_metadata(target_schedule)),
    }
    record = WanTrajectoryRecord(
        schema_version=1,
        cache_item=str(cache_item),
        cache_digest=_external_digest(cache_item),
        conditioning_digest=_conditioning_digest(conditioning),
        clean_latent_digest=tensor_digest(clean_video.detach().cpu()),
        noise_digest=tensor_digest(noise.detach().cpu()),
        teacher_checkpoint_digest=_external_digest(teacher_checkpoint),
        scheduler_digest=json_digest(scheduler_metadata),
        source_nfe=int(source_nfe),
        target_nfe=int(target_nfe),
        source_sigmas=source_sigmas,
        target_sigmas=target_sigmas,
        source_timesteps=source_timesteps,
        target_timesteps=target_timesteps,
    )
    trajectory = LoadedWanTrajectory(
        metadata=record,
        noise=noise.detach().cpu().contiguous(),
        clean_latent=clean_video.detach().cpu().contiguous(),
        teacher_states=tuple(teacher_states),
        teacher_scheduler_snapshots=tuple(teacher_snapshots),
        teacher_predictions=tuple(teacher_predictions),
    )
    trajectory = LoadedWanTrajectory(
        metadata=record.__class__(**{**record.__dict__, "tensor_digest": tensors_digest(trajectory)}),
        noise=trajectory.noise,
        clean_latent=trajectory.clean_latent,
        teacher_states=trajectory.teacher_states,
        teacher_scheduler_snapshots=trajectory.teacher_scheduler_snapshots,
        teacher_predictions=trajectory.teacher_predictions,
    )
    trajectory.validate()
    if output_dir is not None:
        target_dir = Path(output_dir)
        trajectory.save(target_dir)
        replayed = replay_teacher_from_record(
            adapter=adapter,
            teacher_role=teacher_role,
            conditioning=conditioning,
            trajectory=trajectory,
        )
        replay_errors = [
            float((actual.float() - expected.float()).abs().max())
            for actual, expected in zip(replayed, trajectory.teacher_states)
        ]
        replay_audit = {
            "record_digest": trajectory.metadata.tensor_digest,
            "boundary_count": len(replay_errors),
            "max_abs_error": max(replay_errors),
            "mean_boundary_max_abs_error": sum(replay_errors) / len(replay_errors),
            "tolerance": 1e-4,
            "passed": max(replay_errors) <= 1e-4,
        }
        (target_dir / "trajectory-replay-audit.json").write_text(
            json.dumps(replay_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not replay_audit["passed"]:
            raise ValueError(
                "native trajectory replay parity failed: max_abs_error=%s" % replay_audit["max_abs_error"]
            )
    return trajectory


def replay_teacher_from_record(
    *,
    adapter: Any,
    teacher_role: ModelRole,
    conditioning: Conditioning,
    trajectory: LoadedWanTrajectory,
) -> tuple[torch.Tensor, ...]:
    """Replay a stored record with the native adapter and return all boundaries."""

    trajectory.validate()
    device = next(teacher_role.model.parameters(), torch.empty((), device="cpu")).device
    clean = ModalLatents(video=trajectory.clean_latent.to(device=device))
    noise = ModalLatents(video=trajectory.noise.to(device=device))
    schedule = adapter.schedule(trajectory.metadata.source_nfe)
    adapter.begin_trajectory(teacher_role, schedule, 0, device)
    state = adapter.add_noise(
        clean,
        noise,
        ModalTimesteps(video=_batch_timestep(trajectory.metadata.source_sigmas[0], clean.video)),
    )
    if state.video is None:
        raise ValueError("native replay returned no initial state")
    states = [state.video.detach().cpu().contiguous()]
    for index, (current_sigma, next_sigma) in enumerate(
        zip(trajectory.metadata.source_sigmas, trajectory.metadata.source_sigmas[1:])
    ):
        prediction = adapter.predict(
            teacher_role,
            state,
            ModalTimesteps(video=_batch_timestep(trajectory.metadata.source_sigmas[index], state.video)),
            conditioning,
        )
        state = adapter.scheduler_step(
            teacher_role,
            state,
            prediction,
            ModalInterval(video=(float(current_sigma), float(next_sigma))),
        )
        if state.video is None:
            raise ValueError("native replay returned no endpoint")
        states.append(state.video.detach().cpu().contiguous())
    return tuple(states)


__all__ = ["build_wan_trajectory_record", "replay_teacher_from_record"]
