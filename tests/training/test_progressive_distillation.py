import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.base import ModelRole
from h3_training.algorithms.progressive_distillation import (
    DistillationStage,
    ProgressiveDistillation,
    ProgressiveDistillationConfig,
    plan_binary_stages,
)
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.data.schema import ModalLatents
from h3_training.data.schema import Conditioning, ModalInterval, ModalPrediction, ModalSchedule, ModalTimesteps, PreparedBatch
from h3_training.engine.state import TrainingFailure
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from h3_training.tiny.model import TinyH3Model
from h3_training.wan_trajectory_builder import build_wan_trajectory_record


class NativeToyModel(torch.nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(weight)))

    def forward(self, noisy, conditioning, timestep):
        assert noisy.video is not None
        return ModalPrediction(video=noisy.video * self.weight)


class NativeToyAdapter:
    native_trajectory_training = True

    def __init__(self):
        self.index = 0

    def schedule(self, nfe):
        return ModalSchedule(video_sigmas=tuple(float(nfe - i) / nfe for i in range(nfe)) + (0.0,))

    def native_schedule_metadata(self, schedule):
        values = schedule.video_sigmas
        assert values is not None
        return {"class": "NativeToyAdapter", "timesteps": [1000.0 * value for value in values[:-1]]}

    def prepare_batch(self, raw, generator):
        return PreparedBatch(
            conditioning=Conditioning(text=raw["conditioning"]),
            latents=ModalLatents(video=raw["clean"]),
            noise=ModalLatents(video=raw["noise"]),
        )

    def resolve_trainable_parameters(self, role, policy):
        return ("weight",)

    def begin_trajectory(self, role, schedule, start_index, device):
        self.index = int(start_index)
        return None

    def capture_native_scheduler_state(self, role):
        return {"index": self.index}

    def add_noise(self, clean, noise, timestep):
        sigma = timestep.video.reshape(-1, 1, 1, 1, 1)
        return ModalLatents(video=(1.0 - sigma) * clean.video + sigma * noise.video)

    def predict(self, role, noisy, timestep, conditioning):
        return role.model(noisy, conditioning, timestep)

    def scheduler_step(self, role, latent, prediction, interval):
        self.index += 1
        return ModalLatents(video=latent.video + (interval.video[1] - interval.video[0]) * prediction.video)

    def prediction_to_clean(self, noisy, prediction, timestep):
        sigma = timestep.video.reshape(-1, 1, 1, 1, 1)
        return ModalLatents(video=noisy.video - sigma * prediction.video)


def clone(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def equal(left, right):
    return all(torch.equal(left[name], right[name]) for name in left)


def fixture():
    torch.manual_seed(7)
    teacher = TinyH3Model()
    student = copy.deepcopy(teacher)
    config = ProgressiveDistillationConfig(DistillationStage(4, 2), learning_rate=2e-3)
    method = ProgressiveDistillation(student, teacher, TinyH3Adapter(), config)
    return method, [[SyntheticH3Dataset(1, 91)[0]]]


def native_rollout_fixture(tmp_path, *, intervals=2):
    adapter = NativeToyAdapter()
    teacher = NativeToyModel(0.5)
    student = NativeToyModel(0.2)
    teacher_role = ModelRole("teacher", teacher, adapter, False)
    clean = ModalLatents(video=torch.full((1, 1, 1, 2, 2), 0.5))
    conditioning = Conditioning(text=torch.ones(1, 1, 1))
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"cache")
    teacher_path = tmp_path / "teacher.safetensors"
    teacher_path.write_bytes(b"teacher")
    record = build_wan_trajectory_record(
        adapter=adapter,
        teacher_role=teacher_role,
        clean=clean,
        conditioning=conditioning,
        cache_item=cache,
        teacher_checkpoint=teacher_path,
        output_dir=None,
        seed=20261008,
        source_nfe=4,
        target_nfe=2,
    )
    config = ProgressiveDistillationConfig(
        DistillationStage(4, 2),
        mode="native_rollout_state",
        teacher_forced_intervals=intervals,
        learning_rate=2e-3,
    )
    method = ProgressiveDistillation(student, teacher, adapter, config)
    method.set_trajectory_record(record)
    method.prepare()
    batch = PreparedBatch(
        conditioning=conditioning,
        latents=clean,
        noise=ModalLatents(video=record.noise.clone()),
    )
    return method, record, batch


def test_native_rollout_state_uses_matching_states_and_no_legacy_velocity(tmp_path):
    method, record, batch = native_rollout_fixture(tmp_path, intervals=2)
    output = method.training_step(batch, iteration=1)
    assert output.metrics["teacher_forced_intervals"] == 2.0
    assert "same_state_endpoint_loss" in output.losses
    assert "same_state_clean_loss" in output.losses
    assert "same_state_flow_loss" in output.losses
    assert "velocity_loss" not in output.losses
    assert output.detached["trajectory_record"] == record.metadata.tensor_digest


def test_native_rollout_state_loss_backpropagates_to_student(tmp_path):
    method, _, batch = native_rollout_fixture(tmp_path, intervals=1)
    output = method.training_step(batch, iteration=1)
    output.losses["total_loss"].backward()
    assert method.student_model.weight.grad is not None
    assert torch.isfinite(method.student_model.weight.grad).all()


def test_binary_stage_requires_two_to_one_nfe():
    with pytest.raises(ValueError, match="twice"):
        DistillationStage(teacher_nfe=12, student_nfe=8)


def test_plan_binary_stages():
    assert plan_binary_stages(16, 4) == (DistillationStage(16, 8), DistillationStage(8, 4))


def test_progressive_distillation_updates_student_not_teacher():
    method, batches = fixture()
    teacher_before = clone(method.teacher_model)
    student_before = clone(method.student_model)
    result = TrainerEngine(TrainerConfig(seed=13)).run(method, batches, max_steps=4)
    assert equal(clone(method.teacher_model), teacher_before)
    assert not equal(clone(method.student_model), student_before)
    assert result.optimizer_steps == {"student": 4}
    assert len(result.loss_trace) == 4
    assert all("total_loss" in item and "flow_loss" in item for item in result.loss_trace)


def test_progressive_distillation_can_keep_optimizer_state_in_fp32():
    torch.manual_seed(7)
    teacher = TinyH3Model()
    student = copy.deepcopy(teacher)
    config = ProgressiveDistillationConfig(
        DistillationStage(4, 2), learning_rate=2e-3, fp32_optimizer_state=True
    )
    method = ProgressiveDistillation(student, teacher, TinyH3Adapter(), config)
    batches = [[SyntheticH3Dataset(1, 91)[0]]]
    result = TrainerEngine(TrainerConfig(seed=13)).run(method, batches, max_steps=1)
    assert result.optimizer_steps == {"student": 1}
    state = next(iter(method._optimizer.state.values()))
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_progressive_distillation_exposes_direct_velocity_alignment_loss():
    method, batches = fixture()
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    output = method.training_step(batch, iteration=1)
    assert "velocity_loss" in output.losses
    assert "clean_anchor_loss" in output.losses
    assert torch.isfinite(output.losses["velocity_loss"])
    assert torch.isfinite(output.losses["clean_anchor_loss"])
    assert output.losses["velocity_loss"].dtype == torch.float32
    assert output.losses["clean_anchor_loss"].dtype == torch.float32
    for name in ("endpoint_loss", "flow_loss", "clean_loss"):
        assert name in output.losses
        assert torch.isfinite(output.losses[name])


def test_native_trajectory_mode_replays_prefix_before_current_interval():
    method, batches = fixture()
    method.adapter.native_trajectory_training = True
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    output = method.training_step(batch, iteration=2)
    assert torch.isfinite(output.losses["total_loss"])
    assert output.metrics["interval_index"] == 1.0


def test_p1_loss_components_are_separately_observable_and_rollout_is_additive():
    method, batches = fixture()
    method.config = replace(
        method.config,
        rollout_steps=2,
        rollout_every=1,
        rollout_endpoint_weight=1.0,
        rollout_clean_weight=0.5,
    )
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    output = method.training_step(batch, iteration=1)
    expected = (
        method.config.endpoint_weight * output.losses["endpoint_loss"]
        + method.config.velocity_weight * output.losses["flow_loss"]
        + method.config.clean_anchor_weight * output.losses["clean_loss"]
        + method.config.rollout_endpoint_weight * output.losses["rollout_endpoint_loss"]
        + method.config.rollout_clean_weight * output.losses["rollout_clean_loss"]
    )
    assert torch.allclose(output.losses["total_loss"], expected)
    assert output.losses["flow_loss"].dtype == torch.float32
    assert output.losses["clean_loss"].dtype == torch.float32
    assert output.losses["rollout_endpoint_loss"].requires_grad
    assert output.losses["rollout_clean_loss"].requires_grad


@pytest.mark.parametrize(
    ("name", "weights"),
    [
        ("endpoint", {"endpoint_weight": 1.0}),
        ("flow", {"endpoint_weight": 0.0, "velocity_weight": 1.0, "clean_anchor_weight": 0.0}),
        ("clean", {"endpoint_weight": 0.0, "velocity_weight": 0.0, "clean_anchor_weight": 1.0}),
        (
            "rollout",
            {
                "endpoint_weight": 0.0,
                "velocity_weight": 0.0,
                "clean_anchor_weight": 0.0,
                "rollout_steps": 2,
                "rollout_endpoint_weight": 1.0,
            },
        ),
    ],
)
def test_loss_ablation_can_isolate_one_signal(name, weights):
    method, batches = fixture()
    method.config = replace(method.config, **weights)
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    output = method.training_step(batch, iteration=1)
    expected = {
        "endpoint": output.losses["endpoint_loss"],
        "flow": output.losses["flow_loss"],
        "clean": output.losses["clean_loss"],
        "rollout": output.losses["rollout_endpoint_loss"],
    }[name]
    assert torch.allclose(output.losses["total_loss"], expected)


def test_rollout_config_rejects_invalid_values():
    stage = DistillationStage(4, 2)
    with pytest.raises(ValueError, match="rollout_steps"):
        ProgressiveDistillationConfig(stage, rollout_steps=-1)
    with pytest.raises(ValueError, match="rollout_every"):
        ProgressiveDistillationConfig(stage, rollout_steps=2, rollout_every=0)
    with pytest.raises(ValueError, match="rollout weight"):
        ProgressiveDistillationConfig(
            stage,
            rollout_steps=2,
            rollout_endpoint_weight=0.0,
            rollout_clean_weight=0.0,
        )
    with pytest.raises(ValueError, match="disabled"):
        ProgressiveDistillationConfig(stage, rollout_endpoint_weight=1.0)


def test_rollout_config_accepts_active_policy():
    config = ProgressiveDistillationConfig(
        DistillationStage(8, 4),
        rollout_steps=4,
        rollout_every=2,
        rollout_endpoint_weight=1.0,
        rollout_clean_weight=0.5,
    )
    assert config.rollout_steps == 4
    assert config.rollout_every == 2


def test_progressive_rollout_adds_finite_losses_and_metrics():
    method, batches = fixture()
    method.config = replace(
        method.config,
        rollout_steps=2,
        rollout_every=1,
        rollout_endpoint_weight=1.0,
        rollout_clean_weight=0.5,
    )
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    output = method.training_step(batch, iteration=1)
    assert output.metrics["rollout_steps"] == 2.0
    assert output.metrics["rollout_index"] == 0.0
    assert torch.isfinite(output.losses["rollout_endpoint_loss"])
    assert torch.isfinite(output.losses["rollout_clean_loss"])
    assert output.losses["rollout_endpoint_loss"].requires_grad
    assert output.losses["rollout_clean_loss"].requires_grad
    assert output.losses["total_loss"].requires_grad


def test_progressive_rollout_is_disabled_without_changing_existing_path():
    method, batches = fixture()
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    output = method.training_step(batch, iteration=1)
    assert output.metrics["rollout_steps"] == 0.0
    assert output.metrics["rollout_index"] == -1.0
    assert output.losses["rollout_endpoint_loss"].item() == 0.0
    assert output.losses["rollout_clean_loss"].item() == 0.0


def test_progressive_rollout_clamps_at_final_student_interval():
    method, batches = fixture()
    method.config = replace(
        method.config,
        rollout_steps=4,
        rollout_every=1,
        rollout_endpoint_weight=1.0,
        rollout_clean_weight=0.5,
    )
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    output = method.training_step(batch, iteration=2)
    assert output.metrics["interval_index"] == 1.0
    assert output.metrics["rollout_steps"] == 1.0


def test_progressive_rollout_frequency_uses_training_iteration():
    method, batches = fixture()
    method.config = replace(
        method.config,
        rollout_steps=2,
        rollout_every=2,
        rollout_endpoint_weight=1.0,
        rollout_clean_weight=0.5,
    )
    method.prepare()
    batch = method.prepare_batch(batches[0], torch.Generator().manual_seed(13))
    first = method.training_step(batch, iteration=1)
    second = method.training_step(batch, iteration=2)
    assert first.metrics["rollout_steps"] == 2.0
    assert second.metrics["rollout_steps"] == 0.0


def test_progressive_resume_matches_uninterrupted(tmp_path):
    full_method, batches = fixture()
    full_engine = TrainerEngine(TrainerConfig(seed=13))
    full_result = full_engine.run(full_method, batches, max_steps=6)

    resumed_method, resumed_batches = fixture()
    first_engine = TrainerEngine(TrainerConfig(seed=13))
    first = first_engine.run(resumed_method, resumed_batches, max_steps=3)
    path = first_engine.save_checkpoint(resumed_method, first.loop_state, tmp_path / "distill.pt")
    resumed_engine = TrainerEngine(TrainerConfig(seed=13))
    resumed = resumed_engine.run(resumed_method, resumed_batches, max_steps=6, resume_from=path)
    assert equal(clone(full_method.student_model), clone(resumed_method.student_model))
    assert resumed.loop_state == full_result.loop_state


def test_video_only_requires_zero_audio_weight():
    method, _ = fixture()
    method.config = replace(method.config, audio_weight=0.0)
    sample = SyntheticH3Dataset(1, 91)[0]
    video_only = replace(sample, latents=ModalLatents(video=sample.latents.video))
    result = TrainerEngine(TrainerConfig(seed=13)).run(method, [[video_only]], max_steps=1)
    assert result.optimizer_steps == {"student": 1}

    invalid, _ = fixture()
    with pytest.raises(TrainingFailure, match="absent audio modality"):
        TrainerEngine(TrainerConfig(seed=13)).run(invalid, [[video_only]], max_steps=1)
