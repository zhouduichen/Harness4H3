import copy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.progressive_distillation import (
    DistillationStage,
    ProgressiveDistillation,
    ProgressiveDistillationConfig,
    plan_binary_stages,
)
from h3_training.data.dataset import SyntheticH3Dataset
from h3_training.data.schema import ModalLatents
from h3_training.engine.state import TrainingFailure
from h3_training.engine.trainer import TrainerConfig, TrainerEngine
from h3_training.tiny.model import TinyH3Model


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
