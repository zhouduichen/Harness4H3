import pytest

torch = pytest.importorskip("torch")

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.algorithms.base import ModelRole
from h3_training.engine.evidence import capture_parent, save_verified_child
from h3_training.engine.state import TrainingFailure
from h3_training.engine.trainer import TrainerEngine
from h3_training.tiny.factory import create_tiny_checkpoint, load_tiny_checkpoint


def role_from(path):
    model, metadata = load_tiny_checkpoint(path)
    return ModelRole("student", model, TinyH3Adapter(), True, dict(metadata))


def test_child_evidence_detects_change_and_reloads(tmp_path):
    parent_path = create_tiny_checkpoint(tmp_path / "parent.pt", model_id="M0000")
    role = role_from(parent_path)
    names = {name for name, _ in role.model.named_parameters() if name.startswith("video_head.")}
    parent = capture_parent(parent_path, role, names)
    role.scheduler_state.update(model_id="M0001", parent_id="M0000")
    role.model.video_head.weight.data.add_(0.01)
    evidence = save_verified_child(role, parent, tmp_path / "child.pt")
    assert evidence.child_sha256 != evidence.parent_sha256
    assert evidence.changed_trainable == 1
    assert evidence.reloaded
    assert evidence.manifest_path.is_file()


def test_unchanged_child_and_parent_mutation_fail(tmp_path):
    parent_path = create_tiny_checkpoint(tmp_path / "parent.pt", model_id="M0000")
    role = role_from(parent_path)
    names = {name for name, _ in role.model.named_parameters() if name.startswith("video_head.")}
    parent = capture_parent(parent_path, role, names)
    with pytest.raises(TrainingFailure, match="unchanged_child"):
        save_verified_child(role, parent, tmp_path / "child.pt")
    parent_path.write_bytes(parent_path.read_bytes() + b"changed")
    role.model.video_head.weight.data.add_(0.01)
    with pytest.raises(TrainingFailure, match="parent_modified"):
        save_verified_child(role, parent, tmp_path / "child.pt")


def test_frozen_tensor_mutation_fails(tmp_path):
    parent_path = create_tiny_checkpoint(tmp_path / "parent.pt", model_id="M0000")
    role = role_from(parent_path)
    trainable = {name for name, _ in role.model.named_parameters() if name.startswith("video_head.")}
    parent = capture_parent(parent_path, role, trainable)
    role.model.video_head.weight.data.add_(0.01)
    role.model.audio_head.weight.data.add_(0.01)
    with pytest.raises(TrainingFailure, match="frozen_tensor_changed"):
        save_verified_child(role, parent, tmp_path / "child.pt")


def test_engine_exposes_verified_child_entrypoint(tmp_path):
    parent_path = create_tiny_checkpoint(tmp_path / "parent.pt", model_id="M0000")
    role = role_from(parent_path)
    trainable = {name for name, _ in role.model.named_parameters() if name.startswith("video_head.")}
    parent = capture_parent(parent_path, role, trainable)
    role.model.video_head.weight.data.add_(0.01)

    class Method:
        student = role

    evidence = TrainerEngine().save_child(Method(), parent, tmp_path / "child.pt")
    assert evidence.reloaded is True


def test_child_reload_failure_is_stable(tmp_path):
    class BrokenReloadAdapter(TinyH3Adapter):
        def reload_role(self, path):
            raise ValueError("deliberate reload failure")

    parent_path = create_tiny_checkpoint(tmp_path / "parent.pt", model_id="M0000")
    model, metadata = load_tiny_checkpoint(parent_path)
    role = ModelRole("student", model, BrokenReloadAdapter(), True, dict(metadata))
    trainable = {name for name, _ in role.model.named_parameters() if name.startswith("video_head.")}
    parent = capture_parent(parent_path, role, trainable)
    role.model.video_head.weight.data.add_(0.01)
    with pytest.raises(TrainingFailure, match="child_reload_failed"):
        save_verified_child(role, parent, tmp_path / "child.pt")
