from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from harness4h3.student.compiler import StudentCompiler
from harness4h3.student.model import build_smoke_student
from harness4h3.student.proposal import StudentProposal, StudentTarget
from harness4h3.student.teacher_service import TeacherService
from harness4h3.student.worker import StudentBatch, StudentTrainWorker
from tests.unit.test_student_proposal import valid_payload


class FakeBackend:
    offline_simulation = True

    def load_teacher(self, checkpoint, device):
        return torch.nn.Linear(1, 1, bias=False).to(device)

    def build_student(self, proposal, target, device):
        return build_smoke_student(proposal, scale=0.125, target=target).to(device)

    def batches(self, teacher, proposal, target, max_steps, device):
        generator = torch.Generator().manual_seed(7)
        for _ in range(max_steps):
            latent = torch.randn(
                (1, target.latent_channels, target.latent_frames, target.latent_height, target.latent_width),
                generator=generator,
                device=device,
            )
            conditioning = torch.randn((1, 2, target.condition_dim), generator=generator, device=device)
            timestep = torch.zeros((1,), device=device)
            # The target is detached and deterministic; optimizer updates the
            # same graph used by the production worker.
            with torch.no_grad():
                target_value = torch.zeros_like(latent)
            yield StudentBatch(latent, conditioning, timestep, target_value)

    def save_student(self, student, path, metadata):
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in student.state_dict().items()},
            str(path),
            metadata={str(key): str(value) for key, value in metadata.items()},
        )


class NonShardedProductionBackend(FakeBackend):
    offline_simulation = False

    def load_teacher(self, checkpoint, device, *, teacher_devices=(), teacher_world_size=1):
        del checkpoint, device, teacher_devices, teacher_world_size
        return TeacherService.from_predictors([lambda noisy, timestep, conditioning: None])


def compile_manifest(tmp_path):
    proposal = StudentProposal.from_dict(valid_payload())
    return StudentCompiler(StudentTarget()).compile(proposal, tmp_path / "compile")


def test_worker_writes_changed_child_and_training_evidence(tmp_path):
    teacher = tmp_path / "teacher.safetensors"
    save_file({"teacher": torch.ones(1)}, str(teacher))
    result = StudentTrainWorker(FakeBackend()).run(
        compile_manifest(tmp_path), teacher, tmp_path / "child", max_steps=2, device="cpu"
    )
    assert result.status == "success"
    assert Path(result.child_checkpoint).is_file()
    assert result.parent_sha256 != result.child_sha256
    assert result.optimizer_steps >= 1
    assert result.offline_simulation is True
    assert result.changed_parameter_count > 0
    assert result.quantization == "int8"
    assert Path(result.full_precision_checkpoint).is_file()
    assert Path(result.quantized_checkpoint).is_file()


def test_worker_refuses_manifest_digest_mismatch(tmp_path):
    manifest = compile_manifest(tmp_path)
    path = Path(manifest.path)
    raw = path.read_text(encoding="utf-8").replace('"proposal_digest": "', '"proposal_digest": "tampered-')
    path.write_text(raw, encoding="utf-8")
    teacher = tmp_path / "teacher.safetensors"
    save_file({"teacher": torch.ones(1)}, str(teacher))
    try:
        StudentTrainWorker(FakeBackend()).run(
            type(manifest).from_path(path), teacher, tmp_path / "child", max_steps=1, device="cpu"
        )
    except ValueError as exc:
        assert "manifest_digest_mismatch" in str(exc)
    else:
        raise AssertionError("tampered manifest should not load")


def test_production_worker_rejects_non_sharded_teacher(tmp_path):
    teacher = tmp_path / "teacher.safetensors"
    save_file({"teacher": torch.ones(1)}, str(teacher))

    result = StudentTrainWorker(NonShardedProductionBackend()).run(
        compile_manifest(tmp_path), teacher, tmp_path / "child", max_steps=1, device="cpu"
    )

    assert result.status == "failed"
    assert result.failure_code == "teacher_not_sharded"
    assert result.offline_simulation is False


def test_progressive_worker_publishes_each_child_as_next_teacher(tmp_path):
    payload = valid_payload()
    payload["training"].update({"method": "progressive_distillation", "source_steps": 4, "target_steps": 1})
    proposal = StudentProposal.from_dict(payload)
    manifest = StudentCompiler(StudentTarget()).compile(proposal, tmp_path / "progressive-compile")
    teacher = tmp_path / "teacher.safetensors"
    save_file({"teacher": torch.ones(1)}, str(teacher))

    result = StudentTrainWorker(FakeBackend()).run(
        manifest, teacher, tmp_path / "progressive-child", max_steps=2, device="cpu"
    )

    assert result.status == "success"
    assert len(result.stage_lineage) == 2
    assert result.stage_lineage[0]["student_nfe"] == 2
    assert result.stage_lineage[1]["teacher_checkpoint"] == result.stage_lineage[0]["student_checkpoint"]
    assert result.stage_lineage[0]["promoted_as_next_teacher"] is True
    assert result.stage_lineage[1]["promoted_as_next_teacher"] is False
    assert all(Path(item["student_checkpoint"]).is_file() for item in result.stage_lineage)


def test_progressive_stages_share_the_candidate_fidelity_budget(tmp_path):
    payload = valid_payload()
    payload["training"].update({"method": "progressive_distillation", "source_steps": 8, "target_steps": 2})
    proposal = StudentProposal.from_dict(payload)
    manifest = StudentCompiler(StudentTarget()).compile(proposal, tmp_path / "progressive-budget-compile")
    teacher = tmp_path / "teacher.safetensors"
    save_file({"teacher": torch.ones(1)}, str(teacher))

    result = StudentTrainWorker(FakeBackend()).run(
        manifest, teacher, tmp_path / "progressive-budget-child", train_steps=6, max_steps=6, device="cpu"
    )

    assert result.status == "success"
    assert result.train_steps == 6
    assert result.cumulative_train_steps == 6
    assert sum(int(item["stage_train_steps"]) for item in result.stage_lineage) == result.train_steps
    assert result.optimizer_steps == sum(result.optimizer_steps_by_role.values())
    assert [item["cumulative_train_steps"] for item in result.stage_lineage] == [3, 6]


def test_dmd2_reports_one_candidate_budget_and_real_role_optimizer_steps(tmp_path):
    proposal = StudentProposal.from_dict(valid_payload())
    manifest = StudentCompiler(StudentTarget()).compile(proposal, tmp_path / "dmd2-budget-compile")
    teacher = tmp_path / "teacher.safetensors"
    save_file({"teacher": torch.ones(1)}, str(teacher))

    result = StudentTrainWorker(FakeBackend()).run(
        manifest, teacher, tmp_path / "dmd2-budget-child", train_steps=4, max_steps=4, device="cpu"
    )

    assert result.status == "success"
    assert result.train_steps == 4
    assert result.cumulative_train_steps == 4
    assert result.optimizer_steps_by_role == {"critic": 4, "student": 2}
    assert result.optimizer_steps == 6
    assert result.stage_lineage[0]["stage_train_steps"] == result.train_steps
