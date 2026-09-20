from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from harness4h3.student.compiler import StudentCompiler
from harness4h3.student.model import build_smoke_student
from harness4h3.student.proposal import StudentProposal, StudentTarget
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
