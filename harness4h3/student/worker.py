"""Fixed H3-to-Student training worker and auditable result records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compiler import CompileManifest
from .model import build_smoke_student, build_student
from .proposal import StudentProposal, StudentTarget
from .quantization import quantize_checkpoint


class StudentTrainingError(RuntimeError):
    """A typed failure that must be returned to the next campaign proposal."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


@dataclass(frozen=True)
class StudentBatch:
    latent: Tensor
    conditioning: Tensor
    timestep: Tensor
    target: Tensor


@dataclass(frozen=True)
class TrainingResult:
    status: str
    proposal_digest: str
    compiler_digest: str
    parent_sha256: Optional[str]
    child_sha256: Optional[str]
    child_checkpoint: Optional[str]
    optimizer_steps: int
    initial_loss: Optional[float]
    final_loss: Optional[float]
    gradient_norm: Optional[float]
    wall_time_s: float
    peak_memory_gb: float
    changed_parameter_count: int
    offline_simulation: bool
    failure_code: Optional[str] = None
    message: str = ""
    full_precision_checkpoint: Optional[str] = None
    quantized_checkpoint: Optional[str] = None
    quantization: str = "none"
    model_size_bytes: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TeacherStudentBackend(Protocol):
    """Trusted implementation boundary; proposal data cannot implement this."""

    offline_simulation: bool

    def load_teacher(self, checkpoint: Path, device: torch.device) -> Any:
        ...

    def build_student(
        self, proposal: StudentProposal, target: StudentTarget, device: torch.device
    ) -> nn.Module:
        ...

    def batches(
        self,
        teacher: Any,
        proposal: StudentProposal,
        target: StudentTarget,
        max_steps: int,
        device: torch.device,
    ) -> Iterable[StudentBatch]:
        ...

    def save_student(self, student: nn.Module, path: Path, metadata: Mapping[str, str]) -> None:
        ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class StudentTrainWorker:
    def __init__(self, backend: TeacherStudentBackend, *, target: Optional[StudentTarget] = None):
        self.backend = backend
        self.target = target or StudentTarget()

    @staticmethod
    def _failure(
        manifest: CompileManifest,
        started: float,
        code: str,
        message: str,
        *,
        parent_sha256: Optional[str] = None,
        offline_simulation: bool = False,
    ) -> TrainingResult:
        return TrainingResult(
            status="failed",
            proposal_digest=manifest.proposal_digest,
            compiler_digest=manifest.manifest_digest,
            parent_sha256=parent_sha256,
            child_sha256=None,
            child_checkpoint=None,
            optimizer_steps=0,
            initial_loss=None,
            final_loss=None,
            gradient_norm=None,
            wall_time_s=time.perf_counter() - started,
            peak_memory_gb=0.0,
            changed_parameter_count=0,
            offline_simulation=offline_simulation,
            failure_code=code,
            message=message,
            quantization="none",
        )

    def run(
        self,
        manifest: CompileManifest,
        teacher_checkpoint: Path,
        output_dir: Path,
        *,
        max_steps: Optional[int] = None,
        device: Optional[str] = None,
        teacher_device: Optional[str] = None,
        student_device: Optional[str] = None,
    ) -> TrainingResult:
        started = time.perf_counter()
        teacher_checkpoint = Path(teacher_checkpoint).resolve()
        output_dir = Path(output_dir).resolve()
        try:
            proposal = StudentProposal.from_dict(manifest.proposal)
            if proposal.digest != manifest.proposal_digest:
                return self._failure(manifest, started, "manifest_digest_mismatch", "proposal digest does not match manifest")
            if not teacher_checkpoint.is_file():
                return self._failure(manifest, started, "teacher_checkpoint_missing", str(teacher_checkpoint), offline_simulation=self.backend.offline_simulation)
            parent_sha256 = sha256_file(teacher_checkpoint)
            selected_student_device = torch.device(student_device or device or ("cuda" if torch.cuda.is_available() else "cpu"))
            selected_teacher_device = torch.device(teacher_device or selected_student_device)
            if (selected_student_device.type == "cuda" or selected_teacher_device.type == "cuda") and not torch.cuda.is_available():
                return self._failure(manifest, started, "device_unavailable", "CUDA is not available", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
            teacher = self.backend.load_teacher(teacher_checkpoint, selected_teacher_device)
            student = self.backend.build_student(proposal, self.target, selected_student_device)
            student.train(True)
            initial_state = {
                name: value.detach().to(device="cpu").clone()
                for name, value in student.state_dict().items()
                if value.is_floating_point()
            }
            optimizer = torch.optim.AdamW(
                [parameter for parameter in student.parameters() if parameter.requires_grad],
                lr=proposal.training.learning_rate,
            )
            if not optimizer.param_groups or not optimizer.param_groups[0]["params"]:
                return self._failure(manifest, started, "no_trainable_parameters", "Student has no trainable parameters", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
            steps = int(max_steps if max_steps is not None else proposal.training.max_steps)
            if steps <= 0:
                return self._failure(manifest, started, "invalid_training_config", "max_steps must be positive", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
            initial_loss = None
            final_loss = None
            maximum_gradient_norm = 0.0
            optimizer_steps = 0
            if selected_student_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected_student_device)
            for batch in self.backend.batches(teacher, proposal, self.target, steps, selected_student_device):
                optimizer.zero_grad(set_to_none=True)
                predicted = student(batch.latent, batch.conditioning, batch.timestep)
                target = batch.target.to(device=predicted.device, dtype=predicted.dtype)
                loss = F.mse_loss(predicted, target)
                if not torch.isfinite(loss):
                    return self._failure(manifest, started, "nonfinite_loss", "Student loss is non-finite", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
                if initial_loss is None:
                    initial_loss = float(loss.detach().cpu())
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).detach().cpu())
                if not math.isfinite(norm) or norm <= 0:
                    return self._failure(manifest, started, "zero_gradient", "Student gradient is zero or non-finite", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
                optimizer.step()
                optimizer_steps += 1
                maximum_gradient_norm = max(maximum_gradient_norm, norm)
                final_loss = float(loss.detach().cpu())
                if optimizer_steps >= steps:
                    break
            if optimizer_steps <= 0 or initial_loss is None or final_loss is None:
                return self._failure(manifest, started, "empty_training_stream", "teacher backend produced no batches", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
            changed = 0
            for name, value in student.state_dict().items():
                before = initial_state.get(name)
                if before is not None and not torch.equal(before, value.detach().to(device="cpu")):
                    changed += 1
            if changed <= 0:
                return self._failure(manifest, started, "unchanged_child", "no Student tensor changed after optimization", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
            output_dir.mkdir(parents=True, exist_ok=True)
            child_path = output_dir / "student.safetensors"
            metadata = {
                "proposal_digest": proposal.digest,
                "compiler_digest": manifest.manifest_digest,
                "teacher_sha256": parent_sha256,
                "architecture_family": proposal.architecture.family,
                "offline_simulation": str(bool(self.backend.offline_simulation)).lower(),
            }
            self.backend.save_student(student, child_path, metadata)
            full_precision_path = child_path
            quantized_path = None
            child_for_evaluation = child_path
            if proposal.deployment.quantization == "int8":
                quantized_path = output_dir / "student-int8.safetensors"
                quantize_checkpoint(
                    child_path,
                    quantized_path,
                    bits=8,
                    metadata={**metadata, "full_precision_sha256": sha256_file(child_path)},
                )
                child_for_evaluation = quantized_path
            elif proposal.deployment.quantization == "int4":
                return self._failure(manifest, started, "quantization_unsupported", "int4 Student quantization is not supported", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
            child_sha256 = sha256_file(child_for_evaluation)
            if child_sha256 == parent_sha256:
                return self._failure(manifest, started, "unchanged_child", "child file hash equals teacher hash", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation)
            peak = torch.cuda.max_memory_allocated(selected_student_device) / float(1024**3) if selected_student_device.type == "cuda" else 0.0
            return TrainingResult(
                status="success",
                proposal_digest=proposal.digest,
                compiler_digest=manifest.manifest_digest,
                parent_sha256=parent_sha256,
                child_sha256=child_sha256,
                child_checkpoint=str(child_for_evaluation),
                optimizer_steps=optimizer_steps,
                initial_loss=initial_loss,
                final_loss=final_loss,
                gradient_norm=maximum_gradient_norm,
                wall_time_s=time.perf_counter() - started,
                peak_memory_gb=float(peak),
                changed_parameter_count=changed,
                offline_simulation=bool(self.backend.offline_simulation),
                full_precision_checkpoint=str(full_precision_path),
                quantized_checkpoint=str(quantized_path) if quantized_path else None,
                quantization=proposal.deployment.quantization,
                model_size_bytes=child_for_evaluation.stat().st_size,
            )
        except StudentTrainingError as exc:
            return self._failure(manifest, started, exc.code, exc.message, offline_simulation=self.backend.offline_simulation)
        except RuntimeError as exc:
            lowered = str(exc).lower()
            code = "training_oom" if "out of memory" in lowered else "training_runtime"
            return self._failure(manifest, started, code, str(exc), offline_simulation=self.backend.offline_simulation)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            return self._failure(manifest, started, "training_config", str(exc), offline_simulation=self.backend.offline_simulation)


class RealH3TeacherBackend:
    """Bridge real H3 cached latents to the registered video Student graph."""

    offline_simulation = False

    def __init__(
        self,
        comfyui_root: Path,
        cache_dir: Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        teacher_targets_dir: Optional[Path] = None,
    ):
        self.comfyui_root = Path(comfyui_root).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.dtype = dtype
        self.teacher_targets_dir = Path(teacher_targets_dir).resolve() if teacher_targets_dir else None
        self.adapter = None

    def load_teacher(self, checkpoint: Path, device: torch.device) -> Any:
        if self.teacher_targets_dir is not None:
            if not self.teacher_targets_dir.is_dir():
                raise StudentTrainingError("teacher_targets_missing", str(self.teacher_targets_dir))
            return {"precomputed_teacher_targets": True}
        from h3_training.adapters.real_h3 import RealMiniMaxH3Adapter

        self.adapter = RealMiniMaxH3Adapter(self.comfyui_root, device=str(device), dtype=self.dtype)
        return self.adapter.load_role(Path(checkpoint), trainable=False)

    def build_student(self, proposal: StudentProposal, target: StudentTarget, device: torch.device) -> nn.Module:
        return build_student(proposal, device=device, target=target).to(dtype=self.dtype)

    def batches(
        self,
        teacher: Any,
        proposal: StudentProposal,
        target: StudentTarget,
        max_steps: int,
        device: torch.device,
    ) -> Iterable[StudentBatch]:
        if self.teacher_targets_dir is not None:
            paths = sorted(self.teacher_targets_dir.glob("*.pt"))
            if not paths:
                raise StudentTrainingError("teacher_targets_missing", str(self.teacher_targets_dir))
            for index in range(max_steps):
                try:
                    raw = torch.load(paths[index % len(paths)], map_location="cpu", weights_only=False)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    raise StudentTrainingError("teacher_targets_corrupt", str(exc)) from exc
                if not isinstance(raw, Mapping):
                    raise StudentTrainingError("teacher_targets_corrupt", "target item must be a mapping")
                required = ("latent", "conditioning", "timestep", "target")
                if any(name not in raw for name in required):
                    raise StudentTrainingError("teacher_targets_corrupt", "target item is missing required tensors")
                yield StudentBatch(
                    latent=torch.as_tensor(raw["latent"]).to(device=device, dtype=self.dtype),
                    conditioning=torch.as_tensor(raw["conditioning"]).to(device=device, dtype=self.dtype),
                    timestep=torch.as_tensor(raw["timestep"]).to(device=device, dtype=torch.float32),
                    target=torch.as_tensor(raw["target"]).to(device=device, dtype=self.dtype),
                )
            return
        if self.adapter is None:
            raise StudentTrainingError("h3_adapter_unavailable", "real H3 adapter was not initialized")
        paths = sorted(self.cache_dir.glob("*.pt"))
        if not paths:
            raise StudentTrainingError("cache_missing", "no H3 cache items under %s" % self.cache_dir)
        generator = torch.Generator(device="cpu").manual_seed(20260920)
        for index in range(max_steps):
            raw = torch.load(paths[index % len(paths)], map_location="cpu", weights_only=False)
            prepared = self.adapter.prepare_batch(raw, generator)
            if prepared.latents is None or prepared.noise is None or prepared.timesteps is None:
                raise StudentTrainingError("invalid_h3_batch", "H3 batch lacks latent/noise/timestep tensors")
            noisy = self.adapter.add_noise(prepared.latents, prepared.noise, prepared.timesteps)
            if noisy.video is None or prepared.timesteps.video is None:
                raise StudentTrainingError("invalid_h3_batch", "H3 batch lacks video tensors")
            with torch.no_grad():
                teacher_prediction = self.adapter.predict(
                    teacher,
                    noisy,
                    prepared.timesteps,
                    prepared.conditioning,
                )
            if teacher_prediction.video is None:
                raise StudentTrainingError("invalid_h3_batch", "H3 teacher returned no video prediction")
            yield StudentBatch(
                latent=noisy.video.to(device=device, dtype=self.dtype),
                conditioning=prepared.conditioning.text.to(device=device, dtype=self.dtype),
                timestep=prepared.timesteps.video.to(device=device, dtype=torch.float32),
                target=teacher_prediction.video.detach().to(device=device, dtype=self.dtype),
            )

    def save_student(self, student: nn.Module, path: Path, metadata: Mapping[str, str]) -> None:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in student.state_dict().items()
        }
        save_file(state, str(path), metadata={str(key): str(value) for key, value in metadata.items()})


__all__ = [
    "RealH3TeacherBackend",
    "StudentBatch",
    "StudentTrainWorker",
    "StudentTrainingError",
    "TrainingResult",
    "sha256_file",
]
