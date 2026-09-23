"""Fixed H3-to-Student training worker and auditable result records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import copy
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from h3_training.adapters.base import DenoisingModelAdapter
from h3_training.algorithms.dmd2 import DMD2, DMD2Config
from h3_training.algorithms.progressive_distillation import (
    DistillationStage,
    ProgressiveDistillation,
    ProgressiveDistillationConfig,
    plan_binary_stages,
)
from h3_training.data.schema import (
    Conditioning,
    ModalInterval,
    ModalLatents,
    ModalPrediction,
    ModalSchedule,
    ModalTimesteps,
    PreparedBatch,
)
from h3_training.engine.state import TrainingFailure
from h3_training.engine.trainer import TrainerConfig, TrainerEngine

from .compiler import CompileManifest
from .model import build_smoke_student, build_student
from .proposal import StudentProposal, StudentTarget
from .quantization import quantize_checkpoint
from .teacher_service import TeacherService, TeacherServiceHandle


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
    target: Optional[Tensor] = None
    audio_latent: Optional[Tensor] = None
    audio_noise: Optional[Tensor] = None
    noise: Optional[Tensor] = None
    audio_timestep: Optional[Tensor] = None


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
    full_precision_sha256: Optional[str] = None
    quantized_checkpoint: Optional[str] = None
    quantization: str = "none"
    model_size_bytes: Optional[int] = None
    parent_kind: str = "teacher_seed"
    parent_checkpoint: Optional[str] = None
    parent_inherited: bool = False
    inherited_parameter_count: int = 0
    algorithm_name: Optional[str] = None
    algorithm_path: Optional[str] = None
    algorithm_dispatch: str = "not_started"
    fidelity: str = "F1"
    train_steps: int = 0
    cumulative_train_steps: int = 0
    evaluation_cases: int = 0
    seed_count: int = 0
    verifier_strength: str = ""
    timeout_s: float = 0.0
    gpu_budget: int = 0
    total_parameter_count: int = 0
    inheritance_ratio: float = 0.0
    initialization_mode: str = "fresh_init"
    teacher_world_size: int = 1
    teacher_devices: tuple[str, ...] = ()
    student_device: str = ""
    teacher_ranks_used: tuple[int, ...] = ()
    online_forward: bool = False
    teacher_sharded: bool = False
    teacher_forward_count: int = 0
    teacher_peak_memory_gb: float = 0.0
    teacher_rank_forward_counts: tuple[tuple[int, int], ...] = ()
    stage_lineage: tuple[Mapping[str, Any], ...] = ()
    optimizer_steps_by_role: Mapping[str, int] = field(default_factory=dict)
    runtime_resource_gate_passed: bool = False
    estimated_training_peak_memory_gb: float = 0.0
    student_free_memory_gb: float = 0.0
    student_memory_safety_margin_gb: float = 0.0
    gpu_allocation: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TeacherStudentBackend(Protocol):
    """Trusted implementation boundary; proposal data cannot implement this."""

    offline_simulation: bool

    def load_teacher(
        self,
        checkpoint: Path,
        device: torch.device,
        *,
        teacher_devices: Sequence[str] = (),
        teacher_world_size: int = 1,
    ) -> Any:
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


class StudentAlgorithmAdapter(DenoisingModelAdapter):
    """Adapt the existing H3 algorithms to the registered Student graph.

    The real H3 worker supplies a loaded teacher role.  Teacher predictions
    are made inside the algorithm step for the current noisy latent and
    timestep; a detached target is retained only for explicitly offline test
    backends, whose TrainingResult is marked ``offline_simulation``.
    """

    def __init__(
        self,
        *,
        teacher_adapter: Optional[DenoisingModelAdapter] = None,
        teacher_role: Any = None,
        teacher_predictor: Optional[TeacherServiceHandle] = None,
        use_role_model_for_teacher: bool = False,
    ):
        self.teacher_adapter = teacher_adapter
        self.teacher_role = teacher_role
        self.teacher_predictor = teacher_predictor
        self.use_role_model_for_teacher = bool(use_role_model_for_teacher)
        self._teacher_target: Optional[torch.Tensor] = None
        self._teacher_audio_clean: Optional[torch.Tensor] = None
        self._teacher_audio_noise: Optional[torch.Tensor] = None
        self._teacher_audio_timestep: Optional[torch.Tensor] = None

    def prepare_batch(self, raw: Any, generator: torch.Generator) -> PreparedBatch:
        if isinstance(raw, PreparedBatch):
            return raw
        if not isinstance(raw, StudentBatch):
            raise TrainingFailure("invalid_training_config", "Student algorithm batches must be StudentBatch values")
        self._teacher_target = raw.target.detach() if raw.target is not None else None
        self._teacher_audio_clean = raw.audio_latent.detach() if raw.audio_latent is not None else None
        self._teacher_audio_noise = raw.audio_noise.detach() if raw.audio_noise is not None else None
        self._teacher_audio_timestep = raw.audio_timestep.detach() if raw.audio_timestep is not None else None
        noise = raw.noise if raw.noise is not None else torch.randn_like(raw.latent)
        audio_noise = raw.audio_noise
        return PreparedBatch(
            conditioning=Conditioning(raw.conditioning),
            latents=ModalLatents(video=raw.latent, audio=raw.audio_latent),
            noise=ModalLatents(video=noise, audio=audio_noise),
            timesteps=ModalTimesteps(video=raw.timestep, audio=raw.audio_timestep),
        )

    @staticmethod
    def _broadcast(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        value = value.to(device=reference.device, dtype=reference.dtype)
        while value.ndim < reference.ndim:
            value = value.unsqueeze(-1)
        return value

    def add_noise(self, clean: ModalLatents, noise: ModalLatents, timestep: ModalTimesteps) -> ModalLatents:
        video = None
        audio = None
        if clean.video is not None:
            if noise.video is None or timestep.video is None:
                raise TrainingFailure("invalid_training_config", "Student video noise inputs are incomplete")
            value = self._broadcast(timestep.video, clean.video)
            video = value * clean.video + (1.0 - value) * noise.video
        if clean.audio is not None:
            if noise.audio is None or timestep.audio is None:
                raise TrainingFailure("invalid_training_config", "Student audio noise inputs are incomplete")
            value = self._broadcast(timestep.audio, clean.audio)
            audio = value * clean.audio + (1.0 - value) * noise.audio
        if video is None and audio is None:
            raise TrainingFailure("invalid_training_config", "Student algorithms require a latent")
        return ModalLatents(video=video, audio=audio)

    def predict(self, role: Any, noisy: ModalLatents, timestep: ModalTimesteps, conditioning: Conditioning) -> ModalPrediction:
        if noisy.video is None or timestep.video is None:
            raise TrainingFailure("invalid_training_config", "Student algorithms require a video prediction")
        if getattr(role, "name", "") == "teacher":
            if self.use_role_model_for_teacher:
                teacher_model = getattr(role, "model", None)
                if teacher_model is None or not callable(teacher_model):
                    raise TrainingFailure("teacher_predictor_missing", "promoted Student teacher has no callable model")
                output = teacher_model(noisy.video, conditioning.text, timestep.video)
                if not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(noisy.video.shape):
                    raise TrainingFailure("teacher_prediction_shape_mismatch", "promoted Student teacher output does not match latent shape")
                return ModalPrediction(video=output)
            if self.teacher_predictor is not None:
                prediction = self.teacher_predictor.predict(noisy, timestep, conditioning)
                return ModalPrediction(
                    video=prediction.video.to(device=noisy.video.device, dtype=noisy.video.dtype)
                    if prediction.video is not None else None,
                    audio=prediction.audio.to(device=noisy.video.device, dtype=noisy.video.dtype)
                    if prediction.audio is not None else None,
                )
            if self.teacher_adapter is not None and self.teacher_role is not None:
                full_noisy = noisy
                if full_noisy.audio is None and self._teacher_audio_clean is not None:
                    audio_clean = self._teacher_audio_clean.to(device=noisy.video.device, dtype=noisy.video.dtype)
                    audio_noise = self._teacher_audio_noise
                    audio_timestep = timestep.audio if timestep.audio is not None else timestep.video
                    if audio_noise is not None and audio_timestep is not None:
                        audio_noise = audio_noise.to(device=noisy.video.device, dtype=noisy.video.dtype)
                        audio_timestep = audio_timestep.to(device=noisy.video.device)
                        full_noisy = ModalLatents(
                            video=noisy.video,
                            audio=self.add_noise(
                                ModalLatents(audio=audio_clean),
                                ModalLatents(audio=audio_noise),
                                ModalTimesteps(audio=audio_timestep),
                            ).audio,
                        )
                prediction = self.teacher_adapter.predict(self.teacher_role, full_noisy, timestep, conditioning)
                if prediction.video is None:
                    raise TrainingFailure("teacher_prediction_missing", "real H3 teacher returned no video prediction")
                return ModalPrediction(video=prediction.video)
            if self._teacher_target is not None:
                target = self._teacher_target.to(device=noisy.video.device, dtype=noisy.video.dtype)
                if tuple(target.shape) != tuple(noisy.video.shape):
                    raise TrainingFailure("teacher_target_shape_mismatch", str(tuple(target.shape)))
                return ModalPrediction(video=target)
            raise TrainingFailure("teacher_predictor_missing", "a real teacher predictor is required")
        output = role.model(noisy.video, conditioning.text, timestep.video)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(noisy.video.shape):
            raise TrainingFailure("student_forward_shape_mismatch", "Student algorithm output does not match latent shape")
        return ModalPrediction(video=output)

    def prediction_to_clean(self, noisy: ModalLatents, prediction: ModalPrediction, timestep: ModalTimesteps) -> ModalLatents:
        if noisy.video is None or prediction.video is None or timestep.video is None:
            raise TrainingFailure("invalid_training_config", "Student prediction cannot be reconstructed")
        # DMD2's reference implementation samples t=0 for its generator
        # update. Keep the endpoint differentiable at that boundary while
        # retaining the usual t-dependent interpolation for other steps.
        return ModalLatents(
            video=noisy.video + (1.0 - self._broadcast(timestep.video, noisy.video)) * prediction.video
        )

    def scheduler_step(self, role: Any, latent: ModalLatents, prediction: ModalPrediction, interval: ModalInterval) -> ModalLatents:
        if latent.video is None or prediction.video is None or interval.video is None:
            raise TrainingFailure("invalid_training_config", "Student scheduler requires a video interval")
        start, end = interval.video
        return ModalLatents(video=latent.video + (float(start) - float(end)) * prediction.video)

    def schedule(self, num_model_evaluations: int) -> ModalSchedule:
        if int(num_model_evaluations) <= 0:
            raise TrainingFailure("invalid_training_config", "Student NFE must be positive")
        values = tuple(1.0 - index / float(num_model_evaluations) for index in range(num_model_evaluations + 1))
        return ModalSchedule(video_sigmas=values)

    def save_role(self, role: Any, path: Path) -> Mapping[str, Any]:
        raise TrainingFailure("invalid_training_config", "Student roles are published by the trusted backend")

    def reload_role(self, path: Path) -> Any:
        raise TrainingFailure("invalid_training_config", "Student roles are reloaded by the trusted backend")

    def resolve_trainable_parameters(self, role: Any, policy: str) -> Iterable[str]:
        names = [name for name, _ in role.model.named_parameters()]
        if policy == "all":
            return names
        prefixes = tuple(item.strip() for item in str(policy).split(",") if item.strip())
        return [name for name in names if name.startswith(prefixes)]


def fidelity_step_budget(max_steps: int, fidelity: str) -> int:
    """Return the cumulative candidate training budget for one fidelity tier."""

    value = int(max_steps)
    if value <= 0:
        raise ValueError("max_steps must be positive")
    normalized = str(fidelity).strip().upper()
    if normalized not in {"F1", "F2", "F3"}:
        raise ValueError("unsupported fidelity: %s" % fidelity)
    divisor = {"F1": 4, "F2": 2, "F3": 1}[normalized]
    return max(1, value // divisor)


def progressive_stage_budgets(total_steps: int, stage_count: int) -> tuple[int, ...]:
    """Split one candidate fidelity budget across progressive stages."""

    total = int(total_steps)
    count = int(stage_count)
    if total <= 0 or count <= 0 or total < count:
        raise ValueError("progressive fidelity budget must cover every stage")
    base, remainder = divmod(total, count)
    return tuple(base + (1 if index < remainder else 0) for index in range(count))


@dataclass(frozen=True)
class FidelitySpec:
    name: str
    train_steps: int
    cumulative_train_steps: int
    evaluation_cases: int
    seed_count: int
    verifier_strength: str
    timeout_s: float
    gpu_budget: int = 1


def fidelity_spec(max_steps: int, fidelity: str) -> FidelitySpec:
    name = str(fidelity).strip().upper()
    if name not in {"F1", "F2", "F3"}:
        raise ValueError("unsupported fidelity: %s" % fidelity)
    cumulative = fidelity_step_budget(max_steps, name)
    previous = fidelity_step_budget(max_steps, {"F1": "F1", "F2": "F1", "F3": "F2"}[name]) if name != "F1" else 0
    return FidelitySpec(
        name=name,
        train_steps=max(1, cumulative - previous),
        cumulative_train_steps=cumulative,
        evaluation_cases={"F1": 1, "F2": 4, "F3": 16}[name],
        seed_count={"F1": 1, "F2": 2, "F3": 4}[name],
        verifier_strength={"F1": "cheap", "F2": "semantic", "F3": "full"}[name],
        timeout_s={"F1": 900.0, "F2": 1800.0, "F3": 3600.0}[name],
        gpu_budget={"F1": 1, "F2": 1, "F3": 2}[name],
    )


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
        parent_kind: str = "teacher_seed",
        parent_checkpoint: Optional[str] = None,
        parent_inherited: bool = False,
        inherited_parameter_count: int = 0,
        algorithm_name: Optional[str] = None,
        algorithm_path: Optional[str] = None,
        algorithm_dispatch: str = "failed",
        fidelity: str = "F1",
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
            parent_kind=parent_kind,
            parent_checkpoint=parent_checkpoint,
            parent_inherited=parent_inherited,
            inherited_parameter_count=int(inherited_parameter_count),
            algorithm_name=algorithm_name,
            algorithm_path=algorithm_path,
            algorithm_dispatch=algorithm_dispatch,
            fidelity=str(fidelity),
        )

    @staticmethod
    def _inherit_parent(student: nn.Module, parent_checkpoint: Path) -> tuple[str, int, int]:
        parent_checkpoint = Path(parent_checkpoint).resolve()
        if not parent_checkpoint.is_file():
            raise StudentTrainingError("parent_checkpoint_missing", str(parent_checkpoint))
        try:
            from .quantization import load_student_state

            state, _metadata = load_student_state(parent_checkpoint)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise StudentTrainingError("parent_checkpoint_corrupt", str(exc)) from exc
        current = student.state_dict()
        matched = {
            name: value.to(dtype=current[name].dtype)
            for name, value in state.items()
            if name in current and tuple(value.shape) == tuple(current[name].shape)
        }
        parameter_names = {name for name, _ in student.named_parameters()}
        total = sum(int(parameter.numel()) for _, parameter in student.named_parameters())
        if not matched:
            raise StudentTrainingError("parent_checkpoint_incompatible", "no Student tensors match the proposed graph")
        with torch.no_grad():
            for name, value in matched.items():
                current[name].copy_(value.to(device=current[name].device))
        inherited = sum(int(value.numel()) for name, value in matched.items() if name in parameter_names)
        return sha256_file(parent_checkpoint), inherited, total

    @staticmethod
    def _algorithm_runs(
        proposal: StudentProposal,
        student: nn.Module,
        teacher: Any,
        adapter: StudentAlgorithmAdapter,
    ) -> tuple[list[Any], str]:
        teacher_model = getattr(teacher, "model", teacher)
        method = proposal.training.method
        if method == "progressive_distillation":
            stages = plan_binary_stages(proposal.training.source_steps, proposal.training.target_steps)
            return [
                ProgressiveDistillation(
                    student,
                    teacher_model,
                    adapter,
                    ProgressiveDistillationConfig(
                        stage=stage,
                        learning_rate=proposal.training.learning_rate,
                        audio_weight=0.0,
                    ),
                )
                for stage in stages
            ], "h3_training.algorithms.progressive_distillation.ProgressiveDistillation"
        if method == "dmd2":
            critic = copy.deepcopy(student)
            return [
                DMD2(
                    student,
                    teacher_model,
                    critic,
                    adapter,
                    DMD2Config(
                        student_learning_rate=proposal.training.learning_rate,
                        critic_learning_rate=proposal.training.critic_learning_rate,
                        data_mode="real_latent",
                        regression_weight=0.1,
                    ),
                )
            ], "h3_training.algorithms.dmd2.DMD2"
        raise StudentTrainingError("algorithm_unsupported", method)

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
        teacher_devices: Sequence[str] = (),
        teacher_world_size: int = 1,
        train_steps: Optional[int] = None,
        parent_checkpoint: Optional[Path] = None,
        parent_candidate_id: Optional[str] = None,
        parent_checkpoint_sha256: Optional[str] = None,
        fidelity: str = "F1",
    ) -> TrainingResult:
        started = time.perf_counter()
        teacher_checkpoint = Path(teacher_checkpoint).resolve()
        output_dir = Path(output_dir).resolve()
        parent_sha256: Optional[str] = None
        inherited_parameter_count = 0
        total_parameter_count = 0
        parent_kind = "teacher_seed"
        inherit_parent = False
        algorithm_name: Optional[str] = None
        algorithm_path: Optional[str] = None
        dispatch = "not_started"
        teacher_service: Optional[TeacherServiceHandle] = None
        try:
            proposal = StudentProposal.from_dict(manifest.proposal)
            algorithm_name = proposal.training.method
            if proposal.digest != manifest.proposal_digest:
                return self._failure(manifest, started, "manifest_digest_mismatch", "proposal digest does not match manifest", algorithm_name=algorithm_name, fidelity=fidelity)
            if not teacher_checkpoint.is_file():
                return self._failure(manifest, started, "teacher_checkpoint_missing", str(teacher_checkpoint), offline_simulation=self.backend.offline_simulation, algorithm_name=algorithm_name, fidelity=fidelity)
            teacher_sha256 = sha256_file(teacher_checkpoint)
            selected_student_device = torch.device(student_device or device or ("cuda" if torch.cuda.is_available() else "cpu"))
            selected_teacher_device = torch.device(teacher_device or selected_student_device)
            selected_teacher_names = tuple(str(item) for item in teacher_devices)
            if selected_student_device.type == "cuda" and str(selected_student_device) in selected_teacher_names:
                return self._failure(
                    manifest,
                    started,
                    "teacher_student_gpu_overlap",
                    "Student GPU overlaps a Teacher GPU",
                    parent_sha256=teacher_sha256,
                    offline_simulation=self.backend.offline_simulation,
                    algorithm_name=algorithm_name,
                    fidelity=fidelity,
                )
            if (selected_student_device.type == "cuda" or selected_teacher_device.type == "cuda") and not torch.cuda.is_available():
                return self._failure(manifest, started, "device_unavailable", "CUDA is not available", parent_sha256=teacher_sha256, offline_simulation=self.backend.offline_simulation, algorithm_name=algorithm_name, fidelity=fidelity)
            load_teacher = self.backend.load_teacher
            load_parameters = inspect.signature(load_teacher).parameters
            load_kwargs = {}
            if "teacher_devices" in load_parameters:
                load_kwargs["teacher_devices"] = tuple(str(item) for item in teacher_devices)
            if "teacher_world_size" in load_parameters:
                load_kwargs["teacher_world_size"] = int(teacher_world_size)
            teacher = load_teacher(teacher_checkpoint, selected_teacher_device, **load_kwargs)
            if not self.backend.offline_simulation:
                if not isinstance(teacher, TeacherServiceHandle) or not teacher.sharded:
                    raise StudentTrainingError(
                        "teacher_not_sharded",
                        "production real H3 training requires the collective FSDP Teacher service",
                    )
                if teacher.world_size != TeacherService.REQUIRED_WORLD_SIZE or len(teacher.devices) != TeacherService.REQUIRED_WORLD_SIZE:
                    raise StudentTrainingError(
                        "teacher_world_size_mismatch",
                        "production real H3 training requires exactly three Teacher ranks",
                    )
                if str(selected_student_device) in set(teacher.devices):
                    raise StudentTrainingError("teacher_student_gpu_overlap", "Student GPU overlaps a Teacher GPU")
            teacher_service = teacher if isinstance(teacher, TeacherServiceHandle) else None
            student = self.backend.build_student(proposal, self.target, selected_student_device)
            student.train(True)
            # M0000 is the immutable teacher seed. Later fidelity stages and
            # later campaign rounds must load a Student child checkpoint.
            inherit_parent = parent_checkpoint is not None and str(parent_candidate_id or "") != "M0000"
            if inherit_parent:
                parent_kind = "student_checkpoint"
                parent_sha256, inherited_parameter_count, total_parameter_count = self._inherit_parent(student, Path(parent_checkpoint))
            else:
                parent_sha256 = teacher_sha256
                total_parameter_count = sum(int(parameter.numel()) for parameter in student.parameters())
            if parent_checkpoint_sha256 is not None and str(parent_checkpoint_sha256) != str(parent_sha256):
                raise StudentTrainingError(
                    "parent_checkpoint_integrity",
                    "parent checkpoint SHA256 does not match the selected parent evidence",
                )
            spec = fidelity_spec(int(max_steps if max_steps is not None else 256), fidelity)
            configured_steps = int(train_steps if train_steps is not None else (max_steps if max_steps is not None else 256))
            if configured_steps <= 0:
                raise StudentTrainingError("invalid_training_config", "train_steps must be positive")
            steps = configured_steps
            uses_declared_fidelity_budget = train_steps is not None and steps == spec.train_steps
            cumulative_train_steps = spec.cumulative_train_steps if uses_declared_fidelity_budget else steps
            # DMD2 alternates critic and Student updates; one optimizer
            # iteration would exercise only the critic and cannot publish a
            # changed Student checkpoint.
            if proposal.training.method == "dmd2" and steps < 2:
                raise StudentTrainingError("invalid_training_config", "DMD2 requires at least two training iterations")
            adapter = StudentAlgorithmAdapter(
                teacher_adapter=getattr(self.backend, "adapter", None),
                teacher_role=teacher,
                teacher_predictor=teacher if isinstance(teacher, TeacherServiceHandle) else None,
            )
            algorithm_name = proposal.training.method
            initial_state = {
                name: value.detach().to(device="cpu").clone()
                for name, value in student.state_dict().items()
                if value.is_floating_point()
            }
            total_steps = 0
            initial_loss = None
            final_loss = None
            maximum_gradient_norm = 0.0
            stage_lineage: list[Mapping[str, Any]] = []
            optimizer_steps_by_role: dict[str, int] = {}
            output_dir.mkdir(parents=True, exist_ok=True)
            if selected_student_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected_student_device)
            if proposal.training.method == "progressive_distillation":
                algorithm_path = "h3_training.algorithms.progressive_distillation.ProgressiveDistillation"
                stages = plan_binary_stages(proposal.training.source_steps, proposal.training.target_steps)
                stage_budgets = progressive_stage_budgets(steps, len(stages))
                active_teacher = teacher
                active_adapter = adapter
                active_teacher_checkpoint = teacher_checkpoint
                consumed_stage_steps = 0
                for stage_index, stage in enumerate(stages):
                    stage_budget = stage_budgets[stage_index]
                    stage_method = ProgressiveDistillation(
                        student,
                        getattr(active_teacher, "model", active_teacher),
                        active_adapter,
                        ProgressiveDistillationConfig(
                            stage=stage,
                            learning_rate=proposal.training.learning_rate,
                            audio_weight=0.0,
                        ),
                    )
                    result = TrainerEngine(
                        TrainerConfig(
                            seed=20260920 + stage_index,
                            parent_sha256=parent_sha256 or "",
                            device=str(selected_student_device),
                        )
                    ).run(
                        stage_method,
                        self.backend.batches(active_teacher, proposal, self.target, stage_budget, selected_student_device),
                        max_steps=stage_budget,
                    )
                    dispatch = "executed"
                    stage_optimizer_steps = sum(int(value) for value in result.optimizer_steps.values())
                    total_steps += stage_optimizer_steps
                    for role, count in result.optimizer_steps.items():
                        optimizer_steps_by_role[str(role)] = optimizer_steps_by_role.get(str(role), 0) + int(count)
                    initial_loss = result.initial_loss if initial_loss is None else initial_loss
                    final_loss = result.final_loss
                    maximum_gradient_norm = max(maximum_gradient_norm, result.max_gradient_norm)
                    stage_path = output_dir / ("student-stage-%02d.safetensors" % (stage_index + 1))
                    self.backend.save_student(
                        student,
                        stage_path,
                        {
                            "proposal_digest": proposal.digest,
                            "compiler_digest": manifest.manifest_digest,
                            "algorithm_name": proposal.training.method,
                            "stage_index": str(stage_index),
                            "teacher_nfe": str(stage.teacher_nfe),
                            "student_nfe": str(stage.student_nfe),
                            "teacher_checkpoint": str(active_teacher_checkpoint),
                            "promoted_teacher": "true",
                        },
                    )
                    stage_sha256 = sha256_file(stage_path)
                    stage_teacher_sha256 = sha256_file(Path(active_teacher_checkpoint))
                    stage_lineage.append(
                        {
                            "stage_index": stage_index,
                            "stage_parent_checkpoint": str(active_teacher_checkpoint),
                            "teacher_checkpoint": str(active_teacher_checkpoint),
                            "stage_teacher_sha256": stage_teacher_sha256,
                            "teacher_nfe": stage.teacher_nfe,
                            "student_checkpoint": str(stage_path),
                            "stage_child_sha256": stage_sha256,
                            "student_sha256": stage_sha256,
                            "student_nfe": stage.student_nfe,
                            "stage_train_steps": stage_budget,
                            "stage_optimizer_steps": stage_optimizer_steps,
                            "stage_optimizer_steps_by_role": {
                                str(role): int(count) for role, count in result.optimizer_steps.items()
                            },
                            "cumulative_train_steps": (
                                (cumulative_train_steps - steps)
                                + consumed_stage_steps + stage_budget
                            ),
                            "promoted_as_next_teacher": stage_index < len(stages) - 1,
                        }
                    )
                    consumed_stage_steps += stage_budget
                    if stage_index < len(stages) - 1:
                        promoted_teacher = copy.deepcopy(student).eval()
                        for parameter in promoted_teacher.parameters():
                            parameter.requires_grad_(False)
                        active_teacher = promoted_teacher
                        active_teacher_checkpoint = stage_path
                        active_adapter = StudentAlgorithmAdapter(use_role_model_for_teacher=True)
            else:
                methods, algorithm_path = self._algorithm_runs(proposal, student, teacher, adapter)
                for method in methods:
                    result = TrainerEngine(
                        TrainerConfig(
                            seed=20260920,
                            parent_sha256=parent_sha256 or "",
                            device=str(selected_student_device),
                        )
                    ).run(
                        method,
                        self.backend.batches(teacher, proposal, self.target, steps, selected_student_device),
                        max_steps=steps,
                    )
                    dispatch = "executed"
                    total_steps += sum(int(value) for value in result.optimizer_steps.values())
                    for role, count in result.optimizer_steps.items():
                        optimizer_steps_by_role[str(role)] = optimizer_steps_by_role.get(str(role), 0) + int(count)
                    initial_loss = result.initial_loss if initial_loss is None else initial_loss
                    final_loss = result.final_loss
                    maximum_gradient_norm = max(maximum_gradient_norm, result.max_gradient_norm)
                    stage_lineage.append(
                        {
                            "stage_index": 0,
                            "stage_train_steps": steps,
                            "stage_optimizer_steps": sum(int(value) for value in result.optimizer_steps.values()),
                            "stage_optimizer_steps_by_role": {
                                str(role): int(count) for role, count in result.optimizer_steps.items()
                            },
                            "cumulative_train_steps": cumulative_train_steps,
                        }
                    )
            changed = sum(
                1
                for name, value in student.state_dict().items()
                if name in initial_state and not torch.equal(initial_state[name], value.detach().to(device="cpu"))
            )
            if total_steps <= 0 or initial_loss is None or final_loss is None:
                if teacher_service is not None:
                    teacher_service.close()
                return self._failure(manifest, started, "empty_training_stream", "algorithm produced no optimizer steps", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)
            if changed <= 0:
                if teacher_service is not None:
                    teacher_service.close()
                return self._failure(manifest, started, "unchanged_child", "no Student tensor changed after algorithm dispatch", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)
            teacher_world = int(getattr(teacher, "world_size", teacher_world_size))
            teacher_device_names = tuple(getattr(teacher, "devices", tuple(str(item) for item in teacher_devices)))
            teacher_rank_usage = tuple(sorted(teacher_service.ranks_used)) if teacher_service is not None else ()
            online_teacher_forward = teacher_service is not None
            teacher_sharded = bool(getattr(teacher_service, "sharded", False)) if teacher_service is not None else False
            teacher_forward_count = int(getattr(teacher_service, "forward_count", 0)) if teacher_service is not None else 0
            teacher_peak_memory = float(getattr(teacher_service, "teacher_peak_memory_gb", 0.0)) if teacher_service is not None else 0.0
            teacher_rank_forward_counts = (
                tuple(sorted((int(rank), int(count)) for rank, count in teacher_service.rank_forward_counts.items()))
                if teacher_service is not None
                else ()
            )
            if teacher_service is not None:
                teacher_service.close()
            output_dir.mkdir(parents=True, exist_ok=True)
            child_path = output_dir / "student.safetensors"
            metadata = {
                "proposal_digest": proposal.digest,
                "compiler_digest": manifest.manifest_digest,
                "teacher_sha256": teacher_sha256,
                "parent_sha256": parent_sha256 or "",
                "parent_candidate_id": str(parent_candidate_id or ""),
                "algorithm_name": algorithm_name,
                "algorithm_dispatch": dispatch,
                "algorithm_path": algorithm_path,
                "fidelity": str(fidelity),
                "architecture_family": proposal.architecture.family,
                "offline_simulation": str(bool(self.backend.offline_simulation)).lower(),
                "teacher_world_size": str(teacher_world),
                "teacher_devices": ",".join(teacher_device_names),
                "teacher_ranks_used": ",".join(str(item) for item in teacher_rank_usage),
                "student_device": str(selected_student_device),
                "online_forward": str(online_teacher_forward).lower(),
                "teacher_sharded": str(teacher_sharded).lower(),
                "teacher_forward_count": str(teacher_forward_count),
                "teacher_peak_memory_gb": str(teacher_peak_memory),
                "teacher_rank_forward_counts": json.dumps(list(teacher_rank_forward_counts), sort_keys=True),
                "stage_lineage": json.dumps(list(stage_lineage), sort_keys=True),
            }
            self.backend.save_student(student, child_path, metadata)
            full_precision_path = child_path
            quantized_path = None
            child_for_evaluation = child_path
            if proposal.deployment.quantization == "int8":
                quantized_path = output_dir / "student-int8.safetensors"
                quantize_checkpoint(child_path, quantized_path, bits=8, metadata={**metadata, "full_precision_sha256": sha256_file(child_path)})
                child_for_evaluation = quantized_path
            elif proposal.deployment.quantization == "int4":
                if teacher_service is not None:
                    teacher_service.close()
                return self._failure(manifest, started, "quantization_unsupported", "int4 Student quantization is not supported", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)
            child_sha256 = sha256_file(child_for_evaluation)
            full_precision_sha256 = sha256_file(full_precision_path)
            if child_sha256 == parent_sha256:
                if teacher_service is not None:
                    teacher_service.close()
                return self._failure(manifest, started, "unchanged_child", "child file hash equals parent hash", parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)
            peak = torch.cuda.max_memory_allocated(selected_student_device) / float(1024**3) if selected_student_device.type == "cuda" else 0.0
            return TrainingResult(
                status="success", proposal_digest=proposal.digest, compiler_digest=manifest.manifest_digest,
                parent_sha256=parent_sha256, child_sha256=child_sha256, child_checkpoint=str(child_for_evaluation),
                optimizer_steps=total_steps, initial_loss=initial_loss, final_loss=final_loss,
                gradient_norm=maximum_gradient_norm, wall_time_s=time.perf_counter() - started,
                peak_memory_gb=float(peak), changed_parameter_count=changed,
                offline_simulation=bool(self.backend.offline_simulation), full_precision_checkpoint=str(full_precision_path),
                full_precision_sha256=full_precision_sha256,
                quantized_checkpoint=str(quantized_path) if quantized_path else None,
                quantization=proposal.deployment.quantization, model_size_bytes=child_for_evaluation.stat().st_size,
                parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None,
                parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count,
                algorithm_name=algorithm_name, algorithm_dispatch=dispatch, fidelity=str(fidelity),
                algorithm_path=algorithm_path,
                train_steps=steps,
                cumulative_train_steps=cumulative_train_steps,
                evaluation_cases=spec.evaluation_cases,
                seed_count=spec.seed_count,
                verifier_strength=spec.verifier_strength,
                timeout_s=spec.timeout_s,
                gpu_budget=spec.gpu_budget,
                total_parameter_count=total_parameter_count,
                inheritance_ratio=(float(inherited_parameter_count) / float(total_parameter_count)) if total_parameter_count else 0.0,
                initialization_mode=("full_resume" if inherited_parameter_count == total_parameter_count and total_parameter_count else "partial_transfer" if inherited_parameter_count else "fresh_init"),
                teacher_world_size=teacher_world,
                teacher_devices=teacher_device_names,
                student_device=str(selected_student_device),
                teacher_ranks_used=teacher_rank_usage,
                online_forward=online_teacher_forward,
                teacher_sharded=teacher_sharded,
                teacher_forward_count=teacher_forward_count,
                teacher_peak_memory_gb=teacher_peak_memory,
                teacher_rank_forward_counts=teacher_rank_forward_counts,
                stage_lineage=tuple(stage_lineage),
                optimizer_steps_by_role=dict(optimizer_steps_by_role),
            )
        except StudentTrainingError as exc:
            if teacher_service is not None:
                teacher_service.close()
            return self._failure(manifest, started, exc.code, exc.message, parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)
        except TrainingFailure as exc:
            if teacher_service is not None:
                teacher_service.close()
            return self._failure(manifest, started, exc.code, str(exc), parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)
        except RuntimeError as exc:
            if teacher_service is not None:
                teacher_service.close()
            lowered = str(exc).lower()
            code = "training_oom" if "out of memory" in lowered else "training_runtime"
            return self._failure(manifest, started, code, str(exc), parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            if teacher_service is not None:
                teacher_service.close()
            return self._failure(manifest, started, "training_config", str(exc), parent_sha256=parent_sha256, offline_simulation=self.backend.offline_simulation, parent_kind=parent_kind, parent_checkpoint=str(parent_checkpoint) if parent_checkpoint else None, parent_inherited=bool(inherit_parent), inherited_parameter_count=inherited_parameter_count, algorithm_name=algorithm_name, algorithm_path=algorithm_path, algorithm_dispatch=dispatch, fidelity=fidelity)


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
        teacher_devices: Sequence[str] = (),
        teacher_world_size: int = 1,
    ):
        self.comfyui_root = Path(comfyui_root).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.dtype = dtype
        self.teacher_targets_dir = Path(teacher_targets_dir).resolve() if teacher_targets_dir else None
        self.teacher_devices = tuple(str(item) for item in teacher_devices)
        self.teacher_world_size = int(teacher_world_size)
        self.adapter = None
        if self.teacher_targets_dir is not None:
            raise ValueError(
                "precomputed teacher targets are proxy_training and cannot be used by the real Student capability"
            )

    def load_teacher(
        self,
        checkpoint: Path,
        device: torch.device,
        *,
        teacher_devices: Sequence[str] = (),
        teacher_world_size: int = 1,
    ) -> Any:
        from h3_training.adapters.real_h3 import RealMiniMaxH3Adapter

        self.adapter = RealMiniMaxH3Adapter(self.comfyui_root, device=str(device), dtype=self.dtype)
        devices = tuple(str(item) for item in teacher_devices) or self.teacher_devices
        world_size = int(teacher_world_size or self.teacher_world_size or len(devices) or 1)
        if world_size != TeacherService.REQUIRED_WORLD_SIZE or len(devices) != TeacherService.REQUIRED_WORLD_SIZE:
            raise StudentTrainingError(
                "teacher_world_size_mismatch",
                "real online H3 Teacher requires exactly three distinct devices/ranks",
            )
        return TeacherService(checkpoint, self.comfyui_root, devices, dtype=self.dtype).start()

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
            if prepared.latents.video is None or prepared.noise.video is None or prepared.timesteps.video is None:
                raise StudentTrainingError("invalid_h3_batch", "H3 batch lacks video tensors")
            yield StudentBatch(
                latent=prepared.latents.video.to(device=device, dtype=self.dtype),
                conditioning=prepared.conditioning.text.to(device=device, dtype=self.dtype),
                timestep=prepared.timesteps.video.to(device=device, dtype=torch.float32),
                target=None,
                audio_latent=prepared.latents.audio.to(device=device, dtype=self.dtype) if prepared.latents.audio is not None else None,
                audio_noise=prepared.noise.audio.to(device=device, dtype=self.dtype) if prepared.noise.audio is not None else None,
                noise=prepared.noise.video.to(device=device, dtype=self.dtype),
                audio_timestep=prepared.timesteps.audio.to(device=device, dtype=torch.float32) if prepared.timesteps.audio is not None else None,
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
    "StudentAlgorithmAdapter",
    "StudentTrainWorker",
    "StudentTrainingError",
    "FidelitySpec",
    "TrainingResult",
    "fidelity_spec",
    "fidelity_step_budget",
    "sha256_file",
]
