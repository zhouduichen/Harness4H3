"""Versioned, JSON-only contracts emitted by the local Student architect."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class ProposalValidationError(ValueError):
    """Raised when an LLM proposal is malformed rather than merely infeasible."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposalValidationError("%s must be an object" % name)
    return value


def _keys(value: Mapping[str, Any], allowed: Sequence[str], name: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ProposalValidationError("%s has unknown field(s): %s" % (name, ", ".join(map(str, unknown))))


def _required(value: Mapping[str, Any], names: Sequence[str], name: str) -> None:
    missing = [field for field in names if field not in value]
    if missing:
        raise ProposalValidationError("%s is missing field(s): %s" % (name, ", ".join(missing)))


def _string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise ProposalValidationError("%s must be a string" % name)
    result = value.strip()
    if nonempty and not result:
        raise ProposalValidationError("%s must not be empty" % name)
    return result


def _integer(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProposalValidationError("%s must be an integer" % name)
    if minimum is not None and value < minimum:
        raise ProposalValidationError("%s must be at least %d" % (name, minimum))
    return value


def _number(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProposalValidationError("%s must be a number" % name)
    result = float(value)
    if not math.isfinite(result):
        raise ProposalValidationError("%s must be finite" % name)
    if minimum is not None and result < minimum:
        raise ProposalValidationError("%s must be at least %s" % (name, minimum))
    return result


@dataclass(frozen=True)
class StudentTarget:
    min_params: int = 1_000_000_000
    max_params: int = 2_000_000_000
    max_peak_memory_gb: float = 72.0
    latent_channels: int = 24
    latent_frames: int = 5
    latent_height: int = 32
    latent_width: int = 32
    condition_dim: int = 5120
    min_hidden_size: int = 1024
    max_hidden_size: int = 3072
    min_depth: int = 12
    max_depth: int = 48

    def __post_init__(self) -> None:
        for name in (
            "min_params",
            "max_params",
            "latent_channels",
            "latent_frames",
            "latent_height",
            "latent_width",
            "condition_dim",
            "min_hidden_size",
            "max_hidden_size",
            "min_depth",
            "max_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.min_params > self.max_params:
            raise ValueError("min_params must not exceed max_params")
        if self.min_hidden_size > self.max_hidden_size or self.min_depth > self.max_depth:
            raise ValueError("target ranges must be ordered")
        if not math.isfinite(float(self.max_peak_memory_gb)) or self.max_peak_memory_gb <= 0:
            raise ValueError("max_peak_memory_gb must be positive and finite")


@dataclass(frozen=True)
class ArchitectureSpec:
    family: str
    latent_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    spatial_patch: int
    temporal_patch: int
    temporal_layers: Tuple[int, ...]
    conditioning: str
    norm: str
    activation: str

    _FIELDS = (
        "family",
        "latent_channels",
        "hidden_size",
        "depth",
        "num_heads",
        "mlp_ratio",
        "spatial_patch",
        "temporal_patch",
        "temporal_layers",
        "conditioning",
        "norm",
        "activation",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ArchitectureSpec":
        raw = _mapping(raw, "architecture")
        _keys(raw, cls._FIELDS, "architecture")
        _required(raw, cls._FIELDS, "architecture")
        layers = raw["temporal_layers"]
        if not isinstance(layers, (list, tuple)):
            raise ProposalValidationError("architecture.temporal_layers must be an array")
        parsed_layers = tuple(_integer(value, "architecture.temporal_layers[]", minimum=0) for value in layers)
        return cls(
            family=_string(raw["family"], "architecture.family"),
            latent_channels=_integer(raw["latent_channels"], "architecture.latent_channels", minimum=1),
            hidden_size=_integer(raw["hidden_size"], "architecture.hidden_size", minimum=1),
            depth=_integer(raw["depth"], "architecture.depth", minimum=1),
            num_heads=_integer(raw["num_heads"], "architecture.num_heads", minimum=1),
            mlp_ratio=_number(raw["mlp_ratio"], "architecture.mlp_ratio", minimum=1.0),
            spatial_patch=_integer(raw["spatial_patch"], "architecture.spatial_patch", minimum=1),
            temporal_patch=_integer(raw["temporal_patch"], "architecture.temporal_patch", minimum=1),
            temporal_layers=parsed_layers,
            conditioning=_string(raw["conditioning"], "architecture.conditioning"),
            norm=_string(raw["norm"], "architecture.norm"),
            activation=_string(raw["activation"], "architecture.activation"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["temporal_layers"] = list(self.temporal_layers)
        return result


@dataclass(frozen=True)
class TrainingSpec:
    method: str
    source_steps: int
    target_steps: int
    max_steps: int
    learning_rate: float
    critic_learning_rate: float
    batch_size: int = 1

    _FIELDS = (
        "method",
        "source_steps",
        "target_steps",
        "max_steps",
        "learning_rate",
        "critic_learning_rate",
        "batch_size",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrainingSpec":
        raw = _mapping(raw, "training")
        _keys(raw, cls._FIELDS, "training")
        _required(raw, cls._FIELDS[:6], "training")
        method = _string(raw["method"], "training.method")
        if method not in {"velocity_distill", "dmd2"}:
            raise ProposalValidationError("training.method is unsupported: %s" % method)
        source_steps = _integer(raw["source_steps"], "training.source_steps", minimum=1)
        target_steps = _integer(raw["target_steps"], "training.target_steps", minimum=1)
        if target_steps > source_steps:
            raise ProposalValidationError("training.target_steps must not exceed source_steps")
        return cls(
            method=method,
            source_steps=source_steps,
            target_steps=target_steps,
            max_steps=_integer(raw["max_steps"], "training.max_steps", minimum=1),
            learning_rate=_number(raw["learning_rate"], "training.learning_rate", minimum=0.0),
            critic_learning_rate=_number(
                raw["critic_learning_rate"], "training.critic_learning_rate", minimum=0.0
            ),
            batch_size=_integer(raw.get("batch_size", 1), "training.batch_size", minimum=1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentSpec:
    precision: str
    quantization: str

    _FIELDS = ("precision", "quantization")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeploymentSpec":
        raw = _mapping(raw, "deployment")
        _keys(raw, cls._FIELDS, "deployment")
        _required(raw, cls._FIELDS, "deployment")
        precision = _string(raw["precision"], "deployment.precision")
        quantization = _string(raw["quantization"], "deployment.quantization")
        if precision not in {"bf16", "fp16"}:
            raise ProposalValidationError("deployment.precision is unsupported: %s" % precision)
        if quantization not in {"none", "int8", "int4"}:
            raise ProposalValidationError("deployment.quantization is unsupported: %s" % quantization)
        return cls(precision, quantization)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    errors: Tuple[str, ...] = ()
    estimated_params: Optional[int] = None
    estimated_peak_memory_gb: Optional[float] = None
    duplicate_key: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.errors


def estimate_parameter_count(architecture: ArchitectureSpec, condition_dim: int = 5120) -> int:
    """Count the parameters of the registered VideoLatentDiT graph.

    The compiler independently counts the constructed PyTorch module and uses
    this formula only for early validation and prompt feedback.
    """

    hidden = architecture.hidden_size
    expansion = int(round(hidden * architecture.mlp_ratio))
    patch_volume = (
        architecture.latent_channels
        * architecture.temporal_patch
        * architecture.spatial_patch
        * architecture.spatial_patch
    )
    patch = patch_volume * hidden + hidden
    output = hidden * patch_volume + patch_volume
    condition = condition_dim * (6 * hidden) + 6 * hidden
    timestep = 1 * (6 * hidden) + 6 * hidden
    # Keep the count integer-valued so persisted evidence never depends on
    # floating-point rounding.
    block = 4 * hidden * hidden + 4 * hidden + 2 * expansion * hidden + expansion + hidden + 2 * hidden
    return int(patch + output + condition + timestep + architecture.depth * block + hidden)


@dataclass(frozen=True)
class StudentProposal:
    schema_version: int
    proposal_id: str
    parent_proposal_id: Optional[str]
    teacher: Mapping[str, str]
    architecture: ArchitectureSpec
    training: TrainingSpec
    deployment: DeploymentSpec

    _FIELDS = (
        "schema_version",
        "proposal_id",
        "parent_proposal_id",
        "teacher",
        "architecture",
        "training",
        "deployment",
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StudentProposal":
        raw = _mapping(raw, "student proposal")
        _keys(raw, cls._FIELDS, "student proposal")
        _required(raw, cls._FIELDS, "student proposal")
        version = _integer(raw["schema_version"], "schema_version", minimum=1)
        if version != 1:
            raise ProposalValidationError("unsupported student proposal schema_version: %s" % version)
        teacher_raw = _mapping(raw["teacher"], "teacher")
        _keys(teacher_raw, ("checkpoint", "adapter"), "teacher")
        _required(teacher_raw, ("checkpoint", "adapter"), "teacher")
        teacher = {
            "checkpoint": _string(teacher_raw["checkpoint"], "teacher.checkpoint"),
            "adapter": _string(teacher_raw["adapter"], "teacher.adapter"),
        }
        parent = raw["parent_proposal_id"]
        if parent is not None:
            parent = _string(parent, "parent_proposal_id")
        return cls(
            schema_version=version,
            proposal_id=_string(raw["proposal_id"], "proposal_id"),
            parent_proposal_id=parent,
            teacher=teacher,
            architecture=ArchitectureSpec.from_mapping(raw["architecture"]),
            training=TrainingSpec.from_mapping(raw["training"]),
            deployment=DeploymentSpec.from_mapping(raw["deployment"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "parent_proposal_id": self.parent_proposal_id,
            "teacher": dict(self.teacher),
            "architecture": self.architecture.to_dict(),
            "training": self.training.to_dict(),
            "deployment": self.deployment.to_dict(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def validate(self, target: StudentTarget = StudentTarget()) -> ValidationReport:
        errors = []
        architecture = self.architecture
        if architecture.family != "video_latent_dit":
            errors.append("architecture.family must be video_latent_dit")
        if architecture.latent_channels != target.latent_channels:
            errors.append(
                "architecture.latent_channels=%d does not match target=%d"
                % (architecture.latent_channels, target.latent_channels)
            )
        if not target.min_hidden_size <= architecture.hidden_size <= target.max_hidden_size:
            errors.append("architecture.hidden_size is outside target bounds")
        if not target.min_depth <= architecture.depth <= target.max_depth:
            errors.append("architecture.depth is outside target bounds")
        if architecture.hidden_size % architecture.num_heads:
            errors.append("architecture.hidden_size must be divisible by num_heads")
        if architecture.num_heads > architecture.hidden_size:
            errors.append("architecture.num_heads must not exceed hidden_size")
        if architecture.spatial_patch not in {1, 2, 4}:
            errors.append("architecture.spatial_patch is unsupported")
        if architecture.temporal_patch not in {1, 5}:
            errors.append("architecture.temporal_patch is unsupported")
        if target.latent_height % architecture.spatial_patch or target.latent_width % architecture.spatial_patch:
            errors.append("architecture.spatial_patch does not divide target latent dimensions")
        if target.latent_frames % architecture.temporal_patch:
            errors.append("architecture.temporal_patch does not divide target latent frames")
        if any(layer >= architecture.depth for layer in architecture.temporal_layers):
            errors.append("architecture.temporal_layers contains a layer outside depth")
        if len(set(architecture.temporal_layers)) != len(architecture.temporal_layers):
            errors.append("architecture.temporal_layers must not contain duplicates")
        if architecture.conditioning != "ada_norm_zero":
            errors.append("architecture.conditioning is unsupported")
        if architecture.norm not in {"rmsnorm", "layernorm"}:
            errors.append("architecture.norm is unsupported")
        if architecture.activation not in {"silu", "gelu"}:
            errors.append("architecture.activation is unsupported")
        estimated = estimate_parameter_count(architecture, target.condition_dim)
        if not target.min_params <= estimated <= target.max_params:
            errors.append(
                "estimated parameter count %d is outside [%d, %d]"
                % (estimated, target.min_params, target.max_params)
            )
        precision_bytes = 2 if self.deployment.precision in {"bf16", "fp16"} else 4
        optimizer_bytes = 8
        # Parameters + gradients + AdamW moments, plus a conservative 20% activation reserve.
        peak_bytes = estimated * (precision_bytes + precision_bytes + optimizer_bytes) * 1.2
        peak_gb = peak_bytes / float(1024**3)
        if peak_gb > target.max_peak_memory_gb:
            errors.append(
                "estimated peak memory %.2f GB exceeds target %.2f GB"
                % (peak_gb, target.max_peak_memory_gb)
            )
        return ValidationReport(tuple(errors), estimated, peak_gb)


__all__ = [
    "ArchitectureSpec",
    "DeploymentSpec",
    "ProposalValidationError",
    "StudentProposal",
    "StudentTarget",
    "TrainingSpec",
    "ValidationReport",
    "canonical_digest",
    "estimate_parameter_count",
]
