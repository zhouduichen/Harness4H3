from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stable_digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_content_sha256(path: str | Path) -> str:
    """Hash checkpoint bytes and relative filenames, never model labels.

    Directory checkpoints (such as Diffusers pipelines) are canonicalized by
    sorted relative paths.  This is deliberately separate from a state-dict
    key digest: a parent/child identity must change when the actual weights or
    config bytes change.
    """

    root = Path(path)
    digest = hashlib.sha256()
    if root.is_file():
        with root.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty: %s" % root)
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelStructure:
    architecture: str
    block_count: int
    hidden_size: int
    ffn_width: int
    parameter_count: int
    checkpoint_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.architecture.strip():
            raise ValueError("architecture must not be empty")
        if self.block_count <= 0 or self.hidden_size <= 0 or self.ffn_width <= 0 or self.parameter_count < 0:
            raise ValueError("model structure dimensions must be positive")
        if len(str(self.checkpoint_sha256)) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    index: int
    parameter_count: int
    module_path: str

    def __post_init__(self) -> None:
        if not self.block_id.strip() or self.index < 0 or self.parameter_count < 0:
            raise ValueError("invalid block reference")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ChannelPruningConfig:
    block_ids: Tuple[str, ...]
    retained_indices: Tuple[int, ...]
    projection_name: str = "ffn"

    def __post_init__(self) -> None:
        if not self.block_ids:
            raise ValueError("block_ids must not be empty")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("block_ids must be unique")
        if not self.retained_indices:
            raise ValueError("retained_indices must not be empty")
        if len(set(self.retained_indices)) != len(self.retained_indices):
            raise ValueError("retained_indices must be unique")
        if any(index < 0 for index in self.retained_indices):
            raise ValueError("retained_indices must be non-negative")
        if not self.projection_name.strip():
            raise ValueError("projection_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class StructuralChange:
    kind: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CheckpointArtifact:
    path: str
    sha256: str
    architecture: str
    structure: ModelStructure
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path.strip() or len(self.sha256) != 64:
            raise ValueError("checkpoint artifact requires path and SHA-256")
        if self.structure.checkpoint_sha256 != self.sha256:
            raise ValueError("checkpoint artifact digest must match structure digest")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CompressionResult:
    status: str
    parent_checkpoint_sha256: str
    child_checkpoint: Optional[CheckpointArtifact]
    structural_change: Mapping[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    message: str = ""
    wall_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed"}:
            raise ValueError("compression status must be success or failed")
        if self.status == "success" and self.child_checkpoint is None:
            raise ValueError("successful compression requires a child checkpoint")
        if self.status == "failed" and not self.failure_type:
            raise ValueError("failed compression requires failure_type")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class OperatorCapability:
    """Persisted capability metadata for an executable Wan operator.

    ``OperatorSpec`` is the search-space-facing schema.  This smaller
    contract is useful at the execution boundary where callers need to know
    whether an operator changes weights, runtime execution, or the sampler.
    """

    name: str
    version: str
    granularity: str
    execution_domain: str
    argument_schema: Mapping[str, Any]
    implementation_ref: str
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip() or not self.granularity.strip():
            raise ValueError("operator capability requires name, version, and granularity")
        if self.execution_domain not in {"structural", "runtime", "sampler", "composite"}:
            raise ValueError("unsupported operator execution domain")
        if not self.implementation_ref.strip():
            raise ValueError("operator capability requires an implementation reference")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ModelVariant:
    """Minimal, restartable identity for structural/runtime/sampler variants.

    A runtime-only variant deliberately keeps ``structural_checkpoint`` null:
    it records behavior layered on the immutable base checkpoint and never
    converts a zero-parameter-change policy into a claimed weight reduction.
    The manifest is a sidecar so adding it cannot mutate checkpoint bytes or
    invalidate the checkpoint artifact digest.
    """

    variant_id: str
    base_checkpoint: str
    structural_checkpoint: Optional[str]
    ordered_operations: tuple[Mapping[str, Any], ...] = ()
    runtime_policy: Mapping[str, Any] = field(default_factory=dict)
    sampler_policy: Mapping[str, Any] = field(default_factory=dict)
    algorithm_id: Optional[str] = None
    search_space_version: Optional[str] = None
    source_revision: str = ""
    runtime_revision: str = ""
    base_checkpoint_sha256: Optional[str] = None
    structural_checkpoint_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.variant_id.startswith("M") or not self.base_checkpoint.strip():
            raise ValueError("model variant requires an ID and base checkpoint")
        if self.algorithm_id is not None and not str(self.algorithm_id).startswith("A"):
            raise ValueError("model variant algorithm_id must be an algorithm ID")
        if self.search_space_version is not None and not str(self.search_space_version).startswith("S"):
            raise ValueError("model variant search_space_version must be a search-space ID")
        for operation in self.ordered_operations:
            if not isinstance(operation, Mapping) or not isinstance(operation.get("operator"), str) or not isinstance(operation.get("args", {}), Mapping):
                raise ValueError("model variant operations must contain operator and args")
        if self.structural_checkpoint is None and self.structural_checkpoint_sha256 is not None:
            raise ValueError("runtime-only model variants cannot have a structural checkpoint digest")

    @property
    def runtime_only(self) -> bool:
        return self.structural_checkpoint is None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "base_checkpoint": self.base_checkpoint,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "structural_checkpoint": self.structural_checkpoint,
            "structural_checkpoint_sha256": self.structural_checkpoint_sha256,
            "ordered_operations": list(self.ordered_operations),
            "runtime_policy": dict(self.runtime_policy),
            "sampler_policy": dict(self.sampler_policy),
            "algorithm_id": self.algorithm_id,
            "search_space_version": self.search_space_version,
            "source_revision": self.source_revision,
            "runtime_revision": self.runtime_revision,
        }

    @property
    def variant_digest(self) -> str:
        return stable_digest(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**_jsonable(asdict(self)), "runtime_only": self.runtime_only, "variant_digest": self.variant_digest}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelVariant":
        return cls(
            variant_id=str(raw["variant_id"]),
            base_checkpoint=str(raw["base_checkpoint"]),
            structural_checkpoint=str(raw["structural_checkpoint"]) if raw.get("structural_checkpoint") else None,
            ordered_operations=tuple(dict(item) for item in raw.get("ordered_operations", [])),
            runtime_policy=dict(raw.get("runtime_policy") or {}),
            sampler_policy=dict(raw.get("sampler_policy") or {}),
            algorithm_id=str(raw["algorithm_id"]) if raw.get("algorithm_id") else None,
            search_space_version=str(raw["search_space_version"]) if raw.get("search_space_version") else None,
            source_revision=str(raw.get("source_revision", "")),
            runtime_revision=str(raw.get("runtime_revision", "")),
            base_checkpoint_sha256=str(raw["base_checkpoint_sha256"]) if raw.get("base_checkpoint_sha256") else None,
            structural_checkpoint_sha256=str(raw["structural_checkpoint_sha256"]) if raw.get("structural_checkpoint_sha256") else None,
        )

    def write_manifest(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read_manifest(cls, path: str | Path) -> "ModelVariant":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_compression(
        cls,
        variant_id: str,
        parent: CheckpointArtifact,
        child: CompressionResult,
        plan: Mapping[str, Any],
        *,
        algorithm_id: Optional[str] = None,
        search_space_version: Optional[str] = None,
        source_revision: str = "",
        runtime_revision: str = "",
    ) -> "ModelVariant":
        if child.status != "success" or child.child_checkpoint is None:
            raise ValueError("model variant requires a successful compression result")

        def operations(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
            if str(raw.get("operator")) != "compose":
                return [{"operator": str(raw.get("operator")), "args": dict(raw.get("args") or {})}]
            steps = raw.get("args", {}).get("steps", []) if isinstance(raw.get("args"), Mapping) else []
            return [dict(step) for step in steps if isinstance(step, Mapping)]

        ordered = tuple(operations(plan))
        structural_names = {"prune_blocks", "prune_layers", "prune_ffn_channels", "prune_attention_heads", "prune_heads", "tica_prune"}
        runtime_names = {"timestep_block_gate", "timestep_attention_policy", "prune_tokens", "token_merge", "kv_token_select", "sparse_runtime", "attention_window"}
        sampler_names = {"sampling_step_skip"}
        structural = any(str(item.get("operator")) in structural_names for item in ordered)
        runtime_policy = {
            str(item["operator"]): dict(item.get("args") or {})
            for item in ordered
            if str(item.get("operator")) in runtime_names
        }
        sampler_policy = {
            str(item["operator"]): dict(item.get("args") or {})
            for item in ordered
            if str(item.get("operator")) in sampler_names
        }
        return cls(
            variant_id=str(variant_id),
            base_checkpoint=parent.path,
            structural_checkpoint=child.child_checkpoint.path if structural else None,
            ordered_operations=ordered,
            runtime_policy=runtime_policy,
            sampler_policy=sampler_policy,
            algorithm_id=algorithm_id,
            search_space_version=search_space_version,
            source_revision=source_revision,
            runtime_revision=runtime_revision,
            base_checkpoint_sha256=parent.sha256,
            structural_checkpoint_sha256=child.child_checkpoint.sha256 if structural else None,
        )


class WanModelAdapter(Protocol):
    architecture_name: str

    def load_model(self, checkpoint: str, *, device: str = "cpu", dtype: str = "float32") -> Any:
        ...

    def inspect(self, model: Any) -> ModelStructure:
        ...

    def get_blocks(self, model: Any) -> tuple[BlockRef, ...]:
        ...

    def remove_blocks(self, model: Any, block_ids: Sequence[str]) -> StructuralChange:
        ...

    def prune_channels(self, model: Any, config: ChannelPruningConfig) -> StructuralChange:
        ...

    def prune_attention_heads(self, model: Any, removed_heads: Sequence[int], attention_names: Sequence[str] = ("attn1", "attn2")) -> StructuralChange:
        ...

    def save_model(self, model: Any, output_dir: Path) -> CheckpointArtifact:
        ...

    def reload_model(self, artifact: CheckpointArtifact) -> Any:
        ...

    # Optional-but-standardized deterministic probe boundary used by the
    # algorithm-level evaluator.  Real Wan adapters already expose these
    # methods for profiling; FakeWanDiT implements the same boundary so CPU
    # evidence follows the production adapter shape.
    def profile_sample(self, model: Any, seed: int = 7) -> Any:
        ...

    def forward_profile(self, model: Any, sample: Any) -> Any:
        ...

    def runtime_profile(self, model: Any, seed: int = 7) -> Any:
        """Optional sequence/mixed-timestep probe for runtime policies."""
        ...

    def estimate_flops(self, model: Any) -> Optional[int]:
        ...

    def tica_calibration_inputs(self, model: Any, manifest: Any) -> tuple[Any, ...]:
        ...

    def forward_denoising(self, model: Any, calibration_input: Any, timestep: float) -> Any:
        ...

    def apply_timestep_block_gating(self, model: Any, rules: Sequence[Mapping[str, Any]]) -> StructuralChange:
        ...

    def apply_timestep_attention_policy(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        ...

    def apply_token_sparsity(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        ...

    def apply_token_merge(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        ...

    def apply_kv_token_selection(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        ...

    def apply_sparse_runtime(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        ...

    def apply_attention_window(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        ...

    def apply_sampling_step_policy(self, model: Any, policy: Mapping[str, Any]) -> StructuralChange:
        ...

    def sampling_steps(self, model: Any, timesteps: Sequence[Any]) -> tuple[Any, ...]:
        ...
