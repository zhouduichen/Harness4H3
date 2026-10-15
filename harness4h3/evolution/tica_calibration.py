"""Versioned calibration inputs for TICA-Prune."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapters.wan.contracts import stable_digest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _tensor_digest(value: Any) -> str:
    try:
        import torch

        buffer = io.BytesIO()
        torch.save(value.detach().cpu().contiguous(), buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return stable_digest(
            {
                "type": type(value).__name__,
                "shape": list(getattr(value, "shape", ())),
                "dtype": str(getattr(value, "dtype", "unknown")),
            }
        )


@dataclass(frozen=True)
class TICACalibrationSpec:
    """Identity and storage metadata for one replayable calibration case."""

    sample_id: str
    prompt: str
    seed: int
    latent_shape: tuple[int, ...]
    timesteps: tuple[float, ...]
    latent_path: str | None = None
    encoder_hidden_states_path: str | None = None
    latent_sha256: str | None = None
    encoder_hidden_states_sha256: str | None = None

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip():
            raise ValueError("TICA calibration sample_id must not be empty")
        if not str(self.prompt).strip():
            raise ValueError("TICA calibration prompt must not be empty")
        if not self.latent_shape or any(int(value) <= 0 for value in self.latent_shape):
            raise ValueError("TICA calibration latent_shape must be positive")
        if not self.timesteps or any(not math.isfinite(float(value)) for value in self.timesteps):
            raise ValueError("TICA calibration timesteps must be finite")
        if any(float(value) < 0.0 or float(value) > 1.0 for value in self.timesteps):
            raise ValueError("TICA calibration timesteps must be normalized to [0, 1]")
        for name in ("latent_sha256", "encoder_hidden_states_sha256"):
            value = getattr(self, name)
            if value is not None and (len(str(value)) != 64 or any(char not in "0123456789abcdefABCDEF" for char in str(value))):
                raise ValueError("TICA %s must be a SHA-256 digest" % name)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def digest(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True)
class TICACalibrationInput:
    """Adapter-ready tensors plus provenance for one calibration case."""

    spec: TICACalibrationSpec
    hidden_states: Any
    encoder_hidden_states: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        return stable_digest(
            {
                "spec": self.spec.to_dict(),
                "hidden_states": _tensor_digest(self.hidden_states),
                "encoder_hidden_states": _tensor_digest(self.encoder_hidden_states),
                "metadata": dict(self.metadata),
            }
        )

    def to_dict_metadata(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "digest": self.digest(),
            "metadata": _jsonable(dict(self.metadata)),
            "hidden_states_shape": list(getattr(self.hidden_states, "shape", ())),
            "encoder_hidden_states_shape": list(getattr(self.encoder_hidden_states, "shape", ())),
        }


@dataclass(frozen=True)
class TICACalibrationManifest:
    """A fixed ordered set of TICA inputs shared by all candidate algorithms."""

    specs: tuple[TICACalibrationSpec, ...]
    provider_name: str = "cached_real_wan"
    manifest_id: str = ""

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("TICA calibration manifest must contain at least one sample")
        if len({item.sample_id for item in self.specs}) != len(self.specs):
            raise ValueError("TICA calibration sample IDs must be unique")
        if not str(self.provider_name).strip():
            raise ValueError("TICA calibration provider_name must not be empty")
        if not self.manifest_id:
            object.__setattr__(self, "manifest_id", "tica-manifest-" + self.digest()[:12])

    @classmethod
    def from_fake(
        cls,
        *,
        hidden_size: int,
        sample_count: int = 2,
        timesteps: Sequence[float] = (0.9, 0.5, 0.1),
        seed: int = 7,
        sample_prefix: str = "fake",
    ) -> "TICACalibrationManifest":
        if int(hidden_size) <= 0 or int(sample_count) <= 0:
            raise ValueError("fake TICA manifest dimensions must be positive")
        specs = tuple(
            TICACalibrationSpec(
                sample_id="%s-%02d" % (sample_prefix, index),
                prompt="fake Wan calibration sample %d (%s)" % (index, sample_prefix),
                seed=int(seed) + index,
                latent_shape=(2, int(hidden_size)),
                timesteps=tuple(float(value) for value in timesteps),
            )
            for index in range(int(sample_count))
        )
        return cls(specs, provider_name="fake_wan_cpu")

    def digest(self) -> str:
        return stable_digest(
            {
                "provider_name": self.provider_name,
                "specs": [item.to_dict() for item in self.specs],
            }
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TICACalibrationManifest":
        if not isinstance(raw, Mapping):
            raise TypeError("TICA calibration manifest must be an object")
        specs = tuple(
            TICACalibrationSpec(
                sample_id=str(item["sample_id"]),
                prompt=str(item["prompt"]),
                seed=int(item["seed"]),
                latent_shape=tuple(int(value) for value in item["latent_shape"]),
                timesteps=tuple(float(value) for value in item["timesteps"]),
                latent_path=str(item["latent_path"]) if item.get("latent_path") else None,
                encoder_hidden_states_path=(
                    str(item["encoder_hidden_states_path"])
                    if item.get("encoder_hidden_states_path")
                    else None
                ),
                latent_sha256=str(item["latent_sha256"]) if item.get("latent_sha256") else None,
                encoder_hidden_states_sha256=(
                    str(item["encoder_hidden_states_sha256"])
                    if item.get("encoder_hidden_states_sha256")
                    else None
                ),
            )
            for item in raw.get("specs", [])
            if isinstance(item, Mapping)
        )
        return cls(
            specs=specs,
            provider_name=str(raw.get("provider_name", "cached_real_wan")),
            manifest_id=str(raw.get("manifest_id", "")),
        )

    def timesteps(self) -> tuple[float, ...]:
        values: list[float] = []
        for spec in self.specs:
            for value in spec.timesteps:
                if float(value) not in values:
                    values.append(float(value))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "provider_name": self.provider_name,
            "specs": [item.to_dict() for item in self.specs],
            "digest": self.digest(),
            "offline_simulation": self.provider_name == "fake_wan_cpu",
        }

    def verify_cached_sources(self) -> None:
        """Verify replayable tensor files before a real Wan run starts."""

        for spec in self.specs:
            sources = (
                ("latent", spec.latent_path, spec.latent_sha256),
                ("encoder_hidden_states", spec.encoder_hidden_states_path, spec.encoder_hidden_states_sha256),
            )
            for name, raw_path, expected in sources:
                if not raw_path or not expected:
                    raise ValueError("TICA %s source/hash is missing for sample %s" % (name, spec.sample_id))
                path = Path(raw_path).expanduser()
                if not path.is_file():
                    raise ValueError("TICA %s source is missing for sample %s: %s" % (name, spec.sample_id, path))
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != str(expected).lower():
                    raise ValueError("TICA %s source hash mismatch for sample %s" % (name, spec.sample_id))

    def samples(self, model: Any, adapter: Any) -> tuple[TICACalibrationInput, ...]:
        loader = getattr(adapter, "tica_calibration_inputs", None)
        if not callable(loader):
            raise ValueError("adapter does not implement TICA calibration input loading")
        return tuple(loader(model, self))


@dataclass(frozen=True)
class TICAEvaluationSplits:
    """Fixed Calibration/Validation/Test manifests with leakage checks."""

    calibration: TICACalibrationManifest
    validation: TICACalibrationManifest
    test: TICACalibrationManifest

    def __post_init__(self) -> None:
        manifests = {
            "calibration": self.calibration,
            "validation": self.validation,
            "test": self.test,
        }
        seen: dict[str, str] = {}
        seen_sources: dict[tuple[str, str], str] = {}
        for role, manifest in manifests.items():
            if not isinstance(manifest, TICACalibrationManifest):
                raise TypeError("TICA %s split must be a TICACalibrationManifest" % role)
            for spec in manifest.specs:
                sample_id = str(spec.sample_id)
                if sample_id in seen:
                    raise ValueError("TICA evaluation split leakage: %s appears in %s and %s" % (sample_id, seen[sample_id], role))
                seen[sample_id] = role
                for source_name, raw_path, source_hash in (
                    ("latent", spec.latent_path, spec.latent_sha256),
                    ("encoder_hidden_states", spec.encoder_hidden_states_path, spec.encoder_hidden_states_sha256),
                ):
                    # A split must not reuse the same cached tensor under a
                    # different sample ID.  Prefer the immutable content
                    # hash; the resolved path catches manifests created
                    # before hashes were added and also handles symlinks.
                    identities = []
                    if source_hash:
                        identities.append((source_name, "sha256:" + str(source_hash).lower()))
                    if raw_path:
                        identities.append((source_name, "path:" + str(Path(raw_path).expanduser().resolve())))
                    for identity in identities:
                        previous_role = seen_sources.get(identity)
                        if previous_role is not None and previous_role != role:
                            raise ValueError(
                                "TICA evaluation split leakage: %s source is reused between %s and %s"
                                % (source_name, previous_role, role)
                            )
                        seen_sources[identity] = role

    def digest(self) -> str:
        return stable_digest(
            {
                "calibration": self.calibration.digest(),
                "validation": self.validation.digest(),
                "test": self.test.digest(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
            "digest": self.digest(),
        }

    def manifest(self, role: str) -> TICACalibrationManifest:
        if role not in {"calibration", "validation", "test"}:
            raise ValueError("unsupported TICA evaluation split: %s" % role)
        return getattr(self, role)


__all__ = ["TICACalibrationInput", "TICACalibrationManifest", "TICACalibrationSpec", "TICAEvaluationSplits"]
