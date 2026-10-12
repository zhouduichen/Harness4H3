from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class ProfileReport:
    model: Mapping[str, Any]
    layers: tuple[Mapping[str, Any], ...]
    block_sensitivity: tuple[Mapping[str, Any], ...]
    channel_sensitivity: tuple[Mapping[str, Any], ...]
    timestep_sensitivity: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]
    hardware: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported profile schema version: %s" % self.schema_version)
        if not isinstance(self.model, Mapping) or not self.model.get("architecture"):
            raise ValueError("profile model architecture is required")
        required_timesteps = {"high_noise", "middle", "low_noise"}
        if set(self.timestep_sensitivity) != required_timesteps:
            raise ValueError("profile must contain high_noise, middle, and low_noise timestep records")
        if not isinstance(self.provenance, Mapping) or "offline_simulation" not in self.provenance:
            raise ValueError("profile provenance must declare offline_simulation")
        if not isinstance(self.hardware, Mapping):
            raise ValueError("profile hardware must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProfileReport":
        if not isinstance(raw, Mapping):
            raise TypeError("profile report must be a mapping")
        required = ("model", "layers", "block_sensitivity", "channel_sensitivity", "timestep_sensitivity", "provenance")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError("profile report missing fields: %s" % ", ".join(missing))
        return cls(
            model=dict(raw["model"]),
            layers=tuple(dict(item) for item in raw["layers"]),
            block_sensitivity=tuple(dict(item) for item in raw["block_sensitivity"]),
            channel_sensitivity=tuple(dict(item) for item in raw["channel_sensitivity"]),
            timestep_sensitivity={str(key): dict(value) for key, value in dict(raw["timestep_sensitivity"]).items()},
            provenance=dict(raw["provenance"]),
            hardware=dict(raw.get("hardware") or {}),
            schema_version=int(raw.get("schema_version", 1)),
        )

    def write_json(self, path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def read_json(cls, path: Path) -> "ProfileReport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
