from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


class CompressionConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CompressionConfig:
    backend: str
    checkpoint: str
    output_root: Path
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = True
    fake: Mapping[str, Any] = field(default_factory=dict)
    controller: Mapping[str, Any] = field(default_factory=dict)
    evaluator: Mapping[str, Any] = field(default_factory=dict)
    memory_enabled: bool = True
    adapt_search_policy: bool = True
    sandbox_timeout_s: float = 10.0
    exploit_ratio: float = 0.7
    seed: int = 7
    advanced_profiling: bool = False
    algorithm_improvement: bool = False
    search_space_evolution: bool = False
    frozen_expanded_search_space: bool = False
    candidate_screening: bool = False
    phase: str = "evolution"
    tica_prune: bool = False
    tica_calibration_manifest: Optional[Path] = None
    tica_validation_manifest: Optional[Path] = None
    tica_test_manifest: Optional[Path] = None
    tica_recipe: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"fake", "diffusers"}:
            raise CompressionConfigError("backend must be fake or diffusers")
        if not self.checkpoint.strip():
            raise CompressionConfigError("checkpoint must not be empty")
        if not str(self.device).strip():
            raise CompressionConfigError("device must not be empty")
        if self.sandbox_timeout_s <= 0:
            raise CompressionConfigError("sandbox_timeout_s must be positive")
        if not 0.0 <= self.exploit_ratio <= 1.0:
            raise CompressionConfigError("exploit_ratio must be in [0, 1]")
        if self.search_space_evolution and self.frozen_expanded_search_space:
            raise CompressionConfigError("search_space_evolution and frozen_expanded_search_space are mutually exclusive")
        if self.phase not in {"evolution", "final_evaluation"}:
            raise CompressionConfigError("phase must be evolution or final_evaluation")

    @classmethod
    def from_path(cls, path: Path) -> "CompressionConfig":
        source = Path(path).resolve()
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise CompressionConfigError("unable to read compression config: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise CompressionConfigError("compression config must be a mapping")
        allowed = {"backend", "checkpoint", "output_root", "device", "dtype", "local_files_only", "fake", "controller", "evaluator", "memory_enabled", "adapt_search_policy", "sandbox_timeout_s", "exploit_ratio", "seed", "advanced_profiling", "algorithm_improvement", "search_space_evolution", "frozen_expanded_search_space", "candidate_screening", "phase", "tica_prune", "tica_calibration_manifest", "tica_validation_manifest", "tica_test_manifest", "tica_recipe"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise CompressionConfigError("unsupported compression config field(s): %s" % ", ".join(unknown))
        output_raw = raw.get("output_root", "var/evolutionary-compression")
        output = Path(str(output_raw)).expanduser()
        if not output.is_absolute():
            output = (source.parent / output).resolve()
        fake = raw.get("fake", {})
        if not isinstance(fake, Mapping):
            raise CompressionConfigError("fake must be a mapping")
        controller = raw.get("controller", {})
        if not isinstance(controller, Mapping):
            raise CompressionConfigError("controller must be a mapping")
        evaluator = raw.get("evaluator", {})
        if not isinstance(evaluator, Mapping):
            raise CompressionConfigError("evaluator must be a mapping")
        recipe = raw.get("tica_recipe", {})
        if not isinstance(recipe, Mapping):
            raise CompressionConfigError("tica_recipe must be a mapping")
        manifest_paths: dict[str, Optional[Path]] = {}
        for field_name in ("tica_calibration_manifest", "tica_validation_manifest", "tica_test_manifest"):
            manifest_raw = raw.get(field_name)
            manifest_path = None
            if manifest_raw:
                manifest_path = Path(str(manifest_raw)).expanduser()
                if not manifest_path.is_absolute():
                    manifest_path = (source.parent / manifest_path).resolve()
            manifest_paths[field_name] = manifest_path
        return cls(
            backend=str(raw.get("backend", "fake")),
            checkpoint=str(raw.get("checkpoint", "fake://baseline")),
            output_root=output,
            device=str(raw.get("device", "cpu")),
            dtype=str(raw.get("dtype", "float32")),
            local_files_only=bool(raw.get("local_files_only", True)),
            fake=dict(fake),
            controller=dict(controller),
            evaluator=dict(evaluator),
            memory_enabled=bool(raw.get("memory_enabled", True)),
            adapt_search_policy=bool(raw.get("adapt_search_policy", True)),
            sandbox_timeout_s=float(raw.get("sandbox_timeout_s", 10.0)),
            exploit_ratio=float(raw.get("exploit_ratio", 0.7)),
            seed=int(raw.get("seed", 7)),
            advanced_profiling=bool(raw.get("advanced_profiling", False)),
            algorithm_improvement=bool(raw.get("algorithm_improvement", False)),
            search_space_evolution=bool(raw.get("search_space_evolution", False)),
            frozen_expanded_search_space=bool(raw.get("frozen_expanded_search_space", False)),
            candidate_screening=bool(raw.get("candidate_screening", False)),
            phase=str(raw.get("phase", "evolution")).strip().lower() or "evolution",
            tica_prune=bool(raw.get("tica_prune", False)),
            tica_calibration_manifest=manifest_paths["tica_calibration_manifest"],
            tica_validation_manifest=manifest_paths["tica_validation_manifest"],
            tica_test_manifest=manifest_paths["tica_test_manifest"],
            tica_recipe=dict(recipe),
        )
