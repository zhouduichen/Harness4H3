"""Preflight validation for matched Wan evolutionary-compression runs.

The protocol is deliberately separate from the GPU evaluator.  It validates
configuration and writes an auditable run manifest before an expensive run is
started; it never performs model loading or inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..adapters.wan.contracts import stable_digest
from .config import CompressionConfig, CompressionConfigError

PROTOCOL_SCHEMA_VERSION = 1
REQUIRED_ARMS = ("fixed", "coding-agent", "agent-memory", "rsi")
DEFAULT_ROUNDS = 3
DEFAULT_EXPECTED_CASES = 16
DEFAULT_LATENCY_REPEATS = 3


class ProtocolValidationError(ValueError):
    """Raised when four runs cannot be treated as one controlled experiment."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_asset(raw: Any, config_path: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _asset_record(raw: Any, config_path: Path, *, require_file: bool = False) -> dict[str, Any]:
    path = _resolve_asset(raw, config_path)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "file" if path.is_file() else "directory" if path.is_dir() else "missing",
    }
    if path.is_file():
        record["sha256"] = _sha256_file(path)
    else:
        record["sha256"] = None
    if require_file and not path.is_file():
        record["required_file"] = True
    return record


def _read_raw_config(path: Path) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolValidationError("unable to read %s: %s" % (path, exc)) from exc
    if not isinstance(raw, Mapping):
        raise ProtocolValidationError("compression config must be a mapping: %s" % path)
    return raw


def _normal_name(name: str) -> str:
    value = str(name).strip().lower().replace("_", "-")
    aliases = {"coding": "coding-agent", "memory": "agent-memory", "full-rsi": "rsi", "complete-rsi": "rsi"}
    return aliases.get(value, value)


def _controller_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    controller = raw.get("controller")
    if not isinstance(controller, Mapping):
        controller = {}
    return dict(_canonical(dict(controller)))


def _path_overlap(left: Path, right: Path) -> bool:
    try:
        return os.path.commonpath((str(left), str(right))) in {str(left), str(right)}
    except ValueError:
        return False


def _config_record(name: str, path: Path, *, allow_missing_assets: bool) -> dict[str, Any]:
    raw = _read_raw_config(path)
    try:
        config = CompressionConfig.from_path(path)
    except (CompressionConfigError, ValueError) as exc:
        raise ProtocolValidationError("invalid %s config: %s" % (name, exc)) from exc
    evaluator = raw.get("evaluator")
    if not isinstance(evaluator, Mapping) or str(evaluator.get("backend", "")).strip() != "wan_video":
        raise ProtocolValidationError("%s must use evaluator.backend=wan_video" % name)
    required = ("wan_config", "evaluation_manifest", "clip_model_path")
    missing = [key for key in required if not str(evaluator.get(key, "")).strip()]
    if missing:
        raise ProtocolValidationError("%s evaluator missing: %s" % (name, ", ".join(missing)))

    source_revision = str(evaluator.get("source_revision", "")).strip()
    if not source_revision or source_revision.lower() in {"unknown", "none", "null"}:
        raise ProtocolValidationError("%s evaluator.source_revision must be pinned" % name)
    assets = {
        "checkpoint": _asset_record(config.checkpoint, path),
        "wan_config": _asset_record(evaluator["wan_config"], path, require_file=True),
        "evaluation_manifest": _asset_record(evaluator["evaluation_manifest"], path, require_file=True),
        "clip_model_path": _asset_record(evaluator["clip_model_path"], path),
    }
    missing_assets = [key for key, value in assets.items() if not value["exists"]]
    invalid_files = [key for key, value in assets.items() if value.get("required_file")]
    if (missing_assets or invalid_files) and not allow_missing_assets:
        details = []
        if missing_assets:
            details.append("missing=" + ",".join(missing_assets))
        if invalid_files:
            details.append("not-files=" + ",".join(invalid_files))
        raise ProtocolValidationError("%s assets are not available (%s)" % (name, "; ".join(details)))

    eval_fields = {
        "wan_config": assets["wan_config"].get("sha256") or assets["wan_config"]["path"],
        "evaluation_manifest": assets["evaluation_manifest"].get("sha256") or assets["evaluation_manifest"]["path"],
        "clip_model_path": assets["clip_model_path"]["path"],
        "device": str(evaluator.get("device", config.device)),
        "pipeline_device": str(evaluator.get("pipeline_device", evaluator.get("device", config.device))),
        "quality_device": str(evaluator.get("quality_device", evaluator.get("device", config.device))),
        "dtype": str(evaluator.get("dtype", config.dtype)),
        "local_files_only": bool(evaluator.get("local_files_only", config.local_files_only)),
        "max_cases": int(evaluator.get("max_cases", 1)),
        "seed_count": int(evaluator.get("seed_count", 1)),
        "sample_frames": int(evaluator.get("sample_frames", 4)),
        "max_quality_drop": float(evaluator.get("max_quality_drop", 0.05)),
        "latency_warmup": int(evaluator.get("latency_warmup", 1)),
        "latency_repeats": int(evaluator.get("latency_repeats", DEFAULT_LATENCY_REPEATS)),
        "visual_quality_gate": dict(evaluator.get("visual_quality_gate") or {}),
        "source_revision": source_revision,
    }
    controller = _controller_record(raw)
    provider = str(controller.get("provider", "deterministic")).strip().lower().replace("_", "-")
    return {
        "name": name,
        "config_path": str(path),
        "config_sha256": _sha256_file(path),
        "backend": config.backend,
        "checkpoint": str(config.checkpoint),
        "checkpoint_asset": assets["checkpoint"],
        "output_root": str(config.output_root.resolve()),
        "device": str(config.device),
        "dtype": str(config.dtype),
        "seed": int(config.seed),
        "memory_enabled": bool(config.memory_enabled),
        "adapt_search_policy": bool(config.adapt_search_policy),
        "exploit_ratio": float(config.exploit_ratio),
        "controller": controller,
        "provider": provider,
        "model": str(controller.get("model", "")),
        "evaluator": eval_fields,
        "assets": assets,
        "source_revision": source_revision,
    }


def _raise(errors: Sequence[str]) -> None:
    if errors:
        raise ProtocolValidationError("protocol validation failed: " + " | ".join(errors))


def validate_protocol(
    specs: Mapping[str, str | Path],
    *,
    rounds: int = DEFAULT_ROUNDS,
    expected_cases: int = DEFAULT_EXPECTED_CASES,
    latency_repeats: int = DEFAULT_LATENCY_REPEATS,
    source_revision: str | None = None,
    allow_missing_assets: bool = False,
) -> dict[str, Any]:
    """Validate four configs and return a stable, serializable protocol report."""

    names = {_normal_name(name): name for name in specs}
    errors: list[str] = []
    missing_arms = [name for name in REQUIRED_ARMS if name not in names]
    extra_arms = sorted(name for name in names if name not in REQUIRED_ARMS)
    if missing_arms:
        errors.append("missing arms: " + ", ".join(missing_arms))
    if extra_arms:
        errors.append("unsupported arms: " + ", ".join(extra_arms))
    if rounds <= 0:
        errors.append("rounds must be positive")
    if expected_cases <= 0:
        errors.append("expected_cases must be positive")
    if latency_repeats <= 0:
        errors.append("latency_repeats must be positive")
    _raise(errors)

    records = {
        name: _config_record(name, Path(specs[names[name]]).expanduser().resolve(), allow_missing_assets=allow_missing_assets)
        for name in REQUIRED_ARMS
    }
    errors = []
    roots = {name: Path(record["output_root"]) for name, record in records.items()}
    if len(set(roots.values())) != len(roots):
        errors.append("output_root must be distinct for all arms")
    for left_name, left_root in roots.items():
        for right_name, right_root in roots.items():
            if left_name < right_name and _path_overlap(left_root, right_root):
                errors.append("output_root overlaps: %s and %s" % (left_name, right_name))

    first = records[REQUIRED_ARMS[0]]
    common_keys = ("backend", "checkpoint", "device", "dtype", "seed", "evaluator", "source_revision")
    for name, record in records.items():
        for key in common_keys:
            if _canonical(record[key]) != _canonical(first[key]):
                errors.append("common field differs: %s" % key)
                break
    # The controller is intentionally the same only for the three adaptive
    # arms; Fixed may use a deterministic provider.
    for name in ("agent-memory", "rsi"):
        for key in ("controller", "model", "evaluator"):
            if _canonical(records[name][key]) != _canonical(records["coding-agent"][key]):
                errors.append("coding-agent and %s differ in %s" % (name, key))
    for name in REQUIRED_ARMS:
        evaluator = records[name]["evaluator"]
        if evaluator["max_cases"] * evaluator["seed_count"] != expected_cases:
            errors.append("%s must have max_cases*seed_count=%d" % (name, expected_cases))
        if evaluator["latency_repeats"] != latency_repeats:
            errors.append("%s latency_repeats=%d, expected %d" % (name, evaluator["latency_repeats"], latency_repeats))
        if source_revision is not None and records[name]["source_revision"] != str(source_revision).strip():
            errors.append("%s source_revision differs from requested revision" % name)

    controls = {
        name: {
            "provider": record["provider"],
            "model": record["model"],
            "memory_enabled": record["memory_enabled"],
            "adapt_search_policy": record["adapt_search_policy"],
            "exploit_ratio": record["exploit_ratio"],
        }
        for name, record in records.items()
    }
    fixed_provider_ok = controls["fixed"]["provider"] in {"fixed", "deterministic", "offline", "mock"}
    if not fixed_provider_ok:
        errors.append("fixed must use a deterministic/fixed controller provider")
    if controls["fixed"]["memory_enabled"] or controls["fixed"]["adapt_search_policy"] or controls["fixed"]["exploit_ratio"] != 1.0:
        errors.append("fixed must disable memory/adaptation and use exploit_ratio=1.0")
    if controls["coding-agent"]["memory_enabled"] or controls["coding-agent"]["adapt_search_policy"]:
        errors.append("coding-agent must disable memory and adaptation")
    if not controls["agent-memory"]["memory_enabled"] or controls["agent-memory"]["adapt_search_policy"]:
        errors.append("agent-memory must enable memory and disable adaptation")
    if not controls["rsi"]["memory_enabled"] or not controls["rsi"]["adapt_search_policy"]:
        errors.append("rsi must enable memory and adaptation")
    for name in ("agent-memory", "rsi"):
        if controls[name]["exploit_ratio"] != controls["coding-agent"]["exploit_ratio"]:
            errors.append("coding-agent and %s must use the same exploit_ratio" % name)
    _raise(errors)

    fingerprint_payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "arms": {name: controls[name] for name in REQUIRED_ARMS},
        "common": {
            key: first[key] for key in ("backend", "checkpoint", "device", "dtype", "seed", "evaluator", "source_revision")
        },
        "adaptive_controller": {
            "controller": records["coding-agent"]["controller"],
            "model": records["coding-agent"]["model"],
        },
        "budget": {"rounds": rounds, "expected_cases": expected_cases, "latency_repeats": latency_repeats},
    }
    fingerprint = stable_digest(fingerprint_payload)
    manifests = {}
    for name in REQUIRED_ARMS:
        record = records[name]
        manifests[name] = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "protocol_fingerprint": fingerprint,
            "arm_name": name,
            "config_sha256": record["config_sha256"],
            "source_revision": record["source_revision"],
            "checkpoint": record["checkpoint_asset"],
            "evaluator": record["evaluator"],
            "budget": {"rounds": rounds, "expected_cases": expected_cases, "latency_repeats": latency_repeats},
            "controls": controls[name],
        }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_fingerprint": fingerprint,
        "required_arms": list(REQUIRED_ARMS),
        "budget": {"rounds": rounds, "expected_cases": expected_cases, "latency_repeats": latency_repeats},
        "arms": records,
        "run_manifests": manifests,
    }


def write_manifests(report: Mapping[str, Any]) -> None:
    """Atomically write all manifests after a complete successful preflight."""

    arms = report.get("arms")
    manifests = report.get("run_manifests")
    if not isinstance(arms, Mapping) or not isinstance(manifests, Mapping):
        raise ProtocolValidationError("invalid protocol report")
    targets: list[tuple[Path, Mapping[str, Any]]] = []
    for name in REQUIRED_ARMS:
        record = arms.get(name)
        manifest = manifests.get(name)
        if not isinstance(record, Mapping) or not isinstance(manifest, Mapping):
            raise ProtocolValidationError("protocol report missing arm %s" % name)
        target = Path(str(record["output_root"])) / "run-manifest.json"
        existing = None
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProtocolValidationError("cannot read existing manifest %s: %s" % (target, exc)) from exc
            if existing != manifest:
                raise ProtocolValidationError("existing run manifest conflicts: %s" % target)
            continue
        targets.append((target, manifest))

    for target, manifest in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".run-manifest-", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise ProtocolValidationError("cannot write run manifest %s: %s" % (target, exc)) from exc
