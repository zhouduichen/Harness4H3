#!/usr/bin/env python3
"""Build matched Wan evolutionary-compression configurations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from harness4h3.evolution.protocol import (
    DEFAULT_EXPECTED_CASES,
    DEFAULT_LATENCY_REPEATS,
    DEFAULT_ROUNDS,
    PAPER_REQUIRED_ARMS,
    REQUIRED_ARMS,
    ProtocolValidationError,
    validate_protocol,
    write_manifests,
)

_DETERMINISTIC_PROVIDERS = {"fixed", "deterministic", "offline", "mock"}
_ASSET_KEYS = ("wan_config", "evaluation_manifest", "clip_model_path")


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolValidationError("unable to read base config %s: %s" % (path, exc)) from exc
    if not isinstance(raw, Mapping):
        raise ProtocolValidationError("base config must be a mapping: %s" % path)
    return raw


def _absolute_asset(value: Any, base_dir: Path) -> str:
    text = str(value).strip()
    if "://" in text:
        return text
    path = Path(text).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".evolution-config-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(value), handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise ProtocolValidationError("cannot write generated config %s: %s" % (path, exc)) from exc


def build_protocol_configs(
    base_config: str | Path,
    output_dir: str | Path,
    run_root: str | Path,
    *,
    source_revision: str,
    paper_protocol: bool = False,
    rounds: int = DEFAULT_ROUNDS,
    expected_cases: int = DEFAULT_EXPECTED_CASES,
    latency_repeats: int = DEFAULT_LATENCY_REPEATS,
    device: str | None = None,
    execution_profile: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Generate matched configs without changing model/evaluation inputs."""

    arms = PAPER_REQUIRED_ARMS if paper_protocol else REQUIRED_ARMS

    base_path = Path(base_config).expanduser().resolve()
    raw = _read_yaml(base_path)
    if str(raw.get("backend", "")).strip() != "diffusers":
        raise ProtocolValidationError("base config must use backend=diffusers")
    evaluator = raw.get("evaluator")
    if not isinstance(evaluator, Mapping) or str(evaluator.get("backend", "")).strip() != "wan_video":
        raise ProtocolValidationError("base config must use evaluator.backend=wan_video")
    revision = str(source_revision).strip()
    if not revision or revision.lower() in {"unknown", "none", "null"}:
        raise ProtocolValidationError("source_revision must be pinned")
    controller = raw.get("controller")
    if not isinstance(controller, Mapping):
        raise ProtocolValidationError("base config requires a controller mapping for Coding Agent arms")
    provider = str(controller.get("provider", "")).strip().lower().replace("_", "-")
    if not provider or provider in _DETERMINISTIC_PROVIDERS:
        raise ProtocolValidationError("base controller provider must be a non-deterministic Coding Agent provider")

    base = copy.deepcopy(dict(raw))
    base["checkpoint"] = _absolute_asset(base["checkpoint"], base_path.parent)
    base_evaluator = dict(base["evaluator"])
    for key in _ASSET_KEYS:
        if not str(base_evaluator.get(key, "")).strip():
            raise ProtocolValidationError("base evaluator missing %s" % key)
        base_evaluator[key] = _absolute_asset(base_evaluator[key], base_path.parent)
    base_evaluator["source_revision"] = revision
    if execution_profile:
        base_evaluator["execution_profile"] = str(Path(execution_profile).expanduser())
    if device:
        base["device"] = str(device)
        for key in ("device", "pipeline_device", "quality_device"):
            base_evaluator[key] = str(device)
    base["evaluator"] = base_evaluator
    adaptive_exploit_ratio = float(base.get("exploit_ratio", 0.7))
    if not 0.0 <= adaptive_exploit_ratio <= 1.0:
        raise ProtocolValidationError("base exploit_ratio must be in [0, 1]")

    output_path = Path(output_dir).expanduser().resolve()
    run_path = Path(run_root).expanduser().resolve()
    targets = {name: output_path / (name + ".yaml") for name in arms}
    if not overwrite:
        existing = [str(path) for path in targets.values() if path.exists()]
        if existing:
            raise ProtocolValidationError("generated config already exists: " + ", ".join(existing))

    generated: dict[str, Path] = {}
    for name in arms:
        config = copy.deepcopy(base)
        config["output_root"] = str(run_path / name)
        if name == "fixed":
            config["memory_enabled"] = False
            config["adapt_search_policy"] = False
            if paper_protocol:
                config["algorithm_improvement"] = False
                config["search_space_evolution"] = False
                config["frozen_expanded_search_space"] = False
            config["exploit_ratio"] = 1.0
            fixed_controller = dict(config.get("controller") or {})
            fixed_controller["provider"] = "deterministic"
            fixed_controller["model"] = "deterministic-fixed"
            config["controller"] = fixed_controller
        elif name == "coding-agent":
            config["memory_enabled"] = False
            config["adapt_search_policy"] = False
            if paper_protocol:
                config["algorithm_improvement"] = True
                config["search_space_evolution"] = False
                config["frozen_expanded_search_space"] = False
            config["exploit_ratio"] = adaptive_exploit_ratio
        elif name == "agent-memory":
            config["memory_enabled"] = True
            config["adapt_search_policy"] = False
            if paper_protocol:
                config["algorithm_improvement"] = True
                config["search_space_evolution"] = False
                config["frozen_expanded_search_space"] = False
            config["exploit_ratio"] = adaptive_exploit_ratio
        elif paper_protocol:
            config["memory_enabled"] = True
            config["adapt_search_policy"] = True
            config["algorithm_improvement"] = True
            config["search_space_evolution"] = name == "rsi"
            config["frozen_expanded_search_space"] = name == "rsi-expanded-frozen"
            config["exploit_ratio"] = adaptive_exploit_ratio
        else:
            config["memory_enabled"] = True
            config["adapt_search_policy"] = True
            config["exploit_ratio"] = adaptive_exploit_ratio
        target = targets[name]
        _atomic_yaml(target, config)
        generated[name] = target
    return generated


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".evolution-protocol-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise ProtocolValidationError("cannot write protocol report %s: %s" % (path, exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for generated YAML files")
    parser.add_argument("--run-root", required=True, type=Path, help="parent directory for independent output roots")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--paper", action="store_true", help="generate the six-arm RSI paper ablation")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--expected-cases", type=int, default=DEFAULT_EXPECTED_CASES)
    parser.add_argument("--latency-repeats", type=int, default=DEFAULT_LATENCY_REPEATS)
    parser.add_argument("--device", help="override model, pipeline, and quality devices for every arm")
    parser.add_argument("--execution-profile", help="shared completed real-CUDA profile used as Agent evidence")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing-assets", action="store_true")
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--output", type=Path, help="optional protocol report path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    generated: dict[str, Path] = {}
    try:
        generated = build_protocol_configs(
            args.base_config,
            args.output_dir,
            args.run_root,
            source_revision=args.source_revision,
            paper_protocol=args.paper,
            rounds=args.rounds,
            expected_cases=args.expected_cases,
            latency_repeats=args.latency_repeats,
            device=args.device,
            execution_profile=args.execution_profile,
            overwrite=args.overwrite,
        )
        report = validate_protocol(
            generated,
            rounds=args.rounds,
            expected_cases=args.expected_cases,
            latency_repeats=args.latency_repeats,
            source_revision=args.source_revision,
            allow_missing_assets=args.allow_missing_assets,
            required_arms=PAPER_REQUIRED_ARMS if args.paper else REQUIRED_ARMS,
        )
        if args.write_manifests:
            write_manifests(report)
        if args.output:
            _write_report(args.output, report)
    except (ProtocolValidationError, OSError, ValueError) as exc:
        print("protocol_config_invalid: %s" % exc)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("protocol_configs_valid fingerprint=%s" % report["protocol_fingerprint"])
        for name in report["required_arms"]:
            print("%s config=%s output_root=%s" % (name, generated[name], report["arms"][name]["output_root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
