"""Run an independent Wan evaluation for completed evolutionary runs.

This tool deliberately sits outside ``EvolutionLoop``.  It evaluates the
final active model from each completed run against a separate manifest so
development-set evidence cannot be mistaken for held-out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness4h3.adapters.wan.contracts import CheckpointArtifact, ModelStructure
from harness4h3.evolution.evaluator import WanVideoEvaluationConfig, WanVideoEvaluator


@dataclass(frozen=True)
class RunSpec:
    name: str
    root: Path
    gpu: int
    candidate_model_id: str | None = None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_from_candidate(path: Path) -> CheckpointArtifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    state = raw["state"]
    provenance = dict(state.get("provenance") or {})
    digest = str(provenance.get("checkpoint_sha256", ""))
    structure = ModelStructure(
        architecture=str(state["architecture_name"]),
        block_count=int(state["num_blocks"]),
        hidden_size=int(state["hidden_size"]),
        ffn_width=int(state["ffn_width"]),
        parameter_count=int(state["parameter_count"]),
        checkpoint_sha256=digest,
        metadata={"offline_simulation": bool(provenance.get("offline_simulation", False))},
    )
    return CheckpointArtifact(
        str(state["checkpoint_path"]),
        digest,
        structure.architecture,
        structure,
        dict(structure.metadata),
    )


def _active_artifacts(
    root: Path,
    candidate_model_id: str | None = None,
) -> tuple[CheckpointArtifact, CheckpointArtifact, str, str]:
    candidates = root / "models" / "candidates"
    baseline = _artifact_from_candidate(candidates / "M0000.json")
    active_id = str(json.loads((root / "models" / "active.json").read_text(encoding="utf-8"))["id"])
    selected_id = candidate_model_id or active_id
    candidate_path = candidates / (selected_id + ".json")
    if candidate_model_id is not None and not candidate_path.is_file():
        raise FileNotFoundError("explicit candidate model record not found: %s" % candidate_path)
    candidate = _artifact_from_candidate(candidate_path)
    selection_mode = "explicit" if candidate_model_id is not None else "active_default"
    return baseline, candidate, active_id, selection_mode


def _parse_candidate_models(values: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("--candidate-model must use NAME=MODEL_ID")
        name, model_id = raw.split("=", 1)
        name = name.strip()
        model_id = model_id.strip()
        if not name or not model_id:
            raise ValueError("--candidate-model must use non-empty NAME=MODEL_ID")
        if name in overrides:
            raise ValueError("duplicate --candidate-model name: %s" % name)
        overrides[name] = model_id
    return overrides


def _evaluate_one(
    spec: RunSpec,
    *,
    manifest: Path,
    wan_config: Path,
    clip_model_path: Path,
    output_root: Path,
    source_revision: str,
    max_cases: int,
    seed_count: int,
    latency_warmup: int,
    latency_repeats: int,
    quality_device: str | None,
) -> None:
    destination = output_root / spec.name
    try:
        baseline, candidate, active_id, selection_mode = _active_artifacts(
            spec.root,
            spec.candidate_model_id,
        )
        evaluator_config = WanVideoEvaluationConfig.from_mapping(
            {
                "wan_config": str(wan_config),
                "evaluation_manifest": str(manifest),
                "clip_model_path": str(clip_model_path),
                "device": "cuda:%d" % spec.gpu,
                "quality_device": quality_device or ("cuda:%d" % spec.gpu),
                "dtype": "bfloat16",
                "local_files_only": True,
                "max_cases": max_cases,
                "seed_count": seed_count,
                "sample_frames": 4,
                "max_quality_drop": 0.05,
                "latency_warmup": latency_warmup,
                "latency_repeats": latency_repeats,
                "source_revision": source_revision,
            },
            destination / "real-evaluation",
        )
        result = WanVideoEvaluator(evaluator_config, baseline=baseline)(candidate, {})
        _write_json(
            destination / "summary.json",
            {
                "status": "completed",
                "run": spec.name,
                "gpu": spec.gpu,
                "active_model_id": active_id,
                "candidate_model_id": spec.candidate_model_id or active_id,
                "candidate_selection": selection_mode,
                "candidate_checkpoint_path": candidate.path,
                "baseline_checkpoint_sha256": baseline.sha256,
                "candidate_checkpoint_sha256": candidate.sha256,
                "evaluation_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "result": result,
            },
        )
    except Exception as exc:  # pragma: no cover - exercised by remote dependencies
        _write_json(
            destination / "summary.json",
            {
                "status": "failed",
                "run": spec.name,
                "gpu": spec.gpu,
                "failure_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def _wait_for_decisions(specs: Sequence[RunSpec], round_id: str, timeout_s: float, poll_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all((spec.root / "rounds" / round_id / "decision.json").is_file() for spec in specs):
            return
        time.sleep(poll_s)
    missing = [spec.name for spec in specs if not (spec.root / "rounds" / round_id / "decision.json").is_file()]
    raise TimeoutError("missing terminal decisions for: %s" % ", ".join(missing))


def _parse_run(raw: str, gpu: int) -> RunSpec:
    if "=" not in raw:
        raise ValueError("--run must use NAME=ROOT")
    name, root = raw.split("=", 1)
    if not name.strip() or not root.strip():
        raise ValueError("--run must use non-empty NAME=ROOT")
    return RunSpec(name.strip(), Path(root).expanduser().resolve(), gpu)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="completed run as NAME=ROOT; repeat once per GPU")
    parser.add_argument(
        "--candidate-model",
        action="append",
        default=[],
        metavar="NAME=MODEL_ID",
        help="explicit candidate model ID for a named run; defaults to active.json",
    )
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids matching --run order")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wan-config", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--wait-round", default=None, help="wait until this round has decision.json for every run")
    parser.add_argument("--wait-timeout-s", type=float, default=24 * 3600)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument(
        "--quality-device",
        default=None,
        help="device for independent CLIP quality evaluation; defaults to the candidate GPU",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpu_ids = [int(item.strip()) for item in str(args.gpus).split(",") if item.strip()]
    if len(gpu_ids) != len(args.run):
        raise ValueError("--gpus count must match --run count")
    candidate_models = _parse_candidate_models(args.candidate_model)
    specs = [_parse_run(raw, gpu) for raw, gpu in zip(args.run, gpu_ids)]
    run_names = {spec.name for spec in specs}
    unknown_names = sorted(set(candidate_models) - run_names)
    if unknown_names:
        raise ValueError("--candidate-model names missing from --run: %s" % ", ".join(unknown_names))
    specs = [
        RunSpec(spec.name, spec.root, spec.gpu, candidate_models.get(spec.name))
        for spec in specs
    ]
    manifest = args.manifest.expanduser().resolve()
    wan_config = args.wan_config.expanduser().resolve()
    clip_model_path = args.clip_model_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.wait_round:
        _wait_for_decisions(specs, str(args.wait_round), float(args.wait_timeout_s), float(args.poll_s))

    context = mp.get_context("spawn")
    processes = []
    for spec in specs:
        process = context.Process(
            target=_evaluate_one,
            kwargs={
                "spec": spec,
                "manifest": manifest,
                "wan_config": wan_config,
                "clip_model_path": clip_model_path,
                "output_root": output_root,
                "source_revision": str(args.source_revision),
                "max_cases": int(args.max_cases),
                "seed_count": int(args.seed_count),
                "latency_warmup": int(args.latency_warmup),
                "latency_repeats": int(args.latency_repeats),
                "quality_device": args.quality_device,
            },
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    summaries = {}
    failed = False
    for spec, process in zip(specs, processes):
        path = output_root / spec.name / "summary.json"
        if path.is_file():
            summaries[spec.name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            failed = True
            summaries[spec.name] = {"status": "missing", "returncode": process.exitcode}
        failed = failed or process.exitcode not in (0, None) or summaries[spec.name].get("status") != "completed"
    _write_json(
        output_root / "summary.json",
        {
            "status": "failed" if failed else "completed",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "runs": summaries,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
