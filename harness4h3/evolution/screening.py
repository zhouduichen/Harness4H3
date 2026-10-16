"""Low-cost, restartable screening before independent Wan evaluation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..adapters.wan.contracts import CheckpointArtifact, WanModelAdapter
from ..operators.wan import WanCompressionRegistry
from ..sandbox.policy import SourcePolicyError, validate_source
from ..sandbox.runner import CompressionPlan
from .candidate import AlgorithmCandidate


class ScreeningStage(str, Enum):
    STATIC = "static"
    DIAGNOSTIC = "diagnostic"
    FULL = "full"


@dataclass(frozen=True)
class ScreeningResult:
    stage: str
    status: str
    elapsed_s: float
    evidence_level: str
    reasons: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    offline_simulation: bool = True

    def __post_init__(self) -> None:
        if self.stage not in {item.value for item in ScreeningStage}:
            raise ValueError("unsupported screening stage")
        if self.status not in {"passed", "rejected", "failed", "skipped"}:
            raise ValueError("unsupported screening status")
        if self.evidence_level not in {"static", "diagnostic", "real_measured", "none"}:
            raise ValueError("unsupported screening evidence level")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScreeningResult":
        return cls(
            stage=str(raw["stage"]),
            status=str(raw["status"]),
            elapsed_s=float(raw.get("elapsed_s", 0.0)),
            evidence_level=str(raw.get("evidence_level", "none")),
            reasons=tuple(str(item) for item in raw.get("reasons", [])),
            metrics=dict(raw.get("metrics") or {}),
            offline_simulation=bool(raw.get("offline_simulation", True)),
        )


DiagnosticEvaluator = Callable[["ScreeningContext", Path], Mapping[str, Any]]
FullEvaluator = Callable[["ScreeningContext"], Mapping[str, Any]]


@dataclass(frozen=True)
class ScreeningContext:
    parent: CheckpointArtifact
    adapter: WanModelAdapter
    profile: Mapping[str, Any]
    algorithm: AlgorithmCandidate
    plan: CompressionPlan
    registry: WanCompressionRegistry
    search_space_version: Optional[Mapping[str, Any]] = None
    diagnostic_evaluator: Optional[DiagnosticEvaluator] = None
    full_evaluator: Optional[FullEvaluator] = None


class CandidateScreeningPipeline:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "screening.json"

    def run(self, context: ScreeningContext) -> tuple[ScreeningResult, ...]:
        previous = self.load()
        if previous and previous[-1].status in {"rejected", "failed"}:
            return previous
        results = list(previous)
        static = _find(results, ScreeningStage.STATIC.value)
        if static is None:
            static = self._static(context)
            results.append(static)
            self._write(results)
        if static.status != "passed":
            return tuple(results)

        diagnostic = _find(results, ScreeningStage.DIAGNOSTIC.value)
        if diagnostic is None:
            diagnostic = self._diagnostic(context)
            results.append(diagnostic)
            self._write(results)
        if diagnostic.status != "passed":
            return tuple(results)

        if context.full_evaluator is None:
            return tuple(results)
        full = _find(results, ScreeningStage.FULL.value)
        if full is None:
            full = self._full(context)
            results.append(full)
            self._write(results)
        return tuple(results)

    def load(self) -> tuple[ScreeningResult, ...]:
        if not self.path.exists():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return tuple(ScreeningResult.from_dict(item) for item in raw.get("results", []))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return ()

    def _static(self, context: ScreeningContext) -> ScreeningResult:
        started = time.monotonic()
        reasons: list[str] = []
        try:
            validate_source(context.algorithm.source)
            extension_specs = {
                str(item["name"]): dict(item)
                for item in (context.search_space_version or {}).get("operators", [])
                if isinstance(item, Mapping) and item.get("name")
            }
            plan = CompressionPlan.from_mapping(context.plan.to_dict(), operator_specs=extension_specs)
            if plan.operator not in context.registry.names():
                reasons.append("operator_not_in_active_registry")
            _validate_static_args(plan, context.profile)
        except (SourcePolicyError, TypeError, ValueError) as exc:
            reasons.append(str(exc))
        return ScreeningResult(
            stage=ScreeningStage.STATIC.value,
            status="passed" if not reasons else "rejected",
            elapsed_s=time.monotonic() - started,
            evidence_level="static",
            reasons=tuple(reasons),
            metrics={"operator": context.plan.operator},
            offline_simulation=True,
        )

    def _diagnostic(self, context: ScreeningContext) -> ScreeningResult:
        started = time.monotonic()
        try:
            output_dir = self.root / "diagnostic"
            if context.diagnostic_evaluator is not None:
                metrics = dict(context.diagnostic_evaluator(context, output_dir))
            else:
                compression = context.registry.execute(
                    context.plan.operator,
                    context.adapter,
                    context.parent,
                    context.plan.args,
                    output_dir,
                )
                if compression.status != "success" or compression.child_checkpoint is None:
                    raise ValueError(compression.failure_type or "diagnostic_compression_failed")
                child = context.adapter.reload_model(compression.child_checkpoint)
                _assert_finite(child)
                child_metadata = context.adapter.inspect(child).metadata
                metrics = {
                    "valid": True,
                    "offline_simulation": bool(child_metadata.get("offline_simulation", True)),
                    "structural_change": dict(compression.structural_change),
                    "child_checkpoint_sha256": compression.child_checkpoint.sha256,
                }
            if metrics.get("valid") is not True:
                raise ValueError("diagnostic_evaluator_rejected")
            offline = bool(metrics.get("offline_simulation", True))
            return ScreeningResult(
                stage=ScreeningStage.DIAGNOSTIC.value,
                status="passed",
                elapsed_s=time.monotonic() - started,
                evidence_level="diagnostic",
                metrics=metrics,
                offline_simulation=offline,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return ScreeningResult(
                stage=ScreeningStage.DIAGNOSTIC.value,
                status="rejected",
                elapsed_s=time.monotonic() - started,
                evidence_level="diagnostic",
                reasons=(str(exc),),
                metrics={"valid": False},
                offline_simulation=True,
            )

    def _full(self, context: ScreeningContext) -> ScreeningResult:
        started = time.monotonic()
        try:
            metrics = dict(context.full_evaluator(context))
            offline = bool(metrics.get("offline_simulation", False))
            return ScreeningResult(
                stage=ScreeningStage.FULL.value,
                status="passed" if metrics.get("valid") is True else "rejected",
                elapsed_s=time.monotonic() - started,
                evidence_level="real_measured" if not offline else "diagnostic",
                reasons=tuple(str(item) for item in metrics.get("reasons", [])),
                metrics=metrics,
                offline_simulation=offline,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return ScreeningResult(
                stage=ScreeningStage.FULL.value,
                status="failed",
                elapsed_s=time.monotonic() - started,
                evidence_level="real_measured",
                reasons=(str(exc),),
                metrics={"valid": False},
                offline_simulation=False,
            )

    def _write(self, results: list[ScreeningResult]) -> None:
        payload = {"schema_version": 1, "results": [item.to_dict() for item in results]}
        fd, temporary = tempfile.mkstemp(prefix="screening.json.tmp-", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _find(results: list[ScreeningResult] | tuple[ScreeningResult, ...], stage: str) -> Optional[ScreeningResult]:
    return next((item for item in results if item.stage == stage), None)


def _validate_static_args(plan: CompressionPlan, profile: Mapping[str, Any]) -> None:
    known_blocks = {
        str(item.get("block_id"))
        for item in profile.get("layers", [])
        if isinstance(item, Mapping) and item.get("block_id")
    }
    if plan.operator == "prune_blocks":
        block_ids = plan.args.get("block_ids")
        if not block_ids or not set(block_ids) <= known_blocks:
            raise ValueError("static_invalid_block_selection")
        if len(set(block_ids)) >= len(known_blocks):
            raise ValueError("static_cannot_remove_all_blocks")
    if plan.operator == "prune_ffn_channels":
        retained = plan.args.get("retained_indices")
        width = int(profile.get("model", {}).get("ffn_width", 0) or 0)
        if not retained or any(int(item) < 0 or int(item) >= width for item in retained):
            raise ValueError("static_invalid_channel_selection")
    if plan.operator == "tica_prune":
        mode = str(plan.args.get("mode", ""))
        if mode not in {"block", "ffn", "block_plus_ffn"}:
            raise ValueError("static_invalid_tica_mode")
        block_ids = plan.args.get("block_ids", [])
        retained = plan.args.get("retained_indices", [])
        width = int(profile.get("model", {}).get("ffn_width", 0) or 0)
        if mode in {"block", "block_plus_ffn"} and (not block_ids or not set(block_ids) <= known_blocks):
            raise ValueError("static_invalid_tica_block_selection")
        if len(set(block_ids)) >= len(known_blocks):
            raise ValueError("static_cannot_remove_all_blocks")
        if mode in {"ffn", "block_plus_ffn"} and (not retained or any(int(item) < 0 or int(item) >= width for item in retained)):
            raise ValueError("static_invalid_tica_channel_selection")


def _assert_finite(model: Any) -> None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        # DiffusionPipeline exposes its trainable modules through named
        # components rather than a Module.parameters() method.  Structural
        # screening must still check the actual Wan transformer after reload.
        transformer = getattr(model, "transformer", None)
        parameters = getattr(transformer, "parameters", None)
    if not callable(parameters):
        raise ValueError("diagnostic_model_parameters_unavailable")
    for parameter in parameters():
        if not bool(parameter.detach().isfinite().all()):
            raise ValueError("diagnostic_model_integrity_failed")


__all__ = ["CandidateScreeningPipeline", "ScreeningContext", "ScreeningResult", "ScreeningStage"]
