from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

from ..archive.model_candidate import ModelCandidate
from ..controller.schemas import CostEstimate, OperatorResult
from ..h3.inspector import H3Inspector
from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .base import ExecutionContext, OperatorValidationError


@dataclass(frozen=True)
class PrebuiltQuantizeOperator:
    """Adopt an immutable, pre-quantized H3 checkpoint produced by a trusted pipeline.

    The Harness deliberately does not implement an ad-hoc H3 weight converter. The
    quantized paths are fixed by deployment configuration and are never controller
    arguments, so this operator can be used to benchmark a known-good NVFP4/GGUF
    artifact without overwriting its parent.
    """

    variants: Mapping[int, Path]
    inspector: H3Inspector
    cost: CostEstimate = CostEstimate(wall_time_s=0.2)
    name: str = "quantize"
    description: str = "Adopt a fixed prebuilt H3 quantized checkpoint (paths are operator configuration, not LLM input)"

    def __post_init__(self) -> None:
        normalized = {int(bits): Path(path).resolve() for bits, path in self.variants.items()}
        if not normalized:
            raise OperatorValidationError("quantize requires at least one configured checkpoint variant")
        object.__setattr__(self, "variants", normalized)

    def schema(self) -> Mapping[str, Any]:
        return {"bits": "int"}

    def _source(self, args: Mapping[str, Any]) -> Tuple[int, Path]:
        if set(args) - {"bits"}:
            raise OperatorValidationError("quantize accepts only bits")
        bits = int(args.get("bits", 4))
        if bits not in self.variants:
            raise OperatorValidationError("no configured quantized checkpoint for %d bits" % bits)
        source = self.variants[bits]
        if not source.is_file():
            raise OperatorValidationError("configured quantized checkpoint is missing: %s" % source)
        return bits, source

    def validate(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> None:
        if not isinstance(args, Mapping) or "bits" not in args:
            raise OperatorValidationError("quantize requires bits")
        if isinstance(args["bits"], bool) or not isinstance(args["bits"], int):
            raise OperatorValidationError("quantize.bits has invalid type")
        bits, source = self._source(args)
        parent_bits = int(parent.quantization.get("bits") or 16)
        if parent_bits <= bits:
            raise OperatorValidationError("model is already quantized to %d bits or lower" % bits)
        if parent.checkpoint_path and Path(parent.checkpoint_path).resolve() == source:
            raise OperatorValidationError("quantized checkpoint must differ from parent")

    def estimate_cost(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        self.validate(parent, args, target)
        return self.cost

    def dry_run(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        return self.estimate_cost(parent, args, target)

    def execute(self, parent: ModelCandidate, args: Mapping[str, Any], runtime: ExecutionContext) -> OperatorResult:
        try:
            bits, source = self._source(args)
            self.validate(parent.state, args, TargetProfile("operator", "unknown", "unknown"))
            inspected = self.inspector.inspect(
                source,
                model_id=runtime.child_model_id,
                parent_model_id=parent.id,
                sampling_steps=parent.state.sampling_steps,
                components=copy.deepcopy(dict(parent.state.components)),
            )
            if inspected.architecture_name != "MiniMax-H3":
                raise ValueError("configured quantized checkpoint is not recognized as MiniMax-H3")
            warnings = list(inspected.warnings) + [
                "quality and hardware metrics are pending held-out benchmark; no metrics were inferred from quantization"
            ]
            runtime_state = dict(inspected.runtime_state)
            runtime_state.update({"operator": self.name, "metrics_stale": True})
            state = inspected.derive(
                runtime.child_model_id,
                parent_model_id=parent.id,
                checkpoint_path=str(source),
                warnings=warnings,
                runtime_state=runtime_state,
                provenance={
                    **dict(inspected.provenance),
                    "operator": self.name,
                    "parent_model_id": parent.id,
                    "prebuilt_variant": True,
                    "quantization_bits": bits,
                },
                measured_metrics={"model_size_gb": inspected.measured_metrics["model_size_gb"]},
            )
            return OperatorResult("success", state, self.cost, metrics={"prebuilt_variant": True})
        except (OSError, TypeError, ValueError, KeyError, OperatorValidationError) as exc:
            return OperatorResult("failed", None, CostEstimate(), failure_type="quantize_invalid", message=str(exc))
