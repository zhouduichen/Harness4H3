from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ..archive.model_candidate import ModelCandidate
from ..controller.schemas import CostEstimate, OperatorResult
from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .base import ExecutionContext, OperatorRegistry, OperatorValidationError


@dataclass(frozen=True)
class RuntimeMemoryOperator:
    """Clone a model with one explicit runtime-memory policy.

    The operator never edits weights or the parent checkpoint. The benchmark
    adapter is responsible for translating the policy into concrete workflow
    controls and for rejecting workflows that cannot honor the requested policy.
    """

    name: str
    description: str
    policy_kind: str
    allowed_args: Mapping[str, Tuple[type, ...]]
    cost: CostEstimate = CostEstimate(wall_time_s=0.05)

    def schema(self) -> Mapping[str, Any]:
        return {key: "/".join(kind.__name__ for kind in kinds) for key, kinds in self.allowed_args.items()}

    def validate(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> None:
        if not isinstance(args, Mapping):
            raise OperatorValidationError("runtime operator arguments must be a mapping")
        unknown = sorted(set(args) - set(self.allowed_args))
        if unknown:
            raise OperatorValidationError("unsupported argument(s) for %s: %s" % (self.name, ", ".join(unknown)))
        missing = sorted(set(self.allowed_args) - set(args))
        if missing:
            raise OperatorValidationError("%s requires argument(s): %s" % (self.name, ", ".join(missing)))
        for key, value in args.items():
            allowed = self.allowed_args[key]
            if isinstance(value, bool) or not isinstance(value, allowed):
                raise OperatorValidationError("%s.%s has invalid type" % (self.name, key))
        if parent.architecture_name.lower().find("h3") < 0:
            raise OperatorValidationError("%s requires an H3 model" % self.name)
        if self.policy_kind == "runtime_offload" and args["mode"] not in {"balanced", "aggressive"}:
            raise OperatorValidationError("runtime_offload.mode must be balanced or aggressive")
        if self.policy_kind == "vae_tiling":
            tile_size = int(args["tile_size"])
            overlap = int(args["overlap"])
            if tile_size < 64 or tile_size > 1024:
                raise OperatorValidationError("vae_tiling.tile_size must be between 64 and 1024")
            if overlap < 0 or overlap >= tile_size:
                raise OperatorValidationError("vae_tiling.overlap must be non-negative and below tile_size")
        if self.policy_kind == "inference_chunking":
            chunk_size = int(args["chunk_size"])
            if chunk_size < 1 or chunk_size > 32:
                raise OperatorValidationError("inference_chunking.chunk_size must be between 1 and 32")

    def estimate_cost(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        self.validate(parent, args, target)
        return self.cost

    def execute(self, parent: ModelCandidate, args: Mapping[str, Any], runtime: ExecutionContext) -> OperatorResult:
        try:
            self.validate(parent.state, args, TargetProfile("runtime", "unknown", "unknown"))
            policy = {"kind": self.policy_kind, "args": copy.deepcopy(dict(args))}
            runtime_state = copy.deepcopy(dict(parent.state.runtime_state))
            runtime_state.update(
                {
                    "runtime_policy": policy,
                    "metrics_stale": True,
                    "runtime_parent_model_id": parent.id,
                }
            )
            provenance = copy.deepcopy(dict(parent.state.provenance))
            provenance.update({"operator": self.name, "parent_model_id": parent.id, "runtime_policy": policy})
            state = parent.state.derive(
                runtime.child_model_id,
                checkpoint_path=parent.state.checkpoint_path,
                runtime_state=runtime_state,
                measured_metrics=copy.deepcopy(dict(parent.state.measured_metrics)),
                provenance=provenance,
            )
            return OperatorResult("success", state, self.cost, metrics={"runtime_policy": policy})
        except (TypeError, ValueError, OperatorValidationError) as exc:
            return OperatorResult("failed", None, CostEstimate(), failure_type="runtime_policy_invalid", message=str(exc))


def build_runtime_registry(backend: Any = None) -> OperatorRegistry:
    """Return the fake Phase-I registry with the M6 runtime operators added."""
    from .fake import build_fake_registry

    registry = build_fake_registry(backend, include_runtime=False)
    registry.register(
        RuntimeMemoryOperator(
            "runtime_offload",
            "Runtime offload policy: reduce residency without changing model weights",
            "runtime_offload",
            {"mode": (str,)},
        )
    )
    registry.register(
        RuntimeMemoryOperator(
            "vae_tiling",
            "Decode VAE tiles to bound temporary activation memory",
            "vae_tiling",
            {"tile_size": (int,), "overlap": (int,)},
        )
    )
    registry.register(
        RuntimeMemoryOperator(
            "inference_chunking",
            "Chunk H3 inference when the workflow exposes a compatible input",
            "inference_chunking",
            {"chunk_size": (int,)},
        )
    )
    return registry
