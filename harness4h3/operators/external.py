from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..archive.model_candidate import ModelCandidate
from ..controller.schemas import CostEstimate, OperatorResult
from ..executor.local import LocalProcessExecutor
from ..h3.checkpoint import sha256_file
from ..h3.state import ModelState
from ..target.profile import TargetProfile
from .base import ExecutionContext, OperatorValidationError


def _checkpoint_digest(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return sha256_file(path)


@dataclass(frozen=True)
class ExternalScriptOperator:
    name: str
    description: str
    command: Tuple[str, ...]
    allowed_args: Mapping[str, Tuple[type, ...]]
    executor: LocalProcessExecutor
    cost: CostEstimate
    timeout_s: float = 3600.0

    def __init__(
        self,
        name: str,
        description: str,
        command: Sequence[str],
        allowed_args: Mapping[str, Tuple[type, ...]],
        executor: LocalProcessExecutor,
        cost: CostEstimate,
        timeout_s: float = 3600.0,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "command", tuple(command))
        object.__setattr__(self, "allowed_args", dict(allowed_args))
        object.__setattr__(self, "executor", executor)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "timeout_s", float(timeout_s))
        if not name or not self.command or timeout_s <= 0:
            raise OperatorValidationError("external operator requires a name, command, and positive timeout")

    def schema(self) -> Mapping[str, Any]:
        return {name: "/".join(kind.__name__ for kind in kinds) for name, kinds in self.allowed_args.items()}

    def validate(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> None:
        unknown = sorted(set(args) - set(self.allowed_args))
        if unknown:
            raise OperatorValidationError("unsupported argument(s) for %s: %s" % (self.name, ", ".join(unknown)))
        for name, value in args.items():
            allowed = self.allowed_args[name]
            if not isinstance(value, allowed) or (isinstance(value, bool) and bool not in allowed):
                raise OperatorValidationError("%s.%s has invalid type" % (self.name, name))

    def estimate_cost(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        self.validate(parent, args, target)
        return self.cost

    def dry_run(self, parent: ModelState, args: Mapping[str, Any], target: TargetProfile) -> CostEstimate:
        return self.estimate_cost(parent, args, target)

    def execute(self, parent: ModelCandidate, args: Mapping[str, Any], runtime: ExecutionContext) -> OperatorResult:
        experiment_dir = Path(runtime.experiment_dir).resolve()
        artifacts_dir = experiment_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        request_path = experiment_dir / "operator_request.json"
        result_path = experiment_dir / "external_result.json"
        request = {
            "operator": self.name,
            "parent": parent.to_dict(),
            "child_model_id": runtime.child_model_id,
            "operator_args": dict(args),
            "artifacts_dir": str(artifacts_dir),
        }
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        parent_path = Path(parent.checkpoint_path).resolve() if "://" not in parent.checkpoint_path else None
        before = _checkpoint_digest(parent_path) if parent_path is not None else None
        process = self.executor.execute(
            self.command + ("--request", request_path.name, "--result", result_path.name),
            experiment_dir,
            timeout_s=self.timeout_s,
        )
        after = _checkpoint_digest(parent_path) if parent_path is not None else None
        if before != after:
            return OperatorResult(
                "failed",
                None,
                CostEstimate(wall_time_s=process.wall_time_s),
                artifacts=[process.stdout_path, process.stderr_path],
                failure_type="parent_modified",
                message="external operator changed or removed the parent checkpoint",
            )
        if not process.ok:
            failure_type = process.failure_type
            message = process.message
            metrics = {"process": process.to_dict()}
            if not process.timed_out and result_path.is_file():
                try:
                    if result_path.stat().st_size <= 8 * 1024 * 1024:
                        failed_result = json.loads(result_path.read_text(encoding="utf-8"))
                        if isinstance(failed_result, Mapping) and failed_result.get("status") == "failed":
                            reported_failure = failed_result.get("failure_type")
                            if isinstance(reported_failure, str) and reported_failure:
                                failure_type = reported_failure
                                message = str(failed_result.get("message") or message)
                                reported_metrics = failed_result.get("metrics")
                                if isinstance(reported_metrics, Mapping):
                                    metrics.update(dict(reported_metrics))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            return OperatorResult(
                "failed",
                None,
                CostEstimate(wall_time_s=process.wall_time_s),
                artifacts=[process.stdout_path, process.stderr_path],
                metrics=metrics,
                failure_type=failure_type,
                message=message,
            )
        try:
            if result_path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("external result exceeds 8 MiB")
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping) or raw.get("status") != "success":
                raise ValueError(str(raw.get("message", "external result did not report success")))
            state = ModelState.from_dict(raw["output_state"])
            if state.model_id != runtime.child_model_id or state.parent_model_id != parent.id:
                raise ValueError("external result has incorrect child or parent model id")
            child_checkpoint = Path(state.checkpoint_path).resolve()
            if child_checkpoint == parent_path:
                raise ValueError("external operator returned the parent checkpoint as its child")
            try:
                child_checkpoint.relative_to(artifacts_dir.resolve())
            except ValueError:
                raise ValueError("child checkpoint must be inside the experiment artifacts directory")
            if not child_checkpoint.is_file():
                raise ValueError("child checkpoint does not exist")
            reported_cost = CostEstimate.from_mapping(raw.get("cost") or {})
            output_cost = CostEstimate(
                wall_time_s=max(process.wall_time_s, reported_cost.wall_time_s),
                gpu_hours=reported_cost.gpu_hours,
                controller_calls=reported_cost.controller_calls,
            )
            return OperatorResult(
                "success",
                state,
                output_cost,
                artifacts=[
                    process.stdout_path,
                    process.stderr_path,
                    str(child_checkpoint),
                    *([str(raw.get("metrics", {}).get("child_evidence_manifest"))]
                      if isinstance(raw.get("metrics"), Mapping)
                      and raw.get("metrics", {}).get("child_evidence_manifest")
                      else []),
                ],
                metrics={"process": process.to_dict(), **dict(raw.get("metrics") or {})},
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return OperatorResult(
                "failed",
                None,
                CostEstimate(wall_time_s=process.wall_time_s),
                artifacts=[process.stdout_path, process.stderr_path],
                metrics={"process": process.to_dict()},
                failure_type="invalid_operator_result",
                message=str(exc),
            )
