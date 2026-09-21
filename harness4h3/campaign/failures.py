"""Deterministic-first structured experiment failure attribution."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FailureReport:
    stage: str
    category: str
    responsible_variables: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    deterministic_fix: Optional[str]
    confidence: float
    prohibited_changes: Tuple[str, ...]
    failure_code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["responsible_variables"] = list(self.responsible_variables)
        value["evidence_ids"] = list(self.evidence_ids)
        value["prohibited_changes"] = list(self.prohibited_changes)
        return value


_KNOWN = {
    "proposal_invalid": ("schema_validation", ("proposal",), "repair the structured proposal fields", ("evaluator", "target_profile")),
    "shape_mismatch": ("architecture_shape", ("architecture",), "repair the registered shape parameters", ("evaluator",)),
    "manifest_digest_mismatch": ("integrity", ("compile_manifest",), None, ("proposal", "evaluator")),
    "checkpoint_missing": ("artifact_missing", ("checkpoint",), "recreate the child artifact before evaluation", ("parent", "evaluator")),
    "worker_oom": ("resource_oom", ("batch_size", "memory_budget"), "reduce the trusted resource recipe", ("target_profile",)),
    "worker_timeout": ("resource_timeout", ("training_budget",), "retry with a bounded resource budget", ("evaluator",)),
    "video_missing": ("generation_artifact_missing", ("generation_output",), "produce a generation artifact before quality scoring", ("quality_metric",)),
    "video_decode_failed": ("generation_decode", ("video_artifact",), "repair or regenerate the video artifact", ("video_decode_failed",)),
    "video_black_frames": ("generation_black_frame", ("sampler", "video_artifact"), "reject the invalid generation artifact", ("video_black_frames",)),
    "video_nonfinite": ("generation_nonfinite", ("video_artifact",), "reject the non-finite generation artifact", ("video_nonfinite",)),
    "quality_invalid": ("metric_invalid", ("quality_metric",), "rerun the fixed quality verifier", ("quality_score",)),
    "metric_missing": ("metric_missing", ("verifier_bank",), "collect the required fixed verifier evidence", ("hard_gate",)),
}


class FailureAttributor:
    def attribute(self, stage: str, result: Mapping[str, Any], evidence_ids: Sequence[str]) -> FailureReport:
        raw_code = str(result.get("failure_code") or result.get("category") or "unknown_failure")
        message = str(result.get("message") or raw_code)
        known = _KNOWN.get(raw_code)
        if known is None:
            category = "unclassified_experimental_failure"
            variables = ("experiment",)
            deterministic_fix = None
            confidence = 0.2
            prohibited = ("evaluator", "target_profile", "verifier_bank")
        else:
            category, variables, deterministic_fix, prohibited = known
            confidence = 1.0
        return FailureReport(
            stage=str(stage),
            category=category,
            responsible_variables=tuple(variables),
            evidence_ids=tuple(dict.fromkeys(str(item) for item in evidence_ids)),
            deterministic_fix=deterministic_fix,
            confidence=confidence,
            prohibited_changes=tuple(prohibited),
            failure_code=raw_code,
            message=message[:2000],
        )


__all__ = ["FailureAttributor", "FailureReport"]
