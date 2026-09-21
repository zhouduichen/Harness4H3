"""Structured, read-only LLM review decisions for long experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple


REVIEW_ACTIONS = ("continue", "stop", "replan", "review_only")
_REVIEW_FIELDS = {
    "action",
    "reason",
    "evidence_ids",
    "confidence",
    "next_review_after_s",
    "risks",
}


def _clip(value: Any, depth: int = 0) -> Any:
    """Bound runtime context before it is placed in an LLM prompt."""

    if depth > 4:
        return "<depth-limited>"
    if isinstance(value, Mapping):
        return {str(key): _clip(item, depth + 1) for key, item in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_clip(item, depth + 1) for item in list(value)[:32]]
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:1997] + "..."
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:2000]


@dataclass(frozen=True)
class ReviewDecision:
    """The only actions an LLM may request while an experiment is running."""

    action: str
    reason: str
    evidence_ids: Tuple[str, ...]
    confidence: float
    next_review_after_s: float
    risks: Tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReviewDecision":
        if not isinstance(raw, Mapping):
            raise ValueError("review decision must be an object")
        unknown = sorted(set(raw) - _REVIEW_FIELDS)
        if unknown:
            raise ValueError("unknown review decision field(s): %s" % ", ".join(str(item) for item in unknown))
        missing = sorted(_REVIEW_FIELDS - set(raw))
        if missing:
            raise ValueError("missing review decision field(s): %s" % ", ".join(missing))
        action = raw["action"]
        if not isinstance(action, str) or action not in REVIEW_ACTIONS:
            raise ValueError("action must be one of %s" % ", ".join(REVIEW_ACTIONS))
        reason = raw["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        evidence = raw["evidence_ids"]
        if not isinstance(evidence, (list, tuple)) or not all(isinstance(item, str) and item for item in evidence):
            raise ValueError("evidence_ids must be a list of non-empty strings")
        confidence = raw["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        delay = raw["next_review_after_s"]
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise ValueError("next_review_after_s must be numeric")
        delay = float(delay)
        if not math.isfinite(delay) or not 0.0 < delay <= 86400.0:
            raise ValueError("next_review_after_s must be finite and in (0, 86400]")
        risks = raw["risks"]
        if not isinstance(risks, (list, tuple)) or not all(isinstance(item, str) for item in risks):
            raise ValueError("risks must be a list of strings")
        return cls(
            action=action,
            reason=reason[:2000],
            evidence_ids=tuple(dict.fromkeys(evidence)),
            confidence=confidence,
            next_review_after_s=delay,
            risks=tuple(item[:2000] for item in risks[:32]),
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["evidence_ids"] = list(self.evidence_ids)
        value["risks"] = list(self.risks)
        return value


def review_json_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": list(REVIEW_ACTIONS)},
            "reason": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            # Keep the grammar portable across the remote vLLM backend.  The
            # parser still enforces strict positivity in ReviewDecision.
            "next_review_after_s": {"type": "number", "minimum": 0, "maximum": 86400},
            "risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": sorted(_REVIEW_FIELDS),
    }


def review_prompt(request: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    bounded = _clip(request)
    return (
        "You are the read-only supervisor for a real MiniMax-H3 experiment. "
        "Return exactly one JSON controller review using the supplied schema. "
        "Review evidence, telemetry, failures, and constraints; do not invent measurements. "
        "Use continue when the active job is healthy, review_only when no action is needed, "
        "replan when the next experiment should be reconsidered, and stop only for a clear "
        "safety or critical-regression signal. A non-zero trusted worker result, checkpoint "
        "load error, timeout, or missing result is recoverable experiment evidence: choose "
        "replan so the main Controller can select a different registered operator; do not "
        "stop the whole campaign unless the evidence shows tampering, data corruption, or "
        "an explicit safety incident. You cannot change TargetProfile, evaluator hard "
        "gates, or evidence. GPU0 may be reserved by the ComfyUI lease, but treat the live gpu_isolation snapshot and scheduler reservation as authoritative; "
        "do not assume a fixed controller GPU, and never request, release, or reassign cards directly. Do not output commands, "
        "paths, or ExperimentPlan fields. A single transient controller_review_unavailable event "
        "or SSH/port-forward error is not by itself a safety failure: if the latest preflight is "
        "ready and the worker/evaluator is still alive, use continue or review_only and allow the "
        "next heartbeat to retry. Request stop for connectivity only after repeated failures or "
        "when the active job cannot be safely observed or controlled. SCHEMA=%s CONTEXT=%s"
        % (json.dumps(schema, ensure_ascii=False, sort_keys=True), json.dumps(bounded, ensure_ascii=False, sort_keys=True))
    )


__all__ = ["REVIEW_ACTIONS", "ReviewDecision", "review_json_schema", "review_prompt"]
