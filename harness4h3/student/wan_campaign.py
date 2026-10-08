"""Trusted contracts for a vLLM-controlled real Wan experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class WanProposalError(ValueError):
    """Raised when a vLLM Wan proposal cannot cross the trusted boundary."""


_PROPOSAL_FIELDS = {
    "schema_version",
    "proposal_id",
    "parent_checkpoint_sha256",
    "steps",
    "teacher_reuse_steps",
    "learning_rate",
    "focus",
    "hypothesis",
}
_FOCUS = {"semantic", "temporal", "stability", "efficiency"}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WanProposalError("%s must be an object" % name)
    return value


def _require(raw: Mapping[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in raw]
    if missing:
        raise WanProposalError("proposal is missing: %s" % ", ".join(missing))


def _string(value: Any, name: str, *, max_length: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WanProposalError("%s must be a non-empty string" % name)
    result = value.strip()
    if max_length is not None and len(result) > max_length:
        raise WanProposalError("%s is too long" % name)
    return result


@dataclass(frozen=True)
class WanTrainingProposal:
    schema_version: int
    proposal_id: str
    parent_checkpoint_sha256: str
    steps: int
    teacher_reuse_steps: int
    learning_rate: float
    focus: str
    hypothesis: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_parent_sha256: str,
    ) -> "WanTrainingProposal":
        raw = _mapping(value, "proposal")
        unknown = sorted(set(raw) - _PROPOSAL_FIELDS)
        if unknown:
            raise WanProposalError("proposal has unknown field(s): %s" % ", ".join(unknown))
        _require(raw, tuple(sorted(_PROPOSAL_FIELDS)))
        if raw["schema_version"] != 1:
            raise WanProposalError("schema_version must be 1")
        proposal_id = _string(raw["proposal_id"], "proposal_id")
        if re.fullmatch(r"wan_[0-9]{4,}", proposal_id) is None:
            raise WanProposalError("proposal_id must match wan_[0-9]{4,}")
        parent = _string(raw["parent_checkpoint_sha256"], "parent_checkpoint_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", parent) is None:
            raise WanProposalError("parent_checkpoint_sha256 must be a lowercase SHA-256 digest")
        expected = str(expected_parent_sha256).strip().lower()
        if parent != expected:
            raise WanProposalError("parent_checkpoint_sha256 does not match trusted parent")
        steps = raw["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 32:
            raise WanProposalError("steps must be an integer in [1, 32]")
        reuse = raw["teacher_reuse_steps"]
        if isinstance(reuse, bool) or not isinstance(reuse, int) or not 1 <= reuse <= steps:
            raise WanProposalError("teacher_reuse_steps must be an integer in [1, steps]")
        learning_rate = raw["learning_rate"]
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise WanProposalError("learning_rate must be numeric")
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or not 1e-8 <= learning_rate <= 1e-4:
            raise WanProposalError("learning_rate must be finite and in [1e-8, 1e-4]")
        focus = _string(raw["focus"], "focus")
        if focus not in _FOCUS:
            raise WanProposalError("focus must be one of: %s" % ", ".join(sorted(_FOCUS)))
        hypothesis = _string(raw["hypothesis"], "hypothesis", max_length=512)
        return cls(
            schema_version=1,
            proposal_id=proposal_id,
            parent_checkpoint_sha256=parent,
            steps=steps,
            teacher_reuse_steps=reuse,
            learning_rate=learning_rate,
            focus=focus,
            hypothesis=hypothesis,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(evaluation: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    quality = evaluation.get("quality_metrics")
    if isinstance(quality, Mapping) and quality.get(key) is not None:
        return float(quality[key])
    direct = evaluation.get(key)
    return float(direct) if direct is not None else float(default)


def candidate_decision(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    teacher_semantic: float,
    semantic_floor_ratio: float,
    quality_floor_ratio: float,
    min_semantic_delta: float,
) -> dict[str, Any]:
    """Return a deterministic promotion decision for one evaluated child."""

    if candidate.get("status") != "success" or not bool(candidate.get("valid", True)):
        return {"accepted": False, "reason": "evaluation_failed"}
    if candidate.get("offline_simulation") is not False:
        return {"accepted": False, "reason": "offline_simulation"}
    if not bool(candidate.get("continuation_verified")):
        return {"accepted": False, "reason": "contract_invalid"}
    semantic = _metric(candidate, "semantic")
    quality = float(candidate.get("quality_score") or _metric(candidate, "score"))
    if semantic < float(teacher_semantic) * float(semantic_floor_ratio):
        return {
            "accepted": False,
            "reason": "semantic_floor",
            "semantic": semantic,
            "semantic_floor": float(teacher_semantic) * float(semantic_floor_ratio),
        }
    parent_semantic = _metric(parent, "semantic")
    parent_quality = float(parent.get("quality_score") or _metric(parent, "score"))
    if quality < parent_quality * float(quality_floor_ratio):
        return {"accepted": False, "reason": "quality_floor", "quality": quality, "parent_quality": parent_quality}
    if semantic >= parent_semantic + float(min_semantic_delta):
        return {"accepted": True, "reason": "semantic_improved", "semantic": semantic, "quality": quality}
    if semantic >= parent_semantic and quality > parent_quality:
        return {"accepted": True, "reason": "quality_improved", "semantic": semantic, "quality": quality}
    return {"accepted": False, "reason": "no_improvement", "semantic": semantic, "quality": quality}


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "WanProposalError",
    "WanTrainingProposal",
    "append_event",
    "atomic_write_json",
    "candidate_decision",
    "sha256_file",
]
