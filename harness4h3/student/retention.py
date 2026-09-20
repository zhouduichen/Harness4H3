"""Checkpoint retention decisions for the autonomous Student campaign."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


_CANDIDATE_ID = re.compile(r"^student_[0-9]{4,}$")


@dataclass(frozen=True)
class RetentionDecision:
    outcome: str
    candidate_id: str
    candidate_path: str
    delete_path: Optional[str]
    protected_paths: Tuple[str, ...]
    reason: str


def retain_after_evaluation(
    candidate_path: Path,
    candidate_id: str,
    *,
    outcome: str,
    protected: Iterable[Path] = (),
) -> RetentionDecision:
    # Keep the final path component unresolved so a symlink cannot be turned
    # into its target before the safety check below.
    path = Path(candidate_path).absolute()
    candidate_id = str(candidate_id)
    protected_paths = tuple(sorted(str(Path(value).resolve()) for value in protected))
    if str(path) in protected_paths:
        return RetentionDecision(outcome, candidate_id, str(path), None, protected_paths, "protected_candidate")
    if outcome in {"accepted", "active", "best", "in_flight"}:
        return RetentionDecision(outcome, candidate_id, str(path), None, protected_paths, "successful_or_protected_outcome")
    if outcome not in {"rejected", "failed", "superseded"}:
        return RetentionDecision(outcome, candidate_id, str(path), None, protected_paths, "unknown_outcome_protected")
    if not _CANDIDATE_ID.fullmatch(candidate_id):
        return RetentionDecision(outcome, candidate_id, str(path), None, protected_paths, "invalid_candidate_id")
    if path.suffix != ".safetensors" or path.name != "student.safetensors" or path.parent.name != candidate_id:
        return RetentionDecision(outcome, candidate_id, str(path), None, protected_paths, "unexpected_checkpoint_layout")
    if path.is_symlink():
        return RetentionDecision(outcome, candidate_id, str(path), None, protected_paths, "symlink_refused")
    return RetentionDecision(outcome, candidate_id, str(path), str(path), protected_paths, "rejected_child_deletable")


def apply_retention(decision: RetentionDecision) -> RetentionDecision:
    """Delete only the exact decision target after experience is durable."""

    if decision.delete_path is None:
        return decision
    target = Path(decision.delete_path)
    if target.is_symlink() or not target.is_file():
        return RetentionDecision(
            decision.outcome,
            decision.candidate_id,
            decision.candidate_path,
            None,
            decision.protected_paths,
            "candidate_already_absent_or_refused",
        )
    target.unlink()
    return RetentionDecision(
        decision.outcome,
        decision.candidate_id,
        decision.candidate_path,
        None,
        decision.protected_paths,
        "candidate_deleted",
    )


__all__ = ["RetentionDecision", "apply_retention", "retain_after_evaluation"]
