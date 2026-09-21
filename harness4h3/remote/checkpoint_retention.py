"""Bounded retention for large remote H3 candidate checkpoints."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, asdict
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Optional

from .ssh import SSHClient


_MODEL_ID = re.compile(r"^M[0-9]{4,}$")
_CHECKPOINT_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class CheckpointRetentionResult:
    policy: str
    checkpoint_path: Optional[str]
    model_id: Optional[str]
    outcome: str
    retained: bool
    deleted: bool
    reason: str
    error: Optional[str] = None
    targets: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["targets"] = [dict(item) for item in self.targets]
        return value


class RemoteCheckpointRetention:
    """Delete only evaluated, rejected child weights in the campaign area.

    A real H3 checkpoint is tens of gigabytes, so keeping every rejected child
    is not a useful default.  This helper never deletes evidence, result JSON,
    logs, or the configured parent.  The accepted model remains available as
    the next lineage parent.
    """

    def __init__(self, client: SSHClient, results_root: str, policy: str = "rejected_candidate_v1"):
        self.client = client
        self.policy = str(policy).strip()
        if self.policy not in {"rejected_candidate_v1", "keep_all"}:
            raise ValueError("unsupported remote checkpoint retention policy: %s" % self.policy)
        raw_root = str(results_root or "").strip()
        if not raw_root or "://" in raw_root:
            raise ValueError("remote checkpoint retention requires a local results_root")
        self.results_root = PurePosixPath(posixpath.normpath(raw_root))
        if not self.results_root.is_absolute():
            raise ValueError("remote checkpoint retention root must be absolute")
        self.checkpoint_root = self.results_root / "continuous"

    @staticmethod
    def _path_value(value: Any) -> Optional[PurePosixPath]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw or "://" in raw:
            return None
        path = PurePosixPath(posixpath.normpath(raw))
        return path if path.is_absolute() else None

    def _candidate_target(self, checkpoint_path: Any, model_id: Any) -> tuple[Optional[PurePosixPath], Optional[str]]:
        model = str(model_id or "").strip()
        if not _MODEL_ID.fullmatch(model):
            return None, "invalid_model_id"
        target = self._path_value(checkpoint_path)
        if target is None:
            return None, "non_local_path_refused"
        try:
            relative = target.relative_to(self.checkpoint_root)
        except ValueError:
            return None, "path_outside_checkpoint_root"
        if tuple(relative.parts) != (model, model + _CHECKPOINT_SUFFIX):
            return None, "unexpected_checkpoint_layout"
        return target, None

    def _remove(self, target: PurePosixPath) -> Mapping[str, Any]:
        return self.client.remove_file(str(target))

    def apply(
        self,
        checkpoint_path: Any,
        model_id: Any,
        outcome: str,
        parent_checkpoint_path: Any = None,
    ) -> CheckpointRetentionResult:
        """Apply retention after the benchmark has classified ``outcome``."""

        checkpoint = str(checkpoint_path) if checkpoint_path is not None else None
        model = str(model_id) if model_id is not None else None
        if self.policy == "keep_all":
            return CheckpointRetentionResult(
                self.policy, checkpoint, model, outcome, True, False, "keep_all_policy"
            )
        if outcome == "accepted_candidate":
            return CheckpointRetentionResult(
                self.policy, checkpoint, model, outcome, True, False, "accepted_candidate_protected"
            )
        if outcome not in {"rejected_candidate", "superseded_candidate"}:
            return CheckpointRetentionResult(
                self.policy, checkpoint, model, outcome, True, False, "unclassified_outcome_protected"
            )
        target, error = self._candidate_target(checkpoint_path, model_id)
        if target is None:
            return CheckpointRetentionResult(
                self.policy, checkpoint, model, outcome, True, False, error or "target_refused"
            )
        parent = self._path_value(parent_checkpoint_path)
        if parent is not None and target == parent:
            return CheckpointRetentionResult(
                self.policy, checkpoint, model, outcome, True, False, "parent_protected"
            )
        if model == "M0000":
            return CheckpointRetentionResult(
                self.policy, checkpoint, model, outcome, True, False, "m0000_protected"
            )
        try:
            response = dict(self._remove(target))
        except Exception as exc:
            return CheckpointRetentionResult(
                self.policy, checkpoint, model, outcome, True, False, "delete_failed", str(exc)
            )
        status = str(response.get("status", "")).strip().lower()
        if status in {"deleted", "missing"}:
            return CheckpointRetentionResult(
                self.policy,
                checkpoint,
                model,
                outcome,
                False,
                status == "deleted",
                (
                    "superseded_candidate_deleted"
                    if outcome == "superseded_candidate" and status == "deleted"
                    else "superseded_candidate_already_absent"
                    if outcome == "superseded_candidate"
                    else "rejected_candidate_deleted"
                    if status == "deleted"
                    else "rejected_candidate_already_absent"
                ),
                targets=(
                    {"path": str(target), "status": status},
                ),
            )
        return CheckpointRetentionResult(
            self.policy,
            checkpoint,
            model,
            outcome,
            True,
            False,
            "delete_refused",
            str(response.get("reason") or "remote_refusal"),
            targets=({"path": str(target), "status": status},),
        )

    def cleanup_failed(self, output_dir: Any, model_id: Any) -> CheckpointRetentionResult:
        """Remove failed-worker checkpoint artifacts from one exact child dir.

        Real workers can leave a named diagnostic checkpoint (for example
        ``M0001.failed-invalid-parent-id.safetensors``) after an exception.
        It is still a full-size model file, so it belongs to the failed-worker
        cleanup set.  The remote ``find`` is rooted at the already validated
        child directory and every returned path is validated again before it
        can be unlinked; evidence, logs, and arbitrary sibling files remain
        untouched.
        """

        model = str(model_id or "").strip()
        if self.policy == "keep_all":
            return CheckpointRetentionResult(self.policy, None, model or None, "failed_experiment", True, False, "keep_all_policy")
        if not _MODEL_ID.fullmatch(model):
            return CheckpointRetentionResult(self.policy, None, model or None, "failed_experiment", True, False, "invalid_model_id")
        directory = self._path_value(output_dir)
        if directory is None or directory.parent != self.checkpoint_root or directory.name != model:
            return CheckpointRetentionResult(self.policy, None, model, "failed_experiment", True, False, "unexpected_failed_output_layout")
        paths = [directory / (model + _CHECKPOINT_SUFFIX), directory / (model + _CHECKPOINT_SUFFIX + ".part")]
        target_records = []
        errors = []
        deleted_any = False
        try:
            discovered = self.client.find(model + ".failed-*.safetensors", root=str(directory))
        except Exception as exc:
            discovered = []
            errors.append("failed checkpoint discovery: %s" % exc)
        for value in discovered:
            target = self._path_value(value)
            if target is None or target.parent != directory or not re.fullmatch(
                re.escape(model) + r"\.failed-[A-Za-z0-9_.-]+\.safetensors", target.name
            ):
                continue
            if target not in paths:
                paths.append(target)
        for target in paths:
            try:
                response = dict(self._remove(target))
                status = str(response.get("status", "")).strip().lower()
                target_records.append({"path": str(target), "status": status})
                deleted_any = deleted_any or status == "deleted"
                if status not in {"deleted", "missing"}:
                    errors.append("%s: %s" % (target, response.get("reason") or "remote_refusal"))
            except Exception as exc:
                errors.append("%s: %s" % (target, exc))
                target_records.append({"path": str(target), "status": "error", "error": str(exc)})
        return CheckpointRetentionResult(
            self.policy,
            None,
            model,
            "failed_experiment",
            bool(errors),
            deleted_any,
            "failed_worker_artifacts_cleaned" if not errors else "failed_worker_cleanup_incomplete",
            "; ".join(errors) if errors else None,
            tuple(target_records),
        )


__all__ = ["CheckpointRetentionResult", "RemoteCheckpointRetention"]
