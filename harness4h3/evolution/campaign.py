"""Persistent parent-branch scheduling for budgeted EvoCompress campaigns."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from .selection import select_representatives


class BranchCampaign:
    """Append auditable round costs and fairly rotate selected parent branches."""

    COST_FIELDS = (
        "parent_load_seconds",
        "profile_seconds",
        "generation_seconds",
        "sandbox_seconds",
        "training_seconds",
        "verification_seconds",
        "evaluation_seconds",
        "retry_seconds",
        "unattributed_seconds",
    )

    def __init__(self, root: Path, *, baseline_quality: float, budget_seconds: float, protocol: Optional[Mapping[str, Any]] = None, deployment_cost_metric: str = "inference_wall_time_s_mean") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "branch-campaign.json"
        self.events_path = self.root / "branch-events.jsonl"
        self.baseline_quality = float(baseline_quality)
        self.budget_seconds = float(budget_seconds)
        self.protocol = dict(protocol or {})
        self.deployment_cost_metric = str(deployment_cost_metric)
        self.protocol_sha256 = hashlib.sha256(json.dumps(self.protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        self.state = self._load()
        self.state.setdefault("schema_version", 1)
        self.state.setdefault("budget_seconds", self.budget_seconds)
        self.state.setdefault("baseline_quality", self.baseline_quality)
        self.state.setdefault("branches", {})
        self.state.setdefault("spent_seconds", 0.0)
        self.state.setdefault("next_index", 0)
        self.state.setdefault("protocol_sha256", self.protocol_sha256)
        self.state.setdefault("pending_branch", None)
        self.state.setdefault("deployment_cost_metric", self.deployment_cost_metric)
        if (self.state["budget_seconds"] != self.budget_seconds or
                self.state["baseline_quality"] != self.baseline_quality or
                self.state["protocol_sha256"] != self.protocol_sha256):
            raise ValueError("campaign budget, quality floor, and protocol are immutable on resume")
        if self.state["deployment_cost_metric"] != self.deployment_cost_metric:
            raise ValueError("deployment cost metric is immutable on resume")
        self._reconcile_event_ledger()
        self._save()

    def _reconcile_event_ledger(self) -> None:
        """Rebuild mutable counters and branch heads from the append-only log."""
        if not self.events_path.exists():
            return
        events = [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        branches = self.state["branches"]
        for branch in branches.values():
            branch["rounds"] = 0
        for event in events:
            branch_id = str(event["branch_id"])
            branch = branches.setdefault(branch_id, {"head_candidate_id": event["parent_model_id"], "rounds": 0})
            branch["rounds"] = int(branch.get("rounds", 0)) + 1
            if event.get("status") == "promoted" and event.get("child_model_id"):
                branch["head_candidate_id"] = event["child_model_id"]
        self.state["spent_seconds"] = sum(float(event.get("total_seconds", 0.0)) for event in events)

    def select(self, front: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        representatives = select_representatives(front, baseline_quality=self.baseline_quality, cost_metric=self.deployment_cost_metric)
        branches = self.state["branches"]
        for item in representatives:
            mode, cid = item.mode, item.candidate_id
            if mode not in branches:
                branches[mode] = {"head_candidate_id": cid, "seed_candidate_id": cid, "rounds": 0}
        self.state["selected_modes"] = [item.mode for item in representatives]
        self._save()
        return [{"mode": item.mode, **branches[item.mode], "selected_candidate_id": item.candidate_id} for item in representatives]

    def next_branch(self, front: list[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
        pending = self.state.get("pending_branch")
        if isinstance(pending, Mapping):
            return dict(pending)
        branches = self.select(front)
        if not branches or self.remaining_seconds <= 0:
            return None
        index = int(self.state["next_index"]) % len(branches)
        branch = branches[index]
        self.state["next_index"] = (index + 1) % len(branches)
        self.state["pending_branch"] = {"mode": branch["mode"], "head_candidate_id": branch["head_candidate_id"], "selected_candidate_id": branch["selected_candidate_id"]}
        self._save()
        return dict(self.state["pending_branch"])

    def record_round(
        self,
        *,
        round_id: str,
        branch_id: str,
        parent_model_id: str,
        child_model_id: Optional[str],
        status: str,
        costs: Mapping[str, float],
        failure_type: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized: dict[str, float] = {}
        for key in self.COST_FIELDS:
            value = float(costs.get(key, 0.0))
            if value < 0:
                raise ValueError("round costs must be non-negative")
            normalized[key] = value
        event = {
            "round_id": str(round_id),
            "branch_id": branch_id,
            "parent_model_id": parent_model_id,
            "child_model_id": child_model_id,
            "status": status,
            "failure_type": failure_type,
            "costs_seconds": normalized,
            "total_seconds": sum(normalized.values()),
        }
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                previous = json.loads(line)
                if previous.get("round_id") == event["round_id"]:
                    if previous != event:
                        raise ValueError("round cost event conflicts with previously recorded event")
                    return previous
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        branch = self.state["branches"].setdefault(branch_id, {"head_candidate_id": parent_model_id, "rounds": 0})
        branch["rounds"] = int(branch.get("rounds", 0)) + 1
        if status == "promoted" and child_model_id:
            branch["head_candidate_id"] = child_model_id
        self.state["spent_seconds"] = float(self.state["spent_seconds"]) + event["total_seconds"]
        self.state["pending_branch"] = None
        self._save()
        return event

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget_seconds - float(self.state["spent_seconds"]))

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("invalid branch campaign state")
        return value

    def _save(self) -> None:
        fd, name = tempfile.mkstemp(prefix="branch-campaign-", suffix=".json", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.state_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
