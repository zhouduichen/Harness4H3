"""Append-only, base-bound decision trace."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .base import ActorIdentity, CampaignBase, CampaignBaseError, canonical_digest


EVENT_TYPES = frozenset(
    {
        "campaign.created",
        "proposal.generated",
        "proposal.validated",
        "critic.completed",
        "proposal.revised",
        "training.started",
        "training.metric",
        "training.completed",
        "evaluation.started",
        "evaluation.completed",
        "gate.decided",
        "archive.updated",
        "parent.selected",
        "campaign.replanned",
        "campaign.stopped",
    }
)


class TraceIntegrityError(ValueError):
    """Raised when an append-only trace cannot be trusted."""


def _optional_id(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TraceIntegrityError("%s must be null or a non-empty string" % name)
    return value.strip()


@dataclass(frozen=True)
class DecisionEvent:
    event_id: str
    sequence: int
    event_type: str
    campaign_id: str
    round_id: Optional[str]
    experiment_id: Optional[str]
    candidate_id: Optional[str]
    parent_candidate_id: Optional[str]
    actor: ActorIdentity
    base_digest: str
    payload: Mapping[str, Any]
    payload_digest: str
    evidence_ids: Tuple[str, ...]
    created_at: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence <= 0:
            raise TraceIntegrityError("event sequence must be positive")
        if not self.event_id or not self.event_type or self.event_type not in EVENT_TYPES:
            raise TraceIntegrityError("unknown or empty event type")
        if not self.campaign_id or not self.base_digest or not self.payload_digest:
            raise TraceIntegrityError("event identity fields must be non-empty")
        if not isinstance(self.actor, ActorIdentity):
            raise TraceIntegrityError("event actor must be an ActorIdentity")
        if not isinstance(self.payload, Mapping):
            raise TraceIntegrityError("event payload must be a mapping")
        try:
            expected = canonical_digest(self.payload)
        except CampaignBaseError as exc:
            raise TraceIntegrityError("event payload is not canonical JSON") from exc
        if expected != self.payload_digest:
            raise TraceIntegrityError("event payload digest mismatch")
        if any(not isinstance(item, str) or not item for item in self.evidence_ids):
            raise TraceIntegrityError("evidence IDs must be non-empty strings")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise TraceIntegrityError("event created_at must be ISO-8601") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "campaign_id": self.campaign_id,
            "round_id": self.round_id,
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "actor": self.actor.to_dict(),
            "base_digest": self.base_digest,
            "payload": copy.deepcopy(dict(self.payload)),
            "payload_digest": self.payload_digest,
            "evidence_ids": list(self.evidence_ids),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DecisionEvent":
        if not isinstance(raw, Mapping):
            raise TraceIntegrityError("event must be a mapping")
        required = {
            "event_id", "sequence", "event_type", "campaign_id", "round_id", "experiment_id",
            "candidate_id", "parent_candidate_id", "actor", "base_digest", "payload",
            "payload_digest", "evidence_ids", "created_at",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise TraceIntegrityError("event is missing field(s): %s" % ", ".join(missing))
        evidence = raw["evidence_ids"]
        if not isinstance(evidence, (list, tuple)):
            raise TraceIntegrityError("event evidence_ids must be an array")
        return cls(
            event_id=str(raw["event_id"]),
            sequence=int(raw["sequence"]),
            event_type=str(raw["event_type"]),
            campaign_id=str(raw["campaign_id"]),
            round_id=_optional_id(raw["round_id"], "round_id"),
            experiment_id=_optional_id(raw["experiment_id"], "experiment_id"),
            candidate_id=_optional_id(raw["candidate_id"], "candidate_id"),
            parent_candidate_id=_optional_id(raw["parent_candidate_id"], "parent_candidate_id"),
            actor=ActorIdentity.from_dict(raw["actor"]),
            base_digest=str(raw["base_digest"]),
            payload=copy.deepcopy(dict(raw["payload"])),
            payload_digest=str(raw["payload_digest"]),
            evidence_ids=tuple(str(item) for item in evidence),
            created_at=str(raw["created_at"]),
        )


class DecisionTrace:
    def __init__(self, path: Path, base: CampaignBase):
        self.path = Path(path)
        self.base = base

    def _read_raw(self) -> tuple[DecisionEvent, ...]:
        if not self.path.exists():
            return ()
        events = []
        seen_ids = set()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise TraceIntegrityError("unable to read decision trace: %s" % exc) from exc
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise TraceIntegrityError("blank event line %d" % line_number)
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceIntegrityError("invalid JSON on event line %d" % line_number) from exc
            try:
                event = DecisionEvent.from_dict(raw)
            except (TypeError, ValueError, KeyError) as exc:
                raise TraceIntegrityError("invalid event line %d: %s" % (line_number, exc)) from exc
            if event.event_id in seen_ids:
                raise TraceIntegrityError("duplicate event id: %s" % event.event_id)
            seen_ids.add(event.event_id)
            events.append(event)
        return tuple(events)

    def verify(self) -> None:
        expected_sequence = 1
        for event in self._read_raw():
            if event.campaign_id != self.base.campaign_id:
                raise TraceIntegrityError("event campaign_id does not match campaign base")
            try:
                self.base.assert_event_base(event.base_digest)
            except CampaignBaseError as exc:
                raise TraceIntegrityError(str(exc)) from exc
            if event.sequence != expected_sequence:
                raise TraceIntegrityError(
                    "event sequence gap: expected %d got %d" % (expected_sequence, event.sequence)
                )
            expected_sequence += 1

    def read(self) -> tuple[DecisionEvent, ...]:
        self.verify()
        return self._read_raw()

    def append(
        self,
        event_type: str,
        *,
        round_id: Optional[str],
        experiment_id: Optional[str],
        candidate_id: Optional[str],
        parent_candidate_id: Optional[str],
        actor: ActorIdentity,
        payload: Mapping[str, Any],
        evidence_ids: Sequence[str],
    ) -> DecisionEvent:
        existing = self.read()
        sequence = len(existing) + 1
        event = DecisionEvent(
            event_id="evt_%06d" % sequence,
            sequence=sequence,
            event_type=event_type,
            campaign_id=self.base.campaign_id,
            round_id=round_id,
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            parent_candidate_id=parent_candidate_id,
            actor=actor,
            base_digest=self.base.digest,
            payload=copy.deepcopy(dict(payload)),
            payload_digest=canonical_digest(payload),
            evidence_ids=tuple(evidence_ids),
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise TraceIntegrityError("unable to append decision trace: %s" % exc) from exc
        return event


__all__ = ["DecisionEvent", "DecisionTrace", "EVENT_TYPES", "TraceIntegrityError"]
