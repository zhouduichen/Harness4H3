from __future__ import annotations

import json

import pytest

from harness4h3.campaign.events import DecisionTrace, TraceIntegrityError
from tests.unit.campaign_fixtures import make_base


def test_event_trace_appends_fsyncable_events_and_reads_them(tmp_path):
    base = make_base()
    trace = DecisionTrace(tmp_path / "events.jsonl", base)
    event = trace.append(
        "campaign.created",
        round_id="R0001",
        experiment_id=None,
        candidate_id=None,
        parent_candidate_id=None,
        actor=base.controller_identity,
        payload={"ok": True},
        evidence_ids=(),
    )
    assert event.sequence == 1
    assert trace.read() == (event,)


def test_event_trace_rejects_wrong_base_and_non_monotonic_sequence(tmp_path):
    base = make_base()
    path = tmp_path / "events.jsonl"
    trace = DecisionTrace(path, base)
    trace.append(
        "campaign.created",
        round_id="R0001",
        experiment_id=None,
        candidate_id=None,
        parent_candidate_id=None,
        actor=base.controller_identity,
        payload={"ok": True},
        evidence_ids=(),
    )
    raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    raw["base_digest"] = "sha256:wrong"
    raw["sequence"] = 2
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="base"):
        trace.verify()


def test_event_trace_rejects_payload_tampering_unknown_event_and_duplicate_ids(tmp_path):
    base = make_base()
    path = tmp_path / "events.jsonl"
    trace = DecisionTrace(path, base)
    for event_type in ("campaign.created", "proposal.generated"):
        trace.append(
            event_type,
            round_id="R0001",
            experiment_id="exp_0001",
            candidate_id="C0001",
            parent_candidate_id="M0000",
            actor=base.controller_identity,
            payload={"event_type": event_type},
            evidence_ids=("evidence-1",),
        )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[1]["payload"]["tampered"] = True
    lines[1]["event_type"] = "unknown.event"
    lines[1]["event_id"] = lines[0]["event_id"]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    with pytest.raises(TraceIntegrityError):
        trace.verify()
