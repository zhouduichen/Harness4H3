from __future__ import annotations

import hashlib
import json

import pytest

from harness4h3.controller.directive import (
    DIRECTIVE_APPLY_AT,
    DIRECTIVE_KIND,
    MAX_INSTRUCTION_CHARS,
    HumanDirective,
    submit_directive,
)
from harness4h3.memory.observation import ObservationStore


def test_directive_normalizes_text_and_writes_a_bounded_observation(tmp_path):
    store = ObservationStore(tmp_path / "observations.jsonl")

    directive, written = submit_directive(store, "  prioritize peak memory next round  ")

    assert written is True
    assert directive.instruction == "prioritize peak memory next round"
    assert directive.apply_at == DIRECTIVE_APPLY_AT
    records = list(store.read())
    assert len(records) == 1
    assert records[0].kind == DIRECTIVE_KIND
    assert records[0].observation_id == "obs-%s" % directive.directive_id
    assert records[0].summary == {
        "directive_id": directive.directive_id,
        "instruction": directive.instruction,
        "apply_at": DIRECTIVE_APPLY_AT,
    }
    assert not any(key in records[0].summary for key in ("command", "cuda_visible_devices", "checkpoint"))


def test_default_identity_is_deterministic_and_retry_is_idempotent(tmp_path):
    store = ObservationStore(tmp_path / "observations.jsonl")

    first, first_written = submit_directive(store, "improve quality without violating memory")
    second, second_written = submit_directive(store, "improve quality without violating memory")

    assert first.directive_id == second.directive_id
    assert first.source_sha256 == second.source_sha256
    assert first_written is True
    assert second_written is False
    assert len(list(store.read())) == 1


def test_explicit_identity_conflict_is_rejected(tmp_path):
    store = ObservationStore(tmp_path / "observations.jsonl")
    submit_directive(store, "first objective", directive_id="night-001")

    with pytest.raises(ValueError, match="different source evidence"):
        submit_directive(store, "different objective", directive_id="night-001")


def test_directive_contract_rejects_blank_oversized_and_invalid_ids():
    with pytest.raises(ValueError, match="must not be blank"):
        HumanDirective.create("   ")
    with pytest.raises(ValueError, match="at most"):
        HumanDirective.create("x" * (MAX_INSTRUCTION_CHARS + 1))
    with pytest.raises(ValueError, match="directive_id"):
        HumanDirective.create("objective", directive_id="bad id")


def test_source_hash_is_canonical_and_does_not_include_creation_time():
    directive = HumanDirective.create("prefer lower peak memory", directive_id="stable")
    payload = json.dumps(
        {
            "apply_at": DIRECTIVE_APPLY_AT,
            "directive_id": "stable",
            "instruction": "prefer lower peak memory",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert directive.source_sha256 == hashlib.sha256(payload).hexdigest()
