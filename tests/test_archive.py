from __future__ import annotations

import pytest

from harness4h3.archive.store import ArchiveError, Candidate, CandidateExists, CandidateStore, default_policy


def test_candidate_is_immutable_and_lineage_is_ordered(tmp_path):
    store = CandidateStore(tmp_path)
    h0 = store.initialize()
    h1 = Candidate("H1", "H0", 1, "prompt", {"path": "prompt.suffix"}, "reason", default_policy())
    store.create(h1)
    with pytest.raises(CandidateExists):
        store.create(h1)
    store.promote("H1")
    assert store.active_id == "H1"
    assert [candidate.id for candidate in store.lineage()] == ["H0", "H1"]


def test_candidate_id_cannot_escape_archive(tmp_path):
    store = CandidateStore(tmp_path)
    with pytest.raises(ArchiveError, match="invalid candidate id"):
        store.get("../../outside")
