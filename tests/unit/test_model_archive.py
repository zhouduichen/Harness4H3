from __future__ import annotations

import pytest

from harness4h3.archive.model_candidate import ModelCandidate
from harness4h3.archive.model_store import ModelCandidateExists, ModelStore
from harness4h3.archive.pareto import ParetoArchive, dominates
from harness4h3.controller.schemas import EvaluationResult, HardwareMetrics
from harness4h3.h3.state import ModelState


def candidate(model_id, parent=None, generation=0, metrics=None):
    state = ModelState.fake_baseline(model_id).derive(model_id, measured_metrics=metrics or {}) if parent else ModelState.fake_baseline(model_id)
    if parent:
        state = ModelState.from_dict({**state.to_dict(), "parent_model_id": parent})
    return ModelCandidate(model_id, parent, generation, state.checkpoint_path, state, "exp" if parent else None, "candidate")


def evaluation(quality, latency, memory, feasible=False):
    return EvaluationResult(
        quality,
        {"quality_score": quality},
        HardwareMetrics(latency_s=latency, peak_memory_gb=memory, model_size_gb=4.0),
        feasible,
    )


def test_model_store_is_immutable_and_supports_branching(tmp_path):
    store = ModelStore(tmp_path)
    root = candidate("M0000")
    store.initialize(root)
    left = candidate("M0001", "M0000", 1)
    right = candidate("M0002", "M0000", 1)
    store.create(left)
    store.create(right)
    with pytest.raises(ModelCandidateExists):
        store.create(left)
    assert store.children("M0000") == [left, right]
    assert store.next_id() == "M0003"


def test_pareto_archive_keeps_quality_and_latency_branches(tmp_path):
    high_quality = evaluation(0.90, 50, 8)
    low_latency = evaluation(0.85, 25, 8)
    assert not dominates(high_quality, low_latency)
    assert not dominates(low_latency, high_quality)
    archive = ParetoArchive(tmp_path)
    archive.update("M0001", high_quality)
    archive.update("M0002", low_latency)
    assert [entry.candidate_id for entry in archive.front()] == ["M0001", "M0002"]


def test_feasible_candidate_dominates_infeasible_candidate_before_soft_metrics():
    feasible = evaluation(0.84, 29, 5.9, feasible=True)
    infeasible = evaluation(0.90, 60, 12, feasible=False)
    assert dominates(feasible, infeasible)
