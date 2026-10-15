from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.evolution.screening import CandidateScreeningPipeline, ScreeningContext, ScreeningStage
from harness4h3.operators.wan import build_wan_compression_registry
from harness4h3.sandbox.runner import CompressionPlan


def _context(tmp_path, plan=None, full_evaluator=None):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    profile = {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 16},
        "layers": [{"block_id": "block.%d" % index} for index in range(4)],
    }
    algorithm = AlgorithmCandidate.from_source(
        "A0001",
        "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
    )
    return ScreeningContext(
        parent=parent,
        adapter=adapter,
        profile=profile,
        algorithm=algorithm,
        plan=plan or CompressionPlan("prune_blocks", {"block_ids": ["block.0"]}),
        registry=build_wan_compression_registry(),
        full_evaluator=full_evaluator,
    )


def test_screening_stops_invalid_plan_before_diagnostic_or_full(tmp_path):
    calls = []
    context = _context(tmp_path, plan=CompressionPlan("unknown_operator", {}), full_evaluator=lambda _: calls.append("full"))
    results = CandidateScreeningPipeline(tmp_path / "screening").run(context)
    assert results[-1].status == "rejected"
    assert results[-1].stage == ScreeningStage.STATIC.value
    assert calls == []


def test_fake_diagnostic_is_not_real_promotion_evidence(tmp_path):
    calls = []

    def full(context):
        calls.append("full")
        return {"valid": True, "offline_simulation": True, "search_score": 0.9}

    pipeline = CandidateScreeningPipeline(tmp_path / "screening")
    results = pipeline.run(_context(tmp_path, full_evaluator=full))
    assert [item.stage for item in results] == ["static", "diagnostic", "full"]
    assert all(item.evidence_level != "real_measured" for item in results[:2])
    assert results[-1].offline_simulation is True
    assert calls == ["full"]
    assert "promotion" not in json.dumps(results[-1].to_dict())


def test_screening_resume_does_not_repeat_full_evaluator(tmp_path):
    calls = []

    def full(context):
        calls.append("full")
        return {"valid": True, "offline_simulation": True, "search_score": 0.9}

    root = tmp_path / "screening"
    first = CandidateScreeningPipeline(root).run(_context(tmp_path, full_evaluator=full))
    second = CandidateScreeningPipeline(root).run(_context(tmp_path, full_evaluator=full))
    assert len(first) == len(second) == 3
    assert calls == ["full"]
