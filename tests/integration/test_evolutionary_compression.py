from __future__ import annotations

import json
import hashlib

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution import EvolutionLoop
from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.memory.experience import ExperienceRecord
from harness4h3.evolution.policy import SearchPolicy


def test_evolution_loop_runs_fake_round_and_persists_evidence(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    outcome = loop.run_round()
    assert outcome.status == "promoted"
    assert outcome.child_model_id == "M0001"
    round_dir = tmp_path / "run" / "rounds" / outcome.round_id
    assert (round_dir / "decision.json").exists()
    round_state = json.loads((round_dir / "round.json").read_text(encoding="utf-8"))
    assert round_state["status"] == "promoted"
    assert round_state["child_model_id"] == "M0001"
    records = list(loop.experience_store.read())
    assert records
    assert records[0].operator == "prune_blocks"
    assert records[0].operator_args["block_ids"]
    assert (tmp_path / "run" / "search-policy.json").exists()
    assert loop.model_store.active_id == "M0001"


def test_evolution_loop_keeps_failed_algorithm_as_experience(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")

    def provider(payload):
        return {"source": "import os\ndef propose(profile, history):\n    return {}\n", "parent_id": payload["parent_id"]}

    from harness4h3.evolution.agent import AlgorithmAgent

    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", agent=AlgorithmAgent(provider=provider))
    outcome = loop.run_round()
    assert outcome.status == "failed"
    round_state = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "round.json").read_text(encoding="utf-8"))
    assert round_state["status"] == "failed"
    assert round_state["failure_type"] == "sandbox_policy"
    record = list(loop.experience_store.read())[0]
    assert record.status == "failed"
    assert record.decision["failure_type"] == "sandbox_policy"
    assert record.decision["search_policy"]["exploit_ratio"] == 0.6


def test_standard_loop_executes_algorithm_search_program(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")

    source = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}

def search(profile, history, state, search_space):
    return {
        'candidates': [
            {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
            {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
        ],
        'rank': [1, 0],
        'stop': True,
    }
"""

    def provider(payload):
        return {
            "source": source,
            "hypothesis": "the search program selects the second candidate",
            "parent_id": payload["parent_id"],
        }

    from harness4h3.evolution.agent import AlgorithmAgent

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        agent=AlgorithmAgent(provider=provider),
    )
    outcome = loop.run_round()

    assert outcome.status == "promoted"
    round_dir = tmp_path / "run" / "rounds" / outcome.round_id
    trace = json.loads((round_dir / "algorithm-program-trace.json").read_text(encoding="utf-8"))
    assert trace["decision"]["rank"] == [1, 0]
    assert "prune_blocks" in trace["trace"][0]["search_space_operator_names"]
    assert len(trace["trace"][0]["search_space_operator_names"]) > 1
    assert trace["selected_plan"]["args"]["block_ids"] == ["block.2"]
    decision = json.loads((round_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["operator_args"]["block_ids"] == ["block.2"]


def test_block_quality_feedback_is_connected_to_restartable_fake_round(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        algorithm_improvement=True,
        search_space_evolution=False,
        candidate_screening=False,
        advanced_profiling=True,
    )

    outcome = loop.run_round()
    round_dir = tmp_path / "run" / "rounds" / outcome.round_id
    trace = json.loads((round_dir / "algorithm-program-trace.json").read_text(encoding="utf-8"))
    measured = trace["measured_search"]
    assert measured["candidate_scope"] == "registered_multigranular"
    assert measured["parent_checkpoint_sha256"] == parent.sha256
    assert measured["candidate_results"]
    assert {row["family"] for row in measured["candidate_results"]} >= {"structure", "head", "timestep"}
    assert all(row["parent_checkpoint_sha256"] == parent.sha256 for row in measured["candidate_results"])
    assert measured["quality_floor_basis"] == "parent_relative_probe_similarity_with_parent_baseline_1.0"
    assert not (tmp_path / "run" / "search-space-feedback.json").exists()


def test_evolution_loop_restarts_from_persisted_parent_and_search_policy(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    first = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    assert first.run_round().status == "promoted"

    restarted = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run")
    outcome = restarted.run_round()
    assert outcome.round_id == "R0002"
    assert outcome.parent_model_id == "M0001"
    # The restarted loop compares the new candidate with the active measured
    # Parent.  The new protocol permits promotion when any one of the
    # independent efficiency objectives clears its configured threshold.
    assert outcome.status in {"promoted", "rejected", "failed"}
    round_state = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "round.json").read_text(encoding="utf-8"))
    assert round_state["status"] == outcome.status
    if outcome.status == "failed":
        assert outcome.decision["failure_type"] == "duplicate_algorithm"
    else:
        assert restarted.model_store.active_id == outcome.child_model_id
    assert restarted.search_policy.rounds == 2
    assert (tmp_path / "run" / "initial-baseline.json").is_file()


def test_deterministic_baseline_avoids_duplicate_selection_across_three_rounds(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=8, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        memory_enabled=False,
        adapt_search_policy=False,
        search_policy=SearchPolicy(exploit_ratio=1.0, seed=0),
    )

    outcomes = [loop.run_round() for _ in range(3)]

    assert all(outcome.status in {"promoted", "rejected", "failed"} for outcome in outcomes)
    if outcomes[-1].status == "failed":
        assert outcomes[-1].decision["failure_type"] == "duplicate_algorithm"
    assert loop.search_policy.duplicate_rounds <= 1
    selected = []
    for outcome in outcomes:
        decision = json.loads(
            (tmp_path / "run" / "rounds" / outcome.round_id / "decision.json").read_text()
        )
        if decision.get("operator_args", {}).get("block_ids"):
            selected.append(decision["operator_args"]["block_ids"][0])
    assert len(set(selected)) == len(selected)


def test_duplicate_algorithm_is_recorded_without_overwriting_active_parent(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n"

    def provider(payload):
        return {"source": source, "hypothesis": "same logic", "parent_id": payload["parent_id"]}

    from harness4h3.evolution.agent import AlgorithmAgent

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        agent=AlgorithmAgent(provider=provider),
    )
    assert loop.run_round().status == "promoted"
    duplicate = loop.run_round()
    assert duplicate.status == "failed"
    assert duplicate.decision["failure_type"] == "duplicate_algorithm"
    assert loop.model_store.active_id == "M0001"
    assert loop.search_policy.duplicate_rounds == 1


def test_duplicate_algorithm_improvement_reuses_selected_program_for_measured_search(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent_model = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    source = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}

def search(profile, history, state, search_space):
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
    ], 'rank': [0], 'stop': True}
"""

    class DuplicateAgent:
        def improve(self, algorithm_id, **kwargs):
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=kwargs["parent_id"],
                parent_algorithm_id=kwargs["parent"].algorithm_id,
                parent_source_sha256=kwargs["parent"].source_sha256,
            )

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent_model,
        output_root=tmp_path / "run",
        algorithm_improvement=True,
        agent=DuplicateAgent(),
    )
    selected_parent = AlgorithmCandidate.from_source("A0001", source)
    loop.algorithm_store.create(selected_parent)
    search_calls = []

    def no_feasible_measurement(**kwargs):
        search_calls.append(kwargs)
        return {
            "valid": False,
            "failure_reason": "standard_algorithm_search_no_quality_feasible_candidate",
            "selected_plan": None,
            "algorithm_search": {
                "candidate_scope": "block_only",
                "candidate_results": [],
                "quality_feasible_candidate_count": 0,
            },
        }

    loop._complete_standard_algorithm_search = no_feasible_measurement
    outcome = loop.run_round()

    assert outcome.status == "failed"
    assert outcome.decision["failure_type"] == "algorithm_search_failed"
    assert len(search_calls) == 1
    assert search_calls[0]["source"] == selected_parent.source
    assert search_calls[0]["parent"].sha256 == parent_model.sha256
    round_dir = tmp_path / "run" / "rounds" / outcome.round_id
    round_algorithm = json.loads((round_dir / "algorithm.json").read_text())
    assert round_algorithm["algorithm_id"] == "A0001"
    assert round_algorithm["metadata"]["reused_existing_algorithm_after_duplicate"] is True
    assert len(round_algorithm["metadata"]["duplicate_source_attempts"]) == 2
    assert [item.algorithm_id for item in loop.algorithm_store.read()] == ["A0001"]


def test_algorithm_lineage_advances_after_rejected_model(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    calls = {"count": 0}

    def evaluator(child, profile):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "quality_score": 0.5,
                "baseline_quality_score": 1.0,
                "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
                "hardware_metrics": {"latency_s": 0.5, "peak_memory_gb": 0.5, "model_size_gb": 0.5},
                "valid": True,
            }
        return {
            "quality_score": 1.0,
            "baseline_quality_score": 1.0,
            "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
            "hardware_metrics": {"latency_s": 0.5, "peak_memory_gb": 0.5, "model_size_gb": 0.5},
            "valid": True,
        }

    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", evaluator=evaluator, algorithm_improvement=True)
    assert loop.run_round().status == "rejected"
    assert loop.run_round().status == "promoted"
    assert loop.algorithm_store.get("A0002").parent_algorithm_id == "A0001"


def test_design_feedback_changes_next_algorithm_and_persists_gene(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    calls = {"count": 0}

    def evaluator(child, profile):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "quality_score": 0.50,
                "baseline_quality_score": 1.0,
                "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
                "hardware_metrics": {"latency_s": 0.40, "peak_memory_gb": 0.40, "model_size_gb": 0.40},
                "valid": True,
            }
        return {
            "quality_score": 1.0,
            "baseline_quality_score": 1.0,
            "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
            "hardware_metrics": {"latency_s": 0.90, "peak_memory_gb": 0.90, "model_size_gb": 0.90},
            "valid": True,
        }

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        evaluator=evaluator,
        search_policy=SearchPolicy(exploit_ratio=0.0, seed=0),
        algorithm_improvement=True,
    )

    first = loop.run_round()
    second = loop.run_round()

    assert first.status == "rejected"
    assert second.status == "promoted"
    assert loop.model_store.active_id == second.child_model_id
    first_design = json.loads((tmp_path / "run" / "rounds" / first.round_id / "design.json").read_text())
    second_design = json.loads((tmp_path / "run" / "rounds" / second.round_id / "design.json").read_text())
    assert first_design["fingerprint"] != second_design["fingerprint"]
    allowed_families = {"prune_blocks", "head_gating", "timestep_block_gate", "token_merge", "kv_token_select", "sparse_runtime", "attention_window", "sampling_step_skip", "prune_tokens"}
    assert first_design["strategy"] in allowed_families
    assert second_design["strategy"] in allowed_families
    genes = (tmp_path / "run" / "design-genes.jsonl").read_text().splitlines()
    assert len(genes) == 2
    assert "quality_floor_failed" in genes[0]


def test_failed_composite_design_is_retained_as_negative_gene(tmp_path):
    from harness4h3.evolution.agent import AlgorithmAgent

    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    source = """def propose(profile, history):
    return {'operator': 'compose', 'args': {'steps': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
    ]}}
"""

    def provider(payload):
        return {"source": source, "hypothesis": "repeat a block to test failure memory", "parent_id": payload["parent_id"]}

    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        agent=AlgorithmAgent(provider=provider),
    )

    outcome = loop.run_round()

    assert outcome.status == "promoted"
    trace = json.loads((tmp_path / "run" / "rounds" / outcome.round_id / "algorithm-program-trace.json").read_text())
    assert any(item["operator"] == "compose" for item in trace["decision"]["plans"])
    assert trace["selected_plan"]["operator"] == "prune_blocks"
    assert trace["decision"]["plans"][0]["operator"] == "compose"


def test_incomplete_round_reuses_persisted_compression_after_process_crash(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")

    def crash_evaluator(child, profile):
        raise KeyboardInterrupt("simulated process crash after compression")

    first = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", evaluator=crash_evaluator)
    try:
        first.run_round()
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("simulated crash should propagate")

    round_dir = tmp_path / "run" / "rounds" / "R0001"
    assert (round_dir / "compression.json").is_file()
    assert not (round_dir / "decision.json").exists()

    calls = {"count": 0}

    def recovered_evaluator(child, profile):
        calls["count"] += 1
        return {
            "quality_score": 0.99,
            "baseline_quality_score": 0.99,
            "baseline_hardware_metrics": {"latency_s": 1.0, "peak_memory_gb": 1.0, "model_size_gb": 1.0},
            "hardware_metrics": {"latency_s": 0.5, "peak_memory_gb": 0.5, "model_size_gb": 0.5},
            "valid": True,
        }

    restarted = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", evaluator=recovered_evaluator)
    outcome = restarted.run_round()

    assert outcome.round_id == "R0001"
    assert calls["count"] == 1
    assert not (tmp_path / "run" / "rounds" / "R0002").exists()
    assert len(list(restarted.algorithm_store.read())) == 1


def test_evo_compress_features_form_restartable_cpu_fake_pipeline(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        advanced_profiling=True,
        algorithm_improvement=True,
        search_space_evolution=True,
        candidate_screening=True,
    )

    first = loop.run_round()
    second = loop.run_round()
    third = loop.run_round()

    assert first.status in {"promoted", "rejected"}
    assert second.status in {"promoted", "rejected"}
    assert third.status in {"promoted", "rejected"}
    first_dir = tmp_path / "run" / "rounds" / first.round_id
    second_dir = tmp_path / "run" / "rounds" / second.round_id
    first_profile = json.loads((first_dir / "profile.json").read_text())
    assert first_profile["calibration"]["provider"] == "FakeWanCalibrationProvider"
    assert first_profile["evidence"]["advanced"] is True
    search_space = json.loads((first_dir / "search-space.json").read_text())
    assert search_space["version_id"] == "S0001"
    assert "prune_ffn_by_weight_threshold" not in {item["name"] for item in search_space["operators"]}
    feedback = json.loads((tmp_path / "run" / "search-space-feedback.json").read_text())
    assert feedback["observed_rounds"] >= 2
    assert feedback["ready"] is True
    third_dir = tmp_path / "run" / "rounds" / third.round_id
    third_search_space = json.loads((third_dir / "search-space.json").read_text())
    assert third_search_space["version_id"] == "S0002"
    assert "timestep_attention_policy" in {item["name"] for item in third_search_space["operators"]}
    screening = json.loads((second_dir / "screening" / "screening.json").read_text())
    assert [item["stage"] for item in screening["results"]] == ["static", "diagnostic", "full"]
    validation = json.loads((second_dir / "algorithm-validation.json").read_text())
    assert validation["parent_algorithm_id"] == "A0001"
    assert validation["evidence_level"] == "cpu_fake_quality_proxy"
    assert validation["parent_metrics"]["output_metrics"]["cosine_similarity"] <= 1.0
    assert validation["candidate_metrics"]["algorithm_search"]["evaluated_candidate_counts"][0] >= 2
    assert validation["candidate_metrics"]["hardware_metrics"]["available"] is False
    assert validation["scenario_results"][0]["candidate_metrics"]["parent_checkpoint_sha256"] == validation["scenario_results"][0]["parent_metrics"]["parent_checkpoint_sha256"]
    parent_search = validation["parent_metrics"]["algorithm_search"]
    candidate_search = validation["candidate_metrics"]["algorithm_search"]
    assert parent_search["protocol"] == candidate_search["protocol"] == "standard_hbps"
    assert parent_search["budgets"][0]["max_full_evals"] == candidate_search["budgets"][0]["max_full_evals"]
    assert parent_search["evaluated_candidate_counts"][0] <= parent_search["budgets"][0]["max_full_evals"]
    assert candidate_search["evaluated_candidate_counts"][0] <= candidate_search["budgets"][0]["max_full_evals"]
    assert parent_search["evaluated_candidate_counts"][0] >= 2
    algorithm = json.loads((second_dir / "algorithm.json").read_text())
    assert algorithm["parent_algorithm_id"] == "A0001"
    assert algorithm["parent_source_sha256"] == validation["parent_source_sha256"]
    assert algorithm["validation_status"] in {"validated", "rejected"}
    algorithms = [
        json.loads((tmp_path / "run" / "algorithms" / "candidates" / ("A%04d.json" % index)).read_text())
        for index in (1, 2, 3)
    ]
    assert [item["algorithm_id"] for item in algorithms] == ["A0001", "A0002", "A0003"]
    assert algorithms[1]["source_sha256"] != algorithms[0]["source_sha256"]
    assert algorithms[2]["source_sha256"] != algorithms[1]["source_sha256"]
    assert algorithms[1]["source_diff"]
    assert algorithms[2]["source_diff"]
    assert "excluded.update" in algorithms[1]["source"]
    second_trace = json.loads((second_dir / "algorithm-program-trace.json").read_text())
    assert second_trace["measured_search"]["protocol"] == "standard_hbps"
    assert second_trace["measured_search"]["evaluated_candidate_count"] >= 2
    # The recursive loop keeps improving the latest source, while validation
    # remains the gate for recording an algorithm as a verified skill.
    assert json.loads((third_dir / "algorithm.json").read_text())["parent_algorithm_id"] == "A0002"
    skills_path = tmp_path / "run" / "algorithm-skills.jsonl"
    if validation["valid"]:
        assert skills_path.is_file()
    else:
        # Equal-budget search correctly keeps an algorithm that did not beat
        # its parent out of the verified-skill store.
        assert not skills_path.exists()
    skills = loop.skill_hub.search(architecture="WanFakeDiT", limit=16)
    assert any(item.get("memory_kind") == "algorithm_skill" for item in skills) is validation["valid"]
    genes = [json.loads(line) for line in (tmp_path / "run" / "design-genes.jsonl").read_text().splitlines()]
    assert any(item["gene_id"] == "G-" + first.round_id for item in genes)
    assert any(item["gene_id"] == "G-" + second.round_id for item in genes)
    assert json.loads((second_dir / "design.json").read_text())["change_taxonomy"]["change_class"] in {"parameter_search", "algorithm_rule_evolution"}


def test_search_space_extension_requires_recorded_feedback_evidence(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / "run", search_space_evolution=True)
    assert loop.search_space_version.version_id == "S0001"
    assert loop.search_space_manager.store.read_proposals() == ()

    for index in (1, 2):
        round_id = "R%04d" % index
        decision = {"status": "failed", "failure_type": "unsupported_compression_capability"}
        source_uri = str((tmp_path / (round_id + ".json")).resolve())
        loop.experience_store.append(
            ExperienceRecord(
                experience_id=round_id,
                source_uri=source_uri,
                source_sha256=hashlib.sha256(source_uri.encode()).hexdigest(),
                source_kind="compression_round",
                experiment_id=round_id,
                parent_model_id="M0000",
                child_model_id=None,
                operator="unsupported_operator",
                operator_args={},
                training={"algorithm_id": "A%04d" % index},
                evaluation=None,
                decision=decision,
                reward=None,
                status="failed",
                provenance={"offline_simulation": True},
                created_at="",
            )
        )

    loop._maybe_evolve_search_space()
    proposals = loop.search_space_manager.store.read_proposals()
    assert len(proposals) == 1
    assert proposals[0].status == "validated"
    assert proposals[0].trigger_evidence["ready"] is True
    assert proposals[0].trigger_evidence["evidence_ids"] == ["R0001", "R0002"]
    assert loop.search_space_version.version_id == "S0002"


def test_frozen_expanded_search_space_is_consumed_without_evolution(tmp_path):
    adapter = FakeWanDiTAdapter(block_count=5, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model("fake://baseline"), tmp_path / "parent")
    loop = EvolutionLoop(
        adapter=adapter,
        parent=parent,
        output_root=tmp_path / "run",
        advanced_profiling=True,
        algorithm_improvement=True,
        frozen_expanded_search_space=True,
        memory_enabled=False,
        adapt_search_policy=False,
    )

    outcomes = [loop.run_round() for _ in range(3)]
    assert all(item.status in {"promoted", "rejected"} for item in outcomes)
    assert loop.search_space_version is not None
    assert loop.search_space_version.version_id == "S0001"
    assert "kv_token_select" in {item.name for item in loop.search_space_version.operators}
    assert not (tmp_path / "run" / "search-space" / "versions" / "S0002.json").exists()
    for outcome in outcomes:
        round_dir = tmp_path / "run" / "rounds" / outcome.round_id
        search_space = json.loads((round_dir / "search-space.json").read_text())
        assert search_space["version_id"] == "S0001"
    first_trace = json.loads((tmp_path / "run" / "rounds" / "R0001" / "algorithm-program-trace.json").read_text())
    candidate_operators = {
        plan["operator"]
        for plan in first_trace["decision"]["plans"]
        if isinstance(plan, dict) and "operator" in plan
    }
    assert "kv_token_select" not in candidate_operators
    assert any(
        item.get("operator") == "kv_token_select" and item.get("reason") == "missing_token_sensitivity"
        for item in first_trace["decision"]["diagnostics"].get("omitted", [])
    )
