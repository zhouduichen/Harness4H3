from __future__ import annotations

from harness4h3.evolution.tica_calibration import TICACalibrationManifest
from harness4h3.evolution.tica_prune import TICAPruner, TICARecipe
from harness4h3.evolution.agent import AlgorithmAgent, _compact_profile, _normalize_block_proposal_source, _revise_search_program
from harness4h3.evolution.algorithm_evaluation import AlgorithmModelEvaluator
from harness4h3.evolution.tica_prune import _run_algorithm_search_program
from harness4h3.operators.wan.search_space import default_search_space
from harness4h3.sandbox import AlgorithmSandbox


def _profile():
    return {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 8},
        "layers": [{"block_id": "block.%d" % index} for index in range(4)],
        "evaluation_split": "calibration",
    }


def _manifest():
    return TICACalibrationManifest.from_fake(hidden_size=8, sample_count=1)


def _measure(plan, full, stage):
    args = plan.get("args", plan)
    count = len(args.get("block_ids", []))
    error = 0.04 + 0.01 * count
    return {
        "valid": True,
        "quality_error": error,
        "risk_metric": "normalized_mse" if full else "local_relative_output_error",
        "risk_value": error,
        "risk_scale": "normalized_mse" if full else "profile_relative_error",
        "risk_is_approximate": not full,
        "quality_score": 1.0 - error,
        "parameter_reduction_ratio": 0.10 + 0.05 * count,
    }


def _run(source, root):
    return TICAPruner().search(
        None,
        None,
        _manifest(),
        _profile(),
        TICARecipe(
            baseline_name="tica_block",
            include_baseline_suite=False,
            max_search_depth=1,
            max_fast_evals=8,
            max_full_evals=4,
        ),
        evaluate_candidate=_measure,
        algorithm_source=source,
        algorithm_history=[{"experience_id": "R0001", "failure_reason": "quality_regression"}],
        search_space={
            "version_id": "S0001",
            "operators": [
                {"name": "prune_blocks", "argument_schema": {"block_ids": "list[str]"}},
                {"name": "prune_ffn_channels", "argument_schema": {"block_ids": "list[str]", "retained_indices": "list[int]"}},
            ],
        },
        sandbox=AlgorithmSandbox(root),
    )


def test_two_algorithm_programs_change_candidate_membership_and_order(tmp_path):
    source_a = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}
def search(profile, history, state, search_space):
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
    ], 'rank': [0, 1], 'stop': True}
"""
    source_b = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}
def search(profile, history, state, search_space):
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
    ], 'rank': [1, 0], 'stop': True}
"""
    first = _run(source_a, tmp_path / "a")
    second = _run(source_b, tmp_path / "b")
    assert first.algorithm_trace[0]["source_sha256"] != second.algorithm_trace[0]["source_sha256"]
    assert first.algorithm_trace[0]["plan_digests"] != second.algorithm_trace[0]["plan_digests"]
    assert first.algorithm_trace[0]["rank"] == [0, 1]
    assert second.algorithm_trace[0]["rank"] == [1, 0]
    assert first.candidates[0].plan["args"]["block_ids"] == ["block.0"]
    assert second.candidates[0].plan["args"]["block_ids"] == ["block.2"]


def test_search_program_receives_current_search_space_and_history(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}
def search(profile, history, state, search_space):
    assert history[0]['failure_reason'] == 'quality_regression'
    assert search_space['version_id'] == 'S0001'
    assert len(search_space['operators']) == 2
    return {'candidates': [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.3']}},
    ], 'rank': [0], 'stop': True}
"""
    result = _run(source, tmp_path / "space")
    assert result.algorithm_trace[0]["search_space_operator_names"] == ["prune_blocks", "prune_ffn_channels"]
    assert result.candidates[0].plan["args"]["block_ids"] == ["block.3"]


def test_generated_program_covers_registered_families_within_budget(tmp_path):
    profile = {
        "model": {"architecture": "WanFakeDiT", "block_count": 4, "ffn_width": 4},
        "layers": [{"block_id": "block.%d" % index, "latency_ms": 1.0 + index} for index in range(4)],
        "block_sensitivity": [
            {"block_id": "block.%d" % index, "importance_score": 0.1 + index * 0.1}
            for index in range(4)
        ],
        "channel_sensitivity": [
            {
                "block_id": "block.%d" % block,
                "channels": [
                    {"channel": channel, "importance_score": float(channel + 1)}
                    for channel in range(4)
                ],
            }
            for block in range(4)
        ],
        "head_sensitivity": [
            {"block_id": "block.%d" % block, "attention": "attn1", "head": head, "importance_score": float(head + 1)}
            for block in range(2)
            for head in range(4)
        ],
        "token_sensitivity": [{
            "keep_ratio": 0.75,
            "token_count": 8,
            "grid_shape": [1, 2, 4],
            "importance_score": 0.8,
            "available": True,
        }],
        "sampling_step_sensitivity": {
            "candidates": [
                {"target_steps": 12, "available": True},
                {"target_steps": 8, "available": True},
            ]
        },
        "provenance": {"offline_simulation": True},
    }
    algorithm = AlgorithmAgent().generate("A0001", parent_id="M0000", profile=profile, history=[], search_choice="explore")
    search_space = default_search_space().to_dict()
    decision = _run_algorithm_search_program(
        algorithm.source,
        AlgorithmSandbox(tmp_path / "sandbox"),
        profile,
        [],
        {"depth": 1, "proposed_plan": None, "budget": {"max_candidates": 5}},
        search_space,
    )

    assert len(decision.plans) <= 5
    operators = {str(plan["operator"]) for plan in decision.plans}
    assert {"prune_blocks", "head_gating", "prune_tokens"}.issubset(operators)
    assert all(
        plan["operator"] != "prune_blocks"
        or (plan["args"]["block_ids"] and len(plan["args"]["block_ids"]) < 4)
        for plan in decision.plans
    )


def test_program_rank_is_profile_derived_instead_of_algorithm_parity(tmp_path):
    profile = {
        "model": {"architecture": "WanFakeDiT", "block_count": 2, "ffn_width": 4},
        "layers": [{"block_id": "block.0"}, {"block_id": "block.1"}],
        "block_sensitivity": [
            {"block_id": "block.0", "importance_score": 0.1},
            {"block_id": "block.1", "importance_score": 0.3},
        ],
        "head_sensitivity": [
            {"block_id": "block.0", "attention": "attn1", "head": 0, "importance_score": 0.1},
            {"block_id": "block.0", "attention": "attn1", "head": 1, "importance_score": 0.9},
        ],
        "token_sensitivity": [{"keep_ratio": 0.75, "token_count": 8, "importance_score": 0.8, "available": True}],
        "provenance": {"offline_simulation": True},
    }
    search_space = default_search_space().to_dict()
    rank_values = []
    for algorithm_id in ("A0001", "A0002"):
        algorithm = AlgorithmAgent().generate(algorithm_id, parent_id="M0000", profile=profile, history=[])
        result = _run_algorithm_search_program(
            algorithm.source,
            AlgorithmSandbox(tmp_path / algorithm_id),
            profile,
            [],
            {"depth": 1, "proposed_plan": None, "budget": {"max_candidates": 16}},
            search_space,
        )
        rank_values.append([(result.plans[index]["operator"], result.plans[index]["args"]) for index in result.rank])

    assert rank_values[0] == rank_values[1]


def test_search_proposes_external_controls_and_ordered_supported_composition(tmp_path):
    profile = {
        "model": {"architecture": "WanDiffusersDiT", "block_count": 2, "ffn_width": 4},
        "layers": [{"block_id": "block.0"}, {"block_id": "block.1"}],
        "block_sensitivity": [{"block_id": "block.0", "importance_score": 0.1}],
        "execution": {
            "cases": [{
                "variant": "dense",
                "generation_wall_time_s": 90.0,
                "recipe": {"num_inference_steps": 16},
                "diagnostic_pass": {
                    "category_cuda_time_ms_total": {"self_attention": 60.0, "ffn": 30.0},
                    "module_timings": [],
                },
            }]
        },
    }
    algorithm = AlgorithmAgent().generate("A0001", parent_id="M0000", profile=profile, history=[])
    decision = _run_algorithm_search_program(
        algorithm.source,
        AlgorithmSandbox(tmp_path / "external-search"),
        profile,
        [],
        {"depth": 1, "proposed_plan": None, "budget": {"max_candidates": 16}},
        default_search_space().to_dict(),
    )

    plans = decision.plans
    assert any(plan["operator"] == "svoo_attention" for plan in plans)
    assert any(plan["operator"] == "magcache" for plan in plans)
    combined = next(plan for plan in plans if plan["operator"] == "compose" and [step["operator"] for step in plan["args"]["steps"]] == ["svoo_attention", "magcache"])
    assert combined["args"]["steps"][0]["args"]["method"] == "svoo"
    assert decision.diagnostics["family_allocated_counts"]
    assert all("allocation_reason" in item and "evidence_refs" in item for item in decision.diagnostics["candidate_evidence"])


def test_standard_algorithm_improvement_retains_registered_granularity(tmp_path):
    profile = {
        "model": {"architecture": "WanFakeDiT", "block_count": 3, "ffn_width": 8},
        "layers": [{"block_id": "block.%d" % index, "latency_ms": 1.0} for index in range(3)],
        "block_sensitivity": [
            {"block_id": "block.%d" % index, "importance_score": 0.1 + index * 0.1}
            for index in range(3)
        ],
        "head_sensitivity": [
            {"block_id": "block.0", "attention": "attn1", "head": 0, "importance_score": 0.1},
            {"block_id": "block.0", "attention": "attn1", "head": 1, "importance_score": 0.5},
        ],
        "token_sensitivity": [{"token_count": 16, "keep_ratio": 0.75, "available": True}],
    }
    agent = AlgorithmAgent()
    parent = agent.generate("A0001", parent_id="M0000", profile=profile, history=[])
    child = agent.improve(
        "A0002",
        parent=parent,
        parent_id="M0000",
        profile=profile,
        history=[{"decision": {"failure_type": "quality_floor_failed"}}],
        failures=["quality_floor_failed"],
        search_space=default_search_space().to_dict(),
    )
    decision = _run_algorithm_search_program(
        child.source,
        AlgorithmSandbox(tmp_path / "improved-sandbox"),
        profile,
        [],
        {"depth": 1, "proposed_plan": None, "budget": {"max_candidates": 4}},
        default_search_space().to_dict(),
    )
    assert "prune_blocks" in {plan["operator"] for plan in decision.plans}
    assert "head_gating" in {plan["operator"] for plan in decision.plans}
    assert "kv_token_select" in {plan["operator"] for plan in decision.plans}


def test_controller_context_preserves_multigranular_evidence():
    profile = {
        "model": {"architecture": "WanDiffusersDiT", "block_count": 2},
        "layers": [{"block_id": "block.0", "latency_ms": 1.0}],
        "block_sensitivity": [{"block_id": "block.0", "importance_score": 0.2}],
        "channel_sensitivity": [{"block_id": "block.0", "channels": [{"channel": index} for index in range(4096)]}],
        "head_sensitivity": [{"block_id": "block.0", "head": index} for index in range(32)],
        "attention_sensitivity": [{"block_id": "block.0", "head": index} for index in range(64)],
        "token_sensitivity": [{"token_count": 64, "keep_ratio": 0.75}],
    }

    block_profile = _compact_profile(profile, block_only=True)
    full_profile = _compact_profile(profile)

    assert block_profile["block_sensitivity"] == profile["block_sensitivity"]
    assert block_profile["layers"] == [{"block_id": "block.0", "latency_ms": 1.0}]
    assert not {
        "channel_sensitivity",
        "head_sensitivity",
        "attention_sensitivity",
        "token_sensitivity",
    }.intersection(block_profile)
    assert "channel_sensitivity" in full_profile
    assert "head_sensitivity" in full_profile

    payloads = []

    def provider(payload):
        payloads.append(payload)
        return {
            "source": "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}\n",
            "hypothesis": "profile-guided Block pruning",
            "parent_id": "M0000",
        }

    AlgorithmAgent(provider=provider).generate(
        "A0001",
        parent_id="M0000",
        profile=profile,
        history=[],
        search_space=default_search_space().to_dict(),
    )
    assert {
        "channel_sensitivity",
        "head_sensitivity",
        "attention_sensitivity",
        "token_sensitivity",
    }.issubset(payloads[0]["profile"])


def test_block_proposal_normalizer_uses_first_ordered_llm_plan(tmp_path):
    source = """def propose(profile, history):
    return [
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}},
        {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}},
    ]
"""
    sandbox = AlgorithmSandbox(tmp_path / "sandbox")

    result = sandbox.run(_normalize_block_proposal_source(source), _profile(), [])

    assert result.status == "success"
    assert result.plan.to_dict() == {
        "operator": "prune_blocks",
        "args": {"block_ids": ["block.2"]},
    }


def test_search_revision_does_not_rotate_candidate_families_by_algorithm_id():
    source = """def search(profile, history, state, search_space):
    token_row = {}
    token_ratio = float(token_row.get('keep_ratio', 0.75) or 0.75)
    ranked = sorted(range(2), key=lambda index: (-index, index))
    return {'rank': ranked}
"""
    assert _revise_search_program(source, "A0002") == source
    assert _revise_search_program(source, "A0003") == source


def test_generated_wan_candidates_use_supported_runtime_heads_and_report_missing_evidence(tmp_path):
    profile = {
        "model": {"architecture": "WanDiffusersDiT", "block_count": 2, "ffn_width": 8960},
        "layers": [{"block_id": "block.0"}, {"block_id": "block.1"}],
        "block_sensitivity": [{"block_id": "block.0", "importance_score": 0.1}],
        "channel_sensitivity": [
            {"block_id": "block.%d" % block, "channels": [{"channel": index, "importance_score": 0.1} for index in range(32)]}
            for block in range(2)
        ],
        "head_sensitivity": [{"block_id": "block.0", "attention": "attn1", "head": 0, "importance_score": 0.1}],
        "token_sensitivity": [{"layout": "wan_latent_grid", "token_count": 64, "keep_ratio": 0.75, "available": True}],
        "provenance": {"offline_simulation": False},
    }
    algorithm = AlgorithmAgent().generate("A0001", parent_id="M0000", profile=profile, history=[])
    decision = _run_algorithm_search_program(
        algorithm.source,
        AlgorithmSandbox(tmp_path / "wan-sandbox"),
        profile,
        [],
        {"depth": 1, "proposed_plan": None, "budget": {"max_candidates": 16}},
        default_search_space().to_dict(),
    )

    operators = {str(plan["operator"]) for plan in decision.plans}
    assert "prune_blocks" in operators
    assert "head_gating" not in operators
    assert "prune_attention_heads" not in operators
    assert any(item.get("reason") == "operator_not_in_active_search_space" for item in decision.diagnostics["omitted"])


def test_algorithm_evaluator_finite_check_uses_transformer_for_pipeline_parameter_mapping():
    class FiniteValue:
        def __bool__(self):
            return True

    class Parameter:
        def detach(self):
            return self

        def isfinite(self):
            return self

        def all(self):
            return FiniteValue()

    class Transformer:
        def parameters(self):
            return iter((Parameter(),))

    class Pipeline:
        parameters = {"transformer": "not-an-iterator"}

        def __init__(self):
            self.transformer = Transformer()

    AlgorithmModelEvaluator._assert_finite(Pipeline())

