import json
from unittest.mock import patch

from harness4h3.adapters.wan import FakeWanDiTAdapter
from harness4h3.evolution.agent import AlgorithmAgent
from harness4h3.evolution.loop import EvolutionLoop
from harness4h3.operators.wan import default_search_space


def run_search(tmp_path, results, plans=None, rank=None, budget=8, prior_plans=(), prior_parent_sha=None):
    plans = plans or [dict(operator='prune_blocks', args={'block_ids': ['block.%d' % i]}) for i in range(3)]
    rank = list(range(len(plans))) if rank is None else rank
    source = "def propose(profile, history):\n    return %r\ndef search(profile, history, state, search_space):\n    return %r\n" % (plans[0], {'candidates': plans, 'rank': rank, 'stop': True})
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model('fake://baseline'), tmp_path / 'parent')
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / 'run', algorithm_improvement=True)
    if prior_plans:
        prior_round = loop.rounds_root / 'R0001'
        prior_round.mkdir(parents=True)
        (prior_round / 'algorithm-program-trace.json').write_text(json.dumps({
            'algorithm_id': 'A0001',
            'measured_search': {
                'parent_checkpoint_sha256': prior_parent_sha or parent.sha256,
                'quality_floor_ratio': 0.5,
                'candidate_results': [
                    {
                        'plan': plan,
                        'valid': True,
                        'quality_score': 0.9,
                        'quality_floor_ratio': 0.5,
                        'quality_feasible': True,
                        'search_score': 0.91,
                        'efficiency_reduction_ratio': 0.1,
                    }
                    for plan in prior_plans
                ],
            },
        }))
    proposed = loop.sandbox.run(source, {}, [])
    with patch('harness4h3.evolution.loop.AlgorithmModelEvaluator') as cls:
        cls.return_value.evaluate.side_effect = results
        result = loop._complete_standard_algorithm_search(source=source, proposed_result=proposed, profile={}, history=[], parent=parent, output_root=tmp_path / 'search', seed=7, search_budget={'max_candidates': 16, 'max_full_evals': budget, 'quality_floor_ratio': 0.5}, search_space_payload=default_search_space().to_dict())
        return result, cls.return_value.evaluate.call_count


def test_all_invalid_search_has_no_winner_and_retains_every_failure(tmp_path):
    result, count = run_search(tmp_path, [{'valid': False, 'failure_reason': 'broken_reload'}] * 3)
    assert count == 3
    assert result['valid'] is False
    assert 'selected_plan' not in result
    assert result['algorithm_search']['complete'] is False
    assert len(result['algorithm_search']['candidate_results']) == 3


def test_search_selects_valid_candidate_and_reports_real_index(tmp_path):
    result, _ = run_search(tmp_path, [
        {'valid': True},
        {'valid': True, 'quality_score': float('nan')},
        {'valid': True, 'quality_score': 0.99},
        {'valid': False, 'quality_score': 1.0},
    ], plans=[
        dict(operator='prune_blocks', args={'block_ids': ['block.%d' % i]}) for i in range(4)
    ], rank=[3, 2, 1, 0])
    assert result['algorithm_search']['selected_rank_index'] == 1
    assert result['algorithm_search']['selected_rank_position'] == 2
    assert result['algorithm_search']['valid_candidate_count'] == 3
    assert result['algorithm_search']['quality_feasible_candidate_count'] == 1


def test_duplicate_does_not_consume_budget_slot(tmp_path):
    a = {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}
    b = {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}
    c = {'operator': 'prune_blocks', 'args': {'block_ids': ['block.2']}}
    result, count = run_search(tmp_path, [{'valid': True, 'quality_score': .8}] * 2, plans=[a, a, b, c], budget=2)
    assert count == 2
    skipped = result['algorithm_search']['not_evaluated']
    assert [row['reason'] for row in skipped] == ['duplicate', 'budget']
    assert result['algorithm_search']['candidate_count'] == 4


def test_same_parent_plan_is_not_measured_again_but_other_candidates_remain_eligible(tmp_path):
    repeated = dict(operator='prune_blocks', args={'block_ids': ['block.0']})
    novel = dict(operator='prune_blocks', args={'block_ids': ['block.1']})
    result, count = run_search(
        tmp_path,
        [{'valid': True, 'quality_score': 0.9}],
        plans=[repeated, novel],
        budget=2,
        prior_plans=[repeated],
    )

    assert count == 1
    search = result['algorithm_search']
    assert search['previously_measured_plan_count'] == 1
    assert [row['reason'] for row in search['not_evaluated']] == ['previously_measured_same_parent']
    assert search['candidate_results'][0]['plan']['args']['block_ids'] == ['block.1']


def test_same_plan_is_eligible_again_for_a_different_parent_checkpoint(tmp_path):
    repeated = dict(operator='prune_blocks', args={'block_ids': ['block.0']})
    result, count = run_search(
        tmp_path,
        [{'valid': True, 'quality_score': 0.9}],
        plans=[repeated],
        budget=1,
        prior_plans=[repeated],
        prior_parent_sha='different-checkpoint',
    )

    assert count == 1
    assert result['algorithm_search']['previously_measured_plan_count'] == 0


def test_new_round_seeds_agent_recovery_with_same_parent_near_misses(tmp_path):
    repeated = dict(operator='prune_blocks', args={'block_ids': ['block.0']})
    novel = dict(operator='prune_blocks', args={'block_ids': ['block.2']})
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model('fake://baseline'), tmp_path / 'parent')
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / 'run', algorithm_improvement=True)
    prior_round = loop.rounds_root / 'R0001'
    prior_round.mkdir(parents=True)
    (prior_round / 'algorithm-program-trace.json').write_text(json.dumps({
        'algorithm_id': 'A0001',
        'measured_search': {
            'parent_checkpoint_sha256': parent.sha256,
            'quality_floor_ratio': 0.98,
            'candidate_results': [{
                'plan': repeated,
                'valid': True,
                'quality_score': 0.97,
                'quality_floor_ratio': 0.98,
                'quality_feasible': False,
                'search_score': 0.972,
                'efficiency_reduction_ratio': 0.02,
            }],
        },
    }))
    source = (
        "def propose(profile, history):\n"
        "    return %r\n"
        "def search(profile, history, state, search_space):\n"
        "    if state.get('depth') > 1 and state.get('measured_candidates'):\n"
        "        return {'candidates': [%r], 'rank': [0], 'stop': True}\n"
        "    return {'candidates': [%r], 'rank': [0], 'stop': True}\n"
    ) % (repeated, novel, repeated)
    proposed = loop.sandbox.run(source, {}, [])

    with patch('harness4h3.evolution.loop.AlgorithmModelEvaluator') as cls:
        cls.return_value.evaluate.return_value = {'valid': True, 'quality_score': 0.97}
        result = loop._complete_standard_algorithm_search(
            source=source,
            proposed_result=proposed,
            profile={},
            history=[],
            parent=parent,
            output_root=tmp_path / 'search',
            seed=7,
            search_budget={'max_candidates': 4, 'max_full_evals': 1, 'max_search_depth': 2, 'quality_floor_ratio': 0.98},
            search_space_payload={
                **default_search_space().to_dict(),
                "operators": [item for item in default_search_space().to_dict()["operators"] if item["name"] == "prune_blocks"],
            },
        )

    search = result['algorithm_search']
    assert cls.return_value.evaluate.call_count == 1
    assert search['initial_search_depth'] == 2
    assert search['prior_measurement_candidate_count'] == 1
    assert search['candidate_results'][0]['plan'] == novel


def run_generated_block_search(tmp_path, quality_for_plan):
    profile = {
        'model': {'architecture': 'WanFakeDiT', 'block_count': 4, 'ffn_width': 16},
        'layers': [{'block_id': 'block.%d' % i, 'latency_ms': 1.0} for i in range(4)],
        'block_sensitivity': [
            {'block_id': 'block.%d' % i, 'importance_score': 0.1 + i * 0.1, 'available': True}
            for i in range(4)
        ],
    }
    algorithm = AlgorithmAgent().generate('A0001', parent_id='M0000', profile=profile, history=[])
    adapter = FakeWanDiTAdapter(block_count=4, hidden_size=8, ffn_width=16)
    parent = adapter.save_model(adapter.load_model('fake://baseline'), tmp_path / 'parent')
    loop = EvolutionLoop(adapter=adapter, parent=parent, output_root=tmp_path / 'run', algorithm_improvement=True)
    proposed = loop.sandbox.run(algorithm.source, profile, [])
    assert proposed.status == 'success'

    def evaluate(candidate):
        removed = candidate.plan.args['block_ids']
        return {
            'valid': True,
            'quality_score': quality_for_plan(removed),
            'parameter_reduction_ratio': 0.1 * len(removed),
            'parent_checkpoint_sha256': parent.sha256,
        }

    with patch('harness4h3.evolution.loop.AlgorithmModelEvaluator') as cls:
        cls.return_value.evaluate.side_effect = evaluate
        result = loop._complete_standard_algorithm_search(
            source=algorithm.source,
            proposed_result=proposed,
            profile=profile,
            history=[],
            parent=parent,
            output_root=tmp_path / 'search',
            seed=7,
            search_budget={'max_candidates': 8, 'max_full_evals': 8, 'max_search_depth': 2, 'quality_floor_ratio': 0.98},
            search_space_payload={
                **default_search_space().to_dict(),
                "operators": [item for item in default_search_space().to_dict()["operators"] if item["name"] == "prune_blocks"],
            },
        )
        return result, cls, parent


def test_quality_failure_triggers_block_restore_replace_and_selects_feasible_candidate(tmp_path):
    result, evaluator_class, parent = run_generated_block_search(
        tmp_path,
        lambda removed: 0.99 if removed == ['block.1'] else (0.97 if len(removed) == 1 else 0.90),
    )

    search = result['algorithm_search']
    assert result['valid'] is True
    assert result['selected_plan']['args']['block_ids'] == ['block.1']
    assert search['candidate_scope'] == 'registered_multigranular'
    assert evaluator_class.call_args.args[2] is parent
    assert all(row['parent_checkpoint_sha256'] == parent.sha256 for row in search['candidate_results'])
    assert all(row['plan']['operator'] == 'prune_blocks' for row in search['candidate_results'])
    assert search['selection_status'] == 'quality_feasible'
    assert len(search['pareto_front_plan_digests']) >= 1


def test_valid_near_miss_enters_validation_without_lowering_quality_gate(tmp_path):
    result, evaluator_class, parent = run_generated_block_search(tmp_path, lambda removed: 0.95)

    search = result['algorithm_search']
    assert result['valid'] is True
    assert result['validation_only'] is True
    assert result['validation_only_reason'] == 'no_calibration_candidate_met_quality_floor'
    assert result['selected_plan']['operator'] == 'prune_blocks'
    assert search['quality_feasible_candidate_count'] == 0
    assert search['selection_status'] == 'validation_only_near_miss'
    assert search['evaluated_candidate_count'] == evaluator_class.return_value.evaluate.call_count
    assert len(search['candidate_results']) == evaluator_class.return_value.evaluate.call_count
    assert all(row['parent_checkpoint_sha256'] == parent.sha256 for row in search['candidate_results'])
