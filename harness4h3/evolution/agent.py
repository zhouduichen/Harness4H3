from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .candidate import AlgorithmCandidate, build_source_diff


Provider = Callable[[Mapping[str, Any]], Mapping[str, Any]]


ALGORITHM_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source", "hypothesis", "parent_id"],
    "properties": {
        "source": {"type": "string"},
        "hypothesis": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
        "predicted_gain": {"type": "object"},
    },
}

IMPROVEMENT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source", "hypothesis", "parent_id"],
    "properties": {
        "source": {"type": "string"},
        "hypothesis": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
        "predicted_gain": {"type": "object"},
        "change_summary": {"type": "string"},
        "change_rationale": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}

SEARCH_SPACE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["spec", "rationale", "evidence_ids"],
    "properties": {
        "spec": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "version", "argument_schema", "granularity"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "argument_schema": {"type": "object"},
                "granularity": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "trusted_factory_key": {"type": "string"},
                "implementation_source": {"type": "string"},
                "implementation_ref": {"type": "string"},
                "execution_domain": {"type": "string"},
            },
        },
        "rationale": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}


def _seed_source_base(
    block_id: str,
    algorithm_id: str,
    search_choice: str,
    *,
    retention_ratio: float = 0.25,
    channel_block_id: Optional[str] = None,
) -> str:
    return (
        "def propose(profile, history):\n"
        "    # Seed algorithm %s proposes a profile-ranked Block removal; search() compares bounded subsets.\n"
        "    selected = %r\n"
        "    return {'operator': 'prune_blocks', 'args': {'block_ids': [selected]}}\n" % (algorithm_id, block_id)
    )


def _concise_multigranular_search_program_suffix(algorithm_id: str) -> str:
    return '''\ndef search(profile, history, state, search_space):
    # Evidence-ranked multi-granularity search: ALGORITHM_ID.
    import math
    def finite(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    names = [str(x.get('name')) for x in search_space.get('operators', []) if isinstance(x, dict)]
    plans, scores, omitted = [], [], []
    def add(op, args, score=0.0):
        if op not in names:
            omitted.append({'operator': op, 'reason': 'operator_not_in_active_search_space'})
            return
        p = {'operator': op, 'args': args}
        if p not in plans:
            plans.append(p); scores.append(float(score))
    layers = [x for x in profile.get('layers', []) if isinstance(x, dict) and x.get('block_id')]
    ids = [str(x['block_id']) for x in layers]
    costs = {str(x['block_id']): float(x.get('latency_ms', x.get('estimated_flops', 0.0)) or 0.0) for x in layers}
    cost_source = 'structural_profile_proxy'
    execution = profile.get('execution', {})
    execution_cases = execution.get('cases', []) if isinstance(execution, dict) else []
    reference_case = next((x for x in execution_cases if isinstance(x, dict) and x.get('variant') == 'dense'), None)
    if isinstance(reference_case, dict):
        diagnostic = reference_case.get('diagnostic_pass', {})
        module_rows = diagnostic.get('module_timings', []) if isinstance(diagnostic, dict) else []
        measured_costs = {
            str(x.get('block_id')): float(x.get('cuda_time_ms_total', 0.0))
            for x in module_rows
            if isinstance(x, dict) and x.get('category') == 'transformer_block' and x.get('block_id') in ids
            and finite(x.get('cuda_time_ms_total')) and float(x.get('cuda_time_ms_total')) > 0.0
        }
        if measured_costs:
            costs.update(measured_costs)
            cost_source = 'real_cuda_transformer_block_forward_hook'
    block_rows = profile.get('block_timestep_sensitivity') or profile.get('tica_block_sensitivity') or profile.get('block_sensitivity', [])
    block_scores, by_time = {}, {}
    for row in block_rows:
        if not isinstance(row, dict) or row.get('available') is False or row.get('block_id') not in ids: continue
        ts = row.get('per_timestep', {})
        normalized_ts = {}
        if isinstance(ts, dict):
            for key, value in ts.items():
                if not finite(value): continue
                bucket = str(key)
                if bucket not in ('high_noise', 'middle', 'low_noise'):
                    try:
                        timestep = float(key)
                    except (TypeError, ValueError):
                        continue
                    bucket = 'high_noise' if timestep >= 0.8 else ('middle' if 0.4 <= timestep <= 0.6 else ('low_noise' if timestep <= 0.2 else ''))
                if bucket: normalized_ts[bucket] = float(value)
        ts = normalized_ts
        score = sum(ts.values()) / len(ts) if ts else row.get('importance_score')
        if finite(score):
            block_scores[str(row['block_id'])] = float(score); by_time[str(row['block_id'])] = ts
    blocks = sorted(block_scores, key=lambda b: (block_scores[b], -costs.get(b, 0.0), b))
    for b in blocks: add('prune_blocks', {'block_ids': [b]}, costs.get(b, 0.0) / (1.0 + block_scores[b]))
    heads = {}
    for row in profile.get('head_sensitivity', profile.get('attention_sensitivity', [])):
        if isinstance(row, dict) and row.get('available') is not False and row.get('attention', 'attn1') == 'attn1' and row.get('block_id') in ids and isinstance(row.get('head'), int) and not isinstance(row.get('head'), bool) and finite(row.get('importance_score')):
            heads.setdefault(str(row['block_id']), {})[int(row['head'])] = float(row['importance_score'])
    local = []
    for b, scores_by_head in heads.items():
        if len(scores_by_head) > 1:
            for drop in sorted(scores_by_head, key=lambda h: (scores_by_head[h], h)):
                local.append((scores_by_head[drop], b, [h for h in sorted(scores_by_head) if h != drop]))
    for score, b, keep in sorted(local):
        add('head_gating', {'attention': 'attn1', 'block_ids': [b], 'head_indices': keep, 'score': 'activation_l2'}, 1.0 / (1.0 + max(0.0, score)))
    if 'head_gating' in names and not local:
        omitted.append({'operator': 'head_gating', 'reason': 'missing_per_block_head_sensitivity'})
    if 'prune_attention_heads' in names:
        reason = 'wan_structural_head_pruning_not_projection_aligned' if str(profile.get('model', {}).get('architecture', '')) == 'WanDiffusersDiT' else 'structural_head_pruning_not_supported_by_profile_evidence'
        omitted.append({'operator': 'prune_attention_heads', 'reason': reason})
    tokens = [x for x in profile.get('token_sensitivity', []) if isinstance(x, dict) and x.get('available') is not False]
    token = tokens[0] if tokens else {}
    count = int(token.get('token_count', 0) or 0)
    grid = token.get('grid_shape', profile.get('token_grid', profile.get('model', {}).get('token_grid', [])))
    grid_ok = isinstance(grid, (list, tuple)) and len(grid) == 3 and count > 0 and all(isinstance(v, int) and v > 0 for v in grid) and grid[0] * grid[1] * grid[2] == count
    if token.get('layout') == 'wan_latent_grid' and not grid_ok:
        omitted.append({'operator': 'token_merge', 'reason': 'missing_spatiotemporal_token_grid'})
    ratios = sorted(set(float(x['keep_ratio']) for x in tokens if finite(x.get('keep_ratio')) and 0.5 <= float(x['keep_ratio']) < 1.0), reverse=True)
    geom = {'grid_shape': [int(v) for v in grid]} if grid_ok else {}
    if tokens and ratios:
        for ratio in ratios:
            add('prune_tokens', dict({'keep_ratio': ratio, 'score': 'l2_norm'}, **geom), 0.2)
            if token.get('layout') != 'wan_latent_grid' or grid_ok:
                add('token_merge', dict({'keep_ratio': ratio, 'mode': 'spatiotemporal' if grid_ok else 'adjacent'}, **geom), 0.2)
            add('kv_token_select', {'keep_ratio': ratio, 'block_ids': blocks[:2], 'score': 'l2_norm'}, 0.3)
            add('sparse_runtime', {'keep_ratio': ratio, 'attention': 'attn1'}, 0.2)
    else:
        for op in ('prune_tokens', 'token_merge', 'kv_token_select', 'sparse_runtime'):
            if op in names: omitted.append({'operator': op, 'reason': 'missing_token_sensitivity'})
    if count:
        args = {'window_size': max(1, min(32, int(math.sqrt(count)))), 'attention': 'attn1'}
        if grid_ok:
            args['grid_shape'] = [int(v) for v in grid]
            args['window_radius'] = [int(grid[0] > 1), int(grid[1] // 4), int(grid[2] // 4)]
        add('attention_window', args, 0.15)
    sampling = profile.get('sampling_step_sensitivity', {})
    steps = sampling.get('candidates', []) if isinstance(sampling, dict) else []
    targets = [int(x['target_steps']) for x in steps if isinstance(x, dict) and x.get('available') is not False and x.get('target_steps') is not None]
    for target in sorted(set(targets), reverse=True):
        add('sampling_step_skip', {'target_steps': target}, 0.15)
    recipe = reference_case.get('recipe', {}) if isinstance(reference_case, dict) else {}
    step_count = int(recipe.get('num_inference_steps', 16)) if isinstance(recipe, dict) else 16
    svoo_args = {'method': 'svoo', 'model_key': 'wan21-t2v-1.3b', 'config': {}}
    magcache_config = {'threshold': 0.12, 'max_skip_steps': 4, 'retention_ratio': 0.2, 'num_inference_steps': step_count, 'calibrate': False}
    if step_count == 16:
        magcache_config['mag_ratios'] = [1.0, 1.0043765903, 0.9673218131, 0.9690133333, 0.9731017649, 0.9714230895, 0.9706552029, 0.9636064172, 0.9583126307, 0.9527728260, 0.9466742873, 0.9315682659, 0.9141220748, 0.8960618079, 0.8554908037, 0.7984179854]
    magcache_args = {'method': 'magcache', 'config': magcache_config}
    category_costs = diagnostic.get('category_cuda_time_ms_total', {}) if isinstance(reference_case, dict) and isinstance(diagnostic, dict) else {}
    attention_priority = float(category_costs.get('self_attention', category_costs.get('self_attention_cuda_ms', 0.0)) or 0.0) if isinstance(category_costs, dict) else 0.0
    full_generation_s = float(reference_case.get('generation_wall_time_s', 0.0) or 0.0) if isinstance(reference_case, dict) else 0.0
    add('svoo_attention', svoo_args, max(attention_priority, 0.01))
    add('magcache', magcache_args, max(full_generation_s * 1000.0, 0.01))
    if 'compose' in names and 'svoo_attention' in names and 'magcache' in names:
        add('compose', {'steps': [
            {'operator': 'svoo_attention', 'args': svoo_args},
            {'operator': 'magcache', 'args': magcache_args},
        ]}, max(attention_priority, 0.01) + max(full_generation_s * 1000.0, 0.01))
    intervals = {'high_noise': (0.85, 0.95), 'middle': (0.45, 0.55), 'low_noise': (0.05, 0.15)}
    temporal = [(v, b, t) for b, values in by_time.items() for t, v in values.items() if t in intervals]
    temporal.sort()
    if temporal:
        for delta, b, bucket in temporal[:6]:
            low, high = intervals[bucket]
            gate = {'rules': [{'block_id': b, 'min_timestep': low, 'max_timestep': high}], 'policy': 'skip_when_in_range'}
            add('timestep_block_gate', gate, 1.0 / (1.0 + max(0.0, delta)))
            if ratios and 'timestep_attention_policy' in names:
                add('timestep_attention_policy', {'rules': [{'min_timestep': low, 'max_timestep': high, 'keep_ratio': ratios[0], 'block_ids': [b], 'score': 'l2_norm'}], 'policy': 'select_kv_tokens'}, 1.0 / (1.0 + max(0.0, delta)))
    elif 'timestep_block_gate' not in names:
        omitted.append({'operator': 'timestep_block_gate', 'reason': 'operator_not_in_active_search_space'})
    else:
        omitted.append({'operator': 'timestep_block_gate', 'reason': 'missing_block_by_timestep_sensitivity'})
    if 'timestep_attention_policy' not in names:
        omitted.append({'operator': 'timestep_attention_policy', 'reason': 'operator_not_in_active_search_space'})
    elif not temporal or not ratios:
        omitted.append({'operator': 'timestep_attention_policy', 'reason': 'missing_token_or_temporal_sensitivity'})
    proposed = state.get('proposed_plan') if isinstance(state, dict) else None
    proposed_ok = isinstance(proposed, dict) and proposed.get('operator') in names and isinstance(proposed.get('args'), dict)
    if proposed_ok and proposed.get('operator') == 'compose':
        proposed_steps = proposed['args'].get('steps', [])
        proposed_ok = isinstance(proposed_steps, list) and len(proposed_steps) == 2 and all(isinstance(step, dict) and isinstance(step.get('args'), dict) and step.get('operator') in names for step in proposed_steps)
        if proposed_ok:
            proposed_ok = [step['operator'] for step in proposed_steps] == ['svoo_attention', 'magcache']
    if proposed_ok and not state.get('algorithm_improvement') and proposed not in plans:
        plans.insert(0, proposed); scores.insert(0, 10.0)
    elif isinstance(proposed, dict) and proposed.get('operator') == 'compose' and not proposed_ok:
        omitted.append({'operator': 'compose', 'reason': 'unsupported_or_over_budget_interaction'})
    prior_plans = [item.get('plan') for item in state.get('measured_candidates', []) if isinstance(item, dict) and isinstance(item.get('plan'), dict)]
    retained = [(plan, score) for plan, score in zip(plans, scores) if plan not in prior_plans]
    omitted.extend({'operator': plan.get('operator'), 'reason': 'previously_measured_same_parent'} for plan, _score in zip(plans, scores) if plan in prior_plans)
    plans = [plan for plan, _score in retained]; scores = [score for _plan, score in retained]
    if not plans and retained == [] and prior_plans:
        omitted.append({'operator': 'search', 'reason': 'all_evidence_supported_candidates_already_measured'})
    limit = max(1, min(64, int(state.get('budget', {}).get('max_candidates', 16) or 16)))
    family = {'prune_blocks': 'structure', 'head_gating': 'head', 'prune_tokens': 'token', 'token_merge': 'token', 'kv_token_select': 'kv', 'sparse_runtime': 'attention', 'svoo_attention': 'attention', 'magcache': 'cache', 'attention_window': 'attention', 'sampling_step_skip': 'sampling', 'timestep_block_gate': 'timestep', 'timestep_attention_policy': 'timestep', 'compose': 'hybrid'}
    queues, family_order = {}, []
    for i, plan in enumerate(plans):
        key = family.get(plan['operator'], plan['operator'])
        if key not in queues: queues[key] = []; family_order.append(key)
        queues[key].append(i)
    counts = {str(k): max(0, int(v)) for k, v in state.get('family_sample_counts', {}).items()}
    for item in state.get('measured_candidates', []):
        if isinstance(item, dict) and item.get('valid') is True:
            fam = str(item.get('family', ''))
            if fam: counts[fam] = max(counts.get(fam, 0), 1)
    offsets, chosen = {}, []
    family_evidence = {str(k): float(v) for k, v in state.get('family_gpu_evidence', {}).items() if finite(v) and float(v) >= 0.0}
    for item in state.get('measured_candidates', []):
        if not isinstance(item, dict) or item.get('valid') is not True or item.get('quality_feasible') is not True:
            continue
        if item.get('offline_simulation') is True or item.get('evidence_level') not in ('real_wan_gpu', 'measured_video'):
            continue
        fam = str(item.get('family', ''))
        latency_gain = item.get('latency_gain_s')
        evaluation_cost = item.get('evaluation_cost_s')
        if finite(latency_gain) and finite(evaluation_cost) and float(evaluation_cost) > 0:
            family_evidence[fam] = max(family_evidence.get(fam, 0.0), max(0.0, float(latency_gain)) / float(evaluation_cost))
    # Ensure one unmeasured family probe when the budget permits; use real
    # same-parent gain per GPU second afterward, with profile bottlenecks as
    # the deterministic cold-start prior.
    unmeasured = [key for key in family_order if counts.get(key, 0) == 0]
    for key in unmeasured[:limit]:
        chosen.append(queues[key][0]); offsets[key] = 1; counts[key] = 1
    while len(chosen) < min(limit, len(plans)):
        available = [key for key in family_order if offsets.get(key, 0) < len(queues[key])]
        if not available: break
        def priority(key):
            row = queues[key][offsets.get(key, 0)]
            cold = max(0.0, float(scores[row]))
            evidence = family_evidence.get(key)
            count = counts.get(key, 0)
            return (-(float(evidence) if evidence is not None else cold) / (1.0 + count), count, family_order.index(key))
        key = min(available, key=priority)
        chosen.append(queues[key][offsets.get(key, 0)])
        offsets[key] = offsets.get(key, 0) + 1; counts[key] = counts.get(key, 0) + 1
    plans = [plans[i] for i in chosen]
    scores = [scores[i] for i in chosen]
    rank_queues, ranked_families = {}, []
    for i, plan in enumerate(plans):
        family_name = family.get(plan['operator'], plan['operator'])
        if family_name not in rank_queues: rank_queues[family_name] = []; ranked_families.append(family_name)
        rank_queues[family_name].append(i)
    for family_name in ranked_families:
        rank_queues[family_name].sort(key=lambda i: (-scores[i], i))
    family_counts, family_offsets, rank = {}, {}, []
    while len(rank) < len(plans):
        available = [name for name in ranked_families if family_offsets.get(name, 0) < len(rank_queues[name])]
        if not available: break
        family_name = min(available, key=lambda name: (family_counts.get(name, 0), ranked_families.index(name)))
        rank.append(rank_queues[family_name][family_offsets.get(family_name, 0)])
        family_offsets[family_name] = family_offsets.get(family_name, 0) + 1
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
    bottlenecks = {}
    if isinstance(reference_case, dict):
        diagnostic = reference_case.get('diagnostic_pass', {})
        categories = diagnostic.get('category_cuda_time_ms_total', {}) if isinstance(diagnostic, dict) else {}
        if isinstance(categories, dict): bottlenecks = categories
    allocations = {key: sum(1 for plan in plans if family.get(plan['operator'], plan['operator']) == key) for key in sorted(set(family.get(plan['operator'], plan['operator']) for plan in plans))}
    candidate_evidence = []
    for i, plan in enumerate(plans):
        family_name = family.get(plan['operator'], plan['operator'])
        measured_utility = family_evidence.get(family_name)
        allocation_reason = 'paired_real_wan_gain_per_gpu_second' if measured_utility is not None else ('profile_bottleneck_probe' if float(scores[i]) > 0.0 and bottlenecks else 'cold_start_family_probe')
        evidence_refs = ['profile.execution.diagnostic_pass.category_cuda_time_ms_total'] if bottlenecks else []
        if measured_utility is not None: evidence_refs.append('same_parent_real_wan_family_measurement')
        candidate_evidence.append({'candidate_index': i, 'family': family_name, 'profile_score': float(scores[i]), 'measured_gain_per_gpu_second': measured_utility, 'allocation_reason': allocation_reason, 'evidence_refs': evidence_refs, 'family_sample_count': int(state.get('family_sample_counts', {}).get(family_name, 0)), 'module_timing_scope': 'inclusive_nested_for_bottleneck_order_only' if bottlenecks else 'unavailable'})
    return {'candidates': plans, 'rank': rank, 'stop': True, 'diagnostics': {'omitted': omitted, 'generated_operators': [x['operator'] for x in plans], 'family_ranking_order': [plans[i]['operator'] for i in rank], 'family_allocated_counts': allocations, 'family_gpu_utility': family_evidence, 'candidate_evidence': candidate_evidence, 'ranked_blocks': blocks, 'block_cost_source': cost_source, 'inclusive_cuda_bottleneck_categories_ms': bottlenecks}}
'''.replace('ALGORITHM_ID', algorithm_id)


def _search_program_suffix(algorithm_id: str, *, exploration: bool = False) -> str:
    """Build the bounded, evidence-driven multi-granularity search program."""

    return _concise_multigranular_search_program_suffix(algorithm_id)


def _seed_source(
    block_id: str,
    algorithm_id: str,
    search_choice: str,
    *,
    retention_ratio: float = 0.25,
    channel_block_id: Optional[str] = None,
) -> str:
    return _seed_source_base(
        block_id,
        algorithm_id,
        search_choice,
        retention_ratio=retention_ratio,
        channel_block_id=channel_block_id,
    ) + _search_program_suffix(algorithm_id, exploration=search_choice == "explore")


def _ensure_search_program(source: str, algorithm_id: str, *, exploration: bool = False) -> str:
    if "def search(" in source:
        return source
    return source + _search_program_suffix(algorithm_id, exploration=exploration)


def _normalize_block_proposal_source(source: str) -> str:
    """Accept an LLM's ordered plan list as its single seed proposal.

    The standard search protocol reserves candidate lists for ``search`` but
    providers sometimes put the same ordered plans in ``propose``. Preserve
    that ordering by using its first valid Block plan; the trusted Sandbox
    still validates the resulting operator and arguments.
    """

    marker = "# harness4h3_plan_proposal_normalizer"
    if marker in source:
        return source
    return source + "\n" + marker + "\n" + (
        "_harness4h3_original_propose = propose\n"
        "def propose(profile, history):\n"
        "    result = _harness4h3_original_propose(profile, history)\n"
        "    if isinstance(result, (list, tuple)):\n"
        "        for candidate in result:\n"
        "            if (isinstance(candidate, dict)\n"
        "                    and isinstance(candidate.get('operator'), str)\n"
        "                    and isinstance(candidate.get('args'), dict)):\n"
        "                return candidate\n"
        "        return None\n"
        "    return result\n"
    )


def _revise_search_program(source: str, algorithm_id: str) -> str:
    """Keep the candidate generator evidence-driven; never rotate families by ID."""

    return source


@dataclass(frozen=True)
class AlgorithmAgent:
    provider: Optional[Provider] = None

    # Retention ratio is an algorithm-design budget, not a model mutation
    # supplied by the controller.  It is bounded before source generation.
    explore_ffn_retention_ratio: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < float(self.explore_ffn_retention_ratio) < 1.0:
            raise ValueError("explore_ffn_retention_ratio must be in (0, 1)")

    def generate(
        self,
        algorithm_id: str,
        *,
        parent_id: Optional[str],
        parent_algorithm_id: Optional[str] = None,
        profile: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        skills: Sequence[Mapping[str, Any]] = (),
        pareto_front: Sequence[Mapping[str, Any]] = (),
        search_space: Optional[Mapping[str, Any]] = None,
        search_choice: str = "exploit",
        protocol_mode: str = "standard",
    ) -> AlgorithmCandidate:
        if self.provider is None:
            previously_selected = self._previously_selected_blocks(history)
            quality_feedback = self._has_quality_failure(history)
            block_id = self._least_sensitive_block(
                profile,
                excluded=previously_selected,
            )
            source = _seed_source(
                block_id,
                algorithm_id,
                "exploit",
            )
            if quality_feedback:
                hypothesis = "quality feedback requires conservative block-only pruning"
                design_strategy = "conservative_block"
            else:
                hypothesis = "compare profile-ranked Block removal subsets under the measured quality floor"
                design_strategy = "profile_guided_block_search"
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=parent_id,
                parent_algorithm_id=parent_algorithm_id,
                hypothesis=hypothesis,
                metadata={"generator": "deterministic_seed", "operator": "prune_blocks", "design_strategy": design_strategy, "search_choice": search_choice, "predicted_gain": {"latency": 0.05, "model_size": 0.10, "basis": "profile_sensitivity_and_operator_proxy"}},
            )
        compact_history = _compact_context_items(history)
        history_keys = {
            (str(item.get("memory_kind", "")), str(item.get("experience_id", "")))
            for item in history
            if isinstance(item, Mapping)
        }
        compact_skills = _compact_context_items(
            [
                item
                for item in skills
                if not isinstance(item, Mapping)
                or (str(item.get("memory_kind", "")), str(item.get("experience_id", ""))) not in history_keys
            ]
        )
        payload = {
            "profile": _compact_profile(
                profile,
                block_only=False,
            ),
            "history": compact_history,
            "skills": compact_skills,
            "pareto_front": _compact_pareto_front(pareto_front),
            "search_space": _compact_search_space(
                search_space or {}
            ),
            "parent_id": parent_id,
            "search_policy": {"choice": search_choice},
            "protocol_mode": str(protocol_mode).strip().lower() or "standard",
        }
        raw = self.provider(payload)
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm_generation_invalid: provider response must be an object")
        unknown = sorted(set(raw) - {"source", "hypothesis", "parent_id", "predicted_gain"})
        if unknown:
            raise ValueError("algorithm_generation_invalid: unsupported field(s): %s" % ", ".join(unknown))
        response_parent = raw.get("parent_id", parent_id)
        if response_parent != parent_id:
            raise ValueError("algorithm_generation_invalid: provider cannot change parent_id")
        source = raw.get("source")
        if not isinstance(source, str):
            raise ValueError("algorithm_generation_invalid: source must be text")
        if str(protocol_mode).strip().lower() != "tica":
            source = _normalize_block_proposal_source(_ensure_search_program(source, algorithm_id))
        controller = dict(getattr(self.provider, "last_response_metadata", {}) or {})
        return AlgorithmCandidate.from_source(
            algorithm_id,
                source,
                parent_id=parent_id,
                parent_algorithm_id=parent_algorithm_id,
            hypothesis=str(raw.get("hypothesis", "")),
            metadata={"generator": "structured_provider", "search_choice": search_choice, "predicted_gain": dict(raw.get("predicted_gain") or {}), "controller": controller},
        )

    def improve(
        self,
        algorithm_id: str,
        *,
        parent: AlgorithmCandidate,
        parent_id: Optional[str],
        profile: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        failures: Sequence[str] = (),
        skills: Sequence[Mapping[str, Any]] = (),
        search_space: Optional[Mapping[str, Any]] = None,
        protocol_mode: str = "standard",
    ) -> AlgorithmCandidate:
        """Improve an existing source with bounded, explicit feedback context."""

        if not isinstance(parent, AlgorithmCandidate):
            raise TypeError("algorithm improvement requires an AlgorithmCandidate parent")
        feedback = [str(item) for item in failures if str(item).strip()]
        if not feedback:
            feedback = list(_feedback_reasons(history))
        if self.provider is None:
            source, summary, rationale, hypothesis = _deterministic_improvement_source(
                algorithm_id, parent, profile, feedback
            )
            active_names = {
                str(item.get("name", ""))
                for item in (search_space or {}).get("operators", ())
                if isinstance(item, Mapping)
            }
            # TICA's trusted recursive beam search owns candidate generation.
            # Keep that contract unless a validated SearchSpace extension has
            # added a timestep-conditioned runtime policy for the standard
            # Algorithm Program to evaluate alongside it.
            if str(protocol_mode).strip().lower() != "tica" or "timestep_attention_policy" in active_names:
                source = _ensure_search_program(
                    source,
                    algorithm_id,
                    exploration=not any("quality" in item.lower() or "regression" in item.lower() for item in feedback),
                )
                source = _revise_search_program(source, algorithm_id)
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=parent_id,
                parent_algorithm_id=parent.algorithm_id,
                parent_source_sha256=parent.source_sha256,
                source_diff=build_source_diff(parent.source, source, parent_id=parent.algorithm_id, child_id=algorithm_id),
                hypothesis=hypothesis,
                change_summary=summary,
                change_rationale=rationale,
                evidence_ids=_history_ids(history),
                validation_status="pending",
                metadata={
                    "generator": "deterministic_improvement",
                    "algorithm_improvement": True,
                    "parent_source_sha256": parent.source_sha256,
                    "failure_reasons": feedback[:16],
                    "search_space": dict(search_space or {}),
                },
            )

        payload = {
            "parent_algorithm_id": parent.algorithm_id,
            "parent_source_sha256": parent.source_sha256,
            # The parent source is the object being improved.  Keep it intact;
            # bounded history/profile fields are handled separately below.
            "parent_source": parent.source,
            "parent_source_line_count": len(parent.source.splitlines()),
            "profile": _compact_profile(
                profile,
                block_only=False,
            ),
            "history": _compact_context_items(history),
            # Keep retrieved sources from crowding out measured candidate
            # evidence in a bounded controller context.
            "skills": _compact_context_items(skills, limit=2),
            "failure_reasons": feedback[:16],
            "search_space": (
                dict(search_space or {})
            ),
            "parent_id": parent_id,
            "task": "improve_existing_algorithm_source",
            "protocol_mode": str(protocol_mode).strip().lower() or "standard",
        }
        raw = self.provider(payload)
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm_improvement_invalid: provider response must be an object")
        unknown = sorted(set(raw) - {"source", "hypothesis", "parent_id", "predicted_gain", "change_summary", "change_rationale", "evidence_ids"})
        if unknown:
            raise ValueError("algorithm_improvement_invalid: unsupported field(s): %s" % ", ".join(unknown))
        if raw.get("parent_id", parent_id) != parent_id:
            raise ValueError("algorithm_improvement_invalid: provider cannot change parent_id")
        source = raw.get("source")
        if not isinstance(source, str):
            raise ValueError("algorithm_improvement_invalid: source must be text")
        if str(protocol_mode).strip().lower() != "tica":
            source = _normalize_block_proposal_source(_ensure_search_program(source, algorithm_id))
        controller = dict(getattr(self.provider, "last_response_metadata", {}) or {})
        return AlgorithmCandidate.from_source(
            algorithm_id,
            source,
            parent_id=parent_id,
            parent_algorithm_id=parent.algorithm_id,
            parent_source_sha256=parent.source_sha256,
            source_diff=build_source_diff(parent.source, source, parent_id=parent.algorithm_id, child_id=algorithm_id),
            hypothesis=str(raw.get("hypothesis", "")),
            change_summary=str(raw.get("change_summary", "provider source revision")),
            change_rationale=str(raw.get("change_rationale", "provider used parent source and feedback")),
            evidence_ids=tuple(str(item) for item in raw.get("evidence_ids", _history_ids(history))),
            validation_status="pending",
            metadata={
                "generator": "structured_improvement_provider",
                "algorithm_improvement": True,
                "parent_source_sha256": parent.source_sha256,
                "failure_reasons": feedback[:16],
                "predicted_gain": dict(raw.get("predicted_gain") or {}),
                "controller": controller,
            },
        )

    def propose_search_space_change(
        self,
        *,
        search_space: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        failures: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        """Propose a capability extension without granting execution trust."""

        active_names = {
            str(item.get("name"))
            for item in search_space.get("operators", [])
            if isinstance(item, Mapping) and item.get("name")
        }
        if self.provider is not None:
            payload = {
                "task": "propose_search_space_extension",
                "search_space": dict(search_space),
                "history": _compact_context_items(history),
                "failure_reasons": [str(item) for item in failures if str(item).strip()][:16],
            }
            raw = self.provider(payload)
            if not isinstance(raw, Mapping):
                raise ValueError("search_space_proposal_invalid: provider response must be an object")
            unknown = sorted(set(raw) - {"spec", "rationale", "evidence_ids"})
            if unknown:
                raise ValueError("search_space_proposal_invalid: unsupported field(s): %s" % ", ".join(unknown))
            spec = raw.get("spec")
            if not isinstance(spec, Mapping):
                raise ValueError("search_space_proposal_invalid: spec must be an object")
            if str(spec.get("name", "")) in active_names:
                raise ValueError("search_space_proposal_invalid: operator already exists")
            return {
                "spec": dict(spec),
                "rationale": str(raw.get("rationale", "provider identified a search-space limitation")),
                "evidence_ids": tuple(str(item) for item in raw.get("evidence_ids", ())),
            }
        failure_text = " ".join(str(item).lower() for item in failures)
        if (
            "timestep_block_gate" not in active_names
            and any(token in failure_text for token in ("timestep", "dynamic", "latency", "pareto"))
        ):
            implementation_source = """def propose(profile, history):
    layers = profile.get('layers', [])
    block_id = str(layers[0].get('block_id')) if layers else 'block.0'
    return {'operator': 'timestep_block_gate', 'args': {'rules': [{'block_id': block_id, 'min_timestep': 0.0, 'max_timestep': 0.3}], 'policy': 'skip_when_in_range'}}
"""
            return {
                "spec": {
                    "name": "timestep_block_gate",
                    "version": "1",
                    "argument_schema": {"rules": "list[object]", "policy": "str"},
                    "granularity": "runtime_timestep",
                    "constraints": ["normalized_timestep", "known_block_ids", "no_parameter_reduction"],
                    "limitations": ["runtime_only", "cpu_fake_structure_evidence_only", "requires_real_wan_replay_for_hardware_claim"],
                    "trusted_factory_key": "timestep_block_gate",
                    "implementation_source": implementation_source,
                },
                "rationale": "timestep-local quality/latency feedback motivates a dynamic Block gate that preserves parameters while changing the executed path",
                "evidence_ids": tuple(str(item) for item in failures if str(item).strip())[:16] or ("offline-bootstrap",),
            }
        if (
            "timestep_block_gate" in active_names
            and "timestep_attention_policy" not in active_names
            and any(token in failure_text for token in ("timestep", "dynamic", "latency", "attention", "pareto"))
        ):
            implementation_source = """def propose(profile, history):
    layers = profile.get('layers', [])
    block_id = str(layers[0].get('block_id')) if layers else 'block.0'
    return {'operator': 'timestep_attention_policy', 'args': {'rules': [
        {'min_timestep': 0.0, 'max_timestep': 0.3, 'keep_ratio': 0.5, 'block_ids': [block_id], 'score': 'l2_norm'},
        {'min_timestep': 0.7, 'max_timestep': 1.0, 'keep_ratio': 0.75, 'block_ids': [block_id], 'score': 'l2_norm'},
    ], 'policy': 'select_kv_tokens'}}
"""
            return {
                "spec": {
                    "name": "timestep_attention_policy",
                    "version": "1",
                    "argument_schema": {"rules": "list[object]", "policy": "str"},
                    "granularity": "timestep_attention",
                    "constraints": ["per_sample_timestep", "reduced_kv_length", "self_attention_only"],
                    "limitations": ["runtime_only", "cpu_fake_structure_evidence_only", "requires_real_wan_replay_for_hardware_claim"],
                    "trusted_factory_key": "timestep_attention_policy",
                    "implementation_source": implementation_source,
                },
                "rationale": "block-only timestep gating leaves attention/token compute unchanged; add a validated per-sample K/V budget",
                "evidence_ids": tuple(str(item) for item in failures if str(item).strip())[:16] or ("offline-bootstrap",),
            }
        if "prune_ffn_by_weight_threshold" in active_names:
            raise ValueError("search_space_no_unseen_deterministic_extension")
        return {
            "spec": {
                "name": "prune_ffn_by_weight_threshold",
                "version": "1",
                "argument_schema": {"block_ids": "list[str]", "threshold": "float"},
                "granularity": "channel",
                "constraints": ["wan_ffn_width_aligned", "threshold_must_change_width"],
                "limitations": ["weight_proxy_only", "not_real_wan_validated"],
                "trusted_factory_key": "prune_ffn_by_weight_threshold",
            },
            "rationale": "repeated channel-importance plateaus motivate a threshold-based channel operator",
            "evidence_ids": tuple(str(item) for item in failures if str(item).strip())[:16] or ("offline-bootstrap",),
        }

    @staticmethod
    def _least_sensitive_block(profile: Mapping[str, Any], excluded: Optional[set[str]] = None) -> str:
        records = profile.get("block_sensitivity", [])
        excluded = excluded or set()
        layers = {
            str(item.get("block_id")): item
            for item in profile.get("layers", [])
            if isinstance(item, Mapping) and isinstance(item.get("block_id"), str)
        }
        candidates = [item for item in records if isinstance(item, Mapping) and isinstance(item.get("block_id"), str) and item.get("block_id") not in excluded and isinstance(item.get("importance_score"), (int, float))]
        if candidates:
            def score(item: Mapping[str, Any]) -> tuple[float, float, float]:
                layer = layers.get(str(item["block_id"]), {})
                latency = float(layer.get("latency_ms", 0.0) or 0.0)
                flops = float(layer.get("estimated_flops", 0.0) or 0.0)
                importance = float(item["importance_score"])
                pressure = max(latency, flops)
                # Prefer low sensitivity while still targeting measurable work.
                return (importance / max(pressure, 1.0), importance, -pressure)

            return str(min(candidates, key=score)["block_id"])
        layers = profile.get("layers", [])
        for item in layers:
            if isinstance(item, Mapping) and isinstance(item.get("block_id"), str) and item.get("block_id") not in excluded:
                return str(item["block_id"])
        for item in records:
            if isinstance(item, Mapping) and isinstance(item.get("block_id"), str):
                return str(item["block_id"])
        return "block.0"

    @staticmethod
    def _channel_target_block(profile: Mapping[str, Any], excluded: set[str]) -> Optional[str]:
        records = profile.get("channel_sensitivity", [])
        candidates = [
            item
            for item in records
            if isinstance(item, Mapping)
            and item.get("available") is True
            and isinstance(item.get("block_id"), str)
            and item["block_id"] not in excluded
        ]
        if not candidates:
            return None
        return str(candidates[0]["block_id"])

    @staticmethod
    def _has_quality_failure(history: Sequence[Mapping[str, Any]]) -> bool:
        for item in history:
            if not isinstance(item, Mapping):
                continue
            decision = item.get("decision")
            if not isinstance(decision, Mapping):
                continue
            promotion = decision.get("promotion")
            reasons = promotion.get("reasons", []) if isinstance(promotion, Mapping) else []
            if any("quality" in str(reason).lower() for reason in reasons):
                return True
            if "quality" in str(decision.get("failure_type", "")).lower():
                return True
        return False

    @staticmethod
    def _previously_selected_blocks(history: Sequence[Mapping[str, Any]]) -> set[str]:
        selected: set[str] = set()
        for item in history:
            decision = item.get("decision") if isinstance(item, Mapping) else None
            args = decision.get("operator_args") if isinstance(decision, Mapping) else None
            if not isinstance(args, Mapping):
                args = item.get("training", {}).get("operator_args") if isinstance(item, Mapping) and isinstance(item.get("training"), Mapping) else None
            if isinstance(args, Mapping) and isinstance(args.get("block_ids"), list):
                selected.update(str(value) for value in args["block_ids"])
            if isinstance(args, Mapping) and isinstance(args.get("steps"), list):
                for step in args["steps"]:
                    if isinstance(step, Mapping) and isinstance(step.get("args"), Mapping) and isinstance(step["args"].get("block_ids"), list):
                        selected.update(str(value) for value in step["args"]["block_ids"])
        return selected

    def _explore_retention_ratio(self, history: Sequence[Mapping[str, Any]]) -> float:
        """Tighten exploration after Memory records a model-size miss."""

        ratio = float(self.explore_ffn_retention_ratio)
        for item in history:
            evaluation = item.get("evaluation") if isinstance(item, Mapping) else None
            if not isinstance(evaluation, Mapping):
                continue
            hardware = evaluation.get("hardware_metrics")
            baseline = evaluation.get("baseline_hardware_metrics")
            if not isinstance(hardware, Mapping) or not isinstance(baseline, Mapping):
                continue
            try:
                before = float(baseline["model_size_gb"])
                after = float(hardware["model_size_gb"])
            except (KeyError, TypeError, ValueError):
                continue
            if before > 0.0 and (before - after) / before < 0.10:
                ratio = max(0.10, ratio * 0.75)
        return ratio


def _history_ids(history: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        str(item.get("experience_id", item.get("round_id")))
        for item in history
        if isinstance(item, Mapping) and (item.get("experience_id") or item.get("round_id"))
    )[:16]


def _feedback_reasons(history: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    values: list[str] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        decision = item.get("decision")
        if isinstance(decision, Mapping):
            promotion = decision.get("promotion")
            reasons = promotion.get("reasons", []) if isinstance(promotion, Mapping) else []
            if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
                values.extend(str(reason) for reason in reasons if str(reason).strip())
            if decision.get("failure_type"):
                values.append(str(decision["failure_type"]))
        if item.get("failure_reason"):
            values.append(str(item["failure_reason"]))
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result[:16])


def _deterministic_improvement_source(
    algorithm_id: str,
    parent: AlgorithmCandidate,
    profile: Mapping[str, Any],
    failures: Sequence[str],
) -> tuple[str, str, str, str]:
    """Produce a source-level revision, not an unrelated fresh seed."""

    failure_text = " ".join(str(item).lower() for item in failures)
    if "tica_prune" in parent.source:
        formula = "late_log" if any(token in failure_text for token in ("quality", "regression", "interaction")) else "uniform"
        # A recursive child must revise the parent's actual rule.  Toggle the
        # weighting formula when the previous improvement already selected the
        # deterministic default, instead of emitting the same logic again.
        if not any(token in failure_text for token in ("quality", "regression", "interaction")):
            if "'time_weight_formula': 'uniform'" in parent.source:
                formula = "late_log"
            elif "'time_weight_formula': 'late_log'" in parent.source:
                formula = "uniform"
        source = (
            "import math\n\n"
            "def propose(profile, history):\n"
            "    # Improve %s: revise TICA's timestep weighting after measured feedback; keep the trusted structural closure.\n"
            "    tica = profile.get('tica', {})\n"
            "    names = [str(item) for item in tica.get('timesteps', ['0.9', '0.5', '0.1'])]\n"
            "    values = [float(item) for item in names]\n"
            "    raw = [math.log1p(1.0 + 9.0 * (1.0 - value)) for value in values]\n"
            "    if %r == 'uniform':\n"
            "        raw = [1.0 for value in values]\n"
            "    total = sum(raw)\n"
            "    weights = [value / total for value in raw]\n"
            "    ranked = []\n"
            "    for item in profile.get('tica_block_sensitivity', []):\n"
            "        per_timestep = item.get('per_timestep', {})\n"
            "        risk = sum(weights[index] * float(per_timestep.get(names[index], 0.0)) for index in range(len(names)))\n"
            "        ranked.append((risk, str(item.get('block_id'))))\n"
            "    ranked = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=bool(%d))\n"
            "    selected = ranked[0][1] if ranked else str(profile.get('layers', [{'block_id': 'block.0'}])[0]['block_id'])\n"
            "    width = int(profile.get('model', {}).get('ffn_width', 1))\n"
            "    target = max(1, min(width - 1, int(width * 0.75)))\n"
            "    return {'operator': 'tica_prune', 'args': {'block_ids': [selected], 'retained_indices': list(range(target)), 'mode': 'block_plus_ffn', 'time_weight_formula': %r, 'risk_limit': 0.25, 'fast_risk_limit': 0.50, 'activation_weight': 0.5, 'interaction_weight': 0.5, 'beam_width': 2, 'max_fast_evals': 8, 'max_full_evals': 4, 'max_interaction_evals': 2, 'strategy': 'greedy_beam', 'max_search_depth': 3}}\n"
        ) % (parent.algorithm_id, formula, int(str(algorithm_id)[1:]) % 2, formula)
        return (
            source,
            "change TICA timestep weighting while preserving globally aligned Block+FFN execution",
            "measured TICA feedback indicates the parent weighting rule is the actionable design variable",
            "reweight denoising timesteps and keep the same trusted structural mutation for a fair algorithm comparison",
        )
    quality_feedback = "quality" in failure_text or "integrity" in failure_text
    source = (
        "def propose(profile, history):\n"
        "    # Improve %s: revise the Block subset from current sensitivity and measured Block cost.\n"
        "    layers = {str(item['block_id']): item for item in profile.get('layers', []) if isinstance(item, dict) and item.get('block_id')}\n"
        "    scores = {}\n"
        "    rows = profile.get('tica_block_sensitivity') or profile.get('block_sensitivity', [])\n"
        "    for item in rows:\n"
        "        if not isinstance(item, dict) or item.get('available') is False or str(item.get('block_id', '')) not in layers:\n"
        "            continue\n"
        "        values = item.get('per_timestep') or item.get('timesteps') or {}\n"
        "        values = [float(value) for value in values.values() if isinstance(value, (int, float))] if isinstance(values, dict) else []\n"
        "        score = sum(values) / len(values) if values else item.get('importance_score')\n"
        "        if isinstance(score, (int, float)) and score >= 0:\n"
        "            scores[str(item['block_id'])] = float(score)\n"
        "    costs = {key: max(0.0, float(value.get('latency_ms', value.get('estimated_flops', 0.0)) or 0.0)) for key, value in layers.items()}\n"
        "    max_cost = max(costs.values()) if costs else 0.0\n"
        "    ranked = sorted(scores, key=lambda key: (scores[key] / max(costs.get(key, 0.0) / max(max_cost, 1.0), 1e-6), scores[key], -costs.get(key, 0.0), key))\n"
        "    excluded = set()\n"
        "    if %r:\n"
        "        for record in history:\n"
        "            decision = record.get('decision', {}) if isinstance(record, dict) else {}\n"
        "            args = decision.get('operator_args', {}) if isinstance(decision, dict) else {}\n"
        "            if isinstance(args, dict) and isinstance(args.get('block_ids'), list):\n"
        "                excluded.update(str(value) for value in args['block_ids'])\n"
        "    remaining = [key for key in ranked if key not in excluded] or ranked\n"
        "    if not remaining:\n"
        "        raise ValueError('block_improvement_requires_available_sensitivity')\n"
        "    return {'operator': 'prune_blocks', 'args': {'block_ids': [remaining[0]]}}\n"
    ) % (parent.algorithm_id, quality_feedback)
    inherited_search_name = "_harness_parent_search_" + str(algorithm_id).lstrip("A")
    inherited_source = parent.source.replace("def propose(", "def _parent_propose(", 1)
    inherited_source = inherited_source.replace("def search(", "def " + inherited_search_name + "(", 1)
    if "def " + inherited_search_name + "(" not in inherited_source:
        inherited_source += "\n\n" + _concise_multigranular_search_program_suffix(algorithm_id).replace("def search(", "def " + inherited_search_name + "(", 1).lstrip()
    if inherited_source:
        source = inherited_source + "\n\n" + source + "\n\n" + '''def search(profile, history, state, search_space):
    # Re-rank the inherited candidate generator from measured family evidence.
    result = INHERITED_SEARCH(profile, history, state, search_space)
    candidates = result.get('candidates', [])
    old_rank = result.get('rank', list(range(len(candidates))))
    rank_position = {candidate_index: position for position, candidate_index in enumerate(old_rank)}
    family_by_operator = {
        'prune_blocks': 'structure', 'prune_ffn_channels': 'channel', 'head_gating': 'head',
        'prune_attention_heads': 'head', 'prune_heads': 'head', 'prune_tokens': 'token',
        'token_merge': 'token', 'kv_token_select': 'token', 'sparse_runtime': 'attention',
        'attention_window': 'attention', 'sampling_step_skip': 'sampling',
        'timestep_block_gate': 'timestep', 'timestep_attention_policy': 'timestep', 'compose': 'hybrid',
    }
    family_metrics = {}
    rows = state.get('measured_candidates', []) if isinstance(state, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        plan = row.get('plan', {})
        family = row.get('family') or family_by_operator.get(plan.get('operator')) if isinstance(plan, dict) else row.get('family')
        if not family:
            continue
        values = family_metrics.setdefault(str(family), [0.0, 0.0, 0.0, 0])
        quality = row.get('quality_score')
        efficiency = row.get('efficiency_reduction_ratio')
        risk = row.get('risk_value')
        quality = float(quality) if isinstance(quality, (int, float)) else (0.0 if row.get('valid') is False else 1.0)
        efficiency = float(efficiency) if isinstance(efficiency, (int, float)) else 0.0
        risk = float(risk) if isinstance(risk, (int, float)) else max(0.0, 1.0 - quality)
        values[0] += quality; values[1] += efficiency; values[2] += risk; values[3] += 1
    def evidence_score(index):
        plan = candidates[index]
        family = family_by_operator.get(plan.get('operator'), 'other') if isinstance(plan, dict) else 'other'
        measured = family_metrics.get(family)
        if not measured:
            return 0.0
        count = max(1, measured[3])
        quality = measured[0] / count
        efficiency = measured[1] / count
        risk = measured[2] / count
        return (2.0 * quality if QUALITY_FIRST else quality) + (0.5 * efficiency if QUALITY_FIRST else 2.0 * efficiency) - (risk if QUALITY_FIRST else 0.25 * risk)
    order = sorted(range(len(candidates)), key=lambda index: (-evidence_score(index), rank_position.get(index, len(candidates)), index))
    return {**result, 'rank': order, 'diagnostics': {**result.get('diagnostics', {}), 'measured_family_scores': {name: [round(value / max(1, count), 6) for value, count in zip(values[:3], [values[3]] * 3)] for name, values in family_metrics.items()}, 'quality_first': QUALITY_FIRST}}
'''.replace('QUALITY_FIRST', 'True' if quality_feedback else 'False').replace('INHERITED_SEARCH', inherited_search_name)
    if quality_feedback:
        return (
            source,
            "adjust family ranking from measured quality and risk evidence",
            "measured quality failures elevate quality and risk above expected efficiency in candidate ranking",
            "inherit candidate generation and score each registered compression family using its measured quality, efficiency, and risk",
        )
    return (
        source,
        "adjust family ranking from measured quality, efficiency, and risk evidence",
        "the parent ranking does not account for measured family-level quality and efficiency outcomes",
        "inherit candidate generation and adapt the ranking objective to measured family-level quality, efficiency, and risk",
    )


def _compact_execution_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    execution = profile.get("execution")
    if not isinstance(execution, Mapping) or execution.get("available") is not True:
        return {}
    compact: dict[str, Any] = {
        "execution_schema_version": execution.get("execution_schema_version"),
        "offline_simulation": execution.get("offline_simulation"),
        "device": execution.get("device"),
        "gpu_name": execution.get("gpu_name"),
        "dtype": execution.get("dtype"),
        "checkpoint_sha256": execution.get("checkpoint_sha256"),
        "diagnostic_method": execution.get("diagnostic_method"),
        "cases": [],
    }
    for case in execution.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        diagnostic = case.get("diagnostic_pass")
        diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
        modules = [item for item in diagnostic.get("module_timings", ()) if isinstance(item, Mapping)]
        blocks = [item for item in modules if item.get("category") == "transformer_block"]
        blocks.sort(key=lambda item: float(item.get("cuda_time_ms_total", 0.0) or 0.0), reverse=True)
        trace = case.get("kernel_trace")
        trace = trace if isinstance(trace, Mapping) else {}
        operators = [
            item for item in trace.get("operator_summaries", ())
            if isinstance(item, Mapping) and str(item.get("operator", "")) not in {"ProfilerStep*", "Command Buffer Full"}
        ]
        steps = [item for item in diagnostic.get("denoising_step_timings", ()) if isinstance(item, Mapping)]
        compact["cases"].append({
            "variant": case.get("variant"),
            "recipe": case.get("recipe"),
            "warmup": case.get("warmup"),
            "repeats": case.get("repeats"),
            "latency_median_ms": case.get("latency_median_ms"),
            "memory": case.get("memory"),
            "inclusive_category_cuda_time_ms_total": diagnostic.get("category_cuda_time_ms_total"),
            "inclusive_category_cuda_time_ms_by_step": diagnostic.get("category_cuda_time_ms_by_step"),
            "transformer_step_cuda_time_ms": next((item.get("cuda_time_ms_by_step") for item in modules if item.get("category") == "transformer_forward"), {}),
            "top_transformer_blocks": [
                {key: item[key] for key in ("block_id", "cuda_time_ms_total", "cuda_time_ms_by_step") if key in item}
                for item in blocks[:8]
            ],
            "denoising_steps": [
                {key: item[key] for key in ("step_index", "timestep", "wall_time_ms", "peak_allocated_bytes", "peak_reserved_bytes") if key in item}
                for item in steps
            ],
            "attention_head_execution": case.get("head_gating_execution"),
            "step_kernel_operators": [
                {key: item[key] for key in ("operator", "category", "calls", "self_cuda_time_us", "self_device_memory_bytes") if key in item}
                for item in operators[:8]
            ],
        })
    return compact


def _compact_profile(profile: Mapping[str, Any], *, block_only: bool = False) -> dict[str, Any]:
    """Keep provider prompts bounded while retaining compression evidence."""

    compact: dict[str, Any] = {}
    for key in ("model", "provenance", "schema_version", "timestep_sensitivity", "sampling_step_sensitivity", "token_grid"):
        if key in profile:
            compact[key] = profile[key]
    execution = _compact_execution_profile(profile)
    if execution:
        compact["execution"] = execution
    compact["layers"] = [
        {
            field: item[field]
            for field in ("block_id", "index", "parameter_count", "estimated_flops", "latency_ms", "memory_delta_bytes")
            if field in item
        }
        for item in profile.get("layers", [])
        if isinstance(item, Mapping)
    ]
    compact["block_sensitivity"] = [
        {field: item[field] for field in ("block_id", "importance_score", "available", "method") if field in item}
        for item in profile.get("block_sensitivity", [])
        if isinstance(item, Mapping)
    ]
    compact["block_timestep_sensitivity"] = [
        {
            field: item[field]
            for field in ("block_id", "importance_score", "per_timestep", "best_bucket", "worst_bucket", "available", "method")
            if field in item
        }
        for item in profile.get("block_timestep_sensitivity", [])
        if isinstance(item, Mapping)
    ]
    if block_only:
        return compact
    channel_records = []
    for item in profile.get("channel_sensitivity", []):
        if not isinstance(item, Mapping):
            continue
        record = {field: item[field] for field in ("block_id", "width", "available", "method") if field in item}
        channels = [channel for channel in item.get("channels", []) if isinstance(channel, Mapping)]
        record["channel_count"] = len(channels)
        record["channels"] = sorted(
            (
                {field: channel[field] for field in ("channel", "importance_score", "available") if field in channel}
                for channel in channels
            ),
            key=lambda channel: float(channel.get("importance_score", 0.0)),
        )[:4]
        channel_records.append(record)
    compact["channel_sensitivity"] = channel_records
    def local_extremes(rows: Any, per_block: int) -> list[dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for item in rows if isinstance(rows, (list, tuple)) else ():
            if isinstance(item, Mapping) and item.get("block_id") is not None:
                grouped.setdefault(str(item["block_id"]), []).append(item)
        output = []
        for block_id, items in grouped.items():
            usable = [item for item in items if isinstance(item.get("importance_score"), (int, float))]
            usable.sort(key=lambda item: (float(item["importance_score"]), int(item.get("head", 0))))
            chosen = usable[:per_block]
            output.extend(
                {field: item[field] for field in ("block_id", "attention", "head", "importance_score", "available", "method") if field in item}
                for item in chosen
            )
        return output

    compact["head_sensitivity"] = local_extremes(profile.get("head_sensitivity", []), 4)
    compact["attention_sensitivity"] = local_extremes(profile.get("attention_sensitivity", []), 2)
    compact["token_sensitivity"] = [
            {field: item[field] for field in ("keep_ratio", "token_count", "selected_count", "grid_shape", "layout", "importance_score", "available", "method") if field in item}
        for item in profile.get("token_sensitivity", [])
        if isinstance(item, Mapping)
    ]
    return compact


def _compact_search_space(search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Expose active operator schemas without leaking implementation source."""

    compact: dict[str, Any] = {
        key: search_space[key]
        for key in ("version_id", "parent_version_id", "constraints")
        if key in search_space
    }
    compact["operators"] = [
        {
            field: item[field]
            for field in (
                "name",
                "version",
                "argument_schema",
                "granularity",
                "execution_domain",
                "constraints",
                "limitations",
            )
            if field in item
        }
        for item in search_space.get("operators", [])
        if isinstance(item, Mapping)
    ]
    return compact


def _block_only_search_space(search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Present only the Block primitive to standard Agent generation."""

    compact = dict(search_space)
    compact["operators"] = [
        dict(item)
        for item in search_space.get("operators", ())
        if isinstance(item, Mapping) and str(item.get("name", "")) == "prune_blocks"
    ]
    return compact


def _compact_pareto_front(front: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose measured front evidence without allowing an unbounded prompt."""

    compact: list[dict[str, Any]] = []
    for item in front:
        if not isinstance(item, Mapping):
            continue
        record: dict[str, Any] = {}
        for field in ("candidate_id", "quality_score", "search_score", "evaluation_status", "valid"):
            if field in item:
                record[field] = item[field]
        hardware = item.get("hardware_metrics")
        if isinstance(hardware, Mapping):
            record["hardware_metrics"] = {
                field: hardware[field]
                for field in ("latency_s", "inference_wall_time_s_mean", "peak_memory_gb", "model_size_gb", "parameter_count")
                if field in hardware
            }
        compact.append(record)
    return compact


def _compact_memory_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Keep retrieved evidence small enough for a bounded controller prompt."""

    compact: dict[str, Any] = {}
    for field in ("experience_id", "round_id", "memory_kind", "architecture", "operator", "status", "reward", "failure_reason"):
        if field in item:
            value = item[field]
            compact[field] = value[:240] if isinstance(value, str) else value
    for field in ("algorithm_id", "algorithm_parent_id", "change_summary", "change_rationale"):
        if field in item:
            value = item[field]
            compact[field] = value[:240] if isinstance(value, str) else value
    if item.get("memory_kind") == "algorithm_skill" and isinstance(item.get("algorithm_source"), str):
        # The complete immutable source remains in AlgorithmStore.  This
        # bounded copy lets a provider reuse the skill without allowing
        # unbounded historical sources to overflow its prompt.
        compact["algorithm_source"] = item["algorithm_source"][:6000]

    if item.get("memory_kind") == "algorithm_search_feedback":
        for field in (
            "parent_checkpoint_sha256",
            "quality_floor_ratio",
            "quality_floor_basis",
            "search_outcome",
            "candidate_count",
            "measured_candidate_summary_count",
            "quality_feasible_candidate_count",
            "measured_family_counts",
            "pareto_front_plan_digests",
            "selection_status",
        ):
            if field in item:
                compact[field] = item[field]

        tested_sets = item.get("tested_block_sets")
        if isinstance(tested_sets, list):
            compact["tested_block_sets"] = [
                [str(block_id) for block_id in block_ids[:64]]
                for block_ids in tested_sets[:16]
                if isinstance(block_ids, list)
            ]

        feedback_candidates = []
        for candidate in item.get("measured_candidates", ()):
            if not isinstance(candidate, Mapping):
                continue
            plan = candidate.get("plan")
            if not isinstance(plan, Mapping) or not isinstance(plan.get("args"), Mapping):
                continue
            record = {
                "plan": dict(plan),
                "family": candidate.get("family"),
                "candidate_index": candidate.get("rank_index"),
                "rank_position": candidate.get("rank_position"),
                "evaluation_position": candidate.get("evaluation_position"),
                "valid": candidate.get("valid"),
                "quality_score": candidate.get("quality_score"),
                "quality_floor_ratio": candidate.get("quality_floor_ratio", item.get("quality_floor_ratio")),
                "quality_feasible": candidate.get("quality_feasible"),
                "efficiency_reduction_ratio": candidate.get("efficiency_reduction_ratio"),
                "search_score": candidate.get("search_score"),
                "risk_value": candidate.get("risk_value"),
                "video_quality_evaluated": candidate.get("video_quality_evaluated"),
            }
            for field in ("plan_digest", "failure_reason", "recovery_action", "runtime_execution"):
                if candidate.get(field) is not None:
                    record[field] = candidate[field]
            feedback_candidates.append(record)
            if len(feedback_candidates) >= 8:
                break
        if feedback_candidates:
            compact["measured_candidates"] = feedback_candidates

    if item.get("memory_kind") == "algorithm_duplicate_feedback" and item.get("duplicate_algorithm_id"):
        for field in (
            "duplicate_algorithm_id",
            "candidate_source_sha256",
            "duplicate_algorithm_source_sha256",
            "duplicate_attempt",
        ):
            if field in item:
                compact[field] = item[field]

    def compact_args(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        result: dict[str, Any] = {}
        for field in ("block_ids", "retained_indices"):
            values = value.get(field)
            if isinstance(values, list):
                result[field] = list(values[:64])
        return result

    training = item.get("training")
    if isinstance(training, Mapping):
        summary = {}
        for field in ("architecture", "operator", "algorithm_id", "algorithm_parent_id", "hypothesis", "design_strategy", "design_fingerprint", "predicted_gain"):
            if field in training:
                value = training[field]
                summary[field] = value[:240] if isinstance(value, str) else value
        args = compact_args(training.get("operator_args"))
        if args:
            summary["operator_args"] = args
        if summary:
            compact["training"] = summary

    evaluation = item.get("evaluation")
    if isinstance(evaluation, Mapping):
        summary = {}
        for field in ("quality_score", "search_score", "evaluation_status", "valid", "feasible", "source_revision"):
            if field in evaluation:
                summary[field] = evaluation[field]
        quality = evaluation.get("quality_metrics")
        if isinstance(quality, Mapping):
            summary["quality_metrics"] = {field: quality[field] for field in ("semantic", "temporal", "motion", "score_type") if field in quality}
        hardware = evaluation.get("hardware_metrics")
        if isinstance(hardware, Mapping):
            summary["hardware_metrics"] = {
                field: hardware[field]
                for field in ("latency_s", "inference_wall_time_s_mean", "peak_memory_gb", "model_size_gb", "parameter_count")
                if field in hardware
            }
        if summary:
            compact["evaluation"] = summary

    decision = item.get("decision")
    if isinstance(decision, Mapping):
        summary = {field: decision[field] for field in ("status", "failure_type", "stage") if field in decision}
        args = compact_args(decision.get("operator_args"))
        if args:
            summary["operator_args"] = args
        promotion = decision.get("promotion")
        if isinstance(promotion, Mapping):
            summary["promotion"] = {
                field: promotion[field]
                for field in ("accepted", "improvements", "objective_passes", "objective_failures", "missing_metrics", "reasons")
                if field in promotion
            }
        if summary:
            compact["decision"] = summary

    provenance = item.get("provenance")
    if isinstance(provenance, Mapping):
        compact["provenance"] = {field: provenance[field] for field in ("architecture", "offline_simulation") if field in provenance}
    return compact


def _compact_context_items(items: Sequence[Mapping[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    """Deduplicate and compact Memory items without changing stored evidence."""

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        key = (str(item.get("memory_kind", "")), str(item.get("experience_id", "")))
        if key != ("", "") and key in seen:
            continue
        if key != ("", ""):
            seen.add(key)
        result.append(_compact_memory_item(item))
        if len(result) >= limit:
            break
    return result


def build_algorithm_provider(
    provider_name: str,
    model_name: str,
    *,
    endpoint: str,
    api_key_env: str = "OPENAI_API_KEY",
    timeout_s: float = 180.0,
    protocol_mode: str = "standard",
    model_version: str = "",
) -> Provider:
    """Build an LLM callable that returns only the bounded algorithm fields."""

    provider = str(provider_name).strip().lower().replace("-", "_")
    if provider in {"deterministic", "offline", "mock"}:
        raise ValueError("deterministic provider does not need an HTTP callable")
    if not str(model_name).strip() or timeout_s <= 0:
        raise ValueError("algorithm provider model and timeout are required")
    if provider == "ollama":
        url = endpoint.rstrip("/")
        url = url if url.endswith("/api/chat") else url + "/api/chat"
    elif provider in {"vllm", "openai_compatible", "openai"}:
        url = endpoint
        if provider in {"vllm", "openai_compatible"} and not url.rstrip("/").endswith("/chat/completions"):
            url = url.rstrip("/") + "/v1/chat/completions"
    else:
        raise ValueError("unsupported_algorithm_provider: %s" % provider_name)

    def call(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        task = str(payload.get("task", "generate_algorithm"))
        active_protocol = str(payload.get("protocol_mode", protocol_mode)).strip().lower() or "standard"
        if active_protocol not in {"standard", "tica"}:
            raise ValueError("unsupported_algorithm_protocol_mode: %s" % active_protocol)
        if task == "improve_existing_algorithm_source":
            instruction = (
                "Improve the parent algorithm using its source and the supplied failure/history. "
                "Return a complete, concise replacement source under 7000 characters, not a patch or pseudocode; "
                "you may replace verbose scaffolding instead of copying it. Change the actual "
                "candidate-generation, importance-scoring, interaction, or budget-allocation logic "
                "using observed failures. Use per-Block, per-Head, token, attention and timestep "
                "evidence where available. Do not repeat a measured plan for the same checkpoint. "
                "Prefer localized low-sensitivity selections and moderate complementary policies "
                "over whole-layer removal or uniform ratios. Treat under-floor quality as a near miss. "
                "Use runtime counters to identify execution or fallback, but do not claim GPU speedup "
                "without independent latency evidence. Explain the source-level change and the evidence "
                "that motivates it. A comment-only or ordering-only change is not an improvement. "
            )
            response_schema = IMPROVEMENT_RESPONSE_SCHEMA
        elif task == "propose_search_space_extension":
            instruction = (
                "Analyze the current compression search space and propose one capability it "
                "cannot express. Return a declarative OperatorSpec proposal only; do not return "
                "imports, shell commands, or an untrusted factory. If using the "
                "generated_channel_rule factory, implementation_source may contain one complete "
                "bounded propose(profile, history) function that delegates to a base operator; "
                "the trusted validator will run it in the Sandbox. "
            )
            response_schema = SEARCH_SPACE_RESPONSE_SCHEMA
        else:
            instruction = "Generate one concise compression algorithm for a trusted Wan DiT harness. "
            response_schema = ALGORITHM_RESPONSE_SCHEMA
        algorithm_rules = (
            "Return JSON only with source, hypothesis, and parent_id. The source must define "
            "a complete Python function `def propose(profile, history): ...` and return only "
            "a bounded operator/args plan; keep the source concise, never return a function "
            "name or pseudocode. Also define `search(profile, history, state, search_space)` "
            "to generate and rank bounded candidates from the active operator registry; "
            "`search_space['operators']` is the source of allowed operations. "
            "Use per-Block/per-timestep and per-Head sensitivity, token evidence, attention "
            "regions, measured cost, past family coverage and quality feedback. Prefer localized "
            "Block/Head targets, profile-supported timestep gates, moderate KV selection, and "
            "complementary Sparse Runtime combinations when their operators are registered. "
            "Never assume a combination will win; omit unsupported or unmeasured targets with a reason. "
            "Avoid Wan structural Head Pruning when the architecture cannot reload projection dimensions. "
            "The value "
            "returned by `propose` must be one dictionary, never a list, and have exactly "
            "`operator` and `args`. Choose only IDs present in the profile and obey each "
            "registered argument schema. Respect `state['budget']['max_candidates']`. Use "
            "`algorithm_search_feedback.measured_family_counts` to allocate exploration across "
            "families and avoid repeats against the same parent. Probe quality is calibration "
            "screening; independent multi-case video validation decides model promotion. "
            "Never mark an under-floor candidate feasible. Return exactly a mapping with candidates, rank, stop, and "
            "optional diagnostics. Do not train or fine-tune weights. "
            "Only import math, statistics, json, or collections; never import or use random. "
            "The returned parent_id must exactly match the supplied parent_id. "
            "Allowed plans are exactly the registered operators and their declared argument schemas. "
            "Never output paths, devices, shell commands, imports outside the sandbox allowlist, "
            "or evaluator settings.\n\n"
        )
        tica_rules = (
            "This is TICA-Prune protocol mode. Return an algorithm source whose propose(profile, history) "
            "returns exactly {'operator': 'tica_prune', 'args': {...}}. This is a bounded search rule and "
            "initial suggestion for the trusted TICAPruner; it is not a final model mutation and must not "
            "execute model, CUDA, filesystem, or arbitrary imports. Allowed TICA args are block_ids, "
            "retained_indices, mode in {'block','ffn','block_plus_ffn'}, time_weight_formula in "
            "{'uniform','early_log','late_log'}, risk_limit, fast_risk_limit, activation_weight, "
            "interaction_weight, beam_width, max_fast_evals, max_full_evals, max_interaction_evals, "
            "strategy in {'greedy','greedy_beam'}, and max_search_depth. Numeric values must stay within "
            "the supplied search-space bounds. The trusted searcher computes multi-step candidates, runs "
            "full normalized-MSE checks, and owns Pareto selection. You may change timestep weighting, "
            "scoring weights, interaction handling, ranking, and bounded search controls. Do not invent a "
            "new compression primitive; unsupported capability must be proposed through SearchSpaceManager. "
            "The returned parent_id must exactly match the supplied parent_id.\n\n"
        )
        search_space_rules = (
            "Return JSON only with spec, rationale, and evidence_ids. The spec must describe "
            "one new declarative capability not already present in search_space. Do not include "
            "imports, shell commands, paths, devices, evaluator settings, or a factory "
            "implementation. A generated_channel_rule may include bounded implementation_source "
            "only; it must delegate to prune_ffn_channels and will be sandboxed.\n\n"
        )
        if task == "propose_search_space_extension":
            protocol_rules = search_space_rules
        elif active_protocol == "tica":
            protocol_rules = tica_rules
        else:
            protocol_rules = algorithm_rules
        prompt = instruction + protocol_rules
        prompt += json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if provider in {"openai", "openai_compatible", "vllm"} and api_key_env:
            token = os.environ.get(api_key_env, "").strip()
            if token:
                headers["Authorization"] = "Bearer " + token
        if provider == "ollama":
            request_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "format": response_schema,
                "options": {"temperature": 0.2},
            }
        elif provider in {"vllm", "openai_compatible"}:
            request_payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Return JSON only; do not call tools or output unsafe execution instructions."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "stream": False,
                # vLLM deployments commonly support JSON mode but not the
                # OpenAI json_schema extension.  The local sandbox remains
                # the authoritative validator for the returned source.
                "response_format": {"type": "json_object"},
                # Source improvement returns the complete parent program.
                "max_tokens": 3072 if task == "improve_existing_algorithm_source" else 1536,
            }
        else:
            request_payload = {
                "model": model_name,
                "instructions": "Return JSON only; do not call tools or output unsafe execution instructions.",
                "input": prompt,
                "store": False,
                "text": {"format": {"type": "json_schema", "name": "evolution_request", "strict": True, "schema": response_schema}},
            }
        request = urllib.request.Request(url, data=json.dumps(request_payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                detail = ""
            suffix = ": " + detail if detail else ""
            raise RuntimeError("algorithm_provider_request_failed: %s%s" % (exc, suffix)) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError("algorithm_provider_request_failed: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm_provider_response_invalid: response must be an object")
        raw_choices = raw.get("choices")
        choice_meta = raw_choices[0] if isinstance(raw_choices, list) and raw_choices and isinstance(raw_choices[0], Mapping) else {}
        call.last_response_metadata = {
            "provider": provider,
            "requested_model": str(model_name),
            "served_model": str(raw.get("model", "")) or None,
            "configured_model_version": str(model_version) or None,
            "api_response_id": str(raw.get("id", "")) or None,
            "api_created": raw.get("created"),
            "system_fingerprint": raw.get("system_fingerprint"),
            "usage": dict(raw.get("usage", {})) if isinstance(raw.get("usage"), Mapping) else None,
            "finish_reason": choice_meta.get("finish_reason"),
        }
        if provider == "ollama":
            content = raw.get("message", {}).get("content") if isinstance(raw.get("message"), Mapping) else None
        elif provider in {"vllm", "openai_compatible"}:
            choices = raw.get("choices")
            content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else None
        else:
            content = raw.get("output_text")
            if not content:
                for item in raw.get("output", []):
                    for part in item.get("content", []) if isinstance(item, Mapping) else []:
                        if isinstance(part, Mapping) and part.get("type") == "output_text":
                            content = part.get("text")
                            break
                    if content:
                        break
        if not isinstance(content, str):
            raise ValueError("algorithm_provider_response_invalid: no JSON content")
        parsed = json.loads(content)
        if not isinstance(parsed, Mapping):
            raise ValueError("algorithm_provider_response_invalid: content must be an object")
        normalized = dict(parsed)
        # JSON mode does not enforce the response schema.  Treat an omitted
        # or null parent as the trusted request parent, but never accept a
        # different non-null parent.
        if normalized.get("parent_id") is None and payload.get("parent_id") is not None:
            normalized["parent_id"] = payload["parent_id"]
        return normalized

    call.last_response_metadata = {}
    call.provider_name = provider
    call.requested_model = str(model_name)
    return call
