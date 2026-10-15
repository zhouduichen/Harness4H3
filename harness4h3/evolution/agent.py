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
            "required": ["name", "version", "argument_schema", "granularity", "trusted_factory_key"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "argument_schema": {"type": "object"},
                "granularity": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "trusted_factory_key": {"type": "string"},
                "implementation_source": {"type": "string"},
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
    if search_choice == "explore" and channel_block_id is not None:
        return (
            "def propose(profile, history):\n"
            "    # Explore seed %s: compose profile-guided block and FFN pruning; channel evidence target=%r.\n"
            "    width = int(profile['model']['ffn_width'])\n"
            "    retained_indices = list(range(max(1, int(width * %.6f))))\n"
            "    remaining_block_ids = [item['block_id'] for item in profile.get('layers', []) if item.get('block_id') != %r]\n"
            "    return {'operator': 'compose', 'args': {'steps': [\n"
            "        {'operator': 'prune_blocks', 'args': {'block_ids': %r}},\n"
            "        {'operator': 'prune_ffn_channels', 'args': {'block_ids': remaining_block_ids, 'retained_indices': retained_indices}},\n"
            "    ]}}\n"
        ) % (algorithm_id, channel_block_id, retention_ratio, block_id, [block_id])
    if search_choice == "explore":
        return (
            "def propose(profile, history):\n"
            "    # Explore seed %s: a paired FFN-width hypothesis across all profiled blocks.\n"
            "    block_ids = [item['block_id'] for item in profile.get('layers', [])]\n"
            "    width = int(profile['model']['ffn_width'])\n"
            "    retained_indices = list(range(max(1, int(width * %.6f))))\n"
            "    return {'operator': 'prune_ffn_channels', 'args': {'block_ids': block_ids, 'retained_indices': retained_indices}}\n"
        ) % (algorithm_id, retention_ratio)
    return (
        "def propose(profile, history):\n"
        "    # Seed algorithm %s selects the least output-sensitive block reported by the profiler.\n"
        "    selected = %r\n"
        "    return {'operator': 'prune_blocks', 'args': {'block_ids': [selected]}}\n" % (algorithm_id, block_id)
    )


def _search_program_suffix(algorithm_id: str, *, exploration: bool = False) -> str:
    """Add source-controlled candidate generation to a legacy seed source."""

    mode = "explore" if exploration else "exploit"
    return (
        "\n\ndef search(profile, history, state, search_space):\n"
        "    # Algorithm %s controls scoring, candidate generation, order, and stop.\n"
        "    # mode=%s; this code is executed once per search depth.\n"
        "    records = []\n"
        "    for item in profile.get('tica_block_sensitivity', profile.get('block_sensitivity', [])):\n"
        "        if isinstance(item, dict) and item.get('block_id'):\n"
        "            values = item.get('per_timestep', item.get('timesteps', {}))\n"
        "            score = sum(float(value) for value in values.values()) if isinstance(values, dict) else float(item.get('importance_score', 0.0) or 0.0)\n"
        "            records.append((score, str(item['block_id'])))\n"
        "    if not records:\n"
        "        records = [(0.0, str(item.get('block_id'))) for item in profile.get('layers', []) if isinstance(item, dict) and item.get('block_id')]\n"
        "    ordered = [item[1] for item in sorted(records)]\n"
        "    candidates = []\n"
        "    def add(candidate):\n"
        "        if candidate not in candidates:\n"
        "            candidates.append(candidate)\n"
        "    proposed = state.get('proposed_plan') if isinstance(state, dict) else None\n"
        "    if isinstance(proposed, dict) and proposed.get('operator') and isinstance(proposed.get('args'), dict):\n"
        "        add(proposed)\n"
        "    if ordered:\n"
        "        add({'operator': 'prune_blocks', 'args': {'block_ids': [ordered[0]]}})\n"
        "    if len(ordered) > 1:\n"
        "        add({'operator': 'prune_blocks', 'args': {'block_ids': [ordered[1]]}})\n"
        "    names = [str(item.get('name')) for item in search_space.get('operators', []) if isinstance(item, dict)]\n"
        "    if 'timestep_block_gate' in names and ordered:\n"
        "        add({'operator': 'timestep_block_gate', 'args': {'rules': [{'block_id': ordered[0], 'min_timestep': 0.0, 'max_timestep': 0.3}], 'policy': 'skip_when_in_range'}})\n"
        "    head_rows = [item for item in profile.get('head_sensitivity', []) if isinstance(item, dict) and item.get('head') is not None]\n"
        "    if 'prune_attention_heads' in names and head_rows:\n"
        "        add({'operator': 'prune_attention_heads', 'args': {'head_indices': [int(sorted(head_rows, key=lambda item: float(item.get('importance_score', 0.0) or 0.0))[0]['head'])]}})\n"
        "    if 'prune_tokens' in names:\n"
        "        add({'operator': 'prune_tokens', 'args': {'keep_ratio': 0.75}})\n"
        "    if 'sparse_runtime' in names:\n"
        "        add({'operator': 'sparse_runtime', 'args': {'keep_ratio': 0.75, 'attention': 'attn1'}})\n"
        "    if 'sampling_step_skip' in names:\n"
        "        add({'operator': 'sampling_step_skip', 'args': {'target_steps': 8}})\n"
        "    if ordered and 'prune_attention_heads' in names and 'prune_tokens' in names and 'sparse_runtime' in names:\n"
        "        add({'operator': 'compose', 'args': {'steps': [\n"
        "            {'operator': 'prune_blocks', 'args': {'block_ids': [ordered[0]]}},\n"
        "            {'operator': 'prune_attention_heads', 'args': {'head_indices': [0]}},\n"
        "            {'operator': 'prune_tokens', 'args': {'keep_ratio': 0.75}},\n"
        "        ]}})\n"
        "    if not candidates:\n"
        "        fallback = 'block.' + str(int(state.get('depth', 0)))\n"
        "        return {'candidates': [{'operator': 'prune_blocks', 'args': {'block_ids': [fallback]}}], 'rank': [0], 'stop': True}\n"
        "    return {'candidates': candidates, 'rank': list(range(len(candidates))), 'stop': True}\n"
    ) % (algorithm_id, mode)


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
        search_choice: str = "exploit",
        protocol_mode: str = "standard",
    ) -> AlgorithmCandidate:
        if self.provider is None:
            previously_selected = self._previously_selected_blocks(history)
            retention_ratio = self._explore_retention_ratio(history)
            quality_feedback = self._has_quality_failure(history)
            channel_block_id = None if quality_feedback or search_choice != "explore" else self._channel_target_block(profile, excluded=previously_selected)
            block_excluded = set(previously_selected)
            if channel_block_id is not None:
                block_excluded.add(channel_block_id)
            block_id = self._least_sensitive_block(profile, excluded=block_excluded)
            generation_choice = "exploit" if quality_feedback else search_choice
            source = _seed_source(
                block_id,
                algorithm_id,
                generation_choice,
                retention_ratio=retention_ratio,
                channel_block_id=channel_block_id,
            )
            operator = "compose" if channel_block_id is not None else ("prune_ffn_channels" if generation_choice == "explore" else "prune_blocks")
            if quality_feedback:
                hypothesis = "quality feedback requires conservative block-only pruning"
                design_strategy = "conservative_block"
            elif channel_block_id is not None:
                hypothesis = "compose a high-cost low-sensitivity block removal with FFN width pruning at %.1f%% retention" % (retention_ratio * 100.0)
                design_strategy = "profile_guided_composite"
            elif generation_choice == "explore":
                hypothesis = "explore paired FFN width pruning at %.1f%% retention" % (retention_ratio * 100.0)
                design_strategy = "prune_ffn_channels"
            else:
                hypothesis = "remove the lowest measured leave-one-block-out sensitivity block"
                design_strategy = "prune_blocks"
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=parent_id,
                parent_algorithm_id=parent_algorithm_id,
                hypothesis=hypothesis,
                metadata={"generator": "deterministic_seed", "operator": operator, "design_strategy": design_strategy, "search_choice": search_choice, "retention_ratio": retention_ratio if generation_choice == "explore" else None, "predicted_gain": {"latency": 0.05, "model_size": 0.10, "basis": "profile_sensitivity_and_operator_proxy"}},
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
            "profile": _compact_profile(profile),
            "history": compact_history,
            "skills": compact_skills,
            "pareto_front": _compact_pareto_front(pareto_front),
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
        return AlgorithmCandidate.from_source(
            algorithm_id,
                source,
                parent_id=parent_id,
                parent_algorithm_id=parent_algorithm_id,
            hypothesis=str(raw.get("hypothesis", "")),
            metadata={"generator": "structured_provider", "search_choice": search_choice, "predicted_gain": dict(raw.get("predicted_gain") or {})},
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
            source = _ensure_search_program(
                source,
                algorithm_id,
                exploration=not any("quality" in item.lower() or "regression" in item.lower() for item in feedback),
            )
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
            "profile": _compact_profile(profile),
            "history": _compact_context_items(history),
            "skills": _compact_context_items(skills),
            "failure_reasons": feedback[:16],
            "search_space": dict(search_space or {}),
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
    if "quality" in failure_text or "integrity" in failure_text:
        source = (
            "def propose(profile, history):\n"
            "    # Improve %s: quality feedback narrows the mutation to the least sensitive block.\n"
            "    records = [item for item in profile.get('block_sensitivity', []) if item.get('available') and item.get('importance_score') is not None]\n"
            "    if records:\n"
            "        selected = min(records, key=lambda item: float(item['importance_score']))['block_id']\n"
            "    else:\n"
            "        selected = profile.get('layers', [{'block_id': 'block.0'}])[0]['block_id']\n"
            "    return {'operator': 'prune_blocks', 'args': {'block_ids': [selected]}}\n"
        ) % parent.algorithm_id
        return (
            source,
            "replace broad mutation with sensitivity-ranked conservative block pruning",
            "quality or integrity feedback shows that the parent mutation is too aggressive",
            "conserve quality by selecting one least-sensitive block from the measured profile",
        )

    if "prune_ffn_channels" in parent.source:
        source = (
            "def propose(profile, history):\n"
            "    # Improve %s: use low-noise timestep evidence and a tighter width budget after the FFN revision.\n"
            "    width = int(profile.get('model', {}).get('ffn_width', 1))\n"
            "    timestep = profile.get('timestep_sensitivity', {}).get('low_noise', {})\n"
            "    timestep_weight = 1.0 + float(timestep.get('importance_score', 0.0) or 0.0)\n"
            "    target = max(1, int(width * 0.625))\n"
            "    records = []\n"
            "    for item in profile.get('channel_sensitivity', []):\n"
            "        for channel in item.get('channels', []):\n"
            "            if channel.get('importance_score') is not None:\n"
            "                records.append({'channel': channel['channel'], 'score': float(channel['importance_score']) * timestep_weight})\n"
            "    ordered = sorted(records, key=lambda item: float(item['score']))\n"
            "    retained_indices = sorted({int(item['channel']) for item in ordered[:target]})\n"
            "    if not retained_indices:\n"
            "        retained_indices = list(range(target))\n"
            "    block_ids = [item['block_id'] for item in profile.get('layers', [])]\n"
            "    return {'operator': 'prune_ffn_channels', 'args': {'block_ids': block_ids, 'retained_indices': retained_indices}}\n"
        ) % parent.algorithm_id
        return (
            source,
            "add timestep-aware channel ranking and tighten the FFN budget",
            "the previous FFN rule reached an efficiency plateau without using timestep evidence",
            "rank channels with low-noise sensitivity and retain a smaller aligned width",
        )

    source = (
        "def propose(profile, history):\n"
        "    # Improve %s: efficiency feedback changes channel ordering while keeping the trusted operator.\n"
        "    width = int(profile.get('model', {}).get('ffn_width', 1))\n"
        "    target = max(1, int(width * 0.75))\n"
        "    records = []\n"
        "    for item in profile.get('channel_sensitivity', []):\n"
        "        for channel in item.get('channels', []):\n"
        "            if channel.get('importance_score') is not None:\n"
        "                records.append(channel)\n"
        "    ordered = sorted(records, key=lambda item: float(item.get('importance_score', 0.0)))\n"
        "    retained_indices = sorted({int(item['channel']) for item in ordered[:target]})\n"
        "    if not retained_indices:\n"
        "        retained_indices = list(range(target))\n"
        "    block_ids = [item['block_id'] for item in profile.get('layers', [])]\n"
        "    return {'operator': 'prune_ffn_channels', 'args': {'block_ids': block_ids, 'retained_indices': retained_indices}}\n"
    ) % parent.algorithm_id
    return (
        source,
        "change block-only selection into profile-ranked FFN channel selection",
        "efficiency or model-size feedback shows that the parent block choice cannot reach the target",
        "reduce width using the lowest estimated channel contribution while preserving the trusted FFN operator",
    )


def _compact_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider prompts bounded while retaining compression evidence."""

    compact: dict[str, Any] = {}
    for key in ("model", "provenance", "schema_version", "timestep_sensitivity", "sampling_step_sensitivity"):
        if key in profile:
            compact[key] = profile[key]
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
    compact["head_sensitivity"] = [
        {field: item[field] for field in ("block_id", "attention", "head", "importance_score", "available", "method") if field in item}
        for item in profile.get("head_sensitivity", [])
        if isinstance(item, Mapping)
    ][:32]
    compact["attention_sensitivity"] = [
        {field: item[field] for field in ("block_id", "attention", "head", "importance_score", "available", "method") if field in item}
        for item in profile.get("attention_sensitivity", [])
        if isinstance(item, Mapping)
    ][:64]
    compact["token_sensitivity"] = [
        {field: item[field] for field in ("keep_ratio", "token_count", "selected_count", "importance_score", "available", "method") if field in item}
        for item in profile.get("token_sensitivity", [])
        if isinstance(item, Mapping)
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
    for field in ("experience_id", "memory_kind", "architecture", "operator", "status", "reward", "failure_reason"):
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
                "Improve the complete parent algorithm source included in this request. "
                "Read every line of parent_source, identify a concrete failure or limitation, "
                "and return the complete modified source, not a patch, pseudocode, or a fresh "
                "unrelated seed. State exactly what source-level behavior changed and why. "
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
            "a bounded operator/args plan; keep the source body under 500 tokens, never return "
            "a function name or pseudocode. "
            "Allowed plans are exactly `prune_blocks` with args `{"
            "block_ids: list[str]}`, `prune_ffn_channels` with args "
            "`{block_ids: list[str], retained_indices: list[int]}`, or "
            "`compose` with args `{"
            "steps: list[2..3] of the two base plans}`. Compose cannot be nested. "
            "The value returned by `propose` must always have exactly the top-level keys "
            "`operator` and `args`, for example `{'operator': 'compose', 'args': "
            "{'steps': [...]}}`. A top-level `{'steps': [...]}` object is invalid; "
            "never return `steps` directly and never return a multi-step plan without "
            "the `operator: 'compose'` wrapper. "
            "For every `prune_ffn_channels` step, `retained_indices` must be one flat "
            "list of integer channel indices shared by every block in that step. "
            "For the reloadable Wan checkpoint, that step's `block_ids` must contain "
            "every block from `profile['layers']` exactly once, in profile order; "
            "never submit a partial Block subset for FFN pruning. "
            "Never emit nested lists, a per-block mapping, or a different retained list "
            "for each block; those plans are not executable by the Wan operator. "
            "The returned parent_id must exactly match the supplied parent_id. "
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
                # Leave room for a compact executable source while keeping
                # the request below the controller's 16k context limit.
                "max_tokens": 1536,
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

    return call
