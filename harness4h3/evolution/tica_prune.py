"""Human-designed TICA-Prune scoring and bounded search primitives."""

from __future__ import annotations

import math
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ..adapters.wan.contracts import stable_digest
from ..sandbox.policy import SourcePolicyError, validate_source
from ..sandbox.runner import AlgorithmSandbox, CompressionPlan as SandboxCompressionPlan


BASELINE_NAMES = (
    "independent_block",
    "tica_block",
    "interaction_block",
    "tica_block_ffn",
)


def tica_operator_spec() -> dict[str, Any]:
    """Return the sandbox schema for TICA's orchestration operator.

    TICA is deliberately not inserted into the persisted SearchSpaceVersion:
    it orchestrates the existing trusted Block and globally-aligned FFN
    mutations and therefore is an algorithm-rule baseline, not a new primitive
    compression capability.
    """

    return {
        "name": "tica_prune",
        "version": "1",
        "argument_schema": {
            "block_ids": "list[str]",
            "retained_indices": "list[int]",
            "mode": "str",
            "time_weight_formula": "str",
            "risk_limit": "float",
            "fast_risk_limit": "float",
            "activation_weight": "float",
            "interaction_weight": "float",
            "beam_width": "int",
            "max_fast_evals": "int",
            "max_full_evals": "int",
            "max_interaction_evals": "int",
            "strategy": "str",
            "max_search_depth": "int",
        },
        "granularity": "algorithm_rule",
        "constraints": ["wan_ffn_width_aligned", "block_and_ffn_use_trusted_mutations"],
        "limitations": ["not_a_new_primitive_operator", "no_per_layer_ffn_width"],
        "trusted_factory_key": "tica_prune",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class TICARecipe:
    """All algorithm choices that an Agent child is allowed to revise."""

    baseline_name: str = "tica_block_ffn"
    time_weight_formula: str = "early_log"
    risk_limit: float = 0.25
    fast_risk_limit: float = 0.50
    activation_weight: float = 0.5
    interaction_weight: float = 0.5
    beam_width: int = 2
    max_fast_evals: int = 8
    max_full_evals: int = 4
    max_interaction_evals: int = 2
    strategy: str = "greedy_beam"
    time_weights: tuple[float, ...] = ()
    include_baseline_suite: bool = True
    max_search_depth: int = 3

    def __post_init__(self) -> None:
        if self.baseline_name not in BASELINE_NAMES:
            raise ValueError("unsupported TICA baseline: %s" % self.baseline_name)
        if self.time_weight_formula not in {"uniform", "early_log", "late_log"}:
            raise ValueError("unsupported TICA time weight formula: %s" % self.time_weight_formula)
        if not 0.0 <= float(self.risk_limit) <= 1.0:
            raise ValueError("TICA risk_limit must be in [0, 1]")
        if not 0.0 <= float(self.fast_risk_limit) <= 1.0:
            raise ValueError("TICA fast_risk_limit must be in [0, 1]")
        if not 0.0 <= float(self.activation_weight) <= 1.0:
            raise ValueError("TICA activation_weight must be in [0, 1]")
        if not 0.0 <= float(self.interaction_weight) <= 1.0:
            raise ValueError("TICA interaction_weight must be in [0, 1]")
        for name in ("beam_width", "max_fast_evals", "max_full_evals", "max_interaction_evals", "max_search_depth"):
            if int(getattr(self, name)) <= 0:
                raise ValueError("TICA %s must be positive" % name)
        if self.strategy not in {"greedy", "greedy_beam"}:
            raise ValueError("unsupported TICA search strategy: %s" % self.strategy)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None = None) -> "TICARecipe":
        values = dict(raw or {})
        if "per_layer_retained_indices" in values:
            raise ValueError("global FFN alignment forbids per-layer retained indices")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("unsupported TICA recipe field(s): %s" % ", ".join(unknown))
        if "time_weights" in values and isinstance(values["time_weights"], list):
            values["time_weights"] = tuple(float(item) for item in values["time_weights"])
        return cls(**values)

    def budget(self) -> dict[str, int]:
        return {
            "max_fast_evals": int(self.max_fast_evals),
            "max_full_evals": int(self.max_full_evals),
            "max_interaction_evals": int(self.max_interaction_evals),
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TICAScore:
    time_weights: Mapping[str, float]
    block_risk: Mapping[str, float]
    channel_contribution: Mapping[int, float]
    formula: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_weights": dict(self.time_weights),
            "block_risk": dict(self.block_risk),
            "channel_contribution": {str(key): value for key, value in self.channel_contribution.items()},
            "formula": self.formula,
        }


@dataclass(frozen=True)
class TICACandidate:
    candidate_id: str
    kind: str
    plan: Mapping[str, Any]
    parent_candidate_ids: tuple[str, ...] = ()
    fast_metrics: Mapping[str, Any] = field(default_factory=dict)
    full_metrics: Mapping[str, Any] = field(default_factory=dict)
    interaction_metrics: Mapping[str, Any] = field(default_factory=dict)
    budget_cost: Mapping[str, int] = field(default_factory=dict)
    status: str = "generated"
    state_id: str = ""
    plan_digest: str = ""
    risk: Optional[float] = None
    feasible: bool = False
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class AlgorithmSearchDecision:
    """One source-controlled candidate generation decision."""

    plans: tuple[Mapping[str, Any], ...]
    rank: tuple[int, ...]
    stop: bool
    source_sha256: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plans": [_jsonable(item) for item in self.plans],
            "rank": list(self.rank),
            "stop": bool(self.stop),
            "source_sha256": self.source_sha256,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class TICASearchResult:
    recipe: TICARecipe
    candidates: tuple[TICACandidate, ...]
    pareto_front: tuple[Mapping[str, Any], ...]
    budget: Mapping[str, int]
    calibration_digest: str
    evidence_level: str
    score: Mapping[str, Any] = field(default_factory=dict)
    selected_candidate_id: Optional[str] = None
    selected_plan_digest: Optional[str] = None
    search_states: tuple[Mapping[str, Any], ...] = ()
    budget_audit: tuple[Mapping[str, Any], ...] = ()
    seed_plan: Mapping[str, Any] = field(default_factory=dict)
    observed_search_depth: int = 0
    level_stats: tuple[Mapping[str, Any], ...] = ()
    termination_reason: str = "unknown"
    profiling_cost: Mapping[str, Any] = field(default_factory=dict)
    total_cost: Mapping[str, Any] = field(default_factory=dict)
    algorithm_trace: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "pareto_front": [_jsonable(item) for item in self.pareto_front],
            "budget": dict(self.budget),
            "budget_scope": "shared_across_baseline_roots" if self.recipe.include_baseline_suite else "single_algorithm_root",
            "calibration_digest": self.calibration_digest,
            "evidence_level": self.evidence_level,
            "score": _jsonable(self.score),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_plan_digest": self.selected_plan_digest,
            "search_states": [_jsonable(item) for item in self.search_states],
            "budget_audit": [_jsonable(item) for item in self.budget_audit],
            "seed_plan": _jsonable(self.seed_plan),
            "observed_search_depth": int(self.observed_search_depth),
            "configured_max_search_depth": int(self.recipe.max_search_depth),
            "level_stats": [_jsonable(item) for item in self.level_stats],
            "termination_reason": self.termination_reason,
            "profiling_cost": _jsonable(self.profiling_cost),
            "total_cost": _jsonable(self.total_cost),
            "algorithm_trace": [_jsonable(item) for item in self.algorithm_trace],
        }


@dataclass
class TICABudgetManager:
    """One auditable budget ledger for the TICA search.

    An interaction analysis can use cached independent measurements, but a
    new independent model execution always consumes the same full-evaluation
    budget as an ordinary candidate.  Cache hits are recorded separately and
    are never reported as independent measurements.
    """

    limits: Mapping[str, int]
    used: dict[str, int] = field(default_factory=lambda: {"fast_evals": 0, "full_evals": 0, "interaction_evals": 0})
    audit: list[Mapping[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.limits = {
            "max_fast_evals": max(0, int(self.limits.get("max_fast_evals", 0))),
            "max_full_evals": max(0, int(self.limits.get("max_full_evals", 0))),
            "max_interaction_evals": max(0, int(self.limits.get("max_interaction_evals", 0))),
        }
        self.used = {
            "fast_evals": int(self.used.get("fast_evals", 0)),
            "full_evals": int(self.used.get("full_evals", 0)),
            "interaction_evals": int(self.used.get("interaction_evals", 0)),
        }

    def can_consume(self, stage: str, *, cache_hit: bool = False) -> bool:
        if stage == "fast":
            return cache_hit or self.used["fast_evals"] < self.limits["max_fast_evals"]
        if stage == "full":
            return cache_hit or self.used["full_evals"] < self.limits["max_full_evals"]
        if stage == "interaction":
            if self.used["interaction_evals"] >= self.limits["max_interaction_evals"]:
                return False
            return cache_hit or self.used["full_evals"] < self.limits["max_full_evals"]
        raise ValueError("unsupported TICA budget stage: %s" % stage)

    def consume(self, stage: str, *, plan_digest: str = "", cache_hit: bool = False) -> None:
        if not self.can_consume(stage, cache_hit=cache_hit):
            raise RuntimeError("tica_budget_exhausted:%s" % stage)
        if stage == "fast":
            if not cache_hit:
                self.used["fast_evals"] += 1
        elif stage == "full":
            if not cache_hit:
                self.used["full_evals"] += 1
        else:
            self.used["interaction_evals"] += 1
            if not cache_hit:
                self.used["full_evals"] += 1
        self.audit.append(
            {
                "stage": stage,
                "plan_digest": plan_digest or None,
                "cache_hit": bool(cache_hit),
                "fast_evals": self.used["fast_evals"],
                "full_evals": self.used["full_evals"],
                "interaction_evals": self.used["interaction_evals"],
            }
        )

    def record_interaction(self, *, plan_digest: str = "", cache_hit: bool = False) -> None:
        """Record one interaction analysis without hiding its dependencies."""

        if self.used["interaction_evals"] >= self.limits["max_interaction_evals"]:
            raise RuntimeError("tica_budget_exhausted:interaction")
        self.used["interaction_evals"] += 1
        self.audit.append(
            {
                "stage": "interaction",
                "plan_digest": plan_digest or None,
                "cache_hit": bool(cache_hit),
                "dependencies_consumed_separately": True,
                "fast_evals": self.used["fast_evals"],
                "full_evals": self.used["full_evals"],
                "interaction_evals": self.used["interaction_evals"],
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            **dict(self.used),
            "limits": dict(self.limits),
            "audit_entries": len(self.audit),
            "cache_hits": sum(1 for item in self.audit if item.get("cache_hit")),
        }


@dataclass(frozen=True)
class TICASearchState:
    """Canonical state in the compression search graph."""

    state_id: str
    baseline_model_id: str
    block_ids: tuple[str, ...] = ()
    retained_indices: tuple[int, ...] = ()
    mode: str = "block"
    baseline_name: str = "tica_block_ffn"
    algorithm_id: Optional[str] = None
    search_space_version: Optional[str] = None
    parent_state_id: Optional[str] = None
    depth: int = 0
    measured_metrics: Mapping[str, Any] = field(default_factory=dict)
    budget_cost: Mapping[str, int] = field(default_factory=dict)
    status: str = "generated"

    def plan(self, recipe: TICARecipe) -> dict[str, Any]:
        return {
            "block_ids": list(self.block_ids),
            "retained_indices": list(self.retained_indices),
            "mode": self.mode,
            "time_weight_formula": recipe.time_weight_formula,
            "risk_limit": float(recipe.risk_limit),
            "fast_risk_limit": float(recipe.fast_risk_limit),
            "activation_weight": float(recipe.activation_weight),
            "interaction_weight": float(recipe.interaction_weight),
            "beam_width": int(recipe.beam_width),
            "max_fast_evals": int(recipe.max_fast_evals),
            "max_full_evals": int(recipe.max_full_evals),
            "max_interaction_evals": int(recipe.max_interaction_evals),
            "strategy": recipe.strategy,
            "max_search_depth": int(recipe.max_search_depth),
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def replace_state(state: TICASearchState, **changes: Any) -> TICASearchState:
    """Create a canonical child state without mutating a search branch."""

    values = asdict(state)
    values.update(changes)
    values["block_ids"] = tuple(sorted(set(str(item) for item in values["block_ids"])))
    values["retained_indices"] = tuple(sorted(set(int(item) for item in values["retained_indices"])))
    digest = stable_digest(
        {
            "baseline_model_id": values["baseline_model_id"],
            "baseline_name": values["baseline_name"],
            "block_ids": tuple(values["block_ids"]),
            "retained_indices": tuple(values["retained_indices"]),
            "mode": values["mode"],
        }
    )
    values["state_id"] = "TICA-S-" + digest[:16]
    return TICASearchState(**values)


def replace_candidate(candidate: TICACandidate, **changes: Any) -> TICACandidate:
    values = asdict(candidate)
    values.update(changes)
    return TICACandidate(**values)


def _candidate_rank(candidate: TICACandidate) -> tuple[float, float, float, str]:
    metrics = candidate.full_metrics or candidate.fast_metrics
    quality = metrics.get("quality_score")
    reduction = metrics.get("parameter_reduction_ratio")
    quality_value = float(quality) if isinstance(quality, (int, float)) and math.isfinite(float(quality)) else -1.0
    reduction_value = float(reduction) if isinstance(reduction, (int, float)) and math.isfinite(float(reduction)) else 0.0
    interaction_penalty = float(candidate.interaction_metrics.get("risk_penalty", 0.0) or 0.0)
    # Prefer feasible measured states, then quality/cost.  Candidate IDs make
    # ties deterministic without introducing a random search decision.
    return (1.0 if candidate.feasible else 0.0, quality_value - interaction_penalty, reduction_value, candidate.plan_digest)


def _timestep_names(profile: Mapping[str, Any]) -> tuple[str, ...]:
    tica = profile.get("tica") if isinstance(profile.get("tica"), Mapping) else {}
    values = tica.get("timesteps") if isinstance(tica, Mapping) else None
    if isinstance(values, list) and values:
        return tuple(str(item) for item in values)
    records = profile.get("timestep_sensitivity")
    if isinstance(records, Mapping) and records:
        return tuple(str(item) for item in records)
    return ("high_noise", "middle", "low_noise")


def _timestep_value(name: str, index: int, count: int) -> float:
    if name in {"high_noise", "early", "t0"}:
        return 0.9
    if name in {"middle", "mid"}:
        return 0.5
    if name in {"low_noise", "late", "t_last"}:
        return 0.1
    try:
        value = float(name)
        if 0.0 <= value <= 1.0:
            return value
    except (TypeError, ValueError):
        pass
    return 1.0 - (float(index) / max(float(count - 1), 1.0))


def normalized_time_weights(names: Sequence[str], formula: str) -> dict[str, float]:
    if not names:
        raise ValueError("TICA requires at least one timestep")
    raw: list[float] = []
    for index, name in enumerate(names):
        value = _timestep_value(str(name), index, len(names))
        if formula == "uniform":
            score = 1.0
        elif formula == "early_log":
            score = math.log1p(1.0 + 9.0 * value)
        elif formula == "late_log":
            score = math.log1p(1.0 + 9.0 * (1.0 - value))
        else:
            raise ValueError("unsupported TICA time weight formula: %s" % formula)
        raw.append(max(score, 1.0e-12))
    total = sum(raw)
    return {str(name): value / total for name, value in zip(names, raw)}


def _record_error(record: Mapping[str, Any], name: str) -> float | None:
    for key in ("relative_error", "importance_score", "risk", "error"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    per_timestep = record.get("per_timestep") or record.get("timesteps")
    if isinstance(per_timestep, Mapping):
        value = per_timestep.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _channel_value(record: Mapping[str, Any]) -> float | None:
    for key in ("contribution", "activation_contribution", "importance_score"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def tica_score(
    profile: Mapping[str, Any],
    *,
    time_weight_formula: str = "early_log",
    timesteps: Sequence[str] | None = None,
    time_weights: Sequence[float] | None = None,
) -> TICAScore:
    """Compute profile-derived TICA scores without executing a model."""

    names = tuple(str(item) for item in (timesteps or _timestep_names(profile)))
    if time_weights is not None:
        if len(time_weights) != len(names) or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in time_weights):
            raise ValueError("TICA time_weights must match timesteps and contain finite non-negative values")
        total = sum(float(value) for value in time_weights)
        if total <= 0.0:
            raise ValueError("TICA time_weights must contain a positive value")
        weights = {name: float(value) / total for name, value in zip(names, time_weights)}
    else:
        weights = normalized_time_weights(names, time_weight_formula)
    block_risk: dict[str, float] = {}
    block_records = profile.get("tica_block_sensitivity") or profile.get("block_sensitivity") or ()
    for item in block_records:
        if not isinstance(item, Mapping) or not item.get("block_id"):
            continue
        if item.get("available") is False:
            continue
        values = [_record_error(item, name) for name in names]
        usable = [weights[name] * value for name, value in zip(names, values) if value is not None]
        if usable:
            block_risk[str(item["block_id"])] = float(sum(usable))
    channel_contribution: dict[int, list[float]] = {}
    channel_records = profile.get("tica_channel_sensitivity") or profile.get("channel_sensitivity") or ()
    for item in channel_records:
        if not isinstance(item, Mapping):
            continue
        if item.get("available") is False:
            continue
        for channel in item.get("channels", ()):
            if not isinstance(channel, Mapping) or not isinstance(channel.get("channel"), int):
                continue
            value = _channel_value(channel)
            if value is not None:
                channel_contribution.setdefault(int(channel["channel"]), []).append(value)
    averaged_channels = {
        channel: sum(values) / len(values)
        for channel, values in channel_contribution.items()
        if values
    }
    return TICAScore(weights, block_risk, averaged_channels, time_weight_formula)


def _profile_block_ids(profile: Mapping[str, Any], score: TICAScore) -> list[str]:
    ordered = sorted(score.block_risk, key=lambda item: (score.block_risk[item], item))
    if ordered:
        return ordered
    return [
        str(item["block_id"])
        for item in profile.get("layers", ())
        if isinstance(item, Mapping) and item.get("block_id")
    ]


def _retained_indices(profile: Mapping[str, Any], score: TICAScore, *, activation_weight: float = 1.0) -> list[int]:
    width = int((profile.get("model") or {}).get("ffn_width", 0) or 0)
    if width <= 1:
        return [0]
    activation_weight = max(0.0, min(1.0, float(activation_weight)))
    # ``channel_contribution`` is the activation-times-down-weight signal.
    # Blend it with a deterministic structural prior so activation_weight is
    # a real decision variable even when a profile only exposes one signal.
    weighted = {
        int(channel): activation_weight * float(value)
        + (1.0 - activation_weight) * (1.0 / (1.0 + float(channel)))
        for channel, value in score.channel_contribution.items()
    }
    ordered = sorted(weighted, key=lambda channel: (-weighted[channel], channel))
    target = max(1, min(width - 1, int(width * 0.75)))
    if ordered:
        return sorted(int(item) for item in ordered[:target])
    return list(range(target))


def recipe_plan(profile: Mapping[str, Any], recipe: TICARecipe) -> dict[str, Any]:
    score = tica_score(profile, time_weight_formula=recipe.time_weight_formula, time_weights=recipe.time_weights or None)
    blocks = _profile_block_ids(profile, score)
    if not blocks:
        raise ValueError("TICA profile has no block candidates")
    if recipe.baseline_name == "tica_block_ffn" and not score.channel_contribution:
        raise ValueError("TICA FFN activation evidence is unavailable")
    selected = blocks[:2] if recipe.baseline_name == "interaction_block" else blocks[:1]
    mode = "block"
    if recipe.baseline_name == "tica_block_ffn":
        mode = "block_plus_ffn"
    return {
        "block_ids": selected,
        "retained_indices": _retained_indices(profile, score, activation_weight=recipe.activation_weight),
        "mode": mode,
        "time_weight_formula": recipe.time_weight_formula,
        "risk_limit": float(recipe.risk_limit),
        "fast_risk_limit": float(recipe.fast_risk_limit),
        "activation_weight": float(recipe.activation_weight),
        "interaction_weight": float(recipe.interaction_weight),
        "beam_width": int(recipe.beam_width),
        "max_fast_evals": int(recipe.max_fast_evals),
        "max_full_evals": int(recipe.max_full_evals),
        "max_interaction_evals": int(recipe.max_interaction_evals),
        "strategy": recipe.strategy,
        "max_search_depth": int(recipe.max_search_depth),
    }


def build_human_tica_source(algorithm_id: str) -> str:
    """Return the source of the named human baseline.

    The source is deliberately self-contained so the existing Sandbox can
    validate it. It computes the selected block and channels from ProfileReport
    values; no model-specific block ID is embedded in the source.
    """

    return """import math

def propose(profile, history):
    # Human-designed TICA-Prune baseline %s; this is not Agent discovery.
    tica = profile.get('tica', {})
    names = [str(item) for item in tica.get('timesteps', ['0.9', '0.5', '0.1'])]
    values = [float(item) for item in names]
    raw_weights = [math.log1p(1.0 + 9.0 * value) for value in values]
    total = sum(raw_weights)
    weights = [value / total for value in raw_weights]
    records = []
    for item in profile.get('tica_block_sensitivity', profile.get('block_sensitivity', [])):
        values = item.get('per_timestep', item.get('timesteps', {}))
        fallback_names = ['high_noise', 'middle', 'low_noise']
        score = sum(weights[index] * float(values.get(names[index], values.get(fallback_names[index], item.get('importance_score', 0.0)))) for index in range(len(names)))
        records.append((score, str(item.get('block_id'))))
    if not records:
        records = [(0.0, str(item.get('block_id'))) for item in profile.get('layers', [])]
    selected = sorted(records)[0][1]
    width = int(profile.get('model', {}).get('ffn_width', 1))
    channels = []
    for item in profile.get('tica_channel_sensitivity', profile.get('channel_sensitivity', [])):
        for channel in item.get('channels', []):
            contribution = channel.get('contribution', channel.get('importance_score'))
            if contribution is not None:
                channels.append((float(contribution), int(channel.get('channel'))))
    target = max(1, min(width - 1, int(width * 0.75)))
    retained = sorted(set(item[1] for item in sorted(channels, reverse=True)[:target])) if channels else []
    mode = 'block_plus_ffn' if channels else 'block'
    return {'operator': 'tica_prune', 'args': {
        'block_ids': [selected], 'retained_indices': retained,
        'mode': mode, 'time_weight_formula': 'early_log',
        'risk_limit': 0.25, 'fast_risk_limit': 0.50, 'activation_weight': 0.5,
        'interaction_weight': 0.5, 'beam_width': 2,
        'max_fast_evals': 8, 'max_full_evals': 4,
        'max_interaction_evals': 2,
        'strategy': 'greedy_beam', 'max_search_depth': 3,
    }}
""" % str(algorithm_id)


def _first_tensor(value: Any) -> Any:
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value
    return None


def _relative_error(reference: Any, candidate: Any) -> float | None:
    left = _first_tensor(reference)
    right = _first_tensor(candidate)
    if left is None or right is None or tuple(left.shape) != tuple(right.shape):
        return None
    try:
        delta = (right.detach().float() - left.detach().float()).abs().mean()
        scale = left.detach().float().abs().mean().clamp_min(1.0e-12)
        value = float((delta / scale).item())
        return value if math.isfinite(value) else None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _probe_block(adapter: Any, model: Any, block_id: str, sample: Any, timestep: float, forward: Any) -> Any:
    probe = getattr(adapter, "probe_block_removal", None)
    callback = lambda current_model, current_sample: forward(current_model, current_sample, timestep)
    if callable(probe):
        return probe(model, block_id, sample, callback)
    candidate = deepcopy(model)
    adapter.remove_blocks(candidate, (block_id,))
    try:
        return callback(candidate, sample)
    finally:
        del candidate


def _module_at(root: Any, path: str) -> Any:
    value = root
    for part in str(path).split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def _activation_contribution(adapter: Any, model: Any, samples: Sequence[Any], forward: Any) -> tuple[Mapping[str, Any], ...]:
    """Measure activation magnitude times paired down-projection weight."""

    try:
        import torch
    except ImportError:
        return tuple(
            {"block_id": block.block_id, "channels": [], "available": False, "method": "activation_contribution", "reason": "torch_unavailable"}
            for block in adapter.get_blocks(model)
        )
    blocks = adapter.get_blocks(model)
    totals: dict[str, Any] = {}
    counts: dict[str, int] = {}
    down_modules: dict[str, Any] = {}
    handles = []
    for block in blocks:
        try:
            module = _module_at(model, block.module_path)
            components = getattr(adapter, "ffn_components", None)
            parts = components(module) if callable(components) else None
            if parts is None:
                candidates = (
                    ("up_proj", "activation", "down_proj"),
                    ("ffn.up_proj", "ffn.activation", "ffn.down_proj"),
                    ("ffn.net.0", "ffn.net.1", "ffn.net.2"),
                    ("ffn.net.0.proj", "ffn.net.1", "ffn.net.2"),
                )
                for up_path, activation_path, down_path in candidates:
                    try:
                        parts = (_module_at(module, up_path), _module_at(module, activation_path), _module_at(module, down_path))
                        break
                    except (AttributeError, IndexError, TypeError):
                        continue
            if parts is None:
                raise ValueError("ffn_activation_path_unavailable")
            _up, activation, down = parts
            if activation is None or down is None or not hasattr(down, "weight"):
                continue
            width = int(down.weight.shape[1])
            totals[block.block_id] = torch.zeros(width, device=down.weight.device, dtype=torch.float32)
            counts[block.block_id] = 0
            down_modules[block.block_id] = down

            def hook(_module: Any, _inputs: Any, output: Any, block_id: str = block.block_id) -> None:
                tensor = _first_tensor(output)
                if tensor is None:
                    return
                values = tensor.detach().float().abs()
                while values.ndim > 1:
                    values = values.mean(dim=0)
                if values.numel() == totals[block_id].numel():
                    totals[block_id] += values.to(totals[block_id].device)
                    counts[block_id] += 1

            handles.append(activation.register_forward_hook(hook))
        except (AttributeError, IndexError, RuntimeError, TypeError):
            continue
    try:
        for sample in samples:
            for timestep in sample.spec.timesteps:
                forward(model, sample, float(timestep))
    finally:
        for handle in handles:
            handle.remove()
    rows: list[Mapping[str, Any]] = []
    for block in blocks:
        if block.block_id not in totals or not counts.get(block.block_id):
            rows.append({"block_id": block.block_id, "channels": [], "available": False, "method": "activation_contribution", "reason": "activation_hook_unavailable"})
            continue
        down = down_modules.get(block.block_id)
        if down is None or not hasattr(down, "weight"):
            rows.append({"block_id": block.block_id, "channels": [], "available": False, "method": "activation_contribution", "reason": "down_projection_unavailable"})
            continue
        weight = down.weight.detach().float().abs().mean(dim=0).cpu()
        activation_mean = (totals[block.block_id] / float(counts[block.block_id])).detach().cpu()
        channels = [
            {"channel": index, "contribution": float((activation_mean[index] * weight[index]).item()), "available": True}
            for index in range(int(activation_mean.numel()))
        ]
        rows.append({"block_id": block.block_id, "channels": channels, "available": True, "method": "activation_times_down_weight", "sample_count": len(samples)})
    return tuple(rows)


def collect_tica_profile(adapter: Any, model: Any, manifest: Any, *, max_block_probes: int | None = None) -> Mapping[str, Any]:
    """Collect multi-timestep local block risk and activation contributions."""

    samples = tuple(manifest.samples(model, adapter))
    forward = getattr(adapter, "forward_denoising", None)
    if not callable(forward):
        raise ValueError("TICA profile requires adapter.forward_denoising")
    preflight = getattr(adapter, "preflight_loaded_model", None)
    runtime_preflight = dict(preflight(model)) if callable(preflight) else {"runtime_preflight": "not_available"}
    block_refs = adapter.get_blocks(model)
    if max_block_probes is not None:
        block_refs = block_refs[: int(max_block_probes)]
    reference: dict[tuple[str, float], Any] = {}
    for sample in samples:
        for timestep in sample.spec.timesteps:
            reference[(str(sample.spec.sample_id), float(timestep))] = forward(model, sample, float(timestep))
    block_rows: list[Mapping[str, Any]] = []
    for block in block_refs:
        per_timestep: dict[str, list[float]] = {}
        for sample in samples:
            for timestep in sample.spec.timesteps:
                modified = _probe_block(adapter, model, block.block_id, sample, float(timestep), forward)
                error = _relative_error(reference[(str(sample.spec.sample_id), float(timestep))], modified)
                per_timestep.setdefault(str(timestep), []).append(error if error is not None else float("nan"))
        averaged: dict[str, float] = {}
        for name, values in per_timestep.items():
            usable = [value for value in values if math.isfinite(value)]
            if usable:
                averaged[name] = sum(usable) / len(usable)
        block_rows.append(
            {
                "block_id": block.block_id,
                "per_timestep": averaged,
                "available": bool(averaged) and len(averaged) == len(per_timestep),
                "method": "local_denoising_relative_error",
                "sample_count": len(samples),
                "unavailable_reason": None if averaged else "no_finite_output_measurements",
            }
        )
    channel_rows = _activation_contribution(adapter, model, samples, forward)
    return {
        "tica": {
            "calibration_digest": manifest.digest(),
            "timesteps": [str(value) for value in manifest.timesteps()],
            "sample_ids": [str(sample.spec.sample_id) for sample in samples],
            "offline_simulation": manifest.provider_name == "fake_wan_cpu",
        },
        "tica_block_sensitivity": block_rows,
        "tica_channel_sensitivity": channel_rows,
        "evidence": {
            "status": "cpu_fake_quality_proxy" if manifest.provider_name == "fake_wan_cpu" else "real_wan_structure",
            "local_output_measurements": len(block_rows) * sum(len(item.spec.timesteps) for item in samples),
            "activation_measurements": sum(len(item.get("channels", ())) for item in channel_rows),
            "runtime_preflight": runtime_preflight,
        },
        "profiling_cost": {
            "block_probe_count": len(block_rows),
            "local_output_measurements": len(block_rows) * sum(len(item.spec.timesteps) for item in samples),
            "activation_hook_count": sum(1 for item in channel_rows if item.get("available") is True),
            "sample_count": len(samples),
            "timestep_count": len(manifest.timesteps()),
            "scope": "shared_fixed_calibration_profile",
        },
    }


EvaluateCandidate = Callable[[Mapping[str, Any], bool, str], Mapping[str, Any]]


def _operator_specs_from_search_space(search_space: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    for item in search_space.get("operators", ()) if isinstance(search_space, Mapping) else ():
        if not isinstance(item, Mapping) or not str(item.get("name", "")).strip():
            continue
        values[str(item["name"])] = dict(item)
    return values


def _run_algorithm_search_program(
    source: str,
    sandbox: AlgorithmSandbox,
    profile: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    search_space: Mapping[str, Any],
) -> AlgorithmSearchDecision:
    payload = {
        "profile": dict(profile),
        "history": [dict(item) for item in history if isinstance(item, Mapping)],
        "state": dict(state),
        "search_space": dict(search_space),
    }
    result = sandbox.run_program(source, entrypoint="search", payload=payload)
    if result.status != "success" or not isinstance(result.value, Mapping):
        raise ValueError(result.message or "algorithm_search_program_failed")
    raw_candidates = result.value.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("algorithm_search_output_invalid: candidates must be a non-empty list")
    operator_specs = _operator_specs_from_search_space(search_space)
    plans: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        try:
            plan = SandboxCompressionPlan.from_mapping(raw, operator_specs=operator_specs)
        except (TypeError, ValueError) as exc:
            raise ValueError("algorithm_search_output_invalid: %s" % exc) from exc
        digest = stable_digest(plan.to_dict())
        if digest in seen:
            raise ValueError("algorithm_search_output_invalid: duplicate candidate plan")
        seen.add(digest)
        plans.append(plan.to_dict())
    raw_rank = result.value.get("rank", list(range(len(plans))))
    if not isinstance(raw_rank, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_rank):
        raise ValueError("algorithm_search_output_invalid: rank must be a list of indexes")
    if len(set(raw_rank)) != len(raw_rank) or any(item < 0 or item >= len(plans) for item in raw_rank):
        raise ValueError("algorithm_search_output_invalid: rank contains an invalid index")
    rank = tuple([*raw_rank, *[index for index in range(len(plans)) if index not in set(raw_rank)]])
    return AlgorithmSearchDecision(
        plans=tuple(plans),
        rank=rank,
        stop=bool(result.value.get("stop", False)),
        source_sha256=result.source_sha256,
        payload_sha256=stable_digest(payload),
    )


class TICAPruner:
    """Generate candidates with a bounded, stateful Greedy/Beam search."""

    @staticmethod
    def _plan_digest(plan: Mapping[str, Any]) -> str:
        return stable_digest({"operator": "tica_prune", "args": dict(plan)})

    @staticmethod
    def _structure_digest(plan: Mapping[str, Any]) -> str:
        """Digest only executable structure for cross-rule measurement reuse."""

        return stable_digest(
            {
                "operator": "tica_prune",
                "mode": str(plan.get("mode", "block")),
                "block_ids": sorted(str(item) for item in plan.get("block_ids", ())),
                "retained_indices": sorted(int(item) for item in plan.get("retained_indices", ())),
            }
        )

    @staticmethod
    def _baseline_score(profile: Mapping[str, Any], baseline: str, recipe: TICARecipe) -> TICAScore:
        formula = "uniform" if baseline == "independent_block" else recipe.time_weight_formula
        weights = None if baseline == "independent_block" else recipe.time_weights or None
        return tica_score(profile, time_weight_formula=formula, time_weights=weights)

    @staticmethod
    def _parent_model_id(parent: Any, profile: Mapping[str, Any]) -> str:
        value = getattr(parent, "sha256", None) or profile.get("parent_checkpoint_sha256")
        if not value and isinstance(profile.get("tica"), Mapping):
            value = profile["tica"].get("parent_checkpoint_sha256")
        return str(value or "profile-parent")

    def _root_states(
        self,
        parent: Any,
        profile: Mapping[str, Any],
        recipe: TICARecipe,
        *,
        parent_state_id: Optional[str] = None,
        seed_plan: Optional[Mapping[str, Any]] = None,
    ) -> tuple[TICASearchState, ...]:
        names = BASELINE_NAMES if recipe.include_baseline_suite else (recipe.baseline_name,)
        roots: list[TICASearchState] = []
        parent_id = self._parent_model_id(parent, profile)
        for baseline in names:
            score = self._baseline_score(profile, baseline, recipe)
            blocks = _profile_block_ids(profile, score)
            if not blocks:
                continue
            profile_blocks = set(blocks)
            seeded_blocks = [str(item) for item in (seed_plan or {}).get("block_ids", ()) if str(item) in profile_blocks]
            blocks = list(dict.fromkeys([*seeded_blocks, *blocks]))
            if baseline == "interaction_block":
                selected = tuple(blocks[:2])
                mode = "block"
                retained: tuple[int, ...] = ()
            elif baseline == "tica_block_ffn":
                selected = tuple(blocks[:1])
                mode = "block_plus_ffn"
                profile_retained = _retained_indices(profile, score, activation_weight=recipe.activation_weight)
                width = int((profile.get("model") or {}).get("ffn_width", 0) or 0)
                seeded_retained = [
                    int(item)
                    for item in (seed_plan or {}).get("retained_indices", ())
                    if isinstance(item, int) and not isinstance(item, bool) and 0 <= int(item) < width
                ]
                target = max(1, min(width - 1, int(width * 0.75))) if width > 1 else 1
                retained = tuple(list(dict.fromkeys([*seeded_retained, *profile_retained]))[:target])
            else:
                selected = tuple(blocks[:1])
                mode = "block"
                retained = ()
            if len(selected) >= len(blocks) and len(blocks) > 1:
                selected = tuple(blocks[: max(1, len(blocks) - 1)])
            key = stable_digest(
                {
                    "parent": parent_id,
                    "baseline": baseline,
                    "blocks": selected,
                    "retained": retained,
                    "mode": mode,
                }
            )
            roots.append(
                TICASearchState(
                    state_id="TICA-S-" + key[:16],
                    baseline_model_id=parent_id,
                    block_ids=selected,
                    retained_indices=retained,
                    mode=mode,
                    baseline_name=baseline,
                    parent_state_id=parent_state_id,
                    depth=1,
                )
            )
        return tuple(roots)

    @staticmethod
    def _expand_state(state: TICASearchState, profile: Mapping[str, Any], recipe: TICARecipe) -> tuple[TICASearchState, ...]:
        score = tica_score(profile, time_weight_formula=recipe.time_weight_formula, time_weights=recipe.time_weights or None)
        blocks = _profile_block_ids(profile, score)
        candidates: list[TICASearchState] = []
        remaining = [item for item in blocks if item not in state.block_ids]
        allow_more_blocks = state.baseline_name in {"interaction_block", "tica_block_ffn"}
        if allow_more_blocks and remaining and state.mode in {"block", "block_plus_ffn"}:
            for next_block in remaining:
                selected = tuple([*state.block_ids, next_block])
                candidates.append(
                    replace_state(
                        state,
                        block_ids=selected,
                        depth=state.depth + 1,
                        parent_state_id=state.state_id,
                    )
                )
        if state.baseline_name == "tica_block_ffn" and state.mode == "block":
            retained = tuple(_retained_indices(profile, score, activation_weight=recipe.activation_weight))
            candidates.append(
                replace_state(
                    state,
                    mode="block_plus_ffn",
                    retained_indices=retained,
                    depth=state.depth + 1,
                    parent_state_id=state.state_id,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _risk(state: TICASearchState, score: TICAScore, profile: Mapping[str, Any], recipe: TICARecipe) -> Optional[float]:
        block_values = [score.block_risk.get(item) for item in state.block_ids]
        if any(value is None or not math.isfinite(float(value)) for value in block_values):
            return None
        risk = sum(float(value) for value in block_values)
        if state.mode in {"ffn", "block_plus_ffn"}:
            width = int((profile.get("model") or {}).get("ffn_width", 0) or 0)
            contributions = dict(score.channel_contribution)
            if width <= 1 or len(contributions) < width:
                return None
            total = sum(max(float(value), 0.0) for value in contributions.values())
            removed = sum(
                max(float(value), 0.0)
                for channel, value in contributions.items()
                if int(channel) not in set(state.retained_indices)
            )
            ffn_risk = removed / max(total, 1.0e-12)
            risk += float(recipe.activation_weight) * ffn_risk
        return float(risk)

    @staticmethod
    def _synthetic_metrics(state: TICASearchState, profile: Mapping[str, Any], risk: Optional[float], *, full: bool = False) -> dict[str, Any]:
        block_count = max(int((profile.get("model") or {}).get("block_count", len(profile.get("layers", ()))) or 1), 1)
        width = max(int((profile.get("model") or {}).get("ffn_width", 1) or 1), 1)
        removed_blocks = len(state.block_ids)
        retained_ratio = len(state.retained_indices) / float(width) if state.mode in {"ffn", "block_plus_ffn"} else 1.0
        reduction = min(0.95, removed_blocks / float(block_count) + (1.0 - retained_ratio) * 0.5)
        return {
            "valid": risk is not None,
            "quality_error": risk,
            "quality_score": 1.0 - risk if risk is not None else None,
            "risk_metric": "normalized_mse" if full else "local_relative_output_error",
            "risk_value": risk,
            "risk_scale": "normalized_mse" if full else "profile_relative_error",
            "risk_is_approximate": not full,
            "parameter_reduction_ratio": reduction,
            "selected_blocks": list(state.block_ids),
            "mode": state.mode,
            "evaluation_status": "profile_fast_screen",
        }

    def _interaction_plans(self, state: TICASearchState, recipe: TICARecipe) -> tuple[Mapping[str, Any], ...]:
        base = state.plan(recipe)
        plans: list[Mapping[str, Any]] = []
        if len(state.block_ids) > 1:
            for block_id in state.block_ids:
                plans.append({**base, "mode": "block", "block_ids": [block_id], "retained_indices": []})
        if state.mode == "block_plus_ffn":
            plans.append({**base, "mode": "block", "block_ids": list(state.block_ids), "retained_indices": []})
            plans.append({**base, "mode": "ffn", "block_ids": [], "retained_indices": list(state.retained_indices)})
        return tuple(plans)

    def _measure_state(
        self,
        state: TICASearchState,
        profile: Mapping[str, Any],
        recipe: TICARecipe,
        score: TICAScore,
        budget: TICABudgetManager,
        evaluate_candidate: Optional[EvaluateCandidate],
        cache: dict[str, dict[str, Mapping[str, Any]]],
        full_cache: dict[str, Mapping[str, Any]],
    ) -> TICACandidate:
        plan = state.plan(recipe)
        digest = self._plan_digest(plan)
        structure_digest = self._structure_digest(plan)
        state_score = score
        if state.baseline_name in BASELINE_NAMES:
            state_score = self._baseline_score(profile, state.baseline_name, recipe)
        risk = self._risk(state, state_score, profile, recipe)
        entry = cache.setdefault(digest, {})
        cache_hit = False
        fast_metrics: Mapping[str, Any] = entry.get("fast", {})
        full_metrics: Mapping[str, Any] = full_cache.get(structure_digest, entry.get("full", {}))
        if not fast_metrics:
            if not budget.can_consume("fast"):
                return TICACandidate("", state.baseline_name, plan, status="budget_exhausted:fast", state_id=state.state_id, plan_digest=digest, risk=risk)
            budget.consume("fast", plan_digest=digest)
            try:
                fast_metrics = dict(evaluate_candidate(plan, False, "fast") if evaluate_candidate else self._synthetic_metrics(state, profile, risk, full=False))
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                fast_metrics = {"valid": False, "failure_reason": "fast_measurement_failed:%s" % exc}
            entry["fast"] = fast_metrics
        else:
            cache_hit = True
            budget.consume("fast", plan_digest=digest, cache_hit=True)
        fast_risk = self._risk_evidence(fast_metrics, stage="fast")
        if fast_risk is None and "risk_metric" not in fast_metrics:
            legacy_error = fast_metrics.get("quality_error") if isinstance(fast_metrics, Mapping) else None
            if isinstance(legacy_error, (int, float)) and math.isfinite(float(legacy_error)) and float(legacy_error) >= 0.0:
                fast_risk = (float(legacy_error), "legacy_quality_error")
        # Fast local relative error is only a pre-filter.  It has its own
        # threshold and can never stand in for a full normalized-MSE result.
        profile_evidence_available = risk is not None
        fast_measurement_valid = isinstance(fast_metrics, Mapping) and fast_metrics.get("valid") is True
        feasible = profile_evidence_available and fast_measurement_valid and fast_risk is not None and fast_risk[0] <= float(recipe.fast_risk_limit)
        status = "screened"
        if not feasible:
            if not profile_evidence_available or fast_risk is None:
                status = "fast_risk_unknown"
            elif not fast_measurement_valid:
                status = "fast_measurement_invalid"
            else:
                status = "fast_risk_rejected"
        elif not full_metrics:
            if not budget.can_consume("full"):
                status = "budget_exhausted:full"
                feasible = False
            else:
                budget.consume("full", plan_digest=digest)
                try:
                    full_metrics = dict(evaluate_candidate(plan, True, "full") if evaluate_candidate else self._synthetic_metrics(state, profile, risk, full=True))
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    full_metrics = {"valid": False, "failure_reason": "full_measurement_failed:%s" % exc}
                entry["full"] = full_metrics
                full_cache[structure_digest] = full_metrics
        else:
            cache_hit = True
            budget.consume("full", plan_digest=digest, cache_hit=True)
        full_risk = self._risk_evidence(full_metrics, stage="full") if full_metrics else None
        if feasible:
            if not full_metrics or full_metrics.get("valid") is not True:
                feasible = False
                status = "full_measurement_invalid"
            elif full_risk is None:
                feasible = False
                # Preserve the legacy diagnostic label when an older
                # evaluator supplied only an out-of-budget quality_error.
                # It remains infeasible; any explicitly declared but
                # mismatched risk scale stays fail-closed as unavailable.
                legacy_error = full_metrics.get("quality_error") if isinstance(full_metrics, Mapping) else None
                if "risk_metric" not in full_metrics and isinstance(legacy_error, (int, float)) and float(legacy_error) > float(recipe.risk_limit):
                    status = "full_risk_rejected"
                else:
                    status = "full_risk_unavailable"
            elif full_risk[0] > float(recipe.risk_limit):
                feasible = False
                status = "full_risk_rejected"
        interaction: Mapping[str, Any] = {}
        if feasible and full_metrics and state.baseline_name in {"interaction_block", "tica_block_ffn"}:
            if not budget.can_consume("interaction", cache_hit=True):
                interaction = {"available": False, "reason": "interaction_budget_exhausted"}
                feasible = False
            else:
                independent: list[Mapping[str, Any]] = []
                missing = False
                for independent_plan in self._interaction_plans(state, recipe):
                    independent_digest = self._plan_digest(independent_plan)
                    independent_structure_digest = self._structure_digest(independent_plan)
                    independent_entry = cache.setdefault(independent_digest, {})
                    metrics = full_cache.get(independent_structure_digest, independent_entry.get("full"))
                    if not metrics:
                        missing = True
                        if not budget.can_consume("full"):
                            break
                        budget.consume("full", plan_digest=independent_digest)
                        try:
                            metrics = dict(evaluate_candidate(independent_plan, True, "interaction_independent") if evaluate_candidate else {})
                        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                            metrics = {"valid": False, "failure_reason": "interaction_measurement_failed:%s" % exc}
                        independent_entry["full"] = metrics
                        full_cache[independent_structure_digest] = metrics
                    independent.append(metrics)
                if len(independent) != len(self._interaction_plans(state, recipe)):
                    interaction = {"available": False, "reason": "missing_independent_measurement", "independent_count": len(independent)}
                    feasible = False
                else:
                    budget.record_interaction(plan_digest=digest, cache_hit=not missing)
                    interaction = self.measure_interaction(independent, full_metrics)
                    interaction["kind"] = "block_block" if len(state.block_ids) > 1 else "block_ffn"
                    if not interaction.get("available"):
                        feasible = False
                    elif float(interaction.get("interaction_delta", 0.0)) > 0.0:
                        penalty = float(recipe.interaction_weight) * float(interaction["interaction_delta"])
                        interaction = {**dict(interaction), "risk_penalty": penalty}
                        if full_risk is None or float(full_risk[0]) + penalty > float(recipe.risk_limit):
                            feasible = False
                            status = "interaction_risk_rejected"
        if feasible and status == "screened":
            status = "measured"
        return TICACandidate(
            candidate_id="",
            kind=state.baseline_name,
            plan=plan,
            parent_candidate_ids=(state.parent_state_id,) if state.parent_state_id else (),
            fast_metrics=dict(fast_metrics),
            full_metrics=dict(full_metrics),
            interaction_metrics=dict(interaction),
            budget_cost=dict(budget.used),
            status=status if evaluate_candidate else (status if status != "measured" else "generated"),
            state_id=state.state_id,
            plan_digest=digest,
            risk=risk,
            feasible=feasible,
            cache_hit=cache_hit,
        )

    @staticmethod
    def _risk_evidence(metrics: Mapping[str, Any], *, stage: str) -> Optional[tuple[float, str]]:
        """Read a risk only when the producer declares its scale."""

        if not isinstance(metrics, Mapping):
            return None
        value = metrics.get("risk_value")
        metric = str(metrics.get("risk_metric", "")).strip()
        scale = str(metrics.get("risk_scale", "")).strip()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)) or float(value) < 0.0:
            return None
        expected = {
            "fast": ("local_relative_output_error", "profile_relative_error"),
            "full": ("normalized_mse", "normalized_mse"),
        }.get(stage)
        if expected is None:
            raise ValueError("unsupported TICA risk stage: %s" % stage)
        if (metric, scale) != expected:
            return None
        approximate = metrics.get("risk_is_approximate")
        if stage == "fast" and approximate is not True:
            return None
        if stage == "full" and approximate is not False:
            return None
        return float(value), metric

    def _measure_program_plan(
        self,
        plan: Mapping[str, Any],
        recipe: TICARecipe,
        budget: TICABudgetManager,
        evaluate_candidate: Optional[EvaluateCandidate],
    ) -> TICACandidate:
        operator = str(plan.get("operator", ""))
        args = plan.get("args") if isinstance(plan.get("args"), Mapping) else {}
        digest = stable_digest({"operator": operator, "args": dict(args)})
        fast_metrics: Mapping[str, Any] = {}
        full_metrics: Mapping[str, Any] = {}
        if not budget.can_consume("fast"):
            return TICACandidate(
                "",
                operator,
                dict(plan),
                status="budget_exhausted:fast",
                plan_digest=digest,
            )
        budget.consume("fast", plan_digest=digest)
        try:
            fast_metrics = dict(evaluate_candidate(dict(plan), False, "algorithm_program_fast") if evaluate_candidate else {})
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            fast_metrics = {"valid": False, "failure_reason": "fast_measurement_failed:%s" % exc}
        fast_risk = self._risk_evidence(fast_metrics, stage="fast")
        feasible = bool(fast_metrics.get("valid") is True and fast_risk is not None and fast_risk[0] <= float(recipe.fast_risk_limit))
        status = "screened" if feasible else "fast_risk_rejected"
        if not feasible and fast_risk is None:
            status = "fast_risk_unknown"
        if feasible:
            if not budget.can_consume("full"):
                feasible = False
                status = "budget_exhausted:full"
            else:
                budget.consume("full", plan_digest=digest)
                try:
                    full_metrics = dict(evaluate_candidate(dict(plan), True, "algorithm_program_full") if evaluate_candidate else {})
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    full_metrics = {"valid": False, "failure_reason": "full_measurement_failed:%s" % exc}
                full_risk = self._risk_evidence(full_metrics, stage="full")
                if full_metrics.get("valid") is not True:
                    feasible = False
                    status = "full_measurement_invalid"
                elif full_risk is None:
                    feasible = False
                    status = "full_risk_unavailable"
                elif full_risk[0] > float(recipe.risk_limit):
                    feasible = False
                    status = "full_risk_rejected"
                else:
                    status = "measured"
        state_id = "PROGRAM-S-" + digest[:16]
        return TICACandidate(
            "",
            operator,
            dict(plan),
            fast_metrics=dict(fast_metrics),
            full_metrics=dict(full_metrics),
            budget_cost=dict(budget.used),
            status=status,
            state_id=state_id,
            plan_digest=digest,
            feasible=feasible,
        )

    def _search_with_algorithm_program(
        self,
        parent: Any,
        manifest: Any,
        profile: Mapping[str, Any],
        recipe: TICARecipe,
        *,
        algorithm_source: str,
        algorithm_history: Sequence[Mapping[str, Any]],
        search_space: Mapping[str, Any],
        sandbox: AlgorithmSandbox,
        evaluate_candidate: Optional[EvaluateCandidate],
    ) -> TICASearchResult:
        started = time.perf_counter()
        budget = TICABudgetManager(recipe.budget())
        candidates: list[TICACandidate] = []
        states: list[Mapping[str, Any]] = []
        traces: list[Mapping[str, Any]] = []
        measured: list[Mapping[str, Any]] = []
        parent_id = self._parent_model_id(parent, profile)
        root_digest = stable_digest({"parent": parent_id, "state": "uncompressed"})
        root_state_id = "PROGRAM-S-" + root_digest[:16]
        states.append({"state_id": root_state_id, "depth": 0, "status": "baseline", "feasible": True})
        previous_state_id = root_state_id
        termination_reason = "frontier_empty"
        observed_depth = 0
        for depth in range(1, int(recipe.max_search_depth) + 1):
            observed_depth = depth
            state_payload = {
                "depth": depth,
                "parent_state_id": previous_state_id,
                "budget": budget.snapshot(),
                "measured_candidates": list(measured),
                "frontier": [item.get("plan") for item in measured if isinstance(item, Mapping) and item.get("feasible")],
            }
            decision = _run_algorithm_search_program(
                algorithm_source,
                sandbox,
                profile,
                algorithm_history,
                state_payload,
                search_space,
            )
            ordered_plans = [decision.plans[index] for index in decision.rank]
            trace = {
                "depth": depth,
                "source_sha256": decision.source_sha256,
                "payload_sha256": decision.payload_sha256,
                "plan_digests": [stable_digest(plan) for plan in ordered_plans],
                "rank": list(decision.rank),
                "stop": decision.stop,
                "search_space_version": search_space.get("version_id") if isinstance(search_space, Mapping) else None,
                "search_space_operator_names": sorted(_operator_specs_from_search_space(search_space)),
            }
            traces.append(trace)
            generation: list[TICACandidate] = []
            for plan in ordered_plans:
                if budget.used["full_evals"] >= budget.limits["max_full_evals"]:
                    break
                candidate = self._measure_program_plan(plan, recipe, budget, evaluate_candidate)
                candidate = replace_candidate(candidate, candidate_id="TICA-C%04d" % (len(candidates) + 1), parent_candidate_ids=(previous_state_id,))
                candidates.append(candidate)
                generation.append(candidate)
                states.append(
                    {
                        "state_id": candidate.state_id,
                        "parent_state_id": previous_state_id,
                        "depth": depth,
                        "plan_digest": candidate.plan_digest,
                        "operator": candidate.kind,
                        "feasible": candidate.feasible,
                        "status": candidate.status,
                    }
                )
            measured = [
                {
                    "candidate_id": item.candidate_id,
                    "plan": item.plan,
                    "feasible": item.feasible,
                    "status": item.status,
                    "risk": item.full_metrics.get("risk_value", item.fast_metrics.get("risk_value")),
                    "quality_score": item.full_metrics.get("quality_score", item.fast_metrics.get("quality_score")),
                    "parameter_reduction_ratio": item.full_metrics.get("parameter_reduction_ratio", item.fast_metrics.get("parameter_reduction_ratio")),
                }
                for item in generation
            ]
            level_feasible = [item for item in generation if item.feasible]
            if decision.stop:
                termination_reason = "algorithm_stop"
                break
            if not generation:
                termination_reason = "budget_exhausted:full"
                break
            if not level_feasible:
                termination_reason = "algorithm_frontier_empty"
                break
            previous_state_id = max(level_feasible, key=_candidate_rank).state_id
            if budget.used["full_evals"] >= budget.limits["max_full_evals"]:
                termination_reason = "budget_exhausted:full"
                break
        if observed_depth >= int(recipe.max_search_depth) and termination_reason == "frontier_empty":
            termination_reason = "max_depth_reached"
        profiling_cost = profile.get("profiling_cost") if isinstance(profile.get("profiling_cost"), Mapping) else {}
        total_cost = {
            "search_fast_evals": int(budget.used["fast_evals"]),
            "search_full_evals": int(budget.used["full_evals"]),
            "search_interaction_evals": int(budget.used["interaction_evals"]),
            "profiling": dict(profiling_cost),
            "scope": "search_plus_shared_profiling",
        }
        pareto = _pareto_front(candidates)
        feasible = [item for item in candidates if item.feasible and item.full_metrics]
        pareto_ids = {str(item.get("candidate_id")) for item in pareto}
        pareto_candidates = [item for item in feasible if item.candidate_id in pareto_ids]
        selected = max(pareto_candidates or feasible, key=_candidate_rank, default=None)
        return TICASearchResult(
            recipe=recipe,
            candidates=tuple(candidates),
            pareto_front=tuple(pareto),
            budget={**budget.snapshot(), "elapsed_s": time.perf_counter() - started},
            calibration_digest=str(manifest.digest()),
            evidence_level="cpu_fake_quality_proxy" if getattr(manifest, "provider_name", "") == "fake_wan_cpu" else "real_wan_structure",
            score={},
            selected_candidate_id=selected.candidate_id if selected else None,
            selected_plan_digest=selected.plan_digest if selected else None,
            search_states=tuple(states),
            budget_audit=tuple(budget.audit),
            observed_search_depth=observed_depth,
            level_stats=tuple(),
            termination_reason=termination_reason,
            profiling_cost=dict(profiling_cost),
            total_cost=total_cost,
            algorithm_trace=tuple(traces),
        )

    def search(
        self,
        adapter: Any,
        parent: Any,
        manifest: Any,
        profile: Mapping[str, Any],
        recipe: TICARecipe,
        *,
        evaluate_candidate: Optional[EvaluateCandidate] = None,
        seed_plan: Optional[Mapping[str, Any]] = None,
        algorithm_source: Optional[str] = None,
        algorithm_history: Sequence[Mapping[str, Any]] = (),
        search_space: Optional[Mapping[str, Any]] = None,
        sandbox: Optional[AlgorithmSandbox] = None,
    ) -> TICASearchResult:
        del adapter
        split = profile.get("evaluation_split")
        if split is None and isinstance(profile.get("tica"), Mapping):
            split = profile["tica"].get("evaluation_split")
        if str(split or "").lower() == "test":
            raise ValueError("TICA search cannot consume Test split measurements")
        if algorithm_source is not None:
            try:
                validated = validate_source(algorithm_source)
            except SourcePolicyError as exc:
                raise ValueError("algorithm_search_source_invalid: %s" % exc) from exc
            if "search" in validated.functions:
                if sandbox is None:
                    sandbox = AlgorithmSandbox(Path(tempfile.mkdtemp(prefix="algorithm-search-")))
                return self._search_with_algorithm_program(
                    parent,
                    manifest,
                    profile,
                    recipe,
                    algorithm_source=algorithm_source,
                    algorithm_history=algorithm_history,
                    search_space=dict(search_space or {}),
                    sandbox=sandbox,
                    evaluate_candidate=evaluate_candidate,
                )
        started = time.perf_counter()
        score = tica_score(profile, time_weight_formula=recipe.time_weight_formula, time_weights=recipe.time_weights or None)
        budget = TICABudgetManager(recipe.budget())
        cache: dict[str, dict[str, Mapping[str, Any]]] = {}
        full_cache: dict[str, Mapping[str, Any]] = {}
        candidates: list[TICACandidate] = []
        states: list[Mapping[str, Any]] = []
        level_stats: list[Mapping[str, Any]] = []
        termination_reason = "frontier_empty"
        parent_id = self._parent_model_id(parent, profile)
        root_digest = stable_digest({"parent": parent_id, "state": "uncompressed"})
        root_state = TICASearchState(
            state_id="TICA-S-" + root_digest[:16],
            baseline_model_id=parent_id,
            baseline_name="uncompressed",
            depth=0,
            status="baseline",
        )
        states.append({**root_state.to_dict(), "feasible": True, "status": "baseline"})
        frontier = list(
            self._root_states(
                parent,
                profile,
                recipe,
                parent_state_id=root_state.state_id,
                seed_plan=seed_plan,
            )
        )
        seen_states: set[str] = set()
        seen_plans: set[str] = set()
        while frontier:
            depth = max(int(item.depth) for item in frontier)
            level_started = dict(budget.used)
            next_frontier: list[TICASearchState] = []
            generation: list[TICACandidate] = []
            for state in frontier:
                if state.state_id in seen_states or state.depth > int(recipe.max_search_depth):
                    continue
                seen_states.add(state.state_id)
                candidate = self._measure_state(state, profile, recipe, score, budget, evaluate_candidate, cache, full_cache)
                candidate_id = "TICA-C%04d" % (len(candidates) + 1)
                candidate = replace_candidate(candidate, candidate_id=candidate_id)
                candidates.append(candidate)
                generation.append(candidate)
                states.append({**state.to_dict(), "plan_digest": candidate.plan_digest, "feasible": candidate.feasible, "status": candidate.status})
                structure_digest = self._structure_digest(candidate.plan)
                if structure_digest not in seen_plans:
                    seen_plans.add(structure_digest)
                    if candidate.feasible and state.depth < int(recipe.max_search_depth):
                        next_frontier.extend(self._expand_state(state, profile, recipe))
            level_stats.append(
                {
                    "depth": depth,
                    "frontier_count": len(frontier),
                    "candidate_count": len(generation),
                    "feasible_count": sum(1 for item in generation if item.feasible),
                    "full_evals_before": level_started["full_evals"],
                    "full_evals_after": budget.used["full_evals"],
                    "fast_evals_before": level_started["fast_evals"],
                    "fast_evals_after": budget.used["fast_evals"],
                }
            )
            if not next_frontier:
                termination_reason = "frontier_empty_or_max_depth"
                break
            if budget.used["full_evals"] >= budget.limits["max_full_evals"]:
                termination_reason = "budget_exhausted:full"
                break
            next_frontier = [item for item in next_frontier if item.state_id not in seen_states]
            if recipe.strategy == "greedy":
                ranked = sorted(generation, key=lambda item: _candidate_rank(item), reverse=True)
                keep = {item.state_id for item in ranked[:1] if item.feasible}
            else:
                ranked = sorted(generation, key=lambda item: _candidate_rank(item), reverse=True)
                keep = {item.state_id for item in ranked[: max(1, int(recipe.beam_width))] if item.feasible}
            frontier = [item for item in next_frontier if item.parent_state_id in keep or not keep]
            if not frontier:
                termination_reason = "frontier_pruned"
                break
        observed_depth = max((int(item.get("depth", 0)) for item in states), default=0)
        if observed_depth >= int(recipe.max_search_depth) and termination_reason == "frontier_empty_or_max_depth":
            termination_reason = "max_depth_reached"
        profiling_cost = profile.get("profiling_cost") if isinstance(profile.get("profiling_cost"), Mapping) else {}
        if not profiling_cost and isinstance(profile.get("tica"), Mapping):
            raw_cost = profile["tica"].get("profiling_cost")
            profiling_cost = raw_cost if isinstance(raw_cost, Mapping) else {}
        total_cost = {
            "search_fast_evals": int(budget.used["fast_evals"]),
            "search_full_evals": int(budget.used["full_evals"]),
            "search_interaction_evals": int(budget.used["interaction_evals"]),
            "profiling": dict(profiling_cost),
            "scope": "search_plus_shared_profiling",
        }
        pareto = _pareto_front(candidates)
        feasible = [item for item in candidates if item.feasible and item.full_metrics]
        pareto_ids = {str(item.get("candidate_id")) for item in pareto}
        pareto_candidates = [item for item in feasible if item.candidate_id in pareto_ids]
        selected = max(pareto_candidates or feasible, key=_candidate_rank, default=None)
        return TICASearchResult(
            recipe=recipe,
            candidates=tuple(candidates),
            pareto_front=tuple(pareto),
            budget={**budget.snapshot(), "elapsed_s": time.perf_counter() - started},
            calibration_digest=str(manifest.digest()),
            evidence_level="cpu_fake_quality_proxy" if getattr(manifest, "provider_name", "") == "fake_wan_cpu" else "real_wan_structure",
            score=score.to_dict(),
            selected_candidate_id=selected.candidate_id if selected else None,
            selected_plan_digest=selected.plan_digest if selected else None,
            search_states=tuple(states),
            budget_audit=tuple(budget.audit),
            seed_plan=dict(seed_plan or {}),
            observed_search_depth=observed_depth,
            level_stats=tuple(level_stats),
            termination_reason=termination_reason,
            profiling_cost=dict(profiling_cost),
            total_cost=total_cost,
        )

    def measure_interaction(self, independent: Sequence[Mapping[str, Any]], joint: Mapping[str, Any]) -> dict[str, Any]:
        """Calculate interaction from measured errors, never from a constant."""

        values = [self._risk_evidence(item, stage="full") for item in independent]
        joint_evidence = self._risk_evidence(joint, stage="full")
        if (
            not all(value is not None for value in values)
            or joint_evidence is None
            or not all(item.get("valid") is True for item in independent)
            or joint.get("valid") is not True
        ):
            return {"available": False, "interaction_delta": None, "reason": "missing_or_invalid_measured_quality_error"}
        independent_sum = float(sum(float(value[0]) for value in values if value is not None))
        return {
            "available": True,
            "joint_error": float(joint_evidence[0]),
            "independent_error_sum": independent_sum,
            "interaction_delta": float(joint_evidence[0]) - independent_sum,
        }


def _pareto_front(candidates: Sequence[TICACandidate]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not candidate.feasible:
            continue
        metrics = candidate.full_metrics
        quality = metrics.get("quality_error")
        reduction = metrics.get("parameter_reduction_ratio")
        if not isinstance(quality, (int, float)) or not isinstance(reduction, (int, float)):
            continue
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_metrics = other.full_metrics
            other_quality = other_metrics.get("quality_error")
            other_reduction = other_metrics.get("parameter_reduction_ratio")
            if not isinstance(other_quality, (int, float)) or not isinstance(other_reduction, (int, float)):
                continue
            if float(other_quality) <= float(quality) and float(other_reduction) >= float(reduction) and (float(other_quality) < float(quality) or float(other_reduction) > float(reduction)):
                dominated = True
                break
        if not dominated:
            rows.append({"candidate_id": candidate.candidate_id, "kind": candidate.kind, **dict(metrics)})
    return rows


__all__ = [
    "BASELINE_NAMES",
    "TICABudgetManager",
    "TICACandidate",
    "TICAPruner",
    "TICARecipe",
    "TICAScore",
    "TICASearchState",
    "TICASearchResult",
    "build_human_tica_source",
    "normalized_time_weights",
    "recipe_plan",
    "tica_score",
    "tica_operator_spec",
]
