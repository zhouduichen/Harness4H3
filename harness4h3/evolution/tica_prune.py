"""Human-designed TICA-Prune scoring and bounded search primitives."""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from ..adapters.wan.contracts import stable_digest


BASELINE_NAMES = (
    "independent_block",
    "tica_block",
    "interaction_block",
    "tica_block_ffn",
)


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
    activation_weight: float = 0.5
    interaction_weight: float = 0.5
    beam_width: int = 2
    max_fast_evals: int = 8
    max_full_evals: int = 4
    max_interaction_evals: int = 2
    strategy: str = "greedy_beam"
    time_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.baseline_name not in BASELINE_NAMES:
            raise ValueError("unsupported TICA baseline: %s" % self.baseline_name)
        if self.time_weight_formula not in {"uniform", "early_log", "late_log"}:
            raise ValueError("unsupported TICA time weight formula: %s" % self.time_weight_formula)
        if not 0.0 <= float(self.risk_limit) <= 1.0:
            raise ValueError("TICA risk_limit must be in [0, 1]")
        if not 0.0 <= float(self.activation_weight) <= 1.0:
            raise ValueError("TICA activation_weight must be in [0, 1]")
        if not 0.0 <= float(self.interaction_weight) <= 1.0:
            raise ValueError("TICA interaction_weight must be in [0, 1]")
        for name in ("beam_width", "max_fast_evals", "max_full_evals", "max_interaction_evals"):
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

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TICASearchResult:
    recipe: TICARecipe
    candidates: tuple[TICACandidate, ...]
    pareto_front: tuple[Mapping[str, Any], ...]
    budget: Mapping[str, int]
    calibration_digest: str
    evidence_level: str
    score: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "pareto_front": [_jsonable(item) for item in self.pareto_front],
            "budget": dict(self.budget),
            "calibration_digest": self.calibration_digest,
            "evidence_level": self.evidence_level,
            "score": _jsonable(self.score),
        }


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
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    per_timestep = record.get("per_timestep") or record.get("timesteps")
    if isinstance(per_timestep, Mapping):
        value = per_timestep.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _channel_value(record: Mapping[str, Any]) -> float | None:
    for key in ("contribution", "activation_contribution", "importance_score"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def tica_score(
    profile: Mapping[str, Any],
    *,
    time_weight_formula: str = "early_log",
    timesteps: Sequence[str] | None = None,
) -> TICAScore:
    """Compute profile-derived TICA scores without executing a model."""

    names = tuple(str(item) for item in (timesteps or _timestep_names(profile)))
    weights = normalized_time_weights(names, time_weight_formula)
    block_risk: dict[str, float] = {}
    block_records = profile.get("tica_block_sensitivity") or profile.get("block_sensitivity") or ()
    for item in block_records:
        if not isinstance(item, Mapping) or not item.get("block_id"):
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


def _retained_indices(profile: Mapping[str, Any], score: TICAScore) -> list[int]:
    width = int((profile.get("model") or {}).get("ffn_width", 0) or 0)
    if width <= 1:
        return [0]
    ordered = sorted(score.channel_contribution, key=lambda channel: (-score.channel_contribution[channel], channel))
    target = max(1, min(width - 1, int(width * 0.75)))
    if ordered:
        return sorted(int(item) for item in ordered[:target])
    return list(range(target))


def recipe_plan(profile: Mapping[str, Any], recipe: TICARecipe) -> dict[str, Any]:
    score = tica_score(profile, time_weight_formula=recipe.time_weight_formula)
    blocks = _profile_block_ids(profile, score)
    if not blocks:
        raise ValueError("TICA profile has no block candidates")
    selected = blocks[:2] if recipe.baseline_name == "interaction_block" else blocks[:1]
    mode = "block"
    if recipe.baseline_name == "tica_block_ffn":
        mode = "block_plus_ffn"
    return {
        "block_ids": selected,
        "retained_indices": _retained_indices(profile, score),
        "mode": mode,
        "time_weight_formula": recipe.time_weight_formula,
        "risk_limit": float(recipe.risk_limit),
        "activation_weight": float(recipe.activation_weight),
        "interaction_weight": float(recipe.interaction_weight),
        "beam_width": int(recipe.beam_width),
        "max_fast_evals": int(recipe.max_fast_evals),
        "max_full_evals": int(recipe.max_full_evals),
        "max_interaction_evals": int(recipe.max_interaction_evals),
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
    names = ['high_noise', 'middle', 'low_noise']
    raw_weights = [math.log1p(1.0 + 9.0 * value) for value in [0.9, 0.5, 0.1]]
    total = sum(raw_weights)
    weights = [value / total for value in raw_weights]
    records = []
    for item in profile.get('tica_block_sensitivity', profile.get('block_sensitivity', [])):
        values = item.get('per_timestep', item.get('timesteps', {}))
        score = sum(weights[index] * float(values.get(names[index], item.get('importance_score', 0.0))) for index in range(3))
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
    retained = sorted(set(item[1] for item in sorted(channels, reverse=True)[:target])) if channels else list(range(target))
    return {'operator': 'tica_prune', 'args': {
        'block_ids': [selected], 'retained_indices': retained,
        'mode': 'block_plus_ffn', 'time_weight_formula': 'early_log',
        'risk_limit': 0.25, 'activation_weight': 0.5,
        'interaction_weight': 0.5, 'beam_width': 2,
        'max_fast_evals': 8, 'max_full_evals': 4,
        'max_interaction_evals': 2,
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
    handles = []
    for block in blocks:
        try:
            module = _module_at(model, block.module_path)
            activation = next((getattr(module, name) for name in ("activation", "act") if hasattr(module, name)), None)
            down = getattr(module, "down_proj", None)
            if activation is None or down is None or not hasattr(down, "weight"):
                continue
            width = int(down.weight.shape[1])
            totals[block.block_id] = torch.zeros(width, device=down.weight.device, dtype=torch.float32)
            counts[block.block_id] = 0

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
        module = _module_at(model, block.module_path)
        down = getattr(module, "down_proj", None)
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
        averaged = {
            name: sum(value for value in values if math.isfinite(value)) / max(len([value for value in values if math.isfinite(value)]), 1)
            for name, values in per_timestep.items()
        }
        block_rows.append({"block_id": block.block_id, "per_timestep": averaged, "available": bool(averaged), "method": "local_denoising_relative_error", "sample_count": len(samples)})
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
        },
    }


EvaluateCandidate = Callable[[Mapping[str, Any], bool, str], Mapping[str, Any]]


class TICAPruner:
    """Generate bounded TICA candidates and optionally measure them."""

    def search(
        self,
        adapter: Any,
        parent: Any,
        manifest: Any,
        profile: Mapping[str, Any],
        recipe: TICARecipe,
        *,
        evaluate_candidate: Optional[EvaluateCandidate] = None,
    ) -> TICASearchResult:
        del adapter, parent
        started = time.perf_counter()
        score = tica_score(profile, time_weight_formula=recipe.time_weight_formula)
        candidates: list[TICACandidate] = []
        used = {"fast_evals": 0, "full_evals": 0, "interaction_evals": 0}
        baseline_recipes = (
            TICARecipe.from_mapping({**recipe.to_dict(), "baseline_name": name})
            for name in BASELINE_NAMES
        )
        for index, baseline in enumerate(baseline_recipes):
            if used["full_evals"] >= recipe.max_full_evals:
                break
            plan = recipe_plan(profile, baseline)
            fast_metrics = dict(evaluate_candidate(plan, False, "fast") if evaluate_candidate else {})
            used["fast_evals"] += 1
            full_metrics = dict(evaluate_candidate(plan, True, "full") if evaluate_candidate else {})
            used["full_evals"] += 1
            interaction_metrics: Mapping[str, Any] = {}
            if baseline.baseline_name in {"interaction_block", "tica_block_ffn"} and used["interaction_evals"] < recipe.max_interaction_evals:
                interaction_metrics = dict(evaluate_candidate(plan, True, "interaction") if evaluate_candidate else {})
                used["interaction_evals"] += 1
            candidates.append(
                TICACandidate(
                    candidate_id="TICA-C%04d" % (index + 1),
                    kind=baseline.baseline_name,
                    plan=plan,
                    fast_metrics=fast_metrics,
                    full_metrics=full_metrics,
                    interaction_metrics=interaction_metrics,
                    budget_cost=dict(used),
                    status="measured" if evaluate_candidate else "generated",
                )
            )
        pareto = _pareto_front(candidates)
        return TICASearchResult(
            recipe=recipe,
            candidates=tuple(candidates),
            pareto_front=tuple(pareto),
            budget={**used, "elapsed_s": time.perf_counter() - started},
            calibration_digest=str(manifest.digest()),
            evidence_level="cpu_fake_quality_proxy" if getattr(manifest, "provider_name", "") == "fake_wan_cpu" else "real_wan_structure",
            score=score.to_dict(),
        )

    def measure_interaction(self, independent: Sequence[Mapping[str, Any]], joint: Mapping[str, Any]) -> dict[str, Any]:
        """Calculate interaction from measured errors, never from a constant."""

        values = [item.get("quality_error") for item in independent]
        joint_error = joint.get("quality_error")
        if not all(isinstance(value, (int, float)) for value in values) or not isinstance(joint_error, (int, float)):
            return {"available": False, "interaction_delta": None, "reason": "missing_measured_quality_error"}
        independent_sum = float(sum(float(value) for value in values))
        return {
            "available": True,
            "joint_error": float(joint_error),
            "independent_error_sum": independent_sum,
            "interaction_delta": float(joint_error) - independent_sum,
        }


def _pareto_front(candidates: Sequence[TICACandidate]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for candidate in candidates:
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
    "TICACandidate",
    "TICAPruner",
    "TICARecipe",
    "TICAScore",
    "TICASearchResult",
    "build_human_tica_source",
    "normalized_time_weights",
    "recipe_plan",
    "tica_score",
]
