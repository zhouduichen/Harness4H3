from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .candidate import AlgorithmCandidate


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


def _seed_source(
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
        payload = {
            "profile": _compact_profile(profile),
            "history": _compact_context_items(history),
            "skills": _compact_context_items(skills),
            "pareto_front": _compact_pareto_front(pareto_front),
            "parent_id": parent_id,
            "search_policy": {"choice": search_choice},
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


def _compact_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider prompts bounded while retaining compression evidence."""

    compact: dict[str, Any] = {}
    for key in ("model", "provenance", "schema_version", "timestep_sensitivity"):
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
        prompt = (
            "Generate one compression algorithm for a trusted Wan DiT harness. "
            "Return JSON only with source, hypothesis, and parent_id. The source must define "
            "a complete Python function `def propose(profile, history): ...` and return only "
            "a bounded operator/args plan; never return a function name or pseudocode. "
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
            + json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
        )
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
                "format": ALGORITHM_RESPONSE_SCHEMA,
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
                # Algorithm responses contain executable source as well as
                # the plan metadata.  1024 tokens can truncate a valid JSON
                # response mid-string, which is an infrastructure failure
                # rather than an invalid algorithm.
                "max_tokens": 2048,
            }
        else:
            request_payload = {
                "model": model_name,
                "instructions": "Return JSON only; do not call tools or output unsafe execution instructions.",
                "input": prompt,
                "store": False,
                "text": {"format": {"type": "json_schema", "name": "compression_algorithm", "strict": True, "schema": ALGORITHM_RESPONSE_SCHEMA}},
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
