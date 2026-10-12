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
    },
}


def _seed_source(block_id: str, algorithm_id: str, search_choice: str) -> str:
    if search_choice == "explore":
        return (
            "def propose(profile, history):\n"
            "    # Explore seed %s: a paired FFN-width hypothesis across all profiled blocks.\n"
            "    block_ids = [item['block_id'] for item in profile.get('layers', [])]\n"
            "    width = int(profile['model']['ffn_width'])\n"
            "    retained_indices = list(range(max(1, width // 2)))\n"
            "    return {'operator': 'prune_ffn_channels', 'args': {'block_ids': block_ids, 'retained_indices': retained_indices}}\n"
        ) % algorithm_id
    return (
        "def propose(profile, history):\n"
        "    # Seed algorithm %s selects the least output-sensitive block reported by the profiler.\n"
        "    selected = %r\n"
        "    return {'operator': 'prune_blocks', 'args': {'block_ids': [selected]}}\n" % (algorithm_id, block_id)
    )


@dataclass(frozen=True)
class AlgorithmAgent:
    provider: Optional[Provider] = None

    def generate(
        self,
        algorithm_id: str,
        *,
        parent_id: Optional[str],
        profile: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        skills: Sequence[Mapping[str, Any]] = (),
        search_choice: str = "exploit",
    ) -> AlgorithmCandidate:
        if self.provider is None:
            block_id = self._least_sensitive_block(profile)
            source = _seed_source(block_id, algorithm_id, search_choice)
            return AlgorithmCandidate.from_source(
                algorithm_id,
                source,
                parent_id=parent_id,
                hypothesis=("explore paired FFN width pruning" if search_choice == "explore" else "remove the lowest measured leave-one-block-out sensitivity block"),
                metadata={"generator": "deterministic_seed", "operator": ("prune_ffn_channels" if search_choice == "explore" else "prune_blocks"), "search_choice": search_choice},
            )
        payload = {
            "profile": _compact_profile(profile),
            "history": [dict(item) for item in history],
            "skills": [dict(item) for item in skills],
            "parent_id": parent_id,
            "search_policy": {"choice": search_choice},
        }
        raw = self.provider(payload)
        if not isinstance(raw, Mapping):
            raise ValueError("algorithm_generation_invalid: provider response must be an object")
        unknown = sorted(set(raw) - {"source", "hypothesis", "parent_id"})
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
            hypothesis=str(raw.get("hypothesis", "")),
            metadata={"generator": "structured_provider", "search_choice": search_choice},
        )

    @staticmethod
    def _least_sensitive_block(profile: Mapping[str, Any]) -> str:
        records = profile.get("block_sensitivity", [])
        candidates = [item for item in records if isinstance(item, Mapping) and isinstance(item.get("block_id"), str) and isinstance(item.get("importance_score"), (int, float))]
        if candidates:
            return str(min(candidates, key=lambda item: float(item["importance_score"]))["block_id"])
        layers = profile.get("layers", [])
        for item in layers:
            if isinstance(item, Mapping) and isinstance(item.get("block_id"), str):
                return str(item["block_id"])
        return "block.0"


def _compact_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider prompts bounded while retaining compression evidence."""

    compact: dict[str, Any] = {}
    for key in ("model", "provenance", "schema_version", "timestep_sensitivity"):
        if key in profile:
            compact[key] = profile[key]
    compact["layers"] = [
        {
            field: item[field]
            for field in ("block_id", "index", "parameter_count", "estimated_flops", "latency_ms", "memory_delta_bytes", "module_path", "measurement")
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
        )[:16]
        channel_records.append(record)
    compact["channel_sensitivity"] = channel_records
    return compact


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
                "max_tokens": 4096,
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
        return dict(parsed)

    return call
