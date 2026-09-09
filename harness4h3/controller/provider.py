from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

from .context import ControllerContext
from .schemas import ExperimentPlan


class ControllerProvider(Protocol):
    provider_name: str
    model_name: str

    def plan(self, context: ControllerContext) -> Any:
        ...


class ControllerProviderError(RuntimeError):
    pass


def experiment_plan_json_schema(context: ControllerContext) -> Mapping[str, Any]:
    operator_names = [str(item["name"]) for item in context.available_operators]
    argument_properties: Dict[str, Mapping[str, Any]] = {}
    for operator in context.available_operators:
        for name, kind in (operator.get("input_schema") or {}).items():
            kind_name = str(kind).split("/")[0]
            argument_properties[str(name)] = {
                "type": {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}.get(kind_name, "string")
            }
    next_experiment_id = "exp_%04d" % (context.budget_state.used_iterations + 1)
    max_quality_drop = context.target_profile.max_quality_drop
    min_quality_score = context.target_profile.min_quality_score
    nullable_string = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "experiment_id": {"type": "string", "const": next_experiment_id},
            "parent_model_id": {"type": "string", "const": context.current_model_state.model_id},
            "diagnosis": {"type": "string"},
            "objective": {"type": "string"},
            "hypothesis": {"type": "string"},
            "operator": {"type": "string", "enum": operator_names},
            "operator_args": {
                "type": "object",
                "additionalProperties": False,
                "properties": argument_properties,
            },
            "expected_effects": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "quality_score": nullable_string,
                    "latency_s": nullable_string,
                    "peak_memory_gb": nullable_string,
                    "model_size_gb": nullable_string,
                    "energy_j": nullable_string,
                    "sampling_steps": nullable_string,
                },
                "required": ["quality_score", "latency_s", "peak_memory_gb", "model_size_gb", "energy_j", "sampling_steps"],
            },
            "risks": {"type": "array", "items": {"type": "string"}},
            "required_budget": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "wall_time_s": {"type": "number", "minimum": 0},
                    "gpu_hours": {"type": "number", "minimum": 0},
                    "controller_calls": {"type": "integer", "const": 0},
                },
                "required": ["wall_time_s", "gpu_hours", "controller_calls"],
            },
            "acceptance": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_quality_drop": {"type": ["number", "null"], "const": max_quality_drop},
                    "min_quality_score": {"type": ["number", "null"], "const": min_quality_score},
                },
                "required": ["max_quality_drop", "min_quality_score"],
            },
            "stop_conditions": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "critical_regression": {"type": "boolean"},
                    "target_satisfied": {"type": "boolean"},
                    "budget_exhausted": {"type": "boolean"},
                },
                "required": ["critical_regression", "target_satisfied", "budget_exhausted"],
            },
            "rationale": {"type": "string"},
        },
        "required": [
            "experiment_id",
            "parent_model_id",
            "diagnosis",
            "objective",
            "hypothesis",
            "operator",
            "operator_args",
            "expected_effects",
            "risks",
            "required_budget",
            "acceptance",
            "stop_conditions",
            "rationale",
        ],
    }


def _controller_prompt(context: ControllerContext, schema: Mapping[str, Any]) -> str:
    next_number = context.budget_state.used_iterations + 1
    operator_arguments = {
        str(item["name"]): sorted(str(name) for name in (item.get("input_schema") or {}))
        for item in context.available_operators
    }
    return (
        "You are the fixed planning controller for EvoGen-RSI Phase I. "
        "Return exactly one ExperimentPlan JSON object. Select only a listed operator; never emit shell, code, paths, "
        "evaluator changes, benchmark changes, or target changes. Address the first blocking hard constraint, make one "
        "primary model modification, preserve quality, and declare at least the registered operator cost. If model size "
        "or memory is blocked and the model is not yet 4-bit, prefer quantize before step_distill. If a recent experiment "
        "was rejected, change the intervention instead of repeating it. Copy the TargetProfile quality limits exactly into "
        "acceptance: use null for a limit absent from TargetProfile and never invent a stricter limit. operator_args "
        "must contain only the exact keys listed for the selected operator; keys belonging to any other operator are "
        "forbidden. Use quantize bits=4 when fake int4 quantization is selected. Use exactly "
        "experiment_id exp_%04d and parent_model_id %s. The complete JSON schema and controller context follow.\n"
        "OPERATOR_ARGUMENTS=%s\nSCHEMA=%s\nCONTEXT=%s"
        % (
            next_number,
            context.current_model_state.model_id,
            json.dumps(operator_arguments, ensure_ascii=False, sort_keys=True),
            json.dumps(schema, ensure_ascii=False, sort_keys=True),
            json.dumps(context.to_dict(), ensure_ascii=False, sort_keys=True),
        )
    )


def _post_json(url: str, payload: Mapping[str, Any], timeout_s: float, headers: Optional[Mapping[str, str]] = None) -> Mapping[str, Any]:
    request_headers = {"Accept": "application/json", "Content-Type": "application/json", **dict(headers or {})}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ControllerProviderError("controller request failed: %s" % exc)
    if not isinstance(raw, Mapping):
        raise ControllerProviderError("controller response must be a JSON object")
    return raw


@dataclass
class OllamaStructuredController:
    model_name: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 180.0
    provider_name: str = "ollama"
    last_request_id: Optional[str] = None

    def plan(self, context: ControllerContext) -> ExperimentPlan:
        schema = experiment_plan_json_schema(context)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": _controller_prompt(context, schema)}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        raw = _post_json(self.base_url.rstrip("/") + "/api/chat", payload, self.timeout_s)
        self.last_request_id = str(raw.get("created_at") or "") or None
        try:
            content = raw["message"]["content"]
            parsed = json.loads(content)
            return ExperimentPlan.from_dict(parsed)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ControllerProviderError("ollama returned an invalid ExperimentPlan: %s" % exc)


@dataclass
class OpenAIResponsesController:
    model_name: str
    endpoint: str = "https://api.openai.com/v1/responses"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 180.0
    provider_name: str = "openai"
    last_request_id: Optional[str] = None

    def plan(self, context: ControllerContext) -> ExperimentPlan:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise ControllerProviderError("%s is not set" % self.api_key_env)
        schema = experiment_plan_json_schema(context)
        payload = {
            "model": self.model_name,
            "instructions": "Return one safe model-optimization ExperimentPlan. Do not call tools or output executable code.",
            "input": _controller_prompt(context, schema),
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "experiment_plan", "strict": True, "schema": schema}},
        }
        raw = _post_json(self.endpoint, payload, self.timeout_s, {"Authorization": "Bearer " + api_key})
        self.last_request_id = str(raw.get("id") or "") or None
        output_text = raw.get("output_text")
        if not output_text:
            for item in raw.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        output_text = content["text"]
                        break
                if output_text:
                    break
        try:
            return ExperimentPlan.from_dict(json.loads(str(output_text)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ControllerProviderError("OpenAI returned an invalid ExperimentPlan: %s" % exc)


class RuleBasedMockController:
    provider_name = "offline"
    model_name = "rule-based-mock"

    def plan(self, context: ControllerContext) -> ExperimentPlan:
        state = context.current_model_state
        target = context.target_profile
        metrics = state.measured_metrics
        quantized = int(state.quantization.get("bits", 16)) <= 4
        memory_blocked = target.max_peak_memory_gb is not None and float(metrics["peak_memory_gb"]) > target.max_peak_memory_gb
        size_blocked = target.max_model_size_gb is not None and float(metrics["model_size_gb"]) > target.max_model_size_gb
        latency_blocked = target.max_latency_s is not None and float(metrics["latency_s"]) > target.max_latency_s
        if (memory_blocked or size_blocked) and not quantized:
            operator = "quantize"
            args: Mapping[str, Any] = {"bits": 4}
            diagnosis = "memory_or_model_size"
            objective = "reduce model size, memory, and latency"
            hypothesis = "four-bit fake quantization reduces memory with bounded quality loss"
            effects = {"peak_memory_gb": "decrease", "model_size_gb": "decrease", "latency_s": "decrease"}
            required = {"wall_time_s": 0.2, "gpu_hours": 0.0}
        elif latency_blocked or memory_blocked:
            operator = "step_distill"
            args = {"target_steps": 8}
            diagnosis = "latency_or_residual_memory"
            objective = "reduce sampling latency and residual working memory"
            hypothesis = "fake step distillation reduces sampling work while preserving the quality floor"
            effects = {"latency_s": "decrease", "peak_memory_gb": "decrease"}
            required = {"wall_time_s": 0.3, "gpu_hours": 0.01}
        else:
            operator = "inspect"
            args = {}
            diagnosis = "no_blocking_constraint"
            objective = "confirm normalized model state"
            hypothesis = "inspection will confirm that no model modification is needed"
            effects = {"model_state": "unchanged"}
            required = {"wall_time_s": 0.02, "gpu_hours": 0.0}
        number = context.budget_state.used_iterations + 1
        acceptance = {"max_quality_drop": target.max_quality_drop if target.max_quality_drop is not None else 1.0}
        if target.min_quality_score is not None:
            acceptance["min_quality_score"] = target.min_quality_score
        return ExperimentPlan(
            experiment_id="exp_%04d" % number,
            parent_model_id=state.model_id,
            diagnosis=diagnosis,
            objective=objective,
            hypothesis=hypothesis,
            operator=operator,
            operator_args=args,
            expected_effects=effects,
            risks=["quality regression"],
            required_budget=required,
            acceptance=acceptance,
            stop_conditions={"critical_regression": True},
            rationale="select the first registered operator that addresses the current hard constraint",
        )
