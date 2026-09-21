from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness4h3.controller.context import ControllerContext
from harness4h3.controller.directive import HumanDirective
from harness4h3.controller.provider import (
    ControllerProviderError,
    ControllerUnavailableError,
    OllamaStructuredController,
    OpenAICompatibleController,
    OpenAIResponsesController,
    RuleBasedMockController,
    build_controller_from_config,
    experiment_plan_json_schema,
)
from harness4h3.controller.provider import (
    _compact_experience,
    _compact_observation,
    _compact_quality_metrics,
    _controller_prompt,
    _parse_json_object,
    _prompt_context,
    _safe_completion_tokens,
)
from harness4h3.controller.schemas import BudgetState, ExperimentPlan
from harness4h3.h3.state import ModelState
from harness4h3.operators.fake import build_fake_registry
from harness4h3.target.profile import TargetProfile


def context():
    return ControllerContext(
        TargetProfile("mobile", "mobile", "fake", max_peak_memory_gb=6, max_latency_s=30, max_quality_drop=0.05),
        ModelState.fake_baseline(),
        BudgetState(4, 2, max_controller_calls=4),
        build_fake_registry().visible(),
    )


def plan_payload():
    return {
        "experiment_id": "exp_0001",
        "parent_model_id": "M0000",
        "diagnosis": "memory",
        "objective": "reduce memory",
        "hypothesis": "int4 quantization reduces memory",
        "operator": "quantize",
        "operator_args": {"bits": 4},
        "expected_effects": {
            "quality_score": "slight decrease",
            "latency_s": "decrease",
            "peak_memory_gb": "decrease",
            "model_size_gb": "decrease",
            "energy_j": "decrease",
            "sampling_steps": None,
        },
        "risks": ["quality regression"],
        "required_budget": {"wall_time_s": 0.2, "gpu_hours": 0, "controller_calls": 0},
        "acceptance": {"max_quality_drop": 0.05, "min_quality_score": None},
        "stop_conditions": {"critical_regression": True, "target_satisfied": True, "budget_exhausted": True},
        "rationale": "memory is the first hard constraint",
        "resource_request": {
            "gpu_count": 0,
            "min_gpu_count": 0,
            "max_gpu_count": 0,
            "elastic": False,
            "distributed": False,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "replan",
        },
    }


def round_policy_payload():
    return {
        "schema_version": 1,
        "round_id": "R0001",
        "substrate_digest": "sha256:substrate",
        "search_mode": "runtime_efficiency",
        "allowed_operators": ["quantize"],
        "axis_budget": {"max_trials": 2, "max_gpu_hours": 1.0},
        "objective": {"quality_floor": 0.8},
        "fixed_evaluation": {"split": "heldout", "recipe_digest": "sha256:evaluator"},
        "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
        "stop_conditions": ["critical_regression", "budget_exhausted"],
        "source_observation_ids": ["obs-1"],
        "created_at": "2026-09-19T00:00:00+00:00",
    }


def test_experiment_plan_round_policy_is_optional_but_strictly_carried():
    raw = {**plan_payload(), "round_policy": round_policy_payload()}
    plan = ExperimentPlan.from_dict(raw)
    assert plan.round_policy == raw["round_policy"]
    assert plan.to_dict()["round_policy"]["round_id"] == "R0001"

    with pytest.raises(ValueError, match="round_policy"):
        ExperimentPlan.from_dict({**plan_payload(), "round_policy": "not-an-object"})


def test_experiment_plan_schema_exposes_round_policy_contract():
    policy = experiment_plan_json_schema(context())["properties"]["round_policy"]
    assert policy["additionalProperties"] is False
    assert "uniqueItems" not in policy["properties"]["allowed_operators"]
    assert set(policy["required"]) == {
        "schema_version",
        "round_id",
        "substrate_digest",
        "search_mode",
        "allowed_operators",
        "axis_budget",
        "objective",
        "fixed_evaluation",
        "resource_policy",
        "stop_conditions",
        "source_observation_ids",
        "created_at",
    }


def test_prompt_context_preserves_bounded_discovery_telemetry():
    value = context()
    value = ControllerContext(
        value.target_profile,
        value.current_model_state,
        value.budget_state,
        value.available_operators,
        discovery_digest={
            "schema_version": 1,
            "source_digest": "sha256:digest",
            "telemetry": {
                "items": [
                    {
                        "window_events": 3,
                        "underutilized_gpu_indices": ["2"],
                        "power_target_w": 300.0,
                        "last_power_feedback": {
                            "target_power_w": 300.0,
                            "per_gpu": {
                                "2": {
                                    "power_w_avg": 120.0,
                                    "utilization_gpu_pct_avg": 45.0,
                                    "lane": "controller",
                                }
                            },
                        },
                    }
                ]
            },
        },
    )

    compact = _prompt_context(value)

    telemetry = compact["discovery_digest"]["telemetry"]["items"][0]["telemetry"]
    assert telemetry["underutilized_gpu_indices"] == ["2"]
    assert telemetry["last_power_feedback"]["per_gpu"]["2"]["lane"] == "controller"


@pytest.fixture
def controller_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            requests.append((self.path, payload, dict(self.headers)))
            if self.path == "/api/chat":
                response = {"created_at": "request-1", "message": {"content": json.dumps(plan_payload())}}
            elif self.path == "/v1/chat/completions":
                schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
                content = (
                    {
                        "action": "review_only",
                        "reason": "telemetry is healthy",
                        "evidence_ids": ["evt-1"],
                        "confidence": 0.8,
                        "next_review_after_s": 60,
                        "risks": [],
                    }
                    if schema_name == "controller_review"
                    else plan_payload()
                )
                response = {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(content)},
                            "finish_reason": "stop",
                        }
                    ],
                }
            else:
                response = {"id": "resp-1", "output_text": json.dumps(plan_payload())}
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_port, requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_ollama_provider_sends_schema_and_parses_experiment_plan(controller_server):
    url, requests = controller_server
    provider = OllamaStructuredController("qwen", url)
    plan = provider.plan(context())
    assert plan.operator == "quantize"
    request = requests[0][1]
    assert request["stream"] is False
    assert request["think"] is False
    assert request["format"]["properties"]["operator"]["enum"] == ["inspect", "quantize", "step_distill", "rollback"]
    assert request["format"]["properties"]["experiment_id"]["const"] == "exp_0001"
    assert request["format"]["properties"]["parent_model_id"]["const"] == "M0000"
    assert request["format"]["properties"]["acceptance"]["properties"]["max_quality_drop"]["const"] == 0.05
    assert request["format"]["properties"]["acceptance"]["properties"]["min_quality_score"]["const"] is None
    assert request["format"]["properties"]["required_budget"]["properties"]["controller_calls"]["const"] == 0


def test_controller_schema_declares_elastic_resource_range():
    resource = experiment_plan_json_schema(context())["properties"]["resource_request"]
    assert resource["required"] == [
        "gpu_count",
        "min_gpu_count",
        "max_gpu_count",
        "elastic",
        "distributed",
        "exclusive",
        "evaluation_workers",
        "on_unavailable",
    ]
    assert resource["properties"]["min_gpu_count"]["minimum"] == 0
    assert resource["properties"]["max_gpu_count"]["maximum"] == 4


def test_controller_schema_preserves_operator_argument_enum():
    operators = (
        {"name": "quantize", "description": "INT8", "input_schema": {"bits": {"type": "int", "enum": [8]}}},
    )
    value = ControllerContext(
        context().target_profile,
        context().current_model_state,
        context().budget_state,
        operators,
    )
    assert experiment_plan_json_schema(value)["properties"]["operator_args"]["properties"]["bits"] == {
        "type": "integer",
        "enum": [8],
    }


def test_rule_controller_uses_configured_quantize_variant():
    operators = (
        {"name": "quantize", "description": "INT8", "input_schema": {"bits": {"type": "int", "enum": [8]}}},
    )
    value = ControllerContext(
        TargetProfile("l40", "gpu", "l40x4", max_model_size_gb=6),
        ModelState.fake_baseline(),
        BudgetState(4, 2),
        operators,
    )
    plan = RuleBasedMockController().plan(value)
    assert plan.operator == "quantize"
    assert plan.operator_args == {"bits": 8}


def test_unconsumed_human_directive_is_preserved_in_bounded_prompt_context():
    directive = HumanDirective.create("下一轮优先降低 peak_memory", directive_id="goal-001")
    observation = directive.to_observation().to_dict()
    value = ControllerContext(
        context().target_profile,
        context().current_model_state,
        context().budget_state,
        context().available_operators,
        observations=[
            {"observation_id": "old-%d" % index, "kind": "evaluation", "summary": {"quality_score": 0.8}}
            for index in range(8)
        ]
        + [observation],
        unconsumed_observation_ids=[observation["observation_id"]],
    )

    compact = _prompt_context(value)
    prompt = _controller_prompt(value, experiment_plan_json_schema(value))
    assert any(item.get("observation_id") == observation["observation_id"] for item in compact["observations"])
    directive_view = next(item for item in compact["observations"] if item.get("kind") == "human_directive")
    assert directive_view["summary"]["instruction"] == "下一轮优先降低 peak_memory"
    assert observation["observation_id"] in compact["unconsumed_observation_ids"]
    assert "advisory objectives for this next plan" in prompt
    assert "下一轮优先降低 peak_memory" in prompt
    assert "cannot override hard gates" in prompt
    assert "EXACT_NEW_OBSERVATION_IDS" in prompt
    assert observation["observation_id"] in prompt


def test_prompt_context_truncates_oversized_human_directive():
    observation = {
        "observation_id": "directive-oversized",
        "kind": "human_directive",
        "summary": {
            "directive_id": "oversized",
            "instruction": "x" * 5000,
            "apply_at": "next_boundary",
        },
    }

    compact = _compact_observation(observation)

    assert len(compact["summary"]["instruction"]) == 1200
    assert compact["summary"]["instruction_truncated"] is True


def test_prompt_makes_bounded_resource_wait_cpu_only():
    base = context()
    value = ControllerContext(
        base.target_profile,
        base.current_model_state,
        base.budget_state,
        tuple(
            item
            for item in base.available_operators
            if item.get("name") in {"prune_blocks", "quantize"}
        ),
        planning_intent="resource_recovery_cpu",
    )

    prompt = _controller_prompt(value, experiment_plan_json_schema(value))

    assert "resource_recovery_cpu" in prompt
    assert "only one of the listed CPU-only operators prune_blocks or quantize" in prompt
    assert "Do not select distill, recovery_finetune, step_distill, or dmd2" in prompt


def test_primary_prompt_requests_gpu_candidate_diversity_for_overlap():
    prompt = _controller_prompt(context(), experiment_plan_json_schema(context()))

    assert "preserve useful diversity" in prompt
    assert "distributed GPU-training candidate" in prompt
    assert "isolated speculative sibling" in prompt
    assert "keep min_gpu_count=2 and max_gpu_count=4" in prompt
    assert "SCHEMA=" not in prompt


def test_prompt_context_preserves_previous_benchmark_recipe_without_raw_task_payloads():
    prior = {
        "experience_id": "xp-eval-M0001",
        "experiment_id": "exp_0001",
        "parent_model_id": "M0000",
        "child_model_id": "M0001",
        "operator": "prune_blocks",
        "operator_args": {"ratio": 0.1},
        "status": "rejected",
        "decision": {"status": "rejected", "violations": ["quality_drop"]},
        "evaluation": {
            "quality_score": 0.84,
            "feasible": False,
            "benchmark_recipe": {
                "target_profile_id": "l40x4_h3",
                "split": "heldout",
                "quality_scope": "structural_proxy",
                "workflow_template": "/srv/harness/examples/workflow.json",
                "comfyui_cache_policy": "idle_release",
                "optimization_capabilities": {
                    "lpl": {
                        "status": "not_executable",
                        "safe_to_plan": False,
                        "execution_contract": {
                            "operator": "step_distill",
                            "operator_args": "lpl_target_steps",
                            "creates_checkpoint": False,
                        },
                        "reason": "old live ComfyUI process",
                        "extension_installed": True,
                        "live_node_registered": False,
                    },
                    "ci_dl": {
                        "status": "active",
                        "safe_to_plan": False,
                        "runtime_confirmed": True,
                        "implementation": "dynamic VBAR block prefetch",
                    },
                },
            },
            "tasks": [{"task_id": "task-%d" % index, "raw_frames": "x" * 10000} for index in range(20)],
        },
    }
    value = ControllerContext(
        context().target_profile,
        context().current_model_state,
        context().budget_state,
        context().available_operators,
        recent_experiments=[prior],
    )

    compact = _prompt_context(value)
    recipe = compact["recent_experiments"][0]["evaluation"]["benchmark_recipe"]
    assert recipe["target_profile_id"] == "l40x4_h3"
    assert recipe["comfyui_cache_policy"] == "idle_release"
    assert recipe["optimization_capabilities"]["lpl"]["live_node_registered"] is False
    assert recipe["optimization_capabilities"]["lpl"]["execution_contract"]["creates_checkpoint"] is False
    assert recipe["optimization_capabilities"]["ci_dl"]["runtime_confirmed"] is True
    assert len(compact["recent_experiments"][0]["evaluation"]["tasks"]) == 8
    assert "raw_frames" not in json.dumps(compact, ensure_ascii=False)


def test_prompt_context_includes_compact_live_gpu_capacity():
    value = ControllerContext(
        context().target_profile,
        context().current_model_state,
        context().budget_state,
        context().available_operators,
        campaign_summary={
            "gpu_capacity": {
                "gpu_count": 4,
                "memory_waterline_mb": 26624,
                "reserved_gpu_indices": [0],
                "free_above_waterline_indices": [1, 2, 3],
                "free_above_waterline_count": 3,
                "compute_processes_present": True,
                "compute_process_gpu_indices": [0],
                "compute_process_mapping_unknown": False,
                "comfyui_reserved_gpu_indices": [0],
                "per_gpu": {"0": {"memory_used_mb": 20000}},
                "internal_process_output": "must not be copied",
            },
            "pipeline_telemetry": {
                "window_events": 10,
                "underutilized_gpu_indices": ["2"],
                "overlap_wait_s": {"last_s": 4.0},
                "last_power_feedback": {
                    "target_power_w": 300.0,
                    "under_target_gpu_indices": ["2"],
                    "per_gpu": {"2": {"power_w_avg": 48.0, "lane": "idle"}},
                },
                "secret_raw_log": "must not be copied",
            },
        },
    )

    compact = _prompt_context(value)
    capacity = compact["campaign_summary"]["gpu_capacity"]
    assert capacity["free_above_waterline_indices"] == [1, 2, 3]
    assert capacity["comfyui_reserved_gpu_indices"] == [0]
    assert capacity["compute_process_gpu_indices"] == [0]
    assert capacity["compute_process_mapping_unknown"] is False
    assert "internal_process_output" not in json.dumps(compact, ensure_ascii=False)
    telemetry = compact["campaign_summary"]["pipeline_telemetry"]
    assert telemetry["underutilized_gpu_indices"] == ["2"]
    assert telemetry["last_power_feedback"]["per_gpu"]["2"]["power_w_avg"] == 48.0
    assert "secret_raw_log" not in json.dumps(compact, ensure_ascii=False)


def test_prompt_context_uses_round_policy_and_digest_without_history_duplication():
    value = ControllerContext(
        context().target_profile,
        context().current_model_state,
        context().budget_state,
        context().available_operators,
        recent_experiments=[{"experiment_id": "legacy", "operator": "quantize"}],
        round_policy={
            "schema_version": 1,
            "round_id": "R0012",
            "allowed_operators": ["quantize", "step_distill"],
            "resource_policy": {"min_training_gpus": 2, "controller_overlap_gpus": 1},
        },
        discovery_digest={
            "schema_version": 1,
            "source_digest": "sha256:digest",
            "source_observation_ids": ["obs-1"],
            "recent_experiments": [
                {
                    "experiment_id": "exp-1",
                    "plan": {"operator": "quantize", "operator_args": {"bits": 4}},
                    "checkpoint_path": "/data/models/secret.safetensors",
                }
            ],
            "failure_counts": {"worker_oom": 1},
        },
    )

    compact = _prompt_context(value)

    assert compact["round_policy"]["round_id"] == "R0012"
    assert compact["discovery_digest"]["source_digest"] == "sha256:digest"
    assert compact["discovery_digest"]["recent_experiments"][0]["operator"] == "quantize"
    assert compact["recent_experiments"] == []
    encoded = json.dumps(compact, ensure_ascii=False)
    assert "checkpoint_path" not in encoded
    assert "secret.safetensors" not in encoded


def test_prompt_context_preserves_compact_training_power_experience():
    compact = _compact_experience(
        {
            "experience_id": "xp-power",
            "experiment_id": "exp_0002",
            "operator": "step_distill",
            "status": "training_only_unvalidated",
            "training": {
                "training_power_sampling": {
                    "power_w_avg": 1040.0,
                    "power_w_peak": 1180.0,
                    "target_power_w": 300.0,
                    "samples": 120,
                    "sample_rows": 480,
                    "per_gpu": {
                        "0": {
                            "power_w_avg": 290.0,
                            "power_w_peak": 300.0,
                            "utilization_gpu_pct_avg": 98.0,
                            "utilization_gpu_pct_peak": 100.0,
                            "samples": 120,
                            "raw_trace": "x" * 10000,
                        }
                    },
                }
            },
        }
    )

    power = compact["training"]["training_power_sampling"]
    assert power["target_power_w"] == 300.0
    assert power["per_gpu"]["0"]["utilization_gpu_pct_avg"] == 98.0
    assert "raw_trace" not in json.dumps(compact, ensure_ascii=False)


def test_compact_evaluation_preserves_bounded_per_gpu_power_evidence():
    compact = _compact_quality_metrics(
        {
            "quality_score": 0.91,
            "power_sampling": {
                "power_w_avg": 620.0,
                "target_power_w": 300.0,
                "per_gpu": {
                    "0": {
                        "power_w_avg": 290.0,
                        "utilization_gpu_pct_avg": 97.0,
                        "lane": "comfyui",
                    },
                    "1": {
                        "power_w_avg": 64.0,
                        "utilization_gpu_pct_avg": 12.0,
                        "lane": "idle",
                        "raw_trace": "x" * 10000,
                    },
                },
            },
        }
    )

    power = compact["power_sampling"]
    assert power["per_gpu"]["0"]["utilization_gpu_pct_avg"] == 97.0
    assert power["underutilized_gpu_indices"] == ["1"]
    assert "raw_trace" not in json.dumps(compact, ensure_ascii=False)


def test_prompt_context_bounds_worker_failure_decision_message():
    compact = _compact_experience(
        {
            "experience_id": "xp-failed",
            "experiment_id": "exp_0003",
            "operator": "step_distill",
            "status": "failed",
            "decision": {
                "status": "failed",
                "failure_type": "checkpoint_corrupt",
                "message": "traceback " + ("x" * 100_000),
            },
        }
    )
    decision = compact["decision"]
    assert len(decision["message"]) == 1600
    assert decision["message_truncated"] is True
    assert len(json.dumps(compact, ensure_ascii=False)) < 3000


def test_openai_responses_provider_uses_strict_schema_and_store_false(controller_server, monkeypatch):
    url, requests = controller_server
    monkeypatch.setenv("TEST_OPENAI_KEY", "not-a-real-key")
    provider = OpenAIResponsesController("fixed-model", url + "/v1/responses", "TEST_OPENAI_KEY")
    plan = provider.plan(context())
    assert plan.parent_model_id == "M0000"
    request = requests[0][1]
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert requests[0][2]["Authorization"] == "Bearer not-a-real-key"


def test_openai_compatible_provider_uses_vllm_structured_chat_schema(controller_server):
    url, requests = controller_server
    provider = OpenAICompatibleController("qwen3.5-controller", url + "/v1/chat/completions")
    plan = provider.plan(context())

    assert plan.operator == "quantize"
    assert provider.last_request_id == "chatcmpl-1"
    request = requests[0][1]
    assert request["model"] == "qwen3.5-controller"
    assert request["temperature"] == 0.25
    assert request["n"] == 4
    assert request["stream"] is False
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["name"] == "experiment_plan"


def test_openai_compatible_provider_selects_from_batched_plan_choices():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            requests.append(payload)
            schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
            if schema_name == "experiment_plan_selection":
                content = {"selected_index": 1, "reason": "candidate 1 has the better declared tradeoff"}
                choices = [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}}]
            else:
                choices = []
                for index in range(4):
                    candidate = plan_payload()
                    candidate["rationale"] = "candidate %d" % index
                    choices.append(
                        {
                            "index": index,
                            "message": {"role": "assistant", "content": json.dumps(candidate)},
                        }
                    )
            response = {"id": "chatcmpl-%d" % len(requests), "choices": choices}
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleController(
            "qwen3.5-controller",
            "http://127.0.0.1:%d/v1/chat/completions" % server.server_port,
        )
        plan = provider.plan(context())
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert plan.rationale == "candidate 1"
    assert provider.last_candidate_count == 4
    assert provider.last_selected_index == 1
    assert provider.last_selection_fallback is None
    assert len(provider.last_eligible_candidates) == 4
    assert {item.rationale for item in provider.last_eligible_candidates} == {
        "candidate 0",
        "candidate 1",
        "candidate 2",
        "candidate 3",
    }
    assert requests[0]["n"] == 4
    assert requests[1]["response_format"]["json_schema"]["name"] == "experiment_plan_selection"


def test_openai_compatible_provider_falls_back_when_selector_returns_bad_index():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            requests.append(payload)
            schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
            if schema_name == "experiment_plan_selection":
                content = {"selected_index": 99, "reason": "invalid test index"}
                choices = [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}}]
            else:
                choices = [
                    {
                        "index": index,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({**plan_payload(), "rationale": "first candidate" if index == 0 else "other"}),
                        },
                    }
                    for index in range(2)
                ]
            response = {"id": "chatcmpl-%d" % len(requests), "choices": choices}
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAICompatibleController(
            "qwen3.5-controller",
            "http://127.0.0.1:%d/v1/chat/completions" % server.server_port,
        )
        plan = provider.plan(context())
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert plan.rationale == "first candidate"
    assert provider.last_selected_index == 0
    assert provider.last_selection_fallback is not None


def test_openai_compatible_provider_filters_known_infeasible_candidates_before_selection():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            requests.append(payload)
            schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
            if schema_name == "experiment_plan_selection":
                content = {"selected_index": 0, "reason": "choose the only eligible first candidate"}
                choices = [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}}]
            else:
                invalid = {**plan_payload(), "operator_args": {"bits": 8}, "rationale": "already quantized"}
                valid = {
                    **plan_payload(),
                    "operator": "step_distill",
                    "operator_args": {"target_steps": 25},
                    "resource_request": {
                        "gpu_count": 2,
                        "min_gpu_count": 2,
                        "max_gpu_count": 4,
                        "elastic": True,
                        "distributed": True,
                        "exclusive": False,
                        "evaluation_workers": 1,
                        "on_unavailable": "wait",
                    },
                    "rationale": "valid training candidate",
                }
                choices = [
                    {"index": 0, "message": {"role": "assistant", "content": json.dumps(invalid)}},
                    {"index": 1, "message": {"role": "assistant", "content": json.dumps(valid)}},
                ]
            response = {"id": "chatcmpl-%d" % len(requests), "choices": choices}
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        quantized = ModelState.fake_baseline().derive(
            "M0000",
            quantization={"bits": 8, "scheme": "int8"},
        )
        value = ControllerContext(
            context().target_profile,
            quantized,
            context().budget_state,
            context().available_operators,
        )
        provider = OpenAICompatibleController(
            "qwen3.5-controller",
            "http://127.0.0.1:%d/v1/chat/completions" % server.server_port,
        )
        plan = provider.plan(value)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert plan.operator == "step_distill"
    assert provider.last_candidate_count == 2
    assert provider.last_candidate_eligible_count == 1
    assert provider.last_candidate_filter_rejections == [{"index": 0, "reason": "quantize_not_lower_than_current"}]
    assert provider.last_selected_index == 1
    assert provider.last_selection_fallback == "single_candidate"
    assert len(requests) == 1


def test_known_candidate_guard_rejects_arguments_from_a_different_operator():
    raw = plan_payload()
    raw.update(
        {
            "operator": "step_distill",
            "operator_args": {"target_steps": 16, "bits": 8},
            "resource_request": {
                "gpu_count": 2,
                "min_gpu_count": 2,
                "max_gpu_count": 4,
                "elastic": True,
                "distributed": True,
                "exclusive": False,
                "evaluation_workers": 1,
                "on_unavailable": "wait",
            },
        }
    )
    plan = ExperimentPlan.from_dict(raw)

    assert OpenAICompatibleController._known_candidate_guard(context(), plan) == "operator_args_unknown:bits"


def test_openai_compatible_provider_uses_safe_rule_fallback_when_all_candidates_are_stale():
    state = ModelState.fake_baseline().derive(
        "M0000",
        sampling_steps=1,
        measured_metrics={"latency_s": 100.0, "peak_memory_gb": 7.0},
    )
    value = ControllerContext(
        context().target_profile,
        state,
        context().budget_state,
        context().available_operators,
    )
    stale = {
        **plan_payload(),
        "operator": "step_distill",
        "operator_args": {"target_steps": 1},
        "resource_request": {
            "gpu_count": 2,
            "min_gpu_count": 2,
            "max_gpu_count": 4,
            "elastic": True,
            "distributed": True,
            "exclusive": False,
            "evaluation_workers": 1,
            "on_unavailable": "wait",
        },
    }
    provider = OpenAICompatibleController("qwen3.5-controller")
    provider._chat_json = lambda *args, **kwargs: {
        "id": "chatcmpl-stale-batch",
        "choices": [
            {"message": {"content": json.dumps(stale)}},
            {"message": {"content": json.dumps(stale)}},
        ],
    }

    plan = provider.plan(value)

    assert plan.operator == "quantize"
    assert plan.operator_args == {"bits": 4}
    assert provider.last_candidate_eligible_count == 0
    assert provider.last_selection_fallback == "rule_based_after_all_candidates_ineligible"
    assert provider.last_selected_index is None


def test_openai_compatible_provider_retries_one_malformed_structured_candidate():
    provider = OpenAICompatibleController("qwen3.5-controller", candidate_count=1)
    calls = []

    def fake_chat_json(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"id": "chatcmpl-malformed", "choices": [{"message": {"content": "{}"}}]}
        return {"id": "chatcmpl-retry", "choices": [{"message": {"content": json.dumps(plan_payload())}}]}

    provider._chat_json = fake_chat_json
    plan = provider.plan(context())

    assert plan.operator == "quantize"
    assert len(calls) == 2
    assert calls[1]["max_tokens"] >= calls[0]["max_tokens"]


def test_openai_compatible_provider_doubles_budget_for_length_truncation():
    provider = OpenAICompatibleController(
        "qwen3.5-controller",
        candidate_count=1,
        plan_max_tokens=2048,
        extended_plan_max_tokens=2048,
        candidate_max_tokens=2048,
    )
    calls = []

    def fake_chat_json(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "id": "chatcmpl-length",
                "choices": [
                    {
                        "message": {"content": '{"experiment_id":"exp_0001"}'},
                        "finish_reason": "length",
                    }
                ],
            }
        return {"id": "chatcmpl-length-retry", "choices": [{"message": {"content": json.dumps(plan_payload())}}]}

    provider._chat_json = fake_chat_json
    plan = provider.plan(context())

    assert plan.operator == "quantize"
    assert [call["max_tokens"] for call in calls] == [2048, 4096]


def test_openai_compatible_probe_budget_can_fit_full_plan_schema():
    provider = OpenAICompatibleController("qwen3.5-controller")

    assert provider.probe_max_tokens == 1536


def test_structured_provider_parses_fenced_json_with_short_prefix():
    parsed = _parse_json_object("Here is the plan:\n```json\n" + json.dumps(plan_payload()) + "\n```")

    assert parsed["operator"] == "quantize"
    assert parsed["operator_args"] == {"bits": 4}


def test_vllm_plan_enables_reasoning_and_escalates_for_human_directive(controller_server):
    url, requests = controller_server
    provider = OpenAICompatibleController("qwen3.5-controller", url + "/v1/chat/completions")

    provider.plan(context())
    assert requests[0][1]["chat_template_kwargs"]["enable_thinking"] is False
    assert requests[0][1]["max_tokens"] <= 4096

    directive = HumanDirective.create("先降低 peak_memory", directive_id="long-reasoning-001")
    directive_context = ControllerContext(
        context().target_profile,
        context().current_model_state,
        context().budget_state,
        context().available_operators,
        observations=[directive.to_observation().to_dict()],
        unconsumed_observation_ids=[directive.to_observation().observation_id],
    )
    provider.plan(directive_context)
    assert requests[1][1]["chat_template_kwargs"]["enable_thinking"] is False
    assert requests[1][1]["max_tokens"] <= 4096


def test_safe_completion_tokens_has_long_prompt_emergency_budget():
    assert _safe_completion_tokens("x" * 41_000, 4096) == 256
    with pytest.raises(ControllerProviderError, match="too large"):
        _safe_completion_tokens("x" * 50_000, 4096)


def test_openai_compatible_provider_reviews_with_separate_schema(controller_server):
    url, requests = controller_server
    provider = OpenAICompatibleController("qwen3.5-controller", url + "/v1/chat/completions")
    decision = provider.review({"phase": "training", "trigger": "heartbeat", "evidence_ids": ["evt-1"]})

    assert decision.action == "review_only"
    request = requests[0][1]
    assert request["response_format"]["json_schema"]["name"] == "controller_review"
    assert request["max_tokens"] == 768
    assert request["chat_template_kwargs"]["enable_thinking"] is False
    assert "GPU0" in request["messages"][1]["content"]


def test_openai_compatible_provider_probe_is_structured_and_recorded(controller_server):
    url, requests = controller_server
    provider = OpenAICompatibleController("qwen3.5-controller", url + "/v1/chat/completions")
    probe = provider.probe(context())

    assert probe.parent_model_id == "M0000"
    assert provider.probed is True
    assert provider.last_probe_request_id == "chatcmpl-1"
    assert "Connectivity probe only" in requests[0][1]["messages"][1]["content"]


def test_openai_compatible_provider_does_not_hide_unavailable_endpoint():
    provider = OpenAICompatibleController("qwen3.5-controller", "http://127.0.0.1:1/v1/chat/completions", timeout_s=0.2)

    with pytest.raises(ControllerUnavailableError):
        provider.plan(context())


def test_controller_factory_defaults_to_local_vllm_without_rule_fallback():
    provider = build_controller_from_config()

    assert isinstance(provider, OpenAICompatibleController)
    assert provider.provider_name == "vllm"
    assert provider.model_name == "qwen3.5-controller"
    assert provider.remote_port == 8000
    assert provider.context_window_tokens == 16384


def test_default_vllm_config_bounds_primary_plan_latency():
    provider = build_controller_from_config("configs/controller.yaml")

    assert isinstance(provider, OpenAICompatibleController)
    assert provider.timeout_s == 240
    assert provider.plan_max_tokens == 2048
    assert provider.extended_plan_max_tokens == 2048
    assert provider.candidate_count == 1
    assert provider.candidate_max_tokens == 2048


def test_controller_factory_reads_config_and_allows_explicit_test_backend(tmp_path):
    config_path = tmp_path / "controller.yaml"
    config_path.write_text(
        "default_provider: vllm\n"
        "providers:\n"
        "  vllm:\n"
        "    model: configured-model\n"
        "    endpoint: http://127.0.0.1:9100/v1/chat/completions\n"
        "    remote_port: 9100\n",
        encoding="utf-8",
    )

    configured = build_controller_from_config(config_path)
    explicit_test = build_controller_from_config(config_path, provider_name="rulebased")

    assert isinstance(configured, OpenAICompatibleController)
    assert configured.model_name == "configured-model"
    assert configured.endpoint.endswith(":9100/v1/chat/completions")
    assert configured.remote_port == 9100
    assert configured.plan_max_tokens == 4096
    assert configured.extended_plan_max_tokens == 6144
    assert isinstance(explicit_test, RuleBasedMockController)


def test_runtime_context_exposes_design_gene_and_rule_controller_switches_layer():
    current = ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "model_id": "M0001",
            "parent_model_id": "M0000",
            "architecture_name": "MiniMax-H3",
            "quantization": {"bits": 4, "scheme": "nvfp4"},
            "measured_metrics": {
                "quality_score": 0.99,
                "latency_s": 90.0,
                "peak_memory_gb": 16.3,
                "model_size_gb": 12.5,
                "energy_j": 1.0,
            },
        }
    )
    context_value = ControllerContext(
        TargetProfile("rtx", "gpu", "local", max_peak_memory_gb=16.0, max_quality_drop=0.05),
        current,
        BudgetState(4, 2),
        tuple({"name": name, "description": "", "input_schema": {"mode": "str"}} for name in ("runtime_offload",)),
        validated_design_genes=[{"gene_id": "H3-NVFP4-Quantization-001", "status": "validated_m5_5"}],
        validated_evaluation={"stage": "M5.5", "validated": True},
    )
    assert context_value.to_dict()["validated_design_genes"][0]["status"] == "validated_m5_5"
    assert context_value.to_dict()["validated_evaluation"]["validated"] is True
    plan = RuleBasedMockController().plan(context_value)
    assert plan.operator == "runtime_offload"
    assert plan.operator_args == {"mode": "aggressive"}


def test_rule_controller_moves_to_next_runtime_operator_after_rejection():
    current = ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "model_id": "M0001",
            "parent_model_id": "M0000",
            "architecture_name": "MiniMax-H3",
            "quantization": {"bits": 4, "scheme": "nvfp4"},
            "measured_metrics": {"quality_score": 0.99, "latency_s": 90.0, "peak_memory_gb": 16.3, "model_size_gb": 12.5},
        }
    )
    operators = tuple(
        {"name": name, "description": "", "input_schema": {"mode": "str"}}
        for name in ("runtime_offload", "vae_decode_offload")
    )
    value = ControllerContext(
        TargetProfile("rtx", "gpu", "local", max_peak_memory_gb=16.0, max_quality_drop=0.05),
        current,
        BudgetState(4, 4),
        operators,
        relevant_failures=[{"operator": "runtime_offload", "failure_type": "acceptance_rejected"}],
    )
    plan = RuleBasedMockController().plan(value)
    assert plan.operator == "vae_decode_offload"


def test_rule_controller_receives_goal_and_selects_structural_pruning():
    current = ModelState.from_dict(
        {
            **ModelState.fake_baseline().to_dict(),
            "architecture_name": "MiniMax-H3-FL2VA",
            "sampling_steps": 32,
            "measured_metrics": {
                "quality_score": 0.99,
                "latency_s": 345.0,
                "peak_memory_gb": 47.0,
                "model_size_gb": 66.0,
                "energy_j": 100.0,
            },
        }
    )
    value = ControllerContext(
        TargetProfile("l40", "gpu", "l40x4", max_latency_s=300, max_quality_drop=0.05),
        current,
        BudgetState(4, 2),
        (
            {"name": "prune_blocks", "description": "real structural block pruning", "input_schema": {"ratio": "float"}},
            {"name": "distill", "description": "real teacher output distillation", "input_schema": {"dataset_fraction": "float", "training_steps": "int"}},
        ),
        goal={"goal_id": "l40", "objective": "meet latency without exceeding quality drop"},
    )
    plan = RuleBasedMockController().plan(value)
    assert value.goal["objective"].startswith("meet latency")
    assert plan.operator == "prune_blocks"
    assert plan.operator_args == {"ratio": 0.1}
    assert plan.resource_request["gpu_count"] == 0
    assert plan.resource_request["elastic"] is False


def test_rule_controller_declares_elastic_training_range():
    value = ControllerContext(
        TargetProfile("l40", "gpu", "l40x4", max_latency_s=300, max_quality_drop=0.05),
        ModelState.fake_baseline(),
        BudgetState(4, 2),
        ({"name": "distill", "description": "", "input_schema": {"dataset_fraction": "float", "training_steps": "int"}},),
        recent_experiments=[{"operator": "prune_blocks", "status": "rejected"}],
    )
    plan = RuleBasedMockController().plan(value)
    assert plan.operator == "distill"
    assert plan.resource_request == {
        "gpu_count": 4,
        "min_gpu_count": 2,
        "max_gpu_count": 4,
        "elastic": True,
        "distributed": True,
        "exclusive": False,
        "evaluation_workers": 1,
        "on_unavailable": "wait",
    }


def test_rule_controller_requires_real_training_evidence_before_step_distill():
    value = ControllerContext(
        TargetProfile("l40", "gpu", "l40x4", max_latency_s=300, max_quality_drop=0.05),
        ModelState.fake_baseline(),
        BudgetState(4, 2),
        (
            {"name": "recovery_finetune", "description": "", "input_schema": {"training_steps": "int"}},
            {"name": "step_distill", "description": "", "input_schema": {"target_steps": "int"}},
        ),
        current_system={"id": "S0000"},
    )
    plan = RuleBasedMockController().plan(value)
    assert plan.operator == "recovery_finetune"
