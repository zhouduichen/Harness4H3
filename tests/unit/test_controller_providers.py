from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness4h3.controller.context import ControllerContext
from harness4h3.controller.provider import OllamaStructuredController, OpenAIResponsesController, RuleBasedMockController
from harness4h3.controller.schemas import BudgetState
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
    }


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
