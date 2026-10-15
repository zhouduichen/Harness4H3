from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace
import pytest

from harness4h3.adapters.wan.contracts import CheckpointArtifact, ModelStructure
from harness4h3 import cli as harness_cli
from harness4h3.evolution.agent import build_algorithm_provider
from harness4h3.evolution.candidate import AlgorithmCandidate
from harness4h3.evolution.config import CompressionConfig, CompressionConfigError
from harness4h3.evolution.final_evaluation import FinalEvaluationRequest, FinalEvaluationRunner
from harness4h3.evolution.improvement import AlgorithmValidator
from harness4h3.evolution.tica_prune import TICAPruner, TICARecipe, tica_operator_spec
from harness4h3.evolution.tica_calibration import TICACalibrationManifest
from harness4h3.sandbox import AlgorithmSandbox
from tools.preflight_wan_tica import preflight


def _profile():
    width = 8
    return {
        "evaluation_split": "calibration",
        "model": {"architecture": "WanFakeDiT", "block_count": 5, "ffn_width": width},
        "layers": [{"block_id": "block.%d" % index} for index in range(5)],
        "tica": {"timesteps": ["0.9", "0.5", "0.1"]},
        "tica_block_sensitivity": [
            {"block_id": "block.%d" % index, "per_timestep": {"0.9": 0.01, "0.5": 0.01, "0.1": 0.01}}
            for index in range(5)
        ],
        "tica_channel_sensitivity": [
            {"block_id": "block.%d" % index, "available": True, "channels": [{"channel": channel, "contribution": float(channel + 1)} for channel in range(width)]}
            for index in range(5)
        ],
    }


def _measure(plan, full, stage):
    return {
        "valid": True,
        "quality_score": 0.99,
        "parameter_reduction_ratio": 0.2,
        "risk_metric": "normalized_mse" if full else "local_relative_output_error",
        "risk_value": 0.01,
        "risk_scale": "normalized_mse" if full else "profile_relative_error",
        "risk_is_approximate": not full,
        "stage": stage,
    }


def test_search_records_observed_depth_and_budget_termination():
    result = TICAPruner().search(
        None,
        None,
        type("Manifest", (), {"digest": lambda self: "m" * 64, "provider_name": "fake_wan_cpu"})(),
        _profile(),
        TICARecipe(
            baseline_name="tica_block_ffn",
            include_baseline_suite=False,
            max_search_depth=3,
            max_fast_evals=100,
            max_full_evals=100,
            max_interaction_evals=100,
        ),
        evaluate_candidate=_measure,
    )
    assert result.observed_search_depth == 3
    assert {item["depth"] for item in result.level_stats} >= {1, 2, 3}
    assert result.budget["full_evals"] <= result.budget["limits"]["max_full_evals"]


def test_complete_search_validator_keeps_parent_and_child_comparison_separate(tmp_path):
    source = "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': [profile['algorithm_marker']]}}\n"
    parent = AlgorithmCandidate.from_source("A0001", source)
    child = AlgorithmCandidate.from_source(
        "A0002",
        "def propose(profile, history):\n    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.1']}}\n",
        parent_algorithm_id=parent.algorithm_id,
        parent_source_sha256=parent.source_sha256,
    )
    calls = []

    def complete_search(result, profile):
        calls.append(result.plan.args["block_ids"][0])
        better = result.plan.args["block_ids"][0] == "block.1"
        return {
            "valid": True,
            "offline_simulation": True,
            "search_score": 0.95 if better else 0.80,
            "quality_score": 0.99,
            "parameter_reduction_ratio": 0.2,
            "parent_checkpoint_sha256": "p" * 64,
            "calibration_digest": "c" * 64,
            "search_budget": {"max_fast_evals": 8, "max_full_evals": 4, "max_interaction_evals": 2},
            "pareto_front": [{"candidate_id": "selected"}],
        }

    record = AlgorithmValidator().validate(
        parent,
        child,
        AlgorithmSandbox(tmp_path / "sandbox"),
        _profile(),
        [],
        complete_search,
        complete_search=complete_search,
        algorithm_profile={"algorithm_marker": "block.0"},
    )
    assert record.valid is True
    assert calls == ["block.0", "block.1"]
    assert all(item["evaluation_mode"] == "complete_search" for item in record.scenario_results)
    assert record.scenario_results[0]["parent_metrics"]["pareto_front"]


def test_tica_validation_profiles_do_not_collect_validation_search_evidence(monkeypatch):
    import harness4h3.evolution.loop as evolution_loop_module

    class Adapter:
        def reload_model(self, artifact):
            return object()

    loop = evolution_loop_module.EvolutionLoop.__new__(evolution_loop_module.EvolutionLoop)
    loop.adapter = Adapter()
    loop.tica_prune = True
    loop.advanced_profiling = False
    parent = SimpleNamespace(sha256="p" * 64)
    calibration_profile = {
        "model": {"checkpoint_sha256": parent.sha256, "architecture": "WanFakeDiT"},
        "tica": {"calibration_digest": "c" * 64},
        "tica_block_sensitivity": [{"block_id": "block.0"}],
        "tica_channel_sensitivity": [{"block_id": "block.0"}],
        "provenance": {"seed": 7},
    }

    monkeypatch.setattr(
        evolution_loop_module,
        "profile_model",
        lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {"model": {"architecture": "WanFakeDiT"}, "provenance": {"seed": 8}}),
    )
    monkeypatch.setattr(
        evolution_loop_module,
        "collect_tica_profile",
        lambda *args, **kwargs: pytest.fail("Validation must not be profiled for TICA search"),
    )
    profiles = loop._build_validation_profiles(parent, calibration_profile)
    assert len(profiles) == 2
    assert all(item["evaluation_split"] == "validation" for item in profiles)
    assert all("tica_block_sensitivity" not in item for item in profiles)
    assert all(item["provenance"]["search_profile_source"] == "calibration_only" for item in profiles)


def test_evolution_cli_defers_test_manifest_until_final_evaluation(monkeypatch, tmp_path):
    config_path = tmp_path / "evolution.yaml"
    config_path.write_text(
        "backend: fake\n"
        "checkpoint: fake://baseline\n"
        "output_root: %s\n"
        "tica_prune: true\n"
        "tica_calibration_manifest: calibration.json\n"
        "tica_validation_manifest: validation.json\n"
        "tica_test_manifest: test.json\n" % (tmp_path / "out"),
        encoding="utf-8",
    )
    loaded = []
    captured = {}

    def load_manifest(path):
        loaded.append(path.name)
        if path.name == "test.json":
            raise AssertionError("Evolution must not open the Test manifest")
        return TICACalibrationManifest.from_fake(hidden_size=8, sample_prefix=path.stem)

    class FakeLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_round(self):
            return SimpleNamespace(to_dict=lambda: {"status": "completed"})

    monkeypatch.setattr(harness_cli, "_load_tica_manifest", load_manifest)
    monkeypatch.setattr(harness_cli, "_compression_context", lambda config: (object(), object()))
    monkeypatch.setattr(harness_cli, "EvolutionLoop", FakeLoop)
    args = SimpleNamespace(compression_config=str(config_path), rounds=1, json=True)

    assert harness_cli.cmd_evolve_compression(args) == 0
    assert loaded == ["calibration.json", "validation.json"]
    assert captured["tica_splits"].test is None


def test_final_evaluation_does_not_change_frozen_ids_or_adaptive_state(tmp_path):
    digest = "a" * 64
    structure = ModelStructure("WanFakeDiT", 2, 4, 8, 10, digest)
    artifact = CheckpointArtifact(str(tmp_path / "checkpoint"), digest, "WanFakeDiT", structure)

    class Evaluator:
        class Config:
            evaluation_split = "test"

        config = Config()

        def __init__(self):
            self.calls = 0

        def evaluate_final(self, value):
            self.calls += 1
            return {"valid": False, "quality_score": 0.01, "promotable": False}

    evaluator = Evaluator()
    request = FinalEvaluationRequest("M0003", "A0002", digest, "b" * 64, "rev-1", "protocol-1")
    payload = FinalEvaluationRunner(evaluator, tmp_path / "final.json").run(request, artifact)
    assert payload["protocol_digest"] == "protocol-1"
    assert payload["frozen_request"]["model_id"] == "M0003"
    assert payload["adaptive_state_mutated"] is False
    assert payload["evaluation"]["promotable"] is False
    assert evaluator.calls == 1
    assert json.loads((tmp_path / "final.json").read_text())["phase"] == "final_evaluation"


def test_tica_controller_prompt_is_mode_specific(monkeypatch, tmp_path):
    requests = []
    tica_source = (
        "def propose(profile, history):\n"
        "    return {'operator': 'tica_prune', 'args': {"
        "'block_ids': ['block.0'], 'retained_indices': [0, 1, 2, 3], "
        "'mode': 'block_plus_ffn', 'time_weight_formula': 'late_log', "
        "'risk_limit': 0.25, 'fast_risk_limit': 0.5, 'activation_weight': 0.5, "
        "'interaction_weight': 0.5, 'beam_width': 2, 'max_fast_evals': 8, "
        "'max_full_evals': 4, 'max_interaction_evals': 2, "
        "'strategy': 'greedy_beam', 'max_search_depth': 3}}\n"
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({
                "source": tica_source,
                "hypothesis": "tica",
                "parent_id": "M0000",
            })}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: (requests.append(json.loads(request.data.decode())) or Response()))
    provider = build_algorithm_provider("vllm", "test-model", endpoint="http://127.0.0.1:8000/v1/chat/completions")
    tica_response = provider({"parent_id": "M0000", "profile": {}, "history": [], "skills": [], "protocol_mode": "tica"})
    provider({"parent_id": "M0000", "profile": {}, "history": [], "skills": [], "protocol_mode": "standard"})
    tica_prompt = requests[0]["messages"][1]["content"]
    standard_prompt = requests[1]["messages"][1]["content"]
    assert "TICA-Prune protocol mode" in tica_prompt
    assert "trusted TICAPruner" in tica_prompt
    assert "TICA-Prune protocol mode" not in standard_prompt
    assert "Allowed plans are exactly" in standard_prompt

    candidate = AlgorithmCandidate.from_source("A-TICA", tica_response["source"])
    sandbox = AlgorithmSandbox(tmp_path / "sandbox", operator_specs=[tica_operator_spec()])
    sandbox_result = sandbox.run(candidate.source, _profile(), [])
    assert sandbox_result.status == "success"
    assert sandbox_result.plan is not None
    assert sandbox_result.plan.operator == "tica_prune"
    recipe_fields = {
        name: sandbox_result.plan.args[name]
        for name in (
            "time_weight_formula", "risk_limit", "fast_risk_limit", "activation_weight",
            "interaction_weight", "beam_width", "max_fast_evals", "max_full_evals",
            "max_interaction_evals", "strategy", "max_search_depth",
        )
    }
    search = TICAPruner().search(
        None,
        None,
        type("Manifest", (), {"digest": lambda self: "m" * 64, "provider_name": "fake_wan_cpu"})(),
        _profile(),
        TICARecipe.from_mapping(recipe_fields),
        evaluate_candidate=_measure,
    )
    assert search.candidates
    assert search.candidates[0].plan["time_weight_formula"] == "late_log"


def test_wan_preflight_is_static_and_verifies_all_split_sources(tmp_path):
    checkpoint = tmp_path / "wan"
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "model_index.json").write_text("{}", encoding="utf-8")
    (checkpoint / "transformer" / "config.json").write_text(json.dumps({"num_layers": 2, "hidden_size": 8}), encoding="utf-8")
    manifest_paths = []
    for role in ("calibration", "validation", "test"):
        latent = tmp_path / (role + "-latent.bin")
        hidden = tmp_path / (role + "-hidden.bin")
        latent.write_bytes(role.encode() + b"-latent")
        hidden.write_bytes(role.encode() + b"-hidden")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "provider_name": "cached_real_wan",
            "specs": [{
                "sample_id": role + "-0",
                "prompt": role,
                "seed": 7,
                "latent_shape": [1, 1, 1, 1, 1],
                "timesteps": [0.5],
                "latent_path": str(latent),
                "encoder_hidden_states_path": str(hidden),
                "latent_sha256": digest(latent),
                "encoder_hidden_states_sha256": digest(hidden),
            }],
        }
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_paths.append(path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "backend: diffusers\ncheckpoint: %s\noutput_root: %s\n" % (checkpoint, tmp_path / "out")
        + "tica_prune: true\n"
        + "tica_calibration_manifest: %s\n" % manifest_paths[0]
        + "tica_validation_manifest: %s\n" % manifest_paths[1]
        + "tica_test_manifest: %s\n" % manifest_paths[2],
        encoding="utf-8",
    )
    result = preflight(config)
    assert result["loads_model"] is False
    assert result["starts_gpu"] is False
    assert result["runtime_probe"]["status"] == "not_run"


def test_compression_phase_rejects_unknown_phase(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("backend: fake\ncheckpoint: fake://baseline\nphase: test\n", encoding="utf-8")
    with pytest.raises(CompressionConfigError, match="phase"):
        CompressionConfig.from_path(config)
