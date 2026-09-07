from __future__ import annotations

from harness4h3.archive.store import Candidate, default_policy
from harness4h3.config import Target, WorkflowConfig
from harness4h3.harness.context import build_execution_request
from harness4h3.harness.state import Task


def candidate(policy=None):
    return Candidate("H0", None, 0, "baseline", {}, "baseline", policy or default_policy())


def config(tmp_path):
    return WorkflowConfig(
        tmp_path / "workflow.json",
        Target("prompt", "text"),
        Target("seed", "seed"),
        {"steps": Target("sampler", "steps"), "cfg": Target("sampler", "cfg")},
    )


def test_context_renders_prompt_seed_and_allowlisted_workflow_without_mutating_template(tmp_path, workflow):
    policy = default_policy()
    policy["prompt"]["prefix"] = "Stable subject."
    policy["workflow"]["steps"] = 7
    task = Task("t1", "Walk forward.", "dev", 42, {"lighting": "bright"})
    request = build_execution_request(workflow, task, candidate(policy), config(tmp_path))
    assert request.workflow["prompt"]["inputs"]["text"].startswith("Stable subject.")
    assert "lighting: bright" in request.prompt
    assert request.workflow["seed"]["inputs"]["seed"] == 42
    assert request.workflow["sampler"]["inputs"]["steps"] == 7
    assert workflow["prompt"]["inputs"]["text"] == "original"


def test_context_rejects_non_allowlisted_candidate_workflow_key(tmp_path, workflow):
    policy = default_policy()
    policy["workflow"]["model"] = "other"
    task = Task("t1", "Walk.", "dev")
    import pytest

    with pytest.raises(ValueError, match="non-allowlisted"):
        build_execution_request(workflow, task, candidate(policy), config(tmp_path))


def test_context_budget_preserves_mutation_suffix(tmp_path, workflow):
    policy = default_policy()
    policy["context"]["max_prompt_chars"] = 80
    policy["prompt"]["suffix"] = "Keep every frame bright and stable."
    task = Task("t1", "A" * 200, "dev")
    request = build_execution_request(workflow, task, candidate(policy), config(tmp_path))
    assert len(request.prompt) == 80
    assert request.prompt.endswith("Keep every frame bright and stable.")
