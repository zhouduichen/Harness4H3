from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ..archive.store import Candidate
from ..config import WorkflowConfig
from .state import Task


@dataclass(frozen=True)
class ExecutionRequest:
    prompt: str
    workflow: Mapping[str, Any]
    context_digest: str


def _set_target(workflow: Dict[str, Any], node_id: str, input_name: str, value: Any) -> None:
    if node_id not in workflow or input_name not in workflow[node_id].get("inputs", {}):
        raise ValueError("workflow target %s:%s is unavailable" % (node_id, input_name))
    workflow[node_id]["inputs"][input_name] = value


def _render_prompt(task: Task, policy: Mapping[str, Any]) -> str:
    prompt_policy = policy.get("prompt") or {}
    context_policy = policy.get("context") or {}
    prefix = str(prompt_policy.get("prefix", "")).strip()
    suffix = str(prompt_policy.get("suffix", "")).strip()
    middle = [task.prompt]
    if bool(context_policy.get("include_constraints", True)) and task.constraints:
        constraints = ["%s: %s" % (key, task.constraints[key]) for key in sorted(task.constraints)]
        middle.append("Constraints: " + "; ".join(constraints))
    middle_text = "\n\n".join(middle)
    head = prefix + "\n\n" if prefix else ""
    tail = "\n\n" + suffix if suffix else ""
    prompt = head + middle_text + tail
    max_chars = int(context_policy.get("max_prompt_chars", 4000))
    if max_chars <= 0:
        raise ValueError("context.max_prompt_chars must be positive")
    if len(prompt) <= max_chars:
        return prompt
    middle_budget = max_chars - len(head) - len(tail)
    if middle_budget >= 0:
        return head + middle_text[:middle_budget] + tail
    if suffix:
        return suffix[-max_chars:]
    return prompt[:max_chars]


def build_execution_request(
    template: Mapping[str, Any],
    task: Task,
    candidate: Candidate,
    config: WorkflowConfig,
) -> ExecutionRequest:
    workflow = copy.deepcopy(dict(template))
    prompt = _render_prompt(task, candidate.policy)
    _set_target(workflow, config.prompt_target.node_id, config.prompt_target.input_name, prompt)
    if config.seed_target is not None:
        _set_target(workflow, config.seed_target.node_id, config.seed_target.input_name, task.seed)
    workflow_policy = candidate.policy.get("workflow") or {}
    for key, value in workflow_policy.items():
        if key not in config.mutable:
            raise ValueError("candidate mutates non-allowlisted workflow key %s" % key)
        target = config.mutable[key]
        _set_target(workflow, target.node_id, target.input_name, value)
    digest_payload = {
        "task_id": task.id,
        "prompt": prompt,
        "candidate": candidate.id,
        "workflow_policy": workflow_policy,
    }
    digest = hashlib.sha256(json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return ExecutionRequest(prompt=prompt, workflow=workflow, context_digest=digest)
