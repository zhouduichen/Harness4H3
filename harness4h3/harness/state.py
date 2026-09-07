from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping

import yaml


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    split: str
    seed: int = 42
    constraints: Mapping[str, Any] = field(default_factory=dict)
    expected: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Task":
        task_id = str(raw.get("id", "")).strip()
        prompt = str(raw.get("prompt", "")).strip()
        split = str(raw.get("split", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", task_id) or not prompt or split not in {"sanity", "dev", "heldout"}:
            raise ValueError("task requires id, prompt, and split sanity/dev/heldout")
        return cls(
            id=task_id,
            prompt=prompt,
            split=split,
            seed=int(raw.get("seed", 42)),
            constraints=dict(raw.get("constraints") or {}),
            expected=dict(raw.get("expected") or {}),
        )


@dataclass
class TaskState:
    task_id: str
    goal: str
    step: int = 0
    recent_history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    done: bool = False

    def observe(self, event: Mapping[str, Any]) -> None:
        self.step += 1
        self.recent_history.append(dict(event))


def load_tasks(path: Path) -> List[Task]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("tasks"), list):
        raise ValueError("task manifest requires a tasks list")
    tasks = [Task.from_dict(item) for item in raw["tasks"]]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task ids must be unique")
    return tasks
