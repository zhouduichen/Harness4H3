from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from harness4h3.archive.store import Candidate, CandidateStore
from harness4h3.config import AppConfig, load_config
from harness4h3.evaluator.evaluator import SubprocessEvaluator
from harness4h3.harness.loop import HarnessRunner, load_workflow
from harness4h3.harness.state import Task
from harness4h3.memory.trajectory import Trajectory, TrajectoryStore
from harness4h3.model.minimax_h3 import MiniMaxH3Adapter
from harness4h3.self_improve.evolve import diagnose, propose_mutation, should_promote


SEEDS = (42, 3407, 8888)
DEV_SCENES = (
    "A yellow dragon turns toward a window, turns back, blinks, and smiles.",
    "A yellow dragon walks two steps, stops, and gently swings its tail.",
    "A yellow dragon waves, points toward a clock, and lowers its arm.",
    "A yellow dragon sits on a stool, stands up, and looks at the camera.",
    "A yellow dragon opens a small book, turns one page, and closes it.",
    "A yellow dragon looks left, looks right, then nods once.",
    "A yellow dragon lifts a blue cup, pauses, and places it back down.",
    "A yellow dragon takes one step backward and gives a thumbs-up.",
    "A yellow dragon catches a soft ball and holds it in both hands.",
    "A yellow dragon rotates slowly once and returns to its starting pose.",
)
HELDOUT_SCENES = (
    "A yellow dragon picks up a red ball and places it on a table.",
    "A yellow dragon opens a wooden door, looks outside, and closes it.",
    "A yellow dragon pours water from a pitcher into a clear glass.",
    "A yellow dragon ties a green ribbon around a small gift box.",
    "A yellow dragon stacks three colored blocks and taps the top block.",
    "A yellow dragon unfolds a paper map and points to a marked location.",
    "A yellow dragon rolls a toy car forward and catches it on return.",
    "A yellow dragon places a flower into a narrow vase and adjusts it.",
    "A yellow dragon hangs a small picture frame on a wall hook.",
    "A yellow dragon lights a lantern and carries it across a dim room.",
    "A yellow dragon pulls a chair toward a desk and sits down.",
    "A yellow dragon opens an umbrella, turns it once, and closes it.",
    "A yellow dragon folds a blue towel and puts it on a shelf.",
    "A yellow dragon balances a spoon on one finger and catches it.",
    "A yellow dragon draws a circle on a board and sets down the chalk.",
    "A yellow dragon winds a small music box and listens beside it.",
    "A yellow dragon puts on a red scarf and checks it in a mirror.",
    "A yellow dragon carries a basket to a bench and sets it down.",
    "A yellow dragon turns a desk lamp on, reads a note, and turns it off.",
    "A yellow dragon opens a window curtain and watches falling snow.",
)
VARIANTS = (
    "Use a locked composition, smooth natural motion, and a stable background.",
    "Use a slow camera push-in while preserving subject identity and geometry.",
    "Keep warm balanced lighting, readable silhouettes, and coherent object contact.",
)


def build_tasks() -> Tuple[List[Task], List[Task], List[Task]]:
    sanity = [
        Task("sanity-%d" % seed, "A yellow dragon waves once in a bright stable room.", "sanity", seed, expected={"width": 352, "height": 640, "frames": 22})
        for seed in SEEDS
    ]

    def expand(scenes: Sequence[str], split: str) -> List[Task]:
        tasks = []
        for prompt_index, (scene, variant) in enumerate(((scene, variant) for scene in scenes for variant in VARIANTS), 1):
            for seed in SEEDS:
                tasks.append(
                    Task(
                        "%s-%03d-s%d" % (split, prompt_index, seed),
                        scene + " " + variant,
                        split,
                        seed,
                        constraints={"subject": "fixed yellow dragon", "avoid": "deformation, flicker, identity drift"},
                        expected={"width": 352, "height": 640, "frames": 22},
                    )
                )
        return tasks

    return sanity, expand(DEV_SCENES, "dev"), expand(HELDOUT_SCENES, "heldout")


class ResourceGuard:
    def __init__(self, base_url: str, limit: float = 0.95):
        self.base_url = base_url.rstrip("/")
        self.limit = limit
        self.tripped = False
        self.peak_ratio = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop.wait(1):
            try:
                with urllib.request.urlopen(self.base_url + "/system_stats", timeout=5) as response:
                    data = json.load(response)
                devices = data.get("devices") or []
                if not devices:
                    continue
                total = float(devices[0].get("vram_total", 0))
                free = float(devices[0].get("vram_free", total))
                ratio = (total - free) / total if total else 0.0
                self.peak_ratio = max(self.peak_ratio, ratio)
                if ratio >= self.limit:
                    self.tripped = True
                    body = json.dumps({}).encode()
                    request = urllib.request.Request(self.base_url + "/interrupt", data=body, headers={"Content-Type": "application/json"}, method="POST")
                    urllib.request.urlopen(request, timeout=5).close()
                    return
            except Exception:
                continue


def build_components(config: AppConfig) -> Tuple[HarnessRunner, TrajectoryStore, CandidateStore, str]:
    template = load_workflow(config.workflow.template)
    base_url = os.environ.get("COMFYUI_BASE_URL", config.backend.base_url).strip().rstrip("/")
    backend = MiniMaxH3Adapter(
        base_url,
        config.backend.request_timeout_s,
        config.backend.poll_interval_s,
        config.backend.task_timeout_s,
    )
    evaluator = SubprocessEvaluator(config.evaluator.command, config.evaluator.timeout_s)
    trajectories = TrajectoryStore(config.runtime.trajectory_path)
    archive = CandidateStore(config.runtime.archive_dir)
    baseline_policy = {
        "prompt": {"prefix": "", "suffix": ""},
        "context": {"include_constraints": True, "max_prompt_chars": 4000},
        "workflow": {
            key: template[target.node_id]["inputs"][target.input_name]
            for key, target in config.workflow.mutable.items()
        },
    }
    archive.initialize(baseline_policy)
    runner = HarnessRunner(template, config.workflow, backend, evaluator, trajectories, config.runtime.output_dir)
    return runner, trajectories, archive, base_url


def latest_by_task(history: Iterable[Trajectory], version: str, split: str) -> Dict[str, Trajectory]:
    result = {}
    for item in history:
        if item.harness_version == version and item.split == split:
            result[item.task_id] = item
    return result


def is_retryable_interruption(item: Trajectory) -> bool:
    if item.failure_type != "backend_execution":
        return False
    return "execution_interrupted" in json.dumps(item.steps, ensure_ascii=False)


def run_missing(
    runner: HarnessRunner,
    store: TrajectoryStore,
    tasks: Sequence[Task],
    candidate: Candidate,
    base_url: str,
) -> List[Trajectory]:
    existing = latest_by_task(store.read(), candidate.id, tasks[0].split) if tasks else {}
    total = len(tasks)
    for index, task in enumerate(tasks, 1):
        if task.id in existing and not is_retryable_interruption(existing[task.id]):
            continue
        guard = ResourceGuard(base_url)
        guard.start()
        result = runner.run_task(task, candidate)
        guard.stop()
        existing[task.id] = result
        print(json.dumps({"event": "task", "version": candidate.id, "split": task.split, "index": index, "total": total, "task_id": task.id, "score": result.score, "failure": result.failure_type, "peak_vram_ratio": round(guard.peak_ratio, 4)}), flush=True)
        if guard.tripped:
            raise RuntimeError("VRAM safety limit reached; study stopped after %s" % task.id)
    return [existing[task.id] for task in tasks]


def paired_scores(parent: Sequence[Trajectory], child: Sequence[Trajectory]) -> List[Tuple[float, float]]:
    p = {item.task_id: float(item.score or 0) for item in parent}
    c = {item.task_id: float(item.score or 0) for item in child}
    return [(p[key], c[key]) for key in sorted(set(p) & set(c))]


def grouped_differences(parent: Sequence[Trajectory], child: Sequence[Trajectory]) -> List[float]:
    pairs = paired_scores(parent, child)
    by_group: Dict[str, List[float]] = {}
    child_by_id = {item.task_id: float(item.score or 0) for item in child}
    parent_by_id = {item.task_id: float(item.score or 0) for item in parent}
    for task_id in set(parent_by_id) & set(child_by_id):
        group = task_id.rsplit("-s", 1)[0]
        by_group.setdefault(group, []).append(child_by_id[task_id] - parent_by_id[task_id])
    return [mean(values) for values in by_group.values()]


def bootstrap_ci(values: Sequence[float], samples: int = 10000) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(42)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return means[int(samples * 0.025)], means[int(samples * 0.975)]


def run_study(config: AppConfig, max_generations: int) -> Mapping[str, Any]:
    sanity, dev, heldout = build_tasks()
    runner, trajectories, archive, base_url = build_components(config)
    h0 = archive.get("H0")
    run_missing(runner, trajectories, dev, h0, base_url)
    last_candidate: Optional[Candidate] = None

    for _ in range(max_generations):
        parent = archive.active()
        parent_dev = run_missing(runner, trajectories, dev, parent, base_url)
        diagnosis = diagnose(parent_dev, config.evolution.recurrence_threshold, config.evolution.low_score_threshold)
        if diagnosis is None:
            break
        existing_children = [item for item in archive.lineage() if item.parent == parent.id]
        dropped = [item for item in existing_children if (archive.outcome(item.id) or {}).get("status") == "dropped"]
        if dropped:
            last_candidate = dropped[-1]
            break
        pending = [item for item in existing_children if archive.outcome(item.id) is None]
        candidate = pending[-1] if pending else propose_mutation(parent, diagnosis, archive.next_id())
        if candidate is None:
            break
        if not pending:
            archive.create(candidate)
        last_candidate = candidate
        sanity_results = run_missing(runner, trajectories, sanity, candidate, base_url)
        child_dev = run_missing(runner, trajectories, dev, candidate, base_url)
        pairs = paired_scores(parent_dev, child_dev)
        parent_score = mean(pair[0] for pair in pairs)
        child_score = mean(pair[1] for pair in pairs)
        sanity_passed = all(not item.critical_regression and float(item.score or 0) > 0 for item in sanity_results)
        critical = any(item.critical_regression for item in child_dev)
        regressions = sum(1 for p, c in pairs if c < p)
        promote = should_promote(parent_score, child_score, sanity_passed, critical, regressions, config.evolution)
        status = "promoted" if promote else "dropped"
        archive.record_outcome(candidate.id, {"status": status, "parent_id": parent.id, "candidate_id": candidate.id, "parent_score": parent_score, "candidate_score": child_score, "sanity_passed": sanity_passed, "critical_regression": critical, "regression_count": regressions})
        if not promote:
            break
        archive.promote(candidate.id)

    comparison = last_candidate or archive.active()
    h0_heldout = run_missing(runner, trajectories, heldout, h0, base_url)
    candidate_heldout = run_missing(runner, trajectories, heldout, comparison, base_url) if comparison.id != h0.id else h0_heldout
    differences = grouped_differences(h0_heldout, candidate_heldout)
    lower, upper = bootstrap_ci(differences)
    report = {
        "baseline": h0.id,
        "candidate": comparison.id,
        "candidate_status": (archive.outcome(comparison.id) or {}).get("status", "active"),
        "dev_prompts": len(DEV_SCENES) * len(VARIANTS),
        "heldout_prompts": len(HELDOUT_SCENES) * len(VARIANTS),
        "seeds_per_prompt": len(SEEDS),
        "h0_heldout_score": mean(float(item.score or 0) for item in h0_heldout),
        "candidate_heldout_score": mean(float(item.score or 0) for item in candidate_heldout),
        "heldout_delta": mean(differences) if differences else 0.0,
        "bootstrap_95_ci": [lower, upper],
        "evidence_of_improvement": bool(differences and lower > 0 and mean(differences) >= 0.02 and archive.active_id != "H0"),
    }
    report_path = config.runtime.archive_dir.parent / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/power-study.yaml")
    parser.add_argument("--max-generations", type=int, default=10)
    args = parser.parse_args()
    run_study(load_config(Path(args.config)), args.max_generations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
