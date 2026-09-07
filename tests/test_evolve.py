from __future__ import annotations

from harness4h3.archive.store import CandidateStore
from harness4h3.config import EvolutionConfig
from harness4h3.harness.state import Task
from harness4h3.memory.trajectory import Trajectory
from harness4h3.archive.store import Candidate, default_policy
from harness4h3.self_improve.evolve import Diagnosis, EvolutionController, diagnose, propose_mutation


def trajectory(task_id, score, failure=None, critical=False, version="H0", split="dev"):
    return Trajectory(task_id, version, split, {}, [], None, score, failure, {"wall_time": 1.0}, critical_regression=critical)


def test_diagnosis_requires_recurring_failure():
    assert diagnose([trajectory("a", 0, "low_luma"), trajectory("b", 0, "low_luma")]).failure_type == "low_luma"
    assert diagnose([trajectory("a", 0, "low_luma"), trajectory("b", 1.0)]) is None


def test_evolution_promotes_only_better_candidate(tmp_path):
    store = CandidateStore(tmp_path)
    store.initialize()
    tasks = [Task("sanity", "test", "sanity"), Task("a", "one", "dev"), Task("b", "two", "dev")]
    history = [trajectory("a", 0.5), trajectory("b", 0.5)]

    def run_batch(selected, candidate):
        score = 0.8 if candidate.id != "H0" else 0.5
        return [trajectory(task.id, score, version=candidate.id, split=task.split) for task in selected]

    config = EvolutionConfig(0.01, 2, 0, 0.65)
    outcome = EvolutionController(store, config, run_batch).evolve(tasks, history)
    assert outcome.status == "promoted"
    assert store.active_id == "H1"


def test_evolution_drops_critical_regression(tmp_path):
    store = CandidateStore(tmp_path)
    store.initialize()
    tasks = [Task("sanity", "test", "sanity"), Task("a", "one", "dev"), Task("b", "two", "dev")]
    history = [trajectory("a", 0.5), trajectory("b", 0.5)]

    def run_batch(selected, candidate):
        critical = candidate.id != "H0" and any(task.split == "dev" for task in selected)
        return [trajectory(task.id, 0.8, critical=critical, version=candidate.id, split=task.split) for task in selected]

    config = EvolutionConfig(0.01, 2, 0, 0.65)
    outcome = EvolutionController(store, config, run_batch).evolve(tasks, history)
    assert outcome.status == "dropped"
    assert store.active_id == "H0"


def test_low_score_mutations_advance_without_repeating_same_patch():
    diagnosis = Diagnosis("low_score", ["a", "b"], "repeated low score")
    h0 = Candidate("H0", None, 0, "baseline", {}, "", default_policy())
    h1 = propose_mutation(h0, diagnosis, "H1")
    assert h1 is not None
    h2 = propose_mutation(h1, diagnosis, "H2")
    assert h2 is not None
    assert h2.patch != h1.patch
