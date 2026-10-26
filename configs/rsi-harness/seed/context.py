def build_context(task, evidence):
    return {
        "task": task,
        "recent_evidence": list(evidence)[-8:],
    }
