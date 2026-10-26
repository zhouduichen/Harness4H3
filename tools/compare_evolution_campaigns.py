#!/usr/bin/env python3
"""Compare matched single and Pareto EvoCompress campaign ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness4h3.evolution.evaluator import PromotionPolicy


def _summarize(root: Path) -> dict[str, Any]:
    state_path = root / "branch-campaign.json"
    events_path = root / "branch-events.jsonl"
    if not state_path.is_file() or not events_path.is_file():
        raise ValueError("%s is missing its campaign state or cost ledger" % root)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    raw_events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reconciliation_path = root / "accounting-reconciliation.jsonl"
    reconciliations = {}
    if reconciliation_path.is_file():
        reconciliations = {
            str(item["round_id"]): item
            for item in (json.loads(line) for line in reconciliation_path.read_text(encoding="utf-8").splitlines() if line.strip())
        }
    events = []
    for raw_event in raw_events:
        event = dict(raw_event)
        correction = reconciliations.get(str(event.get("round_id")))
        if correction is not None:
            costs = dict(event["costs_seconds"])
            costs["unattributed_seconds"] = max(0.0, float(correction["measured_round_wall_seconds"]) - float(event["total_seconds"]))
            event["costs_seconds"] = costs
            event["total_seconds"] = float(correction["measured_round_wall_seconds"])
        events.append(event)
    cumulative = 0.0
    first_qualified = None
    failed_cost = 0.0
    cost_fields = (
        "parent_load_seconds", "profile_seconds", "generation_seconds", "sandbox_seconds",
        "training_seconds", "verification_seconds", "evaluation_seconds", "retry_seconds", "unattributed_seconds",
    )
    totals = {key: 0.0 for key in cost_fields}
    for event in events:
        costs = event["costs_seconds"]
        cumulative += float(event["total_seconds"])
        for key in totals:
            totals[key] += float(costs.get(key, 0.0))
        if event.get("status") == "promoted" and first_qualified is None:
            first_qualified = {"candidate_id": event.get("child_model_id"), "cumulative_search_seconds": cumulative}
        if event.get("status") != "promoted":
            failed_cost += float(event["total_seconds"])
    front_path = root / "pareto-front.json"
    front = json.loads(front_path.read_text(encoding="utf-8")).get("entries", []) if front_path.is_file() else []
    return {
        "campaign": state,
        "rounds": len(events),
        "spent_seconds": cumulative,
        "ledger_spent_seconds": float(state.get("spent_seconds", 0.0)),
        "accounting_reconciled": bool(reconciliations),
        "stage_totals_seconds": totals,
        "first_qualified": first_qualified,
        "failed_or_rejected_seconds": failed_cost,
        "pareto_front": front,
        "events": events,
        "root": str(root),
    }


def _prefix(summary: dict[str, Any], budget_seconds: float) -> dict[str, Any]:
    events = summary["events"]
    cumulative = 0.0
    included = []
    for event in events:
        next_total = cumulative + float(event["total_seconds"])
        if next_total > budget_seconds:
            break
        cumulative = next_total
        included.append(event)
    first_qualified = None
    failed_cost = 0.0
    totals = {key: 0.0 for key in (
        "parent_load_seconds", "profile_seconds", "generation_seconds", "sandbox_seconds",
        "training_seconds", "verification_seconds", "evaluation_seconds", "retry_seconds", "unattributed_seconds",
    )}
    for event in included:
        for key in totals:
            totals[key] += float(event["costs_seconds"].get(key, 0.0))
        if event.get("status") == "promoted" and first_qualified is None:
            first_qualified = {
                "candidate_id": event.get("child_model_id"),
                "cumulative_search_seconds": sum(float(item["total_seconds"]) for item in included[: included.index(event) + 1]),
            }
        if event.get("status") != "promoted":
            failed_cost += float(event["total_seconds"])
    return {
        "rounds": len(included),
        "spent_seconds": cumulative,
        "stage_totals_seconds": totals,
        "first_qualified": first_qualified,
        "failed_or_rejected_seconds": failed_cost,
        "events": included,
        "root": summary["root"],
    }


def _front_at_budget(root: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initial_path = root / "initial-pareto-front.json"
    raw = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.is_file() else {"entries": []}
    entries = {str(item["candidate_id"]): dict(item) for item in raw.get("entries", []) if item.get("candidate_id")}
    for event in events:
        candidate_id = event.get("child_model_id")
        if event.get("status") != "promoted" or not candidate_id:
            continue
        path = root / "models" / "candidates" / (str(candidate_id) + ".json")
        if not path.is_file():
            front_path = root / "pareto-front.json"
            if front_path.is_file():
                final_entries = json.loads(front_path.read_text(encoding="utf-8")).get("entries", [])
                match = next((item for item in final_entries if str(item.get("candidate_id")) == str(candidate_id)), None)
                if match is not None:
                    entries[str(candidate_id)] = dict(match)
            continue
        candidate = json.loads(path.read_text(encoding="utf-8"))
        metrics = candidate.get("state", {}).get("measured_metrics") or {}
        entries[str(candidate_id)] = {"candidate_id": str(candidate_id), **dict(metrics)}
    values = list(entries.values())
    return [
        item for item in values
        if item.get("valid") is True and item.get("offline_simulation") is not True
        and not any(
            other.get("candidate_id") != item.get("candidate_id")
            and PromotionPolicy._dominates(other, item)
            for other in values
        )
    ]


def compare(single_root: Path, pareto_root: Path) -> dict[str, Any]:
    single, pareto = _summarize(single_root), _summarize(pareto_root)
    left, right = single["campaign"], pareto["campaign"]
    for field in ("protocol_sha256", "budget_seconds", "baseline_quality", "deployment_cost_metric"):
        if left.get(field) != right.get(field):
            raise ValueError("campaigns are not matched: %s differs" % field)
    matched_budget = min(float(single["spent_seconds"]), float(pareto["spent_seconds"]))
    single_matched = _prefix(single, matched_budget)
    pareto_matched = _prefix(pareto, matched_budget)
    single_matched["pareto_front"] = _front_at_budget(single_root, single_matched["events"])
    pareto_matched["pareto_front"] = _front_at_budget(pareto_root, pareto_matched["events"])
    single.pop("events", None)
    single.pop("root", None)
    pareto.pop("events", None)
    pareto.pop("root", None)
    single_matched.pop("events", None)
    single_matched.pop("root", None)
    pareto_matched.pop("events", None)
    pareto_matched.pop("root", None)
    return {
        "protocol_sha256": left["protocol_sha256"],
        "budget_seconds": left["budget_seconds"],
        "matched_compute_seconds": matched_budget,
        "single": single,
        "pareto": pareto,
        "single_at_matched_compute": single_matched,
        "pareto_at_matched_compute": pareto_matched,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", required=True, type=Path)
    parser.add_argument("--pareto", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.single, args.pareto)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
