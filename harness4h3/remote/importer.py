"""Import remote trainer metadata without transferring checkpoint bytes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..memory.experience import ExperienceRecord
from .ssh import RemoteError, SSHClient


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _complete_evaluation(value: Any) -> bool:
    evaluation = _mapping(value)
    hardware = _mapping(evaluation.get("hardware"))
    quality = evaluation.get("quality_score", evaluation.get("quality"))
    return all(isinstance(item, (int, float)) for item in (quality, hardware.get("latency_s"), hardware.get("peak_memory_gb"), hardware.get("energy_j")))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_item(item: Any) -> Tuple[str, str, Mapping[str, Any], Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    if isinstance(item, Mapping):
        source_uri = str(item.get("source_uri") or item.get("uri") or "")
        source_sha256 = str(item.get("source_sha256") or item.get("sha256") or "")
        result = _mapping(item.get("result") if "result" in item else item)
        evidence = _mapping(item.get("evidence")) if item.get("evidence") is not None else None
        request = _mapping(item.get("request")) if item.get("request") is not None else None
        if not source_uri and item.get("path"):
            source_uri = "file://%s" % Path(str(item["path"])).resolve()
        return source_uri, source_sha256, result, evidence, request
    path = Path(item)
    raw = json.loads(path.read_text(encoding="utf-8"))
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "file://%s" % path.resolve(), digest, _mapping(raw), None, None


def normalize_trainer_result(
    source_uri: str,
    source_sha256: str,
    result: Mapping[str, Any],
    request: Optional[Mapping[str, Any]] = None,
    evidence: Optional[Mapping[str, Any]] = None,
) -> ExperienceRecord:
    """Convert one worker result into a versioned experience record."""

    request = _mapping(request)
    evidence = _mapping(evidence)
    output_state = _mapping(result.get("output_state") or result.get("child_state"))
    metrics = dict(_mapping(result.get("metrics")))
    metrics.update(_mapping(result.get("training_metrics")))
    training = dict(_mapping(result.get("training")))
    training.update(metrics)
    cost = _mapping(result.get("cost"))
    if cost:
        # Keep measured worker cost with the durable training experience so a
        # RoundPolicy can enforce its GPU-hour budget after a restart.  This
        # is metadata only; it never affects checkpoint lineage.
        training.setdefault("cost", dict(cost))
    algorithm_state = _mapping(output_state.get("algorithm_state"))
    operator = result.get("operator") or algorithm_state.get("operator") or request.get("operator")
    operator_args = dict(_mapping(request.get("operator_args")))
    if not operator_args:
        operator_args = {str(key): value for key, value in algorithm_state.items() if key != "operator"}
    experiment_id = str(
        result.get("experiment_id")
        or request.get("experiment_id")
        or request.get("id")
        or output_state.get("model_id")
        or Path(source_uri).stem
    )
    status_value = str(result.get("status", "failed")).strip().lower()
    failed = status_value not in {"success", "succeeded", "ok", "completed"}
    child_id = None if failed else (output_state.get("model_id") or result.get("child_model_id") or request.get("child_model_id"))
    parent_id = output_state.get("parent_model_id") or result.get("parent_model_id") or request.get("parent_model_id")
    evaluation_value: Optional[Mapping[str, Any]] = None
    for candidate in (result.get("evaluation"), evidence.get("evaluation"), evidence.get("benchmark")):
        if isinstance(candidate, Mapping):
            evaluation_value = dict(candidate)
            break
    if failed:
        record_status = "failed"
    elif _complete_evaluation(evaluation_value):
        record_status = "evaluated_candidate"
    else:
        record_status = "training_only_unvalidated"
    decision = dict(_mapping(result.get("decision")))
    decision.setdefault("status", "failed" if failed else ("evaluated_candidate" if record_status == "evaluated_candidate" else "not_evaluated"))
    if failed:
        decision.setdefault("failure_type", result.get("failure_type") or "remote_training_failed")
        if result.get("message"):
            decision.setdefault("message", str(result["message"]))
    checkpoint = output_state.get("checkpoint_path") or result.get("checkpoint_path")
    provenance = dict(_mapping(result.get("provenance")))
    provenance.update({"source_uri": source_uri, "source_sha256": source_sha256})
    if output_state:
        provenance.setdefault("output_state", dict(output_state))
    if checkpoint:
        provenance.setdefault("remote_checkpoint_path", str(checkpoint))
    if evidence.get("path"):
        provenance.setdefault("evidence_path", str(evidence["path"]))
    return ExperienceRecord(
        experience_id="xp-%s-%s" % (str(child_id or experiment_id), source_sha256[:12]),
        source_uri=source_uri,
        source_sha256=source_sha256,
        source_kind="trainer_result",
        experiment_id=experiment_id,
        parent_model_id=str(parent_id) if parent_id else None,
        child_model_id=str(child_id) if child_id else None,
        operator=str(operator) if operator else None,
        operator_args=operator_args,
        training=training,
        evaluation=evaluation_value,
        decision=decision,
        reward=None,
        status=record_status,
        provenance=provenance,
        created_at=str(result.get("created_at") or request.get("created_at") or _now()),
    )


@dataclass(frozen=True)
class ImportSummary:
    imported: int
    skipped_duplicates: int
    corrupt: Tuple[Mapping[str, Any], ...]
    records: Tuple[ExperienceRecord, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "imported": self.imported,
            "skipped_duplicates": self.skipped_duplicates,
            "corrupt": [dict(item) for item in self.corrupt],
            "records": [item.to_dict() for item in self.records],
        }


class RemoteResultImporter:
    def __init__(self, store, client: Optional[SSHClient] = None):
        self.store = store
        self.client = client

    def import_results(self, files: Iterable[Any], requests: Iterable[Any] = ()) -> ImportSummary:
        request_by_child: Dict[str, Mapping[str, Any]] = {}
        for item in requests:
            try:
                _, _, raw, _, _ = _source_item(item)
                child_id = raw.get("child_model_id") or _mapping(raw.get("output_state")).get("model_id")
                if child_id:
                    request_by_child[str(child_id)] = raw
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        imported = 0
        duplicates = 0
        corrupt: List[Mapping[str, Any]] = []
        records: List[ExperienceRecord] = []
        for item in files:
            source_uri = str(item.get("source_uri") or item.get("uri") or "") if isinstance(item, Mapping) else str(item)
            try:
                uri, digest, result, evidence, request = _source_item(item)
                child_id = _mapping(result.get("output_state") or result.get("child_state")).get("model_id") or result.get("child_model_id")
                request = request or request_by_child.get(str(child_id))
                record = normalize_trainer_result(uri, digest, result, request=request, evidence=evidence)
                if self.store.append(record):
                    imported += 1
                    records.append(record)
                else:
                    duplicates += 1
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RemoteError) as exc:
                corrupt.append({"source_uri": source_uri, "error": str(exc)})
        return ImportSummary(imported, duplicates, tuple(corrupt), tuple(records))

    def discover_remote(self, root: Optional[str] = None) -> ImportSummary:
        if self.client is None:
            raise RemoteError("discover_remote requires an SSH client")
        if hasattr(self.client, "result_bundle"):
            files = []
            for item in self.client.result_bundle(root=root):
                path = str(item.get("path", ""))
                result = _mapping(item.get("result"))
                if not path or not result:
                    continue
                files.append(
                    {
                        "source_uri": "ssh://%s%s" % (self.client.config.host, path),
                        "source_sha256": str(item.get("sha256", "")),
                        "result": result,
                    }
                )
            return self.import_results(files)
        files: List[Mapping[str, Any]] = []
        for path in self.client.find("trainer_result_*.json", root=root):
            result = self.client.read_json(path)
            evidence = None
            evidence_path = result.get("evidence_path") if isinstance(result, Mapping) else None
            checkpoint = _mapping(result.get("output_state") if isinstance(result, Mapping) else {}).get("checkpoint_path")
            for candidate in (evidence_path, (str(checkpoint) + ".evidence.json") if checkpoint else None):
                if candidate:
                    try:
                        evidence = self.client.read_json(str(candidate))
                        break
                    except RemoteError:
                        pass
            files.append(
                {
                    "source_uri": "ssh://%s%s" % (self.client.config.host, path),
                    "source_sha256": self.client.sha256(path),
                    "result": result,
                    "evidence": evidence,
                }
            )
        return self.import_results(files)
