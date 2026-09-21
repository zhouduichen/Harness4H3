"""Target-device export, deployment, and evidence contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..campaign.gates import MetricEvidence


EDGE_METRICS = (
    "edge_exported",
    "edge_quantized",
    "edge_runtime",
    "edge_device",
    "edge_latency",
    "edge_memory",
    "edge_energy",
    "edge_thermal",
)


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EdgeEvidence:
    metric_name: str
    value: float
    artifact_sha256: str
    target_device_id: str
    measurement_reference: str
    metric_version: str = "edge-v1"
    hard: bool = True

    def __post_init__(self) -> None:
        if self.metric_name not in EDGE_METRICS:
            raise ValueError("unsupported edge metric: %s" % self.metric_name)
        if not self.artifact_sha256 or not self.target_device_id or not self.measurement_reference:
            raise ValueError("edge evidence identity is incomplete")

    def to_metric_evidence(self) -> MetricEvidence:
        return MetricEvidence(
            metric_name=self.metric_name,
            metric_version=self.metric_version,
            input_reference=self.measurement_reference,
            value=float(self.value),
            confidence_or_validity=True,
            evidence_source="target-device",
            device_profile_id=self.target_device_id,
            hard=self.hard,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EdgeEvidence":
        required = {
            "metric_name",
            "value",
            "artifact_sha256",
            "target_device_id",
            "measurement_reference",
        }
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise ValueError("edge evidence is missing required fields")
        return cls(
            metric_name=str(raw["metric_name"]),
            value=float(raw["value"]),
            artifact_sha256=str(raw["artifact_sha256"]),
            target_device_id=str(raw["target_device_id"]),
            measurement_reference=str(raw["measurement_reference"]),
            metric_version=str(raw.get("metric_version", "edge-v1")),
            hard=bool(raw.get("hard", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": float(self.value),
            "artifact_sha256": self.artifact_sha256,
            "target_device_id": self.target_device_id,
            "measurement_reference": self.measurement_reference,
            "metric_version": self.metric_version,
            "hard": self.hard,
        }


def validate_edge_evidence(items: Sequence[EdgeEvidence], *, target_device_id: str | None = None) -> tuple[EdgeEvidence, ...]:
    evidence = tuple(items)
    if any(not isinstance(item, EdgeEvidence) for item in evidence):
        raise ValueError("target-device evidence contains an invalid item")
    if len(evidence) != len(EDGE_METRICS) or {item.metric_name for item in evidence} != set(EDGE_METRICS):
        raise ValueError("target-device evidence must contain exactly one item for every edge metric")
    if len({item.artifact_sha256 for item in evidence}) != 1:
        raise ValueError("target-device evidence has inconsistent artifact identity")
    devices = {item.target_device_id for item in evidence}
    if "server" in devices or len(devices) != 1:
        raise ValueError("target-device evidence has inconsistent device identity")
    if target_device_id is not None and devices != {str(target_device_id)}:
        raise ValueError("target-device evidence is for the wrong target device")
    return evidence


class TargetDeviceRunner(Protocol):
    def export(self, checkpoint: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Path:
        ...

    def quantize(self, exported: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Path:
        ...

    def compile(self, quantized: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Path:
        ...

    def deploy(self, compiled: Path, output_dir: Path, proposal: Mapping[str, Any]) -> str:
        ...

    def benchmark(self, deployment_id: str, compiled: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Mapping[str, float]:
        ...


class TargetDeviceEvaluator:
    """Run the complete target-device artifact path and emit edge-only evidence."""

    def __init__(self, runner: TargetDeviceRunner, *, target_device_id: str):
        self.runner = runner
        self.target_device_id = str(target_device_id).strip()
        if not self.target_device_id:
            raise ValueError("target_device_id must not be empty")

    def evaluate(self, checkpoint: Path, proposal: Any, round_dir: Path) -> tuple[EdgeEvidence, ...]:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("target device checkpoint does not exist: %s" % checkpoint)
        output_dir = Path(round_dir).resolve() / "target-device"
        output_dir.mkdir(parents=True, exist_ok=True)
        proposal_payload = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal)
        exported = Path(self.runner.export(checkpoint, output_dir / "export", proposal_payload)).resolve()
        quantized = Path(self.runner.quantize(exported, output_dir / "quantize", proposal_payload)).resolve()
        compiled = Path(self.runner.compile(quantized, output_dir / "compile", proposal_payload)).resolve()
        for name, path in (("export", exported), ("quantize", quantized), ("compile", compiled)):
            if not path.is_file():
                raise ValueError("target device %s artifact is missing" % name)
        deployment_id = str(self.runner.deploy(compiled, output_dir / "deploy", proposal_payload)).strip()
        if not deployment_id:
            raise ValueError("target device deployment returned no identity")
        measurements = dict(self.runner.benchmark(deployment_id, compiled, output_dir / "benchmark", proposal_payload))
        artifact_hash = artifact_sha256(compiled)
        reference = str(output_dir / "benchmark" / "edge-evidence.json")
        payload = {
            "deployment_id": deployment_id,
            "artifact_sha256": artifact_hash,
            "target_device_id": self.target_device_id,
            "measurements": measurements,
        }
        Path(reference).parent.mkdir(parents=True, exist_ok=True)
        Path(reference).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        values = {
            "edge_exported": 1.0,
            "edge_quantized": 1.0,
            "edge_runtime": 1.0,
            "edge_device": 1.0,
            "edge_latency": float(measurements["latency_s"]),
            "edge_memory": float(measurements["memory_gb"]),
            "edge_energy": float(measurements["energy_j"]),
            "edge_thermal": float(measurements["thermal_c"]),
        }
        return tuple(
            EdgeEvidence(name, value, artifact_hash, self.target_device_id, reference)
            for name, value in values.items()
        )


class FakeTargetDeviceRunner:
    """Deterministic target runner used by CPU contract tests."""

    def export(self, checkpoint: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Path:
        del proposal
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "student.export"
        shutil.copyfile(checkpoint, target)
        return target

    def quantize(self, exported: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Path:
        del proposal
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "student.quantized"
        shutil.copyfile(exported, target)
        return target

    def compile(self, quantized: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Path:
        del proposal
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "student.runtime"
        shutil.copyfile(quantized, target)
        return target

    def deploy(self, compiled: Path, output_dir: Path, proposal: Mapping[str, Any]) -> str:
        del compiled, proposal
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "deployment.json").write_text("{\"status\": \"deployed\"}\n", encoding="utf-8")
        return "fake-edge-deployment"

    def benchmark(self, deployment_id: str, compiled: Path, output_dir: Path, proposal: Mapping[str, Any]) -> Mapping[str, float]:
        del deployment_id, compiled, proposal
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"latency_s": 0.0125, "memory_gb": 0.5, "energy_j": 1.25, "thermal_c": 48.0}


__all__ = [
    "EDGE_METRICS",
    "EdgeEvidence",
    "FakeTargetDeviceRunner",
    "TargetDeviceEvaluator",
    "TargetDeviceRunner",
    "artifact_sha256",
    "validate_edge_evidence",
]
