"""Safe meta-device compilation for declarative Student proposals."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch.fx import symbolic_trace

from .model import build_student
from .proposal import StudentProposal, StudentTarget, ValidationReport, canonical_digest


class CompileError(ValueError):
    """Raised when a validated proposal cannot be compiled safely."""


@dataclass(frozen=True)
class CompileManifest:
    path: str
    compiler_version: str
    graph_status: str
    graph_digest: str
    proposal_digest: str
    proposal: Mapping[str, Any]
    target: Mapping[str, Any]
    parameter_count: int
    estimated_peak_memory_gb: float
    memory_estimator_version: str
    memory_breakdown: Mapping[str, float]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    conditioning_shape: Tuple[int, ...]
    module_names: Tuple[str, ...]
    manifest_digest: str

    def _payload(self) -> Dict[str, Any]:
        return {
            "compiler_version": self.compiler_version,
            "graph_status": self.graph_status,
            "graph_digest": self.graph_digest,
            "proposal_digest": self.proposal_digest,
            "proposal": dict(self.proposal),
            "target": dict(self.target),
            "parameter_count": self.parameter_count,
            "estimated_peak_memory_gb": self.estimated_peak_memory_gb,
            "memory_estimator_version": self.memory_estimator_version,
            "memory_breakdown": dict(self.memory_breakdown),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "conditioning_shape": list(self.conditioning_shape),
            "module_names": list(self.module_names),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["manifest_digest"] = self.manifest_digest
        return payload

    @classmethod
    def from_path(cls, path: Path) -> "CompileManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise CompileError("compile manifest must be an object")
        expected = canonical_digest({key: value for key, value in raw.items() if key != "manifest_digest"})
        if str(raw.get("manifest_digest", "")) != expected:
            raise CompileError("manifest_digest_mismatch")
        return cls(
            path=str(Path(path).resolve()),
            compiler_version=str(raw["compiler_version"]),
            graph_status=str(raw["graph_status"]),
            graph_digest=str(raw["graph_digest"]),
            proposal_digest=str(raw["proposal_digest"]),
            proposal=dict(raw["proposal"]),
            target=dict(raw["target"]),
            parameter_count=int(raw["parameter_count"]),
            estimated_peak_memory_gb=float(raw["estimated_peak_memory_gb"]),
            memory_estimator_version=str(raw.get("memory_estimator_version", "student-memory-v1")),
            memory_breakdown={str(key): float(value) for key, value in dict(raw.get("memory_breakdown") or {}).items()},
            input_shape=tuple(int(value) for value in raw["input_shape"]),
            output_shape=tuple(int(value) for value in raw["output_shape"]),
            conditioning_shape=tuple(int(value) for value in raw["conditioning_shape"]),
            module_names=tuple(str(value) for value in raw["module_names"]),
            manifest_digest=str(raw["manifest_digest"]),
        )


class StudentCompiler:
    VERSION = "student-compiler-v1"

    def __init__(self, target: StudentTarget = StudentTarget()):
        self.target = target

    @staticmethod
    def _report_error(report: ValidationReport) -> CompileError:
        return CompileError("proposal_invalid: %s" % "; ".join(report.errors))

    def compile(self, proposal: StudentProposal, output_dir: Path) -> CompileManifest:
        report = proposal.validate(self.target)
        if not report.ok:
            raise self._report_error(report)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model = build_student(proposal, device="meta", target=self.target)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if report.estimated_params != parameter_count:
            raise CompileError(
                "parameter_count_mismatch: estimated=%s constructed=%s"
                % (report.estimated_params, parameter_count)
            )
        input_shape = (
            1,
            self.target.latent_channels,
            self.target.latent_frames,
            self.target.latent_height,
            self.target.latent_width,
        )
        conditioning_shape = (1, 4, self.target.condition_dim)
        timestep_shape = (1,)
        # Construction on meta uses PyTorch's default float32 parameters. The
        # deployment precision is validated separately; compile-time shape
        # checking must not fail on an incidental dtype mismatch.
        video = torch.empty(input_shape, device="meta", dtype=torch.float32)
        conditioning = torch.empty(conditioning_shape, device="meta", dtype=torch.float32)
        timestep = torch.empty(timestep_shape, device="meta", dtype=torch.float32)
        try:
            with torch.no_grad():
                output = model(video, conditioning, timestep)
            traced = symbolic_trace(model)
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise CompileError("graph_compile_failed: %s" % exc) from exc
        output_shape = tuple(int(value) for value in output.shape)
        if output_shape != input_shape:
            raise CompileError("output shape %s does not match input shape %s" % (output_shape, input_shape))
        graph_text = str(traced.graph)
        module_names = tuple(name for name, _ in model.named_modules() if name)
        graph_digest = canonical_digest({"graph": graph_text, "modules": module_names})
        manifest_payload = {
            "compiler_version": self.VERSION,
            "graph_status": "compiled",
            "graph_digest": graph_digest,
            "proposal_digest": proposal.digest,
            "proposal": proposal.to_dict(),
            "target": asdict(self.target),
            "parameter_count": int(parameter_count),
            "estimated_peak_memory_gb": float(report.estimated_peak_memory_gb or 0.0),
            "memory_estimator_version": "student-memory-v2-dmd2-aware",
            "memory_breakdown": dict(report.memory_breakdown),
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "conditioning_shape": list(conditioning_shape),
            "module_names": list(module_names),
        }
        manifest_digest = canonical_digest(manifest_payload)
        manifest_path = output_dir / "compile_manifest.json"
        if manifest_path.exists():
            existing = CompileManifest.from_path(manifest_path)
            if existing.manifest_digest != manifest_digest:
                raise CompileError("refusing to overwrite a different compile manifest")
            return existing
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix="compile-manifest-", suffix=".json", dir=str(output_dir))
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({**manifest_payload, "manifest_digest": manifest_digest}, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return CompileManifest(
            path=str(manifest_path),
            compiler_version=self.VERSION,
            graph_status="compiled",
            graph_digest=graph_digest,
            proposal_digest=proposal.digest,
            proposal=proposal.to_dict(),
            target=asdict(self.target),
            parameter_count=int(parameter_count),
            estimated_peak_memory_gb=float(report.estimated_peak_memory_gb or 0.0),
            memory_estimator_version="student-memory-v2-dmd2-aware",
            memory_breakdown=dict(report.memory_breakdown),
            input_shape=input_shape,
            output_shape=output_shape,
            conditioning_shape=conditioning_shape,
            module_names=module_names,
            manifest_digest=manifest_digest,
        )


__all__ = ["CompileError", "CompileManifest", "StudentCompiler"]
