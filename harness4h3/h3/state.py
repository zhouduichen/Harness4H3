from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ModelState:
    model_id: str
    parent_model_id: Optional[str]
    checkpoint_path: str
    architecture_name: str
    parameter_count: Optional[int] = None
    trainable_parameter_count: Optional[int] = None
    num_blocks: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    ffn_width: Optional[int] = None
    dtype: str = "float16"
    quantization: Mapping[str, Any] = field(default_factory=dict)
    sampling_steps: Optional[int] = None
    components: Mapping[str, Any] = field(default_factory=dict)
    algorithm_state: Mapping[str, Any] = field(default_factory=dict)
    runtime_state: Mapping[str, Any] = field(default_factory=dict)
    measured_metrics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.model_id or not self.checkpoint_path or not self.architecture_name:
            raise ValueError("model state requires model_id, checkpoint_path, and architecture_name")
        for name in ("parameter_count", "trainable_parameter_count", "num_blocks", "hidden_size", "num_attention_heads", "ffn_width", "sampling_steps"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError("%s must be non-negative" % name)

    @classmethod
    def fake_baseline(cls, model_id: str = "M0000") -> "ModelState":
        return cls(
            model_id=model_id,
            parent_model_id=None,
            checkpoint_path="fake://%s" % model_id,
            architecture_name="FakeMiniMaxH3",
            parameter_count=13_000_000_000,
            trainable_parameter_count=0,
            num_blocks=40,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_width=11008,
            dtype="float16",
            quantization={"bits": 16, "scheme": "none"},
            sampling_steps=50,
            components={"teacher": "MiniMax-H3"},
            algorithm_state={"distillation": "none"},
            runtime_state={"backend": "fake"},
            measured_metrics={
                "quality_score": 0.90,
                "latency_s": 60.0,
                "peak_memory_gb": 12.0,
                "model_size_gb": 10.0,
                "energy_j": 120.0,
                "throughput": 1.0 / 60.0,
            },
            provenance={"kind": "fake_baseline"},
        )

    def derive(self, model_id: str, **changes: Any) -> "ModelState":
        payload = {
            "model_id": model_id,
            "parent_model_id": self.model_id,
            "checkpoint_path": "fake://%s" % model_id,
        }
        payload.update(changes)
        return replace(self, **copy.deepcopy(payload))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelState":
        fields = cls.__dataclass_fields__
        return cls(**{key: copy.deepcopy(value) for key, value in raw.items() if key in fields})
