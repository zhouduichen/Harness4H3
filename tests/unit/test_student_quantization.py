from pathlib import Path

import torch
from safetensors.torch import save_file

from harness4h3.student.quantization import load_student_state, quantize_checkpoint


def test_int8_checkpoint_has_round_trip_state_and_metadata(tmp_path: Path):
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "student-int8.safetensors"
    original = {"weight": torch.tensor([-2.0, 0.25, 1.5]), "counter": torch.tensor([3], dtype=torch.int64)}
    save_file(original, str(source))

    count, size = quantize_checkpoint(source, destination, bits=8, metadata={"proposal_digest": "p"})
    restored, metadata = load_student_state(destination)

    assert count == 1
    assert size == destination.stat().st_size
    assert metadata["quantization"] == "int8"
    assert restored["counter"].item() == 3
    assert torch.allclose(restored["weight"], original["weight"], atol=0.02)
