from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_FILES = (
    ROOT / "h3_training" / "algorithms" / "recovery_finetune.py",
    ROOT / "h3_training" / "algorithms" / "progressive_distillation.py",
    ROOT / "h3_training" / "algorithms" / "dmd2.py",
)


def test_algorithms_use_adapter_boundary_instead_of_tiny_implementation_details():
    forbidden = (
        "TinyH3Model",
        "h3_training.tiny",
        "video_head",
        "audio_head",
        "torch.save",
        "torch.load",
    )
    for path in ALGORITHM_FILES:
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path.name
