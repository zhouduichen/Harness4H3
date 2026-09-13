import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from h3_training.algorithms.base import StepOutput, TrainingMethod
from h3_training.data.schema import ModalLatents, ModalSchedule, TrainingSample


ROOT = Path(__file__).resolve().parents[2]


def test_modal_latents_requires_a_modality():
    with pytest.raises(ValueError, match="at least one modality"):
        ModalLatents()


def test_modal_latents_rejects_nonfinite_and_batch_mismatch():
    with pytest.raises(ValueError, match="finite"):
        ModalLatents(video=torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="batch sizes"):
        ModalLatents(video=torch.zeros(1, 2), audio=torch.zeros(2, 2))


def test_schedule_requires_monotonic_terminal_grid():
    assert ModalSchedule(video_sigmas=(1.0, 0.5, 0.0)).nfe == 2
    with pytest.raises(ValueError, match="strictly decreasing"):
        ModalSchedule(video_sigmas=(1.0, 0.5, 0.6, 0.0))
    with pytest.raises(ValueError, match="end at sigma 0"):
        ModalSchedule(audio_sigmas=(1.0, 0.1))


def test_training_sample_keeps_raw_fields_optional():
    sample = TrainingSample("s0", "a prompt", 7)
    assert sample.latents is None
    assert sample.text_embedding is None


def test_step_output_enforces_total_scalar_loss():
    with pytest.raises(ValueError, match="total_loss"):
        StepOutput(losses={})
    with pytest.raises(ValueError, match="scalar"):
        StepOutput(losses={"total_loss": torch.ones(2)})


def test_training_method_is_abstract():
    with pytest.raises(TypeError):
        TrainingMethod()


def test_plain_harness_import_does_not_import_torch():
    subprocess.run(
        [sys.executable, "-c", "import sys; sys.modules['torch'] = None; import harness4h3"],
        cwd=ROOT,
        check=True,
    )
