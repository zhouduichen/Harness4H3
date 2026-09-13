import pytest

pytest.importorskip("torch")

from h3_training.adapters.h3_contract import H3AdapterContract
from h3_training.engine.state import TrainingFailure


def test_h3_contract_exposes_confirmed_schedule_and_fails_without_loader(tmp_path):
    adapter = H3AdapterContract()
    schedule = adapter.schedule(4)
    assert schedule.video_sigmas[1] != schedule.audio_sigmas[1]
    assert schedule.video_sigmas[-1] == schedule.audio_sigmas[-1] == 0.0
    with pytest.raises(TrainingFailure, match="h3_adapter_unavailable"):
        adapter.load_role(tmp_path / "parent", trainable=False)
