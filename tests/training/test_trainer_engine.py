import pytest

torch = pytest.importorskip("torch")
from torch import nn

from h3_training.algorithms.base import StepOutput, TrainingMethod
from h3_training.data.schema import Conditioning, PreparedBatch
from h3_training.engine.trainer import TrainerConfig, TrainerEngine


class CountingMethod(TrainingMethod):
    algorithm_name = "counting"

    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def training_step(self, batch, iteration):
        loss = (self.weight(batch.conditioning.text) - 1).square().mean()
        return StepOutput({"total_loss": loss})

    def optimizers(self, iteration):
        return {"student": self.optimizer}

    def grad_clip_targets(self, iteration):
        return {"student": self.weight}


def batches(count=4):
    return [PreparedBatch(Conditioning(torch.ones(1, 1))) for _ in range(count)]


def test_accumulation_steps_once_for_two_microbatches():
    method = CountingMethod()
    result = TrainerEngine(TrainerConfig(gradient_accumulation_steps=2)).run(method, batches(), max_steps=1)
    assert result.optimizer_steps["student"] == 1
    assert result.loop_state.microbatches_consumed == 2


def test_nonfinite_loss_has_stable_failure():
    method = CountingMethod()
    method.weight.weight.data.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite_loss"):
        TrainerEngine().run(method, batches(), max_steps=1)


def test_zero_gradient_has_stable_failure():
    method = CountingMethod()
    method.weight.weight.data.fill_(1.0)
    with pytest.raises(RuntimeError, match="zero_gradient"):
        TrainerEngine().run(method, batches(), max_steps=1)


def test_oom_has_stable_failure():
    class OOMMethod(CountingMethod):
        def training_step(self, batch, iteration):
            raise RuntimeError("DefaultCPUAllocator: out of memory")

    with pytest.raises(RuntimeError, match="training_oom"):
        TrainerEngine().run(OOMMethod(), batches(), max_steps=1)
