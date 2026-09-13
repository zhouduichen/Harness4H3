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


def test_named_nonfinite_loss_is_rejected_even_when_total_is_finite():
    class BadAuxiliaryLoss(CountingMethod):
        def training_step(self, batch, iteration):
            output = super().training_step(batch, iteration)
            return StepOutput({**output.losses, "auxiliary": output.losses["total_loss"] * float("nan")})

    with pytest.raises(RuntimeError, match="nonfinite_loss.*auxiliary"):
        TrainerEngine().run(BadAuxiliaryLoss(), batches(), max_steps=1)


def test_engine_consumes_a_stream_without_materializing_the_loader():
    class StreamingBatches:
        def __iter__(self):
            for _ in range(2):
                yield PreparedBatch(Conditioning(torch.ones(1, 1)))

        def __len__(self):
            raise AssertionError("the training engine must not ask for loader length")

    method = CountingMethod()
    result = TrainerEngine(TrainerConfig(max_gradient_norm=10.0)).run(
        method, StreamingBatches(), max_steps=2
    )
    assert result.loop_state.sampler_position == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_engine_moves_model_and_prepared_batch_to_cuda():
    method = CountingMethod()
    result = TrainerEngine(
        TrainerConfig(device="cuda", max_gradient_norm=10.0)
    ).run(method, batches(), max_steps=1)
    assert result.optimizer_steps["student"] == 1
    assert next(method.parameters()).is_cuda
