"""Source-confirmed MiniMax-H3 scheduler contract, fail-closed for training."""

from pathlib import Path

from h3_training.adapters.tiny import TinyH3Adapter
from h3_training.engine.state import TrainingFailure


class H3AdapterContract(TinyH3Adapter):
    """Record confirmed H3 math without pretending a real loader exists.

    H3 uses data-ward velocity, ``t = 1 - sigma``, separate modality
    schedules, video shift 12, and audio shift 3. The inherited pure schedule
    and conversion methods encode those facts. Model I/O fails until the L40
    integration supplies a validated loader/exporter/backward path.
    """

    def _unavailable(self, capability: str):
        raise TrainingFailure("h3_adapter_unavailable", f"missing real MiniMax-H3 {capability}")

    def load_role(self, path: Path, trainable: bool = False):
        self._unavailable("loader")

    def prepare_batch(self, raw, generator):
        self._unavailable("batch preparation")

    def predict(self, role, noisy, timestep, conditioning):
        self._unavailable("forward/backward path")

    def save_role(self, role, path):
        self._unavailable("child exporter")

    def reload_role(self, path):
        self._unavailable("child reload path")

    def resolve_trainable_parameters(self, role, policy):
        self._unavailable("trainable parameter policy")
