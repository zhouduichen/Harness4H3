"""Model profiling and sensitivity reports."""

from .calibration import CalibrationCorpus, CalibrationSample, CalibrationSampleSpec, FakeWanCalibrationProvider
from .layer_profiler import profile_model, profile_layers
from .report import ProfileReport
from .sensitivity import ChannelImportanceMethod, measure_block_sensitivity, measure_channel_sensitivity, measure_timestep_sensitivity

__all__ = [
    "ProfileReport",
    "CalibrationCorpus",
    "CalibrationSample",
    "CalibrationSampleSpec",
    "FakeWanCalibrationProvider",
    "ChannelImportanceMethod",
    "measure_block_sensitivity",
    "measure_channel_sensitivity",
    "measure_timestep_sensitivity",
    "profile_layers",
    "profile_model",
]
