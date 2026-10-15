"""Model profiling and sensitivity reports."""

from .calibration import CalibrationCorpus, CalibrationSample, CalibrationSampleSpec, FakeWanCalibrationProvider
from .layer_profiler import profile_model, profile_layers
from .report import ProfileReport
from .sensitivity import ChannelImportanceMethod, measure_attention_sensitivity, measure_block_sensitivity, measure_channel_sensitivity, measure_head_sensitivity, measure_sampling_step_sensitivity, measure_timestep_sensitivity, measure_token_sensitivity
from ..evolution.tica_calibration import TICACalibrationInput, TICACalibrationManifest, TICACalibrationSpec, TICAEvaluationSplits

__all__ = [
    "ProfileReport",
    "CalibrationCorpus",
    "CalibrationSample",
    "CalibrationSampleSpec",
    "FakeWanCalibrationProvider",
    "ChannelImportanceMethod",
    "measure_block_sensitivity",
    "measure_channel_sensitivity",
    "measure_attention_sensitivity",
    "measure_head_sensitivity",
    "measure_sampling_step_sensitivity",
    "measure_timestep_sensitivity",
    "measure_token_sensitivity",
    "TICACalibrationInput",
    "TICACalibrationManifest",
    "TICACalibrationSpec",
    "TICAEvaluationSplits",
    "profile_layers",
    "profile_model",
]
