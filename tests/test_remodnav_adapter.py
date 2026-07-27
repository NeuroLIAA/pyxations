"""Focused tests for the Polars-native REMoDNaV adapter path."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from pyxations.methods.eyemovement import REMoDNaV as remodnav_module


class _FakeClassifier:
    """Small deterministic stand-in for REMoDNaV's classifier."""

    last_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).last_kwargs = kwargs

    def preproc(self, eye_data, savgol_length):
        return eye_data

    def __call__(self, preprocessed, classify_isp=True, sort_events=True):
        return [
            (0.05, 0.10, "SACC", 1, 2, 3, 4, 0.5, 20, 10, 12),
            (0.101, 0.14, "FIXA", 3, 4, 3.5, 4.5, 0.2, 2, 1, 1.5),
        ]


def test_run_from_samples_does_not_mutate_config_or_invent_pupil(monkeypatch, tmp_path):
    monkeypatch.setattr(remodnav_module, "EyegazeClassifier", _FakeClassifier)
    samples = pl.DataFrame(
        {
            "tSample": np.arange(20, dtype=float) * 10,
            "X": np.arange(20, dtype=float),
            "Y": np.arange(20, dtype=float) + 1,
        }
    )
    config = {"eye": "Best"}

    detector = remodnav_module.RemodnavDetection(tmp_path, samples)
    fixations, saccades = detector.run_eye_movement_from_samples(
        100, config=config
    )

    assert config == {"eye": "Best"}
    assert isinstance(fixations, pl.DataFrame)
    assert isinstance(saccades, pl.DataFrame)
    assert fixations.height == 1
    assert saccades.height == 1
    assert math.isnan(fixations["pupilAvg"][0])
    assert fixations["tStart"][0] == 101.0
    assert saccades["tEnd"][0] == 100.0


def test_detect_eye_movements_processes_both_available_eyes(monkeypatch, tmp_path):
    monkeypatch.setattr(remodnav_module, "EyegazeClassifier", _FakeClassifier)
    sample_index = np.arange(20, dtype=float)
    samples = pl.DataFrame(
        {
            "tSample": sample_index * 10,
            "Rate_recorded": [100.0] * 20,
            "Calib_index": [1] * 20,
            "Eyes_recorded": ["LR"] * 20,
            "LX": sample_index,
            "LY": sample_index + 1,
            "LPupil": sample_index + 2,
            "RX": sample_index + 3,
            "RY": sample_index + 4,
            "RPupil": sample_index + 5,
        }
    )

    detector = remodnav_module.RemodnavDetection(tmp_path, samples)
    fixations, saccades = detector.detect_eye_movements()

    assert set(fixations["eye"].to_list()) == {"L", "R"}
    assert set(saccades["eye"].to_list()) == {"L", "R"}
    assert fixations["pupilAvg"].is_not_null().all()


def test_low_sample_rate_uses_nyquist_safe_lowpass(monkeypatch, tmp_path):
    monkeypatch.setattr(remodnav_module, "EyegazeClassifier", _FakeClassifier)
    samples = pl.DataFrame(
        {
            "tSample": [0.0, 200.0, 400.0],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 1.0, 2.0],
        }
    )

    detector = remodnav_module.RemodnavDetection(tmp_path, samples)
    detector.run_eye_movement_from_samples(5.0, config={"savgol_length": 0.0})

    assert _FakeClassifier.last_kwargs["lowpass_cutoff_freq"] == 2.0

    with pytest.raises(ValueError, match="Nyquist"):
        detector.run_eye_movement_from_samples(
            5.0,
            config={
                "savgol_length": 0.0,
                "lowpass_cutoff_freq": 2.5,
            },
        )
