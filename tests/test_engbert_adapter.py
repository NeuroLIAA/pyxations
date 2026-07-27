"""Focused tests for the Polars-native Engbert adapter path."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from pyxations.methods.eyemovement import engbert as engbert_module


def test_constant_generic_stream_returns_one_fixation_with_pupil(tmp_path):
    samples = pl.DataFrame(
        {
            "tSample": np.arange(6, dtype=float) * 10.0,
            "Rate_recorded": [100.0] * 6,
            "Calib_index": [2] * 6,
            "Eyes_recorded": ["M"] * 6,
            "X": [100.0] * 6,
            "Y": [200.0] * 6,
            "Pupil": np.arange(1, 7, dtype=float),
        }
    )

    detector = engbert_module.EngbertDetection(tmp_path, samples)
    fixations, saccades = detector.detect_eye_movements()

    assert isinstance(fixations, pl.DataFrame)
    assert isinstance(saccades, pl.DataFrame)
    assert fixations.height == 1
    assert saccades.is_empty()
    assert fixations["eye"][0] == "U"
    assert fixations["tStart"][0] == 0.0
    assert fixations["tEnd"][0] == 50.0
    assert fixations["duration"][0] == 60.0
    assert fixations["pupilAvg"][0] == 3.5
    assert fixations["Calib_index"][0] == 2


def test_both_eyes_are_processed_and_missing_pupil_stays_missing(tmp_path):
    sample_index = np.arange(6, dtype=float)
    samples = pl.DataFrame(
        {
            "tSample": sample_index * 10.0,
            "Rate_recorded": [100.0] * 6,
            "Calib_index": [1] * 6,
            "Eyes_recorded": ["LR"] * 6,
            "LX": sample_index,
            "LY": sample_index + 1.0,
            "RX": sample_index + 2.0,
            "RY": sample_index + 3.0,
        }
    )

    detector = engbert_module.EngbertDetection(tmp_path, samples)
    fixations, saccades = detector.detect_eye_movements(vfac=1_000.0)

    assert set(fixations["eye"].to_list()) == {"L", "R"}
    assert saccades.is_empty()
    assert all(math.isnan(value) for value in fixations["pupilAvg"].to_list())


def test_saccade_rows_and_inter_saccadic_fixations_preserve_schema(
    monkeypatch, tmp_path
):
    def fake_microsacc_plugin(*args, **kwargs):
        return np.array(
            [[1, 2, 2, 15, 20, 5, np.pi / 2, 4, np.pi / 2, np.nan, 1, 2, 3, 4]],
            dtype=float,
        )

    monkeypatch.setattr(engbert_module, "microsacc_plugin", fake_microsacc_plugin)
    samples = pl.DataFrame(
        {
            "tSample": np.arange(6, dtype=float) * 10.0,
            "Rate_recorded": [100.0] * 6,
            "X": np.arange(6, dtype=float),
            "Y": np.arange(6, dtype=float) + 1.0,
        }
    )

    detector = engbert_module.EngbertDetection(tmp_path, samples)
    fixations, saccades = detector.detect_eye_movements(degperpixel=0.5)

    assert saccades.columns == list(engbert_module._SACCADE_COLUMNS)
    assert fixations.columns == list(engbert_module._FIXATION_COLUMNS)
    assert saccades.height == 1
    assert fixations.height == 2
    assert saccades["tStart"][0] == 10.0
    assert saccades["tEnd"][0] == 20.0
    assert saccades["duration"][0] == 20.0
    assert saccades["ampDeg"][0] == 2.0
    assert saccades["vPeak"][0] == 10.0
    assert saccades["thetaDeg"][0] == 90.0
    assert fixations["tStart"].to_list() == [0.0, 30.0]
