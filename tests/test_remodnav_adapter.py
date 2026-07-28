"""Focused tests for the Polars-native REMoDNaV adapter path."""

from __future__ import annotations

import math
from collections import namedtuple

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
    fixations, saccades = detector.run_eye_movement_from_samples(100, config=config)

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


def test_remodnav_helpers_accept_supported_event_containers():
    values = (0.0, 0.1, "FIXA", 1, 2, 1.5, 2.5, 0.2, 3, 2, 2.5)
    Event = namedtuple("Event", remodnav_module._EVENT_COLUMNS)
    structured = np.array(
        [values],
        dtype=[
            (name, "S8" if name == "label" else "f8")
            for name in remodnav_module._EVENT_COLUMNS
        ],
    )[0]

    records = remodnav_module._normalise_remodnav_events(
        [
            dict(zip(remodnav_module._EVENT_COLUMNS, values)),
            Event(*values),
            structured,
            values,
        ]
    )
    assert len(records) == 4
    assert records[2]["label"] == "FIXA"

    with pytest.raises(ValueError, match="expected"):
        remodnav_module._normalise_remodnav_events([(1, 2)])
    with pytest.raises(ValueError, match="missing fields"):
        remodnav_module._normalise_remodnav_events([{"label": "FIXA"}])
    with pytest.raises(ValueError, match="Missing required"):
        remodnav_module._column_to_numpy(pl.DataFrame({"x": [1]}), "missing")
    with pytest.raises(TypeError, match="Polars DataFrame"):
        remodnav_module._make_frame({}, [], ["x"])
    assert math.isnan(remodnav_module._nanmean_or_nan(np.array([np.nan])))


def test_remodnav_constant_metadata_validation():
    with pytest.raises(ValueError, match="empty chunk"):
        remodnav_module._validate_constant(np.array([]), "metadata")
    assert math.isnan(
        remodnav_module._validate_constant(np.array([np.nan, np.nan]), "metadata")
    )
    assert remodnav_module._validate_constant(
        np.array([None, None], dtype=object), "metadata"
    ) is None
    with pytest.raises(ValueError, match="constant"):
        remodnav_module._validate_constant(np.array([1, 2]), "metadata")


def test_remodnav_detector_validation_and_empty_paths(tmp_path):
    empty_samples = pl.DataFrame(
        schema={"tSample": pl.Float64, "Rate_recorded": pl.Float64}
    )
    detector = remodnav_module.RemodnavDetection(tmp_path, empty_samples)
    fixations, saccades = detector.detect_eye_movements()
    assert fixations.is_empty() and saccades.is_empty()

    invalid_rate = pl.DataFrame(
        {"tSample": [0.0], "Rate_recorded": [0.0], "X": [1.0], "Y": [1.0]}
    )
    with pytest.raises(ValueError, match="Rate_recorded"):
        remodnav_module.RemodnavDetection(
            tmp_path, invalid_rate
        ).detect_eye_movements()

    samples = pl.DataFrame({"tSample": [0.0], "X": [1.0], "Y": [1.0]})
    detector = remodnav_module.RemodnavDetection(tmp_path, samples)
    with pytest.raises(ValueError, match="sample_rate"):
        detector.run_eye_movement_from_samples(0)
    with pytest.raises(ValueError, match="sample_rate"):
        detector.run_eye_movement([1], [1], 0)
    with pytest.raises(ValueError, match="Screen size"):
        detector.run_eye_movement([1], [1], 100, screen_size=0)
    with pytest.raises(ValueError, match="equal lengths"):
        detector.run_eye_movement([1], [1, 2], 100)
    with pytest.raises(ValueError, match="times"):
        detector.run_eye_movement([1], [1], 100, times=[0, 1])
    with pytest.raises(ValueError, match="pupil_data"):
        detector.run_eye_movement([1], [1], 100, pupil_data=[1, 2])


def test_remodnav_chunk_validation_paths(tmp_path):
    samples = pl.DataFrame(
        {
            "tSample": [0.0, 10.0],
            "Rate_recorded": [100.0, 100.0],
            "Calib_index": [1, 1],
            "Eyes_recorded": ["M", "M"],
            "other": [1.0, 2.0],
        }
    )
    detector = remodnav_module.RemodnavDetection(tmp_path, samples)
    fixations, saccades = detector.detect_on_chunk(np.array([], dtype=int))
    assert fixations.is_empty() and saccades.is_empty()
    with pytest.raises(ValueError, match="Samples must contain"):
        detector.detect_on_chunk(np.array([0, 1]))

    detector.samples = samples.with_columns(
        pl.lit(float("nan")).alias("X"),
        pl.lit(float("nan")).alias("Y"),
    )
    fixations, saccades = detector.detect_on_chunk(np.array([0, 1]))
    assert fixations.is_empty() and saccades.is_empty()
