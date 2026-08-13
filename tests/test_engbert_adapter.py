"""Focused tests for the Polars-native Engbert adapter path."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

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


def test_engbert_numeric_helpers_and_edge_cases():
    frame = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="Missing required"):
        engbert_module._column_to_numpy(frame, "missing")
    np.testing.assert_array_equal(
        engbert_module._column_to_numpy(
            frame, "missing", required=False, default=3, dtype=float
        ),
        [3.0, 3.0],
    )
    with pytest.raises(TypeError, match="Polars DataFrame"):
        engbert_module._make_frame({}, [], ["x"])

    values = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(engbert_module._smooth_1d(values, 0), values)
    np.testing.assert_allclose(
        engbert_module._smooth_1d(values, 1),
        [4 / 3, 2, 8 / 3],
    )
    assert engbert_module._smooth_1d(values, 2).shape == values.shape
    np.testing.assert_array_equal(engbert_module._smooth_1d(values, 99), values)

    with pytest.raises(ValueError, match="shape"):
        engbert_module.vecvel(np.array([1.0, 2.0]), 100)
    assert np.isnan(engbert_module.vecvel(np.array([[1.0, 2.0]]), 100)).all()
    with pytest.raises(ValueError, match="Sampling rate"):
        engbert_module.vecvel(np.ones((2, 2)), 0)

    assert math.isnan(engbert_module._robust_std([np.nan]))
    assert engbert_module._find_runs([]).shape == (0, 2)
    np.testing.assert_array_equal(
        engbert_module._find_runs([True, True, False, True]),
        [[0, 1], [3, 3]],
    )
    assert math.isnan(engbert_module._nanmean_or_nan([np.nan]))
    assert engbert_module._fallback_sigma(np.array([np.nan])) == 1.0
    assert engbert_module._fallback_sigma(np.array([0.0])) == 1.0


def test_microsaccade_plugin_filters_and_reports_events():
    positions = np.column_stack([np.arange(6, dtype=float), np.zeros(6)])
    velocities = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [0.0, 0.0], [np.nan, 0.0], [0, 0]]
    )

    events = engbert_module.microsacc_plugin(
        positions,
        velocities,
        vfac=2,
        mindur_samples=2,
        sdx=1,
        sdy=1,
    )
    assert events.shape == (1, 14)
    assert events[0, :3].tolist() == [1.0, 2.0, 2.0]

    assert engbert_module.microsacc_plugin(
        positions,
        velocities,
        vfac=2,
        mindur_samples=3,
        sdx=1,
        sdy=1,
    ).shape == (0, 14)

    nonfinite_velocity = np.full((6, 2), np.nan)
    assert engbert_module.microsacc_plugin(
        positions,
        nonfinite_velocity,
        vfac=2,
        mindur_samples=1,
        sdx=1,
        sdy=1,
    ).shape == (0, 14)


def test_engbert_chunking_and_fixation_helpers():
    with pytest.raises(ValueError, match="positive"):
        engbert_module._compute_px2deg(0, 60, 1920)

    filled = engbert_module._forward_backward_fill(np.array([np.nan, 2.0, np.nan, 4.0]))
    np.testing.assert_array_equal(filled, [2.0, 2.0, 2.0, 4.0])
    assert np.isnan(
        engbert_module._forward_backward_fill(np.array([np.nan, np.nan]))
    ).all()

    empty_chunks, empty_rates = engbert_module._split_into_chunks(
        np.array([]), None, 100
    )
    assert empty_chunks.size == empty_rates.size == 0
    with pytest.raises(ValueError, match="finite"):
        engbert_module._split_into_chunks(np.array([0.0, np.nan]), None, 100)
    with pytest.raises(ValueError, match="monotonically"):
        engbert_module._split_into_chunks(np.array([2.0, 1.0]), None, 100)
    with pytest.raises(ValueError, match="fallback"):
        engbert_module._split_into_chunks(np.array([0.0, 1.0]), None)
    with pytest.raises(ValueError, match="same number"):
        engbert_module._split_into_chunks(np.array([0.0, 1.0]), np.array([100.0]))
    with pytest.raises(ValueError, match="greater than zero"):
        engbert_module._split_into_chunks(np.array([0.0, 1.0]), np.array([0.0, 0.0]))

    chunks, rates = engbert_module._split_into_chunks(
        np.array([0.0, 10.0, 40.0, 50.0]),
        np.array([100.0, 100.0, 50.0, 50.0]),
    )
    np.testing.assert_array_equal(chunks, [0, 0, 1, 1])
    np.testing.assert_array_equal(rates, [100.0, 100.0, 50.0, 50.0])

    records = []
    kwargs = {
        "coordinates": np.array([[np.nan, np.nan], [1.0, 2.0]]),
        "pupil_values": None,
        "chunk_start_ms": 0,
        "sample_rate": 100,
        "eye_label": "U",
        "calibration": 1,
        "eyes_recorded": "M",
        "chunk_id": 0,
    }
    engbert_module._append_fixation_record(records, 1, 0, **kwargs)
    engbert_module._append_fixation_record(records, 0, 0, **kwargs)
    assert records == []
    engbert_module._append_fixation_record(records, 0, 1, **kwargs)
    assert records[0]["xAvg"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vfac": 0}, "vfac"),
        ({"mindur_ms": -1}, "mindur_ms"),
        ({"smoothlevel": -1}, "smoothlevel"),
        ({"degperpixel": 0}, "degperpixel"),
    ],
)
def test_engbert_detector_validates_configuration(tmp_path, kwargs, message):
    detector = engbert_module.EngbertDetection(
        tmp_path,
        pl.DataFrame(
            {
                "tSample": [0.0, 10.0],
                "Rate_recorded": [100.0, 100.0],
                "X": [1.0, 2.0],
                "Y": [1.0, 2.0],
            }
        ),
    )
    with pytest.raises(ValueError, match=message):
        detector.detect_eye_movements(**kwargs)


def test_engbert_detector_empty_missing_and_local_threshold_paths(tmp_path):
    empty = engbert_module.EngbertDetection(
        tmp_path,
        pl.DataFrame(
            schema={
                "tSample": pl.Float64,
                "X": pl.Float64,
                "Y": pl.Float64,
            }
        ),
    )
    fixations, saccades = empty.detect_eye_movements(sample_rate_fallback=100)
    assert fixations.is_empty() and saccades.is_empty()

    with pytest.raises(ValueError, match="tSample"):
        engbert_module.EngbertDetection(
            tmp_path, pl.DataFrame({"X": [1.0], "Y": [1.0]})
        ).detect_eye_movements(sample_rate_fallback=100)

    with pytest.raises(ValueError, match="No supported gaze"):
        engbert_module.EngbertDetection(
            tmp_path,
            pl.DataFrame({"tSample": [0.0, 10.0], "Rate_recorded": [100.0, 100.0]}),
        ).detect_eye_movements()

    samples = pl.DataFrame(
        {
            "tSample": [0.0, 10.0, 40.0, 60.0],
            "Rate_recorded": [100.0, 100.0, 50.0, 50.0],
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    fixations, saccades = engbert_module.EngbertDetection(
        tmp_path, samples
    ).detect_eye_movements(globalthresh=False, vfac=1_000)
    assert fixations.get_column("chunk").to_list() == [0, 1]
    assert saccades.is_empty()
