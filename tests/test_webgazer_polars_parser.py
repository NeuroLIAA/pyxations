from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from pyxations.formats.webgazer.parse import WebGazerParse, process_session


def _write_webgazer_csv(path: Path) -> Path:
    frame = pl.DataFrame(
        {
            "trial_index": [-1, 0, 1],
            "time_elapsed": [500.0, 1000.0, 1600.0],
            "rastoc-type": ["calibration-stimulus", "", ""],
            "condition": ["calibration", "left", "right"],
            "rt": [None, 300.0, 200.0],
            "webgazer_data": [
                None,
                json.dumps(
                    [
                        {"x": 0.0, "y": 10.0, "t": 0.0},
                        {"x": 10.0, "y": 20.0, "t": 100.0},
                        {"x": 30.0, "y": 40.0, "t": 300.0},
                    ]
                ),
                json.dumps(
                    [
                        {"x": 100.0, "y": 110.0, "t": 0.0},
                        {"x": 110.0, "y": 120.0, "t": 100.0},
                        {"x": 120.0, "y": 130.0, "t": 200.0},
                    ]
                ),
            ],
        }
    )
    frame.write_csv(path)
    return path


class _FakeRemodnav:
    calls: list[tuple[pl.DataFrame, dict]] = []

    def __init__(self, session_folder_path: Path, samples: pl.DataFrame):
        del session_folder_path
        self.samples = samples

    def detect_eye_movements(self, **config):
        type(self).calls.append((self.samples, config))
        return (
            pl.DataFrame(
                {
                    "tStart": [float(self.samples["tSample"][0])],
                    "tEnd": [float(self.samples["tSample"][-1])],
                    "pupilAvg": [float("nan")],
                }
            ),
            pl.DataFrame(
                {
                    "tStart": [float(self.samples["tSample"][0])],
                    "tEnd": [float(self.samples["tSample"][-1])],
                }
            ),
        )


class _FakeEngbert:
    calls: list[tuple[pl.DataFrame, dict]] = []

    def __init__(self, session_folder_path: Path, samples: pl.DataFrame):
        del session_folder_path
        self.samples = samples

    def detect_eye_movements(self, **config):
        type(self).calls.append((self.samples, config))
        return (
            pl.DataFrame(
                {
                    "tStart": [float(self.samples["tSample"][0])],
                    "tEnd": [float(self.samples["tSample"][-1])],
                }
            ),
            pl.DataFrame(
                {
                    "tStart": [float(self.samples["tSample"][0])],
                    "tEnd": [float(self.samples["tSample"][-1])],
                }
            ),
        )


@pytest.fixture
def detector_mapping(monkeypatch):
    import pyxations.bids_formatting as bids_formatting

    _FakeRemodnav.calls = []
    _FakeEngbert.calls = []
    monkeypatch.setattr(
        bids_formatting,
        "EYE_MOVEMENT_DETECTION_DICT",
        {"remodnav": _FakeRemodnav, "engbert": _FakeEngbert},
    )


def test_webgazer_parser_regularizes_each_trial_and_preserves_quality_metrics(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_webgazer_csv(tmp_path / "webgazer.csv")
    output = tmp_path / "derivatives"

    returned = WebGazerParse(output, "feather").parse(
        source,
        "remodnav",
        overwrite=True,
        screen_width=1024,
        screen_height=768,
        behavioral_columns=["condition", "rt"],
    )

    assert isinstance(returned, pl.DataFrame)
    assert len(_FakeRemodnav.calls) == 2
    first_config = _FakeRemodnav.calls[0][1]
    assert first_config["savgol_length"] == 0.0
    assert first_config["max_pso_dur"] == 0.1
    assert first_config["screen_width"] == 1024
    assert first_config["lowpass_cutoff_freq"] == pytest.approx(
        (1000.0 / 150.0) * 0.4
    )

    samples = pl.read_ipc(output / "samples.feather")
    calibration = pl.read_ipc(output / "calib.feather")
    fixations = pl.read_ipc(output / "remodnav_events" / "fix.feather")
    blinks = pl.read_ipc(output / "remodnav_events" / "blink.feather")

    # Trial 0 is irregular (100 ms, then 200 ms), so it is resampled at the
    # inferred median interval of 150 ms. Trial 1 stays at 100 ms intervals.
    np.testing.assert_allclose(
        samples["tSample"].to_numpy(),
        [0.0, 150.0, 300.0, 700.0, 800.0, 900.0],
    )
    np.testing.assert_allclose(
        samples["X"].to_numpy(),
        [0.0, 15.0, 30.0, 100.0, 110.0, 120.0],
    )
    np.testing.assert_allclose(
        samples["Rate_recorded"].to_numpy(),
        [1000.0 / 150.0] * 3 + [10.0] * 3,
    )
    assert samples["segment_index"].to_list() == [0, 0, 0, 1, 1, 1]
    assert samples["trial_index"].to_list() == [0, 0, 0, 1, 1, 1]
    assert samples["is_interpolated"].to_list() == [False, True, False, False, False, False]
    assert samples["condition"].to_list() == ["left"] * 3 + ["right"] * 3
    assert "Pupil" not in samples.columns

    assert calibration.height == 1
    assert calibration["trial_index"].to_list() == [-1]
    assert fixations.height == 2
    assert blinks.is_empty()
    assert (output / "events.tsv").exists()


def test_webgazer_strict_policy_rejects_irregular_timestamps(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_webgazer_csv(tmp_path / "webgazer.csv")

    with pytest.raises(ValueError, match="Irregular WebGazer sampling"):
        WebGazerParse(tmp_path / "out", "feather").parse(
            source,
            "remodnav",
            overwrite=True,
            sampling_policy="strict",
            max_interval_deviation=0.20,
        )


def test_webgazer_supports_configurable_csv_and_json_field_names(
    tmp_path: Path,
    detector_mapping,
):
    source = tmp_path / "custom.csv"
    pl.DataFrame(
        {
            "trial": [7],
            "elapsed_ms": [500.0],
            "gaze": [
                json.dumps(
                    [
                        {"gx": 1.0, "gy": 2.0, "timestamp": 0.0},
                        {"gx": 3.0, "gy": 4.0, "timestamp": 50.0},
                        {"gx": 5.0, "gy": 6.0, "timestamp": 100.0},
                    ]
                )
            ],
        }
    ).write_csv(source)

    WebGazerParse(tmp_path / "out", "feather").parse(
        source,
        "engbert",
        overwrite=True,
        webgazer_data_column="gaze",
        row_time_column="elapsed_ms",
        trial_index_column="trial",
        calibration_type_column=None,
        x_field="gx",
        y_field="gy",
        time_field="timestamp",
        target_sample_rate=20,
        screen_width=1280,
    )

    samples = pl.read_ipc(tmp_path / "out" / "samples.feather")
    assert samples["trial_index"].to_list() == [7, 7, 7]
    assert samples["Rate_recorded"].to_list() == [20.0, 20.0, 20.0]
    assert _FakeEngbert.calls[0][1] == {
        "sample_rate_fallback": 20.0,
        "screen_width_px": 1280,
    }


def test_webgazer_explicit_time_segmentation_is_polars_native(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_webgazer_csv(tmp_path / "webgazer.csv")
    output = tmp_path / "derivatives"

    WebGazerParse(output, "feather").parse(
        source,
        "remodnav",
        overwrite=True,
        start_times={"search": [0.0, 700.0]},
        end_times={"search": [301.0, 901.0]},
        trial_labels={"search": ["first", "second"]},
    )

    samples = pl.read_ipc(output / "samples.feather")
    assert samples["phase"].to_list() == ["search"] * 6
    assert samples["trial_number"].to_list() == [0, 0, 0, 1, 1, 1]
    assert samples["trial_label"].to_list() == ["first"] * 3 + ["second"] * 3


def test_webgazer_rejects_message_segmentation_without_standard_messages(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_webgazer_csv(tmp_path / "webgazer.csv")

    with pytest.raises(ValueError, match="message-based trial segmentation"):
        WebGazerParse(tmp_path / "out", "feather").parse(
            source,
            "remodnav",
            overwrite=True,
            start_msgs={"search": ["START"]},
            end_msgs={"search": ["END"]},
        )


def test_webgazer_reports_invalid_json_with_row_number(
    tmp_path: Path,
    detector_mapping,
):
    source = tmp_path / "invalid.csv"
    pl.DataFrame(
        {
            "trial_index": [0],
            "time_elapsed": [1000.0],
            "webgazer_data": ["[{not valid json]"],
        }
    ).write_csv(source)

    with pytest.raises(ValueError, match="Invalid WebGazer JSON in CSV row 0"):
        WebGazerParse(tmp_path / "out", "feather").parse(
            source,
            "remodnav",
            overwrite=True,
            calibration_type_column=None,
        )


def test_process_session_requires_exactly_one_webgazer_csv(
    tmp_path: Path,
    detector_mapping,
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No jsPsych WebGazer CSV"):
        process_session(
            input_dir,
            "remodnav",
            tmp_path / "out-empty",
            True,
            "feather",
        )

    _write_webgazer_csv(input_dir / "one.csv")
    _write_webgazer_csv(input_dir / "two.csv")
    process_session(
        input_dir,
        "remodnav",
        tmp_path / "out-multiple",
        True,
        "feather",
    )
    assert not (tmp_path / "out-multiple" / "samples.feather").exists()
