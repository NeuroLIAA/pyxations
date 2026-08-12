from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from pyxations import bids


def _recording(**updates):
    values = {
        "samples": pl.DataFrame(
            {
                "timestamp": [2.0, 1.0, 1.0],
                "x_coordinate": [np.inf, 10.0, 20.0],
                "y_coordinate": [2.0, 3.0, 4.0],
                "pupil_size": [-1.0, 5.0, 6.0],
            }
        ),
        "recorded_eye": "left",
        "sampling_frequency": 100.0,
        "timestamp_unit": "ms",
        "coordinate_unit": "pixel",
        "coordinate_description": "screen",
    }
    values.update(updates)
    return bids.EyeRecording(**values)


def test_eye_recording_normalization_and_validation():
    normalized = _recording().normalized()

    assert normalized.samples["timestamp"].to_list() == [1.0, 2.0]
    assert normalized.samples["x_coordinate"].to_list() == [10.0, None]
    assert normalized.samples["pupil_size"].to_list() == [5.0, None]

    with pytest.raises(ValueError, match="missing required"):
        _recording(samples=pl.DataFrame({"timestamp": [1.0]})).normalized()
    with pytest.raises(ValueError, match="RecordedEye"):
        _recording(recorded_eye="middle").normalized()
    with pytest.raises(ValueError, match="No samples"):
        _recording(
            samples=pl.DataFrame(
                {
                    "timestamp": [None],
                    "x_coordinate": [1.0],
                    "y_coordinate": [1.0],
                }
            )
        ).normalized()
    with pytest.raises(ValueError, match="SamplingFrequency"):
        _recording(sampling_frequency=0).normalized()
    with pytest.raises(ValueError, match="distinct timestamps"):
        bids._sampling_frequency([1.0, 1.0], units_per_second=1000)


def test_webgazer_reader_reports_malformed_and_empty_payloads(tmp_path):
    path = tmp_path / "webgazer.csv"
    with pytest.raises(ValueError, match="webgazer_data"):
        bids._read_webgazer(path, source=pl.DataFrame({"other": [1]}))
    with pytest.raises(ValueError, match="Invalid WebGazer JSON"):
        bids._read_webgazer(
            path,
            source=pl.DataFrame(
                {"webgazer_data": ["{"], "time_elapsed": [0], "trial_index": [0]}
            ),
        )
    with pytest.raises(ValueError, match="No WebGazer samples"):
        bids._read_webgazer(
            path,
            source=pl.DataFrame(
                {
                    "webgazer_data": ['[{"x": 1}]', None],
                    "time_elapsed": [0, 0],
                    "trial_index": [0, 1],
                }
            ),
        )


def test_edf_converter_boundary_without_requiring_edf2asc(tmp_path, monkeypatch):
    asc = tmp_path / "recording.asc"
    asc.write_text("sample", encoding="utf-8")
    assert bids._eyelink_ascii_path(asc, tmp_path) == asc

    edf = tmp_path / "recording.edf"
    edf.write_bytes(b"not decoded by this unit test")
    monkeypatch.setattr(bids.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="edf2asc"):
        bids._eyelink_ascii_path(edf, tmp_path)

    monkeypatch.setattr(bids.shutil, "which", lambda name: "edf2asc")

    def successful_run(command, **kwargs):
        Path(command[-1]).write_text("converted ASC", encoding="utf-8")
        return SimpleNamespace(stdout="warning", stderr="", returncode=1)

    monkeypatch.setattr(bids.subprocess, "run", successful_run)
    converted = bids._eyelink_ascii_path(edf, tmp_path)
    assert converted.read_text(encoding="utf-8") == "converted ASC"

    converted.unlink()
    monkeypatch.setattr(
        bids.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="converter output", stderr="converter error", returncode=1
        ),
    )
    with pytest.raises(RuntimeError, match="converter error"):
        bids._eyelink_ascii_path(edf, tmp_path)


def test_eyelink_reader_handles_right_eye_and_malformed_lines(tmp_path):
    path = tmp_path / "right.asc"
    path.write_text(
        """** VERSION: TEST
SAMPLES GAZE RIGHT RATE 500 TRACKING CR
MSG not-a-time ignored
MSG 100 start
EFIX R bad 120 20 1 2 3
ESACC R 100 120 20 1 2 3 4 5 6
EBLINK R 121 125 4
!CAL VALIDATION HV9 RIGHT GOOD ERROR 0.5 avg.
GAZE_COORDS 0 0 800 600
!MODE RECORD CR 500 2 1 R
100 10 20 3
102 11 21 4
""",
        encoding="utf-8",
    )

    bundle = bids._read_eyelink_bundle(path)

    assert [item.recorded_eye for item in bundle.recordings] == ["right"]
    assert bundle.recordings[0].sampling_frequency == 500
    assert set(bundle.events["trial_type"]) == {"message", "saccade", "blink"}
    assert bundle.metadata == {"ScreenWidth": 800, "ScreenHeight": 600}
    assert not bundle.calibration.is_empty()

    empty = tmp_path / "empty.asc"
    empty.write_text("MSG invalid ignored\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No EyeLink samples"):
        bids._read_eyelink_bundle(empty)


def test_primary_recording_and_writer_input_validation(tmp_path):
    malformed = tmp_path / "broken.csv"
    malformed.write_bytes(b"\xff\xfe")
    assert not bids._is_primary_recording(malformed, "gaze")
    assert not bids._is_primary_recording(tmp_path / "notes.txt", "gaze")

    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="Unknown eye-tracking"):
        bids.write_bids_dataset(tmp_path, source, "dataset", format_name="unknown")
    with pytest.raises(ValueError, match="session_substrings"):
        bids.write_bids_dataset(
            tmp_path, source, "dataset", format_name="gaze", session_substrings=0
        )
    with pytest.raises(FileNotFoundError, match="Input directory"):
        bids.write_bids_dataset(
            tmp_path, tmp_path / "missing", "dataset", format_name="gaze"
        )
    with pytest.raises(ValueError, match="No gaze recordings"):
        bids.write_bids_dataset(tmp_path, source, "dataset", format_name="gaze")

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file").write_text("content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        bids.write_bids_dataset(
            tmp_path,
            source,
            "occupied",
            format_name="gaze",
        )


def test_validator_command_and_success_paths(tmp_path, monkeypatch):
    def executable_only(name):
        return "C:/bin/bids-validator" if name == "bids-validator" else None

    monkeypatch.setattr(bids.shutil, "which", executable_only)
    assert bids.validator_command() == ["C:/bin/bids-validator"]

    monkeypatch.setattr(
        bids.shutil,
        "which",
        lambda name: "C:/bin/deno" if name == "deno" else None,
    )
    assert bids.validator_command()[:4] == [
        "C:/bin/deno",
        "run",
        "-ERWN",
        "jsr:@bids/validator@3.0.1",
    ]

    monkeypatch.setattr(bids.shutil, "which", lambda name: None)
    assert bids.validator_command() is None
    with pytest.raises(FileNotFoundError, match="BIDS dataset"):
        bids.validate_bids_dataset(tmp_path / "missing", command=["validator"])
    with pytest.raises(RuntimeError, match="unavailable"):
        bids.validate_bids_dataset(tmp_path)

    completed = subprocess.CompletedProcess([], 0, stdout="{}", stderr="")
    monkeypatch.setattr(bids.subprocess, "run", lambda *args, **kwargs: completed)
    assert bids.validate_bids_dataset(tmp_path, command=["validator"]) is completed


def _webgazer_export(folder: Path) -> Path:
    """A jsPsych export whose gaze trials start late and skip an index."""
    gaze = json.dumps([{"x": 1.0, "y": 2.0, "t": 0}, {"x": 3.0, "y": 4.0, "t": 17}])
    path = folder / "s01_A_task-anti.csv"
    pl.DataFrame(
        {
            # 0 and 1 are instruction screens, 32 is skipped by jsPsych.
            "trial_index": [0, 1, 29, 30, 31, 33],
            "time_elapsed": [10, 20, 1_000, 2_000, 3_000, 4_000],
            "webgazer_data": [None, None, gaze, gaze, gaze, gaze],
        }
    ).write_csv(path)
    return path


def test_webgazer_trial_numbering_is_sequential_and_keeps_the_source_index(tmp_path):
    source = pl.read_csv(_webgazer_export(tmp_path), infer_schema_length=None)

    numbering = bids.webgazer_trial_numbering(source)

    # Instruction screens carry no gaze, so they are not trials at all.
    assert numbering == {29: 0, 30: 1, 31: 2, 33: 3}

    recording = bids._read_webgazer(tmp_path / "s01_A_task-anti.csv", source=source)[0]
    samples = recording.samples
    assert sorted(samples.get_column("trial_number").unique().to_list()) == [0, 1, 2, 3]
    assert sorted(samples.get_column("source_trial_index").unique().to_list()) == [
        29,
        30,
        31,
        33,
    ]


def test_webgazer_samples_and_events_share_one_numbering(tmp_path):
    """A mismatch here would leave every trial without its behavioral row."""
    source_path = _webgazer_export(tmp_path)

    events = bids._prepare_task_events(
        [source_path],
        source_root=tmp_path,
        primary_source=source_path,
        format_name="webgazer",
    )
    samples = bids._read_webgazer(source_path)[0].samples

    sample_trials = set(samples.get_column("trial_number").to_list())
    event_trials = set(
        events.filter(pl.col("trial_number").is_not_null())
        .get_column("trial_number")
        .to_list()
    )
    assert sample_trials == event_trials == {0, 1, 2, 3}

    # The instruction screens survive as events but are not numbered as trials.
    assert events.get_column("trial_number").null_count() == 2
    assert events.get_column("source_trial_index").to_list() == [0, 1, 29, 30, 31, 33]


def test_unsupported_gaze_exports_name_the_format_that_is_expected(tmp_path):
    """A Titta or Gorilla user should learn why their file was rejected."""
    tobii_path = tmp_path / "s01_A_task-look.txt"
    pl.DataFrame(
        {"Recording timestamp": [0, 1], "something_else": [1.0, 2.0]}
    ).write_csv(tobii_path, separator="\t")
    with pytest.raises(ValueError, match="Titta"):
        bids._read_tobii(tobii_path)

    webgazer_path = tmp_path / "s01_A_task-look.csv"
    pl.DataFrame({"trial_index": [0], "gaze": ["[]"]}).write_csv(webgazer_path)
    with pytest.raises(ValueError, match="Gorilla"):
        bids._read_webgazer(webgazer_path)
