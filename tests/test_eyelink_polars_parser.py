from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")

# Keep these focused parser tests independent from the package-level dataset
# orchestration and optional detector imports.
if "pyxations.bids_formatting" not in sys.modules:
    bids_stub = types.ModuleType("pyxations.bids_formatting")
    bids_stub.dataset_to_bids = lambda *args, **kwargs: None
    bids_stub.compute_derivatives_for_dataset = lambda *args, **kwargs: None
    sys.modules["pyxations.bids_formatting"] = bids_stub

from pyxations.formats.eyelink import parse as eyelink_parse
from pyxations.formats.eyelink.parse import (
    EyelinkParse,
    _apply_best_eye,
    _find_best_eye_polars,
    _keep_eye_polars,
    _parse_ascii_tables,
)


def _write_ascii(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "** DATE: synthetic",
                "!CAL",
                "GAZE_COORDS 0 0 1919 1079",
                "CAL VALIDATION LEFT ERROR 0.30 avg. 0.40 max",
                "!MODE RECORD CR 1000 2 0 LR",
                "RECCFG CR 500 2 0 LR RATE 500 TRACKING",
                "MSG 95 START trial-a",
                "100 10 20 3.1 30 40 3.2",
                "102 . . . 31 41 3.3",
                "EFIX L 100 120 20 11 21 3.0",
                "EFIX L 125 135 10 12 22 .",
                "EFIX R 101 121 20 31 41 3.2",
                "ESACC L 120 140 20 11 21 20 30 2.5 300",
                "ESACC R 121 141 20 31 41 40 50 2.7 320",
                "EBLINK L 150 160 10",
                "EBLINK R 151 161 10",
                "MSG 200 END trial-a",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _calibration_frame(lines: list[str], calib_indexes: list[int] | None = None) -> pl.DataFrame:
    indexes = calib_indexes or [1] * len(lines)
    return pl.DataFrame(
        {
            "line": lines,
            "Line_number": list(range(len(lines))),
            "Calib_index": indexes,
        }
    )


@pytest.mark.parametrize(
    ("left_error", "right_error", "expected"),
    [
        (0.20, 0.40, "L"),
        (0.50, 0.30, "R"),
        (0.25, 0.25, "R"),
    ],
)
def test_find_best_eye_polars_compares_validation_error(
    left_error: float, right_error: float, expected: str
):
    calibration = _calibration_frame(
        [
            f"CAL VALIDATION LEFT ERROR {left_error} avg.",
            f"CAL VALIDATION RIGHT ERROR {right_error} avg.",
        ]
    )

    assert _find_best_eye_polars(calibration) == expected


def test_find_best_eye_polars_preserves_missing_and_aborted_rules():
    assert _find_best_eye_polars(_calibration_frame(["GAZE_COORDS 0 0 1919 1079"])) == "M"
    assert (
        _find_best_eye_polars(
            _calibration_frame(
                [
                    "CAL VALIDATION LEFT ERROR 0.20 avg.",
                    "CAL VALIDATION R ABORTED",
                ]
            )
        )
        == "L"
    )
    assert _find_best_eye_polars(_calibration_frame(["CAL VALIDATION R ABORTED"])) == "R"


def test_find_best_eye_polars_uses_asc_line_order():
    calibration = pl.DataFrame(
        {
            "line": [
                "CAL VALIDATION RIGHT ERROR 0.10 avg.",
                "CAL VALIDATION LEFT ERROR 0.30 avg.",
            ],
            "Line_number": [20, 10],
            "Calib_index": [1, 1],
        }
    )

    assert _find_best_eye_polars(calibration) == "R"


def test_apply_best_eye_reuses_previous_eye_when_later_validation_missing(tmp_path: Path):
    calibrations = _calibration_frame(
        [
            "CAL VALIDATION LEFT ERROR 0.10 avg.",
            "CAL VALIDATION RIGHT ERROR 0.20 avg.",
            "GAZE_COORDS 0 0 1919 1079",
        ],
        [1, 1, 2],
    )
    samples = pl.DataFrame(
        {
            "tSample": [100.0, 200.0],
            "LX": [10.0, 11.0],
            "LY": [20.0, 21.0],
            "LPupil": [3.0, 3.1],
            "RX": [30.0, 31.0],
            "RY": [40.0, 41.0],
            "RPupil": [4.0, 4.1],
            "Line_number": [10, 20],
            "Eyes_recorded": ["LR", "LR"],
            "Rate_recorded": [500.0, 500.0],
            "Calib_index": [1, 2],
        }
    )
    empty_fixations = pl.DataFrame(schema={"eye": pl.String})
    empty_blinks = pl.DataFrame(schema={"eye": pl.String})
    empty_saccades = pl.DataFrame(schema={"eye": pl.String})

    selected_samples, selected_fixations, selected_blinks, selected_saccades = (
        _apply_best_eye(
            calibrations,
            samples,
            empty_fixations,
            empty_blinks,
            empty_saccades,
            session_folder_path=tmp_path / "sub-01" / "ses-01",
        )
    )

    assert selected_samples["X"].to_list() == [10.0, 11.0]
    assert selected_samples["Pupil"].to_list() == [3.0, 3.1]
    assert selected_fixations.is_empty()
    assert selected_blinks.is_empty()
    assert selected_saccades.is_empty()


def test_ascii_streaming_builds_typed_polars_tables(tmp_path: Path):
    ascii_path = _write_ascii(tmp_path / "recording.asc")

    (
        headers,
        calibrations,
        messages,
        samples,
        fixations,
        saccades,
        blinks,
        screen_resolution,
    ) = _parse_ascii_tables(ascii_path, msg_keywords=["START", "END"])

    assert all(
        isinstance(frame, pl.DataFrame)
        for frame in (
            headers,
            calibrations,
            messages,
            samples,
            fixations,
            saccades,
            blinks,
        )
    )
    assert screen_resolution == (1919, 1079)
    assert headers.tail(1)["line"].item() == "** SCREEN SIZE: 1919 1079"
    assert messages["timestamp"].to_list() == [95.0, 200.0]
    assert messages["message"].to_list() == ["START trial-a", "END trial-a"]

    assert samples["tSample"].to_list() == [100.0, 102.0]
    assert samples["Rate_recorded"].to_list() == [500.0, 500.0]
    assert samples["Eyes_recorded"].to_list() == ["LR", "LR"]
    assert samples["LPupil"][0] == pytest.approx(3.1)
    assert samples["RPupil"].to_list() == pytest.approx([3.2, 3.3])
    assert samples["LX"][1] is None

    assert fixations["eye"].to_list() == ["L", "L", "R"]
    assert fixations["pupilAvg"].to_list() == [3.0, None, 3.2]
    assert saccades["vPeak"].to_list() == pytest.approx([300.0, 320.0])
    assert blinks["duration"].to_list() == [10.0, 10.0]


def test_keep_eye_filters_polars_tables_and_preserves_pupil(tmp_path: Path):
    (
        _,
        _,
        _,
        samples,
        fixations,
        saccades,
        blinks,
        _,
    ) = _parse_ascii_tables(
        _write_ascii(tmp_path / "recording.asc"),
        msg_keywords=["START", "END"],
    )

    selected_samples, selected_fixations, selected_blinks, selected_saccades = (
        _keep_eye_polars("R", samples, fixations, blinks, saccades)
    )

    assert selected_samples.columns == [
        "tSample",
        "X",
        "Y",
        "Pupil",
        "Line_number",
        "Eyes_recorded",
        "Rate_recorded",
        "Calib_index",
    ]
    assert selected_samples["Pupil"].to_list() == pytest.approx([3.2, 3.3])
    assert selected_fixations["eye"].to_list() == ["R"]
    assert selected_blinks["eye"].to_list() == ["R"]
    assert selected_saccades["eye"].to_list() == ["R"]


def test_eyelink_vendor_events_remain_polars_through_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ascii_path = _write_ascii(tmp_path / "recording.asc")
    fake_edf = tmp_path / "recording.edf"
    fake_edf.write_bytes(b"")
    output = tmp_path / "derivatives"

    monkeypatch.setattr(
        eyelink_parse,
        "convert_edf_to_ascii",
        lambda edf_file_path, output_dir: ascii_path,
    )
    returned = EyelinkParse(output, "feather").parse(
        fake_edf,
        "eyelink",
        ["START", "END"],
        False,
        True,
        True,
        start_times={"search": [95]},
        end_times={"search": [200]},
        trial_labels={"search": ["trial-a"]},
    )

    assert isinstance(returned, pl.DataFrame)
    samples = pl.read_ipc(output / "samples.feather")
    messages = pl.read_ipc(output / "msg.feather")
    fixations = pl.read_ipc(output / "eyelink_events" / "fix.feather")
    saccades = pl.read_ipc(output / "eyelink_events" / "sacc.feather")
    blinks = pl.read_ipc(output / "eyelink_events" / "blink.feather")

    assert samples["phase"].to_list() == ["search", "search"]
    assert samples["trial_label"].to_list() == ["trial-a", "trial-a"]
    assert samples["LPupil"][0] == pytest.approx(3.1)
    assert messages["message"].to_list() == ["START trial-a", "END trial-a"]
    assert fixations.height == 3
    assert fixations["pupilAvg"].null_count() == 1
    assert set(saccades["dir"].to_list()) <= {
        "right",
        "left",
        "up",
        "down",
        "",
    }
    assert blinks.height == 2


def test_force_best_eye_uses_polars_scoring_and_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ascii_path = _write_ascii(tmp_path / "recording.asc")
    fake_edf = tmp_path / "recording.edf"
    fake_edf.write_bytes(b"")
    output = tmp_path / "derivatives-best-eye"

    monkeypatch.setattr(
        eyelink_parse,
        "convert_edf_to_ascii",
        lambda edf_file_path, output_dir: ascii_path,
    )
    returned = EyelinkParse(output, "feather").parse(
        fake_edf,
        "eyelink",
        ["START", "END"],
        True,
        True,
        True,
        start_times={"search": [95]},
        end_times={"search": [200]},
    )

    assert returned.columns[:4] == ["tSample", "X", "Y", "Pupil"]
    assert returned["Pupil"][0] == pytest.approx(3.1)
    assert pl.read_ipc(output / "eyelink_events" / "fix.feather")["eye"].to_list() == [
        "L",
        "L",
    ]
    assert pl.read_ipc(output / "eyelink_events" / "blink.feather")["eye"].to_list() == [
        "L"
    ]
    assert pl.read_ipc(output / "eyelink_events" / "sacc.feather")["eye"].to_list() == [
        "L"
    ]


class _FakeDetector:
    last_samples: pl.DataFrame | None = None

    def __init__(self, session_folder_path: Path, samples: pl.DataFrame):
        del session_folder_path
        type(self).last_samples = samples

    def detect_eye_movements(self):
        return (
            pl.DataFrame(
                {
                    "eye": ["L"],
                    "tStart": [100.0],
                    "tEnd": [120.0],
                    "duration": [20.0],
                    "xAvg": [11.0],
                    "yAvg": [21.0],
                    "pupilAvg": [3.0],
                    "Line_number": [8],
                    "Eyes_recorded": ["LR"],
                    "Rate_recorded": [500.0],
                    "Calib_index": [1],
                }
            ),
            pl.DataFrame(
                {
                    "eye": ["L"],
                    "tStart": [120.0],
                    "tEnd": [140.0],
                    "duration": [20.0],
                    "xStart": [11.0],
                    "yStart": [21.0],
                    "xEnd": [20.0],
                    "yEnd": [30.0],
                    "ampDeg": [2.5],
                    "vPeak": [300.0],
                    "Line_number": [10],
                    "Eyes_recorded": ["LR"],
                    "Rate_recorded": [500.0],
                    "Calib_index": [1],
                }
            ),
        )


def test_external_detector_receives_polars_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ascii_path = _write_ascii(tmp_path / "recording.asc")
    fake_edf = tmp_path / "recording.edf"
    fake_edf.write_bytes(b"")

    monkeypatch.setattr(
        eyelink_parse,
        "convert_edf_to_ascii",
        lambda edf_file_path, output_dir: ascii_path,
    )
    monkeypatch.setattr(
        eyelink_parse,
        "_detection_registry",
        lambda: {"fake": _FakeDetector},
    )

    EyelinkParse(tmp_path / "out", "feather").parse(
        fake_edf,
        "fake",
        ["START", "END"],
        False,
        True,
        True,
        start_times={"search": [95]},
        end_times={"search": [200]},
    )

    assert isinstance(_FakeDetector.last_samples, pl.DataFrame)
    np.testing.assert_allclose(
        _FakeDetector.last_samples["RPupil"].to_numpy(),
        [3.2, 3.3],
    )
