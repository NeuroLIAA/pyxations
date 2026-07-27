from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyxations.formats.gazepoint.parse import GazePointParse


CSV_TEXT = """,time,TIME,TIMETICK,BPOGX,BPOGY,LPD,BKDUR
0,0.000,0.000,1,0.10,0.20,3.0,0.000
1,0.016,0.016,2,0.20,0.30,3.2,0.000
2,0.032,0.032,3,0.30,0.40,3.4,0.010
"""


def _write_csv(path: Path) -> Path:
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


class _FakeRemodnav:
    last_samples: pl.DataFrame | None = None
    last_config: dict | None = None
    last_sample_rate: float | None = None

    def __init__(self, session_folder_path: Path, samples: pl.DataFrame):
        del session_folder_path
        type(self).last_samples = samples

    def run_eye_movement_from_samples(
        self,
        sample_rate: float,
        *,
        config: dict,
        eye: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        assert eye == "Best"
        type(self).last_config = config
        type(self).last_sample_rate = sample_rate
        return (
            pl.DataFrame(
                {
                    "tStart": [5.0],
                    "tEnd": [15.0],
                    "pupilAvg": [float(np.nanmean(config["pupil_data"]))],
                }
            ),
            pl.DataFrame({"tStart": [21.0], "tEnd": [30.0]}),
        )


class _FakeEngbert:
    last_samples: pl.DataFrame | None = None
    last_config: dict | None = None

    def __init__(self, session_folder_path: Path, samples: pl.DataFrame):
        del session_folder_path
        type(self).last_samples = samples

    def detect_eye_movements(self, **config):
        type(self).last_config = config
        return (
            pl.DataFrame({"tStart": [5.0], "tEnd": [15.0]}),
            pl.DataFrame({"tStart": [21.0], "tEnd": [30.0]}),
        )


@pytest.fixture
def detector_mapping(monkeypatch):
    import pyxations.bids_formatting as bids_formatting

    monkeypatch.setattr(
        bids_formatting,
        "EYE_MOVEMENT_DETECTION_DICT",
        {"remodnav": _FakeRemodnav, "engbert": _FakeEngbert},
    )


def test_gazepoint_parser_stays_polars_and_preserves_pupil(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_csv(tmp_path / "gaze.csv")
    output = tmp_path / "derivatives"

    parser = GazePointParse(output, "feather")
    parser.parse(
        source,
        "remodnav",
        overwrite=True,
        sample_rate=60,
        start_times={"search": [0.0]},
        end_times={"search": [20.0]},
        trial_labels={"search": ["first"]},
    )

    assert isinstance(_FakeRemodnav.last_samples, pl.DataFrame)
    assert _FakeRemodnav.last_sample_rate == 60
    np.testing.assert_allclose(
        _FakeRemodnav.last_config["pupil_data"],
        np.array([3.0, 3.2, 3.4]),
    )

    samples = pl.read_ipc(output / "samples.feather")
    fixations = pl.read_ipc(output / "remodnav_events" / "fix.feather")
    saccades = pl.read_ipc(output / "remodnav_events" / "sacc.feather")
    blinks = pl.read_ipc(output / "remodnav_events" / "blink.feather")

    assert "Pupil" in samples.columns
    assert samples["Pupil"].to_list() == [3.0, 3.2, 3.4]
    assert samples["tSample"].to_list() == pytest.approx([0.0, 16.0, 32.0])
    assert samples["BKDUR"].to_list() == pytest.approx([0.0, 0.0, 10.0])
    assert samples["phase"].to_list() == ["search", "search", ""]
    assert samples["trial_number"].to_list() == [0, 0, -1]
    assert samples["trial_label"].to_list() == ["first", "first", ""]

    assert fixations["phase"].to_list() == ["search"]
    assert fixations["pupilAvg"].to_list() == pytest.approx([3.2])
    assert saccades["phase"].to_list() == [""]
    assert blinks["tStart"].to_list() == pytest.approx([22.0])
    assert blinks["tEnd"].to_list() == pytest.approx([32.0])
    assert blinks["duration"].to_list() == pytest.approx([10.0])
    assert blinks["phase"].to_list() == [""]


def test_gazepoint_parser_uses_engbert_polars_path(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_csv(tmp_path / "gaze.csv")
    output = tmp_path / "derivatives"

    parser = GazePointParse(output, "feather")
    parser.parse(
        source,
        "engbert",
        overwrite=True,
        sample_rate=120,
        detector_config={"vfac": 6.0},
    )

    assert isinstance(_FakeEngbert.last_samples, pl.DataFrame)
    assert _FakeEngbert.last_config == {
        "vfac": 6.0,
        "sample_rate_fallback": 120.0,
    }
    assert (output / "samples.feather").exists()
    assert (output / "engbert_events" / "fix.feather").exists()
    assert (output / "engbert_events" / "sacc.feather").exists()
    assert (output / "engbert_events" / "blink.feather").exists()


def test_gazepoint_rejects_message_segmentation_without_messages(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_csv(tmp_path / "gaze.csv")
    parser = GazePointParse(tmp_path / "derivatives", "feather")

    with pytest.raises(ValueError, match="message-based trial segmentation"):
        parser.parse(
            source,
            "remodnav",
            overwrite=True,
            start_msgs={"search": ["START"]},
            end_msgs={"search": ["END"]},
        )
