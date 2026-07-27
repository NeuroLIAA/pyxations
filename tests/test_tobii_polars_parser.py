from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")

# Avoid importing the full package-level orchestration in these focused parser tests.
if "pyxations.bids_formatting" not in sys.modules:
    sys.modules["pyxations.bids_formatting"] = types.ModuleType(
        "pyxations.bids_formatting"
    )

from pyxations.formats.tobii.parse import TobiiParse, process_session


TSV_COLUMNS = [
    "Recording timestamp",
    "Eyetracker timestamp",
    "Gaze2d_Left.x",
    "Gaze2d_Left.y",
    "Gaze2d_Right.x",
    "Gaze2d_Right.y",
    "PupilDiam_Left",
    "PupilDiam_Right",
    "Validity_Left",
    "Validity_Right",
    "Event value",
    "Event message",
]


def _write_tobii_export(path: Path) -> Path:
    rows = [
        [1_000_000, 8_000_000, 100, 200, 110, 210, 3.1, 3.2, 0, 0, 0, 0],
        [1_003_333, 8_003_333, 101, 201, 111, 211, 3.2, 3.3, 0, 0, 0, 0],
        [1_004_000, None, None, None, None, None, None, None, None, None, 1, "START"],
        [1_006_666, 8_006_666, 102, 202, 112, 212, 3.3, 3.4, 0, 0, 0, 0],
        [1_010_000, None, None, None, None, None, None, None, None, None, 2, "END"],
        [1_010_000, 8_010_000, 103, 203, 113, 213, 3.4, 3.5, 0, 0, 0, 0],
    ]
    frame = pl.DataFrame(rows, schema=TSV_COLUMNS, orient="row")
    frame.write_csv(path, separator="\t", null_value="")
    return path


class _FakeRemodnav:
    last_samples: pl.DataFrame | None = None
    last_config: dict | None = None

    def __init__(self, session_folder_path: Path, samples: pl.DataFrame):
        del session_folder_path
        type(self).last_samples = samples

    def detect_eye_movements(self, **config):
        type(self).last_config = config
        return (
            pl.DataFrame(
                {
                    "tStart": [4.5],
                    "tEnd": [6.0],
                    "pupilAvg": [3.3],
                }
            ),
            pl.DataFrame({"tStart": [6.5], "tEnd": [8.0]}),
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
            pl.DataFrame({"tStart": [1.0], "tEnd": [2.0]}),
            pl.DataFrame({"tStart": [2.2], "tEnd": [3.0]}),
        )


@pytest.fixture
def detector_mapping(monkeypatch):
    import pyxations.bids_formatting as bids_formatting

    monkeypatch.setattr(
        bids_formatting,
        "EYE_MOVEMENT_DETECTION_DICT",
        {"remodnav": _FakeRemodnav, "engbert": _FakeEngbert},
        raising=False,
    )


def test_tobii_parser_normalizes_time_and_preserves_binocular_pupil(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_tobii_export(tmp_path / "tobii.txt")
    output = tmp_path / "derivatives"

    parser = TobiiParse(output, "feather")
    returned = parser.parse(source, "remodnav", overwrite=True)

    assert isinstance(returned, pl.DataFrame)
    assert isinstance(_FakeRemodnav.last_samples, pl.DataFrame)
    assert _FakeRemodnav.last_config == {
        "savgol_length": 0.195,
        "max_pso_dur": 0.3,
    }

    samples = pl.read_ipc(output / "samples.feather")
    messages = pl.read_ipc(output / "msg.feather")
    blinks = pl.read_ipc(output / "remodnav_events" / "blink.feather")

    np.testing.assert_allclose(samples["tSample"].to_numpy(), [0.0, 3.333, 6.666, 10.0])
    np.testing.assert_allclose(samples["LPupil"].to_numpy(), [3.1, 3.2, 3.3, 3.4])
    np.testing.assert_allclose(samples["RPupil"].to_numpy(), [3.2, 3.3, 3.4, 3.5])
    assert samples["LX"].to_list() == [100.0, 101.0, 102.0, 103.0]
    assert samples["RX"].to_list() == [110.0, 111.0, 112.0, 113.0]
    assert samples["Eyes_recorded"].to_list() == ["LR"] * 4
    assert samples["Rate_recorded"][0] == pytest.approx(300.03, rel=1e-3)

    assert messages["message"].to_list() == ["START", "END"]
    assert messages["timestamp"].to_list() == pytest.approx([4.0, 10.0])
    assert blinks.is_empty()


def test_tobii_message_segmentation_uses_polars_messages(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_tobii_export(tmp_path / "tobii.txt")
    output = tmp_path / "derivatives"

    TobiiParse(output, "feather").parse(
        source,
        "remodnav",
        overwrite=True,
        start_msgs={"search": ["START"]},
        end_msgs={"search": ["END"]},
        trial_labels={"search": ["trial-a"]},
        use_regex=False,
    )

    samples = pl.read_ipc(output / "samples.feather")
    fixations = pl.read_ipc(output / "remodnav_events" / "fix.feather")
    saccades = pl.read_ipc(output / "remodnav_events" / "sacc.feather")

    assert samples["phase"].to_list() == ["", "", "search", "search"]
    assert samples["trial_number"].to_list() == [-1, -1, 0, 0]
    assert samples["trial_label"].to_list() == ["", "", "trial-a", "trial-a"]
    assert fixations["phase"].to_list() == ["search"]
    assert saccades["phase"].to_list() == ["search"]


def test_tobii_engbert_receives_inferred_rate_fallback(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_tobii_export(tmp_path / "tobii.txt")
    output = tmp_path / "derivatives"

    TobiiParse(output, "feather").parse(
        source,
        "engbert",
        overwrite=True,
        detector_config={"vfac": 6.0},
    )

    assert isinstance(_FakeEngbert.last_samples, pl.DataFrame)
    assert _FakeEngbert.last_config["vfac"] == 6.0
    assert _FakeEngbert.last_config["sample_rate_fallback"] == pytest.approx(
        300.03, rel=1e-3
    )
    assert (output / "engbert_events" / "fix.feather").exists()


def test_tobii_explicit_sample_rate_overrides_inference(
    tmp_path: Path,
    detector_mapping,
):
    source = _write_tobii_export(tmp_path / "tobii.txt")
    output = tmp_path / "derivatives"

    TobiiParse(output, "feather").parse(
        source,
        "engbert",
        overwrite=True,
        sample_rate=120,
    )

    samples = pl.read_ipc(output / "samples.feather")
    assert samples["Rate_recorded"].to_list() == [120.0] * 4
    assert _FakeEngbert.last_config["sample_rate_fallback"] == 120.0


def test_process_session_requires_exactly_one_tobii_file(
    tmp_path: Path,
    detector_mapping,
):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No Tobii"):
        process_session(
            input_dir,
            "remodnav",
            tmp_path / "out-empty",
            True,
            "feather",
        )

    _write_tobii_export(input_dir / "one.txt")
    _write_tobii_export(input_dir / "two.txt")
    process_session(
        input_dir,
        "remodnav",
        tmp_path / "out-multiple",
        True,
        "feather",
    )
    assert not (tmp_path / "out-multiple" / "samples.feather").exists()
