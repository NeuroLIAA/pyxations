import json
import shutil
import sys
from pathlib import Path

import polars as pl
import pytest

from pyxations import (
    Experiment,
    bids_formatting,
    compute_derivatives_for_dataset,
    dataset_to_bids,
)
from pyxations.bids import (
    BIDSValidationError,
    validate_bids_dataset,
    validator_command,
)
from pyxations.export.bids import (
    BIDSDerivativeExport,
    initialize_bids_derivative,
)
from pyxations.tables import SessionTables


def _write_eyelink(folder: Path) -> None:
    (folder / "s01_A_task-look.asc").write_text(
        """** VERSION: EYELINK II 1
SAMPLES GAZE LEFT RIGHT RATE 1000.00 TRACKING CR FILTER 2
!MODE RECORD CR 1000 2 1 LR
MSG 1000 beginning_of_stimuli
1000 100.0 200.0 500.0 110.0 210.0 510.0
1001 101.0 201.0 501.0 111.0 211.0 511.0
1002 102.0 202.0 502.0 112.0 212.0 512.0
EFIX L 1000 1002 2 101.0 201.0 501.0
""",
        encoding="utf-8",
    )


def _write_gazepoint(folder: Path) -> None:
    pl.DataFrame(
        {
            "TIME": [0.0, 1 / 60, 2 / 60],
            "LPOGX": [0.2, 0.21, 0.22],
            "LPOGY": [0.3, 0.31, 0.32],
            "LPD": [3.1, 3.2, 3.3],
            "RPOGX": [0.25, 0.26, 0.27],
            "RPOGY": [0.35, 0.36, 0.37],
            "RPD": [3.0, 3.1, 3.2],
        }
    ).write_csv(folder / "s01_A_task-look.csv")


def _write_tobii(folder: Path) -> None:
    pl.DataFrame(
        {
            "Eyetracker timestamp": [1_000_000, 1_016_667, 1_033_334],
            "Gaze2d_Left.x": [100.0, 101.0, 102.0],
            "Gaze2d_Left.y": [200.0, 201.0, 202.0],
            "PupilDiam_Left": [3.1, 3.2, 3.3],
            "Gaze2d_Right.x": [110.0, 111.0, 112.0],
            "Gaze2d_Right.y": [210.0, 211.0, 212.0],
            "PupilDiam_Right": [3.0, 3.1, 3.2],
        }
    ).write_csv(folder / "s01_A_task-look.txt", separator="\t")


def _write_webgazer(folder: Path) -> None:
    samples = [
        {"x": 100, "y": 200, "t": 0},
        {"x": 101, "y": 201, "t": 17},
        {"x": 102, "y": 202, "t": 34},
    ]
    pl.DataFrame(
        {
            "trial_index": [0],
            "time_elapsed": [1_000],
            "webgazer_data": [json.dumps(samples)],
        }
    ).write_csv(folder / "s01_A_task-look.csv")


WRITERS = {
    "eyelink": _write_eyelink,
    "gaze": _write_gazepoint,
    "tobii": _write_tobii,
    "webgazer": _write_webgazer,
}


def _make_dataset(
    tmp_path: Path, format_name: str, *, include_behavior: bool = False
) -> Path:
    source = tmp_path / f"{format_name}-source"
    source.mkdir()
    WRITERS[format_name](source)
    if include_behavior:
        pl.DataFrame(
            {
                "participant": ["s01"],
                "trial_type": ["target"],
                "response_time": [0.42],
            }
        ).write_csv(source / "s01_A_task-look_behavior.csv")
    return dataset_to_bids(
        tmp_path,
        source,
        f"{format_name}-dataset",
        format_name=format_name,
        authors=["Pyxations test suite"],
    )


@pytest.mark.parametrize(
    ("format_name", "eye_count"),
    [("eyelink", 2), ("gaze", 2), ("tobii", 2), ("webgazer", 1)],
)
def test_dataset_to_bids_writes_standardized_recordings(
    tmp_path, format_name, eye_count
):
    dataset = _make_dataset(tmp_path, format_name)

    assert (dataset / "dataset_description.json").is_file()
    assert (dataset / "participants.tsv").is_file()
    source_files = sorted(
        path.name for path in (dataset / "sourcedata").iterdir() if path.is_file()
    )
    assert len(source_files) == 1
    physio = list((dataset / "sub-0001" / "ses-A" / "beh").glob("*_physio.tsv.gz"))
    sidecars = list((dataset / "sub-0001" / "ses-A" / "beh").glob("*_physio.json"))
    assert len(physio) == eye_count
    assert len(sidecars) == eye_count
    if format_name == "eyelink":
        assert (
            len(
                list(
                    (dataset / "sub-0001" / "ses-A" / "beh").glob(
                        "*_physioevents.tsv.gz"
                    )
                )
            )
            == eye_count
        )

    participants = pl.read_csv(dataset / "participants.tsv", separator="\t")
    assert participants.item(0, "participant_id") == "sub-0001"
    for sidecar in sidecars:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        assert metadata["PhysioType"] == "eyetrack"
        assert metadata["StartTime"] == 0.0
        assert metadata["RecordedEye"] in {"left", "right", "cyclopean"}
        assert metadata["SampleCoordinateSystem"] == "custom"
        assert metadata["SampleCoordinateSystemDescription"]
        assert metadata["SamplingFrequency"] > 0
        assert metadata["Columns"][:3] == [
            "timestamp",
            "x_coordinate",
            "y_coordinate",
        ]
        assert metadata["x_coordinate"]["Units"]
        assert metadata["y_coordinate"]["Units"]


@pytest.mark.parametrize("format_name", ["gaze", "webgazer"])
def test_dataset_to_bids_preserves_source_folder_verbatim(tmp_path, format_name):
    source = tmp_path / f"{format_name}-source"
    source.mkdir()
    WRITERS[format_name](source)
    behavioral = source / "behavioral"
    behavioral.mkdir()
    pl.DataFrame(
        {
            "participant": ["s01"],
            "trial_type": ["target"],
            "response_time": [0.42],
        }
    ).write_csv(
        behavioral / "s01_A_task-look_behavior.csv",
    )
    documentation = source / "documentation"
    documentation.mkdir()
    (documentation / "notes.txt").write_text(
        "Source-layout preservation test.\n",
        encoding="utf-8",
    )
    dataset = dataset_to_bids(
        tmp_path,
        source,
        f"{format_name}-dataset",
        format_name=format_name,
        authors=["Pyxations test suite"],
    )
    archived = dataset / "sourcedata"
    source_paths = sorted(
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    )
    archived_paths = sorted(
        path.relative_to(archived) for path in archived.rglob("*") if path.is_file()
    )
    assert archived_paths == source_paths
    for relative_path in source_paths:
        assert (archived / relative_path).read_bytes() == (
            source / relative_path
        ).read_bytes()

    event_file = next((dataset / "sub-0001" / "ses-A" / "beh").glob("*_events.tsv"))
    events = pl.read_csv(event_file, separator="\t")
    assert "behavioral/s01_A_task-look_behavior.csv" in set(
        events.get_column("source_file")
    )


def test_derivatives_are_scheduled_from_raw_bids_session(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path, "gaze", include_behavior=True)
    calls = []

    def capture_process_session(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(bids_formatting, "process_session", capture_process_session)

    derivatives = compute_derivatives_for_dataset(
        dataset,
        "gaze",
    )

    assert derivatives == dataset.with_name(f"{dataset.name}_derivatives")
    assert len(calls) == 1
    assert calls[0][0][0] == dataset / "sub-0001" / "ses-A"


def test_derivatives_do_not_require_sourcedata(tmp_path):
    dataset = _make_dataset(tmp_path, "eyelink")
    shutil.rmtree(dataset / "sourcedata")

    derivatives = compute_derivatives_for_dataset(
        dataset,
        "eyelink",
        detection_algorithm="eyelink",
        num_processes=1,
        overwrite=True,
    )

    assert list((derivatives / "sub-0001" / "ses-A" / "beh").glob("*_physio.tsv.gz"))


@pytest.mark.parametrize(
    ("num_processes", "error"),
    [(0, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_derivative_worker_count_is_validated(tmp_path, num_processes, error):
    with pytest.raises(error):
        compute_derivatives_for_dataset(
            tmp_path / "unused",
            "gaze",
            num_processes=num_processes,
        )


def _make_derivative_dataset(tmp_path: Path, format_name: str) -> Path:
    raw = _make_dataset(tmp_path, format_name)
    derivatives = raw.with_name(f"{raw.name}_derivatives")
    initialize_bids_derivative(raw, derivatives)
    session = derivatives / "sub-0001" / "ses-A"

    samples = pl.DataFrame(
        {
            "tSample": [1_000.0, 1_017.0, 1_034.0],
            "X": [100.0, 101.0, 102.0],
            "Y": [200.0, 201.0, 202.0],
            "Pupil": [3.0, 3.1, 3.2],
            "trial_number": [0, 0, 0],
            "phase": ["look", "look", "look"],
            "Calib_index": [1, 1, 1],
        }
    )
    fixations = pl.DataFrame(
        {
            "tStart": [1_000.0],
            "tEnd": [1_017.0],
            "duration": [17.0],
            "xAvg": [100.5],
            "yAvg": [200.5],
            "trial_number": [0],
            "phase": ["look"],
        }
    )
    saccades = pl.DataFrame(
        {
            "tStart": [1_017.0],
            "tEnd": [1_034.0],
            "duration": [17.0],
            "xStart": [101.0],
            "yStart": [201.0],
            "xEnd": [102.0],
            "yEnd": [202.0],
            "trial_number": [0],
            "phase": ["look"],
        }
    )
    blinks = pl.DataFrame(
        schema={
            "tStart": pl.Float64,
            "tEnd": pl.Float64,
            "duration": pl.Float64,
            "trial_number": pl.Int64,
            "phase": pl.String,
        }
    )
    exporter = BIDSDerivativeExport()
    exporter.write_session(
        session,
        SessionTables(
            samples=samples,
            fixations=fixations,
            saccades=saccades,
            blinks=blinks,
        ),
        detection_algorithm="remodnav",
    )
    return derivatives


def test_bids_derivatives_are_canonical_and_reversible(tmp_path):
    derivatives = _make_derivative_dataset(tmp_path, "gaze")
    session = derivatives / "sub-0001" / "ses-A"
    description = json.loads(
        (derivatives / "dataset_description.json").read_text(encoding="utf-8")
    )
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "Pyxations"
    assert not list(derivatives.rglob("*.feather"))
    assert not list(derivatives.rglob("*.hdf5"))

    physio = list((session / "beh").glob("*_physio.tsv.gz"))
    events = list((session / "beh").glob("*_physioevents.tsv.gz"))
    assert len(physio) == 1
    assert len(events) == 1
    physio_metadata = json.loads(
        physio[0].with_suffix("").with_suffix(".json").read_text(encoding="utf-8")
    )
    assert physio_metadata["Columns"][:4] == [
        "timestamp",
        "x_coordinate",
        "y_coordinate",
        "pupil_size",
    ]
    assert not {
        "pyx_tsample",
        "pyx_x",
        "pyx_y",
        "pyx_pupil",
    }.intersection(physio_metadata["Columns"])
    assert physio_metadata["PyxationsCanonicalColumnMap"] == {
        "timestamp": "tSample",
        "x_coordinate": "X",
        "y_coordinate": "Y",
        "pupil_size": "Pupil",
    }

    bundle = BIDSDerivativeExport().read_session(session, "remodnav")
    assert bundle.samples.columns == [
        "tSample",
        "X",
        "Y",
        "Pupil",
        "trial_number",
        "phase",
        "Calib_index",
    ]
    assert bundle.samples.schema == {
        "tSample": pl.Float64,
        "X": pl.Float64,
        "Y": pl.Float64,
        "Pupil": pl.Float64,
        "trial_number": pl.Int64,
        "phase": pl.String,
        "Calib_index": pl.Int64,
    }
    assert bundle.samples["tSample"].to_list() == [
        1_000.0,
        1_017.0,
        1_034.0,
    ]
    assert bundle.fixations["trial_number"].to_list() == [0]
    assert bundle.saccades["xEnd"].to_list() == [102.0]
    assert bundle.blinks.is_empty()

    experiment = Experiment(derivatives.with_name("gaze-dataset"))
    experiment.load_data("remodnav")
    trial = experiment["0001"]["A"][0]
    assert trial.samples()["phase"].to_list() == ["look", "look", "look"]
    assert trial.fixations()["xAvg"].to_list() == [100.5]


@pytest.mark.parametrize("format_name", list(WRITERS))
def test_bids_derivatives_pass_official_validator(tmp_path, format_name):
    command = validator_command()
    if command is None:
        pytest.skip("Official BIDS Validator or Deno is not installed")
    derivatives = _make_derivative_dataset(tmp_path, format_name)
    validate_bids_dataset(derivatives, command=command)


@pytest.mark.parametrize("format_name", list(WRITERS))
def test_dataset_to_bids_passes_official_validator(tmp_path, format_name):
    command = validator_command()
    if command is None:
        pytest.skip("Official BIDS Validator or Deno is not installed")
    dataset = _make_dataset(tmp_path, format_name)
    validate_bids_dataset(dataset, command=command)


def test_validate_bids_dataset_reports_validator_failure(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    command = [
        sys.executable,
        "-c",
        "import sys; print('validator details'); sys.exit(1)",
    ]
    with pytest.raises(BIDSValidationError, match="validator details"):
        validate_bids_dataset(dataset, command=command)
