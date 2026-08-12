from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import polars as pl
import pytest

import pyxations.export.bids as export_module
from pyxations.export.bids import BIDSDerivativeExport


def test_export_column_frequency_and_version_helpers(monkeypatch):
    forward, reverse = export_module._column_mapping(["A B", "a-b", ""])
    assert forward == {
        "A B": "pyx_a_b",
        "a-b": "pyx_a_b_2",
        "": "pyx_column",
    }
    assert reverse["pyx_a_b_2"] == "a-b"

    assert export_module._infer_frequency(pl.Series([1.0])) == 1.0
    assert export_module._infer_frequency(pl.Series([1.0, 1.0])) == 1.0
    assert export_module._infer_frequency(pl.Series([0.0, 0.5, 1.0])) == 2.0

    monkeypatch.setattr(
        export_module,
        "version",
        lambda name: (_ for _ in ()).throw(PackageNotFoundError(name)),
    )
    assert export_module._package_version() == "development"


def test_export_source_metadata_and_column_selection(tmp_path):
    exporter = BIDSDerivativeExport()
    derivative = tmp_path / "dataset_derivatives"
    session = derivative / "sub-0001" / "ses-A"
    session.mkdir(parents=True)
    raw_beh = tmp_path / "dataset" / "sub-0001" / "ses-A" / "beh"
    raw_beh.mkdir(parents=True)
    (raw_beh / "invalid_physio.json").write_text("{", encoding="utf-8")

    assert exporter._raw_sidecars(session) == []
    assert exporter._source_prefix(session, []) == "sub-0001_ses-A_task-eyetracking"

    left = pl.DataFrame({"X": [1.0], "Y": [2.0], "eye": ["L"]})
    right = left.with_columns(pl.lit("R").alias("eye"))
    assert exporter._sample_columns(left)[-1] == "left"
    assert exporter._sample_columns(right)[-1] == "right"
    assert exporter._sample_columns(left.drop("eye"))[-1] == "cyclopean"
    assert exporter._sample_columns(
        pl.DataFrame({"Gaze2d_Left.x": [1.0], "Gaze2d_Left.y": [2.0]})
    ) == ("Gaze2d_Left.x", "Gaze2d_Left.y", None, "left")
    with pytest.raises(ValueError, match="gaze-coordinate"):
        exporter._sample_columns(pl.DataFrame({"other": [1]}))

    with pytest.raises(ValueError, match="timestamp column"):
        exporter._time_column(pl.DataFrame({"X": [1.0], "Y": [2.0]}))
    assert exporter._time_scale(pl.DataFrame({"t_acum": [1]}), "t_acum", {}) == 1000
    assert (
        exporter._time_scale(
            pl.DataFrame({"tSample": [1], "TIMETICK": [1]}), "tSample", {}
        )
        == 1
    )
    assert (
        exporter._time_scale(
            pl.DataFrame({"tSample": [1], "Gaze2d_Left.x": [1]}), "tSample", {}
        )
        == 1_000_000
    )


def test_export_rejects_invalid_samples_and_missing_derivatives(tmp_path):
    exporter = BIDSDerivativeExport()
    with pytest.raises(ValueError, match="no valid timestamps"):
        exporter.write_session(
            tmp_path / "sub-0001" / "ses-A",
            export_module.SessionTables(
                samples=pl.DataFrame(
                    {
                        "tSample": [None],
                        "X": [1.0],
                        "Y": [2.0],
                    }
                )
            ),
            detection_algorithm="engbert",
        )

    with pytest.raises(FileNotFoundError, match="No BIDS derivatives"):
        exporter.read_session(tmp_path / "sub-0001" / "ses-A", "engbert")


def test_event_writer_ignores_unusable_event_tables(tmp_path):
    exporter = BIDSDerivativeExport()
    assert exporter._write_events(
        destination=tmp_path,
        base="sub-0001_ses-A_task-test",
        tables={"fix": pl.DataFrame(), "sacc": pl.DataFrame()},
        sample_time_origin=0,
        sample_time_scale=1000,
        sample_duration=1,
    ) == (None, None)

    assert exporter._write_events(
        destination=tmp_path,
        base="sub-0001_ses-A_task-test",
        tables={"fix": pl.DataFrame({"unrelated": [1]})},
        sample_time_origin=0,
        sample_time_scale=1000,
        sample_duration=1,
    ) == (None, None)

    assert exporter._write_events(
        destination=tmp_path,
        base="sub-0001_ses-A_task-test",
        tables={"fix": pl.DataFrame({"tStart": [None], "tEnd": [None]})},
        sample_time_origin=0,
        sample_time_scale=1000,
        sample_duration=1,
    ) == (None, None)
