from __future__ import annotations

import pytest
import pandas as pd

pl = pytest.importorskip("polars")

from pyxations.export.feather import FeatherExport


def test_feather_round_trip_preserves_polars_schema_and_nulls(tmp_path):
    exporter = FeatherExport()
    original = pl.DataFrame(
        {
            "trial_number": pl.Series([1, 1, 2], dtype=pl.Int64),
            "tSample": pl.Series([0.0, 8.5, 17.0], dtype=pl.Float64),
            "pupil": pl.Series([3.1, None, 3.4], dtype=pl.Float64),
            "eye": pl.Series(["L", "L", "R"], dtype=pl.String),
        }
    )

    exporter.save(original, tmp_path / "nested", "samples")
    restored = exporter.read(tmp_path / "nested", "samples")

    assert restored.equals(original)
    assert restored.schema == original.schema


def test_feather_reader_removes_legacy_columns(tmp_path):
    exporter = FeatherExport()
    legacy = pl.DataFrame(
        {
            "__index_level_0__": [0, 1],
            "line_number": [10, 11],
            "X": [100.0, 110.0],
        }
    )
    legacy.write_ipc(tmp_path / "samples.feather")

    restored = exporter.read(tmp_path, "samples")

    assert restored.columns == ["X"]


def test_feather_export_rejects_unsupported_frames(tmp_path):
    exporter = FeatherExport()

    with pytest.raises(TypeError, match="Polars or pandas-like"):
        exporter.save({"X": [1.0]}, tmp_path, "samples")


def test_feather_export_normalizes_pandas_without_pyarrow(tmp_path):
    exporter = FeatherExport()
    original = pd.DataFrame({"X": [1.0, 2.0], "eye": ["L", "R"]})

    exporter.save(original, tmp_path, "samples")

    assert exporter.read(tmp_path, "samples").to_dict(as_series=False) == {
        "X": [1.0, 2.0],
        "eye": ["L", "R"],
    }


def test_feather_export_rejects_empty_data_name(tmp_path):
    exporter = FeatherExport()
    frame = pl.DataFrame({"X": [1.0]})

    with pytest.raises(ValueError, match="non-empty string"):
        exporter.save(frame, tmp_path, "")
    with pytest.raises(ValueError, match="non-empty string"):
        exporter.read(tmp_path, "")


def test_feather_extension_is_stable():
    assert FeatherExport().extension() == ".feather"
