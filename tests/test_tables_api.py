from __future__ import annotations

import gzip
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

import pyxations
from pyxations.tables import (
    SessionTables,
    as_polars,
    frame_payload,
    json_value,
    payload_frame,
    read_tsv,
    tabular_frame,
    write_tsv,
)


def test_session_tables_clone_and_validation():
    samples = pl.DataFrame({"tSample": [0.0], "X": [1.0]})
    tables = SessionTables(samples=samples, sampling_frequency=100)

    assert tables.samples.equals(samples)
    assert tables.samples is not samples
    assert tables.sampling_frequency == 100.0
    assert tables.fixations.is_empty()

    clone = tables.clone(screen_width=1920)
    assert clone.screen_width == 1920
    assert clone.samples.equals(samples)
    assert clone.samples is not tables.samples

    assert as_polars(None).is_empty()
    with pytest.raises(TypeError, match="must be a Polars DataFrame"):
        as_polars({"x": [1]}, name="samples")
    with pytest.raises(ValueError, match="sampling_frequency"):
        SessionTables(samples=samples, sampling_frequency=np.inf)


def test_json_values_and_payload_round_trip():
    value = {
        "boolean": np.bool_(True),
        "integer": np.int64(2),
        "finite": np.float64(1.5),
        "infinite": np.inf,
        "date": date(2026, 1, 2),
        "datetime": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        "sequence": ("text", None),
        "plain_integer": 3,
        "nan_like": np.array(np.nan),
        "fallback": object(),
    }
    converted = json_value(value)

    assert converted["boolean"] is True
    assert converted["integer"] == 2
    assert converted["finite"] == 1.5
    assert converted["infinite"] is None
    assert converted["date"] == "2026-01-02"
    assert converted["datetime"] == "2026-01-02T03:04:05+00:00"
    assert converted["sequence"] == ["text", None]
    assert converted["plain_integer"] == 3
    assert converted["nan_like"] is None
    assert isinstance(converted["fallback"], str)

    frame = pl.DataFrame({"a": [1], "b": ["value"]})
    payload = frame_payload(frame)
    assert payload == {
        "Columns": ["a", "b"],
        "Records": [{"a": 1, "b": "value"}],
    }
    assert payload_frame(payload).equals(frame)
    assert payload_frame(None).is_empty()
    assert payload_frame({"Columns": ["a", "missing"], "Records": [{"a": 1}]}).columns == [
        "a",
        "missing",
    ]
    assert payload_frame({"Columns": ["a"], "Records": []}).columns == ["a"]


def test_tabular_conversion_and_tsv_io(tmp_path):
    frame = pl.DataFrame(
        {
            "finite": [1.0, np.inf],
            "nested": [[1, 2], None],
            "text": ["one", "two"],
        }
    )
    converted = tabular_frame(frame)

    assert converted["finite"].to_list() == [1.0, None]
    assert converted["nested"].to_list() == ["[1, 2]", None]
    assert tabular_frame(pl.DataFrame()).is_empty()
    assert tabular_frame(pl.DataFrame({"x": [1]})).equals(pl.DataFrame({"x": [1]}))

    plain = write_tsv(
        tmp_path / "nested" / "table.tsv",
        converted,
        include_header=True,
        compressed=False,
    )
    compressed = write_tsv(
        tmp_path / "table.tsv.gz",
        converted,
        include_header=True,
        compressed=True,
    )

    assert read_tsv(plain, has_header=True).shape == (2, 3)
    assert read_tsv(
        compressed,
        columns=["finite", "nested", "text"],
        has_header=True,
        schema_overrides={"finite": pl.Float64},
    ).shape == (2, 3)


def test_compressed_tsv_is_a_valid_gzip_archive(tmp_path):
    frame = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})

    path = write_tsv(
        tmp_path / "table.tsv.gz", frame, include_header=True, compressed=True
    )

    # Decompress outside Polars: an archive that Polars happens to tolerate
    # would still be unreadable by every other tool that consumes BIDS.
    assert gzip.decompress(path.read_bytes()) == b"a\tb\n1\tx\n2\ty\nn/a\tz\n"


def test_compressed_tsv_is_byte_reproducible(tmp_path):
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    kwargs = {"include_header": True, "compressed": True}

    first = write_tsv(tmp_path / "first.tsv.gz", frame, **kwargs)
    second = write_tsv(tmp_path / "second.tsv.gz", frame, **kwargs)

    assert first.read_bytes() == second.read_bytes()


def test_lazy_optional_public_import(monkeypatch):
    assert pyxations.__getattr__("RemodnavDetection").__name__ == "RemodnavDetection"
    with pytest.raises(AttributeError, match="does_not_exist"):
        pyxations.__getattr__("does_not_exist")

    def missing_dependency(*args, name):
        error = ModuleNotFoundError(f"No module named {name!r}")
        error.name = name
        raise error

    monkeypatch.setattr(
        pyxations,
        "import_module",
        lambda *args: missing_dependency(name="remodnav"),
    )
    with pytest.raises(ImportError, match=r"pyxations\[remodnav\]"):
        pyxations.__getattr__("RemodnavDetection")

    monkeypatch.setattr(
        pyxations,
        "import_module",
        lambda *args: missing_dependency(name="unrelated"),
    )
    with pytest.raises(ModuleNotFoundError, match="unrelated"):
        pyxations.__getattr__("RemodnavDetection")
