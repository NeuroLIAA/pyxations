"""Polars-native table models and BIDS tabular I/O helpers."""

from __future__ import annotations

import gzip
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def empty_frame() -> pl.DataFrame:
    """Return a new empty Polars frame for dataclass defaults.

    Returns
    -------
    polars.DataFrame
        New schema-less empty table.
    """

    return pl.DataFrame()


def as_polars(frame: Any | None, *, name: str = "table") -> pl.DataFrame:
    """Clone a Polars table, or create an empty one for ``None``.

    Parameters
    ----------
    frame : polars.DataFrame or None
        Table to clone. ``None`` creates a schema-less empty table.
    name : str, default "table"
        Human-readable value name used in type errors.

    Returns
    -------
    polars.DataFrame
        Independent clone of ``frame`` or a new empty table.

    Raises
    ------
    TypeError
        If ``frame`` is neither a Polars DataFrame nor ``None``.
    """

    if frame is None:
        return pl.DataFrame()
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    raise TypeError(f"{name} must be a Polars DataFrame, got {type(frame)!r}.")


@dataclass(slots=True)
class SessionTables:
    """Canonical in-memory representation of one eye-tracking session."""

    samples: pl.DataFrame
    fixations: pl.DataFrame = field(default_factory=empty_frame)
    saccades: pl.DataFrame = field(default_factory=empty_frame)
    blinks: pl.DataFrame = field(default_factory=empty_frame)
    messages: pl.DataFrame = field(default_factory=empty_frame)
    calibration: pl.DataFrame = field(default_factory=empty_frame)
    header: pl.DataFrame = field(default_factory=empty_frame)
    behavioral_events: pl.DataFrame = field(default_factory=empty_frame)
    sampling_frequency: float | None = None
    screen_width: int | None = None
    screen_height: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "samples",
            "fixations",
            "saccades",
            "blinks",
            "messages",
            "calibration",
            "header",
            "behavioral_events",
        ):
            setattr(self, name, as_polars(getattr(self, name), name=name))

        if self.sampling_frequency is not None:
            frequency = float(self.sampling_frequency)
            if not math.isfinite(frequency) or frequency <= 0:
                raise ValueError(
                    "sampling_frequency must be finite and greater than zero"
                )
            self.sampling_frequency = frequency

    def clone(self, **updates: Any) -> SessionTables:
        """Return an independent copy, optionally replacing selected fields.

        Parameters
        ----------
        **updates : object
            Field values to replace in the cloned session container.

        Returns
        -------
        SessionTables
            Independent container whose table fields are cloned.
        """

        values = {
            "samples": self.samples,
            "fixations": self.fixations,
            "saccades": self.saccades,
            "blinks": self.blinks,
            "messages": self.messages,
            "calibration": self.calibration,
            "header": self.header,
            "behavioral_events": self.behavioral_events,
            "sampling_frequency": self.sampling_frequency,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
        }
        values.update(updates)
        return SessionTables(**values)


def json_value(value: Any) -> Any:
    """Convert scalar or nested table values to JSON-safe values.

    Parameters
    ----------
    value : object
        Scalar, sequence, mapping, date, or Polars series to normalize.

    Returns
    -------
    object
        Recursively normalized value accepted by the standard JSON encoder.
    """

    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, pl.Series):
        return [json_value(item) for item in value.to_list()]
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    try:
        if bool(np.isnan(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def frame_payload(frame: Any | None) -> dict[str, Any]:
    """Serialize a table into an explicitly column-ordered JSON payload.

    Parameters
    ----------
    frame : polars.DataFrame or None
        Table to serialize.

    Returns
    -------
    dict
        Payload containing ordered ``Columns`` and row ``Records``.
    """

    table = as_polars(frame)
    records = [
        {column: json_value(value) for column, value in row.items()}
        for row in table.to_dicts()
    ]
    return {"Columns": table.columns, "Records": records}


def payload_frame(payload: Mapping[str, Any] | None) -> pl.DataFrame:
    """Deserialize a canonical table payload.

    Parameters
    ----------
    payload : mapping or None
        Payload containing optional ``Columns`` and ``Records`` entries.

    Returns
    -------
    polars.DataFrame
        Reconstructed table, preserving the recorded column order.
    """

    if not payload:
        return pl.DataFrame()
    columns = list(payload.get("Columns", []))
    records = list(payload.get("Records", []))
    if not records:
        return pl.DataFrame({column: [] for column in columns})
    table = pl.DataFrame(records, strict=False)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        table = table.with_columns(pl.lit(None).alias(column) for column in missing)
    return table.select(columns)


def _tabular_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(json_value(value), ensure_ascii=False)


def tabular_frame(frame: Any | None) -> pl.DataFrame:
    """Return a TSV-safe frame with nested objects encoded as JSON strings.

    Parameters
    ----------
    frame : polars.DataFrame or None
        Table to normalize for BIDS tabular output.

    Returns
    -------
    polars.DataFrame
        Clone with non-finite floats nulled and nested values JSON-encoded.
    """

    table = as_polars(frame)
    if table.is_empty() or not table.columns:
        return table

    expressions = []
    for column, dtype in table.schema.items():
        if dtype.is_float():
            expressions.append(
                pl.when(pl.col(column).is_finite())
                .then(pl.col(column))
                .otherwise(None)
                .alias(column)
            )
        elif dtype.is_nested() or dtype == pl.Object:
            expressions.append(
                pl.col(column)
                .map_elements(_tabular_json, return_dtype=pl.String)
                .alias(column)
            )
    return table.with_columns(expressions) if expressions else table.clone()


def write_tsv(
    path: str | Path,
    frame: Any,
    *,
    include_header: bool,
    compressed: bool,
) -> Path:
    """Write a deterministic BIDS TSV, optionally gzip-compressed.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination filename.
    frame : polars.DataFrame
        Table to serialize.
    include_header : bool
        Whether to write column names as the first row.
    compressed : bool
        Whether to create a deterministic gzip stream.

    Returns
    -------
    pathlib.Path
        Destination path after the file has been written.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = tabular_frame(frame)

    if compressed:
        # Compress into memory rather than onto the open file. Polars writes
        # straight to the file descriptor whenever the object it is given
        # exposes fileno(), which a wrapper chain over a real file does, so
        # writing through GzipFile that way silently bypasses compression and
        # produces a corrupt archive. A BytesIO has no fileno(), so the text
        # actually goes through the compressor.
        buffer = io.BytesIO()
        with (
            gzip.GzipFile(
                filename="", fileobj=buffer, mode="wb", mtime=0
            ) as gzip_stream,
            io.TextIOWrapper(gzip_stream, encoding="utf-8", newline="") as text_stream,
        ):
            table.write_csv(
                text_stream,
                separator="\t",
                include_header=include_header,
                null_value="n/a",
            )
        destination.write_bytes(buffer.getvalue())
    else:
        table.write_csv(
            destination,
            separator="\t",
            include_header=include_header,
            null_value="n/a",
        )
    return destination


def read_tsv(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    has_header: bool,
    schema_overrides: Mapping[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """Read a BIDS TSV using Polars with stable null handling.

    Parameters
    ----------
    path : str or pathlib.Path
        Plain or gzip-compressed TSV file.
    columns : sequence of str, optional
        Column names for a headerless table.
    has_header : bool
        Whether the first row contains column names.
    schema_overrides : mapping, optional
        Polars data types to impose on selected columns.

    Returns
    -------
    polars.DataFrame
        Parsed table with BIDS ``n/a`` values represented as nulls.
    """

    options: dict[str, Any] = {
        "separator": "\t",
        "has_header": has_header,
        "null_values": ["n/a"],
        "infer_schema_length": None,
        "truncate_ragged_lines": False,
    }
    if columns is not None:
        options["new_columns"] = list(columns)
    if schema_overrides is not None:
        options["schema_overrides"] = dict(schema_overrides)
    return pl.read_csv(path, **options)
