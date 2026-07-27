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
    """Return a new empty Polars frame for dataclass defaults."""

    return pl.DataFrame()


def as_polars(frame: Any | None, *, name: str = "table") -> pl.DataFrame:
    """Clone a Polars table, or create an empty one for ``None``."""

    if frame is None:
        return pl.DataFrame()
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    raise TypeError(
        f"{name} must be a Polars DataFrame, got {type(frame)!r}."
    )


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
        """Return an independent copy, optionally replacing selected fields."""

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
    """Convert scalar or nested table values to JSON-safe values."""

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


def frame_payload(
    frame: Any | None,
    *,
    columns_key: str = "Columns",
    records_key: str = "Records",
) -> dict[str, Any]:
    """Serialize a table into an explicitly column-ordered JSON payload."""

    table = as_polars(frame)
    records = [
        {column: json_value(value) for column, value in row.items()}
        for row in table.to_dicts()
    ]
    return {columns_key: table.columns, records_key: records}


def payload_frame(payload: Mapping[str, Any] | None) -> pl.DataFrame:
    """Deserialize either historical or canonical table payload keys."""

    if not payload:
        return pl.DataFrame()
    columns = list(payload.get("Columns", payload.get("columns", [])))
    records = list(payload.get("Records", payload.get("data", [])))
    if not records:
        return pl.DataFrame({column: [] for column in columns})
    table = pl.DataFrame(records, strict=False)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        table = table.with_columns(pl.lit(None).alias(column) for column in missing)
    return table.select(columns)


def _tabular_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(json_value(value), ensure_ascii=False)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def tabular_frame(frame: Any | None) -> pl.DataFrame:
    """Return a TSV-safe frame with nested objects encoded as JSON strings."""

    table = as_polars(frame)
    if table.is_empty() or not table.columns:
        return table
    return pl.DataFrame(
        [
            {
                column: _tabular_value(value)
                for column, value in row.items()
            }
            for row in table.to_dicts()
        ],
        schema=table.columns,
        orient="row",
        strict=False,
    )


def write_tsv(
    path: str | Path,
    frame: Any,
    *,
    include_header: bool,
    compressed: bool,
) -> Path:
    """Write a deterministic BIDS TSV, optionally gzip-compressed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = tabular_frame(frame)

    if compressed:
        with destination.open("wb") as binary_stream, gzip.GzipFile(
            filename="", fileobj=binary_stream, mode="wb", mtime=0
        ) as gzip_stream, io.TextIOWrapper(
            gzip_stream, encoding="utf-8", newline=""
        ) as text_stream:
            table.write_csv(
                text_stream,
                separator="\t",
                include_header=include_header,
                null_value="n/a",
            )
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
) -> pl.DataFrame:
    """Read a BIDS TSV using Polars with stable null handling."""

    options: dict[str, Any] = {
        "separator": "\t",
        "has_header": has_header,
        "null_values": ["n/a"],
        "infer_schema_length": None,
        "truncate_ragged_lines": False,
    }
    if columns is not None:
        options["new_columns"] = list(columns)
    return pl.read_csv(path, **options)
