"""Source-independent behavioral table ingestion and column normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import polars as pl


def _column_name(value: str, *, prefix: str = "behavioral") -> str:
    """Create a stable BIDS-tabular column name from an external label."""

    column = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower()
    if not column:
        return f"{prefix}_field"
    if column[0].isdigit():
        return f"{prefix}_{column}"
    return column


def normalize_behavioral_events(
    events: pl.DataFrame,
    *,
    column_map: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Map source columns onto experiment-level behavioral concepts.

    Mapping is intentionally independent of the source format. This lets a
    PsychoPy log, CSV, TSV, or future adapter satisfy the same experiment
    schema without embedding task semantics in the format parser.
    """

    if not column_map:
        return events

    normalized = {
        str(source): _column_name(str(destination))
        for source, destination in column_map.items()
    }
    missing = sorted(set(normalized) - set(events.columns))
    if missing:
        raise ValueError(f"Behavioral columns not found for renaming: {missing}")

    resulting = [normalized.get(column, column) for column in events.columns]
    if len(resulting) != len(set(resulting)):
        raise ValueError("Behavioral column mapping creates duplicate columns")
    return events.rename(normalized)


def read_behavioral_events(
    path: str | Path,
    *,
    column_map: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Read a CSV, TSV, or PsychoPy log as a behavioral event table.

    The returned table retains source values and applies the same optional
    experiment-level column mapping regardless of its input format. BIDS
    timing normalization is performed later by the dataset writer.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        events = pl.read_csv(path, infer_schema_length=None)
    elif suffix == ".tsv":
        events = pl.read_csv(path, separator="\t", infer_schema_length=None)
    elif suffix == ".log":
        from .psychopy import psychopy_log_to_events

        events = psychopy_log_to_events(path)
    else:
        raise ValueError(
            f"Unsupported behavioral file format {suffix!r}; "
            "expected .csv, .tsv, or .log"
        )
    return normalize_behavioral_events(events, column_map=column_map)
