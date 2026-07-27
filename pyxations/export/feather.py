"""Polars-backed Apache Arrow IPC/Feather export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


class FeatherExport:
    """Read and write Pyxations derivative tables as Arrow IPC files.

    Pyxations uses Polars as its canonical in-memory dataframe backend.
    Pandas-like frames are normalized through plain Python columns, avoiding a
    PyArrow dependency while preserving the legacy export API.
    """

    _LEGACY_COLUMNS = ("__index_level_0__", "line_number")

    def save(
        self,
        df: Any,
        path: str | Path,
        data_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Write a Polars dataframe to ``<path>/<data_name>.feather``.

        Parameters
        ----------
        df
            Polars or pandas-like table to write.
        path
            Destination directory.
        data_name
            File stem without the ``.feather`` extension.

        Raises
        ------
        TypeError
            If ``df`` is not a supported dataframe.
        ValueError
            If ``data_name`` is empty.
        """
        if isinstance(df, pl.DataFrame):
            frame = df
        elif hasattr(df, "to_dict") and hasattr(df, "columns"):
            try:
                frame = pl.DataFrame(df.to_dict(orient="list"))
            except TypeError as exc:
                raise TypeError(
                    "FeatherExport.save() accepts Polars or pandas-like "
                    "DataFrame objects."
                ) from exc
        else:
            raise TypeError(
                "FeatherExport.save() accepts Polars or pandas-like "
                "DataFrame objects. "
                f"Received {type(df).__name__}."
            )
        if not isinstance(data_name, str) or not data_name.strip():
            raise ValueError("data_name must be a non-empty string.")

        destination = Path(path) / f"{data_name}.feather"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.write_ipc(destination)

    def read(self, path: str | Path, data_name: str) -> pl.DataFrame:
        """Read a derivative table as a Polars dataframe.

        Legacy pandas index columns and Pyxations' historical ``line_number``
        storage column are removed when present, matching the previous reader
        behavior.
        """
        if not isinstance(data_name, str) or not data_name.strip():
            raise ValueError("data_name must be a non-empty string.")

        source = Path(path) / f"{data_name}.feather"
        frame = pl.read_ipc(source, memory_map=False)
        return frame.drop(list(self._LEGACY_COLUMNS), strict=False)

    def extension(self) -> str:
        """Return the filename extension used by this exporter."""
        return ".feather"
