"""Shared storage helpers for vendor parsers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyxations.export import get_exporter


def _frame_is_empty(frame: Any | None) -> bool:
    """Return whether a pandas- or Polars-like dataframe has no rows."""
    if frame is None:
        return True
    if hasattr(frame, "is_empty"):
        return bool(frame.is_empty())
    if hasattr(frame, "empty"):
        return bool(frame.empty)
    try:
        return len(frame) == 0
    except TypeError as exc:
        raise TypeError(
            "Expected a dataframe-like object supporting is_empty(), empty, or len()."
        ) from exc


class BidsParse:
    """Base class that stores parser outputs with the configured exporter.

    The historical class name is retained for compatibility, although this
    helper stores derivative tables and does not itself implement BIDS.
    """

    def __init__(self, session_folder_path: Any, export_method: str) -> None:
        self.session_folder_path = Path(session_folder_path)
        self.export_method = get_exporter(export_method)
        self.detection_algorithm: str | None = None

    def save_dataframe(self, df: Any, path: Any, data_name: str, key: str) -> None:
        self.export_method.save(df, path, data_name, key)

    def store_dataframes(
        self,
        dfSamples: Any,
        dfCalib: Any | None = None,
        dfFix: Any | None = None,
        dfSacc: Any | None = None,
        dfHeader: Any | None = None,
        dfBlink: Any | None = None,
        dfMsg: Any | None = None,
    ) -> None:
        """Store samples and event tables without assuming pandas.

        Empty fixation, saccade, and blink tables are intentionally written so
        every processed session has a predictable derivative layout.
        """
        if self.detection_algorithm is None:
            raise RuntimeError("detection_algorithm must be set before storing tables.")

        if getattr(self.export_method, "is_bids", False):
            self.export_method.save_derivatives(
                session_path=self.session_folder_path,
                samples=dfSamples,
                calibration=dfCalib,
                fixations=dfFix,
                saccades=dfSacc,
                header=dfHeader,
                blinks=dfBlink,
                messages=dfMsg,
                detection_algorithm=self.detection_algorithm,
            )
            return

        self.session_folder_path.mkdir(parents=True, exist_ok=True)
        self.save_dataframe(dfSamples, self.session_folder_path, "samples", key="samples")

        optional_tables = (
            (dfCalib, "calib", "calib"),
            (dfHeader, "header", "header"),
            (dfMsg, "msg", "msg"),
        )
        for frame, data_name, key in optional_tables:
            if not _frame_is_empty(frame):
                self.save_dataframe(frame, self.session_folder_path, data_name, key=key)

        events_path = self.session_folder_path / f"{self.detection_algorithm}_events"
        events_path.mkdir(parents=True, exist_ok=True)
        for frame, data_name, key in (
            (dfFix, "fix", "fix"),
            (dfSacc, "sacc", "sacc"),
            (dfBlink, "blink", "blink"),
        ):
            if frame is None:
                raise ValueError(f"{data_name} dataframe cannot be None.")
            self.save_dataframe(frame, events_path, data_name, key=key)
