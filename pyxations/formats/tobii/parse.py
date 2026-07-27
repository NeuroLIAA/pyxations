"""Parser for Tobii tab-separated text exports.

The parser keeps the vendor columns but adds Pyxations' canonical sample fields,
normalizes timestamps to milliseconds, preserves binocular pupil measurements,
and keeps the complete parsing/detection/preprocessing path in Polars.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing, SessionMetadata

logger = logging.getLogger(__name__)

_DEFAULT_TIMESTAMP_UNITS_PER_MS = 1000.0  # Tobii text timestamps are microseconds.
_REQUIRED_COLUMNS = (
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
)
_EVENT_COLUMNS = {
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "duration": pl.Float64,
}
_SEGMENTATION_KEYS = {
    "trial_labels",
    "start_times",
    "end_times",
    "allow_open_last",
    "require_nonoverlap",
    "start_msgs",
    "end_msgs",
    "durations",
    "case_insensitive",
    "use_regex",
    "return_match_token",
}


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {frame.columns}"
        )


def _read_tobii_text(file_path: Path) -> pl.DataFrame:
    """Read a Tobii TSV while retaining event rows with missing gaze values."""
    try:
        return pl.read_csv(
            file_path,
            separator="\t",
            null_values=["", "NaN", "nan", "NA"],
            infer_schema_length=10_000,
            truncate_ragged_lines=False,
        )
    except pl.exceptions.ComputeError as exc:
        raise ValueError(f"Could not parse Tobii text export {file_path}: {exc}") from exc


def _timestamp_origin(frame: pl.DataFrame) -> float:
    origin = frame.select(pl.col("Recording timestamp").cast(pl.Float64).min()).item()
    if origin is None or not np.isfinite(float(origin)):
        raise ValueError("Tobii export contains no valid Recording timestamp values.")
    return float(origin)


def _normalise_timestamp_expr(origin: float, units_per_ms: float) -> pl.Expr:
    return (
        (pl.col("Recording timestamp").cast(pl.Float64) - pl.lit(origin))
        / pl.lit(units_per_ms)
    )


def _infer_sample_rate(timestamps_ms: pl.Series) -> float:
    values = np.asarray(timestamps_ms.to_numpy(), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(
            "Cannot infer the Tobii sample rate from fewer than two valid samples. "
            "Pass sample_rate explicitly."
        )
    positive_diffs = np.diff(values)
    positive_diffs = positive_diffs[np.isfinite(positive_diffs) & (positive_diffs > 0)]
    if positive_diffs.size == 0:
        raise ValueError(
            "Cannot infer the Tobii sample rate because sample timestamps do not advance. "
            "Pass sample_rate explicitly."
        )
    median_interval_ms = float(np.median(positive_diffs))
    return 1000.0 / median_interval_ms


def _build_samples(
    source: pl.DataFrame,
    *,
    origin: float,
    timestamp_units_per_ms: float,
    sample_rate: float | None,
) -> tuple[pl.DataFrame, float]:
    # Event rows have no Eyetracker timestamp. Keep all actual sample rows,
    # including invalid samples, so timing gaps and validity remain observable.
    samples = source.with_row_index("line_number").filter(
        pl.col("Eyetracker timestamp").is_not_null()
    )
    if samples.is_empty():
        raise ValueError("Tobii export contains no gaze sample rows.")

    samples = (
        samples.with_columns(
            _normalise_timestamp_expr(origin, timestamp_units_per_ms).alias("tSample"),
            pl.col("Gaze2d_Left.x").cast(pl.Float64, strict=False).alias("LX"),
            pl.col("Gaze2d_Left.y").cast(pl.Float64, strict=False).alias("LY"),
            pl.col("Gaze2d_Right.x").cast(pl.Float64, strict=False).alias("RX"),
            pl.col("Gaze2d_Right.y").cast(pl.Float64, strict=False).alias("RY"),
            pl.col("PupilDiam_Left").cast(pl.Float64, strict=False).alias("LPupil"),
            pl.col("PupilDiam_Right").cast(pl.Float64, strict=False).alias("RPupil"),
            pl.col("Validity_Left").cast(pl.Int64, strict=False).alias("LValidity"),
            pl.col("Validity_Right").cast(pl.Int64, strict=False).alias("RValidity"),
        )
        .sort("tSample")
    )

    resolved_rate = float(sample_rate) if sample_rate is not None else _infer_sample_rate(
        samples.get_column("tSample")
    )
    if not np.isfinite(resolved_rate) or resolved_rate <= 0:
        raise ValueError("sample_rate must be finite and greater than zero.")

    samples = samples.with_columns(
        pl.lit(resolved_rate, dtype=pl.Float64).alias("Rate_recorded"),
        pl.lit("LR").alias("Eyes_recorded"),
        pl.lit(0, dtype=pl.Int64).alias("Calib_index"),
    )
    return samples, resolved_rate


def _build_messages(
    source: pl.DataFrame,
    *,
    origin: float,
    timestamp_units_per_ms: float,
) -> pl.DataFrame:
    message_text = pl.col("Event message").cast(pl.String, strict=False).fill_null("")
    return (
        source.with_row_index("line_number")
        .filter((message_text.str.strip_chars() != "") & (message_text != "0"))
        .select(
            _normalise_timestamp_expr(origin, timestamp_units_per_ms).alias("timestamp"),
            message_text.alias("message"),
            pl.col("Event value").alias("event_value"),
            pl.col("line_number"),
        )
        .sort("timestamp")
    )


def _empty_blinks() -> pl.DataFrame:
    """Return an explicitly typed empty blink table.

    Tobii's generic text export used here does not contain blink intervals.
    Validity samples are preserved, but Pyxations does not infer blinks from
    those codes in this parser.
    """
    return pl.DataFrame(schema=_EVENT_COLUMNS)


def _run_detector(
    detector_class: type,
    *,
    detection_algorithm: str,
    session_folder_path: Path,
    samples: pl.DataFrame,
    sample_rate: float,
    detector_config: Mapping[str, Any] | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    config = dict(detector_config or {})
    detector = detector_class(
        session_folder_path=session_folder_path,
        samples=samples,
    )

    if detection_algorithm == "remodnav":
        config.setdefault("savgol_length", 0.195)
        config.setdefault("max_pso_dur", 0.3)
        return detector.detect_eye_movements(**config)

    if detection_algorithm == "engbert":
        config.setdefault("sample_rate_fallback", sample_rate)
        return detector.detect_eye_movements(**config)

    raise ValueError(
        f"Tobii does not define an invocation adapter for {detection_algorithm!r}."
    )


def _apply_requested_segmentation(
    preprocessing: PreProcessing,
    options: Mapping[str, Any],
) -> None:
    prefer_durations = bool(options.get("prefer_durations", False))
    have_explicit_times = "start_times" in options and "end_times" in options
    have_durations = "start_msgs" in options and "durations" in options
    have_message_times = "start_msgs" in options and "end_msgs" in options

    if not (have_explicit_times or have_durations or have_message_times):
        logger.info("No Tobii trial-segmentation configuration supplied.")
        return

    if have_explicit_times:
        method_name = "split_all_into_trials"
    elif have_durations and (prefer_durations or not have_message_times):
        method_name = "split_all_into_trials_by_durations"
    else:
        method_name = "split_all_into_trials_by_msgs"

    method = getattr(preprocessing, method_name)
    allowed_parameters = set(inspect.signature(method).parameters)
    parameters = {
        key: value
        for key, value in options.items()
        if key in _SEGMENTATION_KEYS and key in allowed_parameters
    }
    preprocessing.process({method_name: parameters})


def process_session(
    eye_tracking_data_path: Path,
    detection_algorithm: str,
    session_folder_path: Path,
    overwrite: bool,
    exp_format: str,
    **kwargs: Any,
) -> None:
    """Process the single Tobii text export in a session directory."""
    text_files = sorted(
        file for file in Path(eye_tracking_data_path).iterdir()
        if file.suffix.lower() == ".txt"
    )
    if not text_files:
        raise FileNotFoundError(f"No Tobii .txt export found in {eye_tracking_data_path}.")
    if len(text_files) > 1:
        logger.warning(
            "More than one Tobii text file found in %s; skipping the session.",
            eye_tracking_data_path,
        )
        return

    Path(session_folder_path).mkdir(parents=True, exist_ok=True)
    TobiiParse(session_folder_path, exp_format).parse(
        text_files[0], detection_algorithm, overwrite, **kwargs
    )


class TobiiParse(BidsParse):
    """Parse Tobii tab-separated text exports into Polars derivative tables."""

    def parse(
        self,
        file_path: Path,
        detection_algorithm: str,
        overwrite: bool,
        **kwargs: Any,
    ) -> pl.DataFrame:
        del overwrite  # Dataset-level orchestration handles overwrite behavior.

        from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT

        try:
            detector_class = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm]
        except KeyError as exc:
            available = sorted(EYE_MOVEMENT_DETECTION_DICT)
            raise ValueError(
                f"Unknown detection algorithm {detection_algorithm!r}. "
                f"Available algorithms: {available}."
            ) from exc

        timestamp_units_per_ms = float(
            kwargs.pop("timestamp_units_per_ms", _DEFAULT_TIMESTAMP_UNITS_PER_MS)
        )
        if not np.isfinite(timestamp_units_per_ms) or timestamp_units_per_ms <= 0:
            raise ValueError("timestamp_units_per_ms must be finite and greater than zero.")

        raw_sample_rate = kwargs.pop("sample_rate", None)
        sample_rate = None if raw_sample_rate is None else float(raw_sample_rate)
        raw_detector_config = kwargs.pop("detector_config", None)
        if raw_detector_config is not None and not isinstance(raw_detector_config, Mapping):
            raise TypeError("detector_config must be a mapping or None.")

        source = _read_tobii_text(Path(file_path))
        _require_columns(source, _REQUIRED_COLUMNS, "Tobii text export")
        origin = _timestamp_origin(source)

        samples, resolved_sample_rate = _build_samples(
            source,
            origin=origin,
            timestamp_units_per_ms=timestamp_units_per_ms,
            sample_rate=sample_rate,
        )
        messages = _build_messages(
            source,
            origin=origin,
            timestamp_units_per_ms=timestamp_units_per_ms,
        )
        blinks = _empty_blinks()

        fixations, saccades = _run_detector(
            detector_class,
            detection_algorithm=detection_algorithm,
            session_folder_path=self.session_folder_path,
            samples=samples,
            sample_rate=resolved_sample_rate,
            detector_config=raw_detector_config,
        )
        if not isinstance(fixations, pl.DataFrame) or not isinstance(saccades, pl.DataFrame):
            raise TypeError(
                "The Tobii detector must return Polars DataFrames when given Polars samples."
            )

        preprocessing = PreProcessing(
            samples,
            fixations,
            saccades,
            blinks,
            messages,
            self.session_folder_path,
            metadata=SessionMetadata(
                coords_unit="px",
                time_unit="ms",
                pupil_unit="mm",
            ),
        )
        _apply_requested_segmentation(preprocessing, kwargs)

        self.detection_algorithm = detection_algorithm
        self.store_dataframes(
            preprocessing.samples,
            dfFix=preprocessing.fixations,
            dfSacc=preprocessing.saccades,
            dfBlink=preprocessing.blinks,
            dfMsg=preprocessing.user_messages,
        )
        return source
