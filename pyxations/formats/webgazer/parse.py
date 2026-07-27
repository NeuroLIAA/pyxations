"""Parser for jsPsych WebGazer CSV exports.

The jsPsych WebGazer extension stores one JSON array per trial. Each gaze
sample contains x/y coordinates and a timestamp in milliseconds relative to
that trial's start. Webcam sampling is asynchronous and irregular, so this
adapter never assumes a fixed 30 Hz rate. Instead it measures each trial's
sampling quality and, by default, interpolates that trial onto a regular grid
before passing it to event detectors that require regular sampling.

Other WebGazer/Gorilla exports may use different schemas. The relevant CSV
column names and JSON field names are configurable, but the adapter deliberately
does not claim to parse arbitrary webcam-eye-tracking files.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing, SessionMetadata

logger = logging.getLogger(__name__)

_REQUIRED_EVENT_COLUMNS = ("tStart", "tEnd")
_SEGMENTATION_KEYS = {
    "trial_labels",
    "start_times",
    "end_times",
    "allow_open_last",
    "require_nonoverlap",
}
_MESSAGE_SEGMENTATION_KEYS = {
    "start_msgs",
    "end_msgs",
    "durations",
}

_REMODNAV_FIXATION_SCHEMA = {
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "xStart": pl.Float64,
    "yStart": pl.Float64,
    "xEnd": pl.Float64,
    "yEnd": pl.Float64,
    "ampDeg": pl.Float64,
    "vPeak": pl.Float64,
    "med_vel": pl.Float64,
    "avg_vel": pl.Float64,
    "duration": pl.Float64,
    "xAvg": pl.Float64,
    "yAvg": pl.Float64,
    "pupilAvg": pl.Float64,
    "Calib_index": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "eye": pl.String,
}
_REMODNAV_SACCADE_SCHEMA = {
    key: dtype
    for key, dtype in _REMODNAV_FIXATION_SCHEMA.items()
    if key not in {"xAvg", "yAvg", "pupilAvg"}
}
_ENGBERT_FIXATION_SCHEMA = {
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "duration": pl.Float64,
    "xAvg": pl.Float64,
    "yAvg": pl.Float64,
    "pupilAvg": pl.Float64,
    "eye": pl.String,
    "Calib_index": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "chunk": pl.Int64,
}
_ENGBERT_SACCADE_SCHEMA = {
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "duration": pl.Float64,
    "xStart": pl.Float64,
    "yStart": pl.Float64,
    "xEnd": pl.Float64,
    "yEnd": pl.Float64,
    "ampDeg": pl.Float64,
    "vPeak": pl.Float64,
    "distDeg": pl.Float64,
    "thetaDeg": pl.Float64,
    "eye": pl.String,
    "Calib_index": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "chunk": pl.Int64,
}
_BLINK_SCHEMA = {
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "duration": pl.Float64,
}
_MESSAGE_SCHEMA = {
    "timestamp": pl.Float64,
    "message": pl.String,
}


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {frame.columns}"
        )


def _read_webgazer_csv(file_path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(
            file_path,
            null_values=["", "NaN", "nan", "NA", "null", "None"],
            infer_schema_length=10_000,
            truncate_ragged_lines=False,
        )
    except pl.exceptions.ComputeError as exc:
        raise ValueError(f"Could not parse WebGazer CSV {file_path}: {exc}") from exc


def _coerce_float(value: Any, *, label: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"WebGazer row {row_number} contains a non-numeric {label}: {value!r}."
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"WebGazer row {row_number} contains a non-finite {label}: {value!r}."
        )
    return result


def _decode_trial_samples(
    value: Any,
    *,
    row_number: int,
    x_field: str,
    y_field: str,
    time_field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode and clean one trial's WebGazer JSON sample array."""
    if value is None:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            0,
        )

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return (
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
                0,
            )
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid WebGazer JSON in CSV row {row_number}: {exc.msg}."
            ) from exc
    else:
        decoded = value

    if not isinstance(decoded, list):
        raise ValueError(
            f"WebGazer data in CSV row {row_number} must be a JSON array, "
            f"got {type(decoded).__name__}."
        )

    raw_count = len(decoded)
    rows: list[tuple[float, float, float]] = []
    for sample_index, sample in enumerate(decoded):
        if not isinstance(sample, Mapping):
            raise ValueError(
                f"WebGazer sample {sample_index} in CSV row {row_number} must be "
                f"an object, got {type(sample).__name__}."
            )
        missing = [field for field in (x_field, y_field, time_field) if field not in sample]
        if missing:
            raise ValueError(
                f"WebGazer sample {sample_index} in CSV row {row_number} is "
                f"missing fields {missing}."
            )
        try:
            x = float(sample[x_field])
            y = float(sample[y_field])
            timestamp = float(sample[time_field])
        except (TypeError, ValueError, OverflowError):
            # A prediction may be unavailable while WebGazer is initializing or
            # the face is temporarily lost. Keep this observable through the
            # dropped-sample count instead of inventing coordinates.
            continue
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(timestamp):
            rows.append((timestamp, x, y))

    if not rows:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            raw_count,
        )

    values = np.asarray(rows, dtype=float)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]

    # Keep the last prediction for duplicate timestamps. The extension is
    # asynchronous, and duplicate callbacks do not constitute extra timepoints.
    reverse_unique = np.unique(values[::-1, 0], return_index=True)[1]
    keep = np.sort(values.shape[0] - 1 - reverse_unique)
    values = values[keep]

    return values[:, 0], values[:, 1], values[:, 2], raw_count


def _sampling_statistics(local_times_ms: np.ndarray) -> dict[str, float]:
    positive_intervals = np.diff(local_times_ms)
    positive_intervals = positive_intervals[
        np.isfinite(positive_intervals) & (positive_intervals > 0)
    ]
    if positive_intervals.size == 0:
        raise ValueError(
            "At least two advancing WebGazer timestamps are required in each "
            "trial used for eye-movement detection."
        )

    median_interval = float(np.median(positive_intervals))
    mean_interval = float(np.mean(positive_intervals))
    standard_deviation = float(np.std(positive_intervals))
    jitter_cv = standard_deviation / mean_interval if mean_interval > 0 else float("inf")
    maximum_relative_deviation = float(
        np.max(np.abs(positive_intervals - median_interval)) / median_interval
    )
    return {
        "median_interval_ms": median_interval,
        "inferred_rate_hz": 1000.0 / median_interval,
        "jitter_cv": jitter_cv,
        "maximum_relative_deviation": maximum_relative_deviation,
        "minimum_interval_ms": float(np.min(positive_intervals)),
        "maximum_interval_ms": float(np.max(positive_intervals)),
    }


def _regular_grid(
    local_times_ms: np.ndarray,
    *,
    sample_rate_hz: float,
) -> np.ndarray:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("WebGazer sample rates must be finite and greater than zero.")

    start = float(local_times_ms[0])
    stop = float(local_times_ms[-1])
    interval = 1000.0 / sample_rate_hz
    grid = np.arange(start, stop + interval * 0.25, interval, dtype=float)
    grid = grid[grid <= stop + 1e-9]
    if grid.size < 2 and local_times_ms.size >= 2:
        grid = np.asarray([start, stop], dtype=float)
    return grid


def _timestamp_base(
    *,
    timestamp_mode: str,
    row_time_ms: float,
    local_times_ms: np.ndarray,
) -> float:
    if timestamp_mode == "trial_end":
        # jsPsych's `time_elapsed` belongs to the completed trial row, while the
        # sample `t` values are relative to that trial's start.
        return row_time_ms - float(local_times_ms[-1])
    if timestamp_mode == "trial_start":
        return row_time_ms
    if timestamp_mode == "absolute":
        return 0.0
    raise ValueError(
        "timestamp_mode must be 'trial_end', 'trial_start', or 'absolute', "
        f"got {timestamp_mode!r}."
    )


def _build_samples(
    source: pl.DataFrame,
    *,
    webgazer_data_column: str,
    row_time_column: str,
    trial_index_column: str,
    x_field: str,
    y_field: str,
    time_field: str,
    timestamp_mode: str,
    sampling_policy: str,
    target_sample_rate: float | None,
    max_interval_deviation: float,
    minimum_samples_per_trial: int,
    normalize_timestamps: bool,
) -> pl.DataFrame:
    if sampling_policy not in {"resample", "strict"}:
        raise ValueError(
            "sampling_policy must be 'resample' or 'strict', "
            f"got {sampling_policy!r}."
        )
    if target_sample_rate is not None and (
        not np.isfinite(target_sample_rate) or target_sample_rate <= 0
    ):
        raise ValueError("target_sample_rate must be finite and greater than zero.")
    if sampling_policy == "strict" and target_sample_rate is not None:
        raise ValueError(
            "target_sample_rate is only valid with sampling_policy='resample'."
        )
    if not np.isfinite(max_interval_deviation) or max_interval_deviation < 0:
        raise ValueError("max_interval_deviation must be finite and non-negative.")
    if minimum_samples_per_trial < 2:
        raise ValueError("minimum_samples_per_trial must be at least two.")

    rows: list[dict[str, Any]] = []
    segment_index = 0

    for row_number, row in enumerate(source.iter_rows(named=True)):
        local_times, x, y, raw_count = _decode_trial_samples(
            row.get(webgazer_data_column),
            row_number=row_number,
            x_field=x_field,
            y_field=y_field,
            time_field=time_field,
        )
        if local_times.size == 0:
            continue
        if local_times.size < minimum_samples_per_trial:
            logger.warning(
                "Skipping WebGazer CSV row %d: only %d usable samples (minimum=%d).",
                row_number,
                local_times.size,
                minimum_samples_per_trial,
            )
            continue

        statistics = _sampling_statistics(local_times)
        if (
            sampling_policy == "strict"
            and statistics["maximum_relative_deviation"] > max_interval_deviation
        ):
            raise ValueError(
                "Irregular WebGazer sampling in CSV row "
                f"{row_number}: maximum interval deviation is "
                f"{statistics['maximum_relative_deviation']:.3f}, above the "
                f"configured limit {max_interval_deviation:.3f}. Use "
                "sampling_policy='resample' to regularize the trial explicitly."
            )

        resolved_rate = (
            float(target_sample_rate)
            if target_sample_rate is not None
            else statistics["inferred_rate_hz"]
        )
        if sampling_policy == "resample":
            output_local_times = _regular_grid(
                local_times,
                sample_rate_hz=resolved_rate,
            )
            output_x = np.interp(output_local_times, local_times, x)
            output_y = np.interp(output_local_times, local_times, y)
        else:
            output_local_times = local_times
            output_x = x
            output_y = y

        row_time_ms = _coerce_float(
            row.get(row_time_column),
            label=row_time_column,
            row_number=row_number,
        )
        base = _timestamp_base(
            timestamp_mode=timestamp_mode,
            row_time_ms=row_time_ms,
            local_times_ms=local_times,
        )
        absolute_times = base + output_local_times

        nearest_distance = np.min(
            np.abs(output_local_times[:, None] - local_times[None, :]), axis=1
        )
        interpolated = nearest_distance > 1e-7
        trial_index = row.get(trial_index_column)
        dropped_count = raw_count - local_times.size

        for sample_index, (
            absolute_time,
            local_time,
            x_value,
            y_value,
            is_interpolated,
        ) in enumerate(
            zip(
                absolute_times,
                output_local_times,
                output_x,
                output_y,
                interpolated,
            )
        ):
            rows.append(
                {
                    "tSample": float(absolute_time),
                    "webgazer_trial_time": float(local_time),
                    "X": float(x_value),
                    "Y": float(y_value),
                    "Rate_recorded": float(resolved_rate),
                    "Eyes_recorded": "U",
                    "Calib_index": 0,
                    "trial_index": trial_index,
                    "line_number": row_number,
                    "sample_index_in_segment": sample_index,
                    "segment_index": segment_index,
                    "is_interpolated": bool(is_interpolated),
                    "sampling_policy": sampling_policy,
                    "timestamp_mode": timestamp_mode,
                    "raw_sample_count": int(raw_count),
                    "usable_raw_sample_count": int(local_times.size),
                    "dropped_raw_sample_count": int(dropped_count),
                    "sampling_interval_median_ms": statistics["median_interval_ms"],
                    "sampling_rate_inferred_hz": statistics["inferred_rate_hz"],
                    "sampling_jitter_cv": statistics["jitter_cv"],
                    "sampling_max_relative_deviation": statistics[
                        "maximum_relative_deviation"
                    ],
                    "sampling_interval_min_ms": statistics["minimum_interval_ms"],
                    "sampling_interval_max_ms": statistics["maximum_interval_ms"],
                    "source_row_time": row_time_ms,
                }
            )
        segment_index += 1

    if not rows:
        raise ValueError(
            "The WebGazer CSV contains no trials with enough valid gaze samples."
        )

    samples = pl.DataFrame(rows).sort(
        ["tSample", "segment_index", "sample_index_in_segment"]
    )
    if normalize_timestamps:
        origin = samples.select(pl.col("tSample").min()).item()
        samples = samples.with_columns((pl.col("tSample") - float(origin)).alias("tSample"))
    return samples


def _empty_events(detection_algorithm: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    if detection_algorithm == "remodnav":
        return (
            pl.DataFrame(schema=_REMODNAV_FIXATION_SCHEMA),
            pl.DataFrame(schema=_REMODNAV_SACCADE_SCHEMA),
        )
    if detection_algorithm == "engbert":
        return (
            pl.DataFrame(schema=_ENGBERT_FIXATION_SCHEMA),
            pl.DataFrame(schema=_ENGBERT_SACCADE_SCHEMA),
        )
    raise ValueError(f"Unsupported WebGazer detection algorithm {detection_algorithm!r}.")


def _validate_detector_result(frame: Any, *, name: str) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"The WebGazer detector must return a Polars DataFrame for {name}, "
            f"got {type(frame)!r}."
        )
    missing = [column for column in _REQUIRED_EVENT_COLUMNS if column not in frame.columns]
    if missing and not frame.is_empty():
        raise ValueError(f"Detector {name} output is missing columns: {missing}.")
    return frame


def _run_detector_by_segment(
    detector_class: type,
    *,
    detection_algorithm: str,
    session_folder_path: Path,
    samples: pl.DataFrame,
    detector_config: Mapping[str, Any] | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    base_config = dict(detector_config or {})
    fixation_frames: list[pl.DataFrame] = []
    saccade_frames: list[pl.DataFrame] = []

    segments = samples.partition_by("segment_index", maintain_order=True)
    for segment in segments:
        if segment.height < 2:
            logger.warning(
                "Skipping WebGazer detector segment %s with fewer than two samples.",
                segment.get_column("segment_index")[0],
            )
            continue

        detector = detector_class(
            session_folder_path=session_folder_path,
            samples=segment,
        )
        config = dict(base_config)
        if detection_algorithm == "remodnav":
            # Low-rate webcam streams cannot reliably support the historical
            # 195 ms Savitzky-Golay default. Disable it unless explicitly set.
            rate = float(segment.get_column("Rate_recorded")[0])
            config.setdefault("savgol_length", 0.0)
            config.setdefault("max_pso_dur", 0.1)
            config.setdefault("lowpass_cutoff_freq", min(4.0, rate * 0.4))
        elif detection_algorithm == "engbert":
            rate = float(segment.get_column("Rate_recorded")[0])
            config.setdefault("sample_rate_fallback", rate)
        else:
            raise ValueError(
                f"WebGazer does not define a detector adapter for "
                f"{detection_algorithm!r}."
            )

        fixations, saccades = detector.detect_eye_movements(**config)
        fixation_frames.append(_validate_detector_result(fixations, name="fixations"))
        saccade_frames.append(_validate_detector_result(saccades, name="saccades"))

    empty_fixations, empty_saccades = _empty_events(detection_algorithm)
    fixations = (
        pl.concat(fixation_frames, how="diagonal_relaxed")
        if fixation_frames
        else empty_fixations
    )
    saccades = (
        pl.concat(saccade_frames, how="diagonal_relaxed")
        if saccade_frames
        else empty_saccades
    )
    if "tEnd" in fixations.columns:
        fixations = fixations.sort("tEnd")
    if "tEnd" in saccades.columns:
        saccades = saccades.sort("tEnd")
    return fixations, saccades


def _build_calibration(
    source: pl.DataFrame,
    *,
    calibration_type_column: str | None,
    calibration_value: str,
) -> pl.DataFrame:
    source_with_lines = (
        source.clone()
        if "line_number" in source.columns
        else source.with_row_index("line_number")
    )
    if calibration_type_column is None or calibration_type_column not in source.columns:
        return source_with_lines.head(0)
    return source_with_lines.filter(
        pl.col(calibration_type_column).cast(pl.String, strict=False)
        == calibration_value
    )


def _apply_requested_segmentation(
    preprocessing: PreProcessing,
    options: Mapping[str, Any],
) -> None:
    unsupported = sorted(_MESSAGE_SEGMENTATION_KEYS.intersection(options))
    if unsupported:
        raise ValueError(
            "This jsPsych WebGazer adapter does not extract a standardized message "
            "stream, so message-based trial segmentation is unavailable. Use "
            "explicit start_times/end_times or the existing trial_index column. "
            f"Received: {unsupported}."
        )

    have_start = "start_times" in options
    have_end = "end_times" in options
    if not have_start and not have_end:
        return
    if have_start != have_end:
        raise ValueError("Both start_times and end_times are required for segmentation.")

    method = preprocessing.split_all_into_trials
    allowed_parameters = set(inspect.signature(method).parameters)
    parameters = {
        key: value
        for key, value in options.items()
        if key in _SEGMENTATION_KEYS and key in allowed_parameters
    }
    preprocessing.process({"split_all_into_trials": parameters})


def _save_events_tsv(
    source: pl.DataFrame,
    *,
    output_path: Path,
    behavioral_columns: Sequence[str],
    row_time_column: str,
    trial_index_column: str,
) -> None:
    _require_columns(source, [row_time_column, trial_index_column], "events.tsv")
    events = source.filter(pl.col(row_time_column).is_not_null())
    if events.is_empty():
        return

    time_expression = pl.col(row_time_column).cast(pl.Float64, strict=False)
    onset_origin = events.select(time_expression.min()).item()
    events = events.with_columns(
        ((time_expression - float(onset_origin)) / 1000.0).round(4).alias("onset"),
        (
            (pl.col("rt").cast(pl.Float64, strict=False) / 1000.0).round(4)
            if "rt" in events.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("duration"),
        pl.col(trial_index_column).alias("trial_index"),
    )
    reserved = {"onset", "duration", "trial_index", trial_index_column}
    available = [
        column
        for column in behavioral_columns
        if column in events.columns and column not in reserved
    ]
    events.select(["onset", "duration", "trial_index", *available]).write_csv(
        output_path,
        separator="\t",
    )


def process_session(
    eye_tracking_data_path: Path,
    detection_algorithm: str,
    session_folder_path: Path,
    overwrite: bool,
    exp_format: str,
    **kwargs: Any,
) -> None:
    """Process the single jsPsych WebGazer CSV in a session directory."""
    csv_files = sorted(
        file
        for file in Path(eye_tracking_data_path).iterdir()
        if file.suffix.lower() == ".csv"
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No jsPsych WebGazer CSV found in {eye_tracking_data_path}."
        )
    if len(csv_files) > 1:
        logger.warning(
            "More than one CSV file found in %s; skipping the WebGazer session.",
            eye_tracking_data_path,
        )
        return

    Path(session_folder_path).mkdir(parents=True, exist_ok=True)
    WebGazerParse(session_folder_path, exp_format).parse(
        csv_files[0], detection_algorithm, overwrite, **kwargs
    )


class WebGazerParse(BidsParse):
    """Parse the documented jsPsych WebGazer trial-array schema into Polars."""

    def parse(
        self,
        file_path: Path,
        detection_algorithm: str,
        overwrite: bool,
        **kwargs: Any,
    ) -> pl.DataFrame:
        del overwrite  # Dataset-level orchestration handles overwrite behavior.
        self.session_folder_path.mkdir(parents=True, exist_ok=True)

        from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT

        try:
            detector_class = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm]
        except KeyError as exc:
            available = sorted(EYE_MOVEMENT_DETECTION_DICT)
            raise ValueError(
                f"Unknown detection algorithm {detection_algorithm!r}. "
                f"Available algorithms: {available}."
            ) from exc

        webgazer_data_column = str(kwargs.pop("webgazer_data_column", "webgazer_data"))
        row_time_column = str(kwargs.pop("row_time_column", "time_elapsed"))
        trial_index_column = str(kwargs.pop("trial_index_column", "trial_index"))
        calibration_type_column = kwargs.pop(
            "calibration_type_column", "rastoc-type"
        )
        if calibration_type_column is not None:
            calibration_type_column = str(calibration_type_column)
        calibration_value = str(
            kwargs.pop("calibration_value", "calibration-stimulus")
        )
        x_field = str(kwargs.pop("x_field", "x"))
        y_field = str(kwargs.pop("y_field", "y"))
        time_field = str(kwargs.pop("time_field", "t"))

        timestamp_mode = str(kwargs.pop("timestamp_mode", "trial_end"))
        sampling_policy = str(kwargs.pop("sampling_policy", "resample"))
        raw_target_rate = kwargs.pop("target_sample_rate", None)
        target_sample_rate = (
            None if raw_target_rate is None else float(raw_target_rate)
        )
        max_interval_deviation = float(
            kwargs.pop("max_interval_deviation", 0.20)
        )
        minimum_samples_per_trial = int(
            kwargs.pop("minimum_samples_per_trial", 2)
        )
        normalize_timestamps = bool(kwargs.pop("normalize_timestamps", True))
        behavioral_columns = kwargs.pop("behavioral_columns", None)
        screen_width = kwargs.pop("screen_width", None)
        screen_height = kwargs.pop("screen_height", None)
        raw_detector_config = kwargs.pop("detector_config", None)
        if raw_detector_config is not None and not isinstance(
            raw_detector_config, Mapping
        ):
            raise TypeError("detector_config must be a mapping or None.")
        detector_config = dict(raw_detector_config or {})

        source = _read_webgazer_csv(Path(file_path))
        _require_columns(
            source,
            [webgazer_data_column, row_time_column, trial_index_column],
            "jsPsych WebGazer CSV",
        )

        # Preserve backward compatibility with detector options historically
        # supplied directly to compute_derivatives_for_dataset().
        detector_parameters = set(
            inspect.signature(detector_class.detect_eye_movements).parameters
        ) - {"self"}
        for key in list(kwargs):
            if key in detector_parameters:
                detector_config[key] = kwargs.pop(key)

        # Translate shared screen metadata to each detector's API.
        if screen_width is not None:
            if detection_algorithm == "engbert":
                detector_config.setdefault("screen_width_px", int(screen_width))
            else:
                detector_config.setdefault("screen_width", int(screen_width))

        samples = _build_samples(
            source,
            webgazer_data_column=webgazer_data_column,
            row_time_column=row_time_column,
            trial_index_column=trial_index_column,
            x_field=x_field,
            y_field=y_field,
            time_field=time_field,
            timestamp_mode=timestamp_mode,
            sampling_policy=sampling_policy,
            target_sample_rate=target_sample_rate,
            max_interval_deviation=max_interval_deviation,
            minimum_samples_per_trial=minimum_samples_per_trial,
            normalize_timestamps=normalize_timestamps,
        )
        calibration = _build_calibration(
            source,
            calibration_type_column=calibration_type_column,
            calibration_value=calibration_value,
        )
        blinks = pl.DataFrame(schema=_BLINK_SCHEMA)
        messages = pl.DataFrame(schema=_MESSAGE_SCHEMA)

        fixations, saccades = _run_detector_by_segment(
            detector_class,
            detection_algorithm=detection_algorithm,
            session_folder_path=self.session_folder_path,
            samples=samples,
            detector_config=detector_config,
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
                pupil_unit="unavailable",
                screen_width=None if screen_width is None else int(screen_width),
                screen_height=None if screen_height is None else int(screen_height),
                extra={
                    "source_adapter": "jspsych-webgazer",
                    "sampling_policy": sampling_policy,
                    "timestamp_mode": timestamp_mode,
                },
            ),
        )
        _apply_requested_segmentation(preprocessing, kwargs)

        if behavioral_columns:
            if isinstance(behavioral_columns, str):
                behavioral_columns = [behavioral_columns]
            behavioral_source = source.with_columns(
                pl.col(trial_index_column).alias("trial_index")
            )
            preprocessing.add_trial_metadata(
                behavioral_source,
                list(behavioral_columns),
            )
            _save_events_tsv(
                source,
                output_path=self.session_folder_path / "events.tsv",
                behavioral_columns=list(behavioral_columns),
                row_time_column=row_time_column,
                trial_index_column=trial_index_column,
            )

        ignored_keys = {
            "prefer_durations",
            "case_insensitive",
            "use_regex",
            "return_match_token",
            "msg_keywords",
            # screen_height is stored in metadata; detectors currently use width.
        }
        unexpected = sorted(set(kwargs) - ignored_keys - _SEGMENTATION_KEYS)
        if unexpected:
            logger.warning("Ignoring unsupported WebGazer parser arguments: %s", unexpected)

        self.detection_algorithm = detection_algorithm
        self.store_dataframes(
            preprocessing.samples,
            dfCalib=calibration,
            dfFix=preprocessing.fixations,
            dfSacc=preprocessing.saccades,
            dfBlink=preprocessing.blinks,
            dfMsg=preprocessing.user_messages,
        )
        return source


def get_samples_for_remodnav(
    df_sample: pl.DataFrame,
    rate_recorded: float = 60.0,
    r_pupil: float | None = None,
    l_pupil: float | None = None,
) -> pl.DataFrame:
    """Compatibility helper adding the historical binocular REMoDNaV fields.

    New WebGazer parsing uses generic X/Y columns and does not fabricate pupil
    measurements. This helper is retained for external callers that still rely
    on the older shape.
    """
    pupil_left = float("nan") if l_pupil is None else float(l_pupil)
    pupil_right = float("nan") if r_pupil is None else float(r_pupil)
    return df_sample.with_columns(
        pl.lit(float(rate_recorded)).alias("Rate_recorded"),
        pl.col("X").alias("LX"),
        pl.col("X").alias("RX"),
        pl.col("Y").alias("LY"),
        pl.col("Y").alias("RY"),
        pl.lit(pupil_left).alias("LPupil"),
        pl.lit(pupil_right).alias("RPupil"),
        pl.lit(1).alias("Calib_index"),
        pl.lit("LR").alias("Eyes_recorded"),
    )
