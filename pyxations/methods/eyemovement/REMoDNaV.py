"""Polars-native REMoDNaV eye-movement detection adapter."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl
from remodnav.clf import EyegazeClassifier

from pyxations.methods.eyemovement.eye_movement_detection import EyeMovementDetection

logger = logging.getLogger(__name__)

_EVENT_COLUMNS = (
    "start_time",
    "end_time",
    "label",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "amp",
    "peak_vel",
    "med_vel",
    "avg_vel",
)

_BASE_OUTPUT_COLUMNS = (
    "tStart",
    "tEnd",
    "xStart",
    "yStart",
    "xEnd",
    "yEnd",
    "ampDeg",
    "vPeak",
    "med_vel",
    "avg_vel",
    "duration",
)

_FIXATION_OUTPUT_COLUMNS = _BASE_OUTPUT_COLUMNS + (
    "xAvg",
    "yAvg",
    "pupilAvg",
    "Calib_index",
    "Eyes_recorded",
    "Rate_recorded",
    "eye",
)

_SACCADE_OUTPUT_COLUMNS = _BASE_OUTPUT_COLUMNS + (
    "Calib_index",
    "Eyes_recorded",
    "Rate_recorded",
    "eye",
)


def _column_names(frame: pl.DataFrame) -> list[str]:
    """Return dataframe column names."""
    return list(frame.columns)


def _column_to_numpy(
    frame: pl.DataFrame, name: str, *, dtype: Any | None = None
) -> np.ndarray:
    """Extract a dataframe column as a one-dimensional NumPy array."""
    if name not in _column_names(frame):
        raise ValueError(
            f"Missing required sample column {name!r}. "
            f"Available columns: {_column_names(frame)}"
        )

    values = frame[name]
    if hasattr(values, "to_numpy"):
        array = values.to_numpy()
    else:
        array = np.asarray(values)

    array = np.asarray(array).reshape(-1)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _make_frame(
    template: pl.DataFrame,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> pl.DataFrame:
    """Create an ordered Polars dataframe."""
    if not isinstance(template, pl.DataFrame):
        raise TypeError(
            "REMoDNaV expects a Polars DataFrame, "
            f"got {type(template)!r}."
        )
    ordered_rows = [{column: row.get(column) for column in columns} for row in rows]
    if ordered_rows:
        return pl.DataFrame(ordered_rows, strict=False).select(list(columns))
    return pl.DataFrame({column: [] for column in columns})


def _frame_to_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Convert a Polars dataframe into row dictionaries."""
    return frame.to_dicts()


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.mean(values[finite]))


def _normalise_remodnav_events(events: Any) -> list[dict[str, Any]]:
    """Normalize REMoDNaV event output into dictionaries.

    REMoDNaV versions have returned tuple-like rows, mappings, named tuples, and
    structured NumPy records.  Supporting all of these here keeps the remaining
    detector logic independent of the precise container used by REMoDNaV.
    """
    records: list[dict[str, Any]] = []

    for event in events:
        if isinstance(event, Mapping):
            raw = dict(event)
        elif hasattr(event, "_asdict"):
            raw = dict(event._asdict())
        elif isinstance(event, np.void) and event.dtype.names:
            raw = {name: event[name] for name in event.dtype.names}
        else:
            values = list(event)
            if len(values) != len(_EVENT_COLUMNS):
                raise ValueError(
                    "Unexpected REMoDNaV event shape: expected "
                    f"{len(_EVENT_COLUMNS)} fields, received {len(values)}."
                )
            raw = dict(zip(_EVENT_COLUMNS, values))

        missing = [column for column in _EVENT_COLUMNS if column not in raw]
        if missing:
            raise ValueError(f"REMoDNaV event is missing fields: {missing}")

        record = {column: raw[column] for column in _EVENT_COLUMNS}
        label = record["label"]
        record["label"] = label.decode() if isinstance(label, bytes) else str(label)
        for column in _EVENT_COLUMNS:
            if column != "label":
                record[column] = float(record[column])
        records.append(record)

    return records


def _validate_constant(values: np.ndarray, name: str) -> Any:
    """Return the first value and reject chunks with changing metadata."""
    if values.size == 0:
        raise ValueError(f"Cannot detect events in an empty chunk ({name}).")

    first = values[0]

    def values_equal(left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is right
        try:
            if bool(np.isnan(left)) and bool(np.isnan(right)):
                return True
        except (TypeError, ValueError):
            pass
        return bool(left == right)

    if not all(values_equal(value, first) for value in values):
        raise ValueError(f"{name} must be constant within each continuous sample chunk.")
    return first


class RemodnavDetection(EyeMovementDetection):
    """Detect fixations and saccades with REMoDNaV."""

    def __init__(self, session_folder_path: Any, samples: Any):
        self.session_folder_path = session_folder_path
        self.out_folder = session_folder_path / "remodnav_events"
        self.samples = samples

    def detect_eye_movements(
        self,
        min_pursuit_dur: float = 10.0,
        max_pso_dur: float = 0.0,
        min_fix_dur: float = 0.05,
        sac_max_vel: float = 1000.0,
        fix_max_amp: float = 1.5,
        sac_time_thresh: float = 0.002,
        drop_fix_from_blink: bool = True,
        screen_size: float = 38.0,
        screen_width: int = 1920,
        screen_distance: float = 60.0,
        savgol_length: float = 0.195,
        lowpass_cutoff_freq: float | None = None,
    ) -> tuple[Any, Any]:
        """Detect eye movements in all continuous chunks and recorded eyes.

        Input timestamps are expected in milliseconds.  The returned dataframe
        type matches ``self.samples``.
        """
        self.out_folder.mkdir(parents=True, exist_ok=True)

        timestamps = _column_to_numpy(self.samples, "tSample", dtype=float)
        sample_rates = _column_to_numpy(self.samples, "Rate_recorded", dtype=float)
        if timestamps.size == 0:
            return (
                _make_frame(self.samples, [], _FIXATION_OUTPUT_COLUMNS),
                _make_frame(self.samples, [], _SACCADE_OUTPUT_COLUMNS),
            )
        if timestamps.size != sample_rates.size:
            raise ValueError("tSample and Rate_recorded must contain the same number of rows.")
        if not np.isfinite(sample_rates).all() or np.any(sample_rates <= 0):
            raise ValueError("Rate_recorded values must be finite and greater than zero.")

        # Preserve the existing discontinuity rule while applying the expected
        # interval from the preceding sample when the rate changes.
        expected_intervals = 1000.0 / sample_rates[:-1]
        chunk_starts = np.flatnonzero(np.diff(timestamps) > expected_intervals) + 1
        chunk_indices = np.split(np.arange(timestamps.size), chunk_starts)

        fixation_records: list[dict[str, Any]] = []
        saccade_records: list[dict[str, Any]] = []

        for indices in chunk_indices:
            fixations, saccades = self.detect_on_chunk(
                indices,
                min_pursuit_dur=min_pursuit_dur,
                max_pso_dur=max_pso_dur,
                min_fix_dur=min_fix_dur,
                sac_max_vel=sac_max_vel,
                fix_max_amp=fix_max_amp,
                sac_time_thresh=sac_time_thresh,
                drop_fix_from_blink=drop_fix_from_blink,
                screen_size=screen_size,
                screen_width=screen_width,
                screen_distance=screen_distance,
                savgol_length=savgol_length,
                lowpass_cutoff_freq=lowpass_cutoff_freq,
            )
            fixation_records.extend(_frame_to_records(fixations))
            saccade_records.extend(_frame_to_records(saccades))

        fixation_records.sort(key=lambda row: row["tEnd"])
        saccade_records.sort(key=lambda row: row["tEnd"])
        return (
            _make_frame(self.samples, fixation_records, _FIXATION_OUTPUT_COLUMNS),
            _make_frame(self.samples, saccade_records, _SACCADE_OUTPUT_COLUMNS),
        )

    def run_eye_movement_from_samples(
        self,
        sample_rate: float,
        x_label: str = "X",
        y_label: str = "Y",
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Run REMoDNaV on two columns in ``self.samples``.

        Missing pupil measurements remain missing (NaN); they are no longer
        represented by synthetic zeros.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        options = dict(config or {})
        gazex_data = _column_to_numpy(self.samples, x_label, dtype=float)
        gazey_data = _column_to_numpy(self.samples, y_label, dtype=float)
        timestamps = _column_to_numpy(self.samples, "tSample", dtype=float)
        if not (gazex_data.size == gazey_data.size == timestamps.size):
            raise ValueError("Gaze coordinates and timestamps must have equal lengths.")

        starting_time = float(np.nanmin(timestamps))
        pupil_data = options.pop("pupil_data", None)
        if pupil_data is None:
            pupil_data = np.full(gazex_data.size, np.nan, dtype=float)

        times = np.arange(gazex_data.size, dtype=float) / float(sample_rate)
        return self.run_eye_movement(
            gazex_data,
            gazey_data,
            sample_rate,
            times=times,
            starting_time=starting_time,
            pupil_data=pupil_data,
            **options,
            **kwargs,
        )

    def run_eye_movement(
        self,
        gazex_data: Any,
        gazey_data: Any,
        sample_rate: float,
        min_pursuit_dur: float = 10.0,
        max_pso_dur: float = 0.0,
        min_fix_dur: float = 0.05,
        min_saccade_duration: float = 0.04,
        sac_max_vel: float = 1000.0,
        fix_max_amp: float = 1.5,
        sac_time_thresh: float = 0.002,
        drop_fix_from_blink: bool = True,
        screen_size: float = 38.0,
        screen_width: int = 1920,
        screen_distance: float = 60.0,
        calib_index: Any = 0,
        savgol_length: float = 0.19,
        eyes_recorded: Any = None,
        starting_time: float | None = None,
        times: Any = None,
        pupil_data: Any = None,
        eye: Any = None,
        lowpass_cutoff_freq: float | None = None,
    ) -> tuple[Any, Any]:
        """Run REMoDNaV for one eye stream and return event tables."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")
        if screen_size <= 0 or screen_width <= 0 or screen_distance <= 0:
            raise ValueError("Screen size, width, and viewing distance must be positive.")

        gaze_x = np.asarray(gazex_data, dtype=float).reshape(-1)
        gaze_y = np.asarray(gazey_data, dtype=float).reshape(-1)
        if gaze_x.size != gaze_y.size:
            raise ValueError("gazex_data and gazey_data must have equal lengths.")

        if times is None:
            sample_times = np.arange(gaze_x.size, dtype=float) / float(sample_rate)
        else:
            sample_times = np.asarray(times, dtype=float).reshape(-1)
        if sample_times.size != gaze_x.size:
            raise ValueError("times must have the same length as the gaze arrays.")

        if pupil_data is None:
            pupil = np.full(gaze_x.size, np.nan, dtype=float)
        else:
            pupil = np.asarray(pupil_data, dtype=float).reshape(-1)
        if pupil.size != gaze_x.size:
            raise ValueError("pupil_data must have the same length as the gaze arrays.")

        time_offset_ms = 0.0 if starting_time is None else float(starting_time)
        eye_data = np.rec.fromarrays((gaze_x, gaze_y), names=("x", "y"))
        px2deg = math.degrees(math.atan2(0.5 * screen_size, screen_distance)) / (
            0.5 * screen_width
        )

        if lowpass_cutoff_freq is None:
            resolved_lowpass_cutoff = min(4.0, sample_rate * 0.4)
        else:
            resolved_lowpass_cutoff = float(lowpass_cutoff_freq)
        if (
            not np.isfinite(resolved_lowpass_cutoff)
            or resolved_lowpass_cutoff <= 0
            or resolved_lowpass_cutoff >= sample_rate / 2.0
        ):
            raise ValueError(
                "lowpass_cutoff_freq must be finite, greater than zero, and "
                "below the Nyquist frequency (sample_rate / 2)."
            )

        logger.info("Running REMoDNaV detection for %s eye", eye)
        classifier = EyegazeClassifier(
            px2deg=px2deg,
            sampling_rate=sample_rate,
            min_pursuit_duration=min_pursuit_dur,
            max_pso_duration=max_pso_dur,
            min_fixation_duration=min_fix_dur,
            min_saccade_duration=min_saccade_duration,
            lowpass_cutoff_freq=resolved_lowpass_cutoff,
        )
        preprocessed = classifier.preproc(eye_data, savgol_length=savgol_length)
        events = _normalise_remodnav_events(
            classifier(preprocessed, classify_isp=True, sort_events=True)
        )

        fixation_events = [event for event in events if event["label"] == "FIXA"]
        saccade_events = [
            event for event in events if event["label"] in {"SACC", "ISAC"}
        ]

        filtered_fixations = [
            event for event in fixation_events if event["amp"] <= fix_max_amp
        ]
        filtered_saccades = [
            event for event in saccade_events if event["peak_vel"] <= sac_max_vel
        ]
        # Preserve the former start-x ordering before the optional adjacency filter.
        filtered_fixations.sort(key=lambda event: event["start_x"])
        filtered_saccades.sort(key=lambda event: event["start_x"])

        logger.info(
            "Kept %d/%d fixations and %d/%d saccades after amplitude/velocity filtering",
            len(filtered_fixations),
            len(fixation_events),
            len(filtered_saccades),
            len(saccade_events),
        )

        if drop_fix_from_blink and filtered_fixations:
            saccade_ends = np.asarray(
                [event["end_time"] for event in filtered_saccades], dtype=float
            )
            filtered_fixations = [
                fixation
                for fixation in filtered_fixations
                if saccade_ends.size
                and np.any(
                    (saccade_ends > fixation["start_time"] - sac_time_thresh)
                    & (saccade_ends < fixation["start_time"] + sac_time_thresh)
                )
            ]

        fixation_rows: list[dict[str, Any]] = []
        for event in filtered_fixations:
            within = (sample_times > event["start_time"]) & (
                sample_times < event["end_time"]
            )
            row = self._event_row(
                event,
                time_offset_ms=time_offset_ms,
                sample_rate=sample_rate,
                calib_index=calib_index,
                eyes_recorded=eyes_recorded,
                eye=eye,
            )
            row.update(
                xAvg=_nanmean_or_nan(gaze_x[within]),
                yAvg=_nanmean_or_nan(gaze_y[within]),
                pupilAvg=_nanmean_or_nan(pupil[within]),
            )
            fixation_rows.append(row)

        saccade_rows = [
            self._event_row(
                event,
                time_offset_ms=time_offset_ms,
                sample_rate=sample_rate,
                calib_index=calib_index,
                eyes_recorded=eyes_recorded,
                eye=eye,
            )
            for event in filtered_saccades
        ]

        return (
            _make_frame(self.samples, fixation_rows, _FIXATION_OUTPUT_COLUMNS),
            _make_frame(self.samples, saccade_rows, _SACCADE_OUTPUT_COLUMNS),
        )

    @staticmethod
    def _event_row(
        event: Mapping[str, Any],
        *,
        time_offset_ms: float,
        sample_rate: float,
        calib_index: Any,
        eyes_recorded: Any,
        eye: Any,
    ) -> dict[str, Any]:
        return {
            "tStart": event["start_time"] * 1000.0 + time_offset_ms,
            "tEnd": event["end_time"] * 1000.0 + time_offset_ms,
            "xStart": event["start_x"],
            "yStart": event["start_y"],
            "xEnd": event["end_x"],
            "yEnd": event["end_y"],
            "ampDeg": event["amp"],
            "vPeak": event["peak_vel"],
            "med_vel": event["med_vel"],
            "avg_vel": event["avg_vel"],
            "duration": (event["end_time"] - event["start_time"]) * 1000.0,
            "Calib_index": calib_index,
            "Eyes_recorded": eyes_recorded,
            "Rate_recorded": sample_rate,
            "eye": eye,
        }

    def detect_on_chunk(
        self,
        indices: np.ndarray,
        min_pursuit_dur: float = 10.0,
        max_pso_dur: float = 0.0,
        min_fix_dur: float = 0.05,
        sac_max_vel: float = 1000.0,
        fix_max_amp: float = 1.5,
        sac_time_thresh: float = 0.002,
        drop_fix_from_blink: bool = True,
        screen_size: float = 38.0,
        screen_width: int = 1920,
        screen_distance: float = 60.0,
        savgol_length: float = 0.19,
        lowpass_cutoff_freq: float | None = None,
    ) -> tuple[Any, Any]:
        """Detect events for a continuous set of sample row indices."""
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return (
                _make_frame(self.samples, [], _FIXATION_OUTPUT_COLUMNS),
                _make_frame(self.samples, [], _SACCADE_OUTPUT_COLUMNS),
            )

        rates = _column_to_numpy(self.samples, "Rate_recorded", dtype=float)[indices]
        calib_values = _column_to_numpy(self.samples, "Calib_index")[indices]
        eyes_values = _column_to_numpy(self.samples, "Eyes_recorded")[indices]
        timestamps = _column_to_numpy(self.samples, "tSample", dtype=float)[indices]

        sample_rate = float(_validate_constant(rates, "Rate_recorded"))
        calib_index = _validate_constant(calib_values, "Calib_index")
        eyes_recorded = _validate_constant(eyes_values, "Eyes_recorded")
        starting_time = float(timestamps[0])
        times = np.arange(indices.size, dtype=float) / sample_rate

        columns = set(_column_names(self.samples))
        streams: list[tuple[str, str, str, str | None]] = []
        if {"LX", "LY"}.issubset(columns):
            streams.append(("L", "LX", "LY", "LPupil" if "LPupil" in columns else None))
        if {"RX", "RY"}.issubset(columns):
            streams.append(("R", "RX", "RY", "RPupil" if "RPupil" in columns else None))
        if not streams and {"X", "Y"}.issubset(columns):
            streams.append(("U", "X", "Y", "Pupil" if "Pupil" in columns else None))
        if not streams:
            raise ValueError(
                "Samples must contain LX/LY, RX/RY, or generic X/Y gaze columns."
            )

        fixation_records: list[dict[str, Any]] = []
        saccade_records: list[dict[str, Any]] = []

        for eye, x_column, y_column, pupil_column in streams:
            gaze_x = _column_to_numpy(self.samples, x_column, dtype=float)[indices]
            gaze_y = _column_to_numpy(self.samples, y_column, dtype=float)[indices]
            if not np.isfinite(gaze_x).any() or not np.isfinite(gaze_y).any():
                continue

            pupil = (
                _column_to_numpy(self.samples, pupil_column, dtype=float)[indices]
                if pupil_column is not None
                else np.full(indices.size, np.nan, dtype=float)
            )
            fixations, saccades = self.run_eye_movement(
                gaze_x,
                gaze_y,
                sample_rate,
                min_pursuit_dur=min_pursuit_dur,
                max_pso_dur=max_pso_dur,
                min_fix_dur=min_fix_dur,
                min_saccade_duration=0.04,
                sac_max_vel=sac_max_vel,
                fix_max_amp=fix_max_amp,
                sac_time_thresh=sac_time_thresh,
                drop_fix_from_blink=drop_fix_from_blink,
                screen_size=screen_size,
                screen_width=screen_width,
                screen_distance=screen_distance,
                calib_index=calib_index,
                savgol_length=savgol_length,
                eyes_recorded=eyes_recorded,
                starting_time=starting_time,
                times=times,
                pupil_data=pupil,
                eye=eye,
                lowpass_cutoff_freq=lowpass_cutoff_freq,
            )
            fixation_records.extend(_frame_to_records(fixations))
            saccade_records.extend(_frame_to_records(saccades))

        return (
            _make_frame(self.samples, fixation_records, _FIXATION_OUTPUT_COLUMNS),
            _make_frame(self.samples, saccade_records, _SACCADE_OUTPUT_COLUMNS),
        )
