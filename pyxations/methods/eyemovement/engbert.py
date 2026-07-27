"""Engbert–Kliegl eye-movement detection adapter.

The detector performs its numerical work with NumPy and does not depend on
pandas.  Output tables use the same dataframe family as the input samples,
which preserves existing pandas callers while allowing parsers to migrate to
Polars independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pyxations.methods.eyemovement.eye_movement_detection import EyeMovementDetection

_SACCADE_COLUMNS = (
    "tStart",
    "tEnd",
    "duration",
    "xStart",
    "yStart",
    "xEnd",
    "yEnd",
    "ampDeg",
    "vPeak",
    "distDeg",
    "thetaDeg",
    "eye",
    "Calib_index",
    "Eyes_recorded",
    "Rate_recorded",
    "chunk",
)

_FIXATION_COLUMNS = (
    "tStart",
    "tEnd",
    "duration",
    "xAvg",
    "yAvg",
    "pupilAvg",
    "eye",
    "Calib_index",
    "Eyes_recorded",
    "Rate_recorded",
    "chunk",
)


def _column_names(frame: Any) -> list[str]:
    """Return dataframe column names without depending on its implementation."""
    return list(frame.columns)


def _column_to_numpy(
    frame: Any,
    name: str,
    *,
    dtype: Any | None = None,
    required: bool = True,
    default: Any = np.nan,
) -> np.ndarray:
    """Extract a dataframe column as a one-dimensional NumPy array."""
    columns = _column_names(frame)
    if name not in columns:
        if required:
            raise ValueError(
                f"Missing required sample column {name!r}. Available columns: {columns}"
            )
        return np.full(len(frame), default, dtype=dtype if dtype is not None else object)

    values = frame[name]
    if hasattr(values, "to_numpy"):
        array = values.to_numpy()
    else:
        array = np.asarray(values)

    array = np.asarray(array).reshape(-1)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _is_polars_frame(frame: Any) -> bool:
    return frame.__class__.__module__.split(".", 1)[0] == "polars"


def _make_frame(
    template: Any,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Any:
    """Create a dataframe matching ``template`` without importing pandas."""
    ordered_rows = [{column: row.get(column) for column in columns} for row in rows]

    if _is_polars_frame(template):
        import polars as pl

        if ordered_rows:
            return pl.DataFrame(ordered_rows).select(list(columns))
        return pl.DataFrame({column: [] for column in columns})

    frame_class = template.__class__
    try:
        return frame_class(ordered_rows, columns=list(columns))
    except TypeError as exc:
        raise TypeError(
            "Unsupported dataframe type. EngbertDetection expects a pandas or "
            f"Polars DataFrame, got {frame_class.__module__}.{frame_class.__name__}."
        ) from exc


def _smooth_1d(values: np.ndarray, smoothlevel: int) -> np.ndarray:
    """Smooth a one-dimensional signal with the historical Engbert kernels."""
    if smoothlevel == 0 or values.size < 3:
        return values
    if smoothlevel == 1:
        kernel = np.array([1.0, 1.0, 1.0]) / 3.0
    elif smoothlevel == 2:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    else:
        # Preserve the former behavior for unsupported levels: no smoothing.
        kernel = np.array([1.0])
    pad = (len(kernel) - 1) // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def vecvel(gaze_xy: np.ndarray, fs: float, smoothlevel: int = 1) -> np.ndarray:
    """Calculate two-dimensional gaze velocity in pixels per second."""
    gaze_xy = np.asarray(gaze_xy, dtype=float)
    if gaze_xy.ndim != 2 or gaze_xy.shape[1] != 2:
        raise ValueError("gaze_xy must have shape (n_samples, 2).")
    if gaze_xy.shape[0] < 2:
        return np.full_like(gaze_xy, np.nan, dtype=float)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and greater than zero.")

    x = _smooth_1d(gaze_xy[:, 0], smoothlevel)
    y = _smooth_1d(gaze_xy[:, 1], smoothlevel)

    vx = np.empty_like(x)
    vy = np.empty_like(y)
    vx[1:-1] = (x[2:] - x[:-2]) * (fs / 2.0)
    vy[1:-1] = (y[2:] - y[:-2]) * (fs / 2.0)
    vx[0] = (x[1] - x[0]) * fs
    vy[0] = (y[1] - y[0]) * fs
    vx[-1] = (x[-1] - x[-2]) * fs
    vy[-1] = (y[-1] - y[-2]) * fs
    vx[~np.isfinite(vx)] = np.nan
    vy[~np.isfinite(vy)] = np.nan
    return np.column_stack([vx, vy])


def _robust_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    median_squared = float(np.median(finite**2))
    variance = median_squared - median**2
    return float(np.sqrt(max(variance, 1e-12)))


def velthresh(vxy: np.ndarray) -> tuple[float, float]:
    """Return robust horizontal and vertical velocity thresholds."""
    return _robust_std(vxy[:, 0]), _robust_std(vxy[:, 1])


def _find_runs(mask: np.ndarray) -> np.ndarray:
    """Return inclusive start/end indices for contiguous true runs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty((0, 2), dtype=int)
    differences = np.diff(mask.astype(int))
    starts = np.where(differences == 1)[0] + 1
    ends = np.where(differences == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return np.column_stack([starts, ends]).astype(int)


def microsacc_plugin(
    pos_xy: np.ndarray,
    vel_xy: np.ndarray,
    vfac: float,
    mindur_samples: int,
    sdx: float,
    sdy: float,
) -> np.ndarray:
    """Return detected saccades using the Engbert velocity criterion.

    Columns are onset, offset, duration in samples, average velocity, peak
    velocity, travelled distance, angle, amplitude, direction, epoch, x0, y0,
    x1, and y1.
    """
    vx, vy = vel_xy[:, 0], vel_xy[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        criterion = (vx / sdx) ** 2 + (vy / sdy) ** 2
    runs = _find_runs(criterion > (vfac**2))

    saccades: list[list[float]] = []
    for start, end in runs:
        if (end - start + 1) < mindur_samples:
            continue
        segment_velocity = np.hypot(vx[start : end + 1], vy[start : end + 1])
        if not np.isfinite(segment_velocity).any():
            continue
        peak_velocity = float(np.nanmax(segment_velocity))
        average_velocity = float(np.nanmean(segment_velocity))

        x0, y0 = pos_xy[start, 0], pos_xy[start, 1]
        x1, y1 = pos_xy[end, 0], pos_xy[end, 1]
        amplitude = float(np.hypot(x1 - x0, y1 - y0))
        theta = float(np.arctan2(y1 - y0, x1 - x0))

        segment = pos_xy[start : end + 1]
        distance = float(
            np.nansum(np.hypot(np.diff(segment[:, 0]), np.diff(segment[:, 1])))
        )

        saccades.append(
            [
                float(start),
                float(end),
                float(end - start + 1),
                average_velocity,
                peak_velocity,
                distance,
                theta,
                amplitude,
                theta,
                np.nan,
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            ]
        )
    return np.asarray(saccades, dtype=float).reshape(-1, 14)


def _available_eye_columns(columns: Sequence[str]) -> dict[str, bool]:
    """Return the available left, right, and generic gaze streams."""
    column_set = set(columns)
    return {
        "has_L": {"LX", "LY"}.issubset(column_set),
        "has_R": {"RX", "RY"}.issubset(column_set),
        "has_generic": {"X", "Y"}.issubset(column_set),
    }


def _compute_px2deg(
    screen_size_cm: float,
    screen_distance_cm: float,
    screen_width_px: int,
) -> float:
    """Calculate visual degrees represented by one horizontal pixel."""
    if screen_size_cm <= 0 or screen_distance_cm <= 0 or screen_width_px <= 0:
        raise ValueError("Screen size, distance, and pixel width must be positive.")
    return float(
        np.degrees(np.arctan2(0.5 * screen_size_cm, screen_distance_cm))
        / (0.5 * screen_width_px)
    )


def _forward_backward_fill(values: np.ndarray) -> np.ndarray:
    """Fill missing numeric values using pandas-compatible ffill then bfill."""
    values = np.asarray(values, dtype=float).copy()
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return values

    first_valid = int(finite_indices[0])
    values[:first_valid] = values[first_valid]
    for index in range(first_valid + 1, values.size):
        if not np.isfinite(values[index]):
            values[index] = values[index - 1]
    return values


def _split_into_chunks(
    timestamps: np.ndarray,
    recorded_rates: np.ndarray | None,
    fallback_fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create chunks where sampling rate is stable and time gaps are reasonable."""
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if not np.isfinite(timestamps).all():
        raise ValueError("tSample values must be finite.")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("tSample values must be monotonically non-decreasing.")

    if recorded_rates is not None and np.isfinite(recorded_rates).any():
        sample_rates = _forward_backward_fill(recorded_rates)
    else:
        if fallback_fs is None or not np.isfinite(fallback_fs) or fallback_fs <= 0:
            raise ValueError(
                "Rate_recorded is unavailable and no valid sample_rate_fallback was provided."
            )
        sample_rates = np.full(timestamps.size, float(fallback_fs))

    if sample_rates.size != timestamps.size:
        raise ValueError("Rate_recorded and tSample must contain the same number of rows.")
    if not np.isfinite(sample_rates).all() or np.any(sample_rates <= 0):
        raise ValueError("Sampling rates must be finite and greater than zero.")

    chunk_breaks = np.zeros(timestamps.size, dtype=int)
    for index in range(1, timestamps.size):
        rate_changed = not np.isclose(
            sample_rates[index], sample_rates[index - 1], rtol=0, atol=1e-6
        )
        expected_interval = 1000.0 / sample_rates[index - 1]
        observed_interval = timestamps[index] - timestamps[index - 1]
        large_gap = observed_interval > (1.5 * expected_interval)
        if rate_changed or large_gap:
            chunk_breaks[index] = 1

    return np.cumsum(chunk_breaks), sample_rates


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.mean(values[finite]))


def _fallback_sigma(values: np.ndarray) -> float:
    """Return a finite nonzero velocity scale for degenerate recordings."""
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    median = float(np.median(finite))
    return median if median > 0 else 1.0


class EngbertDetection(EyeMovementDetection):
    """Detect saccades and inter-saccadic fixations with Engbert–Kliegl."""

    def __init__(self, session_folder_path: Any, samples: Any):
        self.session_folder_path = session_folder_path
        self.out_folder = session_folder_path / "engbert_events"
        self.samples = samples

    def detect_eye_movements(
        self,
        vfac: float = 5.0,
        mindur_ms: float = 6.0,
        smoothlevel: int = 1,
        globalthresh: bool = True,
        degperpixel: float | None = None,
        screen_size_cm: float = 38.0,
        screen_width_px: int = 1920,
        screen_distance_cm: float = 60.0,
        sample_rate_fallback: float | None = None,
    ) -> tuple[Any, Any]:
        """Detect fixations and saccades, returning times in milliseconds.

        The returned dataframe type matches ``self.samples``.  Gaze samples may
        contain left/right columns (``LX``, ``LY``, ``RX``, ``RY``) or generic
        columns (``X``, ``Y``).  Pupil measurements are summarized when a
        corresponding pupil column is available; otherwise ``pupilAvg`` is NaN.
        """
        if not np.isfinite(vfac) or vfac <= 0:
            raise ValueError("vfac must be finite and greater than zero.")
        if not np.isfinite(mindur_ms) or mindur_ms < 0:
            raise ValueError("mindur_ms must be finite and non-negative.")
        if not isinstance(smoothlevel, int) or smoothlevel < 0:
            raise ValueError("smoothlevel must be a non-negative integer.")

        columns = _column_names(self.samples)
        timestamps = _column_to_numpy(self.samples, "tSample", dtype=float)
        if timestamps.size == 0:
            return (
                _make_frame(self.samples, [], _FIXATION_COLUMNS),
                _make_frame(self.samples, [], _SACCADE_COLUMNS),
            )

        if degperpixel is None:
            degperpixel = _compute_px2deg(
                screen_size_cm, screen_distance_cm, screen_width_px
            )
        elif not np.isfinite(degperpixel) or degperpixel <= 0:
            raise ValueError("degperpixel must be finite and greater than zero.")

        recorded_rates = (
            _column_to_numpy(self.samples, "Rate_recorded", dtype=float)
            if "Rate_recorded" in columns
            else None
        )
        chunk_ids, sample_rates = _split_into_chunks(
            timestamps,
            recorded_rates,
            fallback_fs=sample_rate_fallback,
        )

        calibration = _column_to_numpy(
            self.samples, "Calib_index", required=False, default=np.nan
        )
        eyes_recorded = _column_to_numpy(
            self.samples, "Eyes_recorded", required=False, default=np.nan
        )

        eye_columns = _available_eye_columns(columns)
        coordinate_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        if eye_columns["has_L"]:
            coordinate_arrays["L"] = (
                _column_to_numpy(self.samples, "LX", dtype=float),
                _column_to_numpy(self.samples, "LY", dtype=float),
                _column_to_numpy(self.samples, "LPupil", dtype=float)
                if "LPupil" in columns
                else None,
            )
        if eye_columns["has_R"]:
            coordinate_arrays["R"] = (
                _column_to_numpy(self.samples, "RX", dtype=float),
                _column_to_numpy(self.samples, "RY", dtype=float),
                _column_to_numpy(self.samples, "RPupil", dtype=float)
                if "RPupil" in columns
                else None,
            )
        if eye_columns["has_generic"]:
            coordinate_arrays["U"] = (
                _column_to_numpy(self.samples, "X", dtype=float),
                _column_to_numpy(self.samples, "Y", dtype=float),
                _column_to_numpy(self.samples, "Pupil", dtype=float)
                if "Pupil" in columns
                else None,
            )

        if not coordinate_arrays:
            raise ValueError(
                "No supported gaze coordinate columns found. Expected LX/LY, "
                "RX/RY, or X/Y."
            )

        saccade_records: list[dict[str, Any]] = []
        fixation_records: list[dict[str, Any]] = []

        for chunk_id in np.unique(chunk_ids):
            indices = np.flatnonzero(chunk_ids == chunk_id)
            if indices.size == 0:
                continue

            sample_rate = float(sample_rates[indices[0]])
            chunk_start_ms = float(timestamps[indices[0]])
            calib_value = calibration[indices[0]]
            eyes_value = eyes_recorded[indices[0]]

            streams: list[tuple[str, np.ndarray, np.ndarray | None]] = []
            for eye_label in ("L", "R"):
                if eye_label not in coordinate_arrays:
                    continue
                x_values, y_values, pupil_values = coordinate_arrays[eye_label]
                xy = np.column_stack([x_values[indices], y_values[indices]])
                if np.isfinite(xy).any():
                    streams.append(
                        (
                            eye_label,
                            xy,
                            pupil_values[indices] if pupil_values is not None else None,
                        )
                    )

            # Preserve the historical preference for explicit left/right streams.
            if not streams and "U" in coordinate_arrays:
                x_values, y_values, pupil_values = coordinate_arrays["U"]
                xy = np.column_stack([x_values[indices], y_values[indices]])
                if np.isfinite(xy).any():
                    streams.append(
                        (
                            "U",
                            xy,
                            pupil_values[indices] if pupil_values is not None else None,
                        )
                    )

            if not streams:
                continue

            global_sigmas: dict[str, tuple[float, float]] = {}
            if globalthresh:
                for eye_label, xy, _ in streams:
                    global_sigmas[eye_label] = velthresh(
                        vecvel(xy, sample_rate, smoothlevel=smoothlevel)
                    )

            minimum_samples = max(
                1, int(round(mindur_ms * sample_rate / 1000.0))
            )

            for eye_label, xy, pupil_values in streams:
                valid_coordinates = np.isfinite(xy).all(axis=1)
                if not valid_coordinates.any():
                    continue

                velocity = vecvel(xy, sample_rate, smoothlevel=smoothlevel)
                if globalthresh:
                    sigma_x, sigma_y = global_sigmas[eye_label]
                else:
                    sigma_x, sigma_y = velthresh(velocity)

                if not np.isfinite(sigma_x) or sigma_x <= 1e-6:
                    sigma_x = _fallback_sigma(velocity[:, 0])
                if not np.isfinite(sigma_y) or sigma_y <= 1e-6:
                    sigma_y = _fallback_sigma(velocity[:, 1])

                saccades = microsacc_plugin(
                    xy,
                    velocity,
                    vfac=vfac,
                    mindur_samples=minimum_samples,
                    sdx=sigma_x,
                    sdy=sigma_y,
                )

                def append_fixation(start_index: int, end_index: int) -> None:
                    if end_index < start_index:
                        return
                    segment = xy[start_index : end_index + 1]
                    finite_segment = np.isfinite(segment).all(axis=1)
                    if not finite_segment.any():
                        return

                    pupil_average = (
                        _nanmean_or_nan(pupil_values[start_index : end_index + 1])
                        if pupil_values is not None
                        else float("nan")
                    )
                    fixation_records.append(
                        {
                            "tStart": chunk_start_ms
                            + (start_index / sample_rate) * 1000.0,
                            "tEnd": chunk_start_ms
                            + (end_index / sample_rate) * 1000.0,
                            "duration": (end_index - start_index + 1)
                            / sample_rate
                            * 1000.0,
                            "xAvg": _nanmean_or_nan(segment[:, 0]),
                            "yAvg": _nanmean_or_nan(segment[:, 1]),
                            "pupilAvg": pupil_average,
                            "eye": eye_label,
                            "Calib_index": calib_value,
                            "Eyes_recorded": eyes_value,
                            "Rate_recorded": sample_rate,
                            "chunk": int(chunk_id),
                        }
                    )

                if saccades.size == 0:
                    valid_indices = np.flatnonzero(valid_coordinates)
                    append_fixation(int(valid_indices[0]), int(valid_indices[-1]))
                    continue

                onset_indices = saccades[:, 0].astype(int)
                offset_indices = saccades[:, 1].astype(int)
                start_times = chunk_start_ms + (onset_indices / sample_rate) * 1000.0
                end_times = chunk_start_ms + (offset_indices / sample_rate) * 1000.0
                durations = (saccades[:, 2] / sample_rate) * 1000.0

                for event_index in range(saccades.shape[0]):
                    saccade_records.append(
                        {
                            "tStart": float(start_times[event_index]),
                            "tEnd": float(end_times[event_index]),
                            "duration": float(durations[event_index]),
                            "xStart": float(saccades[event_index, 10]),
                            "yStart": float(saccades[event_index, 11]),
                            "xEnd": float(saccades[event_index, 12]),
                            "yEnd": float(saccades[event_index, 13]),
                            "ampDeg": float(saccades[event_index, 7] * degperpixel),
                            "vPeak": float(saccades[event_index, 4] * degperpixel),
                            "distDeg": float(saccades[event_index, 5] * degperpixel),
                            "thetaDeg": float(
                                saccades[event_index, 6] * (180.0 / np.pi)
                            ),
                            "eye": eye_label,
                            "Calib_index": calib_value,
                            "Eyes_recorded": eyes_value,
                            "Rate_recorded": sample_rate,
                            "chunk": int(chunk_id),
                        }
                    )

                order = np.argsort(onset_indices)
                sorted_onsets = onset_indices[order]
                sorted_offsets = offset_indices[order]

                if sorted_onsets[0] > 0:
                    append_fixation(0, int(sorted_onsets[0] - 1))

                for event_index in range(len(sorted_onsets) - 1):
                    append_fixation(
                        int(sorted_offsets[event_index] + 1),
                        int(sorted_onsets[event_index + 1] - 1),
                    )

                last_offset = int(sorted_offsets[-1])
                if last_offset < (indices.size - 1):
                    append_fixation(last_offset + 1, indices.size - 1)

        fixation_records.sort(key=lambda row: row["tEnd"])
        saccade_records.sort(key=lambda row: row["tEnd"])
        return (
            _make_frame(self.samples, fixation_records, _FIXATION_COLUMNS),
            _make_frame(self.samples, saccade_records, _SACCADE_COLUMNS),
        )
