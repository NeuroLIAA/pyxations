"""GazePoint CSV parser."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from pyxations.formats.generic import BidsParse

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("TIME", "BPOGX", "BPOGY", "LPD", "BKDUR")
_SEGMENT_COLUMNS = ("phase", "trial_number", "trial_label")
_SECONDS_TO_MILLISECONDS = 1000.0


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {frame.columns}"
        )


def _drop_unnamed_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop CSV index columns produced by prior pandas exports."""
    unnamed = [
        column
        for column in frame.columns
        if (
            not str(column).strip()
            or str(column).startswith("Unnamed:")
            or str(column) == "column_1"
        )
    ]
    return frame.drop(unnamed) if unnamed else frame


def _normalise_trial_mapping(
    value: Mapping[str, Sequence[Any]] | None,
    *,
    name: str,
) -> dict[str, list[Any]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping from phase names to sequences.")
    return {str(phase): list(entries) for phase, entries in value.items()}


def _validate_trial_intervals(
    start_times: Mapping[str, Sequence[float]],
    end_times: Mapping[str, Sequence[float]],
    trial_labels: Mapping[str, Sequence[str]] | None,
    *,
    allow_open_last: bool,
    require_nonoverlap: bool,
) -> dict[str, list[tuple[float, float, str]]]:
    """Validate trial intervals and return normalized phase specifications."""
    if set(start_times) != set(end_times):
        missing_end = sorted(set(start_times) - set(end_times))
        missing_start = sorted(set(end_times) - set(start_times))
        raise ValueError(
            "start_times and end_times must define the same phases. "
            f"Missing end phases: {missing_end}; missing start phases: {missing_start}."
        )

    normalized: dict[str, list[tuple[float, float, str]]] = {}
    for phase, raw_starts in start_times.items():
        starts = [float(value) for value in raw_starts]
        ends = [float(value) for value in end_times[phase]]

        if allow_open_last and len(starts) == len(ends) + 1:
            starts = starts[:-1]
        if len(starts) != len(ends):
            raise ValueError(
                f"[{phase}] start_times and end_times must have the same length, "
                f"got {len(starts)} and {len(ends)}."
            )

        labels = (
            [str(value) for value in trial_labels[phase]]
            if trial_labels is not None and phase in trial_labels
            else [""] * len(starts)
        )
        if len(labels) != len(starts):
            raise ValueError(
                f"[{phase}] Expected {len(starts)} trial labels, got {len(labels)}."
            )

        intervals: list[tuple[float, float, str]] = []
        previous_end: float | None = None
        for index, (start, end, label) in enumerate(zip(starts, ends, labels)):
            if start >= end:
                raise ValueError(
                    f"[{phase}] Trial {index} has a non-positive interval: "
                    f"start={start}, end={end}."
                )
            if require_nonoverlap and previous_end is not None and start < previous_end:
                raise ValueError(
                    f"[{phase}] Trial {index} starts at {start}, before the previous "
                    f"trial ended at {previous_end}."
                )
            intervals.append((start, end, label))
            previous_end = end
        normalized[phase] = intervals

    return normalized


def _segment_frame(
    frame: pl.DataFrame,
    intervals_by_phase: Mapping[str, Sequence[tuple[float, float, str]]],
    *,
    sample_table: bool,
) -> pl.DataFrame:
    """Assign phase and trial columns using millisecond interval boundaries."""
    required = ("tSample",) if sample_table else ("tStart", "tEnd")
    _require_columns(frame, required, "GazePoint trial segmentation")

    result = frame.with_columns(
        pl.lit("").alias("phase"),
        pl.lit(-1, dtype=pl.Int64).alias("trial_number"),
        pl.lit("").alias("trial_label"),
    )

    for phase, intervals in intervals_by_phase.items():
        for trial_number, (start, end, label) in enumerate(intervals):
            if sample_table:
                condition = pl.col("tSample").is_between(start, end, closed="both")
            else:
                condition = (pl.col("tStart") >= start) & (pl.col("tEnd") <= end)
            result = result.with_columns(
                pl.when(condition)
                .then(pl.lit(phase))
                .otherwise(pl.col("phase"))
                .alias("phase"),
                pl.when(condition)
                .then(pl.lit(trial_number, dtype=pl.Int64))
                .otherwise(pl.col("trial_number"))
                .alias("trial_number"),
                pl.when(condition)
                .then(pl.lit(label))
                .otherwise(pl.col("trial_label"))
                .alias("trial_label"),
            )
    return result


def _apply_trial_segmentation(
    samples: pl.DataFrame,
    fixations: pl.DataFrame,
    saccades: pl.DataFrame,
    blinks: pl.DataFrame,
    *,
    start_times: Mapping[str, Sequence[float]] | None,
    end_times: Mapping[str, Sequence[float]] | None,
    trial_labels: Mapping[str, Sequence[str]] | None,
    allow_open_last: bool,
    require_nonoverlap: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if start_times is None and end_times is None:
        return samples, fixations, saccades, blinks
    if start_times is None or end_times is None:
        raise ValueError("Both start_times and end_times are required for segmentation.")

    intervals = _validate_trial_intervals(
        start_times,
        end_times,
        trial_labels,
        allow_open_last=allow_open_last,
        require_nonoverlap=require_nonoverlap,
    )
    return (
        _segment_frame(samples, intervals, sample_table=True),
        _segment_frame(fixations, intervals, sample_table=False),
        _segment_frame(saccades, intervals, sample_table=False),
        _segment_frame(blinks, intervals, sample_table=False),
    )


def process_session(
    eye_tracking_data_path: Path,
    detection_algorithm: str,
    session_folder_path: Path,
    force_best_eye: bool,
    keep_ascii: bool,
    overwrite: bool,
    exp_format: str,
    **kwargs: Any,
) -> None:
    """Process the single GazePoint CSV in a session directory."""
    del force_best_eye, keep_ascii  # GazePoint exports one binocular gaze stream.

    csv_files = sorted(
        file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == ".csv"
    )
    if not csv_files:
        raise FileNotFoundError(f"No GazePoint CSV found in {eye_tracking_data_path}.")
    if len(csv_files) > 1:
        logger.warning(
            "More than one CSV file found in %s; skipping the session.",
            eye_tracking_data_path,
        )
        return

    session_folder_path.mkdir(parents=True, exist_ok=True)
    GazePointParse(session_folder_path, exp_format).parse(
        csv_files[0], detection_algorithm, overwrite, **kwargs
    )


class GazePointParse(BidsParse):
    """Parse a GazePoint CSV into Polars tables with millisecond timestamps."""

    def parse(
        self,
        file_path: Path,
        detection_algorithm: str,
        overwrite: bool,
        **kwargs: Any,
    ) -> None:
        del overwrite  # Overwrite is handled by the dataset-level orchestrator.

        from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT

        try:
            detector_class = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm]
        except KeyError as exc:
            available = sorted(EYE_MOVEMENT_DETECTION_DICT)
            raise ValueError(
                f"Unknown detection algorithm {detection_algorithm!r}. "
                f"Available algorithms: {available}."
            ) from exc

        source = _drop_unnamed_columns(pl.read_csv(file_path))
        _require_columns(source, _REQUIRED_COLUMNS, "GazePoint CSV")
        source = source.with_columns(
            (pl.col("TIME").cast(pl.Float64) * _SECONDS_TO_MILLISECONDS).alias("TIME"),
            (pl.col("BKDUR").cast(pl.Float64) * _SECONDS_TO_MILLISECONDS).alias("BKDUR"),
        )

        samples = (
            source.with_row_index("line_number")
            .rename(
                {
                    "TIME": "tSample",
                    "BPOGX": "X",
                    "BPOGY": "Y",
                    "LPD": "Pupil",
                }
            )
        )

        blinks = (
            source.with_row_index("line_number")
            .filter(pl.col("BKDUR") > 0)
            .rename({"TIME": "tEnd", "BKDUR": "duration"})
            .with_columns((pl.col("tEnd") - pl.col("duration")).alias("tStart"))
        )

        sample_rate = float(kwargs.pop("sample_rate", 60.0))
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")

        detector = detector_class(
            session_folder_path=self.session_folder_path,
            samples=samples,
        )
        self.detection_algorithm = detection_algorithm

        if detection_algorithm == "remodnav":
            config = {
                "savgol_length": 0.19,
                "max_pso_dur": 0.4,
                "pupil_data": samples["Pupil"].to_numpy(),
            }
            detector_config = kwargs.pop("detector_config", {})
            if detector_config is None:
                detector_config = {}
            if not isinstance(detector_config, Mapping):
                raise TypeError("detector_config must be a mapping.")
            config.update(detector_config)
            fixations, saccades = detector.run_eye_movement_from_samples(
                sample_rate,
                config=config,
                eye="Best",
            )
        elif detection_algorithm == "engbert":
            raw_detector_config = kwargs.pop("detector_config", {})
            if raw_detector_config is None:
                raw_detector_config = {}
            if not isinstance(raw_detector_config, Mapping):
                raise TypeError("detector_config must be a mapping.")
            detector_config = dict(raw_detector_config)
            detector_config.setdefault("sample_rate_fallback", sample_rate)
            fixations, saccades = detector.detect_eye_movements(**detector_config)
        else:  # Defensive: custom mappings should define an explicit adapter path.
            raise ValueError(
                f"GazePoint does not yet define an invocation adapter for "
                f"{detection_algorithm!r}."
            )

        start_times = _normalise_trial_mapping(
            kwargs.pop("start_times", None), name="start_times"
        )
        end_times = _normalise_trial_mapping(
            kwargs.pop("end_times", None), name="end_times"
        )
        trial_labels = _normalise_trial_mapping(
            kwargs.pop("trial_labels", None), name="trial_labels"
        )

        unsupported_message_keys = {
            "start_msgs",
            "end_msgs",
            "durations",
        }.intersection(kwargs)
        if unsupported_message_keys:
            raise ValueError(
                "GazePoint CSV parsing does not currently extract message events, so "
                "message-based trial segmentation is unavailable. Use explicit "
                "start_times/end_times instead. Received: "
                f"{sorted(unsupported_message_keys)}."
            )

        samples, fixations, saccades, blinks = _apply_trial_segmentation(
            samples,
            fixations,
            saccades,
            blinks,
            start_times=start_times,
            end_times=end_times,
            trial_labels=trial_labels,
            allow_open_last=bool(kwargs.pop("allow_open_last", True)),
            require_nonoverlap=bool(kwargs.pop("require_nonoverlap", True)),
        )

        ignored_keys = {
            "prefer_durations",
            "case_insensitive",
            "use_regex",
            "return_match_token",
            "behavioral_columns",
            "msg_keywords",
        }
        unexpected = sorted(set(kwargs) - ignored_keys)
        if unexpected:
            logger.warning("Ignoring unsupported GazePoint parser arguments: %s", unexpected)

        self.store_dataframes(
            samples,
            dfBlink=blinks,
            dfFix=fixations,
            dfSacc=saccades,
        )
