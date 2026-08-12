from __future__ import annotations

import json
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

Number = int | float
PathLike = str | Path
DataFrame = pl.DataFrame

_X_COORDINATE_COLUMNS = ("LX", "RX", "X", "xStart", "xEnd", "xAvg")
_Y_COORDINATE_COLUMNS = ("LY", "RY", "Y", "yStart", "yEnd", "yAvg")
_COORDINATE_COLUMNS = _X_COORDINATE_COLUMNS + _Y_COORDINATE_COLUMNS


@dataclass
class SessionMetadata:
    """Lightweight metadata saved alongside a processed recording."""

    coords_unit: str = "px"
    time_unit: str = "ms"
    pupil_unit: str = "arbitrary"
    screen_width: int | None = None
    screen_height: int | None = None
    extra: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return JSON-serializable metadata.

        Returns
        -------
        dict
            Recording units, screen dimensions, and additional metadata.
        """
        return {
            "coords_unit": self.coords_unit,
            "time_unit": self.time_unit,
            "pupil_unit": self.pupil_unit,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "extra": self.extra,
        }


class PreProcessing:
    """Preprocess Polars eye-tracking tables.

    Parameters
    ----------
    samples
        Sample-level table, normally containing ``tSample`` in milliseconds and
        gaze columns such as ``LX``/``LY``, ``RX``/``RY``, or ``X``/``Y``.
    fixations
        Fixation-event table, normally containing ``tStart`` and ``tEnd``.
    saccades
        Saccade-event table, normally containing ``tStart``, ``tEnd``, and
        start/end coordinates.
    blinks
        Blink-event table, normally containing ``tStart`` and ``tEnd``.
    user_messages
        Message table containing ``timestamp`` and ``message`` when
        message-based trial segmentation is required.
    session_path
        Directory used for metadata, recipes, and provenance sidecars.
    metadata
        Optional recording metadata.

    Notes
    -----
    Tables are normalized to :class:`polars.DataFrame` at the boundary.
    Methods preserve Polars schemas throughout the preprocessing pipeline.
    """

    VERSION = "0.3.0"

    def __init__(
        self,
        samples: DataFrame,
        fixations: DataFrame,
        saccades: DataFrame,
        blinks: DataFrame,
        user_messages: DataFrame,
        session_path: PathLike,
        metadata: SessionMetadata | None = None,
    ):
        self.samples = self._copy_frame(samples, "samples")
        self.fixations = self._copy_frame(fixations, "fixations")
        self.saccades = self._copy_frame(saccades, "saccades")
        self.blinks = self._copy_frame(blinks, "blinks")
        self.user_messages = self._copy_frame(user_messages, "user_messages")
        self.session_path = Path(session_path)
        self.metadata = metadata or SessionMetadata()

        if "message" in self.user_messages.columns:
            self.user_messages = self.user_messages.with_columns(
                pl.col("message").cast(pl.String, strict=False)
            )

    # ------------------------------- Utilities ------------------------------- #

    @staticmethod
    def _copy_frame(df: DataFrame, name: str = "table") -> DataFrame:
        if isinstance(df, pl.DataFrame):
            return df.clone()
        raise TypeError(
            f"PreProcessing {name} must be a Polars DataFrame, got {type(df)!r}."
        )

    @staticmethod
    def _require_columns(
        df: DataFrame,
        cols: Sequence[str],
        context: str,
    ) -> None:
        missing = [column for column in cols if column not in df.columns]
        if missing:
            raise ValueError(
                f"[{context}] Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

    @staticmethod
    def _assert_nonoverlap(
        starts: Sequence[Number],
        ends: Sequence[Number],
        key: str,
        session: Path,
    ) -> None:
        if len(starts) != len(ends):
            raise ValueError(
                f"[{key}] start_times and end_times must have the same length, "
                f"got {len(starts)} vs {len(ends)} in session: {session}"
            )
        for index, (start, end) in enumerate(zip(starts, ends)):
            if not start < end:
                raise ValueError(
                    f"[{key}] Non-positive interval at trial {index}: "
                    f"start={start}, end={end} in session: {session}"
                )
            if index < len(starts) - 1 and end > starts[index + 1]:
                raise ValueError(
                    f"[{key}] Overlapping trials {index}–{index + 1}: "
                    f"end[i]={end} > start[i+1]={starts[index + 1]} "
                    f"in session: {session}"
                )

    @staticmethod
    def _ensure_columns_exist(
        df: DataFrame,
        cols: Sequence[str],
    ) -> list[str]:
        return [column for column in cols if column in df.columns]

    def _save_json_sidecar(self, obj: dict, filename: str) -> None:
        self.session_path.mkdir(parents=True, exist_ok=True)
        with (self.session_path / filename).open("w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2, ensure_ascii=False)

    # ---------------------------- Public API: Meta ---------------------------- #

    def set_metadata(
        self,
        coords_unit: str | None = None,
        time_unit: str | None = None,
        pupil_unit: str | None = None,
        screen_width: int | None = None,
        screen_height: int | None = None,
        **extra,
    ) -> None:
        """Update session-level metadata used by preprocessing operations.

        Parameters
        ----------
        coords_unit : str, optional
            Unit used for gaze coordinates.
        time_unit : str, optional
            Unit used for timestamps and durations.
        pupil_unit : str, optional
            Unit used for pupil measurements.
        screen_width : int, optional
            Display width in coordinate units.
        screen_height : int, optional
            Display height in coordinate units.
        **extra : object
            Additional JSON-compatible recording metadata.
        """
        if coords_unit is not None:
            self.metadata.coords_unit = coords_unit
        if time_unit is not None:
            self.metadata.time_unit = time_unit
        if pupil_unit is not None:
            self.metadata.pupil_unit = pupil_unit
        if screen_width is not None:
            self.metadata.screen_width = screen_width
        if screen_height is not None:
            self.metadata.screen_height = screen_height
        self.metadata.extra.update(extra)

    # ----------------------- Public API: Message Parsing ---------------------- #

    def get_timestamps_from_messages(
        self,
        messages_dict: dict[str, list[str]],
        *,
        case_insensitive: bool = True,
        use_regex: bool = True,
        return_match_token: bool = False,
    ) -> dict[str, list[int]]:
        """Extract ordered timestamps by matching message patterns.

        Python's regular-expression engine is deliberately used instead of a
        dataframe-specific expression so literal/regex behavior stays explicit.

        Parameters
        ----------
        messages_dict : dict of str to list of str
            Phase names mapped to message tokens or regular expressions.
        case_insensitive : bool, default True
            Whether matching ignores letter case.
        use_regex : bool, default True
            Whether tokens are interpreted as regular expressions.
        return_match_token : bool, default False
            Whether to retain the matched token in ``user_messages``.

        Returns
        -------
        dict of str to list of int
            Ordered timestamps for every requested phase.

        Raises
        ------
        ValueError
            If required message columns are absent, a token list is empty, or
            a regular expression is invalid.
        """
        df = self.user_messages
        self._require_columns(
            df,
            ["timestamp", "message"],
            "get_timestamps_from_messages",
        )

        timestamps = df.get_column("timestamp").to_list()
        messages = df.get_column("message").to_list()
        matched_tokens = (
            df.get_column("matched_token").to_list()
            if "matched_token" in df.columns
            else [None] * df.height
        )

        flags = re.IGNORECASE if case_insensitive else 0
        timestamps_dict: dict[str, list[int]] = {}

        for key, tokens in messages_dict.items():
            if not tokens:
                raise ValueError(
                    f"[{key}] Empty token list passed to get_timestamps_from_messages."
                )

            prepared_tokens = [
                token if use_regex else re.escape(token) for token in tokens
            ]
            try:
                combined_pattern = re.compile("|".join(prepared_tokens), flags=flags)
                individual_patterns = [
                    re.compile(pattern, flags=flags) for pattern in prepared_tokens
                ]
            except re.error as error:
                raise ValueError(
                    f"[{key}] Invalid message pattern in {tokens}: {error}"
                ) from error

            hits: list[tuple[int, int]] = []
            for row_index, (timestamp, message) in enumerate(zip(timestamps, messages)):
                message_text = "" if message is None else str(message)
                if combined_pattern.search(message_text) is None:
                    continue

                try:
                    timestamp_value = int(timestamp)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        f"[{key}] Invalid timestamp {timestamp!r} for matched "
                        f"message {message_text!r} in session: {self.session_path}"
                    ) from error

                hits.append((row_index, timestamp_value))
                if return_match_token:
                    matched_tokens[row_index] = next(
                        (
                            token
                            for token, pattern in zip(tokens, individual_patterns)
                            if pattern.search(message_text) is not None
                        ),
                        None,
                    )

            if not hits:
                raise ValueError(
                    f"[{key}] No timestamps found for messages {tokens} "
                    f"in session: {self.session_path}"
                )

            hits.sort(key=lambda hit: hit[1])
            timestamps_dict[key] = [timestamp for _, timestamp in hits]

        if return_match_token:
            normalized_tokens = [
                None if self._metadata_value_is_missing(token) else str(token)
                for token in matched_tokens
            ]
            self.user_messages = df.with_columns(
                pl.Series(
                    name="matched_token",
                    values=normalized_tokens,
                    dtype=pl.String,
                )
            )

        return timestamps_dict

    # ---------------------- Public API: Trial Segmentation -------------------- #

    def split_all_into_trials(
        self,
        start_times: dict[str, list[Number]],
        end_times: dict[str, list[Number]],
        trial_labels: dict[str, list[str]] | None = None,
        *,
        allow_open_last: bool = True,
        require_nonoverlap: bool = True,
    ) -> None:
        """Segment samples and events using explicit millisecond intervals.

        Parameters
        ----------
        start_times : dict of str to list of float
            Per-phase trial start times in milliseconds.
        end_times : dict of str to list of float
            Per-phase trial end times in milliseconds.
        trial_labels : dict of str to list of str, optional
            Labels aligned with each phase's intervals.
        allow_open_last : bool, default True
            Whether an unmatched final start extends to the last sample.
        require_nonoverlap : bool, default True
            Whether overlapping trial intervals raise an error.

        Raises
        ------
        ValueError
            If phase definitions are missing, inconsistent, or overlap when
            overlap checking is enabled.
        """
        missing_end_keys = [key for key in start_times if key not in end_times]
        if missing_end_keys:
            raise ValueError(
                f"Missing end-time definitions for phases: {missing_end_keys}."
            )

        self.samples = self._split_into_trials_df(
            self.samples,
            start_times,
            end_times,
            trial_labels,
            sample_table=True,
            allow_open_last=allow_open_last,
            require_nonoverlap=require_nonoverlap,
        )
        self.fixations = self._split_into_trials_df(
            self.fixations,
            start_times,
            end_times,
            trial_labels,
            sample_table=False,
            allow_open_last=allow_open_last,
            require_nonoverlap=require_nonoverlap,
        )
        self.saccades = self._split_into_trials_df(
            self.saccades,
            start_times,
            end_times,
            trial_labels,
            sample_table=False,
            allow_open_last=allow_open_last,
            require_nonoverlap=require_nonoverlap,
        )
        self.blinks = self._split_into_trials_df(
            self.blinks,
            start_times,
            end_times,
            trial_labels,
            sample_table=False,
            allow_open_last=allow_open_last,
            require_nonoverlap=require_nonoverlap,
        )

    def split_all_into_trials_by_msgs(
        self,
        start_msgs: dict[str, list[str]],
        end_msgs: dict[str, list[str]],
        trial_labels: dict[str, list[str]] | None = None,
        *,
        case_insensitive: bool = True,
        use_regex: bool = True,
        return_match_token: bool = False,
        allow_open_last: bool = True,
        require_nonoverlap: bool = True,
    ) -> None:
        """Segment tables using matched start and end messages.

        Parameters
        ----------
        start_msgs : dict of str to list of str
            Phase names mapped to start-message patterns.
        end_msgs : dict of str to list of str
            Phase names mapped to end-message patterns.
        trial_labels : dict of str to list of str, optional
            Labels aligned with each phase's intervals.
        case_insensitive : bool, default True
            Whether message matching ignores letter case.
        use_regex : bool, default True
            Whether message tokens are regular expressions.
        return_match_token : bool, default False
            Whether matched tokens are retained in ``user_messages``.
        allow_open_last : bool, default True
            Whether an unmatched final start extends to the last sample.
        require_nonoverlap : bool, default True
            Whether overlapping intervals raise an error.
        """
        matching_options = {
            "case_insensitive": case_insensitive,
            "use_regex": use_regex,
            "return_match_token": return_match_token,
        }
        starts = self.get_timestamps_from_messages(start_msgs, **matching_options)
        ends = self.get_timestamps_from_messages(end_msgs, **matching_options)
        self.split_all_into_trials(
            starts,
            ends,
            trial_labels,
            allow_open_last=allow_open_last,
            require_nonoverlap=require_nonoverlap,
        )

    def split_all_into_trials_by_durations(
        self,
        start_msgs: dict[str, list[str]],
        durations: dict[str, list[Number]],
        trial_labels: dict[str, list[str]] | None = None,
        *,
        case_insensitive: bool = True,
        use_regex: bool = True,
        return_match_token: bool = False,
        allow_open_last: bool = True,
        require_nonoverlap: bool = True,
    ) -> None:
        """Segment tables using matched start messages and trial durations.

        Parameters
        ----------
        start_msgs : dict of str to list of str
            Phase names mapped to start-message patterns.
        durations : dict of str to list of float
            Trial durations aligned with each phase's matched starts.
        trial_labels : dict of str to list of str, optional
            Labels aligned with each phase's intervals.
        case_insensitive : bool, default True
            Whether message matching ignores letter case.
        use_regex : bool, default True
            Whether message tokens are regular expressions.
        return_match_token : bool, default False
            Whether matched tokens are retained in ``user_messages``.
        allow_open_last : bool, default True
            Whether an unmatched final start extends to the last sample.
        require_nonoverlap : bool, default True
            Whether overlapping intervals raise an error.

        Raises
        ------
        ValueError
            If a phase has no duration definition or too few durations.
        """
        starts = self.get_timestamps_from_messages(
            start_msgs,
            case_insensitive=case_insensitive,
            use_regex=use_regex,
            return_match_token=return_match_token,
        )
        end_times: dict[str, list[Number]] = {}
        for key, start_values in starts.items():
            if key not in durations:
                raise ValueError(f"[{key}] No trial durations were provided.")
            duration_values = durations[key]
            if len(duration_values) < len(start_values):
                raise ValueError(
                    f"[{key}] Provided {len(duration_values)} durations but found "
                    f"{len(start_values)} start times in session: {self.session_path}"
                )
            end_times[key] = [
                start + duration
                for start, duration in zip(start_values, duration_values)
            ]

        self.split_all_into_trials(
            starts,
            end_times,
            trial_labels,
            allow_open_last=allow_open_last,
            require_nonoverlap=require_nonoverlap,
        )

    def _split_into_trials_df(
        self,
        data: DataFrame,
        start_times: dict[str, list[Number]],
        end_times: dict[str, list[Number]],
        trial_labels: dict[str, list[str]] | None = None,
        *,
        sample_table: bool,
        allow_open_last: bool = True,
        require_nonoverlap: bool = True,
    ) -> DataFrame:
        if sample_table:
            self._require_columns(data, ["tSample"], "split_into_trials(samples)")
        else:
            self._require_columns(
                data,
                ["tStart", "tEnd"],
                "split_into_trials(events)",
            )

        df = data.with_columns(
            pl.lit("").alias("phase"),
            pl.lit(-1, dtype=pl.Int64).alias("trial_number"),
            pl.lit("").alias("trial_label"),
        )

        for key, raw_starts in start_times.items():
            start_list = list(raw_starts)
            end_list = list(end_times[key])

            if allow_open_last and len(start_list) == len(end_list) + 1:
                start_list = start_list[:-1]

            if require_nonoverlap:
                self._assert_nonoverlap(
                    start_list,
                    end_list,
                    key,
                    self.session_path,
                )
            elif len(start_list) != len(end_list):
                raise ValueError(
                    f"[{key}] start_times and end_times length mismatch: "
                    f"{len(start_list)} vs {len(end_list)} "
                    f"in session: {self.session_path}"
                )

            labels = (
                trial_labels.get(key)
                if trial_labels is not None and key in trial_labels
                else None
            )
            if labels is not None and len(labels) != len(start_list):
                raise ValueError(
                    f"[{key}] Computed {len(start_list)} trials but got "
                    f"{len(labels)} trial labels in session: {self.session_path}"
                )

            for trial_number, (start, end) in enumerate(zip(start_list, end_list)):
                label = labels[trial_number] if labels is not None else ""
                condition = (
                    pl.col("tSample").is_between(start, end, closed="both")
                    if sample_table
                    else (pl.col("tStart") >= start) & (pl.col("tEnd") <= end)
                )
                df = df.with_columns(
                    pl.when(condition)
                    .then(pl.lit(str(key)))
                    .otherwise(pl.col("phase"))
                    .alias("phase"),
                    pl.when(condition)
                    .then(pl.lit(trial_number, dtype=pl.Int64))
                    .otherwise(pl.col("trial_number"))
                    .alias("trial_number"),
                    pl.when(condition)
                    .then(pl.lit(str(label)))
                    .otherwise(pl.col("trial_label"))
                    .alias("trial_label"),
                )

        return df

    # ------------------------- Public API: QC / Flags ------------------------- #

    def bad_samples(
        self,
        screen_height: int | None = None,
        screen_width: int | None = None,
        *,
        mark_nan_as_bad: bool = True,
        inclusive_bounds: bool = True,
    ) -> None:
        """Mark rows with out-of-screen or missing coordinates as bad.

        Parameters
        ----------
        screen_height : int, optional
            Display height, overriding session metadata.
        screen_width : int, optional
            Display width, overriding session metadata.
        mark_nan_as_bad : bool, default True
            Whether null and NaN coordinates are marked bad.
        inclusive_bounds : bool, default True
            Whether coordinates exactly on zero or the upper bound are valid.

        Raises
        ------
        ValueError
            If screen dimensions are unavailable or not positive.
        """
        height = (
            screen_height if screen_height is not None else self.metadata.screen_height
        )
        width = screen_width if screen_width is not None else self.metadata.screen_width
        if height is None or width is None:
            raise ValueError(
                "bad_samples requires screen_height and screen_width (either "
                "passed or set via set_metadata())."
            )
        if height <= 0 or width <= 0:
            raise ValueError("Screen dimensions must be positive.")

        def mark(df: DataFrame) -> DataFrame:
            coordinate_columns = self._ensure_columns_exist(
                df,
                _COORDINATE_COLUMNS,
            )
            if not coordinate_columns:
                return df.with_columns(pl.lit(False).alias("bad"))

            x_columns = [
                column
                for column in _X_COORDINATE_COLUMNS
                if column in coordinate_columns
            ]
            y_columns = [
                column
                for column in _Y_COORDINATE_COLUMNS
                if column in coordinate_columns
            ]

            bad_expression = pl.lit(False)
            for column, upper_bound in (
                *((column, width) for column in x_columns),
                *((column, height) for column in y_columns),
            ):
                values = pl.col(column).cast(pl.Float64, strict=False)
                missing = values.is_null() | values.is_nan()
                if inclusive_bounds:
                    outside = (values < 0) | (values > upper_bound)
                else:
                    outside = (values <= 0) | (values >= upper_bound)
                outside = outside.fill_null(False)
                outside = outside | missing if mark_nan_as_bad else outside & ~missing
                bad_expression = bad_expression | outside

            return df.with_columns(bad_expression.alias("bad"))

        self.samples = mark(self.samples)
        self.fixations = mark(self.fixations)
        self.saccades = mark(self.saccades)

    # ---------------------- Public API: Saccade Direction --------------------- #

    def saccades_direction(self, tol_deg: float = 15.0) -> None:
        """Compute saccade angles and cardinal directions.

        Parameters
        ----------
        tol_deg : float, default 15.0
            Angular tolerance around each cardinal direction, in degrees.

        Raises
        ------
        ValueError
            If the tolerance is invalid or required numeric coordinates are
            absent.
        """
        if not np.isfinite(tol_deg) or tol_deg < 0 or tol_deg > 90:
            raise ValueError("tol_deg must be a finite value between 0 and 90.")

        required = ["xStart", "xEnd", "yStart", "yEnd"]
        self._require_columns(self.saccades, required, "saccades_direction")

        try:
            coordinates = {
                column: self.saccades.get_column(column)
                .cast(pl.Float64, strict=True)
                .to_numpy()
                for column in required
            }
        except pl.exceptions.InvalidOperationError as error:
            raise ValueError(
                "[saccades_direction] Coordinate columns must be numeric."
            ) from error

        x_difference = coordinates["xEnd"] - coordinates["xStart"]
        y_difference = coordinates["yEnd"] - coordinates["yStart"]
        degrees = np.degrees(np.arctan2(y_difference, x_difference)).astype(
            float,
            copy=False,
        )

        right = (-tol_deg < degrees) & (degrees < tol_deg)
        left = (degrees > 180 - tol_deg) | (degrees < -180 + tol_deg)
        down = ((90 - tol_deg) < degrees) & (degrees < (90 + tol_deg))
        up = ((-90 - tol_deg) < degrees) & (degrees < (-90 + tol_deg))

        directions = np.full(degrees.shape, "", dtype=object)
        directions[right] = "right"
        directions[left] = "left"
        directions[down] = "down"
        directions[up] = "up"

        self.saccades = self.saccades.with_columns(
            pl.Series("deg", degrees, dtype=pl.Float64),
            pl.Series("dir", directions.tolist(), dtype=pl.String),
        )

    # -------------------------- Public API: Orchestrator ---------------------- #

    def process(
        self,
        functions_and_params: dict[str, dict],
        *,
        log_recipe: bool = True,
        recipe_filename: str = "preprocessing_recipe.json",
        provenance_filename: str = "preprocessing_provenance.json",
    ) -> None:
        """Run a declarative preprocessing recipe.

        Parameters
        ----------
        functions_and_params : dict of str to dict
            Ordered public method names and their keyword arguments.
        log_recipe : bool, default True
            Whether to write declaration and completion sidecars.
        recipe_filename : str, default "preprocessing_recipe.json"
            Filename for the declared recipe.
        provenance_filename : str, default "preprocessing_provenance.json"
            Filename for completed-step provenance.

        Raises
        ------
        AttributeError
            If a recipe names an unknown or private operation.
        TypeError
            If a named attribute is not callable or its parameters are not a
            dictionary.
        """
        if log_recipe:
            self._save_json_sidecar(
                {
                    "declared_recipe": functions_and_params,
                    "tool_version": self.VERSION,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "session_path": str(self.session_path),
                },
                recipe_filename,
            )

        for function_name, parameters in functions_and_params.items():
            if function_name.startswith("_") or not hasattr(self, function_name):
                raise AttributeError(
                    f"Unknown preprocessing function '{function_name}'. "
                    f"Available: {self._public_recipe_methods()}"
                )
            function = getattr(self, function_name)
            if not callable(function):
                raise TypeError(
                    f"Preprocessing attribute '{function_name}' is not callable."
                )
            if not isinstance(parameters, dict):
                raise TypeError(
                    f"Parameters for '{function_name}' must be a dict, "
                    f"got {type(parameters)}"
                )
            function(**parameters)

        if log_recipe:
            self._save_json_sidecar(
                {
                    "completed_recipe": list(functions_and_params.keys()),
                    "tool_version": self.VERSION,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "metadata": self.metadata.to_dict(),
                },
                provenance_filename,
            )

    def _public_recipe_methods(self) -> list[str]:
        return sorted(
            name
            for name in dir(self)
            if not name.startswith("_") and callable(getattr(self, name))
        )

    # -------------------- Public API: Behavioral Metadata -------------------- #

    def add_trial_metadata(
        self,
        metadata_df: DataFrame,
        columns: Sequence[str],
    ) -> None:
        """Propagate trial-level behavioral metadata into ``samples``.

        ``metadata_df`` must contain one consistent record per ``trial_index``.
        Exact duplicate records are accepted; conflicting duplicate values
        raise an error. Existing requested columns in samples are replaced.

        Parameters
        ----------
        metadata_df : polars.DataFrame
            Trial-indexed behavioral metadata.
        columns : sequence of str
            Metadata columns to propagate into the sample table.

        Raises
        ------
        ValueError
            If either table lacks ``trial_index`` or duplicate trials contain
            conflicting values.
        """
        metadata_df = self._copy_frame(metadata_df, "metadata_df")
        if "trial_index" not in metadata_df.columns:
            raise ValueError(
                "[add_trial_metadata] metadata_df must contain a 'trial_index' column."
            )
        if "trial_index" not in self.samples.columns:
            raise ValueError(
                "[add_trial_metadata] samples must contain a 'trial_index' "
                "column. Make sure the parser preserves it."
            )

        requested = [
            column for column in dict.fromkeys(columns) if column != "trial_index"
        ]
        available = [column for column in requested if column in metadata_df.columns]
        skipped = [column for column in requested if column not in metadata_df.columns]
        if skipped:
            warnings.warn(
                "[add_trial_metadata] Columns not found in metadata_df and "
                f"skipped: {skipped}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not available:
            return

        unique_records = self._validate_and_deduplicate_trial_metadata(
            metadata_df.select("trial_index", *available).to_dicts(),
            available,
        )

        existing_columns = [
            column for column in available if column in self.samples.columns
        ]
        samples = (
            self.samples.drop(existing_columns)
            if existing_columns
            else self.samples.clone()
        )

        if not unique_records:
            self.samples = samples.with_columns(
                *(pl.lit(None).alias(column) for column in available)
            )
            return

        trial_metadata = pl.DataFrame(unique_records).select(
            "trial_index",
            *available,
        )
        row_order_column = "__pyxations_row_order__"
        while row_order_column in samples.columns:
            row_order_column = f"_{row_order_column}"

        try:
            self.samples = (
                samples.with_row_index(row_order_column)
                .join(trial_metadata, on="trial_index", how="left")
                .sort(row_order_column)
                .drop(row_order_column)
            )
        except (pl.exceptions.SchemaError, pl.exceptions.ComputeError) as error:
            raise ValueError(
                "[add_trial_metadata] Could not join metadata on trial_index. "
                "Check that sample and metadata join-key values use compatible "
                "types."
            ) from error

    @classmethod
    def _validate_and_deduplicate_trial_metadata(
        cls,
        records: Sequence[dict],
        columns: Sequence[str],
    ) -> list[dict]:
        by_trial: dict[object, dict] = {}
        ordered: list[dict] = []

        for record in records:
            trial_index = record["trial_index"]
            if cls._metadata_value_is_missing(trial_index):
                raise ValueError(
                    "[add_trial_metadata] trial_index cannot contain missing values."
                )
            try:
                existing = by_trial.get(trial_index)
            except TypeError as error:
                raise ValueError(
                    "[add_trial_metadata] trial_index values must be hashable."
                ) from error

            if existing is None and trial_index not in by_trial:
                normalized = {
                    "trial_index": trial_index,
                    **{
                        column: (
                            None
                            if cls._metadata_value_is_missing(record.get(column))
                            else record.get(column)
                        )
                        for column in columns
                    },
                }
                by_trial[trial_index] = normalized
                ordered.append(normalized)
                continue

            conflicting = [
                column
                for column in columns
                if not cls._metadata_values_equal(
                    existing.get(column),
                    record.get(column),
                )
            ]
            if conflicting:
                raise ValueError(
                    "[add_trial_metadata] Conflicting metadata for trial_index "
                    f"{trial_index!r} in columns: {conflicting}."
                )

        return ordered

    @staticmethod
    def _metadata_value_is_missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, (float, np.floating)):
            return bool(np.isnan(value))
        if isinstance(value, (complex, np.complexfloating)):
            return bool(np.isnan(value.real) or np.isnan(value.imag))
        return False

    @classmethod
    def _metadata_values_equal(cls, left: object, right: object) -> bool:
        if cls._metadata_value_is_missing(left) and cls._metadata_value_is_missing(
            right
        ):
            return True
        try:
            equal = left == right
        except (TypeError, ValueError):
            return False
        return bool(equal) if isinstance(equal, (bool, np.bool_)) else False
