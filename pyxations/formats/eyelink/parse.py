"""EyeLink EDF/ASC parser.

The parser streams an EyeLink ASC export directly into typed Polars tables.
Sample, message, blink, fixation, saccade, and calibration tables remain Polars
through preprocessing, best-eye selection, and Feather storage.
"""

from __future__ import annotations

import inspect
import logging
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing, SessionMetadata

logger = logging.getLogger(__name__)

_RATE_PATTERN = re.compile(r"\bRATE\s+([0-9]+(?:\.[0-9]+)?)\s+TRACKING\b")
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

_HEADER_SCHEMA = {"line": pl.String, "Line_number": pl.Int64}
_CALIBRATION_SCHEMA = {
    "line": pl.String,
    "Line_number": pl.Int64,
    "Calib_index": pl.Int64,
}
_MESSAGE_SCHEMA = {
    "timestamp": pl.Float64,
    "message": pl.String,
    "Line_number": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "Calib_index": pl.Int64,
}
_SAMPLE_SCHEMA = {
    "tSample": pl.Float64,
    "LX": pl.Float64,
    "LY": pl.Float64,
    "LPupil": pl.Float64,
    "RX": pl.Float64,
    "RY": pl.Float64,
    "RPupil": pl.Float64,
    "Line_number": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "Calib_index": pl.Int64,
}
_BLINK_SCHEMA = {
    "eye": pl.String,
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "duration": pl.Float64,
    "Line_number": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "Calib_index": pl.Int64,
}
_FIXATION_SCHEMA = {
    "eye": pl.String,
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "duration": pl.Float64,
    "xAvg": pl.Float64,
    "yAvg": pl.Float64,
    "pupilAvg": pl.Float64,
    "Line_number": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "Calib_index": pl.Int64,
}
_SACCADE_SCHEMA = {
    "eye": pl.String,
    "tStart": pl.Float64,
    "tEnd": pl.Float64,
    "duration": pl.Float64,
    "xStart": pl.Float64,
    "yStart": pl.Float64,
    "xEnd": pl.Float64,
    "yEnd": pl.Float64,
    "ampDeg": pl.Float64,
    "vPeak": pl.Float64,
    "Line_number": pl.Int64,
    "Eyes_recorded": pl.String,
    "Rate_recorded": pl.Float64,
    "Calib_index": pl.Int64,
}


def _frame_from_records(
    records: list[dict[str, Any]], schema: Mapping[str, pl.DataType]
) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(records, schema=schema, strict=False)


def _numeric_token(token: str, *, context: str, allow_missing: bool = True) -> float | None:
    if token in {".", "", "NA", "NaN", "nan"}:
        if allow_missing:
            return None
        raise ValueError(f"Missing numeric value while parsing {context}.")
    try:
        value = float(token)
    except (TypeError, ValueError, OverflowError) as exc:
        if allow_missing:
            return None
        raise ValueError(f"Invalid numeric value {token!r} while parsing {context}.") from exc
    if np.isfinite(value):
        return value
    if allow_missing:
        return None
    raise ValueError(f"Non-finite numeric value {token!r} while parsing {context}.")


def _required_number(token: str, *, context: str) -> float:
    value = _numeric_token(token, context=context, allow_missing=False)
    assert value is not None
    return value


def _record_metadata(
    *, line_number: int, eyes_recorded: str, rate_recorded: float, calib_index: int
) -> dict[str, Any]:
    return {
        "Line_number": line_number,
        "Eyes_recorded": eyes_recorded,
        "Rate_recorded": rate_recorded,
        "Calib_index": calib_index,
    }


def _extract_screen_resolution(calibration: pl.DataFrame) -> tuple[int, int]:
    gaze_lines = calibration.filter(pl.col("line").str.contains("GAZE_COORDS"))
    if gaze_lines.is_empty():
        raise ValueError(
            "EyeLink calibration data contain no GAZE_COORDS line; pass "
            "screen_width and screen_height explicitly or verify the ASC export."
        )
    line = gaze_lines.get_column("line")[0]
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", line)]
    if len(numbers) < 2:
        raise ValueError(f"Could not extract screen dimensions from calibration line: {line!r}.")
    return int(numbers[-2]), int(numbers[-1])


def _parse_sample_line(
    line: str,
    *,
    eyes_recorded: str,
    line_number: int,
    rate_recorded: float,
    calib_index: int,
) -> dict[str, Any]:
    tokens = line.split()
    if not tokens:
        raise ValueError(f"Empty EyeLink sample at ASC line {line_number}.")

    row: dict[str, Any] = {column: None for column in _SAMPLE_SCHEMA}
    row.update(
        _record_metadata(
            line_number=line_number,
            eyes_recorded=eyes_recorded,
            rate_recorded=rate_recorded,
            calib_index=calib_index,
        )
    )
    row["tSample"] = _required_number(tokens[0], context=f"sample line {line_number}")

    if eyes_recorded == "LR":
        if len(tokens) < 7:
            raise ValueError(
                f"Binocular EyeLink sample at ASC line {line_number} has fewer than 7 fields."
            )
        for column, token in zip(
            ("LX", "LY", "LPupil", "RX", "RY", "RPupil"), tokens[1:7]
        ):
            row[column] = _numeric_token(token, context=f"sample line {line_number}")
    elif eyes_recorded in {"L", "R"}:
        if len(tokens) < 4:
            raise ValueError(
                f"Monocular EyeLink sample at ASC line {line_number} has fewer than 4 fields."
            )
        prefix = eyes_recorded
        for column, token in zip(
            (f"{prefix}X", f"{prefix}Y", f"{prefix}Pupil"), tokens[1:4]
        ):
            row[column] = _numeric_token(token, context=f"sample line {line_number}")
    else:
        raise ValueError(
            f"EyeLink sample at ASC line {line_number} has unknown recorded-eye mode "
            f"{eyes_recorded!r}."
        )
    return row


def _parse_message_line(
    line: str,
    *,
    line_number: int,
    eyes_recorded: str,
    rate_recorded: float,
    calib_index: int,
) -> dict[str, Any]:
    payload = line[4:] if line.startswith("MSG ") else line
    timestamp_text, separator, message = payload.partition(" ")
    if not separator or not message:
        raise ValueError(f"Malformed EyeLink MSG record at ASC line {line_number}: {line!r}.")
    return {
        "timestamp": _required_number(timestamp_text, context=f"MSG line {line_number}"),
        "message": message,
        **_record_metadata(
            line_number=line_number,
            eyes_recorded=eyes_recorded,
            rate_recorded=rate_recorded,
            calib_index=calib_index,
        ),
    }


def _parse_event_line(
    line: str,
    *,
    line_number: int,
    eyes_recorded: str,
    rate_recorded: float,
    calib_index: int,
    event_type: str,
) -> dict[str, Any]:
    tokens = line.split()
    expected = {"EBLINK": 5, "EFIX": 8, "ESACC": 11}[event_type]
    if len(tokens) < expected:
        raise ValueError(
            f"Malformed EyeLink {event_type} record at ASC line {line_number}: {line!r}."
        )
    metadata = _record_metadata(
        line_number=line_number,
        eyes_recorded=eyes_recorded,
        rate_recorded=rate_recorded,
        calib_index=calib_index,
    )
    if event_type == "EBLINK":
        return {
            "eye": tokens[1],
            "tStart": _required_number(tokens[2], context=f"EBLINK line {line_number}"),
            "tEnd": _required_number(tokens[3], context=f"EBLINK line {line_number}"),
            "duration": _required_number(tokens[4], context=f"EBLINK line {line_number}"),
            **metadata,
        }
    if event_type == "EFIX":
        values = [
            _numeric_token(token, context=f"EFIX line {line_number}")
            for token in tokens[2:8]
        ]
        return {
            "eye": tokens[1],
            **dict(
                zip(
                    ("tStart", "tEnd", "duration", "xAvg", "yAvg", "pupilAvg"),
                    values,
                )
            ),
            **metadata,
        }
    values = [
        _numeric_token(token, context=f"ESACC line {line_number}")
        for token in tokens[2:11]
    ]
    return {
        "eye": tokens[1],
        **dict(
            zip(
                (
                    "tStart",
                    "tEnd",
                    "duration",
                    "xStart",
                    "yStart",
                    "xEnd",
                    "yEnd",
                    "ampDeg",
                    "vPeak",
                ),
                values,
            )
        ),
        **metadata,
    }


def _parse_ascii_tables(
    ascii_file_path: Path,
    *,
    msg_keywords: Sequence[str] | None,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    tuple[int, int],
]:
    """Stream one ASC file into typed Polars tables."""
    header_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    message_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    fixation_records: list[dict[str, Any]] = []
    saccade_records: list[dict[str, Any]] = []
    blink_records: list[dict[str, Any]] = []

    calibration_flag = False
    start_flag = False
    recorded_eye = ""
    rate_recorded = 0.0
    calib_index = 0
    keywords = (
        (msg_keywords,)
        if isinstance(msg_keywords, str)
        else tuple(msg_keywords or ())
    )

    with Path(ascii_file_path).open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream):
            line = raw_line.strip().replace("\t", " ")
            tokens = line.split()
            first_token = tokens[0] if tokens else ""

            if "!MODE RECORD" in line and tokens:
                recorded_eye = tokens[-1]
            rate_match = _RATE_PATTERN.search(line)
            if rate_match:
                rate_recorded = float(rate_match.group(1))

            if len(line) < 2:
                continue
            if line.startswith("*"):
                header_records.append({"line": line, "Line_number": line_number})
                continue
            if "!CAL" in line and not calibration_flag:
                calibration_flag = True
                calib_index += 1
                calibration_records.append(
                    {"line": line, "Line_number": line_number, "Calib_index": calib_index}
                )
                continue
            if "!MODE RECORD" in line and calibration_flag:
                calibration_flag = False
                start_flag = True
                continue
            if calibration_flag and not (
                first_token == "MSG" and keywords and any(keyword in line for keyword in keywords)
            ):
                calibration_records.append(
                    {"line": line, "Line_number": line_number, "Calib_index": calib_index}
                )
                continue
            if not start_flag:
                continue

            common = {
                "line_number": line_number,
                "eyes_recorded": recorded_eye,
                "rate_recorded": rate_recorded,
                "calib_index": calib_index,
            }
            if first_token == "MSG" and keywords and any(keyword in line for keyword in keywords):
                message_records.append(_parse_message_line(line, **common))
            elif first_token == "ESACC":
                saccade_records.append(_parse_event_line(line, event_type="ESACC", **common))
            elif first_token == "EFIX":
                fixation_records.append(_parse_event_line(line, event_type="EFIX", **common))
            elif first_token == "EBLINK":
                blink_records.append(_parse_event_line(line, event_type="EBLINK", **common))
            elif first_token and (first_token[0].isdigit() or first_token.startswith("-")):
                sample_records.append(_parse_sample_line(line, **common))

    headers = _frame_from_records(header_records, _HEADER_SCHEMA)
    calibrations = _frame_from_records(calibration_records, _CALIBRATION_SCHEMA)
    messages = _frame_from_records(message_records, _MESSAGE_SCHEMA)
    samples = _frame_from_records(sample_records, _SAMPLE_SCHEMA)
    fixations = _frame_from_records(fixation_records, _FIXATION_SCHEMA).filter(
        pl.all_horizontal(
            [
                pl.col(column).is_not_null()
                for column in ("tStart", "tEnd", "duration", "xAvg", "yAvg")
            ]
        )
    )
    saccades = _frame_from_records(saccade_records, _SACCADE_SCHEMA).filter(
        pl.all_horizontal(
            [
                pl.col(column).is_not_null()
                for column in (
                    "tStart",
                    "tEnd",
                    "duration",
                    "xStart",
                    "yStart",
                    "xEnd",
                    "yEnd",
                    "ampDeg",
                    "vPeak",
                )
            ]
        )
    )
    blinks = _frame_from_records(blink_records, _BLINK_SCHEMA)

    if samples.is_empty():
        raise ValueError(f"EyeLink ASC file {ascii_file_path} contains no calibrated samples.")
    screen_resolution = _extract_screen_resolution(calibrations)
    headers = pl.concat(
        [
            headers,
            pl.DataFrame(
                {
                    "line": [f"** SCREEN SIZE: {screen_resolution[0]} {screen_resolution[1]}"],
                    "Line_number": [-1],
                },
                schema=_HEADER_SCHEMA,
            ),
        ],
        how="vertical",
    )
    return (
        headers,
        calibrations,
        messages,
        samples,
        fixations,
        saccades,
        blinks,
        screen_resolution,
    )


def _keep_eye_polars(
    eye: str,
    samples: pl.DataFrame,
    fixations: pl.DataFrame,
    blinks: pl.DataFrame,
    saccades: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if eye not in {"L", "R"}:
        raise ValueError(f"Best eye must be 'L' or 'R', got {eye!r}.")
    prefix = eye
    required_sample_columns = [
        "tSample",
        f"{prefix}X",
        f"{prefix}Y",
        f"{prefix}Pupil",
        "Line_number",
        "Eyes_recorded",
        "Rate_recorded",
        "Calib_index",
    ]
    missing = [column for column in required_sample_columns if column not in samples.columns]
    if missing:
        raise ValueError(f"Cannot retain EyeLink eye {eye}; sample columns are missing: {missing}.")

    selected_samples = samples.select(required_sample_columns).rename(
        {f"{prefix}X": "X", f"{prefix}Y": "Y", f"{prefix}Pupil": "Pupil"}
    )

    def select_events(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        if "eye" not in frame.columns:
            raise ValueError("EyeLink event table has no 'eye' column for best-eye filtering.")
        selected = frame.filter(pl.col("eye") == eye)
        if {"xAvg", "yAvg"}.issubset(selected.columns):
            required = ("tStart", "tEnd", "duration", "xAvg", "yAvg")
        elif {"xStart", "yStart", "xEnd", "yEnd"}.issubset(selected.columns):
            required = (
                "tStart",
                "tEnd",
                "duration",
                "xStart",
                "yStart",
                "xEnd",
                "yEnd",
            )
        else:
            required = ("tStart", "tEnd", "duration")
        return selected.filter(
            pl.all_horizontal(
                [pl.col(column).is_not_null() for column in required]
            )
        )

    return (
        selected_samples,
        select_events(fixations),
        select_events(blinks),
        select_events(saccades),
    )


def _find_best_eye_from_lines(lines: Sequence[str]) -> str:
    """Apply the historical EyeLink calibration scoring rules to ordered lines."""
    normalized_lines = [str(line) for line in lines]
    validation_positions = [
        index
        for index, line in enumerate(normalized_lines)
        if "CAL VALIDATION" in line
    ]
    if not validation_positions:
        return "M"

    last_position = validation_positions[-1]
    last_message = normalized_lines[last_position]
    previous_message = (
        normalized_lines[last_position - 1] if last_position > 0 else None
    )

    def is_validation(message: str | None) -> bool:
        return message is not None and "CAL VALIDATION" in message

    def is_aborted(message: str | None) -> bool:
        return message is not None and "ABORTED" in message

    def named_eye(message: str) -> str:
        # Preserve the legacy fallback: anything not explicitly left is right.
        return "L" if ("LEFT" in message or "L ABORTED" in message) else "R"

    def validation_error(message: str) -> float:
        tokens = message.split()
        try:
            error_index = tokens.index("ERROR")
            value = float(tokens[error_index + 1])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "Could not parse EyeLink calibration ERROR value from "
                f"validation record: {message!r}."
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"EyeLink calibration ERROR value must be finite, got {value!r}."
            )
        return value

    if is_aborted(last_message):
        if not is_validation(previous_message) or is_aborted(previous_message):
            return named_eye(last_message)
        return named_eye(previous_message)

    if not is_validation(previous_message) or is_aborted(previous_message):
        return named_eye(last_message)

    assert previous_message is not None
    left_message = last_message if "LEFT" in last_message else previous_message
    right_message = last_message if "RIGHT" in last_message else previous_message
    left_error = validation_error(left_message)
    right_error = validation_error(right_message)
    return "L" if left_error < right_error else "R"


def _find_best_eye_polars(calibration: pl.DataFrame) -> str:
    """Return the historical EyeLink best-eye choice for one calibration.

    Parameters
    ----------
    calibration
        Polars table containing at least the ``line`` column. ``Line_number``
        is used to restore ASC order when available.

    Returns
    -------
    str
        ``"L"`` or ``"R"`` for the selected eye, or ``"M"`` when no
        validation record is available. Tied validation errors select the
        right eye to preserve the historical behavior.
    """
    if "line" not in calibration.columns:
        raise ValueError("EyeLink calibration table must contain a 'line' column.")
    if calibration.is_empty():
        return "M"

    ordered = calibration
    if "Line_number" in ordered.columns:
        ordered = ordered.sort("Line_number")
    return _find_best_eye_from_lines(ordered.get_column("line").to_list())


def _apply_best_eye(
    calibrations: pl.DataFrame,
    samples: pl.DataFrame,
    fixations: pl.DataFrame,
    blinks: pl.DataFrame,
    saccades: pl.DataFrame,
    *,
    session_folder_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Score and retain the best EyeLink eye using Polars only."""
    if "Calib_index" not in calibrations.columns:
        raise ValueError("EyeLink calibration table must contain 'Calib_index'.")

    calibration_indexes = sorted(calibrations.get_column("Calib_index").unique().to_list())
    best_eyes = [
        _find_best_eye_polars(
            calibrations.filter(pl.col("Calib_index") == calibration_index)
        )
        for calibration_index in calibration_indexes
    ]

    if not best_eyes or best_eyes[0] == "M":
        logger.warning(
            "The first calibration validation for subject %s in session %s is missing; "
            "best-eye filtering was not applied.",
            session_folder_path.parent.name,
            session_folder_path.name,
        )
        return samples, fixations, blinks, saccades

    for index in range(1, len(best_eyes)):
        if best_eyes[index] == "M":
            logger.warning(
                "Calibration validation %s is missing for subject %s in session %s; "
                "using previous best eye %s.",
                calibration_indexes[index],
                session_folder_path.parent.name,
                session_folder_path.name,
                best_eyes[index - 1],
            )
            best_eyes[index] = best_eyes[index - 1]

    def calibration_rows(
        frame: pl.DataFrame, calibration_index: int
    ) -> pl.DataFrame:
        if "Calib_index" in frame.columns:
            return frame.filter(pl.col("Calib_index") == calibration_index)
        if frame.is_empty():
            return frame
        raise ValueError(
            "EyeLink event table must contain 'Calib_index' for best-eye "
            "filtering."
        )

    pieces = [
        _keep_eye_polars(
            best_eye,
            samples.filter(pl.col("Calib_index") == calibration_index),
            calibration_rows(fixations, calibration_index),
            calibration_rows(blinks, calibration_index),
            calibration_rows(saccades, calibration_index),
        )
        for calibration_index, best_eye in zip(calibration_indexes, best_eyes)
    ]
    return tuple(
        pl.concat([piece[position] for piece in pieces], how="vertical_relaxed")
        for position in range(4)
    )  # type: ignore[return-value]


def _detection_registry() -> dict[str, type]:
    """Load built-in detector classes without importing dataset orchestration."""
    from pyxations.methods.eyemovement.REMoDNaV import RemodnavDetection
    from pyxations.methods.eyemovement.engbert import EngbertDetection

    return {"remodnav": RemodnavDetection, "engbert": EngbertDetection}


def _segmentation_step(options: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    prefer_durations = bool(options.get("prefer_durations", False))
    have_explicit_times = "start_times" in options and "end_times" in options
    have_durations = "start_msgs" in options and "durations" in options
    have_message_times = "start_msgs" in options and "end_msgs" in options

    if not (have_explicit_times or have_durations or have_message_times):
        raise ValueError(
            "Provide one of: (start_times & end_times), (start_msgs & durations), "
            "or (start_msgs & end_msgs)."
        )

    if have_explicit_times:
        method_name = "split_all_into_trials"
    elif have_durations and (prefer_durations or not have_message_times):
        method_name = "split_all_into_trials_by_durations"
    else:
        method_name = "split_all_into_trials_by_msgs"

    method = getattr(PreProcessing, method_name)
    allowed = set(inspect.signature(method).parameters) - {"self"}
    parameters = {
        key: value
        for key, value in options.items()
        if key in _SEGMENTATION_KEYS and key in allowed
    }
    return method_name, parameters


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
    edf_files = sorted(
        file for file in Path(eye_tracking_data_path).iterdir() if file.suffix.lower() == ".edf"
    )
    if not edf_files:
        raise FileNotFoundError(f"No EyeLink EDF file found in {eye_tracking_data_path}.")
    if len(edf_files) > 1:
        logger.warning("More than one EDF file found in %s; skipping folder.", eye_tracking_data_path)
        return

    Path(session_folder_path).mkdir(parents=True, exist_ok=True)
    (Path(session_folder_path) / "eyelink_events").mkdir(parents=True, exist_ok=True)
    msg_keywords = kwargs.pop("msg_keywords", None)
    EyelinkParse(session_folder_path, exp_format).parse(
        edf_files[0],
        detection_algorithm,
        msg_keywords,
        force_best_eye,
        keep_ascii,
        overwrite,
        **kwargs,
    )


def convert_edf_to_ascii(edf_file_path: Path, output_dir: Path) -> Path:
    """Convert an EDF file to ASCII with EyeLink's ``edf2asc`` utility."""
    if not shutil.which("edf2asc"):
        raise FileNotFoundError(
            "edf2asc not found. Install EyeLink software and ensure edf2asc is on PATH."
        )
    if output_dir is None:
        raise ValueError("Output directory must be specified.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ascii_file_path = output_dir / Path(edf_file_path).with_suffix(".asc").name
    if not ascii_file_path.exists():
        subprocess.run(
            ["edf2asc", "-failsafe", str(edf_file_path), str(ascii_file_path)],
            check=True,
        )
    return ascii_file_path


class EyelinkParse(BidsParse):
    """Parse EyeLink EDF exports into Polars derivative tables."""

    def parse(
        self,
        edf_file_path: Path,
        detection_algorithm: str,
        msg_keywords: Sequence[str] | None,
        force_best_eye: bool,
        keep_ascii: bool,
        overwrite: bool,
        **kwargs: Any,
    ) -> pl.DataFrame:
        ascii_file_path = convert_edf_to_ascii(Path(edf_file_path), self.session_folder_path)
        extension = self.export_method.extension()
        events_path = self.session_folder_path / f"{detection_algorithm}_events"
        expected_outputs = (
            self.session_folder_path / f"header{extension}",
            self.session_folder_path / f"calib{extension}",
            self.session_folder_path / f"samples{extension}",
            events_path / f"fix{extension}",
            events_path / f"sacc{extension}",
            events_path / f"blink{extension}",
        )
        if not overwrite and all(output.exists() for output in expected_outputs):
            return pl.DataFrame()

        (
            headers,
            calibrations,
            messages,
            samples,
            vendor_fixations,
            vendor_saccades,
            blinks,
            screen_resolution,
        ) = _parse_ascii_tables(ascii_file_path, msg_keywords=msg_keywords)

        raw_screen_width = kwargs.get("screen_width")
        raw_screen_height = kwargs.get("screen_height")
        screen_width = int(screen_resolution[0] if raw_screen_width is None else raw_screen_width)
        screen_height = int(screen_resolution[1] if raw_screen_height is None else raw_screen_height)

        if detection_algorithm == "eyelink":
            fixations, saccades = vendor_fixations, vendor_saccades
        else:
            detection_registry = _detection_registry()
            try:
                detector_class = detection_registry[detection_algorithm]
            except KeyError as exc:
                available = ["eyelink", *sorted(detection_registry)]
                raise ValueError(
                    f"Unknown detection algorithm {detection_algorithm!r}. "
                    f"Available algorithms: {available}."
                ) from exc
            detector = detector_class(
                session_folder_path=self.session_folder_path,
                samples=samples,
            )
            detector_parameters = {
                key: value
                for key, value in kwargs.items()
                if key in inspect.signature(detector.detect_eye_movements).parameters
            }
            fixations, saccades = detector.detect_eye_movements(**detector_parameters)
            if not isinstance(fixations, pl.DataFrame) or not isinstance(saccades, pl.DataFrame):
                raise TypeError(
                    "EyeLink detectors must return Polars DataFrames when given Polars samples."
                )

        if force_best_eye:
            samples, fixations, blinks, saccades = _apply_best_eye(
                calibrations,
                samples,
                fixations,
                blinks,
                saccades,
                session_folder_path=self.session_folder_path,
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
                pupil_unit="arbitrary",
                screen_width=screen_width,
                screen_height=screen_height,
            ),
        )

        bad_parameters = {
            key: kwargs[key]
            for key in ("screen_height", "screen_width", "mark_nan_as_bad", "inclusive_bounds")
            if key in kwargs
        }
        segmentation_method, segmentation_parameters = _segmentation_step(kwargs)
        direction_parameters = {"tol_deg": kwargs["tol_deg"]} if "tol_deg" in kwargs else {}
        preprocessing.process(
            {
                "bad_samples": bad_parameters,
                segmentation_method: segmentation_parameters,
                "saccades_direction": direction_parameters,
            }
        )

        if not keep_ascii:
            ascii_file_path.unlink(missing_ok=True)

        self.detection_algorithm = detection_algorithm
        self.store_dataframes(
            preprocessing.samples,
            dfCalib=calibrations,
            dfFix=preprocessing.fixations,
            dfSacc=preprocessing.saccades,
            dfHeader=headers,
            dfBlink=preprocessing.blinks,
            dfMsg=preprocessing.user_messages,
        )
        return preprocessing.samples
