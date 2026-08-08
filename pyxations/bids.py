"""BIDS eye-tracking writing and validation utilities.

The writer targets the eye-tracking additions in BIDS 1.11.1. Vendor files are
kept under ``sourcedata`` while standardized, per-eye physiological recordings
are written to the raw BIDS dataset.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from pyxations.behavior import read_behavioral_events
from pyxations.tables import (
    SessionTables,
    empty_frame,
    frame_payload,
    payload_frame,
    read_tsv,
    write_tsv,
)

BIDS_VERSION = "1.11.1"
BIDS_VALIDATOR_VERSION = "3.0.1"


class BIDSValidationError(RuntimeError):
    """Raised when the official BIDS Validator reports an invalid dataset."""


@dataclass
class EyeRecording:
    """Canonical sample-level recording for one eye."""

    samples: pl.DataFrame
    recorded_eye: str
    sampling_frequency: float
    timestamp_unit: str
    coordinate_unit: str
    coordinate_description: str
    pupil_unit: str | None = None
    pupil_description: str | None = None
    manufacturer: str | None = None

    def normalized(self) -> EyeRecording:
        """Validate the recording and return a canonical copy of it.

        Checks that the required sample columns are present, that the recorded
        eye is one of the accepted labels and that the sampling frequency is
        positive, then returns an equivalent recording with a float sampling
        frequency.

        Returns
        -------
        EyeRecording
            A validated copy, ready to be written to BIDS.

        Raises
        ------
        ValueError
            If the ``timestamp``, ``x_coordinate`` or ``y_coordinate`` columns
            are missing, if ``recorded_eye`` is not ``"left"``, ``"right"`` or
            ``"cyclopean"``, or if ``sampling_frequency`` is not positive.
        """
        required = ["timestamp", "x_coordinate", "y_coordinate"]
        missing = [column for column in required if column not in self.samples]
        if missing:
            raise ValueError(f"Eye recording is missing required columns: {missing}")
        if self.recorded_eye not in {"left", "right", "cyclopean"}:
            raise ValueError(f"Unsupported RecordedEye value: {self.recorded_eye}")

        columns = required + (
            ["pupil_size"] if "pupil_size" in self.samples.columns else []
        )
        columns += [column for column in self.samples.columns if column not in columns]
        numeric = required + (
            ["pupil_size"] if "pupil_size" in self.samples.columns else []
        )
        samples = self.samples.select(columns).with_columns(
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .replace([float("inf"), float("-inf")], None)
            .alias(column)
            for column in numeric
        )
        if "pupil_size" in samples:
            samples = samples.with_columns(
                pl.when(pl.col("pupil_size") > 0)
                .then(pl.col("pupil_size"))
                .otherwise(None)
                .alias("pupil_size")
            )
        samples = (
            samples.filter(pl.col("timestamp").is_not_null())
            .sort("timestamp", maintain_order=True)
            .unique(subset=["timestamp"], keep="first", maintain_order=True)
        )
        if samples.is_empty():
            raise ValueError(f"No samples were found for the {self.recorded_eye} eye")

        frequency = float(self.sampling_frequency)
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError(
                f"SamplingFrequency must be positive, got {self.sampling_frequency}"
            )
        return EyeRecording(
            samples=samples,
            recorded_eye=self.recorded_eye,
            sampling_frequency=frequency,
            timestamp_unit=self.timestamp_unit,
            coordinate_unit=self.coordinate_unit,
            coordinate_description=self.coordinate_description,
            pupil_unit=self.pupil_unit,
            pupil_description=self.pupil_description,
            manufacturer=self.manufacturer,
        )


@dataclass
class SourceRecordingBundle:
    """Normalized raw-BIDS content extracted from one vendor recording."""

    recordings: list[EyeRecording]
    events: pl.DataFrame = field(default_factory=empty_frame)
    calibration: pl.DataFrame = field(default_factory=empty_frame)
    header: pl.DataFrame = field(default_factory=empty_frame)
    metadata: dict = field(default_factory=dict)


def bids_label(value: str, *, fallback: str) -> str:
    """Normalize a value for use as a BIDS entity label."""

    label = re.sub(r"[^A-Za-z0-9]+", "", str(value))
    return label or fallback


def _task_from_filename(path: Path, default: str) -> str:
    match = re.search(r"(?:^|_)task-([A-Za-z0-9]+)", path.stem)
    return bids_label(match.group(1) if match else default, fallback="eyetracking")


def _session_from_filename(path: Path, session_substrings: int) -> str:
    tokens = path.stem.split("_")
    selected = tokens[1 : 1 + session_substrings]
    raw = "".join(selected) if selected else "1"
    if raw.lower().startswith("ses-"):
        raw = raw[4:]
    return bids_label(raw, fallback="1")


def _sampling_frequency(
    timestamps: pl.Series | Sequence[float], *, units_per_second: float
) -> float:
    numeric = pl.Series(timestamps).cast(pl.Float64, strict=False).drop_nulls()
    differences = numeric.sort().diff().filter(numeric.sort().diff() > 0)
    if differences.is_empty():
        raise ValueError("At least two distinct timestamps are needed")
    return float(units_per_second / float(differences.median()))


def _recording(
    data: pl.DataFrame,
    *,
    eye: str,
    timestamp: str,
    x: str,
    y: str,
    pupil: str | None,
    timestamp_unit: str,
    units_per_second: float,
    coordinate_unit: str,
    coordinate_description: str,
    pupil_unit: str | None,
    pupil_description: str | None,
    manufacturer: str,
    sampling_frequency: float | None = None,
) -> EyeRecording:
    columns = {
        timestamp: "timestamp",
        x: "x_coordinate",
        y: "y_coordinate",
    }
    if pupil and pupil in data:
        columns[pupil] = "pupil_size"
    samples = data.select(list(columns)).rename(columns)
    frequency = sampling_frequency or _sampling_frequency(
        samples["timestamp"], units_per_second=units_per_second
    )
    return EyeRecording(
        samples=samples,
        recorded_eye=eye,
        sampling_frequency=frequency,
        timestamp_unit=timestamp_unit,
        coordinate_unit=coordinate_unit,
        coordinate_description=coordinate_description,
        pupil_unit=pupil_unit if "pupil_size" in samples else None,
        pupil_description=pupil_description if "pupil_size" in samples else None,
        manufacturer=manufacturer,
    ).normalized()


def _read_gazepoint(
    path: Path, *, data: pl.DataFrame | None = None
) -> list[EyeRecording]:
    data = pl.read_csv(path, infer_schema_length=None) if data is None else data
    timestamp = "TIME" if "TIME" in data else "time"
    definitions = [
        ("left", "LPOGX", "LPOGY", "LPD"),
        ("right", "RPOGX", "RPOGY", "RPD"),
    ]
    recordings = []
    for eye, x, y, pupil in definitions:
        if {timestamp, x, y}.issubset(data.columns):
            recordings.append(
                _recording(
                    data,
                    eye=eye,
                    timestamp=timestamp,
                    x=x,
                    y=y,
                    pupil=pupil,
                    timestamp_unit="s",
                    units_per_second=1.0,
                    coordinate_unit="arbitrary",
                    coordinate_description=(
                        "GazePoint normalized point-of-gaze coordinates."
                    ),
                    pupil_unit="pixel",
                    pupil_description="Pupil diameter reported by GazePoint.",
                    manufacturer="GazePoint",
                )
            )
    return recordings


def _read_tobii(path: Path, *, data: pl.DataFrame | None = None) -> list[EyeRecording]:
    data = (
        pl.read_csv(path, separator="\t", infer_schema_length=None)
        if data is None
        else data
    )
    timestamp = (
        "Eyetracker timestamp"
        if "Eyetracker timestamp" in data
        else "Recording timestamp"
    )
    definitions = [
        ("left", "Gaze2d_Left.x", "Gaze2d_Left.y", "PupilDiam_Left"),
        ("right", "Gaze2d_Right.x", "Gaze2d_Right.y", "PupilDiam_Right"),
    ]
    recordings = []
    for eye, x, y, pupil in definitions:
        if {timestamp, x, y}.issubset(data.columns):
            recordings.append(
                _recording(
                    data,
                    eye=eye,
                    timestamp=timestamp,
                    x=x,
                    y=y,
                    pupil=pupil,
                    timestamp_unit="us",
                    units_per_second=1_000_000.0,
                    coordinate_unit="pixel",
                    coordinate_description=(
                        "Tobii two-dimensional display-area gaze coordinates."
                    ),
                    pupil_unit="mm",
                    pupil_description="Pupil diameter reported by Tobii.",
                    manufacturer="Tobii",
                )
            )
    if not recordings:
        raise ValueError(
            f"{path} has no Tobii gaze columns. The tobii reader expects the "
            "tab-separated export, with Gaze2d_Left/Gaze2d_Right columns and a "
            "recording or eyetracker timestamp. HDF5 containers written by "
            "Titta or PsychoPy's ioHub are a different format and are not "
            "supported yet."
        )
    return recordings


def _webgazer_gaze_rows(source: pl.DataFrame) -> pl.DataFrame:
    """Return the rows of a WebGazer export that actually carry gaze data."""

    return source.filter(
        pl.col("webgazer_data").is_not_null()
        & (pl.col("webgazer_data").cast(pl.String).str.strip_chars() != "")
    )


def webgazer_trial_numbering(source: pl.DataFrame) -> dict[int, int]:
    """Map jsPsych ``trial_index`` values to sequential trial numbers.

    A jsPsych export numbers every screen it presented, including instructions
    and calibration, so the trials that carry gaze are an arbitrary subset such
    as ``29, 30, 31, 33``. Every other Pyxations input format numbers trials
    ``0, 1, 2, ...`` in presentation order, so the raw jsPsych indices are
    renumbered to match and the originals are kept in ``source_trial_index``.

    The same mapping is applied to gaze samples and to the behavioral events
    table, which are read from the same source file by different code paths;
    were they to disagree, no trial would find its behavioral row.

    Parameters
    ----------
    source : polars.DataFrame
        The WebGazer export, read verbatim.

    Returns
    -------
    dict
        Mapping of original ``trial_index`` to sequential trial number. Screens
        that carry no gaze data are absent, and therefore have no trial number.
    """

    if "trial_index" not in source or "webgazer_data" not in source:
        return {}
    indices = (
        _webgazer_gaze_rows(source)
        .get_column("trial_index")
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    return {int(original): number for number, original in enumerate(indices)}


def _read_webgazer(
    path: Path, *, source: pl.DataFrame | None = None
) -> list[EyeRecording]:
    source = pl.read_csv(path, infer_schema_length=None) if source is None else source
    if "webgazer_data" not in source:
        raise ValueError(
            f"{path} has no webgazer_data column. The webgazer reader expects "
            "the export written by jsPsych, which carries each trial's gaze "
            "samples as JSON in that column. Other platforms, such as Gorilla, "
            "structure WebGazer data differently and are not supported yet."
        )

    numbering = webgazer_trial_numbering(source)
    samples: list[dict[str, float]] = []
    for row in _webgazer_gaze_rows(source).iter_rows(named=True):
        payload = row["webgazer_data"]
        try:
            gaze_samples = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid WebGazer JSON in {path}") from error
        base_time = float(row.get("time_elapsed") or 0.0)
        source_index = row.get("trial_index")
        trial_number = numbering.get(
            int(source_index) if source_index is not None else None, source_index
        )
        for sample in gaze_samples:
            if not {"t", "x", "y"}.issubset(sample):
                continue
            samples.append(
                {
                    "timestamp": base_time + float(sample["t"]),
                    "x_coordinate": sample["x"],
                    "y_coordinate": sample["y"],
                    "trial_number": trial_number,
                    "source_trial_index": source_index,
                }
            )
    frame = pl.DataFrame(samples, strict=False)
    if frame.is_empty():
        raise ValueError(f"No WebGazer samples found in {path}")
    return [
        EyeRecording(
            samples=frame,
            recorded_eye="cyclopean",
            sampling_frequency=_sampling_frequency(
                frame["timestamp"], units_per_second=1_000.0
            ),
            timestamp_unit="ms",
            coordinate_unit="pixel",
            coordinate_description=(
                "WebGazer browser viewport coordinates for the combined gaze estimate."
            ),
            manufacturer="WebGazer",
        ).normalized()
    ]


def _eyelink_ascii_path(path: Path, output_directory: Path) -> Path:
    if path.suffix.lower() == ".asc":
        return path
    executable = shutil.which("edf2asc")
    if executable is None:
        raise FileNotFoundError(
            "edf2asc is required to convert EyeLink EDF files. Install the "
            "EyeLink Developers Kit or provide an ASC input file."
        )
    output_path = output_directory / f"{path.stem}.asc"
    result = subprocess.run(
        [executable, "-failsafe", str(path), str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    # Some EDF2ASC releases return a non-zero status for a recoverable
    # end-of-file warning after reporting a successful conversion. A
    # non-empty ASC output is the reliable success criterion in that case.
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"edf2asc failed for {path}:\n{result.stdout}\n{result.stderr}"
        )
    return output_path


def _read_eyelink_bundle(path: Path) -> SourceRecordingBundle:
    with tempfile.TemporaryDirectory(prefix="pyxations-edf-") as directory:
        ascii_path = _eyelink_ascii_path(path, Path(directory))
        left: list[dict[str, float]] = []
        right: list[dict[str, float]] = []
        events: list[dict] = []
        calibration: list[dict] = []
        header: list[dict] = []
        recorded = ""
        declared_frequency: float | None = None
        calibration_index = 0
        calibration_active = False
        screen_width: int | None = None
        screen_height: int | None = None

        with ascii_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, raw_line in enumerate(stream):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("*"):
                    header.append({"line": line, "Line_number": line_number})
                rate_match = re.search(r"RATE\s+([0-9]+(?:\.[0-9]+)?)\s+TRACKING", line)
                if rate_match:
                    declared_frequency = float(rate_match.group(1))
                if "!CAL" in line and not calibration_active:
                    calibration_index += 1
                    calibration_active = True
                if calibration_active:
                    calibration.append(
                        {
                            "line": line.replace("\t", " "),
                            "Line_number": line_number,
                            "Calib_index": calibration_index,
                        }
                    )
                if "GAZE_COORDS" in line:
                    coordinates = re.search(
                        r"GAZE_COORDS\s+[-0-9.]+\s+[-0-9.]+\s+"
                        r"([-0-9.]+)\s+([-0-9.]+)",
                        line,
                    )
                    if coordinates:
                        screen_width = int(float(coordinates.group(1)))
                        screen_height = int(float(coordinates.group(2)))
                if "!MODE RECORD" in line:
                    eye_match = re.search(r"\b(LR|RL|L|R)\s*$", line)
                    if eye_match:
                        recorded = eye_match.group(1)
                    calibration_active = False

                fields = line.split()
                if not fields:
                    continue
                kind = fields[0]
                if kind == "MSG" and len(fields) >= 3:
                    try:
                        onset = float(fields[1])
                    except ValueError:
                        continue
                    events.append(
                        {
                            "onset": onset,
                            "duration": 0.0,
                            "trial_type": "message",
                            "message": " ".join(fields[2:]),
                            "eye": "n/a",
                            "end_timestamp": onset,
                            "calibration_index": calibration_index,
                            "line_number": line_number,
                        }
                    )
                    continue
                if kind in {"EFIX", "ESACC", "EBLINK"} and len(fields) >= 5:
                    try:
                        onset = float(fields[2])
                        end = float(fields[3])
                        duration = float(fields[4]) / 1_000.0
                    except ValueError:
                        continue
                    event = {
                        "onset": onset,
                        "duration": duration,
                        "trial_type": {
                            "EFIX": "fixation",
                            "ESACC": "saccade",
                            "EBLINK": "blink",
                        }[kind],
                        "message": "n/a",
                        "eye": fields[1],
                        "end_timestamp": end,
                        "calibration_index": calibration_index,
                        "line_number": line_number,
                    }
                    try:
                        if kind == "EFIX" and len(fields) >= 8:
                            event.update(
                                {
                                    "x_avg": float(fields[5]),
                                    "y_avg": float(fields[6]),
                                    "pupil_avg": float(fields[7]),
                                }
                            )
                        elif kind == "ESACC" and len(fields) >= 11:
                            event.update(
                                {
                                    "x_start": float(fields[5]),
                                    "y_start": float(fields[6]),
                                    "x_end": float(fields[7]),
                                    "y_end": float(fields[8]),
                                    "amplitude": float(fields[9]),
                                    "peak_velocity": float(fields[10]),
                                }
                            )
                    except ValueError:
                        pass
                    events.append(event)
                    continue
                if not re.fullmatch(r"-?\d+(?:\.\d+)?", fields[0]):
                    continue
                numeric = [
                    float(value) if re.fullmatch(r"-?\d+(?:\.\d+)?", value) else None
                    for value in fields[:7]
                ]
                eye_mode = recorded
                if not eye_mode:
                    eye_mode = "LR" if len(numeric) >= 7 else "L"
                if set(eye_mode) == {"L", "R"} and len(numeric) >= 7:
                    left.append(
                        {
                            "timestamp": numeric[0],
                            "x_coordinate": numeric[1],
                            "y_coordinate": numeric[2],
                            "pupil_size": numeric[3],
                            "calibration_index": calibration_index,
                            "line_number": line_number,
                        }
                    )
                    right.append(
                        {
                            "timestamp": numeric[0],
                            "x_coordinate": numeric[4],
                            "y_coordinate": numeric[5],
                            "pupil_size": numeric[6],
                            "calibration_index": calibration_index,
                            "line_number": line_number,
                        }
                    )
                elif "R" in eye_mode and len(numeric) >= 4:
                    right.append(
                        {
                            "timestamp": numeric[0],
                            "x_coordinate": numeric[1],
                            "y_coordinate": numeric[2],
                            "pupil_size": numeric[3],
                            "calibration_index": calibration_index,
                            "line_number": line_number,
                        }
                    )
                elif len(numeric) >= 4:
                    left.append(
                        {
                            "timestamp": numeric[0],
                            "x_coordinate": numeric[1],
                            "y_coordinate": numeric[2],
                            "pupil_size": numeric[3],
                            "calibration_index": calibration_index,
                            "line_number": line_number,
                        }
                    )

    recordings = []
    for eye, rows in (("left", left), ("right", right)):
        if not rows:
            continue
        frame = pl.DataFrame(rows, strict=False)
        frequency = declared_frequency or _sampling_frequency(
            frame["timestamp"], units_per_second=1_000.0
        )
        recordings.append(
            EyeRecording(
                samples=frame,
                recorded_eye=eye,
                sampling_frequency=frequency,
                timestamp_unit="ms",
                coordinate_unit="pixel",
                coordinate_description=(
                    "EyeLink gaze coordinates in the display coordinate system "
                    "configured during acquisition."
                ),
                pupil_unit="arbitrary",
                pupil_description=(
                    "Pupil area or diameter in EyeLink arbitrary units; consult "
                    "the acquisition configuration."
                ),
                manufacturer="SR Research",
            ).normalized()
        )
    if not recordings:
        raise ValueError(f"No EyeLink samples found in {path}")
    if screen_width and screen_height:
        header.append(
            {
                "line": f"** SCREEN SIZE: {screen_width} {screen_height}",
                "Line_number": -1,
            }
        )
    return SourceRecordingBundle(
        recordings=recordings,
        events=pl.DataFrame(events, strict=False) if events else empty_frame(),
        calibration=(
            pl.DataFrame(calibration, strict=False) if calibration else empty_frame()
        ),
        header=pl.DataFrame(header, strict=False) if header else empty_frame(),
        metadata={
            "ScreenWidth": screen_width,
            "ScreenHeight": screen_height,
        },
    )


def _read_eyelink(path: Path) -> list[EyeRecording]:
    return _read_eyelink_bundle(path).recordings


READERS = {
    "eyelink": _read_eyelink,
    "gazepoint": _read_gazepoint,
    "gaze": _read_gazepoint,
    "tobii": _read_tobii,
    "webgazer": _read_webgazer,
}


def _read_source_bundle(path: Path, format_name: str) -> SourceRecordingBundle:
    """Read a vendor recording once and retain BIDS-relevant auxiliary data."""

    if format_name == "eyelink":
        return _read_eyelink_bundle(path)
    if format_name in {"gaze", "gazepoint"}:
        data = pl.read_csv(path, infer_schema_length=None)
        events = empty_frame()
        if {"TIME", "BKDUR"}.issubset(data.columns):
            blink = data.with_columns(
                pl.col("BKDUR").cast(pl.Float64, strict=False).fill_null(0),
                pl.col("TIME").cast(pl.Float64, strict=False),
            ).filter(pl.col("BKDUR") > 0)
            if not blink.is_empty():
                events = blink.select(
                    (pl.col("TIME") - pl.col("BKDUR")).alias("onset"),
                    pl.col("BKDUR").alias("duration"),
                    pl.lit("blink").alias("trial_type"),
                    pl.lit("n/a").alias("message"),
                    pl.lit("cyclopean").alias("eye"),
                    pl.col("TIME").alias("end_timestamp"),
                )
        return SourceRecordingBundle(
            recordings=_read_gazepoint(path, data=data),
            events=events,
        )
    if format_name == "tobii":
        data = pl.read_csv(path, separator="\t", infer_schema_length=None)
        return SourceRecordingBundle(recordings=_read_tobii(path, data=data))
    if format_name == "webgazer":
        data = pl.read_csv(path, infer_schema_length=None)
        calibration = (
            data.filter(pl.col("rastoc-type") == "calibration-stimulus")
            if "rastoc-type" in data
            else empty_frame()
        )
        if not calibration.is_empty():
            compact_columns = [
                column
                for column in (
                    "trial_index",
                    "time_elapsed",
                    "rastoc-type",
                    "calibration-id",
                    "calibration-point-id",
                    "stimulus-coordinate",
                    "success",
                )
                if column in calibration
            ]
            calibration = calibration.select(compact_columns)
        return SourceRecordingBundle(
            recordings=_read_webgazer(path, source=data),
            calibration=calibration,
        )
    raise ValueError(f"Unsupported eye-tracking format: {format_name}")


PRIMARY_EXTENSIONS = {
    "eyelink": {".edf", ".asc"},
    "gazepoint": {".csv"},
    "gaze": {".csv"},
    "tobii": {".txt"},
    "webgazer": {".csv"},
}


def _is_primary_recording(path: Path, format_name: str) -> bool:
    """Return whether a source file contains samples for the selected tracker."""

    if path.suffix.lower() not in PRIMARY_EXTENSIONS[format_name]:
        return False
    if format_name == "eyelink":
        return True

    try:
        columns = set(
            pl.scan_csv(
                path,
                separator="\t" if format_name == "tobii" else ",",
                infer_schema_length=0,
            )
            .collect_schema()
            .names()
        )
    except (OSError, UnicodeDecodeError, pl.exceptions.PolarsError):
        return False

    if format_name in {"gaze", "gazepoint"}:
        timestamp = "TIME" if "TIME" in columns else "time"
        return any(
            {timestamp, x, y}.issubset(columns)
            for x, y in (("LPOGX", "LPOGY"), ("RPOGX", "RPOGY"))
        )
    if format_name == "tobii":
        timestamp = (
            "Eyetracker timestamp"
            if "Eyetracker timestamp" in columns
            else "Recording timestamp"
        )
        return any(
            {timestamp, x, y}.issubset(columns)
            for x, y in (
                ("Gaze2d_Left.x", "Gaze2d_Left.y"),
                ("Gaze2d_Right.x", "Gaze2d_Right.y"),
            )
        )
    return "webgazer_data" in columns


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_recording(
    recording: EyeRecording,
    *,
    destination: Path,
    prefix: str,
    extra_metadata: dict | None = None,
) -> tuple[Path, Path]:
    # Vendor readers normalize each stream once. Avoid an additional full
    # sort/copy here, which is significant for long recordings.
    columns = list(recording.samples.columns)
    tsv_path = destination / f"{prefix}_physio.tsv.gz"
    json_path = destination / f"{prefix}_physio.json"
    destination.mkdir(parents=True, exist_ok=True)
    write_tsv(
        tsv_path,
        recording.samples,
        include_header=False,
        compressed=True,
    )

    metadata: dict = {
        "SamplingFrequency": recording.sampling_frequency,
        "StartTime": 0.0,
        "Columns": columns,
        "PhysioType": "eyetrack",
        "RecordedEye": recording.recorded_eye,
        "SampleCoordinateSystem": "custom",
        "SampleCoordinateSystemDescription": recording.coordinate_description,
        "timestamp": {
            "Description": (
                "Continuously increasing timestamp issued by the eye tracker."
            ),
            "Units": recording.timestamp_unit,
            "Origin": "Eye-tracker clock",
        },
        "x_coordinate": {
            "LongName": "Gaze position (x)",
            "Description": recording.coordinate_description,
            "Units": recording.coordinate_unit,
        },
        "y_coordinate": {
            "LongName": "Gaze position (y)",
            "Description": recording.coordinate_description,
            "Units": recording.coordinate_unit,
        },
    }
    task_match = re.search(r"_task-([^_]+)", prefix)
    if task_match:
        metadata["TaskName"] = task_match.group(1)
    if recording.manufacturer:
        metadata["Manufacturer"] = recording.manufacturer
    if "pupil_size" in columns:
        metadata["pupil_size"] = {
            "Description": recording.pupil_description
            or "Pupil size reported by the eye tracker.",
            "Units": recording.pupil_unit or "arbitrary",
        }
    for column in columns:
        if column not in metadata:
            metadata[column] = {
                "Description": (
                    "Auxiliary value retained by Pyxations from the source "
                    f"recording ({column})."
                )
            }
    if extra_metadata:
        metadata.update(extra_metadata)
    _write_json(json_path, metadata)
    return tsv_path, json_path


def _write_physio_events(
    events: pl.DataFrame,
    *,
    destination: Path,
    prefix: str,
) -> tuple[Path, Path] | None:
    if events is None or events.is_empty():
        return None
    frame = events.clone()
    required = ["onset", "duration", "trial_type"]
    for column in required:
        if column not in frame:
            frame = frame.with_columns(pl.lit("n/a").alias(column))
    columns = required + [column for column in frame.columns if column not in required]
    frame = frame.select(columns)
    path = destination / f"{prefix}_physioevents.tsv.gz"
    json_path = destination / f"{prefix}_physioevents.json"
    destination.mkdir(parents=True, exist_ok=True)
    write_tsv(path, frame, include_header=False, compressed=True)
    metadata = {
        "Columns": columns,
        "Description": (
            "Messages and tracker-reported events retained from the raw "
            "eye-tracking recording."
        ),
        "OnsetSource": "timestamp",
    }
    for column in columns:
        metadata[column] = {
            "Description": (
                f"Raw eye-tracking event field retained by Pyxations ({column})."
            )
        }
    _write_json(json_path, metadata)
    return path, json_path


def _events_for_recording(
    events: pl.DataFrame,
    *,
    recorded_eye: str,
    available_eyes: Sequence[str],
) -> pl.DataFrame:
    """Select events associated with one per-eye physiological recording."""

    if events is None or events.is_empty() or "eye" not in events:
        return events
    eye_values = {
        "left": {"l", "left"},
        "right": {"r", "right"},
        "cyclopean": {"c", "cyclopean", "both", "binocular"},
    }.get(recorded_eye, {recorded_eye.lower()})
    shared_values = {"", "n/a", "na", "none", "unknown", "all"}
    if "cyclopean" not in available_eyes:
        # Some trackers report only a device-wide blink/event stream while
        # exposing separate left/right sample streams. Retain that shared
        # stream alongside each corresponding recording.
        shared_values.update({"c", "cyclopean", "both", "binocular"})
    return events.filter(
        pl.col("eye")
        .fill_null("n/a")
        .cast(pl.String)
        .str.to_lowercase()
        .is_in(eye_values | shared_values)
    )


def _read_behavioral_table(
    path: Path,
    *,
    behavioral_column_map: Mapping[str, str] | None = None,
) -> pl.DataFrame | None:
    try:
        return read_behavioral_events(
            path,
            column_map=behavioral_column_map,
        )
    except (OSError, UnicodeDecodeError, pl.exceptions.PolarsError):
        return None


def _prepare_task_events(
    files: Sequence[Path],
    *,
    source_root: Path,
    primary_source: Path,
    format_name: str,
    behavioral_column_map: Mapping[str, str] | None = None,
) -> pl.DataFrame | None:
    candidates = list(files)
    if format_name == "webgazer":
        candidates.insert(0, primary_source)
    seen = set()
    unique_candidates = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(path)

    tabular_candidates = [
        path
        for path in unique_candidates
        if path.suffix.lower() in {".csv", ".tsv"}
        and (format_name == "webgazer" or not _is_primary_recording(path, format_name))
    ]
    log_candidates = [
        path for path in unique_candidates if path.suffix.lower() == ".log"
    ]
    candidate_groups = [tabular_candidates, log_candidates]

    tables = []
    # Captured before the gaze payload is dropped below, since the numbering is
    # derived from which screens actually carry gaze.
    webgazer_numbering: dict[int, int] = {}
    for group in candidate_groups:
        for path in group:
            table = _read_behavioral_table(
                path,
                behavioral_column_map=behavioral_column_map,
            )
            if table is None or table.is_empty():
                continue
            if format_name == "webgazer" and "webgazer_data" in table:
                webgazer_numbering.update(webgazer_trial_numbering(table))
                table = table.drop(
                    [
                        column
                        for column in (
                            "webgazer_data",
                            "webgazer_targets",
                            "last-estimations",
                            "validation-results",
                        )
                        if column in table
                    ]
                )
                if "trial_index" in table:
                    table = table.unique(
                        subset=["trial_index"], keep="last", maintain_order=True
                    )
            table = table.clone()
            try:
                source_file = path.relative_to(source_root).as_posix()
            except ValueError:
                source_file = path.name
            table = table.with_columns(pl.lit(source_file).alias("source_file"))
            tables.append(table)
        if tables:
            break
    if not tables:
        return None
    events = pl.concat(tables, how="diagonal_relaxed")
    if "trial_number" not in events and "trial_index" in events:
        if webgazer_numbering:
            # Renumber through the same mapping used for the gaze samples, so a
            # trial finds its behavioral row. Screens without gaze, such as
            # instructions, are not trials and get no number.
            events = events.with_columns(
                pl.col("trial_index")
                .cast(pl.Int64, strict=False)
                .replace_strict(
                    webgazer_numbering, default=None, return_dtype=pl.Int64
                )
                .alias("trial_number"),
                pl.col("trial_index").alias("source_trial_index"),
            )
        else:
            events = events.with_columns(pl.col("trial_index").alias("trial_number"))
    if "onset" not in events:
        if "time_elapsed" in events:
            events = events.with_columns(
                (
                    pl.col("time_elapsed").cast(pl.Float64, strict=False)
                    - pl.col("time_elapsed").cast(pl.Float64, strict=False).min()
                )
                .truediv(1_000.0)
                .alias("onset")
            )
        else:
            events = events.with_columns(pl.lit(None, dtype=pl.Float64).alias("onset"))
    if "duration" not in events:
        if "rt" in events:
            events = events.with_columns(
                pl.col("rt")
                .cast(pl.Float64, strict=False)
                .truediv(1_000.0)
                .alias("duration")
            )
        else:
            events = events.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("duration")
            )
    # BIDS tabular files do not support embedded tabs or newlines. Browser
    # experiments commonly store multiline HTML and pretty-printed JSON in
    # behavioral columns, so retain their content in a single-line form.
    string_columns = [
        column for column, dtype in events.schema.items() if dtype == pl.String
    ]
    if string_columns:
        events = events.with_columns(
            pl.col(column)
            .str.replace_all(r"[\t\r\n]+", " ")
            .str.strip_chars()
            .alias(column)
            for column in string_columns
        )
    return events.sort("onset", nulls_last=True, maintain_order=True)


def _write_task_events(
    events: pl.DataFrame | None,
    *,
    destination: Path,
    prefix: str,
) -> tuple[Path, Path] | None:
    if events is None or events.is_empty():
        return None
    frame = events.clone()
    columns = ["onset", "duration"] + [
        column for column in frame.columns if column not in {"onset", "duration"}
    ]
    frame = frame.select(columns)
    path = destination / f"{prefix}_events.tsv"
    json_path = destination / f"{prefix}_events.json"
    destination.mkdir(parents=True, exist_ok=True)
    write_tsv(path, frame, include_header=True, compressed=False)
    metadata = {
        column: {
            "Description": (f"Behavioral trial field retained by Pyxations ({column}).")
        }
        for column in columns
    }
    if "psychopy_onset" in columns:
        metadata["psychopy_onset"] = {
            "Description": (
                "Trial timestamp from PsychoPy's local clock. This value is "
                "retained for provenance and is not assumed to be synchronized "
                "with the eye-tracker clock."
            ),
            "Units": "s",
        }
    if "psychopy_trial_interval" in columns:
        metadata["psychopy_trial_interval"] = {
            "Description": (
                "Time from this PsychoPy New trial record to the next one."
            ),
            "Units": "s",
        }
    if "keypresses" in columns:
        metadata["keypresses"] = {
            "Description": (
                "Ordered keypresses logged by PsychoPy between this trial "
                "record and the next."
            )
        }
    task_match = re.search(r"_task-([^_]+)", prefix)
    if task_match:
        metadata["TaskName"] = task_match.group(1)
    _write_json(json_path, metadata)
    return path, json_path


def write_bids_dataset(
    target_folder_path: str | Path,
    files_folder_path: str | Path,
    dataset_name: str,
    *,
    session_substrings: int = 1,
    format_name: str = "eyelink",
    task_name: str = "eyetracking",
    authors: Sequence[str] | None = None,
    behavioral_column_map: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert supported vendor recordings into a validated BIDS layout.

    Original vendor files are retained under ``sourcedata``. Standardized
    sample-level recordings are emitted as per-eye physiological files.
    """

    format_name = format_name.lower()
    if format_name not in READERS:
        raise ValueError(
            f"Unknown eye-tracking format {format_name!r}; "
            f"choose one of {sorted(READERS)}"
        )
    if session_substrings < 1:
        raise ValueError("session_substrings must be at least 1")

    source_root = Path(files_folder_path)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {source_root}")
    dataset_root = Path(target_folder_path) / dataset_name
    if dataset_root.exists() and any(dataset_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists and is not empty: {dataset_root}. "
                "Pass overwrite=True to replace it."
            )
        source_resolved = source_root.resolve()
        dataset_resolved = dataset_root.resolve()
        if (
            source_resolved == dataset_resolved
            or dataset_resolved in source_resolved.parents
        ):
            raise ValueError(
                "Refusing to overwrite a dataset containing its source files"
            )
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_root,
        dataset_root / "sourcedata",
        copy_function=shutil.copy2,
    )

    primary = [
        path
        for path in sorted(source_root.rglob("*"))
        if path.is_file() and _is_primary_recording(path, format_name)
    ]
    if format_name == "eyelink":
        edf_stems = {
            path.stem.lower() for path in primary if path.suffix.lower() == ".edf"
        }
        primary = [
            path
            for path in primary
            if path.suffix.lower() != ".asc" or path.stem.lower() not in edf_stems
        ]
    if not primary:
        raise ValueError(f"No {format_name} recordings found in {source_root}")

    subject_names = sorted({path.name.split("_")[0] for path in primary})
    subject_map = {
        old: str(index).zfill(4) for index, old in enumerate(subject_names, start=1)
    }
    participant_rows = []
    for old, subject in subject_map.items():
        participant_rows.append(
            {
                "participant_id": f"sub-{subject}",
                "subject_id": subject,
                "old_subject_id": old,
            }
        )

    _write_json(
        dataset_root / "dataset_description.json",
        {
            "Name": dataset_name,
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "raw",
            "Authors": list(authors or ["NeuroLIAA"]),
            "GeneratedBy": [
                {
                    "Name": "Pyxations",
                    "Description": "Multi-vendor eye-tracking conversion to BIDS.",
                }
            ],
        },
    )
    write_tsv(
        dataset_root / "participants.tsv",
        pl.DataFrame(participant_rows),
        include_header=True,
        compressed=False,
    )
    _write_json(
        dataset_root / "participants.json",
        {
            "subject_id": {
                "Description": "Pyxations' zero-padded internal subject identifier."
            },
            "old_subject_id": {
                "Description": "Subject identifier present in the source filename."
            },
        },
    )
    (dataset_root / "README").write_text(
        "Eye-tracking dataset converted to BIDS by Pyxations.\n",
        encoding="utf-8",
        newline="\n",
    )

    for old_subject, subject in subject_map.items():
        subject_primary = [
            path for path in primary if path.name.split("_")[0] == old_subject
        ]
        by_session: dict[str, list[Path]] = {}
        for path in subject_primary:
            session = _session_from_filename(path, session_substrings)
            by_session.setdefault(session, []).append(path)

        all_subject_files = [
            path
            for path in sorted(source_root.rglob("*"))
            if path.is_file() and path.name.split("_")[0] == old_subject
        ]
        for session, session_primary in by_session.items():
            session_sources = [
                path
                for path in all_subject_files
                if _session_from_filename(path, session_substrings) == session
            ]
            for run_index, source in enumerate(session_primary, start=1):
                bundle = _read_source_bundle(source, format_name)
                task = _task_from_filename(source, task_name)
                base = f"sub-{subject}_ses-{session}_task-{task}"
                if len(session_primary) > 1:
                    base += f"_run-{run_index:02d}"
                destination = dataset_root / f"sub-{subject}" / f"ses-{session}" / "beh"
                available_eyes = [
                    recording.recorded_eye for recording in bundle.recordings
                ]
                for eye_index, recording in enumerate(bundle.recordings, start=1):
                    prefix = f"{base}_recording-eye{eye_index}"
                    _write_recording(
                        recording,
                        destination=destination,
                        prefix=prefix,
                        extra_metadata=(
                            {
                                "PyxationsCalibration": frame_payload(
                                    bundle.calibration
                                ),
                                "PyxationsHeader": frame_payload(bundle.header),
                                **bundle.metadata,
                            }
                            if eye_index == 1
                            else None
                        ),
                    )
                    _write_physio_events(
                        _events_for_recording(
                            bundle.events,
                            recorded_eye=recording.recorded_eye,
                            available_eyes=available_eyes,
                        ),
                        destination=destination,
                        prefix=prefix,
                    )
                task_events = _prepare_task_events(
                    session_sources,
                    source_root=source_root,
                    primary_source=source,
                    format_name=format_name,
                    behavioral_column_map=behavioral_column_map,
                )
                _write_task_events(
                    task_events,
                    destination=destination,
                    prefix=base,
                )
    return dataset_root


def _read_bids_table(path: Path, metadata: dict) -> pl.DataFrame:
    return read_tsv(
        path,
        columns=list(metadata["Columns"]),
        has_header=False,
    )


def _milliseconds_per_unit(unit: str | None) -> float:
    normalized = str(unit or "s").lower()
    return {
        "s": 1_000.0,
        "second": 1_000.0,
        "seconds": 1_000.0,
        "ms": 1.0,
        "millisecond": 1.0,
        "milliseconds": 1.0,
        "us": 0.001,
        "µs": 0.001,
        "microsecond": 0.001,
        "microseconds": 0.001,
        "ns": 0.000001,
    }.get(normalized, 1.0)


def read_bids_task_events(session_path: str | Path) -> pl.DataFrame:
    """Read and combine BIDS task-event tables for one raw session."""

    behavior = Path(session_path) / "beh"
    tables = []
    for path in sorted(behavior.glob("*_events.tsv")):
        table = read_tsv(path, has_header=True).with_columns(
            pl.lit(path.name).alias("bids_events_file")
        )
        tables.append(table)
    return pl.concat(tables, how="diagonal_relaxed") if tables else empty_frame()


def read_raw_bids_session(session_path: str | Path) -> SessionTables:
    """Load normalized samples and source events from a raw BIDS session."""

    session = Path(session_path)
    behavior = session / "beh"
    physio_paths = sorted(
        path
        for path in behavior.glob("*_physio.tsv.gz")
        if "_physioevents." not in path.name
    )
    if not physio_paths:
        raise FileNotFoundError(f"No raw BIDS physio files found in {behavior}")

    sample_streams = []
    first_metadata = None
    frequencies = []
    for path in physio_paths:
        metadata = json.loads(
            path.with_suffix("").with_suffix(".json").read_text(encoding="utf-8")
        )
        first_metadata = first_metadata or metadata
        frequencies.append(float(metadata["SamplingFrequency"]))
        frame = _read_bids_table(path, metadata)
        time_scale = _milliseconds_per_unit(metadata.get("timestamp", {}).get("Units"))
        stream = frame.select(
            pl.col("timestamp")
            .cast(pl.Float64, strict=False)
            .mul(time_scale)
            .alias("tSample")
        )
        eye = metadata.get("RecordedEye", "cyclopean")
        prefix = {"left": "L", "right": "R"}.get(eye)
        if prefix:
            stream = stream.with_columns(
                frame.get_column("x_coordinate")
                .cast(pl.Float64, strict=False)
                .alias(f"{prefix}X"),
                frame.get_column("y_coordinate")
                .cast(pl.Float64, strict=False)
                .alias(f"{prefix}Y"),
            )
            if "pupil_size" in frame:
                stream = stream.with_columns(
                    frame.get_column("pupil_size")
                    .cast(pl.Float64, strict=False)
                    .alias(f"{prefix}Pupil")
                )
        else:
            stream = stream.with_columns(
                frame.get_column("x_coordinate")
                .cast(pl.Float64, strict=False)
                .alias("X"),
                frame.get_column("y_coordinate")
                .cast(pl.Float64, strict=False)
                .alias("Y"),
            )
            if "pupil_size" in frame:
                stream = stream.with_columns(
                    frame.get_column("pupil_size")
                    .cast(pl.Float64, strict=False)
                    .alias("Pupil")
                )
        auxiliary = {
            "calibration_index": "Calib_index",
            "line_number": "Line_number",
            "trial_number": "trial_number",
        }
        for source_column, target_column in auxiliary.items():
            if source_column in frame and target_column not in stream:
                stream = stream.with_columns(
                    frame.get_column(source_column)
                    .cast(pl.Float64, strict=False)
                    .alias(target_column)
                )
        sample_streams.append(stream)

    samples = sample_streams[0]
    for stream in sample_streams[1:]:
        duplicate_auxiliary = [
            column
            for column in ("Calib_index", "Line_number", "trial_number")
            if column in stream and column in samples
        ]
        stream = stream.drop(duplicate_auxiliary)
        if samples.height == stream.height and samples.get_column("tSample").equals(
            stream.get_column("tSample")
        ):
            samples = samples.hstack(stream.drop("tSample"))
        else:
            samples = samples.join(stream, on="tSample", how="full", coalesce=True)
    samples = samples.sort("tSample", maintain_order=True)
    sampling_frequency = float(np.nanmedian(frequencies))
    additions = [pl.lit(sampling_frequency).alias("Rate_recorded")]
    if "Calib_index" not in samples:
        additions.append(pl.lit(1).alias("Calib_index"))
    if "Line_number" not in samples:
        additions.append(
            pl.int_range(0, samples.height, eager=True).alias("Line_number")
        )
    if {"LX", "RX"}.intersection(samples.columns):
        eyes_recorded = (
            "LR"
            if {"LX", "RX"}.issubset(samples.columns)
            else "L"
            if "LX" in samples
            else "R"
        )
        additions.append(pl.lit(eyes_recorded).alias("Eyes_recorded"))
    samples = samples.with_columns(additions)

    event_frames = []
    for path in sorted(behavior.glob("*_physioevents.tsv.gz")):
        metadata = json.loads(
            path.with_suffix("").with_suffix(".json").read_text(encoding="utf-8")
        )
        event_frames.append(_read_bids_table(path, metadata))
    events = (
        pl.concat(event_frames, how="diagonal_relaxed")
        if event_frames
        else empty_frame()
    )
    if not events.is_empty():
        # Device-wide messages may be associated with every per-eye
        # physiological recording. Reconstruct them only once in memory.
        events = events.unique(maintain_order=True)
    time_scale = _milliseconds_per_unit(
        first_metadata.get("timestamp", {}).get("Units")
    )
    if not events.is_empty():
        events = events.with_columns(
            pl.col("onset")
            .cast(pl.Float64, strict=False)
            .mul(time_scale)
            .alias("tStart")
        )
        if "end_timestamp" in events:
            events = events.with_columns(
                pl.col("end_timestamp")
                .cast(pl.Float64, strict=False)
                .mul(time_scale)
                .alias("tEnd")
            )
        else:
            duration = (
                pl.col("duration").cast(pl.Float64, strict=False)
                if "duration" in events
                else pl.lit(None, dtype=pl.Float64)
            )
            events = events.with_columns(
                (pl.col("tStart") + duration * 1_000.0).alias("tEnd")
            )
        events = events.with_columns(
            (pl.col("tEnd") - pl.col("tStart")).alias("duration_ms")
        )

    def selected(event_type: str, mapping: dict[str, str]) -> pl.DataFrame:
        if events.is_empty() or "trial_type" not in events:
            return pl.DataFrame({column: [] for column in mapping.values()})
        result = events.filter(pl.col("trial_type") == event_type)
        available = {
            source: target for source, target in mapping.items() if source in result
        }
        return result.select(list(available)).rename(available)

    common = {
        "eye": "eye",
        "tStart": "tStart",
        "tEnd": "tEnd",
        "duration_ms": "duration",
        "line_number": "Line_number",
        "calibration_index": "Calib_index",
    }
    fixations = selected(
        "fixation",
        {
            **common,
            "x_avg": "xAvg",
            "y_avg": "yAvg",
            "pupil_avg": "pupilAvg",
        },
    )
    saccades = selected(
        "saccade",
        {
            **common,
            "x_start": "xStart",
            "y_start": "yStart",
            "x_end": "xEnd",
            "y_end": "yEnd",
            "amplitude": "ampDeg",
            "peak_velocity": "vPeak",
        },
    )
    blinks = selected("blink", common)
    messages = selected(
        "message",
        {
            "tStart": "timestamp",
            "message": "message",
            "line_number": "Line_number",
            "calibration_index": "Calib_index",
        },
    )
    if "Eyes_recorded" in samples:
        session_eye = samples.get_column("Eyes_recorded").item(0)
        for table in (fixations, saccades, blinks, messages):
            table = table.with_columns(
                pl.lit(session_eye).alias("Eyes_recorded"),
                pl.lit(sampling_frequency).alias("Rate_recorded"),
            )
            if table is fixations:
                fixations = table
            elif table is saccades:
                saccades = table
            elif table is blinks:
                blinks = table
            else:
                messages = table

    return SessionTables(
        samples=samples,
        fixations=fixations,
        saccades=saccades,
        blinks=blinks,
        messages=messages,
        calibration=payload_frame(first_metadata.get("PyxationsCalibration")),
        header=payload_frame(first_metadata.get("PyxationsHeader")),
        behavioral_events=read_bids_task_events(session),
        sampling_frequency=sampling_frequency,
        screen_width=first_metadata.get("ScreenWidth"),
        screen_height=first_metadata.get("ScreenHeight"),
    )


def validator_command() -> list[str] | None:
    """Return the available official BIDS Validator command."""

    executable = shutil.which("bids-validator")
    if executable:
        return [executable]
    deno = shutil.which("deno")
    if deno:
        return [
            deno,
            "run",
            "-ERWN",
            f"jsr:@bids/validator@{BIDS_VALIDATOR_VERSION}",
        ]
    return None


def validate_bids_dataset(
    dataset_path: str | Path, *, command: Sequence[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Validate a dataset with the official BIDS Validator.

    Raises
    ------
    RuntimeError
        If no validator executable or Deno runtime is available.
    BIDSValidationError
        If validation reports one or more errors.
    """

    dataset = Path(dataset_path)
    if not dataset.is_dir():
        raise FileNotFoundError(f"BIDS dataset not found: {dataset}")
    validator = list(command) if command is not None else validator_command()
    if validator is None:
        raise RuntimeError(
            "The official BIDS Validator is unavailable. Install the "
            "`bids-validator` command or the Deno runtime."
        )
    result = subprocess.run(
        [*validator, str(dataset), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        report = "\n".join(part for part in (result.stdout, result.stderr) if part)
        raise BIDSValidationError(
            f"BIDS validation failed for {dataset} (exit {result.returncode}):\n"
            f"{report}"
        )
    return result
