"""BIDS eye-tracking writing and validation utilities.

The writer targets the eye-tracking additions in BIDS 1.11.1. Vendor files are
kept under ``sourcedata`` while standardized, per-eye physiological recordings
are written to the raw BIDS dataset.
"""

from __future__ import annotations

import json
import gzip
import io
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


BIDS_VERSION = "1.11.1"
BIDS_VALIDATOR_VERSION = "3.0.1"


class BIDSValidationError(RuntimeError):
    """Raised when the official BIDS Validator reports an invalid dataset."""


@dataclass
class EyeRecording:
    """Canonical sample-level recording for one eye."""

    samples: pd.DataFrame
    recorded_eye: str
    sampling_frequency: float
    timestamp_unit: str
    coordinate_unit: str
    coordinate_description: str
    pupil_unit: str | None = None
    pupil_description: str | None = None
    manufacturer: str | None = None

    def normalized(self) -> "EyeRecording":
        required = ["timestamp", "x_coordinate", "y_coordinate"]
        missing = [column for column in required if column not in self.samples]
        if missing:
            raise ValueError(f"Eye recording is missing required columns: {missing}")
        if self.recorded_eye not in {"left", "right", "cyclopean"}:
            raise ValueError(f"Unsupported RecordedEye value: {self.recorded_eye}")

        columns = required + (
            ["pupil_size"] if "pupil_size" in self.samples.columns else []
        )
        columns += [
            column for column in self.samples.columns if column not in columns
        ]
        samples = self.samples.loc[:, columns].copy()
        for column in required + (
            ["pupil_size"] if "pupil_size" in samples.columns else []
        ):
            samples[column] = pd.to_numeric(samples[column], errors="coerce")
        samples.replace([np.inf, -np.inf], np.nan, inplace=True)
        if "pupil_size" in samples:
            samples.loc[samples["pupil_size"] <= 0, "pupil_size"] = np.nan
        samples.dropna(subset=["timestamp"], inplace=True)
        samples.sort_values("timestamp", inplace=True, kind="stable")
        samples.drop_duplicates("timestamp", keep="first", inplace=True)
        samples.reset_index(drop=True, inplace=True)
        if samples.empty:
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
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    header: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)


def _bids_label(value: str, *, fallback: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "", str(value))
    return label or fallback


def _task_from_filename(path: Path, default: str) -> str:
    match = re.search(r"(?:^|_)task-([A-Za-z0-9]+)", path.stem)
    return _bids_label(match.group(1) if match else default, fallback="eyetracking")


def _session_from_filename(path: Path, session_substrings: int) -> str:
    tokens = path.stem.split("_")
    selected = tokens[1 : 1 + session_substrings]
    raw = "".join(selected) if selected else "1"
    if raw.lower().startswith("ses-"):
        raw = raw[4:]
    return _bids_label(raw, fallback="1")


def _sampling_frequency(
    timestamps: pd.Series | Sequence[float], *, units_per_second: float
) -> float:
    numeric = pd.to_numeric(pd.Series(timestamps), errors="coerce").dropna()
    differences = numeric.sort_values().diff()
    differences = differences[differences > 0]
    if differences.empty:
        raise ValueError("At least two distinct timestamps are needed")
    return float(units_per_second / differences.median())


def _recording(
    data: pd.DataFrame,
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
    samples = data.loc[:, list(columns)].rename(columns=columns)
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
    path: Path, *, data: pd.DataFrame | None = None
) -> list[EyeRecording]:
    data = pd.read_csv(path) if data is None else data
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


def _read_tobii(
    path: Path, *, data: pd.DataFrame | None = None
) -> list[EyeRecording]:
    data = pd.read_csv(path, sep="\t") if data is None else data
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
    return recordings


def _read_webgazer(
    path: Path, *, source: pd.DataFrame | None = None
) -> list[EyeRecording]:
    source = pd.read_csv(path) if source is None else source
    if "webgazer_data" not in source:
        raise ValueError(f"{path} does not contain a webgazer_data column")

    samples: list[dict[str, float]] = []
    for row in source.loc[source["webgazer_data"].notna()].itertuples(index=False):
        payload = getattr(row, "webgazer_data")
        try:
            gaze_samples = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid WebGazer JSON in {path}") from error
        base_time = float(getattr(row, "time_elapsed", 0.0))
        trial_number = getattr(row, "trial_index", np.nan)
        for sample in gaze_samples:
            if not {"t", "x", "y"}.issubset(sample):
                continue
            samples.append(
                {
                    "timestamp": base_time + float(sample["t"]),
                    "x_coordinate": sample["x"],
                    "y_coordinate": sample["y"],
                    "trial_number": trial_number,
                }
            )
    frame = pd.DataFrame(samples)
    if frame.empty:
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
                rate_match = re.search(
                    r"RATE\s+([0-9]+(?:\.[0-9]+)?)\s+TRACKING", line
                )
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
                numeric = pd.to_numeric(
                    pd.Series(fields[:7]), errors="coerce"
                ).tolist()
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
        frame = pd.DataFrame(rows)
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
        events=pd.DataFrame(events),
        calibration=pd.DataFrame(calibration),
        header=pd.DataFrame(header),
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
        data = pd.read_csv(path)
        events = pd.DataFrame()
        if {"TIME", "BKDUR"}.issubset(data.columns):
            blink = data.loc[
                pd.to_numeric(data["BKDUR"], errors="coerce").fillna(0) > 0
            ]
            if not blink.empty:
                duration = pd.to_numeric(
                    blink["BKDUR"], errors="coerce"
                )
                end = pd.to_numeric(blink["TIME"], errors="coerce")
                events = pd.DataFrame(
                    {
                        "onset": end - duration,
                        "duration": duration,
                        "trial_type": "blink",
                        "message": "n/a",
                        "eye": "cyclopean",
                        "end_timestamp": end,
                    }
                )
        return SourceRecordingBundle(
            recordings=_read_gazepoint(path, data=data),
            events=events,
        )
    if format_name == "tobii":
        data = pd.read_csv(path, sep="\t")
        return SourceRecordingBundle(
            recordings=_read_tobii(path, data=data)
        )
    if format_name == "webgazer":
        data = pd.read_csv(path)
        calibration = (
            data.loc[data["rastoc-type"] == "calibration-stimulus"].copy()
            if "rastoc-type" in data
            else pd.DataFrame()
        )
        if not calibration.empty:
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
            calibration = calibration.loc[:, compact_columns]
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
            pd.read_csv(
                path,
                sep="\t" if format_name == "tobii" else ",",
                nrows=0,
            ).columns
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
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


def _frame_payload(frame: pd.DataFrame) -> dict | None:
    if frame is None or frame.empty:
        return None
    safe = frame.astype(object).where(pd.notna(frame), None)
    return {
        "columns": [str(column) for column in safe.columns],
        "data": json.loads(safe.to_json(orient="values")),
    }


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
    with tsv_path.open("wb") as binary_stream:
        with gzip.GzipFile(
            filename="", fileobj=binary_stream, mode="wb", mtime=0
        ) as gzip_stream:
            with io.TextIOWrapper(
                gzip_stream, encoding="utf-8", newline=""
            ) as text_stream:
                recording.samples.to_csv(
                    text_stream,
                    sep="\t",
                    header=False,
                    index=False,
                    na_rep="n/a",
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
    events: pd.DataFrame,
    *,
    destination: Path,
    prefix: str,
) -> tuple[Path, Path] | None:
    if events is None or events.empty:
        return None
    frame = events.copy()
    required = ["onset", "duration", "trial_type"]
    for column in required:
        if column not in frame:
            frame[column] = "n/a"
    columns = required + [
        column for column in frame.columns if column not in required
    ]
    frame = frame.loc[:, columns]
    path = destination / f"{prefix}_physioevents.tsv.gz"
    json_path = destination / f"{prefix}_physioevents.json"
    destination.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as binary_stream:
        with gzip.GzipFile(
            filename="", fileobj=binary_stream, mode="wb", mtime=0
        ) as gzip_stream:
            with io.TextIOWrapper(
                gzip_stream, encoding="utf-8", newline=""
            ) as text_stream:
                frame.to_csv(
                    text_stream,
                    sep="\t",
                    header=False,
                    index=False,
                    na_rep="n/a",
                )
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
                "Raw eye-tracking event field retained by Pyxations "
                f"({column})."
            )
        }
    _write_json(json_path, metadata)
    return path, json_path


def _events_for_recording(
    events: pd.DataFrame,
    *,
    recorded_eye: str,
    available_eyes: Sequence[str],
) -> pd.DataFrame:
    """Select events associated with one per-eye physiological recording."""

    if events is None or events.empty or "eye" not in events:
        return events
    normalized = events["eye"].fillna("n/a").astype(str).str.lower()
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
    return events.loc[
        normalized.isin(eye_values | shared_values)
    ].reset_index(drop=True)


def _read_behavioral_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return None
    return None


def _prepare_task_events(
    files: Sequence[Path],
    *,
    source_root: Path,
    primary_source: Path,
    format_name: str,
) -> pd.DataFrame | None:
    tables = []
    candidates = list(files)
    if format_name == "webgazer":
        candidates.insert(0, primary_source)
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.resolve() == primary_source.resolve() and format_name != "webgazer":
            continue
        table = _read_behavioral_table(path)
        if table is None or table.empty:
            continue
        if _is_primary_recording(path, format_name) and format_name != "webgazer":
            continue
        if format_name == "webgazer" and "webgazer_data" in table:
            table = table.drop(
                columns=[
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
                table = table.drop_duplicates("trial_index", keep="last")
        table = table.copy()
        try:
            source_file = path.relative_to(source_root).as_posix()
        except ValueError:
            source_file = path.name
        table["source_file"] = source_file
        tables.append(table)
    if not tables:
        return None
    events = pd.concat(tables, ignore_index=True, sort=False)
    if "trial_number" not in events and "trial_index" in events:
        events["trial_number"] = events["trial_index"]
    if "onset" not in events:
        if "time_elapsed" in events:
            elapsed = pd.to_numeric(events["time_elapsed"], errors="coerce")
            events["onset"] = (elapsed - elapsed.min()) / 1_000.0
        else:
            events["onset"] = np.nan
    if "duration" not in events:
        if "rt" in events:
            events["duration"] = (
                pd.to_numeric(events["rt"], errors="coerce") / 1_000.0
            )
        else:
            events["duration"] = np.nan
    # BIDS tabular files do not support embedded tabs or newlines. Browser
    # experiments commonly store multiline HTML and pretty-printed JSON in
    # behavioral columns, so retain their content in a single-line form.
    object_columns = events.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        events[column] = events[column].map(
            lambda value: re.sub(r"[\t\r\n]+", " ", value).strip()
            if isinstance(value, str)
            else value
        )
    events = events.sort_values("onset", na_position="last").reset_index(
        drop=True
    )
    return events


def _write_task_events(
    events: pd.DataFrame | None,
    *,
    destination: Path,
    prefix: str,
) -> tuple[Path, Path] | None:
    if events is None or events.empty:
        return None
    frame = events.copy()
    columns = ["onset", "duration"] + [
        column
        for column in frame.columns
        if column not in {"onset", "duration"}
    ]
    frame = frame.loc[:, columns]
    path = destination / f"{prefix}_events.tsv"
    json_path = destination / f"{prefix}_events.json"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a")
    metadata = {
        column: {
            "Description": (
                "Behavioral trial field retained by Pyxations "
                f"({column})."
            )
        }
        for column in columns
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
        if source_resolved == dataset_resolved or dataset_resolved in source_resolved.parents:
            raise ValueError("Refusing to overwrite a dataset containing its source files")
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
        edf_stems = {path.stem.lower() for path in primary if path.suffix.lower() == ".edf"}
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
    pd.DataFrame(participant_rows).to_csv(
        dataset_root / "participants.tsv", sep="\t", index=False, na_rep="n/a"
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
                destination = (
                    dataset_root / f"sub-{subject}" / f"ses-{session}" / "beh"
                )
                available_eyes = [
                    recording.recorded_eye
                    for recording in bundle.recordings
                ]
                for eye_index, recording in enumerate(
                    bundle.recordings, start=1
                ):
                    prefix = f"{base}_recording-eye{eye_index}"
                    _write_recording(
                        recording,
                        destination=destination,
                        prefix=prefix,
                        extra_metadata=(
                            {
                                "PyxationsCalibration": _frame_payload(
                                    bundle.calibration
                                ),
                                "PyxationsHeader": _frame_payload(
                                    bundle.header
                                ),
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
                )
                _write_task_events(
                    task_events,
                    destination=destination,
                    prefix=base,
                )
    return dataset_root


@dataclass
class RawBIDSSession:
    """Raw BIDS tables needed for derivative computation."""

    samples: pd.DataFrame
    fixations: pd.DataFrame
    saccades: pd.DataFrame
    blinks: pd.DataFrame
    messages: pd.DataFrame
    calibration: pd.DataFrame
    header: pd.DataFrame
    behavioral_events: pd.DataFrame
    sampling_frequency: float
    screen_width: int | None = None
    screen_height: int | None = None


def _payload_frame(payload) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload.get("data", []), columns=payload.get("columns", []))


def _read_bids_table(path: Path, metadata: dict) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(metadata["Columns"]),
        na_values=["n/a"],
        keep_default_na=True,
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


def read_bids_task_events(session_path: str | Path) -> pd.DataFrame:
    """Read and combine BIDS task-event tables for one raw session."""

    behavior = Path(session_path) / "beh"
    tables = []
    for path in sorted(behavior.glob("*_events.tsv")):
        table = pd.read_csv(
            path,
            sep="\t",
            na_values=["n/a"],
            keep_default_na=True,
        )
        table["bids_events_file"] = path.name
        tables.append(table)
    return (
        pd.concat(tables, ignore_index=True, sort=False)
        if tables
        else pd.DataFrame()
    )


def read_raw_bids_session(session_path: str | Path) -> RawBIDSSession:
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
        time_scale = _milliseconds_per_unit(
            metadata.get("timestamp", {}).get("Units")
        )
        stream = pd.DataFrame(
            {
                "tSample": pd.to_numeric(
                    frame["timestamp"], errors="coerce"
                )
                * time_scale
            }
        )
        eye = metadata.get("RecordedEye", "cyclopean")
        prefix = {"left": "L", "right": "R"}.get(eye)
        if prefix:
            stream[f"{prefix}X"] = pd.to_numeric(
                frame["x_coordinate"], errors="coerce"
            )
            stream[f"{prefix}Y"] = pd.to_numeric(
                frame["y_coordinate"], errors="coerce"
            )
            if "pupil_size" in frame:
                stream[f"{prefix}Pupil"] = pd.to_numeric(
                    frame["pupil_size"], errors="coerce"
                )
        else:
            stream["X"] = pd.to_numeric(
                frame["x_coordinate"], errors="coerce"
            )
            stream["Y"] = pd.to_numeric(
                frame["y_coordinate"], errors="coerce"
            )
            if "pupil_size" in frame:
                stream["Pupil"] = pd.to_numeric(
                    frame["pupil_size"], errors="coerce"
                )
        auxiliary = {
            "calibration_index": "Calib_index",
            "line_number": "Line_number",
            "trial_number": "trial_number",
        }
        for source_column, target_column in auxiliary.items():
            if source_column in frame and target_column not in stream:
                stream[target_column] = pd.to_numeric(
                    frame[source_column], errors="coerce"
                )
        sample_streams.append(stream)

    samples = sample_streams[0]
    for stream in sample_streams[1:]:
        duplicate_auxiliary = [
            column
            for column in ("Calib_index", "Line_number", "trial_number")
            if column in stream and column in samples
        ]
        stream = stream.drop(columns=duplicate_auxiliary)
        samples = samples.merge(stream, on="tSample", how="outer", sort=True)
    samples.sort_values("tSample", inplace=True, kind="stable")
    samples.reset_index(drop=True, inplace=True)
    sampling_frequency = float(np.nanmedian(frequencies))
    samples["Rate_recorded"] = sampling_frequency
    if "Calib_index" not in samples:
        samples["Calib_index"] = 1
    if "Line_number" not in samples:
        samples["Line_number"] = np.arange(len(samples))
    if {"LX", "RX"}.intersection(samples.columns):
        samples["Eyes_recorded"] = (
            "LR"
            if {"LX", "RX"}.issubset(samples.columns)
            else "L"
            if "LX" in samples
            else "R"
        )

    event_frames = []
    for path in sorted(behavior.glob("*_physioevents.tsv.gz")):
        metadata = json.loads(
            path.with_suffix("").with_suffix(".json").read_text(encoding="utf-8")
        )
        event_frames.append(_read_bids_table(path, metadata))
    events = (
        pd.concat(event_frames, ignore_index=True, sort=False)
        if event_frames
        else pd.DataFrame()
    )
    if not events.empty:
        # Device-wide messages may be associated with every per-eye
        # physiological recording. Reconstruct them only once in memory.
        events = events.drop_duplicates(ignore_index=True)
    time_scale = _milliseconds_per_unit(
        first_metadata.get("timestamp", {}).get("Units")
    )
    if not events.empty:
        events["tStart"] = (
            pd.to_numeric(events["onset"], errors="coerce") * time_scale
        )
        if "end_timestamp" in events:
            events["tEnd"] = (
                pd.to_numeric(events["end_timestamp"], errors="coerce")
                * time_scale
            )
        else:
            events["tEnd"] = events["tStart"] + (
                pd.to_numeric(events.get("duration"), errors="coerce")
                * 1_000.0
            )
        events["duration_ms"] = events["tEnd"] - events["tStart"]

    def selected(event_type: str, mapping: dict[str, str]) -> pd.DataFrame:
        if events.empty or "trial_type" not in events:
            return pd.DataFrame(columns=list(mapping.values()))
        result = events.loc[events["trial_type"] == event_type].copy()
        available = {
            source: target
            for source, target in mapping.items()
            if source in result
        }
        return result.loc[:, list(available)].rename(columns=available)

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
        for table in (fixations, saccades, blinks, messages):
            table["Eyes_recorded"] = samples["Eyes_recorded"].iloc[0]
            table["Rate_recorded"] = sampling_frequency

    return RawBIDSSession(
        samples=samples,
        fixations=fixations,
        saccades=saccades,
        blinks=blinks,
        messages=messages,
        calibration=_payload_frame(
            first_metadata.get("PyxationsCalibration")
        ),
        header=_payload_frame(first_metadata.get("PyxationsHeader")),
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
