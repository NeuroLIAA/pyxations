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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
        samples = self.samples.loc[:, columns].copy()
        for column in columns:
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


def _read_gazepoint(path: Path) -> list[EyeRecording]:
    data = pd.read_csv(path)
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


def _read_tobii(path: Path) -> list[EyeRecording]:
    data = pd.read_csv(path, sep="\t")
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


def _read_webgazer(path: Path) -> list[EyeRecording]:
    source = pd.read_csv(path)
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
        for sample in gaze_samples:
            if not {"t", "x", "y"}.issubset(sample):
                continue
            samples.append(
                {
                    "timestamp": base_time + float(sample["t"]),
                    "x_coordinate": sample["x"],
                    "y_coordinate": sample["y"],
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


def _read_eyelink(path: Path) -> list[EyeRecording]:
    with tempfile.TemporaryDirectory(prefix="pyxations-edf-") as directory:
        ascii_path = _eyelink_ascii_path(path, Path(directory))
        left: list[dict[str, float]] = []
        right: list[dict[str, float]] = []
        recorded = ""
        declared_frequency: float | None = None

        with ascii_path.open("r", encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                rate_match = re.search(
                    r"RATE\s+([0-9]+(?:\.[0-9]+)?)\s+TRACKING", line
                )
                if rate_match:
                    declared_frequency = float(rate_match.group(1))
                if "!MODE RECORD" in line:
                    eye_match = re.search(r"\b(LR|RL|L|R)\s*$", line)
                    if eye_match:
                        recorded = eye_match.group(1)
                    continue

                fields = line.split()
                if not fields or not re.fullmatch(r"-?\d+(?:\.\d+)?", fields[0]):
                    continue
                numeric = pd.to_numeric(pd.Series(fields), errors="coerce").tolist()
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
                        }
                    )
                    right.append(
                        {
                            "timestamp": numeric[0],
                            "x_coordinate": numeric[4],
                            "y_coordinate": numeric[5],
                            "pupil_size": numeric[6],
                        }
                    )
                elif "R" in eye_mode and len(numeric) >= 4:
                    right.append(
                        {
                            "timestamp": numeric[0],
                            "x_coordinate": numeric[1],
                            "y_coordinate": numeric[2],
                            "pupil_size": numeric[3],
                        }
                    )
                elif len(numeric) >= 4:
                    left.append(
                        {
                            "timestamp": numeric[0],
                            "x_coordinate": numeric[1],
                            "y_coordinate": numeric[2],
                            "pupil_size": numeric[3],
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
    return recordings


READERS = {
    "eyelink": _read_eyelink,
    "gazepoint": _read_gazepoint,
    "gaze": _read_gazepoint,
    "tobii": _read_tobii,
    "webgazer": _read_webgazer,
}

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


def _write_recording(
    recording: EyeRecording,
    *,
    destination: Path,
    prefix: str,
) -> tuple[Path, Path]:
    recording = recording.normalized()
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
    if recording.manufacturer:
        metadata["Manufacturer"] = recording.manufacturer
    if "pupil_size" in columns:
        metadata["pupil_size"] = {
            "Description": recording.pupil_description
            or "Pupil size reported by the eye tracker.",
            "Units": recording.pupil_unit or "arbitrary",
        }
    _write_json(json_path, metadata)
    return tsv_path, json_path


def _copy_sources(
    files: Iterable[Path],
    *,
    destination: Path,
    format_name: str,
) -> None:
    for source in files:
        suffix = source.suffix.lower()
        if _is_primary_recording(source, format_name):
            tags = ["ET"]
            if format_name == "webgazer":
                # A jsPsych/WebGazer export contains both gaze and behavior.
                tags.append("behavioral")
        elif suffix == ".bdf":
            tags = ["EEG"]
        else:
            tags = ["behavioral"]
        for tag in tags:
            folder = destination / tag
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)


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
            source_destination = (
                dataset_root
                / "sourcedata"
                / f"sub-{subject}"
                / f"ses-{session}"
            )
            _copy_sources(
                session_sources,
                destination=source_destination,
                format_name=format_name,
            )

            for run_index, source in enumerate(session_primary, start=1):
                recordings = READERS[format_name](source)
                task = _task_from_filename(source, task_name)
                base = f"sub-{subject}_ses-{session}_task-{task}"
                if len(session_primary) > 1:
                    base += f"_run-{run_index:02d}"
                destination = (
                    dataset_root / f"sub-{subject}" / f"ses-{session}" / "beh"
                )
                for eye_index, recording in enumerate(recordings, start=1):
                    prefix = f"{base}_recording-eye{eye_index}"
                    _write_recording(
                        recording, destination=destination, prefix=prefix
                    )
    return dataset_root


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
