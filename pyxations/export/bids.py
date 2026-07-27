"""BIDS Derivatives storage for processed eye-tracking data."""

from __future__ import annotations

import json
import gzip
import io
import math
import re
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import polars as pl

from pyxations.bids import BIDS_VERSION


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return json.loads(json.dumps(value, default=str))
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, (str, int)) else str(value)


def _frame_payload(frame: pd.DataFrame | pl.DataFrame | None) -> dict:
    if frame is None:
        return {"Columns": [], "Records": []}
    if isinstance(frame, pl.DataFrame):
        columns = frame.columns
        source_records = frame.to_dicts()
    else:
        columns = list(frame.columns)
        source_records = frame.to_dict(orient="records")
    records = [
        {column: _json_value(value) for column, value in row.items()}
        for row in source_records
    ]
    return {"Columns": columns, "Records": records}


def _payload_frame(payload: Mapping | None) -> pl.DataFrame:
    if not payload:
        return pl.DataFrame()
    columns = list(payload.get("Columns", []))
    records = list(payload.get("Records", []))
    if not records:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(records).select(columns)


def _as_pandas(frame: pd.DataFrame | pl.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pl.DataFrame):
        return pd.DataFrame(frame.to_dict(as_series=False))
    return frame.copy()


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _bids_label(value: str, fallback: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "", str(value))
    return label or fallback


def _column_label(value: str) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return label or "column"


def _column_mapping(columns) -> tuple[dict[str, str], dict[str, str]]:
    """Return original-to-BIDS and BIDS-to-original column mappings."""

    original_to_bids: dict[str, str] = {}
    bids_to_original: dict[str, str] = {}
    for original in columns:
        base = f"pyx_{_column_label(original)}"
        candidate = base
        index = 2
        while candidate in bids_to_original and bids_to_original[candidate] != original:
            candidate = f"{base}_{index}"
            index += 1
        original_to_bids[str(original)] = candidate
        bids_to_original[candidate] = str(original)
    return original_to_bids, bids_to_original


def _tabular_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.map(lambda value: value.isoformat() if pd.notna(value) else None)
    if series.dtype == object:
        return series.map(
            lambda value: (
                json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        )
    return series


def _write_headerless_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _unit_scale(unit: str | None) -> float:
    return {
        "s": 1.0,
        "ms": 1_000.0,
        "us": 1_000_000.0,
        "µs": 1_000_000.0,
    }.get(str(unit).lower(), 1_000.0)


def _infer_frequency(seconds: pd.Series) -> float:
    numeric = pd.to_numeric(seconds, errors="coerce").dropna().sort_values()
    differences = numeric.diff()
    differences = differences[differences > 0]
    if differences.empty:
        return 1.0
    return float(1.0 / differences.median())


def _package_version() -> str:
    try:
        return version("pyxations")
    except PackageNotFoundError:
        return "development"


def initialize_bids_derivative(
    raw_dataset: str | Path,
    derivative_dataset: str | Path,
) -> Path:
    """Create dataset-level metadata for a standalone BIDS Derivatives dataset."""

    raw_root = Path(raw_dataset)
    derivative_root = Path(derivative_dataset)
    derivative_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        derivative_root / "dataset_description.json",
        {
            "Name": f"{raw_root.name} Pyxations derivatives",
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "Pyxations",
                    "Version": _package_version(),
                    "Description": (
                        "Eye-movement detection, preprocessing, and trial "
                        "annotation of BIDS eye-tracking recordings."
                    ),
                }
            ],
            "SourceDatasets": [{"URL": f"../{raw_root.name}/"}],
        },
    )
    for filename in ("participants.tsv", "participants.json"):
        source = raw_root / filename
        if source.is_file():
            shutil.copy2(source, derivative_root / filename)
    (derivative_root / "README").write_text(
        "This dataset contains eye-tracking derivatives generated by "
        "Pyxations from the sibling raw BIDS dataset. Processed sample "
        "recordings use the physio suffix, and detected fixations, saccades, "
        "blinks, and retained messages use matching physioevents files. "
        "Human-readable plots may be generated under figures/; those report "
        "artifacts are not part of the standardized BIDS tables.\n",
        encoding="utf-8",
        newline="\n",
    )
    (derivative_root / ".bidsignore").write_text(
        "figures\nfigures/**\n",
        encoding="utf-8",
        newline="\n",
    )
    return derivative_root


class BIDSDerivativeExport:
    """Write and read canonical BIDS eye-tracking derivatives."""

    is_bids = True

    def extension(self):
        return ".tsv.gz"

    @staticmethod
    def _roots(session_path: Path) -> tuple[Path, Path]:
        derivative_root = session_path.parents[1]
        suffix = "_derivatives"
        raw_name = (
            derivative_root.name[: -len(suffix)]
            if derivative_root.name.endswith(suffix)
            else derivative_root.name
        )
        return derivative_root, derivative_root.with_name(raw_name)

    @staticmethod
    def _raw_sidecars(session_path: Path) -> list[tuple[Path, dict]]:
        _, raw_root = BIDSDerivativeExport._roots(session_path)
        folder = (
            raw_root
            / session_path.parent.name
            / session_path.name
            / "beh"
        )
        values = []
        for path in sorted(folder.glob("*_physio.json")):
            try:
                values.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        return values

    @staticmethod
    def _source_prefix(session_path: Path, sidecars) -> str:
        if sidecars:
            stem = sidecars[0][0].name.removesuffix("_physio.json")
            return re.sub(r"_recording-[A-Za-z0-9]+$", "", stem)
        return (
            f"{session_path.parent.name}_{session_path.name}_task-eyetracking"
        )

    @staticmethod
    def _sample_columns(frame: pd.DataFrame):
        if {"X", "Y"}.issubset(frame.columns):
            eye_values = (
                frame["eye"].dropna().astype(str).str.upper().unique().tolist()
                if "eye" in frame
                else []
            )
            recorded_eye = (
                "left"
                if eye_values == ["L"]
                else "right"
                if eye_values == ["R"]
                else "cyclopean"
            )
            return "X", "Y", "Pupil" if "Pupil" in frame else None, recorded_eye
        candidates = [
            (
                "Gaze2d_Left.x",
                "Gaze2d_Left.y",
                "PupilDiam_Left",
                "left",
            ),
            ("LX", "LY", "LPupil", "left"),
            (
                "Gaze2d_Right.x",
                "Gaze2d_Right.y",
                "PupilDiam_Right",
                "right",
            ),
            ("RX", "RY", "RPupil", "right"),
        ]
        for x, y, pupil, eye in candidates:
            if {x, y}.issubset(frame.columns):
                return x, y, pupil if pupil in frame else None, eye
        raise ValueError(
            "Processed samples do not contain a supported gaze-coordinate pair"
        )

    @staticmethod
    def _time_column(frame: pd.DataFrame) -> str:
        for column in ("t_acum", "tSample", "timestamp"):
            if column in frame:
                return column
        raise ValueError("Processed samples do not contain a timestamp column")

    @staticmethod
    def _source_metadata(sidecars, recorded_eye: str) -> dict:
        for _, metadata in sidecars:
            if metadata.get("RecordedEye") == recorded_eye:
                return metadata
        return sidecars[0][1] if sidecars else {}

    @staticmethod
    def _time_scale(frame: pd.DataFrame, time_column: str, metadata: dict) -> float:
        if time_column == "t_acum":
            return 1_000.0
        if {"TIMETICK", "BPOGX"}.intersection(frame.columns):
            return 1.0
        if {"Recording timestamp", "Gaze2d_Left.x"}.intersection(frame.columns):
            return 1_000_000.0
        timestamp_metadata = metadata.get("timestamp", {})
        return _unit_scale(timestamp_metadata.get("Units"))

    @staticmethod
    def _read_auxiliary_json(session_path: Path, filename: str):
        path = session_path / filename
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        finally:
            path.unlink(missing_ok=True)
        return value

    @staticmethod
    def _read_behavioral_events(session_path: Path):
        path = session_path / "events.tsv"
        if not path.is_file():
            return None
        try:
            return _frame_payload(pd.read_csv(path, sep="\t"))
        finally:
            path.unlink(missing_ok=True)

    def save_derivatives(
        self,
        *,
        session_path: Path,
        samples,
        fixations=None,
        saccades=None,
        blinks=None,
        messages=None,
        calibration=None,
        header=None,
        detection_algorithm: str,
    ) -> tuple[Path, Path | None]:
        """Write one processed physio file and its associated event annotations."""

        session_path = Path(session_path)
        sample_frame = _as_pandas(samples)
        sidecars = self._raw_sidecars(session_path)
        prefix = self._source_prefix(session_path, sidecars)
        desc = _bids_label(detection_algorithm.lower(), "pyxations")
        # ``desc`` is not currently permitted by the BIDS eye-tracking physio
        # schema. The required recording label therefore distinguishes both
        # the eye stream and the processing algorithm.
        base = f"{prefix}_recording-eye1{desc}"
        destination = session_path / "beh"

        x_column, y_column, pupil_column, recorded_eye = self._sample_columns(
            sample_frame
        )
        time_column = self._time_column(sample_frame)
        source_metadata = self._source_metadata(sidecars, recorded_eye)
        time_scale = self._time_scale(sample_frame, time_column, source_metadata)
        raw_time = pd.to_numeric(sample_frame[time_column], errors="coerce")
        finite_time = raw_time.dropna()
        if finite_time.empty:
            raise ValueError("Processed samples contain no valid timestamps")
        time_origin = float(finite_time.iloc[0])
        seconds = (raw_time - time_origin) / time_scale

        original_to_bids, bids_to_original = _column_mapping(sample_frame.columns)
        standardized = pd.DataFrame(
            {
                "timestamp": seconds,
                "x_coordinate": pd.to_numeric(
                    sample_frame[x_column], errors="coerce"
                ),
                "y_coordinate": pd.to_numeric(
                    sample_frame[y_column], errors="coerce"
                ),
            }
        )
        if pupil_column:
            standardized["pupil_size"] = pd.to_numeric(
                sample_frame[pupil_column], errors="coerce"
            )
        for original, bids_column in original_to_bids.items():
            standardized[bids_column] = _tabular_series(sample_frame[original])
        standardized.sort_values("timestamp", inplace=True, kind="stable")
        standardized.reset_index(drop=True, inplace=True)

        sampling_frequency = source_metadata.get("SamplingFrequency")
        if sampling_frequency is None and "Rate_recorded" in sample_frame:
            rates = pd.to_numeric(
                sample_frame["Rate_recorded"], errors="coerce"
            ).dropna()
            sampling_frequency = float(rates.median()) if not rates.empty else None
        sampling_frequency = float(
            sampling_frequency or _infer_frequency(standardized["timestamp"])
        )

        coordinate = source_metadata.get("x_coordinate", {})
        coordinate_unit = coordinate.get("Units", "arbitrary")
        coordinate_system = source_metadata.get(
            "SampleCoordinateSystem", "custom"
        )
        coordinate_description = source_metadata.get(
            "SampleCoordinateSystemDescription",
            "Coordinate system retained from the processed source recording.",
        )
        columns = list(standardized.columns)
        metadata = {
            "SamplingFrequency": sampling_frequency,
            "StartTime": 0.0,
            "Columns": columns,
            "PhysioType": "eyetrack",
            "RecordedEye": recorded_eye,
            "SampleCoordinateSystem": coordinate_system,
            "SampleCoordinateSystemDescription": coordinate_description,
            "Description": (
                f"Eye-tracking samples processed by Pyxations using "
                f"{detection_algorithm}."
            ),
            "timestamp": {
                "Description": "Time elapsed since the first processed sample.",
                "Units": "s",
                "Origin": "First sample in the source recording",
            },
            "x_coordinate": {
                "Description": "Processed horizontal gaze coordinate.",
                "Units": coordinate_unit,
            },
            "y_coordinate": {
                "Description": "Processed vertical gaze coordinate.",
                "Units": coordinate_unit,
            },
            "PyxationsColumnMap": bids_to_original,
            "PyxationsTimeOrigin": time_origin,
            "PyxationsTimeScale": time_scale,
            "PyxationsDetectionAlgorithm": detection_algorithm,
            "PyxationsCalibration": _frame_payload(_as_pandas(calibration)),
            "PyxationsHeader": _frame_payload(_as_pandas(header)),
            "PyxationsPreprocessingRecipe": self._read_auxiliary_json(
                session_path, "preprocessing_recipe.json"
            ),
            "PyxationsPreprocessingProvenance": self._read_auxiliary_json(
                session_path, "preprocessing_provenance.json"
            ),
            "PyxationsBehavioralEvents": self._read_behavioral_events(
                session_path
            ),
        }
        if "pupil_size" in standardized:
            pupil_metadata = source_metadata.get("pupil_size", {})
            metadata["pupil_size"] = {
                "Description": pupil_metadata.get(
                    "Description",
                    "Processed pupil diameter or area as reported by the "
                    "source tracker; consult the source metadata for type.",
                ),
                "Units": pupil_metadata.get("Units", "arbitrary"),
            }
        for bids_column, original in bids_to_original.items():
            metadata[bids_column] = {
                "Description": (
                    f"Pyxations analysis column; original name: {original}."
                )
            }

        physio_path = destination / f"{base}_physio.tsv.gz"
        physio_json = destination / f"{base}_physio.json"
        _write_headerless_tsv(physio_path, standardized)
        _write_json(physio_json, metadata)

        event_tables = {
            "fix": _as_pandas(fixations),
            "sacc": _as_pandas(saccades),
            "blink": _as_pandas(blinks),
            "msg": _as_pandas(messages),
        }
        event_path, _ = self._write_events(
            destination=destination,
            base=base,
            tables=event_tables,
            sample_time_origin=time_origin,
            sample_time_scale=time_scale,
            sample_duration=float(standardized["timestamp"].max()),
        )
        return physio_path, event_path

    @staticmethod
    def _event_scale(
        onset: pd.Series,
        *,
        sample_time_origin: float,
        sample_time_scale: float,
        sample_duration: float,
        table_name: str,
    ) -> float:
        if table_name == "msg":
            return sample_time_scale
        candidates = []
        for scale in dict.fromkeys((sample_time_scale, 1_000.0, 1.0)):
            seconds = (
                pd.to_numeric(onset, errors="coerce") - sample_time_origin
            ) / scale
            valid = seconds.dropna()
            score = (
                float(
                    (
                        (valid >= -1.0)
                        & (valid <= max(sample_duration + 1.0, 1.0))
                    ).mean()
                )
                if not valid.empty
                else 0.0
            )
            candidates.append((score, scale))
        return max(candidates, key=lambda item: item[0])[1]

    def _write_events(
        self,
        *,
        destination: Path,
        base: str,
        tables: Mapping[str, pd.DataFrame],
        sample_time_origin: float,
        sample_time_scale: float,
        sample_duration: float,
    ) -> tuple[Path | None, Path | None]:
        all_columns = []
        for frame in tables.values():
            all_columns.extend(str(column) for column in frame.columns)
        original_to_bids, bids_to_original = _column_mapping(
            dict.fromkeys(all_columns)
        )

        prepared = []
        table_columns = {
            name: list(frame.columns) for name, frame in tables.items()
        }
        event_names = {
            "fix": "fixation",
            "sacc": "saccade",
            "blink": "blink",
            "msg": "message",
        }
        for table_name, frame in tables.items():
            if frame.empty:
                continue
            onset_column = next(
                (
                    column
                    for column in ("tStart", "timestamp", "tSample", "tEnd")
                    if column in frame
                ),
                None,
            )
            if onset_column is None:
                continue
            scale = self._event_scale(
                frame[onset_column],
                sample_time_origin=sample_time_origin,
                sample_time_scale=sample_time_scale,
                sample_duration=sample_duration,
                table_name=table_name,
            )
            onset = (
                pd.to_numeric(frame[onset_column], errors="coerce")
                - sample_time_origin
            ) / scale
            if "duration" in frame:
                duration = (
                    pd.to_numeric(frame["duration"], errors="coerce") / scale
                )
            elif "tEnd" in frame and onset_column != "tEnd":
                duration = (
                    pd.to_numeric(frame["tEnd"], errors="coerce")
                    - pd.to_numeric(frame[onset_column], errors="coerce")
                ) / scale
            else:
                duration = pd.Series(0.0, index=frame.index)

            result = pd.DataFrame(
                {
                    "onset": onset,
                    "duration": duration.clip(lower=0),
                    "trial_type": event_names[table_name],
                    "pyxations_table": table_name,
                }
            )
            for original in frame.columns:
                result[original_to_bids[str(original)]] = _tabular_series(
                    frame[original]
                )
            prepared.append(result)

        if not prepared:
            return None, None
        events = pd.concat(prepared, ignore_index=True, sort=False)
        events.dropna(subset=["onset"], inplace=True)
        if events.empty:
            return None, None
        events.sort_values("onset", inplace=True, kind="stable")
        ordered = ["onset", "duration", "trial_type", "pyxations_table"]
        ordered.extend(column for column in events if column not in ordered)
        events = events.loc[:, ordered]

        metadata = {
            "Columns": list(events.columns),
            "Description": (
                "Fixations, saccades, blinks, and messages identified or "
                "retained by the Pyxations processing pipeline."
            ),
            "OnsetSource": "timestamp",
            "onset": {
                "Description": (
                    "Onset in seconds on the timeline of the associated "
                    "processed eye-tracking recording."
                ),
                "Units": "s",
            },
            "duration": {
                "Description": "Event duration.",
                "Units": "s",
            },
            "trial_type": {
                "Description": "Type of eye-movement or message event.",
                "Levels": {
                    "fixation": "Fixation event.",
                    "saccade": "Saccade event.",
                    "blink": "Blink event.",
                    "message": "Message retained from the source recording.",
                },
            },
            "pyxations_table": {
                "Description": (
                    "Original Pyxations table used to reconstruct the analysis "
                    "DataFrame."
                )
            },
            "PyxationsColumnMap": bids_to_original,
            "PyxationsTableColumns": table_columns,
        }
        for bids_column, original in bids_to_original.items():
            metadata[bids_column] = {
                "Description": (
                    f"Pyxations event column; original name: {original}."
                )
            }

        event_path = destination / f"{base}_physioevents.tsv.gz"
        event_json = destination / f"{base}_physioevents.json"
        _write_headerless_tsv(event_path, events)
        _write_json(event_json, metadata)
        return event_path, event_json

    @staticmethod
    def _read_table(path: Path, metadata: Mapping) -> pl.DataFrame:
        columns = list(metadata["Columns"])
        return pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=columns,
            null_values=["n/a"],
        )

    def read_derivatives(
        self, session_path: str | Path, detection_algorithm: str
    ) -> dict[str, pl.DataFrame | None]:
        """Load BIDS derivatives and reconstruct Pyxations analysis tables."""

        session_path = Path(session_path)
        desc = _bids_label(detection_algorithm.lower(), "pyxations")
        physio_files = sorted(
            (session_path / "beh").glob(
                f"*_recording-eye1{desc}_physio.tsv.gz"
            )
        )
        if not physio_files:
            raise FileNotFoundError(
                f"No BIDS derivatives for {detection_algorithm} in {session_path}"
            )
        physio_path = physio_files[0]
        physio_metadata = json.loads(
            physio_path.with_suffix("").with_suffix(".json").read_text(
                encoding="utf-8"
            )
        )
        samples_bids = self._read_table(physio_path, physio_metadata)
        sample_mapping = physio_metadata.get("PyxationsColumnMap", {})
        sample_columns = [
            column
            for column in sample_mapping
            if column in samples_bids.columns
        ]
        samples = samples_bids.select(sample_columns).rename(
            {column: sample_mapping[column] for column in sample_columns}
        )

        output: dict[str, pl.DataFrame | None] = {
            "samples": samples,
            "fix": pl.DataFrame(),
            "sacc": pl.DataFrame(),
            "blink": pl.DataFrame(),
            "msg": pl.DataFrame(),
            "calib": _payload_frame(
                physio_metadata.get("PyxationsCalibration")
            ),
            "header": _payload_frame(physio_metadata.get("PyxationsHeader")),
        }
        event_path = physio_path.with_name(
            physio_path.name.replace("_physio.tsv.gz", "_physioevents.tsv.gz")
        )
        if not event_path.is_file():
            return output
        event_metadata = json.loads(
            event_path.with_suffix("").with_suffix(".json").read_text(
                encoding="utf-8"
            )
        )
        events = self._read_table(event_path, event_metadata)
        event_mapping = event_metadata.get("PyxationsColumnMap", {})
        table_columns = event_metadata.get("PyxationsTableColumns", {})
        for table_name in ("fix", "sacc", "blink", "msg"):
            original_columns = list(table_columns.get(table_name, []))
            rows = events.filter(pl.col("pyxations_table") == table_name)
            bids_columns = [
                bids_column
                for bids_column, original in event_mapping.items()
                if original in original_columns
                and bids_column in rows.columns
            ]
            if rows.is_empty():
                output[table_name] = pl.DataFrame(
                    {column: [] for column in original_columns}
                )
                continue
            reconstructed = rows.select(bids_columns).rename(
                {
                    column: event_mapping[column] for column in bids_columns
                }
            )
            missing_columns = [
                column
                for column in original_columns
                if column not in reconstructed.columns
            ]
            if missing_columns:
                reconstructed = reconstructed.with_columns(
                    [pl.lit(None).alias(column) for column in missing_columns]
                )
            output[table_name] = reconstructed.select(
                original_columns
            )
        return output
