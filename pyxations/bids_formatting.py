"""Raw-to-BIDS conversion and Polars-native derivative orchestration."""

from __future__ import annotations

import inspect
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from pyxations.bids import bids_label, read_raw_bids_session, write_bids_dataset
from pyxations.export.bids import BIDSDerivativeExport, initialize_bids_derivative
from pyxations.methods.eyemovement.engbert import EngbertDetection
from pyxations.pre_processing import PreProcessing
from pyxations.tables import SessionTables, read_tsv

EYE_MOVEMENT_DETECTION_DICT = {"engbert": EngbertDetection}


def _detector_type(name):
    """Load optional detectors only when the corresponding feature is used."""

    if name == "remodnav":
        try:
            from pyxations.methods.eyemovement.REMoDNaV import RemodnavDetection
        except ImportError as exc:
            if exc.name and exc.name.startswith("remodnav"):
                raise ImportError(
                    "REMoDNaV support is optional. Install it with "
                    "`pip install 'pyxations[remodnav]'`."
                ) from exc
            raise
        return RemodnavDetection
    return EYE_MOVEMENT_DETECTION_DICT.get(name)


def _segmentation_recipe(pre_processing, kwargs):
    prefer_durations = kwargs.get("prefer_durations", False)
    have_explicit_times = "start_times" in kwargs and "end_times" in kwargs
    have_durations = "start_msgs" in kwargs and "durations" in kwargs
    have_message_times = "start_msgs" in kwargs and "end_msgs" in kwargs
    if not (have_explicit_times or have_durations or have_message_times):
        return None
    if have_explicit_times:
        function_name = "split_all_into_trials"
    elif have_durations and (prefer_durations or not have_message_times):
        function_name = "split_all_into_trials_by_durations"
    else:
        function_name = "split_all_into_trials_by_msgs"
    allowed = set(inspect.signature(getattr(pre_processing, function_name)).parameters)
    candidates = {
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
    return function_name, {
        key: value
        for key, value in kwargs.items()
        if key in candidates and key in allowed
    }


def _assign_default_trials(pre_processing):
    samples = pre_processing.samples.with_columns(
        (
            pl.col("trial_number").cast(pl.Int64, strict=False).fill_null(0)
            if "trial_number" in pre_processing.samples.columns
            else pl.lit(0, dtype=pl.Int64)
        ).alias("trial_number"),
        (
            pl.col("phase") if "phase" in pre_processing.samples.columns else pl.lit("")
        ).alias("phase"),
    )
    pre_processing.samples = samples

    for table_name in ("fixations", "saccades", "blinks"):
        table = getattr(pre_processing, table_name).with_columns(
            pl.lit(0, dtype=pl.Int64).alias("trial_number"),
            pl.lit("").alias("phase"),
        )
        if not table.is_empty():
            for trial_number, group in samples.partition_by(
                "trial_number", as_dict=True
            ).items():
                trial_value = (
                    trial_number[0] if isinstance(trial_number, tuple) else trial_number
                )
                start = group.get_column("tSample").min()
                end = group.get_column("tSample").max()
                table = table.with_columns(
                    pl.when((pl.col("tStart") >= start) & (pl.col("tEnd") <= end))
                    .then(pl.lit(int(trial_value)))
                    .otherwise(pl.col("trial_number"))
                    .alias("trial_number")
                )
        setattr(pre_processing, table_name, table)


def _detect_from_bids(
    raw,
    *,
    dataset_format,
    detection_algorithm,
    session_folder_path,
    kwargs,
):
    samples = raw.samples.clone()
    blinks = raw.blinks.clone()
    if detection_algorithm == "eyelink":
        if dataset_format != "eyelink":
            raise ValueError(
                "The eyelink detector requires tracker-reported EyeLink events"
            )
        return (
            samples,
            raw.fixations.clone(),
            raw.saccades.clone(),
            blinks,
        )

    detector_type = _detector_type(detection_algorithm)
    if detector_type is None:
        raise ValueError(f"Unknown eye-movement detector: {detection_algorithm}")
    if dataset_format != "eyelink" and not {"X", "Y"}.issubset(samples.columns):
        eye_prefix = "L" if {"LX", "LY"}.issubset(samples.columns) else "R"
        expressions = [
            pl.col(f"{eye_prefix}X").alias("X"),
            pl.col(f"{eye_prefix}Y").alias("Y"),
            pl.lit(eye_prefix).alias("eye"),
        ]
        pupil_column = f"{eye_prefix}Pupil"
        if pupil_column in samples:
            expressions.append(pl.col(pupil_column).alias("Pupil"))
        samples = samples.with_columns(expressions)
    detector = detector_type(
        session_folder_path=session_folder_path,
        samples=samples,
    )
    if detection_algorithm == "remodnav" and {"X", "Y"}.issubset(samples.columns):
        config = {
            "webgazer": {"savgol_length": 0.195, "max_pso_dur": 0.1},
            "gaze": {"savgol_length": 0.19, "max_pso_dur": 0.4},
            "tobii": {"savgol_length": 0.195, "max_pso_dur": 0.3},
        }.get(dataset_format, {})
        accepted_config = set(
            inspect.signature(detector.run_eye_movement).parameters
        ) - {"self", "gazex_data", "gazey_data", "sample_rate"}
        config.update(
            {key: value for key, value in kwargs.items() if key in accepted_config}
        )
        format_rate = {
            "webgazer": 30.0,
            "gaze": 60.0,
            "tobii": 60.0,
        }.get(dataset_format)
        detector_rate = (
            kwargs.get("sample_rate")
            or kwargs.get("sampling_frequency")
            or format_rate
            or raw.sampling_frequency
        )
        if detector_rate is None:
            raise ValueError(
                "Sampling frequency is required for sample-based event detection"
            )
        fixations, saccades = detector.run_eye_movement_from_samples(
            detector_rate,
            config=config,
        )
    else:
        detection_method = detector.detect_eye_movements
        accepted = set(inspect.signature(detection_method).parameters)
        detection_kwargs = {
            key: value for key, value in kwargs.items() if key in accepted
        }
        fixations, saccades = detection_method(**detection_kwargs)
    return samples, fixations, saccades, blinks


def _find_best_eye(calibration: pl.DataFrame) -> str:
    """Return the eye with the best final EyeLink validation."""

    if "line" not in calibration:
        return "M"
    lines = [
        str(value)
        for value in calibration.get_column("line").drop_nulls().to_list()
        if "CAL VALIDATION" in str(value)
    ]
    if not lines:
        return "M"
    candidates: dict[str, float] = {}
    aborted: set[str] = set()
    for line in lines:
        upper = line.upper()
        eye = "L" if "LEFT" in upper or "L ABORTED" in upper else "R"
        if "ABORTED" in upper:
            aborted.add(eye)
            continue
        match = re.search(r"\bERROR\s+([-+]?[0-9]*\.?[0-9]+)", upper)
        candidates[eye] = float(match.group(1)) if match else float("inf")
    valid = {eye: error for eye, error in candidates.items() if eye not in aborted}
    if valid:
        return min(valid, key=valid.get)
    if len(aborted) == 1:
        return "R" if "L" in aborted else "L"
    return "M"


def _keep_eye(
    eye: str,
    samples: pl.DataFrame,
    fixations: pl.DataFrame,
    blinks: pl.DataFrame,
    saccades: pl.DataFrame,
):
    prefix = "R" if eye == "R" else "L"
    source_columns = [f"{prefix}X", f"{prefix}Y", f"{prefix}Pupil"]
    target_columns = ["X", "Y", "Pupil"]
    retained = [
        column
        for column in (
            "tSample",
            *source_columns,
            "Line_number",
            "Eyes_recorded",
            "Rate_recorded",
            "Calib_index",
        )
        if column in samples
    ]
    rename = {
        source: target
        for source, target in zip(source_columns, target_columns)
        if source in retained
    }
    samples = (
        samples.select(retained).rename(rename).with_columns(pl.lit(eye).alias("eye"))
    )

    def selected(table):
        if "eye" in table:
            table = table.filter(
                pl.col("eye").cast(pl.String).str.to_uppercase() == eye
            )
        required = [column for column in ("tStart", "tEnd") if column in table]
        return table.drop_nulls(subset=required) if required else table

    return samples, selected(fixations), selected(blinks), selected(saccades)


def _choose_best_eye(raw, samples, fixations, blinks, saccades):
    required = {"Calib_index", "line"}
    if raw.calibration.is_empty() or not required.issubset(raw.calibration.columns):
        return samples, fixations, blinks, saccades
    if (
        not raw.calibration.get_column("line")
        .cast(pl.String)
        .str.contains("CAL VALIDATION")
        .any()
    ):
        return samples, fixations, blinks, saccades

    selected = []
    for calibration_index, group in raw.calibration.partition_by(
        "Calib_index", as_dict=True, maintain_order=True
    ).items():
        index = (
            calibration_index[0]
            if isinstance(calibration_index, tuple)
            else calibration_index
        )
        eye = _find_best_eye(group)
        if eye not in {"L", "R"}:
            continue

        def calibration_rows(table, calibration_value=index):
            return (
                table.filter(pl.col("Calib_index") == calibration_value)
                if "Calib_index" in table
                else table
            )

        selected.append(
            _keep_eye(
                eye,
                calibration_rows(samples),
                calibration_rows(fixations),
                calibration_rows(blinks),
                calibration_rows(saccades),
            )
        )
    if not selected:
        return samples, fixations, blinks, saccades
    return tuple(
        pl.concat([frames[position] for frames in selected], how="diagonal_relaxed")
        for position in range(4)
    )


def process_bids_session(
    raw_session_path,
    dataset_format,
    detection_algorithm,
    session_folder_path,
    force_best_eye,
    **kwargs,
):
    """Compute and write one derivative session from normalized raw BIDS."""

    raw = read_raw_bids_session(raw_session_path)
    session_folder_path = Path(session_folder_path)
    session_folder_path.mkdir(parents=True, exist_ok=True)

    messages = raw.messages.clone()
    message_keywords = kwargs.pop("msg_keywords", None)
    if message_keywords and not messages.is_empty() and "message" in messages:
        pattern = "(?i)" + "|".join(re.escape(keyword) for keyword in message_keywords)
        messages = messages.filter(
            pl.col("message").cast(pl.String).str.contains(pattern).fill_null(False)
        )

    samples, fixations, saccades, blinks = _detect_from_bids(
        raw,
        dataset_format=dataset_format,
        detection_algorithm=detection_algorithm,
        session_folder_path=session_folder_path,
        kwargs=kwargs,
    )
    if force_best_eye:
        samples, fixations, blinks, saccades = _choose_best_eye(
            raw, samples, fixations, blinks, saccades
        )

    pre_processing = PreProcessing(
        samples,
        fixations,
        saccades,
        blinks,
        messages,
        session_folder_path,
    )
    pre_processing.set_metadata(
        screen_width=raw.screen_width or kwargs.get("screen_width"),
        screen_height=raw.screen_height or kwargs.get("screen_height"),
    )
    segmentation = _segmentation_recipe(pre_processing, kwargs)
    if segmentation:
        name, parameters = segmentation
        recipe = {name: parameters}
        if dataset_format == "eyelink":
            bad_parameters = {
                key: kwargs[key]
                for key in (
                    "screen_height",
                    "screen_width",
                    "mark_nan_as_bad",
                    "inclusive_bounds",
                )
                if key in kwargs
            }
            recipe = {
                "bad_samples": bad_parameters,
                name: parameters,
                "saccades_direction": (
                    {"tol_deg": kwargs["tol_deg"]} if "tol_deg" in kwargs else {}
                ),
            }
        pre_processing.process(recipe)
    else:
        _assign_default_trials(pre_processing)

    behavioral_columns = kwargs.get("behavioral_columns")
    if behavioral_columns and not raw.behavioral_events.is_empty():
        if "trial_number" in raw.behavioral_events:
            metadata = raw.behavioral_events.rename({"trial_number": "trial_index"})
        else:
            metadata = raw.behavioral_events
        if "trial_index" in metadata:
            pre_processing.add_trial_metadata(metadata, behavioral_columns)

    processed = SessionTables(
        samples=pre_processing.samples,
        fixations=pre_processing.fixations,
        saccades=pre_processing.saccades,
        blinks=pre_processing.blinks,
        messages=pre_processing.user_messages,
        calibration=raw.calibration,
        header=raw.header,
        behavioral_events=raw.behavioral_events,
        sampling_frequency=raw.sampling_frequency,
        screen_width=pre_processing.metadata.screen_width,
        screen_height=pre_processing.metadata.screen_height,
    )
    BIDSDerivativeExport().write_session(
        session_folder_path,
        processed,
        detection_algorithm=detection_algorithm,
    )


def dataset_to_bids(
    target_folder_path,
    files_folder_path,
    dataset_name,
    session_substrings=1,
    format_name="eyelink",
    *,
    task_name="eyetracking",
    authors=None,
    overwrite=False,
):
    """Convert vendor recordings to raw BIDS and retain originals in sourcedata."""

    return write_bids_dataset(
        target_folder_path,
        files_folder_path,
        dataset_name,
        session_substrings=session_substrings,
        format_name=format_name,
        task_name=task_name,
        authors=authors,
        overwrite=overwrite,
    )


def process_session(
    raw_session_path,
    dataset_format,
    detection_algorithm,
    session_folder_path,
    force_best_eye,
    overwrite,
    **kwargs,
):
    """Process one raw BIDS session unless its requested output already exists."""

    session_folder_path = Path(session_folder_path)
    label = bids_label(detection_algorithm.lower(), fallback="pyxations")
    if not overwrite and session_folder_path.exists():
        existing = (session_folder_path / "beh").glob(
            f"*_recording-eye1{label}_physio.tsv.gz"
        )
        if next(existing, None) is not None:
            return
    if dataset_format not in {"eyelink", "webgazer", "tobii", "gaze"}:
        raise ValueError(f"Dataset format {dataset_format} not found.")
    process_bids_session(
        raw_session_path,
        dataset_format,
        detection_algorithm,
        session_folder_path,
        force_best_eye,
        **kwargs,
    )


def compute_derivatives_for_dataset(
    bids_dataset_folder,
    dataset_format,
    detection_algorithm="remodnav",
    num_processes: int = 1,
    force_best_eye=True,
    overwrite=False,
    behavioral_columns=None,
    **kwargs,
):
    """Compute canonical BIDS derivatives for every raw BIDS session.

    Processing is serial by default to avoid process-startup and serialization
    costs for small datasets. Set ``num_processes`` above one for large
    collections of independent sessions.
    """

    bids_dataset_folder = Path(bids_dataset_folder)
    if isinstance(num_processes, bool) or not isinstance(num_processes, int):
        raise TypeError("num_processes must be an integer")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    derivatives_folder = Path(f"{bids_dataset_folder}_derivatives")
    initialize_bids_derivative(bids_dataset_folder, derivatives_folder)

    start_times = kwargs.pop("start_times", None)
    end_times = kwargs.pop("end_times", None)
    if behavioral_columns is not None:
        kwargs["behavioral_columns"] = behavioral_columns

    participants = read_tsv(
        bids_dataset_folder / "participants.tsv", has_header=True
    ).with_columns(
        pl.col("subject_id").cast(pl.String).str.pad_start(4, "0"),
        pl.col("old_subject_id").cast(pl.String),
    )
    subject_lookup = dict(
        participants.select("subject_id", "old_subject_id").iter_rows()
    )

    jobs = []
    for subject in sorted(bids_dataset_folder.glob("sub-*")):
        if not subject.is_dir():
            continue
        subject_name = subject_lookup[subject.name[4:]]
        for session in sorted(subject.glob("ses-*")):
            if not session.is_dir():
                continue
            session_name = session.name[4:]
            session_kwargs = dict(kwargs)
            if (
                start_times
                and subject_name in start_times
                and session_name in start_times[subject_name]
            ):
                session_kwargs["start_times"] = start_times[subject_name][session_name]
            if (
                end_times
                and subject_name in end_times
                and session_name in end_times[subject_name]
            ):
                session_kwargs["end_times"] = end_times[subject_name][session_name]
            jobs.append(
                (
                    session,
                    dataset_format,
                    detection_algorithm,
                    derivatives_folder / subject.name / session.name,
                    force_best_eye,
                    overwrite,
                    session_kwargs,
                )
            )

    if num_processes == 1:
        for (
            source,
            format_name,
            algorithm,
            destination,
            choose_eye,
            replace,
            options,
        ) in jobs:
            process_session(
                source,
                format_name,
                algorithm,
                destination,
                choose_eye,
                replace,
                **options,
            )
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    process_session,
                    source,
                    format_name,
                    algorithm,
                    destination,
                    choose_eye,
                    replace,
                    **options,
                )
                for source, format_name, algorithm, destination, choose_eye, replace, options in jobs
            ]
            for future in futures:
                future.result()
    return derivatives_folder
