from pathlib import Path
import inspect
import re
import shutil
import pandas as pd
import polars as pl
from concurrent.futures import ProcessPoolExecutor
from pyxations.methods.eyemovement.engbert import EngbertDetection

from pyxations.export import BIDS_EXPORT
from pyxations.bids import read_raw_bids_session, write_bids_dataset
from pyxations.export.bids import initialize_bids_derivative
from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing

EYE_MOVEMENT_DETECTION_DICT = {"engbert": EngbertDetection}


def _detector_type(name):
    """Load optional detectors only when the corresponding feature is used."""

    if name == "remodnav":
        try:
            from pyxations.methods.eyemovement.REMoDNaV import (
                RemodnavDetection,
            )
        except ImportError as exc:
            if exc.name and exc.name.startswith("remodnav"):
                raise ImportError(
                    "REMoDNaV support is optional. Install it with "
                    "`pip install 'pyxations[remodnav]'`."
                ) from exc
            raise
        return RemodnavDetection
    return EYE_MOVEMENT_DETECTION_DICT.get(name)


def _clean_bids_session_workfiles(session_folder_path, detection_algorithm):
    """Keep only canonical BIDS data after a parser finishes its export."""

    session_folder_path = Path(session_folder_path)
    if not session_folder_path.is_dir():
        return
    work_directories = {
        "events",
        "eyelink_events",
        f"{detection_algorithm}_events",
    }
    for directory_name in work_directories:
        directory = session_folder_path / directory_name
        if directory.is_dir():
            shutil.rmtree(directory)
    for ascii_file in session_folder_path.glob("*.asc"):
        ascii_file.unlink()


def _segmentation_recipe(pre_processing, kwargs):
    prefer_durations = kwargs.get("prefer_durations", False)
    have_explicit_times = (
        "start_times" in kwargs and "end_times" in kwargs
    )
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
    allowed = set(
        inspect.signature(
            getattr(pre_processing, function_name)
        ).parameters
    )
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
    parameters = {
        key: value
        for key, value in kwargs.items()
        if key in candidates and key in allowed
    }
    return function_name, parameters


def _assign_default_trials(pre_processing):
    samples = pre_processing.samples.with_columns(
        (
            pl.col("trial_number").cast(pl.Int64, strict=False).fill_null(0)
            if "trial_number" in pre_processing.samples.columns
            else pl.lit(0, dtype=pl.Int64)
        ).alias("trial_number"),
        (
            pl.col("phase")
            if "phase" in pre_processing.samples.columns
            else pl.lit("")
        ).alias("phase"),
    )
    pre_processing.samples = samples

    for table_name in ("fixations", "saccades", "blinks"):
        table = getattr(pre_processing, table_name)
        table = table.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("trial_number"),
            pl.lit("").alias("phase"),
        )
        if not table.is_empty():
            for trial_number, group in samples.partition_by(
                "trial_number", as_dict=True
            ).items():
                trial_value = trial_number[0] if isinstance(
                    trial_number, tuple
                ) else trial_number
                start = group.get_column("tSample").min()
                end = group.get_column("tSample").max()
                table = table.with_columns(
                    pl.when(
                        (pl.col("tStart") >= start)
                        & (pl.col("tEnd") <= end)
                    )
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
    samples = raw.samples.copy()
    blinks = raw.blinks.copy()
    if detection_algorithm == "eyelink":
        if dataset_format != "eyelink":
            raise ValueError(
                "The eyelink detector requires tracker-reported EyeLink events"
            )
        return (
            samples,
            raw.fixations.copy(),
            raw.saccades.copy(),
            blinks,
        )

    detector_type = _detector_type(detection_algorithm)
    if detector_type is None:
        raise ValueError(
            f"Unknown eye-movement detector: {detection_algorithm}"
        )
    if dataset_format != "eyelink" and not {"X", "Y"}.issubset(samples):
        eye_prefix = "L" if {"LX", "LY"}.issubset(samples) else "R"
        samples["X"] = samples[f"{eye_prefix}X"]
        samples["Y"] = samples[f"{eye_prefix}Y"]
        pupil_column = f"{eye_prefix}Pupil"
        if pupil_column in samples:
            samples["Pupil"] = samples[pupil_column]
        samples["eye"] = eye_prefix
    detector = detector_type(
        session_folder_path=session_folder_path,
        samples=samples,
    )
    if detection_algorithm == "remodnav" and {"X", "Y"}.issubset(samples):
        config = {
            "webgazer": {"savgol_length": 0.195, "max_pso_dur": 0.1},
            "gaze": {"savgol_length": 0.19, "max_pso_dur": 0.4},
            "tobii": {"savgol_length": 0.195, "max_pso_dur": 0.3},
        }.get(dataset_format, {})
        detector_rate = {
            "webgazer": 30.0,
            "gaze": 60.0,
            "tobii": 60.0,
        }.get(dataset_format, raw.sampling_frequency)
        fixations, saccades = detector.run_eye_movement_from_samples(
            detector_rate,
            config=config,
        )
    else:
        detection_method = detector.detect_eye_movements
        accepted = set(inspect.signature(detection_method).parameters)
        detection_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in accepted
        }
        fixations, saccades = detection_method(**detection_kwargs)
    return samples, fixations, saccades, blinks


def process_bids_session(
    raw_session_path,
    dataset_format,
    detection_algorithm,
    session_folder_path,
    force_best_eye,
    keep_ascii,
    overwrite,
    exp_format,
    **kwargs,
):
    """Compute derivatives from normalized raw BIDS tables only."""

    raw = read_raw_bids_session(raw_session_path)
    session_folder_path = Path(session_folder_path)
    session_folder_path.mkdir(parents=True, exist_ok=True)

    messages = raw.messages.copy()
    message_keywords = kwargs.pop("msg_keywords", None)
    if message_keywords and not messages.empty:
        pattern = "|".join(
            re.escape(keyword) for keyword in message_keywords
        )
        messages = messages.loc[
            messages["message"].astype(str).str.contains(
                pattern, case=False, regex=True, na=False
            )
        ].reset_index(drop=True)

    samples, fixations, saccades, blinks = _detect_from_bids(
        raw,
        dataset_format=dataset_format,
        detection_algorithm=detection_algorithm,
        session_folder_path=session_folder_path,
        kwargs=kwargs,
    )

    if (
        force_best_eye
        and not raw.calibration.empty
        and "Calib_index" in raw.calibration
        and "line" in raw.calibration
        and raw.calibration["line"].astype(str).str.contains(
            "CAL VALIDATION"
        ).any()
    ):
        calibration_indexes = raw.calibration["Calib_index"].unique()
        best_eyes = [
            find_besteye(group)
            for _, group in raw.calibration.groupby("Calib_index")
        ]
        selected = [
            keep_eye(
                best_eyes[index],
                samples.loc[samples["Calib_index"] == calibration_index],
                fixations.loc[
                    fixations["Calib_index"] == calibration_index
                ],
                blinks.loc[blinks["Calib_index"] == calibration_index],
                saccades.loc[
                    saccades["Calib_index"] == calibration_index
                ],
            )
            for index, calibration_index in enumerate(calibration_indexes)
            if best_eyes[index] in {"L", "R"}
        ]
        if selected:
            samples, fixations, blinks, saccades = [
                pd.concat([frames[position] for frames in selected])
                for position in range(4)
            ]

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
                    {"tol_deg": kwargs["tol_deg"]}
                    if "tol_deg" in kwargs
                    else {}
                ),
            }
        pre_processing.process(recipe)
    else:
        _assign_default_trials(pre_processing)

    behavioral_columns = kwargs.get("behavioral_columns")
    if behavioral_columns and not raw.behavioral_events.empty:
        metadata = raw.behavioral_events.copy()
        if "trial_number" in metadata:
            pre_processing.add_trial_metadata(
                metadata, behavioral_columns
            )

    parser = BidsParse(session_folder_path, exp_format)
    parser.detection_algorithm = detection_algorithm
    parser.store_dataframes(
        pre_processing.samples,
        dfCalib=raw.calibration,
        dfFix=pre_processing.fixations,
        dfSacc=pre_processing.saccades,
        dfHeader=raw.header,
        dfBlink=pre_processing.blinks,
        dfMsg=pre_processing.user_messages,
    )
    if exp_format == BIDS_EXPORT:
        _clean_bids_session_workfiles(
            session_folder_path, detection_algorithm
        )


def find_besteye(df_cal):
    if df_cal[df_cal['line'].str.contains('CAL VALIDATION')].index.empty:
        return 'M'
    last_index = df_cal[df_cal['line'].str.contains('CAL VALIDATION')].index[-1]
    last_val_msg = df_cal.loc[last_index].values[0]
    second_to_last_index = last_index - 1
    if 'ABORTED' in last_val_msg:
        if not second_to_last_index in df_cal.index or 'CAL VALIDATION' not in df_cal.loc[second_to_last_index].values[0] or 'ABORTED' in df_cal.loc[second_to_last_index].values[0]:
            return 'L' if 'L ABORTED' in last_val_msg else 'R'
        last_val_msg = df_cal.loc[second_to_last_index].values[0]
        return 'L' if ('LEFT' in last_val_msg or 'L ABORTED' in last_val_msg) else 'R'
    
    if not second_to_last_index in df_cal.index or 'CAL VALIDATION' not in df_cal.loc[second_to_last_index].values[0] or 'ABORTED' in df_cal.loc[second_to_last_index].values[0]:
        return 'L' if 'LEFT' in last_val_msg else 'R'    
    left_index = last_index if 'LEFT' in last_val_msg else second_to_last_index
    right_index = last_index if 'RIGHT' in last_val_msg else second_to_last_index
    right_msg = df_cal.loc[right_index].values[0]
    left_msg = df_cal.loc[left_index].values[0]
    lefterror_index, righterror_index = left_msg.split().index('ERROR'), right_msg.split().index('ERROR')
    left_error = float(left_msg.split()[lefterror_index + 1])
    right_error = float(right_msg.split()[righterror_index + 1])

    return 'L' if left_error < right_error else 'R'


def keep_eye(eye, df_samples, df_fix, df_blink, df_sacc):
    if eye == 'R':
        df_samples = df_samples[['tSample', 'RX', 'RY', 'RPupil', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']].copy()
        df_fix = df_fix[df_fix['eye'] == 'R'].reset_index(drop=True)
        df_blink = df_blink[df_blink['eye'] == 'R'].reset_index(drop=True)
        df_sacc = df_sacc[df_sacc['eye'] == 'R'].reset_index(drop=True)
        df_samples.rename(columns={'RX': 'X', 'RY': 'Y', 'RPupil': 'Pupil'}, inplace=True)
    else:
        df_samples = df_samples[['tSample', 'LX', 'LY', 'LPupil', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']].copy()
        df_fix = df_fix[df_fix['eye'] == 'L'].reset_index(drop=True)
        df_blink = df_blink[df_blink['eye'] == 'L'].reset_index(drop=True)
        df_sacc = df_sacc[df_sacc['eye'] == 'L'].reset_index(drop=True)
        df_samples.rename(columns={'LX': 'X', 'LY': 'Y', 'LPupil': 'Pupil'}, inplace=True)
    df_samples["eye"] = eye
    df_blink.dropna(inplace=True)
    df_fix.dropna(inplace=True)
    df_sacc.dropna(inplace=True)
    return df_samples, df_fix, df_blink, df_sacc


def dataset_to_bids(
    target_folder_path,
    files_folder_path,
    dataset_name,
    session_substrings=1,
    format_name='eyelink',
    *,
    task_name='eyetracking',
    authors=None,
    overwrite=False,
):
    """
    Convert eye-tracking recordings to BIDS and retain originals in sourcedata.

    Args:
        target_folder_path (str): Path to the folder where the BIDS dataset will be created.
        files_folder_path (str): Path to the folder containing vendor files.
            Filenames must begin with the subject identifier followed by an
            underscore.
        dataset_name (str): Name of the BIDS dataset.
        session_substrings (int): Number of filename tokens used for the session.
        format_name (str): One of ``eyelink``, ``tobii``, ``gaze``/``gazepoint``,
            or ``webgazer``.
        task_name (str): Fallback BIDS task label when it is absent from a filename.
        authors (sequence of str, optional): Dataset authors.
        overwrite (bool): Replace an existing output dataset when True.

    Returns:
        pathlib.Path
            Root of the written BIDS dataset.
    """
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
def process_session(raw_session_path, dataset_format, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, exp_format, **kwargs):
    # For BIDS, algorithm-specific recording labels let multiple detector
    # outputs coexist in one derivative session.
    if not overwrite and session_folder_path.exists():
        if exp_format == BIDS_EXPORT:
            label = ''.join(
                character for character in detection_algorithm.lower()
                if character.isalnum()
            ) or "pyxations"
            existing = (
                session_folder_path / "beh"
            ).glob(f"*_recording-eye1{label}_physio.tsv.gz")
            if next(existing, None) is not None:
                return
        elif any(session_folder_path.iterdir()):
            return
    
    if dataset_format not in {"eyelink", "webgazer", "tobii", "gaze"}:
        raise ValueError(f"Dataset format {dataset_format} not found.")
    process_bids_session(
        raw_session_path,
        dataset_format,
        detection_algorithm,
        session_folder_path,
        force_best_eye,
        keep_ascii,
        overwrite,
        exp_format,
        **kwargs,
    )


def compute_derivatives_for_dataset(bids_dataset_folder, dataset_format, detection_algorithm='remodnav', num_processes=4,
                                    force_best_eye=True, keep_ascii=True, overwrite=False, exp_format=BIDS_EXPORT,
                                    behavioral_columns=None, **kwargs):
    """Compute eye-tracking derivatives for every BIDS subject/session.

    BIDS TSV.GZ/JSON is the canonical output. The sibling derivatives dataset
    receives its own ``dataset_description.json`` and mirrors the source
    subject/session hierarchy. Feather remains an explicit compatibility
    export through ``exp_format``.

    Returns
    -------
    pathlib.Path
        Root of the written derivatives dataset.
    """
    derivatives_folder = Path(str(bids_dataset_folder) + "_derivatives")
    bids_dataset_folder = Path(bids_dataset_folder)
    derivatives_folder.mkdir(exist_ok=True)

    # ``export_format`` appeared in early examples; retain it as an alias.
    exp_format = kwargs.pop("export_format", exp_format)
    if exp_format == BIDS_EXPORT:
        initialize_bids_derivative(bids_dataset_folder, derivatives_folder)

    # Extract and remove start_times and end_times from kwargs if present
    start_times = kwargs.pop("start_times", None)
    end_times = kwargs.pop("end_times", None)

    if behavioral_columns is not None:
        kwargs["behavioral_columns"] = behavioral_columns

    bids_folders = [
        folder for folder in bids_dataset_folder.iterdir()
        if folder.is_dir() and folder.name.startswith("sub-")
    ]

    participants_file = bids_dataset_folder / "participants.tsv"
    participants_tsv = pd.read_csv(
        participants_file,
        sep="\t",
        dtype={'subject_id': str, 'old_subject_id': str},
    )

    jobs = []
    for subject in bids_folders:
        subject_name = participants_tsv.loc[
            participants_tsv['subject_id'] == subject.name[4:],
            'old_subject_id',
        ].values[0]
        subject_path = bids_dataset_folder / subject.name

        for session in subject_path.iterdir():
            if not (session.name.startswith("ses-") and session.is_dir()):
                continue
            session_name = session.name[4:]

            session_kwargs = dict(kwargs)
            if (
                start_times
                and subject_name in start_times
                and session_name in start_times[subject_name]
            ):
                session_kwargs["start_times"] = start_times[subject_name][
                    session_name
                ]
            if (
                end_times
                and subject_name in end_times
                and session_name in end_times[subject_name]
            ):
                session_kwargs["end_times"] = end_times[subject_name][
                    session_name
                ]

            jobs.append(
                (
                    session,
                    dataset_format,
                    detection_algorithm,
                    derivatives_folder / subject.name / session.name,
                    force_best_eye,
                    keep_ascii,
                    overwrite,
                    exp_format,
                    session_kwargs,
                )
            )

    if num_processes == 1:
        for (
            source_path,
            format_name,
            algorithm,
            destination,
            choose_eye,
            preserve_ascii,
            replace,
            output_format,
            session_kwargs,
        ) in jobs:
            process_session(
                source_path,
                format_name,
                algorithm,
                destination,
                choose_eye,
                preserve_ascii,
                replace,
                output_format,
                **session_kwargs,
            )
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    process_session,
                    source_path,
                    format_name,
                    algorithm,
                    destination,
                    choose_eye,
                    preserve_ascii,
                    replace,
                    output_format,
                    **session_kwargs,
                )
                for (
                    source_path,
                    format_name,
                    algorithm,
                    destination,
                    choose_eye,
                    preserve_ascii,
                    replace,
                    output_format,
                    session_kwargs,
                ) in jobs
            ]
            for future in futures:
                future.result()

    return derivatives_folder
        
