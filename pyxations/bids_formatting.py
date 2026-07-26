from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pyxations.methods.eyemovement.REMoDNaV import RemodnavDetection
from pyxations.methods.eyemovement.engbert import EngbertDetection

import pyxations.formats.eyelink.parse as eyelink_parser 
import pyxations.formats.webgazer.parse as webgazer_parser
import pyxations.formats.tobii.parse as tobii_parser
import pyxations.formats.gazepoint.parse as gaze_parser
from pyxations.export import BIDS_EXPORT
from pyxations.bids import write_bids_dataset
from pyxations.export.bids import initialize_bids_derivative

EYE_MOVEMENT_DETECTION_DICT = {'remodnav': RemodnavDetection, 'engbert': EngbertDetection}


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
def process_session(eye_tracking_data_path, dataset_format, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, exp_format, **kwargs):
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
    
    if dataset_format == 'eyelink':
        eyelink_parser.process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, exp_format, **kwargs)
        
    elif dataset_format == 'webgazer':
        webgazer_parser.process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, overwrite, exp_format, **kwargs)
    elif dataset_format == 'tobii':
        tobii_parser.process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, overwrite, exp_format, **kwargs)
    elif dataset_format == 'gaze':
        gaze_parser.process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, exp_format, **kwargs)
    else:
        raise ValueError(f"Dataset format {dataset_format} not found.")


def compute_derivatives_for_dataset(bids_dataset_folder, dataset_format, detection_algorithm='remodnav', num_processes=4,
                                    force_best_eye=True, keep_ascii=True, overwrite=False, exp_format=BIDS_EXPORT,
                                    behavioral_columns=None, **kwargs):
    """Compute eye-tracking derivatives for every BIDS subject/session.

    BIDS TSV.GZ/JSON is the canonical output. The sibling derivatives dataset
    receives its own ``dataset_description.json`` and mirrors the source
    subject/session hierarchy. Feather and HDF5 remain explicit compatibility
    exports through ``exp_format``.

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

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for subject in bids_folders:
            # To get subject_name go to the bids_dataset_folder and open the "participants.tsv" file
            # There are two columns: subject_id and old_subject_id
            # subject_id equals subject.name[4:] and old_subject_id is the one we want to use in this case


            subject_name = participants_tsv.loc[participants_tsv['subject_id'] == subject.name[4:], 'old_subject_id'].values[0]
            subject_path = bids_dataset_folder / subject.name

            for session in subject_path.iterdir():
                if session.name.startswith("ses-") and session.is_dir():
                    session_name = session.name[4:]  # Remove "ses-" prefix

                    # Build per-session kwargs
                    session_kwargs = dict(kwargs)  # base kwargs
                    if start_times and subject_name in start_times and session_name in start_times[subject_name]:
                        session_kwargs["start_times"] = start_times[subject_name][session_name]
                    if end_times and subject_name in end_times and session_name in end_times[subject_name]:
                        session_kwargs["end_times"] = end_times[subject_name][session_name]

                    source_session = (
                        bids_dataset_folder
                        / "sourcedata"
                        / subject.name
                        / session.name
                    )
                    if not source_session.exists():
                        source_session = session

                    futures.append(
                        executor.submit(
                            process_session,
                            source_session / "ET", dataset_format, detection_algorithm,
                            derivatives_folder / subject.name / session.name,
                            force_best_eye, keep_ascii, overwrite, exp_format,
                            **session_kwargs
                        )
                    )

        for future in futures:
            future.result()

    return derivatives_folder
        
