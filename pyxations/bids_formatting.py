from pathlib import Path
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pyxations.methods.eyemovement.REMoDNaV import RemodnavDetection
from pyxations.methods.eyemovement.engbert import EngbertDetection

from pyxations.formats.webgazer.bids import WebGazerBidsConverter
from pyxations.formats.eyelink.bids import EyeLinkBidsConverter
import pyxations.formats.eyelink.parse as eyelink_parser 
import pyxations.formats.webgazer.parse as webgazer_parser
import pyxations.formats.tobii.parse as tobii_parser
import pyxations.formats.gazepoint.parse as gaze_parser
from pyxations.formats.tobii.bids import TobiiBidsConverter
from pyxations.formats.gazepoint.bids import GazepointBidsConverter
from pyxations.export import FEATHER_EXPORT

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
    df_blink.dropna(inplace=True)
    df_fix.dropna(inplace=True)
    df_sacc.dropna(inplace=True)
    return df_samples, df_fix, df_blink, df_sacc


def get_converter(format_name):
    if format_name == 'webgazer':
        return WebGazerBidsConverter()
    elif format_name == 'eyelink':
        return EyeLinkBidsConverter()
    elif format_name == 'tobii':
        return TobiiBidsConverter()
    elif format_name == 'gaze':
        return GazepointBidsConverter()
    return None


def dataset_to_bids(target_folder_path, files_folder_path, dataset_name, session_substrings=1, format_name='eyelink'):
    """
    Convert a dataset to BIDS format.

    Args:
        target_folder_path (str): Path to the folder where the BIDS dataset will be created.
        files_folder_path (str): Path to the folder containing the EDF files.
        The EDF files are assumed to have the ID of the subject at the beginning of the file name, separated by an underscore.
        dataset_name (str): Name of the BIDS dataset.
        session_substrings (int): Number of substrings to use for the session ID. Default is 1.

    Returns:
        None
    """
    converter = get_converter(format_name)
    
    # Create a metadata tsv file
    metadata = pd.DataFrame(columns=['subject_id', 'old_subject_id'])
    files_folder_path = Path(files_folder_path)
    # List all file paths in the folder
    file_paths = []
    for file_path in files_folder_path.rglob('*'):  # Recursively go through all files
        if file_path.is_file():
            file_paths.append(file_path)
    
    file_paths = [file for file in file_paths if file.suffix.lower() in converter.relevant_extensions()]

    bids_folder_path = Path(target_folder_path) / dataset_name
    bids_folder_path.mkdir(parents=True, exist_ok=True)

    subj_ids = converter.get_subject_ids(file_paths)

    # If all of the subjects have numerical IDs, sort them numerically, else sort them alphabetically
    if all(subject_id.isdigit() for subject_id in subj_ids):
        subj_ids.sort(key=int)
    else:
        subj_ids.sort()
    new_subj_ids = [str(subject_index).zfill(4) for subject_index in range(1, len(subj_ids) + 1)]

    # Create subfolders for each session for each subject
    for subject_id in new_subj_ids:
        old_subject_id = subj_ids[int(subject_id) - 1]
        for file in file_paths:
            file_name = Path(file).name
            session_id = "_".join("".join(file_name.split(".")[:-1]).split("_")[1:session_substrings + 1])
            converter.move_file_to_bids_folder(file, bids_folder_path, subject_id, old_subject_id, session_id)

        metadata.loc[len(metadata.index)] = [subject_id, old_subject_id]
    # Save metadata to tsv file
    metadata.to_csv(bids_folder_path / "participants.tsv", sep="\t", index=False)
    return bids_folder_path


def move_file_to_bids_folder(file_path, bids_folder_path, subject_id, session_id, tag):
    session_folder_path = bids_folder_path / ("sub-" + subject_id) / ("ses-" + session_id) / tag
    session_folder_path.mkdir(parents=True, exist_ok=True)
    new_file_path = session_folder_path / file_path.name
    if not new_file_path.exists():
        shutil.copy(file_path, session_folder_path)


def process_session(eye_tracking_data_path, dataset_format, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, exp_format, **kwargs):
    # If session folder path has files and overwrite is False, return
    if not overwrite and session_folder_path.exists() and any(session_folder_path.iterdir()):
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
                                    force_best_eye=True, keep_ascii=True, overwrite=False, exp_format=FEATHER_EXPORT, **kwargs):
    derivatives_folder = Path(str(bids_dataset_folder) + "_derivatives")
    bids_dataset_folder = Path(bids_dataset_folder)
    derivatives_folder.mkdir(exist_ok=True)

    # Extract and remove start_times and end_times from kwargs if present
    start_times = kwargs.pop("start_times", None)
    end_times = kwargs.pop("end_times", None)

    bids_folders = [
        folder for folder in bids_dataset_folder.iterdir()
        if folder.is_dir() and folder.name.startswith("sub-")
    ]

    participants_file = bids_dataset_folder / "participants.tsv"
    participants_tsv = pd.read_csv(participants_file, sep="\t",dtype={'subject_id': str, 'old_subject_id': str})

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

                    futures.append(
                        executor.submit(
                            process_session,
                            session / "ET", dataset_format, detection_algorithm,
                            derivatives_folder / subject.name / session.name,
                            force_best_eye, keep_ascii, overwrite, exp_format,
                            **session_kwargs
                        )
                    )

        for future in futures:
            future.result()

    return derivatives_folder
        
