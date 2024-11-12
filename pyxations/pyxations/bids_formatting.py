from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .pre_processing import PreProcessing
from .eye_movement_detection import RemodnavDetection
import inspect
from pyxations.formats.webgazer.bids import WebGazerBidsConverter
from pyxations.formats.eyelink.bids import EyeLinkBidsConverter
import pyxations.formats.eyelink.parse as eyelink_parser 
import pyxations.formats.webgazer.parse as webgazer_parser
import pyxations.formats.tobii.parse as tobii_parser
from pyxations.formats.tobii.bids import TobiiBidsConverter


EYE_MOVEMENT_DETECTION_DICT = {'remodnav': RemodnavDetection}



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

def keep_eye(eye,df_samples,df_fix,df_blink,df_sacc):
    if eye == 'R':
        df_samples = df_samples[['tSample', 'RX', 'RY', 'RPupil','Line_number','Eyes_recorded','Rate_recorded','Calib_index']].copy()
        df_fix = df_fix[df_fix['eye'] == 'R'].reset_index(drop=True)
        df_blink = df_blink[df_blink['eye'] == 'R'].reset_index(drop=True)
        df_sacc = df_sacc[df_sacc['eye'] == 'R'].reset_index(drop=True)
        df_samples.rename(columns={'RX': 'X', 'RY': 'Y', 'RPupil': 'Pupil'}, inplace=True)
    else:
        df_samples = df_samples[['tSample', 'LX', 'LY', 'LPupil','Line_number','Eyes_recorded','Rate_recorded','Calib_index']].copy()
        df_fix = df_fix[df_fix['eye'] == 'L'].reset_index(drop=True)
        df_blink = df_blink[df_blink['eye'] == 'L'].reset_index(drop=True)
        df_sacc = df_sacc[df_sacc['eye'] == 'L'].reset_index(drop=True)
        df_samples.rename(columns={'LX': 'X', 'LY': 'Y', 'LPupil': 'Pupil'}, inplace=True)
    df_blink.dropna(inplace=True)
    df_fix.dropna(inplace=True)
    df_sacc.dropna(inplace=True)
    return df_samples,df_fix,df_blink,df_sacc

def get_converter(format_name):
    if format_name == 'webgazer':
        return WebGazerBidsConverter()
    elif format_name == 'eyelink':
        return EyeLinkBidsConverter()
    elif format_name == 'tobii':
        return TobiiBidsConverter()
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

    subj_ids =  converter.get_subject_ids(file_paths)

    # If all of the subjects have numerical IDs, sort them numerically, else sort them alphabetically
    if all(subject_id.isdigit() for subject_id in subj_ids):
        subj_ids.sort(key=int)
    else:
        subj_ids.sort()
    new_subj_ids = [str(subject_index).zfill(4) for subject_index in range(1,len(subj_ids)+1)]

    # Create subfolders for each session for each subject
    for subject_id in new_subj_ids:
        old_subject_id = subj_ids[int(subject_id) - 1]
        for file in file_paths:
            file_name = Path(file).name
            session_id = "_".join("".join(file_name.split(".")[:-1]).split("_")[1:session_substrings+1])
            converter.move_file_to_bids_folder(file, bids_folder_path, subject_id, old_subject_id, session_id)

        metadata.loc[len(metadata.index)] = [subject_id, old_subject_id]
    # Save metadata to tsv file
    metadata.to_csv(bids_folder_path / "participants.tsv", sep="\t", index=False)
    return bids_folder_path


def move_file_to_bids_folder(file_path, bids_folder_path, subject_id, session_id,tag):
    session_folder_path = bids_folder_path / ("sub-" + subject_id) / ("ses-" + session_id) / tag
    session_folder_path.mkdir(parents=True, exist_ok=True)
    new_file_path = session_folder_path / file_path.name
    if not new_file_path.exists():
        shutil.copy(file_path, session_folder_path)
    




def process_session(eye_tracking_data_path, msg_keywords, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):

    if detection_algorithm == 'eyelink':
        eyelink_parser.process_session(eye_tracking_data_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite)
        
    elif detection_algorithm == 'webgazer':
        webgazer_parser.process_session(eye_tracking_data_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite)
    elif detection_algorithm == 'tobii':
        tobii_parser.process_session(eye_tracking_data_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite)

def compute_derivatives_for_dataset(bids_dataset_folder, msg_keywords, detection_algorithm='eyelink', num_processes=4, force_best_eye=True, keep_ascii=True, overwrite=False, **kwargs):
    derivatives_folder = str(bids_dataset_folder) + "_derivatives"
    derivatives_folder = Path(derivatives_folder)
    bids_dataset_folder = Path(bids_dataset_folder)
    derivatives_folder.mkdir(exist_ok=True)


    bids_folders = [folder for folder in bids_dataset_folder.iterdir() if folder.is_dir() and folder.name.startswith("sub-")]
    if detection_algorithm not in EYE_MOVEMENT_DETECTION_DICT and detection_algorithm not in ['eyelink', 'webgazer', 'tobii']:
        raise ValueError(f"Detection algorithm {detection_algorithm} not found.")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:

        futures = [
            executor.submit(process_session, session / "ET", msg_keywords, detection_algorithm, derivatives_folder / subject.name / session.name, force_best_eye, keep_ascii, overwrite, **kwargs)
            for subject in bids_folders
            for session in (bids_dataset_folder / subject.name).iterdir() if session.name.startswith("ses-") and (bids_dataset_folder / subject.name / session.name).is_dir()
        ]

        for future in futures:
            future.result()

    return derivatives_folder
        