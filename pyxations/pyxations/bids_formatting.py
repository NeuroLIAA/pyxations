from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .pre_processing import PreProcessing
from .eye_movement_detection import RemodnavDetection
import inspect
import json


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


def dataset_to_bids(target_folder_path, files_folder_path, dataset_name, session_substrings=1):
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
    # Create a metadata json file
    metadata = {"subject_ids": {}}
    files_folder_path = Path(files_folder_path)
    # List all file paths in the folder
    file_paths = []
    for file_path in files_folder_path.rglob('*'):  # Recursively go through all files
        if file_path.is_file():
            file_paths.append(file_path)
    
    file_paths = [file for file in file_paths if file.suffix.lower() == '.edf' or file.suffix.lower() == '.bdf' or file.suffix.lower() == '.log' or file.suffix.lower() == '.csv']

    bids_folder_path = Path(target_folder_path) / dataset_name

    subj_ids = list(set([Path(file).name.split("_")[0] for file in file_paths if file.suffix.lower() == '.edf' or file.suffix.lower() == '.bdf']))

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
            file_lower = file_name.lower()
            session_id = "_".join("".join(file_name.split(".")[:-1]).split("_")[1:session_substrings+1])
            if file_lower.endswith(".edf") and file_name.split("_")[0] == old_subject_id:
                move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'ET')
            if file_lower.endswith(".bdf") and file_name.split("_")[0] == old_subject_id:
                move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'EEG')
            if (file_lower.endswith(".log") or file_lower.endswith(".csv")) and file_name.split("_")[0] == old_subject_id:                
                move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'behavioral')
        metadata['subject_ids'][subject_id] = old_subject_id
    # Save metadata to json file
    with open(bids_folder_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    return bids_folder_path

def move_file_to_bids_folder(file_path, bids_folder_path, subject_id, session_id,tag):
    session_folder_path = bids_folder_path / ("sub-" + subject_id) / ("ses-" + session_id) / tag
    session_folder_path.mkdir(parents=True, exist_ok=True)
    new_file_path = session_folder_path / file_path.name
    if not new_file_path.exists():
        shutil.copy(file_path, session_folder_path)
    

def convert_edf_to_ascii(edf_file_path, output_dir):
    """
    Convert an EDF file to ASCII format using edf2asc.

    Args:
        edf_file_path (str): Path to the input EDF file.
        output_dir (str): Directory to save the ASCII file. If None, the ASCII file will be saved in the same directory as the input EDF file.

    Returns:
        str: Path to the generated ASCII file.
    """
    # Check if edf2asc is installed
    if not shutil.which("edf2asc"):
        raise FileNotFoundError("edf2asc not found. Please make sure EyeLink software is installed and accessible in the system PATH.")

    # Set output directory
    if output_dir is None:
        raise ValueError("Output directory must be specified.")

    # Generate output file path
    edf_file_name = edf_file_path.name
    ascii_file_name = Path(edf_file_name).with_suffix('.asc')
    ascii_file_path = output_dir / ascii_file_name

    # Run edf2asc command with the -failsafe flag, only run it if the file does not already exist
    if not ascii_file_path.exists():
        subprocess.run(["edf2asc", "-failsafe", edf_file_path, ascii_file_path])

    return ascii_file_path



def parse_edf_eyelink(edf_file_path, msg_keywords, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):
    # Convert EDF to ASCII (only if necessary)
    ascii_file_path = convert_edf_to_ascii(edf_file_path, session_folder_path)

    # Check if all files exist, to avoid unnecessary reprocessing
    existing_files = all([
        (session_folder_path / file_name).exists()
        for file_name in ['header.hdf5', 'msg.hdf5', 'calib.hdf5', 'samples.hdf5']
    ])
    if existing_files and not overwrite:
        return

     # Reading ASCII in chunks to reduce memory usage
    with open(ascii_file_path, 'r') as f:
        lines = (line.strip() for line in f)  # Generator to save memory

        # Pre-allocate variables
        line_data = []
        line_types = []
        eyes_recorded = []
        rates_recorded = []
        calib_indexes = []

        # Initialize flags
        calibration_flag = False
        start_flag = False
        recorded_eye = ''
        rate_recorded = 0.0
        calib_index = 0

        # Process the file line by line
        for line in lines:
            if len(line)<2:
                line_type = 'EMPTY'
            elif line.startswith('*'):
                line_type = 'HEADER'
            # If there is a !CAL in the line, it is a calibration line
            elif '!CAL' in line and not calibration_flag:
                line_type = 'Calibration'
                calibration_flag = True
                calib_index += 1    
            elif '!MODE RECORD' in line and calibration_flag:
                calibration_flag = False
                start_flag = True
            elif calibration_flag:
                line_type = 'Calibration'
            elif not start_flag: # Data before the first successful calibration is discarded. 
                # After the first successul calibration, EVERY sample is taken into account.
                line_type = 'Non_calibrated_samples'
            elif line.split()[0] == 'MSG' and any(keyword in line for keyword in msg_keywords):
                line_type = 'MSG'
            elif line.split()[0] == 'ESACC':
                line_type = 'ESACC'
            elif line.split()[0] == 'EFIX':
                line_type = 'EFIX'
            elif line.split()[0] == 'EBLINK':
                line_type = 'EBLINK'
            elif line.split()[0][0].isdigit() or line.split()[0].startswith('-'):
                line_type = 'SAMPLE'
            else:
                line_type = 'OTHER'
            if '!MODE RECORD' in line:
                recorded_eye = line.split()[-1]            
            if 'RATE' in line and 'TRACKING' in line:
                rate_recorded = float(line.split('RATE')[-1].split('TRACKING')[0])

            # Store relevant information
            line_data.append(line.replace('\n', '').replace('\t', ' '))
            line_types.append(line_type)
            eyes_recorded.append(recorded_eye)
            rates_recorded.append(rate_recorded)
            calib_indexes.append(calib_index)
    
    # Convert to DataFrame (in one step to save memory)
    df = pd.DataFrame({
        'line': line_data,
        'Line_type': line_types,
        'Eyes_recorded': eyes_recorded,
        'Rate_recorded': rates_recorded,
        'Calib_index': calib_indexes
    })
    # Process DataFrame columns (vectorized operations)

    df['Line_number'] = np.arange(len(df))

    
    # Separate lines into different types
    dfHeader = df[df['Line_type'] == 'HEADER'][['line', 'Line_number']].reset_index(drop=True)
    dfCalib = df[df['Line_type'] == 'Calibration'][['line', 'Line_number', 'Calib_index']].reset_index(drop=True)
    dfMsg = df[df['Line_type'] == 'MSG'][['line', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']].reset_index(drop=True)

    # Process samples and events only for required lines
    dfSamples = df[df['Line_type'] == 'SAMPLE'][['line', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']].reset_index(drop=True)
    dfFix = df[df['Line_type'] == 'EFIX'][['line', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']].reset_index(drop=True)
    dfSacc = df[df['Line_type'] == 'ESACC'][['line', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']].reset_index(drop=True)
    dfBlink = df[df['Line_type'] == 'EBLINK'][['line', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']].reset_index(drop=True)
    del df, line_data, line_types, eyes_recorded, rates_recorded, calib_indexes # Free up memory
    # Optimized screen resolution extraction from dfCalib
    gaze_coords_row = dfCalib.loc[dfCalib['line'].str.contains('GAZE_COORDS'), 'line'].values[0]
    screen_res = [str(int(float(res))) for res in gaze_coords_row.split()[5:7]]
    dfHeader.loc[len(dfHeader.index)] = ["** SCREEN SIZE: " + " ".join(screen_res), -1]

    # Screen size extraction optimization
    if 'screen_height' not in kwargs or 'screen_width' not in kwargs:
        screen_size = dfHeader['line'].iloc[-1].split()
        kwargs['screen_width'], kwargs['screen_height'] = int(screen_size[-2]), int(screen_size[-1])

    # Optimized processing of dfMsg to extract timestamp and message
    if dfMsg.empty:
        raise ValueError(f"No messages {msg_keywords} found in the ASC file for session {session_folder_path}.")
    # Extracting timestamp and message in a single step
    dfMsg[['timestamp', 'message']] = dfMsg['line'].str.replace('MSG ', '').str.split(n=1,expand=True).values
    dfMsg.drop(columns=['line'], inplace=True)

    # Convert timestamp to numeric in one operation
    dfMsg['timestamp'] = pd.to_numeric(dfMsg['timestamp'], errors='raise')
    dfMsg = dfMsg[['timestamp', 'message', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']]

   # Optimized blink data extraction and conversion
    dfBlink['line'] = dfBlink['line'].str.replace('EBLINK ', '')
    dfBlink[['eye', 'tStart', 'tEnd', 'duration']] = dfBlink['line'].str.split(expand=True)
    dfBlink.drop(columns=['line'], inplace=True)
    dfBlink = dfBlink[['eye', 'tStart', 'tEnd', 'duration', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']]
    dfBlink[['tStart', 'tEnd', 'duration']] = dfBlink[['tStart', 'tEnd', 'duration']].apply(pd.to_numeric, errors='raise')

    if not dfSamples[dfSamples['Eyes_recorded'] == 'LR'].empty:
        dfSamples.loc[dfSamples[dfSamples['Eyes_recorded'] == 'LR'].index, ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']] = dfSamples[dfSamples['Eyes_recorded'] == 'LR']['line'].str.split(expand=True)[[0, 1, 2, 3, 4, 5, 6]].apply(pd.to_numeric, errors='coerce').values

    for eye, cols in zip(['R', 'L'], [['RX', 'RY', 'RPupil'], ['LX', 'LY', 'LPupil']]):
        if not dfSamples[dfSamples['Eyes_recorded'] == eye].empty:
            dfSamples.loc[dfSamples[dfSamples['Eyes_recorded'] == eye].index, ['tSample'] + cols] = dfSamples[dfSamples['Eyes_recorded'] == eye]['line'].str.split(expand=True)[[0] + list(range(1, len(cols) + 1))].apply(pd.to_numeric, errors='coerce').values

    dfSamples.drop(columns=['line'], inplace=True)

    dfSamples = dfSamples[['tSample'] + [col for col in ['LX', 'LY','LPupil','RX', 'RY', 'RPupil'] if col in dfSamples.columns] + ['Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']]
    if detection_algorithm == 'eyelink':
        # Optimized fixation and saccade processing
        dfFix['line'] = dfFix['line'].str.replace('EFIX ', '')
        dfFix[['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg']] = dfFix['line'].str.split(expand=True)
        dfFix.drop(columns=['line'], inplace=True)
        dfFix[['xAvg', 'yAvg', 'pupilAvg', 'tStart', 'tEnd', 'duration']] = dfFix[['xAvg', 'yAvg', 'pupilAvg', 'tStart', 'tEnd', 'duration']].apply(pd.to_numeric, errors='coerce')
        dfFix.dropna(inplace=True)
        dfFix = dfFix[['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']]

        dfSacc['line'] = dfSacc['line'].str.replace('ESACC ', '')
        dfSacc[['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']] = dfSacc['line'].str.split(expand=True)
        dfSacc.drop(columns=['line'], inplace=True)
        dfSacc[['xStart', 'yStart', 'xEnd', 'yEnd', 'duration', 'ampDeg', 'vPeak', 'tStart', 'tEnd']] = dfSacc[['xStart', 'yStart', 'xEnd', 'yEnd', 'duration', 'ampDeg', 'vPeak', 'tStart', 'tEnd']].apply(pd.to_numeric, errors='coerce')
        dfSacc.dropna(inplace=True)
        dfSacc = dfSacc[['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak', 'Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']]

    else:
        eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=session_folder_path,samples=dfSamples)
        dfFix, dfSacc = eye_movement_detector.detect_eye_movements(**{arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(eye_movement_detector.detect_eye_movements).parameters.keys()})

    # Optimization for selecting best eye
    if force_best_eye:
        calib_indexes = dfCalib['Calib_index'].unique()
        best_eyes = dfCalib.groupby('Calib_index').apply(find_besteye).values
        if best_eyes[0] == 'M':
            # TODO: everything in samples, fix, blink, sacc for the first best_eyes that are equal to 'M' should be marked as uncalibrated, using the calib_indexes to do so.
            raise ValueError("No calibration validation found in the first block.")
        # Replace the 'M' values with the previous value that is not 'M'
        for i in range(1, len(best_eyes)):
            if best_eyes[i] == 'M':
                best_eyes[i] = best_eyes[i - 1]
        dfslist = [keep_eye(best_eyes[i], dfSamples[dfSamples['Calib_index'] == ci], dfFix[dfFix['Calib_index'] == ci], dfBlink[dfBlink['Calib_index'] == ci], dfSacc[dfSacc['Calib_index'] == ci]) for i, ci in enumerate(calib_indexes)]
        dfSamples, dfFix, dfBlink, dfSacc = [pd.concat([dfslist[i][j] for i in range(len(best_eyes))]) for j in range(4)]
        del dfslist


    
    pre_processing = PreProcessing(dfSamples, dfFix,dfSacc,dfBlink, dfMsg)
    pre_processing.process({'bad_samples': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.bad_samples).parameters.keys()},
                            'split_all_into_trials_by_msgs': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.split_all_into_trials_by_msgs).parameters.keys()},
                            'saccades_direction': {},})

    if not keep_ascii:
        ascii_file_path.unlink(missing_ok=True)

    # Save DataFrames to disk in one go to minimize memory usage during processing
    dfHeader.to_hdf((session_folder_path / 'header.hdf5'), key='header', mode='w')
    dfMsg.to_hdf((session_folder_path / 'msg.hdf5'), key='msg', mode='w')
    dfCalib.to_hdf((session_folder_path / 'calib.hdf5'), key='calib', mode='w')
    dfSamples.to_hdf((session_folder_path / 'samples.hdf5'), key='samples', mode='w')
    dfBlink.to_hdf((session_folder_path / 'eyelink_events' / 'blink.hdf5'), key='blink', mode='w')
    dfFix.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'fix.hdf5'), key='fix', mode='w')
    dfSacc.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'sacc.hdf5'), key='sacc', mode='w')



def process_session(eye_tracking_data_path, msg_keywords,detection_algorithm,session_folder_path,force_best_eye,keep_ascii, overwrite, **kwargs):
    edf_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.edf']
    if len(edf_files) > 1:
        print(f"More than one EDF file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = edf_files[0]
    (session_folder_path / 'eyelink_events').mkdir(parents=True, exist_ok=True)

    parse_edf_eyelink(edf_file_path, msg_keywords,detection_algorithm,session_folder_path,force_best_eye,keep_ascii, overwrite, **kwargs)

def compute_derivatives_for_dataset(bids_dataset_folder, msg_keywords, detection_algorithm='eyelink', num_processes=4, force_best_eye=True, keep_ascii=True, overwrite=False, **kwargs):
    derivatives_folder = bids_dataset_folder + "_derivatives"
    derivatives_folder = Path(derivatives_folder)
    bids_dataset_folder = Path(bids_dataset_folder)
    derivatives_folder.mkdir(exist_ok=True)


    bids_folders = [folder for folder in bids_dataset_folder.iterdir() if folder.is_dir() and folder.name.startswith("sub-")]
    if detection_algorithm not in EYE_MOVEMENT_DETECTION_DICT and detection_algorithm != 'eyelink':
        raise ValueError(f"Detection algorithm {detection_algorithm} not found.")



    with ProcessPoolExecutor(max_workers=num_processes) as executor:

        futures = [
            executor.submit(process_session, session / "ET", msg_keywords, detection_algorithm, derivatives_folder / subject.name / session.name, force_best_eye, keep_ascii, overwrite, **kwargs)
            for subject in bids_folders
            for session in (bids_dataset_folder / subject).iterdir() if session.name.startswith("ses-") and (bids_dataset_folder / subject / session).is_dir()
        ]

        for future in futures:
            future.result()

    return derivatives_folder
        