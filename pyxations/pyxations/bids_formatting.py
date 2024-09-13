import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import logging
from .visualization import Visualization
from .pre_processing import PreProcessing
from .eye_movement_detection import RemodnavDetection
import inspect

EYE_MOVEMENT_DETECTION_DICT = {'remodnav': RemodnavDetection}


def find_besteye(df_cal, default='R'):
    if df_cal[df_cal['line'].str.contains('CAL VALIDATION')].index.empty:
        return default
    last_index = df_cal[df_cal['line'].str.contains('CAL VALIDATION')].index[-1]
    last_val_msg = df_cal.loc[last_index].values[0]
    if not len(last_val_msg) or 'ABORTED' in last_val_msg:
        return default
    second_to_last_index = last_index - 1
    second_to_last_msg = df_cal.loc[second_to_last_index].values[0]
    if '!CAL VALIDATION' not in second_to_last_msg:
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
        df_fix = df_fix[df_fix['eye'] == 'R']
        df_blink = df_blink[df_blink['eye'] == 'R']
        df_sacc = df_sacc[df_sacc['eye'] == 'R']
        df_samples.rename(columns={'RX': 'X', 'RY': 'Y', 'RPupil': 'Pupil'}, inplace=True)
    else:
        df_samples = df_samples[['tSample', 'LX', 'LY', 'LPupil','Line_number','Eyes_recorded','Rate_recorded','Calib_index']].copy()
        df_fix = df_fix[df_fix['eye'] == 'L']
        df_blink = df_blink[df_blink['eye'] == 'L']
        df_sacc = df_sacc[df_sacc['eye'] == 'L']
        df_samples.rename(columns={'LX': 'X', 'LY': 'Y', 'LPupil': 'Pupil'}, inplace=True)
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

    # List all file paths in the folder
    file_paths = []
    for root, _, files in os.walk(files_folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    file_paths = [file for file in file_paths if file.lower().endswith(".edf") or file.lower().endswith(".bdf") or file.lower().endswith(".log") or file.lower().endswith(".csv")]

    bids_folder_path = os.path.join(target_folder_path, dataset_name)

    subj_ids = list(set([os.path.basename(file).split("_")[0] for file in file_paths if file.lower().endswith(".edf") or file.lower().endswith(".bdf")]))

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
            file_name = os.path.basename(file)
            file_lower = file_name.lower()
            session_id = "_".join("".join(file_name.split(".")[:-1]).split("_")[1:session_substrings+1])
            if file_lower.endswith(".edf") and file_name.split("_")[0] == old_subject_id:
                move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'ET')
            if file_lower.endswith(".bdf") and file_name.split("_")[0] == old_subject_id:
                move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'EEG')
            if (file_lower.endswith(".log") or file_lower.endswith(".csv")) and file_name.split("_")[0] == old_subject_id:                
                move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'behavioral')
    return bids_folder_path

def move_file_to_bids_folder(file_path, bids_folder_path, subject_id, session_id,tag):
    session_folder_path = os.path.join(bids_folder_path, "sub-" + subject_id, "ses-" + session_id,tag)
    os.makedirs(session_folder_path, exist_ok=True)
    new_file_path = os.path.join(session_folder_path, os.path.basename(file_path))
    if not os.path.exists(new_file_path):
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
    edf_file_name = os.path.basename(edf_file_path)
    ascii_file_name = os.path.splitext(edf_file_name)[0] + ".asc"
    ascii_file_path = os.path.join(output_dir, ascii_file_name)

    # Run edf2asc command with the -failsafe flag, only run it if the file does not already exist
    if not os.path.exists(ascii_file_path):
        subprocess.run(["edf2asc", "-failsafe", edf_file_path, ascii_file_path])

    return ascii_file_path



def parse_edf_eyelink(edf_file_path, msg_keywords,detection_algorithm,session_folder_path,force_best_eye,keep_ascii,**kwargs):
    """
    Parse an EDF file generated by EyeLink system.
    Adapted from `ParseEyeLinkAsc` by DJ. (https://github.com/djangraw/ParseEyeLinkAscFiles)

    Args:
        edf_file_path (str): Path to the input EDF file.
        msg_keywords (list of str): List of strings representing keywords to filter MSG lines.

    Returns:
        tuple: A tuple containing five pandas DataFrames:
            - Header information DataFrame
            - MSG lines DataFrame filtered by msg_keywords
            - Calibration information DataFrame
            - EyeLink events DataFrame
            - Raw sample data DataFrame
    """
    # Configure logging
    derivatives_folder_path = os.path.dirname(os.path.dirname(session_folder_path))
    logging.basicConfig(filename=os.path.join(derivatives_folder_path,'derivatives_generation.log'),level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
    # Convert EDF to ASCII
    ascii_file_path = convert_edf_to_ascii(edf_file_path,session_folder_path)

    #Check if files already exist:
    if os.path.exists(os.path.join(session_folder_path, 'header.hdf5')) and os.path.exists(os.path.join(session_folder_path, 'msg.hdf5')) and os.path.exists(os.path.join(session_folder_path, 'calib.hdf5')) and os.path.exists(os.path.join(session_folder_path, 'samples.hdf5')):
        dfFix = pd.read_hdf(os.path.join(session_folder_path, detection_algorithm+'_events', 'fix.hdf5'))
        dfSacc = pd.read_hdf(os.path.join(session_folder_path, detection_algorithm+'_events', 'sacc.hdf5'))
        dfSamples = pd.read_hdf(os.path.join(session_folder_path, 'samples.hdf5'))
        visualization = Visualization(session_folder_path,detection_algorithm,dfFix,dfSacc,dfSamples)
        return dfFix,dfSacc
    else:
        # ===== READ IN FILES ===== #
        # Read in EyeLink file
        
        f = open(ascii_file_path,'r')
        fileTxt0 = f.read().splitlines(True) # split into lines
        fileTxt0 = np.array(fileTxt0) # concert to np array for simpler indexing
        f.close()

        subject = os.path.basename(edf_file_path).split("_")[0]
        # Separate lines into samples and messages
        logging.info(f'Sorting lines for subject {subject}...')
        nLines = len(fileTxt0)
        lineType = np.array(['OTHER']*nLines,dtype='object')
        eyes_recorded = np.array(['']*nLines,dtype='object')
        rates_recorded = np.array([0.0]*nLines,dtype='float')
        calib_indexes = np.array([0]*nLines,dtype='int')

        calibration_flag = False
        start_flag = False
        recorded_eye = ''
        rate_recorded = 0.0
        calib_index = 0
        for iLine in range(nLines):    
            if len(fileTxt0[iLine])<2:
                lineType[iLine] = 'EMPTY'
            elif fileTxt0[iLine].startswith('*'):
                lineType[iLine] = 'HEADER'
            # If there is a !CAL in the line, it is a calibration line
            elif '!CAL' in fileTxt0[iLine] and not calibration_flag:
                lineType[iLine] = 'Calibration'
                calibration_flag = True
                calib_index += 1    
            elif '!MODE RECORD' in fileTxt0[iLine] and calibration_flag:
                calibration_flag = False
                start_flag = True
            elif calibration_flag:
                lineType[iLine] = 'Calibration'
            elif not start_flag: # Data before the first successful calibration is discarded. 
                # After the first successul calibration, EVERY sample is taken into account.
                lineType[iLine] = 'Non_calibrated_samples'
            elif fileTxt0[iLine].split()[0] == 'MSG' and any(keyword in fileTxt0[iLine] for keyword in msg_keywords):
                lineType[iLine] = 'MSG'
            elif fileTxt0[iLine].split()[0] == 'ESACC':
                lineType[iLine] = 'ESACC'
            elif fileTxt0[iLine].split()[0] == 'EFIX':
                lineType[iLine] = 'EFIX'
            elif fileTxt0[iLine].split()[0] == 'EBLINK':
                lineType[iLine] = 'EBLINK'
            elif fileTxt0[iLine].split()[0][0].isdigit() or fileTxt0[iLine].split()[0].startswith('-'):
                lineType[iLine] = 'SAMPLE'
            else:
                lineType[iLine] = 'OTHER'
            if '!MODE RECORD' in fileTxt0[iLine]:
                recorded_eye = fileTxt0[iLine].split()[-1]            
            if 'RATE' in fileTxt0[iLine] and 'TRACKING' in fileTxt0[iLine]:
                rate_recorded = float(fileTxt0[iLine].split('RATE')[-1].split('TRACKING')[0])
            eyes_recorded[iLine] = recorded_eye
            rates_recorded[iLine] = rate_recorded
            calib_indexes[iLine] = calib_index

        df = pd.DataFrame(fileTxt0)
        df.rename(columns={0:'line'},inplace=True)
        df['line'] = df['line'].apply(lambda x: x.replace('\n','').replace('\t',' '))
        df["Eyes_recorded"] = eyes_recorded
        df["Rate_recorded"] = rates_recorded
        df["Calib_index"] = calib_indexes
        df["Line_type"] = lineType
        df["Line_number"] = np.arange(nLines)

        dfHeader = df[df["Line_type"] == 'HEADER'].reset_index(drop=True)[['line','Line_number']]
        dfCalib = df[df["Line_type"] == 'Calibration'].reset_index(drop=True)[['line','Line_number','Calib_index']]
        dfMsg = df[df["Line_type"] == 'MSG'].reset_index(drop=True)[['line','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        dfSamples = df[df["Line_type"] == 'SAMPLE'].reset_index(drop=True)[['line','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        dfFix = df[df["Line_type"] == 'EFIX'].reset_index(drop=True)[['line','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        dfSacc = df[df["Line_type"] == 'ESACC'].reset_index(drop=True)[['line','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        dfBlink = df[df["Line_type"] == 'EBLINK'].reset_index(drop=True)[['line','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        i_gaze_coords = dfCalib['line'].str.contains('GAZE_COORDS')
        # Get the first row that matches the condition
        i_gaze_coords = i_gaze_coords[i_gaze_coords].index[0]
        # Grab the value of that row in the first column, turn it into a string and split it by spaces, then get the 5th and 6th elements, which are the screen resolution
        screen_res = dfCalib.iloc[i_gaze_coords]['line'].split()[5:7]
        # Parse them as integers and then back to strings
        screen_res = [str(int(float(res))) for res in screen_res]
        dfHeader.loc[len(dfHeader.index)] = ["** SCREEN SIZE: "+ " ".join(screen_res),-1]


        # From dfMsg grab the "line" column and split into three two columns: "timestamp" and "message"
        if len(dfMsg) == 0:
            raise ValueError(f"No messages {msg_keywords} found in the ASC file for session {session_folder_path}.")
        dfMsg['line'] = dfMsg['line'].str.replace('MSG ','')
        dfMsg[['timestamp','message']] = dfMsg['line'].str.split(n=1,expand=True)
        dfMsg.drop(columns=['line'],inplace=True)
        # Reorder columns to havce timestamp first, then message, then line number and calibration index
        dfMsg = dfMsg[['timestamp','message','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        dfMsg['timestamp'] = pd.to_numeric(dfMsg['timestamp'], errors='raise')

        # From dfSamples grab the "line" column where Eye_recorded == LR and split into ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
        dfSamples[['tSample','LX','LY','LPupil','RX','RY','RPupil']] = np.nan
        dfSamples_LR = dfSamples[dfSamples['Eyes_recorded'] == 'LR']

        if len(dfSamples_LR) > 0:
            dfSamples.loc[dfSamples_LR.index,['tSample','LX','LY','LPupil','RX','RY','RPupil']] = dfSamples_LR['line'].str.split(expand=True)[[0,1,2,3,4,5,6]].values

            
        dfSamples_R = dfSamples[dfSamples['Eyes_recorded'] == 'R']
        if len(dfSamples_R) > 0:
            dfSamples.loc[dfSamples_R.index,['tSample','RX','RY','RPupil']] = dfSamples_R['line'].str.split(expand=True)[[0,1,2,3]].values

        dfSamples_L = dfSamples[dfSamples['Eyes_recorded'] == 'L']
        if len(dfSamples_L) > 0:
            dfSamples.loc[dfSamples_L.index,['tSample','LX','LY','LPupil']] = dfSamples_L['line'].str.split(expand=True)[[0,1,2,3]].values


        dfSamples = dfSamples[['tSample','LX','LY','LPupil','RX','RY','RPupil','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        for eye in ['L', 'R']:
            dfSamples['%cX' % eye] = pd.to_numeric(dfSamples['%cX' % eye], errors='coerce')
            dfSamples['%cY' % eye] = pd.to_numeric(dfSamples['%cY' % eye], errors='coerce')
            dfSamples['%cPupil' % eye] = pd.to_numeric(dfSamples['%cPupil' % eye], errors='coerce')
        dfSamples['tSample'] = pd.to_numeric(dfSamples['tSample'], errors='raise')
        # If LX LY RX and RY are all NaN, drop the row
        dfSamples = dfSamples.dropna(subset=['LX','LY','RX','RY'], how='all')

        dfBlink['line'] = dfBlink['line'].str.replace('EBLINK ','')
        # From dfBlink grab the "line" column and split into ['eye', 'tStart', 'tEnd', 'duration']
        dfBlink[['eye','tStart','tEnd','duration']] = dfBlink['line'].str.split(expand=True)
        dfBlink.drop(columns=['line'],inplace=True)
        dfBlink = dfBlink[['eye','tStart','tEnd','duration','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
        dfBlink['tStart'] = pd.to_numeric(dfBlink['tStart'], errors='raise')
        dfBlink['tEnd'] = pd.to_numeric(dfBlink['tEnd'], errors='raise')
        dfBlink['duration'] = pd.to_numeric(dfBlink['duration'], errors='raise')
        if detection_algorithm == 'eyelink':
            # From dfFix grab the "line" column and split into ['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg']
            dfFix['line'] = dfFix['line'].str.replace('EFIX ','')
            dfFix[['eye','tStart','tEnd','duration','xAvg','yAvg','pupilAvg']] = dfFix['line'].str.split(expand=True)
            dfFix.drop(columns=['line'],inplace=True)
            dfFix = dfFix[['eye','tStart','tEnd','duration','xAvg','yAvg','pupilAvg','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
            filtering = dfFix['xAvg'].apply(lambda x: not any(char.isdigit() for char in x)) | dfFix['yAvg'].apply(lambda x: not any(char.isdigit() for char in x))
            dfFix = dfFix[~filtering]
            dfFix['xAvg'] = pd.to_numeric(dfFix['xAvg'], errors='raise')
            dfFix['yAvg'] = pd.to_numeric(dfFix['yAvg'], errors='raise')
            dfFix['duration'] = pd.to_numeric(dfFix['duration'], errors='raise')
            dfFix['pupilAvg'] = pd.to_numeric(dfFix['pupilAvg'], errors='raise')
            dfFix['tStart'] = pd.to_numeric(dfFix['tStart'], errors='raise')
            dfFix['tEnd'] = pd.to_numeric(dfFix['tEnd'], errors='raise')

            dfSacc['line'] = dfSacc['line'].str.replace('ESACC ','')
            # From dfFix grab the "line" column and split into ['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']
            dfSacc[['eye','tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak']] = dfSacc['line'].str.split(expand=True)
            dfSacc.drop(columns=['line'],inplace=True)
            dfSacc = dfSacc[['eye','tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak','Line_number','Eyes_recorded','Rate_recorded','Calib_index']]
            filtering = dfSacc['xStart'].apply(lambda x: not any(char.isdigit() for char in x)) | dfSacc['yStart'].apply(lambda x: not any(char.isdigit() for char in x)) | dfSacc['xEnd'].apply(lambda x: not any(char.isdigit() for char in x)) | dfSacc['yEnd'].apply(lambda x: not any(char.isdigit() for char in x))
            dfSacc = dfSacc[~filtering]

            dfSacc['xStart'] = pd.to_numeric(dfSacc['xStart'], errors='raise')
            dfSacc['yStart'] = pd.to_numeric(dfSacc['yStart'], errors='raise')
            dfSacc['xEnd'] = pd.to_numeric(dfSacc['xEnd'], errors='raise')
            dfSacc['yEnd'] = pd.to_numeric(dfSacc['yEnd'], errors='raise')
            dfSacc['duration'] = pd.to_numeric(dfSacc['duration'], errors='raise')
            dfSacc['ampDeg'] = pd.to_numeric(dfSacc['ampDeg'], errors='raise')
            dfSacc['vPeak'] = pd.to_numeric(dfSacc['vPeak'], errors='raise')
            dfSacc['tStart'] = pd.to_numeric(dfSacc['tStart'], errors='raise')
            dfSacc['tEnd'] = pd.to_numeric(dfSacc['tEnd'], errors='raise')
        else:
            dfFix, dfSacc = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path,dfSamples).detect_eye_movements()
            # TODO: 'Eyes_recorded','Rate_recorded','Calib_index' are not being added to the dfFix and dfSacc DataFrames in the remodnav case. Fix this. They can be copied from the samples DataFrame.

        if force_best_eye:
            calib_indexes = dfCalib['Calib_index'].unique()
            best_eyes = dfCalib.groupby('Calib_index').apply(lambda x: find_besteye(x)).values
            indexes = range(len(best_eyes))
            dfslist = list(map(lambda x: keep_eye(best_eyes[x-1],dfSamples[dfSamples['Calib_index'] == x],dfFix[dfFix['Calib_index'] == x],dfBlink[dfBlink['Calib_index'] == x],dfSacc[dfSacc['Calib_index'] == x]),calib_indexes))
            dfSamples,dfFix,dfBlink,dfSacc = map(lambda x: pd.concat([dfslist[i][x] for i in indexes]),[0,1,2,3])

        dict_events = {'fix': dfFix, 'sacc': dfSacc}
        if not keep_ascii:
            os.remove(ascii_file_path)
        if 'screen_height' not in kwargs or 'screen_width' not in kwargs:
            screen_size = dfHeader["line"].iloc[-1].split()
            kwargs['screen_width'] = int(screen_size[-2])
            kwargs['screen_height'] = int(screen_size[-1])

        visualization = Visualization(session_folder_path,detection_algorithm,dfFix,dfSacc,dfSamples)
        pre_processing = PreProcessing(dfSamples, dfFix,dfSacc,dfBlink, dfMsg)
        pre_processing.process({'bad_samples': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.bad_samples).parameters.keys()},
                                'split_all_into_trials': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.split_all_into_trials).parameters.keys()},
                                'saccades_direction': {},})

        unique_trials = dfFix[dfFix['trial_number'] != -1]['trial_number'].unique()
        bad_samples = dfSamples[(dfSamples['trial_number'] != -1) & (dfSamples['bad'] == True)][['trial_number','bad']].groupby('trial_number').size()

        subject = os.path.basename(os.path.dirname(session_folder_path))
        session = os.path.basename(session_folder_path)
        derivatives_folder_path = os.path.dirname(os.path.dirname(session_folder_path))
        log_file = os.path.join(derivatives_folder_path,'derivatives_processing.log') 
        logging.basicConfig(filename=log_file,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        if not bad_samples.empty:
            for trial in bad_samples.index:
                logging.info(f"Subject {subject} in session {session} has {bad_samples[trial]} bad samples in trial {trial}.")
        # Save the 5 data structures in HDF5 file each, in the derivatives folder
        dfHeader.to_hdf(os.path.join(session_folder_path, 'header.hdf5'), key='header', mode='w')
        dfMsg.to_hdf(os.path.join(session_folder_path, 'msg.hdf5'), key='msg', mode='w')
        dfCalib.to_hdf(os.path.join(session_folder_path, 'calib.hdf5'), key='calib', mode='w')
        dfSamples.to_hdf(os.path.join(session_folder_path, 'samples.hdf5'), key='samples', mode='w')
        dfBlink.to_hdf(os.path.join(session_folder_path,'eyelink_events', 'blink.hdf5'), key='blink', mode='w')
        for key, value in dict_events.items():
            value.to_hdf(os.path.join(session_folder_path,detection_algorithm+'_events', key + '.hdf5'), key=key, mode='w')

    for trial in unique_trials:
        visualization.scanpath(trial_index=trial,**{arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(visualization.scanpath).parameters.keys()})

    return dfFix,dfSacc

def process_session(bids_dataset_folder, subject, session, msg_keywords,detection_algorithm,derivatives_folder,force_best_eye,keep_ascii,**kwargs):
    eye_tracking_data_path = os.path.join(bids_dataset_folder, subject, session, 'ET')
    edf_files = [file for file in os.listdir(eye_tracking_data_path) if file.lower().endswith(".edf")]
    if len(edf_files) > 1:
        logging.warning(f"More than one EDF file found in {eye_tracking_data_path}. Skipping folder.")
        return


    edf_file_path = os.path.join(eye_tracking_data_path, edf_files[0])
    session_folder_path = os.path.join(derivatives_folder, subject, session)
    os.makedirs(os.path.join(session_folder_path, 'eyelink_events'), exist_ok=True)

    return parse_edf_eyelink(edf_file_path, msg_keywords,detection_algorithm,session_folder_path,force_best_eye,keep_ascii,**kwargs)

def compute_derivatives_for_dataset(bids_dataset_folder, msg_keywords,detection_algorithm='eyelink',num_processes=None,force_best_eye=True,keep_ascii=True,**kwargs):
    '''
    Generate the derivatives for a BIDS dataset.

    Args:
        bids_dataset_folder (str): Path to the BIDS dataset folder.
        msg_keywords (list of str): List of strings representing keywords to filter MSG lines. For example: 'trial_start', 'trial_end'.
    '''
    derivatives_folder = bids_dataset_folder + "_derivatives"
    os.makedirs(derivatives_folder, exist_ok=True)

    bids_folders = [folder for folder in os.listdir(bids_dataset_folder) if folder.startswith("sub-")]
    if not (detection_algorithm in EYE_MOVEMENT_DETECTION_DICT) and detection_algorithm != 'eyelink':
        raise ValueError(f"Detection algorithm {detection_algorithm} not found.")
    fixations = []
    saccades = []
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for subject in bids_folders:
            sessions_folders = [folder for folder in os.listdir(os.path.join(bids_dataset_folder, subject)) if folder.startswith("ses-")]
            for session in sessions_folders:
                futures.append(executor.submit(process_session, bids_dataset_folder, subject, session, msg_keywords,detection_algorithm,derivatives_folder,force_best_eye,keep_ascii,**kwargs))
        for futures in futures:
            result = futures.result()
            fixations.append(result[0])
            saccades.append(result[1])
    fixations = pd.concat(fixations)
    saccades = pd.concat(saccades)
    visualization = Visualization(derivatives_folder, detection_algorithm, fixations, saccades,None)
    visualization.plot_multipanel()
    return derivatives_folder
        