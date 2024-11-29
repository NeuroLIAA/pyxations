'''
Created on Oct 31, 2024

@author: placiana
'''
import pandas as pd
import json


def process_session(eye_tracking_data_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.csv']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)

    parse_webgazer(edf_file_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs)


def parse_webgazer(file_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):
    # Convert EDF to ASCII (only if necessary)
    # ascii_file_path = convert_edf_to_ascii(edf_file_path, session_folder_path)
    from pyxations.bids_formatting import find_besteye, EYE_MOVEMENT_DETECTION_DICT, keep_eye
    detection_algorithm = 'remodnav'
    df = pd.read_csv(file_path)
    
    df['line_number'] = df.index
    # columna importante 
    dfSamples = df[df['webgazer_data'].notna()].reset_index()
    dfSamples['data'] = dfSamples['webgazer_data'].apply(json.loads)
    df_exploded = dfSamples.explode('data')
    
    df_exploded['data'] = df_exploded.apply(
        lambda row: {**row['data'], 't_acum': row['data']['t'] + row['time_elapsed']}, axis=1
    )
    
    expanded_df = pd.json_normalize(df_exploded['data'])
    expanded_df = pd.concat(
    [df_exploded[['line_number', 'trial_index', 'time_elapsed']].reset_index(drop=True),  # Keep desired columns
     expanded_df],                    # Expand the data
    axis=1
    )
    
    dfSamples = expanded_df.rename(columns={"x": "X", "y": "Y", 't': 'tSample'})

    # Calibration messages    
    dfCalib = df[df['rastoc-type'] == 'calibration-stimulus']

    # Eye movement
    eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=session_folder_path,samples=dfSamples)
    config = {
        'savgol_length': 0.195,
    }
    
    dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(dfSamples, 60, config=config)


    # Persist
    dfCalib.to_hdf((session_folder_path / 'calib.hdf5'), key='calib', mode='w')
    dfSamples.to_hdf((session_folder_path / 'samples.hdf5'), key='samples', mode='w')
    dfFix.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'fix.hdf5'), key='fix', mode='w')
    dfSacc.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'sacc.hdf5'), key='sacc', mode='w')


def get_samples_for_remodnav(df_samples, rate_recorded=60, r_pupil=1, l_pupil=1):
    df_samples['Rate_recorded'] = rate_recorded
    df_samples['LX'] = df_samples['X'] 
    df_samples['RX'] = df_samples['X']
    df_samples['LY'] = df_samples['Y']
    df_samples['RY'] = df_samples['Y']
    df_samples['LPupil'] = l_pupil
    df_samples['RPupil'] = r_pupil
    df_samples['Calib_index'] = 1
    df_samples['Eyes_recorded'] = 'LR'

    return df_samples
        
        