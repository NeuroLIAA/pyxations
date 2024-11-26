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
    detection_algorithm = 'webgazer'
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
    
    dfCalib.to_hdf((session_folder_path / 'calib.hdf5'), key='calib', mode='w')
    dfSamples.to_hdf((session_folder_path / 'samples.hdf5'), key='samples', mode='w')
    
    #dfSacc =  df[df['isSaccadeExperiment'] == True]
    #(session_folder_path / f'{detection_algorithm}_events').mkdir(parents=True, exist_ok=True)
    #dfSacc.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'sacc.hdf5'), key='sacc', mode='w')
    
    old = True
    
    #output
    if not old:
        dfHeader.to_hdf((session_folder_path / 'header.hdf5'), key='header', mode='w')
        dfMsg.to_hdf((session_folder_path / 'msg.hdf5'), key='msg', mode='w')
    
        dfBlink.to_hdf((session_folder_path / 'eyelink_events' / 'blink.hdf5'), key='blink', mode='w')
        dfFix.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'fix.hdf5'), key='fix', mode='w')
    


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
                if len(line) < 2:
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
                elif not start_flag:  # Data before the first successful calibration is discarded. 
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
        del df, line_data, line_types, eyes_recorded, rates_recorded, calib_indexes  # Free up memory
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
        dfMsg[['timestamp', 'message']] = dfMsg['line'].str.replace('MSG ', '').str.split(n=1, expand=True).values
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
    
        dfSamples = dfSamples[['tSample'] + [col for col in ['LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil'] if col in dfSamples.columns] + ['Line_number', 'Eyes_recorded', 'Rate_recorded', 'Calib_index']]
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
            eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=session_folder_path, samples=dfSamples)
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
        
        pre_processing = PreProcessing(dfSamples, dfFix, dfSacc, dfBlink, dfMsg)
        pre_processing.process({'bad_samples': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.bad_samples).parameters.keys()},
                                'split_all_into_trials_by_msgs': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.split_all_into_trials_by_msgs).parameters.keys()},
                                'saccades_direction': {}, })
    
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

