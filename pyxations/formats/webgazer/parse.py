'''
Created on Oct 31, 2024

@author: placiana
'''
import pandas as pd
import json
from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing
import inspect


def process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, overwrite, exp_format, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.csv']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)
    
    WebGazerParse(session_folder_path, exp_format).parse(edf_file_path, detection_algorithm,
                         overwrite, **kwargs)


class WebGazerParse(BidsParse):

    def parse(self, file_path, detection_algorithm, overwrite, **kwargs):
        # Convert EDF to ASCII (only if necessary)
        # ascii_file_path = convert_edf_to_ascii(edf_file_path, session_folder_path)
        from pyxations.bids_formatting import find_besteye, EYE_MOVEMENT_DETECTION_DICT, keep_eye
        df = pd.read_csv(file_path)
        
        session_folder_path = self.session_folder_path
        
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
            'max_pso_dur': 0.1
        }
        
        dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(dfSamples, 30, config=config)

        dfBlink = pd.DataFrame(columns=dfSamples.columns)
        dfMsg = pd.DataFrame(columns=dfSamples.columns)


        pre_processing = PreProcessing(dfSamples, dfFix,dfSacc,dfBlink, dfMsg, session_folder_path)
        preprocessing_parameters = inspect.signature(pre_processing.split_all_into_trials).parameters.keys()
        if all([arg in kwargs for arg in preprocessing_parameters]):
            pre_processing.process({
                #'bad_samples': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.bad_samples).parameters.keys()},
                'split_all_into_trials': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.split_all_into_trials).parameters.keys()},
                #'saccades_direction': {},
            })
        else:
            print('Skipping preprocessing: not enough parameters.')


    

        self.detection_algorithm = detection_algorithm
        #self.store_dataframes(dfSamples, dfCalib, dfFix, dfSacc, dfBlink, dfMsg)
        pp = pre_processing
        self.store_dataframes(pp.samples, dfCalib, pp.fixations, pp.saccades, pp.blinks, pp.user_messages)
            

        # Save DataFrames to disk in one go to minimize memory usage during processing
        #self.save_dataframe(dfCalib, session_folder_path, 'calib', key='calib')
        #self.save_dataframe(dfSamples, session_folder_path, 'samples', key='samples')
        
        #(session_folder_path / f'{detection_algorithm}_events').mkdir(parents=True, exist_ok=True)
        #self.save_dataframe(dfFix, (session_folder_path / f'{detection_algorithm}_events'), 'fix', key='fix')
        #self.save_dataframe(dfSacc, (session_folder_path / f'{detection_algorithm}_events'), 'sacc', key='sacc')
    

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
        
        