'''
Created on Nov 7, 2024

@author: placiana
'''
import pandas as pd
from pyxations.formats.generic import BidsParse


def process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, exp_format, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.csv']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)


    GazePointParse(session_folder_path, exp_format).parse(edf_file_path, detection_algorithm, overwrite, **kwargs)

class GazePointParse(BidsParse):

    def parse(self, file_path, detection_algorithm, overwrite, **kwargs):
        from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT
        
        df = pd.read_csv(file_path)
        
        dfSample = df.reset_index().rename(columns={
            "index": "line_number", 
            "TIME": "tSample",
            "BPOGX": "X",
            "BPOGY": "Y",
            "LPD": "Pupil"
        })
        
        dfBlink = df[df['BKDUR'] > 0].reset_index().rename(columns={
            "index": "line_number", 
            "TIME": "tEnd",
            "BKDUR": "duration"
        })
        dfBlink['tStart'] = dfBlink['tEnd'] - dfBlink['duration']

        eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=self.session_folder_path,samples=dfSample)
        config = {
            'savgol_length': 0.19,
        }
        self.detection_algorithm = detection_algorithm
        dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(dfSample, 60, config=config)

        
        
        self.store_dataframes(dfSample, dfBlink=dfBlink, dfFix=dfFix, dfSacc=dfSacc)

    