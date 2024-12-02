'''
Created on Nov 7, 2024

@author: placiana
'''
import pandas as pd
from pyxations.formats.generic import BidsParse
from pyxations.export import HDF5_EXPORT

def process_session(eye_tracking_data_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.csv']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)

    exp_format = HDF5_EXPORT
    if 'export_format' in kwargs:
        exp_format = kwargs.get('export_format')
    GazePointParse(exp_format).parse(edf_file_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs)

class GazePointParse(BidsParse):

    def parse(self, file_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):
        # Convert EDF to ASCII (only if necessary)
        # ascii_file_path = convert_edf_to_ascii(edf_file_path, session_folder_path)
        detection_algorithm = 'remodnav'
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
        
        #dfSample.to_hdf((session_folder_path / 'samples.hdf5'), key='samples', mode='w')
        #dfBlink.to_hdf((session_folder_path / 'blink.hdf5'), key='samples', mode='w')
        (session_folder_path / f'{detection_algorithm}_events').mkdir(parents=True, exist_ok=True)
        self.save_dataframe(dfSample, session_folder_path, 'samples', key='samples')
        self.save_dataframe(dfBlink, (session_folder_path / f'{detection_algorithm}_events'), 'blink', key='blink')

        #(session_folder_path / f'{detection_algorithm}_events').mkdir(parents=True, exist_ok=True)
        #self.save_dataframe(dfFix, (session_folder_path / f'{detection_algorithm}_events'), 'fix', key='fix')
        #self.save_dataframe(dfSacc, (session_folder_path / f'{detection_algorithm}_events'), 'sac', key='sacc')
    