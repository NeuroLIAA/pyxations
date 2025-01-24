'''
Created on Nov 7, 2024

@author: placiana
'''
import pandas as pd
from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing
import inspect


def process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, overwrite, exp_format, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.txt']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)

    TobiiParse(session_folder_path, exp_format).parse(
        edf_file_path, detection_algorithm, overwrite, **kwargs)


class TobiiParse(BidsParse):

    def parse(self, file_path, detection_algorithm, overwrite, **kwargs):
        from pyxations.bids_formatting import find_besteye, EYE_MOVEMENT_DETECTION_DICT, keep_eye
        
        # Convert EDF to ASCII (only if necessary)
        # ascii_file_path = convert_edf_to_ascii(edf_file_path, session_folder_path)
        df = pd.read_csv(file_path, sep="\t")
        
        dfSample = df[df['Eyepos3d_Left.x'] > 0].reset_index().rename(columns={"index": "line_number"})
        
        # Reading ASCII in chunks to reduce memory usage
        with open(file_path, 'r') as f:
            lines = (line.strip() for line in f)  # Generator to save memory
            line_data = []
            
            for line in lines:
                linesplit = line.split('\t')
                if len(linesplit) != 30:
                    print(len(linesplit))
                line_data.append(line.replace('\n', '').replace('\t', ' '))
                
        dfSample = dfSample.rename(columns={'Eyetracker timestamp': 'tSample'})
    
        
        # Eye movement detect
        eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=self.session_folder_path, samples=dfSample)
        config = {
            'savgol_length': 0.195,
            'eyes_recorded': 'L',
            'eye': 'L',
            'pupil_data': dfSample['PupilDiam_Left'],
            'max_pso_dur': 0.3
        }
        
        dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(
            dfSample, 60,
            x_label='Gaze3d_Left.x', y_label='Gaze3d_Left.y', config=config, )
        

        # Split into trials
        #placeholder
        dfMsg = dfBlink = pd.DataFrame(columns=dfSample.columns)

        pre_processing = PreProcessing(dfSample, dfFix,dfSacc,dfBlink, dfMsg, self.session_folder_path)
        preprocessing_parameters = inspect.signature(pre_processing.split_all_into_trials).parameters.keys()
        if all([arg in kwargs for arg in preprocessing_parameters]):
            pre_processing.process({
                #'bad_samples': {arg:kwargs[arg] for arg in kwargs if arg in inspect.signature(pre_processing.bad_samples).parameters.keys()},
                'split_all_into_trials': {arg:kwargs[arg] for arg in kwargs if arg in preprocessing_parameters},
                #'saccades_direction': {},
            })
        else:
            print('Skipping preprocessing: not enough parameters.')
        
        

        
        self.detection_algorithm = detection_algorithm
        self.store_dataframes(dfSample, dfFix=dfFix, dfSacc=dfSacc)

        return df
