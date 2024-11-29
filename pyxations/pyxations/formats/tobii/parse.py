'''
Created on Nov 7, 2024

@author: placiana
'''
import pandas as pd

def process_session(eye_tracking_data_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.txt']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)

    parse_tobii(edf_file_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs)


def parse_tobii(file_path, msg_keywords, session_folder_path, force_best_eye, keep_ascii, overwrite, **kwargs):
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

    detection_algorithm = 'remodnav'
    eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=session_folder_path,samples=dfSample)
    config = {
        'savgol_length': 0.195,
        'eyes_recorded': 'L',
        'eye': 'L',
        'pupil_data': dfSample['PupilDiam_Left']
    }
    
    dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(
        dfSample,  60,
        x_label='Gaze3d_Left.x', y_label='Gaze3d_Left.y', config=config)
    
    
    (session_folder_path / f'{detection_algorithm}_events').mkdir(parents=True, exist_ok=True)

    dfSample.to_hdf((session_folder_path / 'samples.hdf5'), key='samples', mode='w')
    dfFix.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'fix.hdf5'), key='fix', mode='w')
    dfSacc.to_hdf((session_folder_path / f'{detection_algorithm}_events' / 'sacc.hdf5'), key='sacc', mode='w')



    return df