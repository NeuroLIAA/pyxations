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
            
            
    
    dfSample.to_hdf((session_folder_path / 'samples.hdf5'), key='samples', mode='w')
    return df