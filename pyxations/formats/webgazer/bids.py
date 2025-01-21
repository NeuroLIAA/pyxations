'''
Created on Nov 4, 2024

@author: placiana
'''
from pathlib import Path



class WebGazerBidsConverter():
    
    def relevant_extensions(self):
        return ['.log', '.csv']

    def get_subject_ids(self, file_paths):
        return list(set([Path(file).name.split("_")[0] for file in file_paths if file.suffix.lower() == '.csv' ]))

    def move_file_to_bids_folder(self, file, bids_folder_path, subject_id, old_subject_id, session_id):
        from pyxations.bids_formatting import move_file_to_bids_folder
        file_name = Path(file).name
        file_lower = file_name.lower()
        if file_lower.endswith(".csv") and file_name.split("_")[0] == old_subject_id:
            move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'ET')
        if (file_lower.endswith(".log") or file_lower.endswith(".csv")) and file_name.split("_")[0] == old_subject_id:                
            move_file_to_bids_folder(file, bids_folder_path, subject_id, session_id, 'behavioral')        


