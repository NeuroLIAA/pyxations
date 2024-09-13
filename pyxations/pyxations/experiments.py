import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from .subjects import Subject


# Abstract Experiment class
class BaseExperiment(ABC):
    @abstractmethod
    def load_data(self, subject_id, trial_id):
        pass

class Experiment(BaseExperiment):       
    def __init__(self, base_path):
        self.base_path = base_path
        self.subjects = self._get_subjects_ids()
        self.paths = self._get_subjects_path()
        self.n = len(self.subjects)
        self.hdf5_filenames = ["blink.hdf5", "fix.hdf5", "sacc.hdf5", "calib.hdf5", "header.hdf5", "msg.hdf5", "samples.hdf5"] 
    
    def _get_subjects_ids(self):
        """Get list of subjects based on BIDS folder structure."""
        return [d for d in os.listdir(self.base_path) if d.startswith("sub-")]

    def _get_subjects_path(self):
        """Get list of subjects based on BIDS folder structure."""
        return [os.path.join(self.base_path, d) for d in os.listdir(self.base_path) if d.startswith("sub-")]
    
    def load_hdf5_file(self, file_path: str) -> pd.DataFrame:
        """
        Load an HDF5 file and add the subject name to the DataFrame.

        Parameters:
        file_path (str): The file path of the HDF5 file to load.

        Returns:
        pd.DataFrame: The loaded DataFrame with an added 'subj_name' column.
        """
        subj_id = re.findall(r"sub-([a-zA-Z0-9]{4,6})", file_path)[0]
        df = pd.read_hdf(file_path)
        df["subj_id"] = subj_id
        return df

    def load_data(self, subject_id, session=None, trial_id=None, detection_algorithm_folder=None):
        """Abstract method to load data, must be implemented by subclasses."""
        
        if not session:
            raise(f"session for subject {subject_id} not found")
        
        if not trial_id:
            raise((f"trial for subject {subject_id} not found"))

        subject_id = f'sub-{subject_id}'
        subject_path_ids = dict(zip(self.subjects, self.paths))
        subj_path = subject_path_ids[subject_id]
        data = {}

        path = os.path.join(subj_path, f"ses-{session}", detection_algorithm_folder)
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".hdf5"):
                    hdf5_data = self.load_hdf5_file(os.path.join(dirpath, filename))
                    dataframe_name = filename.split(".")[0]
                    print("dataframe_name:", dataframe_name)
                    if int(trial_id) in list(hdf5_data['trial_number'].unique()):
                        hdf5_data = hdf5_data.query(f"trial_number == {trial_id}")
                        data[dataframe_name] = hdf5_data 
                    else:   
                        raise ValueError(f"Warning: trial {trial_id} is not present")

        
        print("data:", data)
        return Subject(subject_id, subj_path, session, trial_id, detection_algorithm_folder, data['fix'], data['sacc'], None)

    def get_subject_data(self, subject_id, session, trial_id, detection_algorithm_folder):
        """Load specific subject and trial data, handling missing files internally."""
        data = self.load_data(subject_id, session, trial_id, detection_algorithm_folder)
        return data
 
    def plot_scanpath(self):
        pass

# Concrete ReadingExperiment class
class ReadingExperiment(Experiment):
    def load_data(self, subject_id, trial_id):
        pass

    def analyze_reading_behavior(self, subject_id, trial_id):
        """Analyze specific reading-related metrics for a subject and trial."""
        pass