from pathlib import Path
import pandas as pd
from .visualization import Visualization

class Session:
    """
    Initialize a Session instance.

    Args:
        derivatives_path (str): The path to the dataset directory.
        subject_id (str): The subject's ID.
        session_id (str): The session ID.

    Raises:
        FileNotFoundError: If the dataset path or session path does not exist.
    """
    
    def __init__(self, derivatives_path: str, subject_id: str, session_id: str) -> None:
        self.derivatives_path = Path(derivatives_path)
        self.subject_id = subject_id
        self.session_id = session_id
        self.session_path = self.derivatives_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}"
        self.dataset_path = Path(str(self.derivatives_path).rsplit('_',1)[0])
        self.behavior_path = self.dataset_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}" / "behavioral"
        metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t")
        self.old_subject_id = metadata.loc[metadata["subject_id"] == self.subject_id, "old_subject_id"].values[0]
        
        # Check if the dataset path and session folder exist
        base_path = self.derivatives_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}"
        
        if not self.derivatives_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.derivatives_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Session path not found: {base_path}")
            
    def __repr__(self):
      return f"Session('session_id={self.session_id}', subject_id={self.subject_id}, dataset={self.derivatives_path.name})"
    
    
    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        """Filter fixation data by duration and bad data exclusion.

        Args:
            Session

        Returns:
            None
        """
        self.fix = self.fix.loc[(self.fix["duration"] > min_fix_dur) & (self.fix["duration"] < max_fix_dur) & (self.fix["bad"] == False)].reset_index(drop=True)

    def filter_saccades(self, max_sacc_dur=100):
        """Filter saccades data by duration and bad data exclusion.

        Args:
            Session

        Returns:
            None
        """
        self.sacc = self.sacc.loc[(self.sacc["duration"] < max_sacc_dur) & (self.sacc["bad"] == False)].reset_index(drop=True)

        
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        events_path =  self.session_path / f"{self.detection_algorithm}_events"
        if self.behavior_path.exists() and len(list(self.behavior_path.glob("*.csv"))) == 1:
            behavior_data = pd.read_csv(next(self.behavior_path.glob("*.csv")))
            self.behavior_data = behavior_data
        elif self.behavior_path.exists() and len(list(self.behavior_path.glob("*.csv"))) > 1:
            raise ValueError(f"Multiple behavior files found in {self.behavior_path}. Please ensure only one behavior file is present.")
    
        # Check if paths and files exist
        if not events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {events_path}")

        # Define file paths
        samples_path = self.session_path  / "samples.hdf5"
        calib_path = self.session_path  / "calib.hdf5"
        header_path = self.session_path  / "header.hdf5"
        msg_path = self.session_path  / "msg.hdf5"
        fix_path = events_path / "fix.hdf5"
        sacc_path = events_path / "sacc.hdf5"
        blink_path = events_path / "blink.hdf5"
        
        # Check if specific files exist
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")
        if not fix_path.exists():
            raise FileNotFoundError(f"Fixations file not found: {fix_path}")
        if not sacc_path.exists():
            raise FileNotFoundError(f"Saccades file not found: {sacc_path}")
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        if not header_path.exists():
            raise FileNotFoundError(f"Header file not found: {header_path}")
        if not msg_path.exists():
            raise FileNotFoundError(f"Messsages file not found: {msg_path}")
    

        # Load the data
        self.samples = pd.read_hdf(samples_path)
        self.fix = pd.read_hdf(fix_path)
        self.sacc = pd.read_hdf(sacc_path)
        self.blink = pd.read_hdf(blink_path)
        self.calib = pd.read_hdf(calib_path)
        self.header = pd.read_hdf(header_path)
        self.msg = pd.read_hdf(msg_path)

    def plot_scanpath(self,trial,img_path=None, **kwargs) -> None:
        events_path = self.derivatives_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}" / f"{self.detection_algorithm}_events"
        if not events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {events_path}")

        vis = Visualization(events_path, self.detection_algorithm)
        vis.scanpath(fixations=self.fix, saccades=self.sacc, samples=self.samples, screen_height=1080, screen_width=1920 , trial_index=trial,img_path=img_path, **kwargs)
