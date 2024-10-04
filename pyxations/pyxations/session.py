from pathlib import Path
import pandas as pd
from .visualization import Visualization

class Experiment:

    def __init__(self,dataset_path: str, detection_algorithm: str):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = Path(str(self.dataset_path) + "_derivatives")
        self.metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t")
        self.subjects = self.create_subjects()

    def get_derivatives_path(self):
        return self.derivatives_path
    
    def get_dataset_path(self):
        return self.dataset_path

    def create_subjects(self):
        subjects = []
        for subject_id, old_subject_id in zip(self.metadata["subject_id"], self.metadata["old_subject_id"]):
            subjects.append(Subject(subject_id, old_subject_id, self))
        return subjects

    def __iter__(self):
        return iter(self.subjects)
    
    def __getitem__(self, index):
        return self.subjects[index]
    
    def __len__(self):
        return len(self.subjects)
    
    def __repr__(self):
        return f"Experiment = '{self.dataset_path.name}'"
    
    def __next__(self):
        return next(self.subjects)
    
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        for subject in self.subjects:
            subject.load_data(detection_algorithm)

    def plot_multipanel(self):
        fixations = pd.concat(subject.get_fixations() for subject in self.subjects)
        saccades = pd.concat(subject.get_saccades() for subject in self.subjects)

        vis = Visualization(self.derivatives_path, self.detection_algorithm)
        vis.plot_multipanel(pd.concat(fixations),pd.concat(saccades))


class Subject:

    def __init__(self,subject_id: str,old_subject_id: str, experiment: Experiment):
        self.subject_id = subject_id
        self.old_subject_id = old_subject_id
        self.experiment = experiment
        self.sessions = self.create_sessions()

    def get_subject_path_derivatives(self):
        return self.experiment.get_derivatives_path() / f"sub-{self.subject_id}"
    
    def get_subject_path(self):
        return self.experiment.get_dataset_path() / f"sub-{self.subject_id}"


    def create_sessions(self):
        sessions = []
        for session_id in self.experiment.metadata["session_id"]:
            sessions.append(Session(session_id, self))
        return sessions
    
    def __iter__(self):
        return iter(self.sessions)
    
    def __getitem__(self, index):
        return self.sessions[index]
    
    def __len__(self):
        return len(self.sessions)
    
    def __repr__(self):
        return f"'Subject = {self.subject_id}'," + self.experiment.__repr__()
    
    def __next__(self):
        return next(self.sessions)
    
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        for session in self.sessions:
            session.load_data(detection_algorithm)

    def get_fixations(self):
        return [session.get_fixations() for session in self.sessions]
    
    def get_saccades(self):
        return [session.get_saccades() for session in self.sessions]


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
    
    def __init__(self, session_id: str, subject:Subject) -> None:

        self.subject = subject
        self.session_id = session_id
        self.session_path = subject.get_subject_path_derivatives() / f"ses-{self.session_id}"
        self.behavior_path = subject.get_subject_path() / f"ses-{self.session_id}" / "behavioral"
        self.events_path = self.session_path / f"{self.detection_algorithm}_events"

        if not self.session_path.exists():
            raise FileNotFoundError(f"Session path not found: {self.session_path}")
            
    def __repr__(self):
      return f"'Session = {self.session_id}'," + self.subject.__repr__() 
    
    
    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        """Filter fixation data by duration and bad data exclusion.

        Args:
            Session

        Returns:
            None
        """
        self.fix = self.fix.loc[(self.fix["duration"] > min_fix_dur) & (self.fix["duration"] < max_fix_dur) & (self.fix["bad"] == False) & (self.fix["trial_index"] > -1)].reset_index(drop=True)

    def filter_saccades(self, max_sacc_dur=100):
        """Filter saccades data by duration and bad data exclusion.

        Args:
            Session

        Returns:
            None
        """
        self.sacc = self.sacc.loc[(self.sacc["duration"] < max_sacc_dur) & (self.sacc["bad"] == False) & (self.sacc["trial_index"] > -1)].reset_index(drop=True)

        
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm

        if self.behavior_path.exists() and len(list(self.behavior_path.glob("*.csv"))) == 1:
            behavior_data = pd.read_csv(next(self.behavior_path.glob("*.csv")))
            self.behavior_data = behavior_data
        elif self.behavior_path.exists() and len(list(self.behavior_path.glob("*.csv"))) > 1:
            raise ValueError(f"Multiple behavior files found in {self.behavior_path}. Please ensure only one behavior file is present.")
    
        # Check if paths and files exist
        if not self.events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {self.events_path}")

        # Define file paths
        samples_path = self.session_path  / "samples.hdf5"
        calib_path = self.session_path  / "calib.hdf5"
        header_path = self.session_path  / "header.hdf5"
        msg_path = self.session_path  / "msg.hdf5"
        fix_path = self.events_path / "fix.hdf5"
        sacc_path = self.events_path / "sacc.hdf5"
        blink_path = self.events_path / "blink.hdf5"
        
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

    def get_fixations(self):
        return self.fix
    
    def get_saccades(self):
        return self.sacc

    def plot_scanpath(self,trial,img_path=None, **kwargs) -> None:
        if not self.events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {self.events_path}")

        vis = Visualization(self.events_path, self.detection_algorithm)
        vis.scanpath(fixations=self.fix, saccades=self.sacc, samples=self.samples, screen_height=1080, screen_width=1920 , trial_index=trial,img_path=img_path, **kwargs)
