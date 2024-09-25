from pathlib import Path
import pandas as pd
from .visualization import Visualization

class Session:
    """
    Initialize a Session instance.

    Args:
        dataset_path (str): The path to the dataset directory.
        subject_id (str): The subject's ID.
        session_id (str): The session ID.

    Raises:
        FileNotFoundError: If the dataset path or session path does not exist.
    """
    
    def __init__(self, dataset_path: str, subject_id: str, session_id: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.subject_id = subject_id
        self.session_id = session_id
        
        # Check if the dataset path and session folder exist
        base_path = self.dataset_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}"
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Session path not found: {base_path}")
        
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        events_path = self.dataset_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}" / f"{self.detection_algorithm}_events"
        
        # Check if paths and files exist
        if not events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {events_path}")

        # Define file paths
        samples_path = self.dataset_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}" / "samples.hdf5"
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
        if not blink_path.exists():
            raise FileNotFoundError(f"Blinks file not found: {blink_path}")

        # Load the data
        self.samples = pd.read_hdf(samples_path)
        self.fix = pd.read_hdf(fix_path)
        self.sacc = pd.read_hdf(sacc_path)
        self.blink = pd.read_hdf(blink_path)

    def plot_scanpath(self, **kwargs) -> None:
        events_path = self.dataset_path / f"sub-{self.subject_id}" / f"ses-{self.session_id}" / f"{self.detection_algorithm}_events"
        if not events_path.exists():
            raise FileNotFoundError(f"Algorithm events path not found: {events_path}")

        vis = Visualization(events_path, self.detection_algorithm)
        vis.scanpath(fixations=self.fix, saccades=self.sacc, samples=self.samples, screen_height=1080, screen_width=1920, **kwargs)
