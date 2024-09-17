
from .visualization import Visualization
import os
import pandas as pd

class Session():
    """
    Initialize a Session instance.

    Args:
     
    """       
    def __init__(self,dataset_path:str,  subject_id:str, session_id:str) -> None:
        self.dataset_path = dataset_path
        self.subject_id = subject_id 
        self.session_id = session_id

    def load_data(self, detection_algorithm):
        self.detection_algorithm = detection_algorithm
        path = os.path.join(self.dataset_path, f"sub-{self.subject_id}", f"ses-{self.session_id}", f"{self.detection_algorithm}_events")
        self.samples = pd.read_hdf(os.path.join(self.dataset_path, f"sub-{self.subject_id}", f"ses-{self.session_id}", "samples.hdf5"))
        self.fix = pd.read_hdf(os.path.join(path, "fix.hdf5"))
        self.sacc = pd.read_hdf(os.path.join(path, "sacc.hdf5"))
        self.blink = pd.read_hdf(os.path.join(path, "blink.hdf5"))
    
    def plot_scanpath(self, **kwargs) -> None:
        path = os.path.join(self.dataset_path, f"sub-{self.subject_id}", f"ses-{self.session_id}", f"{self.detection_algorithm}_events")
        vis = Visualization(path, self.detection_algorithm)
        vis.scanpath(fixations=self.fix,saccades=self.sacc, samples=self.samples, screen_height=1080, screen_width=1920, **kwargs)

