
from .visualization import Visualization
import os
import pandas as pd

class Subject():
    """
    Initialize a Subject instance.

    Args:
        subject_id (str): Unique identifier for the subject.
        subj_path (str): Path to the subject's data directory.
        session (str): Identifier for the session.
        trial_id (str): Identifier for the trial.
        detection_algorithm (str): Path to the detection algorithm folder.
    """       
    def __init__(self, subject_id:str, subj_path:str, session:str, trial_id:str, detection_algorithm:str, fix:pd.DataFrame, sacc:pd.DataFrame, blink:pd.DataFrame) -> None:
        self.id = subject_id
        self.path = subj_path
        self.session = session
        self.trial = trial_id
        self.detection_algorithm = detection_algorithm
        self.fix = fix
        self.sacc = sacc
        self.blink = blink
    
    def plot_scanpath(self, screen_res_x:int=1920, screen_res_y:int=1080) -> None:
        session_folder_path = os.path.join(self.path, self.session)
        vis = Visualization(session_folder_path,self.detection_algorithm)
        vis.scanpath(fixations=self.fix, saccades=self.sacc)


class MultipleSubjects():
    """
    """
    pass     
