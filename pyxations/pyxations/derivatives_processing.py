import os
import pandas as pd
from . import Visualization, PostProcessing
from collections import defaultdict

detection_algorithm_folder = "eyelink_events"

def process_derivatives(derivatives_folder_path:str,start_msgs: list[str],end_msgs: list[str]):
    subjects = os.listdir(derivatives_folder_path)
    for subject in subjects:
        sessions = os.listdir(os.path.join(derivatives_folder_path,subject))
        for session in sessions:
            session_folder_path = os.path.join(derivatives_folder_path,subject,session)
            samples = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, "samples.hdf5")) 
            # Here you could perform eye movement detection and then load the resulting saccades and fixations
            sacc_filename = "sacc.hdf5"
            fix_filename = "fix.hdf5"
            saccades = pd.read_hdf(path_or_buf=os.path.join(session_folder_path,detection_algorithm_folder, sacc_filename))
            fixations = pd.read_hdf(path_or_buf=os.path.join(session_folder_path,detection_algorithm_folder, fix_filename))  
            user_messages = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, "msg.hdf5"))                 
            visualization = Visualization(session_folder_path,"eyelink")
            post_processing = PostProcessing(session_folder_path,"eyelink")
            saccades = post_processing.saccades_direction(saccades, sacc_filename)
            saccades = post_processing.split_into_trials(saccades,sacc_filename,user_messages=user_messages,start_msgs=start_msgs,end_msgs=end_msgs)
            visualization.plot_multipanel(fixations, saccades)
            visualization.scanpath(fixations=fixations,tmin=samples['tSample'][100000], tmax=samples['tSample'][110000], img_path=None, saccades=saccades, samples=samples)



def get_ordered_trials_from_psycopy_logs(dataset_folder_path:str):
    #TODO: Implement this function
    dict_trial_labels = defaultdict(lambda: defaultdict(list))
    subjects = os.listdir(dataset_folder_path)
    for subject in subjects:
        sessions = os.listdir(os.path.join(dataset_folder_path,subject))
        for session in sessions:
            dict_trial_labels[subject][session] = []