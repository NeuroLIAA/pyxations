import os
import pandas as pd
from . import Visualization, PostProcessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

detection_algorithm_folder = "eyelink_events"

def process_session(session_folder_path, start_msgs, end_msgs):
    samples = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, "samples.hdf5"))
    sacc_filename = "sacc.hdf5"
    fix_filename = "fix.hdf5"
    saccades = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, detection_algorithm_folder, sacc_filename))
    fixations = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, detection_algorithm_folder, fix_filename))
    user_messages = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, "msg.hdf5"))
    visualization = Visualization(session_folder_path, "eyelink")
    post_processing = PostProcessing(session_folder_path, "eyelink")

    saccades = post_processing.saccades_direction(saccades)
    saccades = post_processing.split_into_trials(saccades, sacc_filename, user_messages=user_messages, start_msgs=start_msgs, end_msgs=end_msgs)
    fixations = post_processing.split_into_trials(fixations, fix_filename, user_messages=user_messages, start_msgs=start_msgs, end_msgs=end_msgs)
    samples = post_processing.split_into_trials(samples, "samples.hdf5", user_messages=user_messages, start_msgs=start_msgs, end_msgs=end_msgs)
    unique_trials = fixations[fixations['trial_number'] != -1]['trial_number'].unique()

    visualization.plot_multipanel(fixations, saccades)
    for trial in unique_trials:
        visualization.scanpath(fixations=fixations, trial_index=trial, saccades=saccades, samples=samples)

def process_derivatives(derivatives_folder_path: str, start_msgs: list[str], end_msgs: list[str]):
    '''
    Process all the sessions for all the subjects in the derivatives folder.
    This will add the direction of the saccades and split the data into trials (for fixations, saccades and samples).
    It will also plot a scanpath for each trial and the multipanel.

    Parameters:
    derivatives_folder_path (str): Path to the derivatives folder.
    start_msgs (list[str]): List of strings to identify the start of the stimuli.
    end_msgs (list[str]): List of strings to identify the end of the stimuli.
    '''

    subjects = [subject for subject in os.listdir(derivatives_folder_path) if os.path.isdir(os.path.join(derivatives_folder_path, subject))]

    with ProcessPoolExecutor() as executor:
        futures = []
        for subject in subjects:
            sessions = [session for session in os.listdir(os.path.join(derivatives_folder_path, subject)) if os.path.isdir(os.path.join(derivatives_folder_path, subject, session))]
            for session in sessions:
                session_folder_path = os.path.join(derivatives_folder_path, subject, session)
                futures.append(executor.submit(process_session, session_folder_path, start_msgs, end_msgs))
        for future in futures:
            future.result()  # This will raise exceptions if any occurred during processing



def get_ordered_trials_from_psycopy_logs(dataset_folder_path:str):
    #TODO: Implement this function
    dict_trial_labels = defaultdict(lambda: defaultdict(list))
    subjects = [subject for subject in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, subject))]
    for subject in subjects:
        sessions = [session for session in os.listdir(os.path.join(dataset_folder_path, subject)) if os.path.isdir(os.path.join(dataset_folder_path, subject, session))]
        for session in sessions:
            dict_trial_labels[subject][session] = []