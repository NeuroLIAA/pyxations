import os
import pandas as pd
from . import Visualization, PostProcessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import logging

detection_algorithm_folder = "eyelink_events"
detection_algorithm = "eyelink"

def process_session(session_folder_path, start_msgs, end_msgs):
    
    samples = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, "samples.hdf5"))
    sacc_filename = "sacc.hdf5"
    fix_filename = "fix.hdf5"
    saccades = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, detection_algorithm_folder, sacc_filename))
    fixations = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, detection_algorithm_folder, fix_filename))
    user_messages = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, "msg.hdf5"))
    visualization = Visualization(session_folder_path,detection_algorithm)
    post_processing = PostProcessing(session_folder_path, detection_algorithm)
    header_filename = "header.hdf5"
    header = pd.read_hdf(path_or_buf=os.path.join(session_folder_path, header_filename))
    # Screen size is in the last row of the header, in the "value" column, it is a string. I need to split it by whitespaces and take the last two elements, which are the width and height of the screen.
    screen_size = header["value"].iloc[-1].split()
    screen_width = int(screen_size[-2])
    screen_height = int(screen_size[-1])

    samples = post_processing.bad_samples(samples,screen_height,screen_width)
    saccades = post_processing.saccades_direction(saccades)
    saccades = post_processing.split_into_trials(saccades, sacc_filename, user_messages=user_messages, start_msgs=start_msgs, end_msgs=end_msgs)
    fixations = post_processing.split_into_trials(fixations, fix_filename, user_messages=user_messages, start_msgs=start_msgs, end_msgs=end_msgs)
    samples = post_processing.split_into_trials(samples, "samples.hdf5", user_messages=user_messages, start_msgs=start_msgs, end_msgs=end_msgs)
    unique_trials = fixations[fixations['trial_number'] != -1]['trial_number'].unique()
    bad_samples = samples[(samples['trial_number'] != -1) & (samples['bad'] == True)][['trial_number','bad']].groupby('trial_number').size()

    subject = os.path.basename(os.path.dirname(session_folder_path))
    session = os.path.basename(session_folder_path)
    derivatives_folder_path = os.path.dirname(os.path.dirname(session_folder_path))
    log_file = os.path.join(derivatives_folder_path,'derivatives_processing.log') 
    logging.basicConfig(filename=log_file,level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if not bad_samples.empty:
        for trial in bad_samples.index:
            logging.info(f"Subject {subject} in session {session} has {bad_samples[trial]} bad samples in trial {trial}.")
    

    visualization.plot_multipanel(fixations, saccades)
    for trial in unique_trials:
        visualization.scanpath(fixations=fixations, trial_index=trial, saccades=saccades, samples=samples)
    return samples,fixations,saccades

def process_derivatives(derivatives_folder_path: str, start_msgs: list[str], end_msgs: list[str],max_workers:int=None):
    '''
    Process all the sessions for all the subjects in the derivatives folder.
    This will add the direction of the saccades and split the data into trials (for fixations, saccades and samples).
    It will also plot a scanpath for each trial and the multipanel.

    Parameters:
    derivatives_folder_path (str): Path to the derivatives folder.
    start_msgs (list[str]): List of strings to identify the start of the stimuli.
    end_msgs (list[str]): List of strings to identify the end of the stimuli.
    '''

    subjects = [subject for subject in os.listdir(derivatives_folder_path) if os.path.isdir(os.path.join(derivatives_folder_path, subject)) and subject.startswith("sub-")]
    fixations = []
    saccades = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for subject in subjects:
            sessions = [session for session in os.listdir(os.path.join(derivatives_folder_path, subject)) if os.path.isdir(os.path.join(derivatives_folder_path, subject, session)) and session.startswith("ses-")]
            for session in sessions:
                session_folder_path = os.path.join(derivatives_folder_path, subject, session)
                futures.append(executor.submit(process_session, session_folder_path, start_msgs, end_msgs))
        for future in futures:
            fixations.append(future.result()[1])  # This will raise exceptions if any occurred during processing
            saccades.append(future.result()[2])
    fixations = pd.concat(fixations)
    saccades = pd.concat(saccades)
    visualization = Visualization(derivatives_folder_path, detection_algorithm)
    visualization.plot_multipanel(fixations, saccades)



def get_ordered_trials_from_psycopy_logs(dataset_folder_path:str):
    #TODO: Implement this function
    dict_trial_labels = defaultdict(lambda: defaultdict(list))
    subjects = [subject for subject in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, subject)) and subject.startswith("sub-")]
    for subject in subjects:
        sessions = [session for session in os.listdir(os.path.join(dataset_folder_path, subject)) if os.path.isdir(os.path.join(dataset_folder_path, subject, session)) and session.startswith("ses-")]
        for session in sessions:
            dict_trial_labels[subject][session] = []