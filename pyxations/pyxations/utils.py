import os
from collections import defaultdict

def sessions_without_samples(derivatives_folder_path:str):
    for subject in [subject for subject in os.listdir(derivatives_folder_path) if os.path.isdir(os.path.join(derivatives_folder_path, subject)) and subject.startswith("sub-")]:
        for session in [session for session in os.listdir(os.path.join(derivatives_folder_path, subject)) if os.path.isdir(os.path.join(derivatives_folder_path, subject, session)) and session.startswith("ses-")]:                
            if not os.path.exists(os.path.join(derivatives_folder_path,subject,session,"samples.hdf5")):
                print(os.path.join(derivatives_folder_path,subject,session,"samples.hdf5"))

def parse_psycopy_log_for_trial_names(log_file_path:str,trial_beginning_delimiter:str,trial_end_delimiter:str):
    with open(log_file_path, "r") as log_file:
        log_lines = log_file.readlines()
    trial_names = []
    for line in log_lines:
        if trial_beginning_delimiter in line and trial_end_delimiter in line:
            trial_name = line.split(trial_beginning_delimiter)[1].split(trial_end_delimiter)[0]
            trial_names.append(trial_name)
    return trial_names

def get_ordered_trials_from_psycopy_logs(dataset_folder_path:str,trial_beginning_delimiter:str,trial_end_delimiter:str):
    dict_trial_labels = defaultdict()
    subjects = [subject for subject in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, subject)) and subject.startswith("sub-")]
    for subject in subjects:
        dict_trial_labels[subject] = defaultdict(list)
        sessions = [session for session in os.listdir(os.path.join(dataset_folder_path, subject)) if os.path.isdir(os.path.join(dataset_folder_path, subject, session)) and session.startswith("ses-")]
        for session in sessions:
            log_files = [log_file for log_file in os.listdir(os.path.join(dataset_folder_path, subject, session,"behavioral")) if log_file.endswith(".log")]
            if len(log_files) > 1:
                raise ValueError(f"More than one log file found in {os.path.join(dataset_folder_path, subject, session,'behavioral')}")
            log_file = log_files[0]
            dict_trial_labels[subject][session] = parse_psycopy_log_for_trial_names(os.path.join(dataset_folder_path, subject, session,"behavioral",log_file),trial_beginning_delimiter,trial_end_delimiter)
    return dict_trial_labels