from pathlib import Path
from collections import defaultdict

def parse_psycopy_log_for_trial_names(log_file_path:Path,trial_beginning_delimiter:str,trial_end_delimiter:str):
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
    dataset_folder_path = Path(dataset_folder_path)
    subjects = [subject for subject in dataset_folder_path.iterdir() if subject.is_dir() and subject.name.startswith("sub-")]
    for subject in subjects:
        dict_trial_labels[subject.name] = defaultdict(list)
        sessions = [session for session in subject.iterdir() if session.is_dir() and session.name.startswith("ses-")]
        for session in sessions:
            log_files = [log_file for log_file in (session / "behavioral").iterdir() if log_file.name.endswith(".log")]
            if len(log_files) > 1:
                raise ValueError(f"More than one log file found in {(session / 'behavioral')}")
            log_file = log_files[0]
            dict_trial_labels[subject.name][session.name] = parse_psycopy_log_for_trial_names(log_file,trial_beginning_delimiter,trial_end_delimiter)
    return dict_trial_labels