from pathlib import Path
import pandas as pd
from .visualization import Visualization
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

class Experiment:

    def __init__(self, dataset_path: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(self.dataset_path.name + "_derivatives")
        self.metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t", 
                                    dtype={"subject_id": str, "old_subject_id": str})
        self.subjects = [
            Subject(subject_id, old_subject_id, self, excluded_sessions.get(subject_id, []), excluded_trials.get(subject_id, {}))
            for subject_id, old_subject_id in zip(self.metadata["subject_id"], self.metadata["old_subject_id"])
            if subject_id not in excluded_subjects and old_subject_id not in excluded_subjects
        ]

    def get_derivatives_path(self):
        return self.derivatives_path
    
    def get_dataset_path(self):
        return self.dataset_path

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
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(subject.load_data, detection_algorithm) for subject in self.subjects]
            for future in as_completed(futures):
                future.result()

    def plot_multipanel(self, display: bool):
        fixations = pd.concat([subject.get_fixations() for subject in self.subjects], ignore_index=True)
        saccades = pd.concat([subject.get_saccades() for subject in self.subjects], ignore_index=True)

        vis = Visualization(self.derivatives_path, self.detection_algorithm)
        vis.plot_multipanel(fixations, saccades, display)

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for subject in self.subjects:
            subject.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for subject in self.subjects:
            subject.filter_saccades(max_sacc_dur)

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        sessions_results = {subject.get_id(): subject.drop_trials_with_nan_threshold(threshold) for subject in self.subjects}
        bad_trials_total = {subject.get_id(): sessions_results[subject.get_id()][2] for subject in self.subjects}
        self.subjects = [subject for subject in self.subjects if sessions_results[subject.get_id()][0]/sessions_results[subject.get_id()][1] <= threshold]
        return bad_trials_total
    
    def plot_scanpaths(self,screen_height,screen_width,display: bool = False):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(subject.plot_scanpaths,screen_height,screen_width,display) for subject in self.subjects]
            for future in as_completed(futures):
                future.result()

    def get_rts(self):
        rts = [subject.get_rts() for subject in self.subjects]
        return pd.concat(rts, ignore_index=True)

    def get_subject(self, subject_id):
        for subject in self.subjects:
            if subject.get_id() == subject_id:
                return subject
        raise ValueError(f"Subject {subject_id} not found")
    
    def get_session(self, subject_id, session_id):
        subject = self.get_subject(subject_id)
        return subject.get_session(session_id)
    
    def get_trial(self, subject_id, session_id, trial_number):
        session = self.get_session(subject_id, session_id)
        return session.get_trial(trial_number)

class Subject:

    def __init__(self, subject_id: str, old_subject_id: str, experiment: Experiment, 
                 excluded_sessions: list = [], excluded_trials: dict = {}):
        self.subject_id = subject_id
        self.old_subject_id = old_subject_id
        self.experiment = experiment
        self._sessions = None  # Lazy load sessions
        self.excluded_sessions = excluded_sessions
        self.excluded_trials = excluded_trials

    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = [
                Session(session_folder.name.split("-")[-1], self, self.excluded_trials.get(session_folder.name.split("-")[-1], {})) 
                for session_folder in self.get_subject_path().glob("ses-*") 
                if session_folder.name.split("-")[-1] not in self.excluded_sessions
            ]
        return self._sessions

    def get_id(self):
        return self.subject_id

    def get_subject_path_derivatives(self):
        return self.experiment.get_derivatives_path() / f"sub-{self.subject_id}"
    
    def get_subject_path(self):
        return self.experiment.get_dataset_path() / f"sub-{self.subject_id}"
 
    def __iter__(self):
        return iter(self.sessions)
    
    def __getitem__(self, index):
        return self.sessions[index]
    
    def __len__(self):
        return len(self.sessions)
    
    def __repr__(self):
        return f"Subject = '{self.subject_id}', " + self.experiment.__repr__()
    
    def __next__(self):
        return next(self.sessions)
    
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        for session in self.sessions:
            session.load_data(detection_algorithm)


    def get_fixations(self):
        return pd.concat([session.get_fixations() for session in self.sessions], ignore_index=True)
    
    def get_saccades(self):
        return pd.concat([session.get_saccades() for session in self.sessions], ignore_index=True)
    
    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for session in self.sessions:
            session.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for session in self.sessions:
            session.filter_saccades(max_sacc_dur)

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        total_sessions = len(self.sessions)
        sessions_results = {session.get_session_id(): session.drop_trials_with_nan_threshold(threshold) for session in self.sessions}
        bad_sessions = [session for session in self.sessions if sessions_results[session.get_session_id()][0]/sessions_results[session.get_session_id()][1] > threshold]
        bad_sessions_count = len(bad_sessions)
        bad_trials_subject = {session.get_session_id(): {"bad_trials": sessions_results[session.get_session_id()][0], "total_trials": sessions_results[session.get_session_id()][1]} for session in self.sessions}

        # If the proportion of bad sessions exceeds the threshold, remove all sessions
        if bad_sessions_count / total_sessions > threshold:
            for session in bad_sessions:
                session.subject = None
            bad_sessions = self.sessions

        # Update sessions with only the valid sessions
        self._sessions = [session for session in self.sessions if session not in bad_sessions]
        


        return bad_sessions_count, total_sessions, bad_trials_subject

    def plot_scanpaths(self,screen_height,screen_width, display: bool = False):
        for session in self.sessions:
            session.plot_scanpaths(screen_height,screen_width,display)

    def get_rts(self):
        rts = [session.get_rts() for session in self.sessions]
        rts = pd.concat(rts, ignore_index=True)
        rts["subject_id"] = self.subject_id
        return rts

    def get_session(self, session_id):
        for session in self.sessions:
            if session.get_session_id() == session_id:
                return session
        raise ValueError(f"Session {session_id} not found for subject {self.subject_id}")

    def get_trial(self, session_id, trial_number):
        session = self.get_session(session_id)
        return session.get_trial(trial_number)

class Session:
    
    def __init__(self, session_id: str, subject: Subject, excluded_trials: list = []):
        self.session_id = session_id
        self.subject = subject
        self.excluded_trials = excluded_trials
        self.session_path = subject.get_subject_path_derivatives() / f"ses-{self.session_id}"
        self.behavior_path = subject.get_subject_path() / f"ses-{self.session_id}" / "behavioral"
        self._trials = None  # Lazy load trials

        if not self.session_path.exists():
            raise FileNotFoundError(f"Session path not found: {self.session_path}")
        
    
    @property
    def trials(self):
        if self._trials is None:
            raise ValueError("Trials not loaded. Please load data first.")
        return self._trials

    def __repr__(self):
        return f"Session = '{self.session_id}', " + self.subject.__repr__()
    
    def get_session_id(self):
        return self.session_id

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        bad_trials = []
        total_trials = len(self.trials)
        # Filter bad trials

        bad_trials = [trial.get_trial_number() for trial in self.trials if trial.is_trial_bad(threshold)]
        if len(bad_trials)/total_trials > threshold:
            bad_trials = self._trials
        self._trials = [trial for trial in self.trials if trial.get_trial_number() not in bad_trials]
        for trial in bad_trials:
            trial.session = None
        return bad_trials, total_trials
    
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        events_path = self.session_path / f"{self.detection_algorithm}_events"
        
        # Check paths and load files efficiently
        samples_path = self.session_path / "samples.hdf5"
        fix_path = events_path / "fix.hdf5"
        sacc_path = events_path / "sacc.hdf5"
        
        samples = pd.read_hdf(samples_path, memory_map=True)
        fix = pd.read_hdf(fix_path, memory_map=True)
        sacc = pd.read_hdf(sacc_path, memory_map=True)
        blink = pd.read_hdf(events_path / "blink.hdf5", memory_map=True) if (events_path / "blink.hdf5").exists() else None

        self.behavior_data = None
        if self.behavior_path.exists():
            behavior_files = list(self.behavior_path.glob("*.csv"))
            if len(behavior_files) == 1:
                self.behavior_data = pd.read_csv(behavior_files[0])

        # Initialize trials
        self._trials = [
            Trial(trial, self, samples, fix, sacc, blink, events_path)
            for trial in samples["trial_number"].unique() 
            if trial != -1 and trial not in self.excluded_trials
        ]

    def get_fixations(self):
        return pd.concat([trial.get_fixations() for trial in self.trials], ignore_index=True)

    def get_saccades(self):
        return pd.concat([trial.get_saccades() for trial in self.trials], ignore_index=True)

    def plot_scanpaths(self,screen_height,screen_width, display: bool = False):
        for trial in self.trials:
            trial.plot_scanpath(screen_height,screen_width,display=display)

    def __iter__(self):
        return iter(self.trials)
    
    def __getitem__(self, index):
        return self.trials[index]
    
    def __len__(self):
        return len(self.trials)
    
    def get_trial(self, trial_number):
        for trial in self.trials:
            if trial.get_trial_number() == trial_number:
                return trial
        raise ValueError(f"Trial {trial_number} not found for session {self.session_id} of subject {self.subject.get_id()}")

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for trial in self.trials:
            trial.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for trial in self.trials:
            trial.filter_saccades(max_sacc_dur)

    def get_rts(self):
        rts = [trial.get_rts() for trial in self.trials]
        rts = pd.concat(rts, ignore_index=True)
        rts["session_id"] = self.session_id
        return rts



class Trial:

    def __init__(self, trial_number: int, session: Session, samples: pd.DataFrame, fix: pd.DataFrame, 
                 sacc: pd.DataFrame, blink: pd.DataFrame, events_path: str):
        self.trial_number = trial_number
        self.session = session
        self.samples = samples[samples["trial_number"] == trial_number].reset_index(drop=True)
        self.fix = fix[fix["trial_number"] == trial_number].reset_index(drop=True)
        self.sacc = sacc[sacc["trial_number"] == trial_number].reset_index(drop=True)
        self.blink = blink[blink["trial_number"] == trial_number].reset_index(drop=True) if blink is not None else None
        self.events_path = events_path
        self.detection_algorithm = events_path.name[:-7]

    def __repr__(self):
        return f"Trial = '{self.trial_number}', " + self.session.__repr__()

    def get_trial_number(self):
        return self.trial_number
    
    def get_fixations(self):
        return self.fix
    
    def get_saccades(self):
        return self.sacc

    def plot_scanpath(self,screen_height,screen_width, **kwargs):
        vis = Visualization(self.events_path, self.detection_algorithm)
        (self.events_path / "plots").mkdir(parents=True, exist_ok=True)
        vis.scanpath(fixations=self.fix, saccades=self.sacc, samples=self.samples, screen_height=screen_height, screen_width=screen_width, 
                      folder_path=self.events_path / "plots", **kwargs)

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        self.fix = self.fix.query(f"{min_fix_dur} < duration < {max_fix_dur} and bad == False").reset_index(drop=True)

    def filter_saccades(self, max_sacc_dur=100):
        self.sacc = self.sacc.query(f"duration < {max_sacc_dur} and bad == False").reset_index(drop=True)

    def save_rts(self):
        if hasattr(self, "rts"):
            return
        rts = self.samples[self.samples["phase"] != ""].groupby(["phase"])["tSample"].agg(lambda x: x.iloc[-1] - x.iloc[0])
        self.rts = rts.reset_index().rename(columns={"tSample": "rt"})
        self.rts["trial_number"] = self.trial_number

    def get_rts(self):
        if not hasattr(self, "rts"):
            self.save_rts()
        return self.rts
    
    def is_trial_bad(self, threshold=0.5):
        nan_values = self.samples.isna().sum().sum()
        bad_values = self.samples["bad"].sum()
        bad_and_nan_percentage = (nan_values + bad_values) / len(self.samples)
        return bad_and_nan_percentage > threshold