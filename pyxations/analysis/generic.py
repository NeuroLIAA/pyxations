from pathlib import Path
import pandas as pd
from pyxations.visualization.visualization import Visualization
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
from pyxations.export import FEATHER_EXPORT, get_exporter


STIMULI_FOLDER = "stimuli"
ITEMS_FOLDER = "items"

def _find_fixation_cutoff(fix_count_list, threshold, max_possible):
    """
    fix_count_list: The list of fixation counts for each trial
    threshold: e.g. 0.95 * sum(fix_list)
    max_possible: max(fix_list), or possibly something else, depending on logic

    Returns: For each element in fix_list, sum the minimum of the element and a given index i, until the sum is greater than or equal to the threshold.
    Then return that index i.
    """

    # If threshold >= sum of fix_list, return max_possible
    if threshold >= sum(fix_count_list):
        return max_possible-1

    for i, val in enumerate(range(max_possible)):
        summation = sum([min(fix_count, val) for fix_count in fix_count_list])
        if summation >= threshold:
            return i

    return max_possible-1

class Experiment:

    def __init__(self, dataset_path: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(self.dataset_path.name + "_derivatives")
        self.metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t", 
                                    dtype={"subject_id": str, "old_subject_id": str})
        self.subjects = { subject_id:
            Subject(subject_id, old_subject_id, self, 
                     excluded_sessions.get(subject_id, []), excluded_trials.get(subject_id, {}),export_format)
            for subject_id, old_subject_id in zip(self.metadata["subject_id"], self.metadata["old_subject_id"])
            if subject_id not in excluded_subjects and old_subject_id not in excluded_subjects
        }
        self.export_format = export_format

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
        for subject in self.subjects.values():
            subject.load_data(detection_algorithm)

    def plot_multipanel(self, display: bool):
        fixations = pd.concat([subject.fixations() for subject in self.subjects.values()], ignore_index=True)
        saccades = pd.concat([subject.saccades() for subject in self.subjects.values()], ignore_index=True)

        vis = Visualization(self.derivatives_path, self.detection_algorithm)
        vis.plot_multipanel(fixations, saccades, display)

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for subject in self.subjects.values():
            subject.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for subject in self.subjects.values():
            subject.filter_saccades(max_sacc_dur)

    def drop_trials_with_nan_threshold(self, phase, threshold=0.1,print_flag=True):
        amount_trials_total = self.get_rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.drop_trials_with_nan_threshold(phase,threshold,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.get_rts().shape[0]} trials with NaN values.")

    def drop_trials_longer_than(self, seconds,phase, print_flag=True):
        amount_trials_total = self.get_rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.drop_trials_longer_than(seconds,phase,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.get_rts().shape[0]} trials longer than {seconds} seconds.")
    
    def plot_scanpaths(self,screen_height,screen_width,display: bool = False):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(subject.plot_scanpaths,screen_height,screen_width,display) for subject in self.subjects.values()]
            for future in as_completed(futures):
                future.result()


    def get_rts(self):
        rts = [subject.get_rts() for subject in self.subjects.values()]
        return pd.concat(rts, ignore_index=True)

    def get_subject(self, subject_id):
        return self.subjects[subject_id]
    
    def get_session(self, subject_id, session_id):
        subject = self.get_subject(subject_id)
        return subject.get_session(session_id)
    
    def get_trial(self, subject_id, session_id, trial_number):
        session = self.get_session(subject_id, session_id)
        return session.get_trial(trial_number)
    
    def fixations(self):
        return pd.concat([subject.fixations() for subject in self.subjects.values()], ignore_index=True)
    
    def saccades(self):
        return pd.concat([subject.saccades() for subject in self.subjects.values()], ignore_index=True)
    
    def samples(self):
        return pd.concat([subject.samples() for subject in self.subjects.values()], ignore_index=True)
    
    def remove_subject(self, subject_id):
        del self.subjects[subject_id]


class Subject:

    def __init__(self, subject_id: str, old_subject_id: str, experiment: Experiment,
                 excluded_sessions: list = [], excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.subject_id = subject_id
        self.old_subject_id = old_subject_id
        self.experiment = experiment
        self._sessions = None  # Lazy load sessions
        self.excluded_sessions = excluded_sessions
        self.excluded_trials = excluded_trials
        self.subject_dataset_path = self.experiment.dataset_path / f"sub-{self.subject_id}"
        self.subject_derivatives_path = self.experiment.derivatives_path / f"sub-{self.subject_id}"
        self.export_format = export_format

    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = { session_folder.name.split("-")[-1] :
                Session(session_folder.name.split("-")[-1], self,
                        self.excluded_trials.get(session_folder.name.split("-")[-1], {}),self.export_format) 
                for session_folder in self.subject_derivatives_path.glob("ses-*") 
                if session_folder.name.split("-")[-1] not in self.excluded_sessions
            }
        return self._sessions

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
    
    def unlink_experiment(self):
        keys = list(self.sessions.keys())
        for session in keys:
            self.sessions[session].unlink_subject()
        self.experiment.remove_subject(self.subject_id)
        self.experiment = None
    
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        for session in self.sessions.values():
            session.load_data(detection_algorithm)


    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for session in self.sessions.values():
            session.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for session in self.sessions.values():
            session.filter_saccades(max_sacc_dur)

    def drop_trials_with_nan_threshold(self,phase, threshold=0.1, print_flag=True):
        total_sessions = len(self.sessions)
        amount_trials_total = self.get_rts().shape[0]
        for session in list(self.sessions.values()):
            session.drop_trials_with_nan_threshold(phase,threshold,False)
        bad_sessions_count = total_sessions - len(self.sessions)


        # If the proportion of bad sessions exceeds the threshold, remove all sessions
        if bad_sessions_count / total_sessions > threshold:
            self.unlink_experiment()
        
        if print_flag:
            print(f"Removed {amount_trials_total - self.get_rts().shape[0]} trials with NaN values.")

    def drop_trials_longer_than(self, seconds,phase, print_flag=True):
        amount_trials_total = self.get_rts().shape[0]
        for session in list(self.sessions.values()):
            session.drop_trials_longer_than(seconds,phase,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.get_rts().shape[0]} trials longer than {seconds} seconds.")


    def plot_scanpaths(self,screen_height,screen_width, display: bool = False):
        for session in self.sessions.values():
            session.plot_scanpaths(screen_height,screen_width,display)

    def get_rts(self):
        rts = [session.get_rts() for session in self.sessions.values()]
        rts = pd.concat(rts, ignore_index=True)
        rts["subject_id"] = self.subject_id
        return rts

    def get_session(self, session_id):
        return self.sessions[session_id]

    def get_trial(self, session_id, trial_number):
        session = self.get_session(session_id)
        return session.get_trial(trial_number)
    
    def fixations(self):
        df = pd.concat([session.fixations() for session in self.sessions.values()], ignore_index=True)
        df["subject_id"] = self.subject_id
        return df
    
    def saccades(self):
        df = pd.concat([session.saccades() for session in self.sessions.values()], ignore_index=True)
        df["subject_id"] = self.subject_id
        return df
    
    def samples(self):
        df = pd.concat([session.samples() for session in self.sessions.values()], ignore_index=True)
        df["subject_id"] = self.subject_id
        return df

    def remove_session(self, session_id):
        del self._sessions[session_id]

class Session():
    
    def __init__(self, session_id: str, subject: Subject, excluded_trials: list = [],export_format = FEATHER_EXPORT):
        self.session_id = session_id
        self.subject = subject
        self.excluded_trials = excluded_trials
        self.session_dataset_path = self.subject.subject_dataset_path / f"ses-{self.session_id}"
        self.session_derivatives_path = self.subject.subject_derivatives_path / f"ses-{self.session_id}"
        self._trials = None  # Lazy load trials
        self.export_format = export_format

        if not self.session_derivatives_path.exists():
            raise FileNotFoundError(f"Session path not found: {self.session_derivatives_path}")
        
    
    @property
    def trials(self):
        if self._trials is None:
            raise ValueError("Trials not loaded. Please load data first.")
        return self._trials

    def __repr__(self):
        return f"Session = '{self.session_id}', " + self.subject.__repr__()
    
    def unlink_subject(self):
        keys = list(self.trials.keys())
        for trial in keys:
            self.trials[trial].unlink_session()
        self.subject.remove_session(self.session_id)
        self.subject = None

    def drop_trials_with_nan_threshold(self, phase, threshold=0.1, print_flag=True):
        bad_trials = []
        total_trials = len(self.trials)
        # Filter bad trials

        bad_trials = [trial for trial in self.trials.keys() if self.trials[trial].is_trial_bad(phase, threshold)]
        if len(bad_trials)/total_trials > threshold:
            bad_trials = self._trials
            self.unlink_subject()
        else:
            for trial in bad_trials:
                self.trials[trial].unlink_session()
        
        if print_flag:
            print(f"Removed {len(bad_trials)} trials with NaN values.")

    def drop_trials_longer_than(self, seconds,phase, print_flag=True):
        bad_trials = []

        # Filter bad trials

        bad_trials = [trial for trial in self.trials.keys() if self.trials[trial].is_trial_longer_than(seconds,phase)]
        for trial in bad_trials:
            self.trials[trial].unlink_session()
        
        if len(self.trials) == 0:
            self.unlink_subject()
               
        if print_flag:
            print(f"Removed {len(bad_trials)} trials longer than {seconds} seconds.")

    def load_behavior_data(self):
        # This should be implemented for each type of experiment
        pass

    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        events_path = self.session_derivatives_path / f"{self.detection_algorithm}_events"
        
        
        exporter = get_exporter(self.export_format)
        file_extension = exporter.extension()
        
        
        # Check paths and load files efficiently
        
        samples = exporter.read(self.session_derivatives_path, 'samples')
        fix = exporter.read(events_path, 'fix')
        sacc = exporter.read(events_path, 'sacc')
        blink = exporter.read(events_path, "blink") if (events_path / ("blink" + file_extension)).exists() else None
   
        # Initialize trials
        self._init_trials(samples,fix,sacc,blink,events_path)


    def _init_trials(self,samples,fix,sacc,blink,events_path):
        cosas = [trial for trial in samples["trial_number"].unique() if trial != -1 and trial not in self.excluded_trials]
        self._trials = {trial:
            Trial(trial, self, samples, fix, sacc, blink, events_path)
            for trial in cosas
        } 

    def plot_scanpaths(self,screen_height,screen_width, display: bool = False):
        for trial in self.trials.values():
            trial.plot_scanpath(screen_height,screen_width,display=display)

    def __iter__(self):
        return iter(self.trials)
    
    def __getitem__(self, index):
        return self.trials[index]
    
    def __len__(self):
        return len(self.trials)
    
    def get_trial(self, trial_number):
        return self._trials[trial_number]

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        for trial in self.trials.values():
            trial.filter_fixations(min_fix_dur, max_fix_dur)

    def filter_saccades(self, max_sacc_dur=100):
        for trial in self.trials.values():
            trial.filter_saccades(max_sacc_dur)

    def get_rts(self):
        rts = [trial.get_rts() for trial in self.trials.values()]
        rts = pd.concat(rts, ignore_index=True)
        rts["session_id"] = self.session_id
        return rts

    def fixations(self):
        df = pd.concat([trial.fixations() for trial in self.trials.values()], ignore_index=True)
        df["session_id"] = self.session_id
        return df
    
    def saccades(self):
        df = pd.concat([trial.saccades() for trial in self.trials.values()], ignore_index=True)
        df["session_id"] = self.session_id
        return df
        
    
    def samples(self):
        df = pd.concat([trial.samples() for trial in self.trials.values()], ignore_index=True)
        df["session_id"] = self.session_id
        return df

    def remove_trial(self, trial_number):
        del self._trials[trial_number]

class Trial:

    def __init__(self, trial_number: int, session: Session, samples: pd.DataFrame, fix: pd.DataFrame, 
                 sacc: pd.DataFrame, blink: pd.DataFrame, events_path: Path):
        self.trial_number = trial_number
        self.session = session
        self._samples = samples[samples["trial_number"] == trial_number].reset_index(drop=True)
        self._fix = fix[fix["trial_number"] == trial_number].reset_index(drop=True)
        self._sacc = sacc[sacc["trial_number"] == trial_number].reset_index(drop=True)
        self._blink = blink[blink["trial_number"] == trial_number].reset_index(drop=True) if blink is not None else None
        start_time = self._samples["tSample"].iloc[0]
        self._samples["tSample"] = self._samples["tSample"] - start_time
        self._fix["tStart"] = self._fix["tStart"] - start_time
        self._fix["tEnd"] = self._fix["tEnd"] - start_time
        self._sacc["tStart"] = self._sacc["tStart"] - start_time
        self._sacc["tEnd"] = self._sacc["tEnd"] - start_time
        if self._blink is not None:
            self._blink["tStart"] = self._blink["tStart"] - start_time
            self._blink["tEnd"] = self._blink["tEnd"] - start_time

        self.events_path = events_path
        self.detection_algorithm = events_path.name[:-7]

    def fixations(self):
        return self._fix
    

    def saccades(self):
        return self._sacc
    

    def samples(self):
        return self._samples

    def __repr__(self):
        return f"Trial = '{self.trial_number}', " + self.session.__repr__()

    def unlink_session(self):
        self.session.remove_trial(self.trial_number)
        self.session = None

    def plot_scanpath(self,screen_height,screen_width, **kwargs):
        vis = Visualization(self.events_path, self.detection_algorithm)
        (self.events_path / "plots").mkdir(parents=True, exist_ok=True)
        vis.scanpath(fixations=self._fix, saccades=self._sacc, samples=self._samples, screen_height=screen_height, screen_width=screen_width, 
                      folder_path=self.events_path / "plots", **kwargs)

    def filter_fixations(self, min_fix_dur=50, max_fix_dur=1000):
        self._fix = self._fix.query(f"{min_fix_dur} < duration < {max_fix_dur} and bad == False").reset_index(drop=True)

    def filter_saccades(self, max_sacc_dur=100):
        self._sacc = self._sacc.query(f"duration < {max_sacc_dur} and bad == False").reset_index(drop=True)

    def save_rts(self):
        if hasattr(self, "rts"):
            return
        rts = self._samples[self._samples["phase"] != ""].groupby(["phase"])["tSample"].agg(lambda x: x.iloc[-1] - x.iloc[0])
        self.rts = rts.reset_index().rename(columns={"tSample": "rt"})
        self.rts["trial_number"] = self.trial_number

    def get_rts(self):
        if not hasattr(self, "rts"):
            self.save_rts()
        return self.rts
    
    def is_trial_bad(self,phase, threshold=0.1):
        samples = self._samples.copy()[self._samples["phase"] == phase].reset_index(drop=True)
        
        # Remove samples that fall within any blink period
        if self._blink is not None and not self._blink.empty:
            blink_start = self._blink["tStart"].values
            blink_end = self._blink["tEnd"].values
            
            # Mask to keep samples that do not fall in any blink interval
            mask = ~samples["tSample"].apply(
                lambda t: any(start < t < end for start, end in zip(blink_start, blink_end))
            )
            samples = samples[mask]
        
        # Compute bad and NaN values
        nan_values = samples.isna().sum().sum()
        bad_values = samples["bad"].sum()
        
        bad_and_nan_percentage = (nan_values + bad_values) / len(samples) if len(samples) > 0 else 1.0
        
        return bad_and_nan_percentage > threshold
    
    def is_trial_longer_than(self, seconds, phase):
        return self.get_rts()[self.get_rts()["phase"] == phase]["rt"].values[0] > seconds * 1000.0
