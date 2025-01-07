from pathlib import Path
import pandas as pd
from pyxations.visualization.visualization import Visualization
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pyxations.export import FEATHER_EXPORT, HDF5_EXPORT
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

STIMULI_FOLDER = "stimuli"
ITEMS_FOLDER = "items"

class Experiment:

    def __init__(self, dataset_path: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(self.dataset_path.name + "_derivatives")
        self.metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t", 
                                    dtype={"subject_id": str, "old_subject_id": str})
        self.subjects = { subject_id:
            Subject(subject_id, old_subject_id, self, self.dataset_path / f"sub-{subject_id}",
                    self.derivatives_path / f"sub-{subject_id}",
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
        #with ThreadPoolExecutor() as executor:
            #futures = [executor.submit(subject.load_data, detection_algorithm) for subject in self.subjects.values()]
            #for future in as_completed(futures):
                #future.result()
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

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        #TODO: TEST after the changes
        sessions_results = {subject: self.subjects[subject].drop_trials_with_nan_threshold(threshold) for subject in self.subjects}
        bad_trials_total = {subject: sessions_results[subject][2] for subject in sessions_results.keys()}
        return bad_trials_total
    
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

    def __init__(self, subject_id: str, old_subject_id: str, experiment: Experiment, subject_dataset_path: Path, subject_derivatives_path: Path,
                 excluded_sessions: list = [], excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.subject_id = subject_id
        self.old_subject_id = old_subject_id
        self.experiment = experiment
        self._sessions = None  # Lazy load sessions
        self.excluded_sessions = excluded_sessions
        self.excluded_trials = excluded_trials
        self.subject_dataset_path = subject_dataset_path
        self.subject_derivatives_path = subject_derivatives_path
        self.export_format = export_format

    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = { session_folder.name.split("-")[-1] :
                Session(session_folder.name.split("-")[-1], self, session_folder, self.subject_dataset_path / session_folder.name / "behavioral",
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

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        total_sessions = len(self.sessions)
        sessions_results = {session: self.sessions[session].drop_trials_with_nan_threshold(threshold) for session in self.sessions}
        bad_sessions_count = total_sessions - len(self.sessions)
        bad_trials_subject = {session: {"bad_trials": sessions_results[session][0], "total_trials": sessions_results[session][1]} for session in sessions_results.keys()}

        # If the proportion of bad sessions exceeds the threshold, remove all sessions
        if bad_sessions_count / total_sessions > threshold:
            self.unlink_experiment()

        return bad_sessions_count, total_sessions, bad_trials_subject

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
    
    def __init__(self, session_id: str, subject: Subject, session_path: Path, behavior_path: Path, excluded_trials: list = [],export_format = FEATHER_EXPORT):
        self.session_id = session_id
        self.subject = subject
        self.excluded_trials = excluded_trials
        self.session_path = session_path
        self.behavior_path = behavior_path
        self._trials = None  # Lazy load trials
        self.export_format = export_format

        if not self.session_path.exists():
            raise FileNotFoundError(f"Session path not found: {self.session_path}")
        
    
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

    def drop_trials_with_nan_threshold(self, threshold=0.5):
        bad_trials = []
        total_trials = len(self.trials)
        # Filter bad trials

        bad_trials = [trial for trial in self.trials.keys() if self.trials[trial].is_trial_bad(threshold)]
        if len(bad_trials)/total_trials > threshold:
            bad_trials = self._trials
            self.unlink_subject()
        else:
            for trial in bad_trials:
                self.trials[trial].unlink_session()
        return bad_trials, total_trials

    def load_behavior_data(self):
        # This should be implemented for each type of experiment
        pass

    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        events_path = self.session_path / f"{self.detection_algorithm}_events"
        
        if self.export_format == FEATHER_EXPORT:
            file_extension = "feather"
            reader = pd.read_feather
        elif self.export_format == HDF5_EXPORT:
            file_extension = "hdf5"
            reader = pd.read_hdf
        else:
            raise ValueError(f"Export format {self.export_format} not found.")

        
        # Check paths and load files efficiently
        samples_path = self.session_path / ("samples." + file_extension)
        fix_path = events_path / ("fix." + file_extension)
        sacc_path = events_path / ("sacc." + file_extension)
        
        samples = reader(samples_path)
        fix = reader(fix_path)
        sacc = reader(sacc_path)
        blink = reader(events_path / ("blink." + file_extension)) if (events_path / ("blink." + file_extension)).exists() else None

        self.behavior_data = None
        if self.behavior_path.exists():
            behavior_files = list(self.behavior_path.glob("*.csv"))
            if len(behavior_files) == 1:
                self.behavior_data = pd.read_csv(behavior_files[0])

        # Initialize trials
        self._init_trials(samples,fix,sacc,blink,events_path)

    def _init_trials(self,samples,fix,sacc,blink,events_path):
        self._trials = {trial:
            Trial(trial, self, samples, fix, sacc, blink, events_path)
            for trial in samples["trial_number"].unique() 
            if trial != -1 and trial not in self.excluded_trials
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
    
    def is_trial_bad(self, threshold=0.5):
        nan_values = self._samples.isna().sum().sum()
        bad_values = self._samples["bad"].sum()
        bad_and_nan_percentage = (nan_values + bad_values) / len(self._samples)
        return bad_and_nan_percentage > threshold
    




class VisualSearchExperiment(Experiment):
    def __init__(self, dataset_path: str,search_phase_name: str,memorization_phase_name: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(self.dataset_path.name + "_derivatives")
        self.metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t", 
                                    dtype={"subject_id": str, "old_subject_id": str})
        self.subjects = { subject_id:
            VisualSearchSubject(subject_id, old_subject_id, self, self.dataset_path / f"sub-{subject_id}",
                    self.derivatives_path / f"sub-{subject_id}", search_phase_name, memorization_phase_name,
                     excluded_sessions.get(subject_id, []), excluded_trials.get(subject_id, {}),export_format)
            for subject_id, old_subject_id in zip(self.metadata["subject_id"], self.metadata["old_subject_id"])
            if subject_id not in excluded_subjects and old_subject_id not in excluded_subjects
        }
        self.export_format = export_format
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name

    def correct_trials(self):
        correct_trials = pd.concat([subject.correct_trials() for subject in self.subjects.values()], ignore_index=True)

        return correct_trials

    def accuracy(self):
        accuracy = pd.concat([subject.accuracy() for subject in self.subjects.values()], ignore_index=True)

        return accuracy
    
    def plot_accuracy(self):
        accuracy = self.accuracy()
        # Change the name of the correct_response column to accuracy
        accuracy = accuracy.sort_values(by=["target_present", "accuracy"])
        # There should be an ax for each pair of target present and memory set size
        n_cols = len(accuracy["target_present"].unique())
        n_rows = len(accuracy["memory_set_size"].unique())
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows),sharey=True)
        if n_cols == 1:
            axs = np.array([axs])
        if n_rows == 1:
            axs = np.array([axs])

        for i, row in enumerate(accuracy["memory_set_size"].unique()):
            for j, col in enumerate(accuracy["target_present"].unique()):
                data = accuracy[(accuracy["memory_set_size"] == row) & (accuracy["target_present"] == col)]
                sns.barplot(x="subject_id", y="accuracy", data=data, ax=axs[i, j],estimator="mean")
                axs[i, j].set_title(f"Memory Set Size {row}, Target Present: {bool(col)}")
                axs[i, j].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.show()
        plt.close()

    def remove_poor_accuracy_sessions(self, threshold=0.5):
        keys = list(self.subjects.keys())
        for subject in keys:
            self.subjects[subject].remove_poor_accuracy_sessions(threshold)

    def scanpaths_by_stimuli(self):
        return pd.concat([subject.scanpaths_by_stimuli() for subject in self.subjects.values()], ignore_index=True)


    def remove_trials_for_stimuli_with_poor_accuracy(self, threshold=0.5):
        scanpaths_by_stimuli = self.scanpaths_by_stimuli()
        grouped = scanpaths_by_stimuli.groupby(["stimulus", "target_present", "memory_set_size"])
        poor_accuracy_stimuli = grouped["correct_response"].mean() < threshold
        poor_accuracy_stimuli = poor_accuracy_stimuli[poor_accuracy_stimuli].index
        subj_keys = list(self.subjects.keys())
        for subject_key in subj_keys:
            subject = self.subjects[subject_key]
            session_keys = list(subject.sessions.keys())
            for session_key in session_keys:
                session = subject.sessions[session_key]
                trial_keys = list(session.trials.keys())
                for trial_key in trial_keys:
                    trial = session.trials[trial_key]
                    if (trial.stimulus, trial.target_present, trial.memory_set_size) in poor_accuracy_stimuli:
                        trial.unlink_session()
                if len(session.trials) == 0:
                    session.unlink_subject()
            if len(subject.sessions) == 0:
                subject.unlink_experiment()

    
    def cumulative_correct_trials(self, max_fixations=20):
        cumulative_correct = pd.concat([subject.cumulative_correct_trials(max_fixations) for subject in self.subjects.values()], ignore_index=True)

        return cumulative_correct

    def cumulative_performance(self, max_fixations=20):
        cumulative_performance = pd.concat([subject.cumulative_performance(max_fixations) for subject in self.subjects.values()], ignore_index=True)

        return cumulative_performance
    
    def plot_cumulative_performance(self, max_fixations=20):
        cumulative_performance = self.cumulative_performance(max_fixations)
        n_rows = len(cumulative_performance["memory_set_size"].unique())
        n_cols = len(cumulative_performance["target_present"].unique())
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows),sharey=True,sharex=True)
        fig.suptitle("Cumulative Performance")
        if n_cols == 1:
            axs = np.array([axs])

        if n_rows == 1:
            axs = np.array([axs])

        # For each fixation number (i.e. first "max_fixations" columns), we need the mean and the standard error
        # The X axis will be the fixation number, the Y axis will be the accuracy
        # The area around the mean will be the standard error
        columns_starting_with_fix = [col for col in cumulative_performance.columns if col.startswith("fix")]
        for i, row in enumerate(cumulative_performance["memory_set_size"].unique()):
            for j, col in enumerate(cumulative_performance["target_present"].unique()):
                data = cumulative_performance[(cumulative_performance["memory_set_size"] == row) & (cumulative_performance["target_present"] == col)]
                data_mean = data[columns_starting_with_fix].mean()
                data_standard_error = data[columns_starting_with_fix].sem()
                axs[i, j].plot(data_mean,color="black")
                axs[i, j].fill_between(data_mean.index, data_mean - data_standard_error, data_mean + data_standard_error, alpha=0.5, color="gray")
                axs[i, j].set_title(f"Memory Set Size {int(row)}, Target Present {bool(col)}")
                axs[i, j].set_xticks(range(max_fixations))
                axs[i, j].set_xticklabels(range(1,max_fixations+1))

        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        plt.close()

    def correct_trials_by_rt_bins(self, bin_end,bin_step):
        correct_trials = pd.concat([subject.correct_trials_by_rt_bins(bin_end,bin_step) for subject in self.subjects.values()], ignore_index=True)

        return correct_trials

    def plot_correct_trials_by_rt_bins(self, bin_end,bin_step):
        correct_trials_per_bin = self.correct_trials_by_rt_bins(bin_end,bin_step)[["rt_bin","target_present","memory_set_size","correct_response"]]
        correct_trials_per_bin = correct_trials_per_bin.groupby(["rt_bin","target_present","memory_set_size"],observed=False).sum().reset_index()

        n_cols = len(correct_trials_per_bin["target_present"].unique())
        n_rows = len(correct_trials_per_bin["memory_set_size"].unique())
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows),sharey=True,sharex=True)
        fig.suptitle("Correct Trials by RT Bins")
        if n_cols == 1:
            axs = np.array([axs])
        if n_rows == 1:
            axs = np.array([axs])
        for i, row in enumerate(correct_trials_per_bin["memory_set_size"].unique()):
            for j, col in enumerate(correct_trials_per_bin["target_present"].unique()):
                data = correct_trials_per_bin[(correct_trials_per_bin["memory_set_size"] == row) & (correct_trials_per_bin["target_present"] == col)]
                sns.barplot(x="rt_bin", y="correct_response", data=data, ax=axs[i, j])
                axs[i, j].set_title(f"Memory Set Size {int(row)}, Target Present {bool(col)}")
                axs[i, j].set_xlabel("RT Bins")
                axs[i, j].set_ylabel("Correct Trials")
        plt.tight_layout()
        plt.show()
        plt.close()        


class VisualSearchSubject(Subject):
    def __init__(self, subject_id: str, old_subject_id: str, experiment: VisualSearchExperiment, subject_dataset_path: Path, subject_derivatives_path: Path,search_phase_name, memorization_phase_name,
                 excluded_sessions: list = [], excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        super().__init__(subject_id, old_subject_id, experiment, subject_dataset_path, subject_derivatives_path, excluded_sessions, excluded_trials, export_format)
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name

    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = { session_folder.name.split("-")[-1] :
                VisualSearchSession(session_folder.name.split("-")[-1], self, session_folder, self.subject_dataset_path / session_folder.name / "behavioral",self._search_phase_name, self._memorization_phase_name,
                        self.excluded_trials.get(session_folder.name.split("-")[-1], {}),self.export_format) 
                for session_folder in self.subject_derivatives_path.glob("ses-*") 
                if session_folder.name.split("-")[-1] not in self.excluded_sessions
            }
        return self._sessions
    
    def scanpaths_by_stimuli(self):
        return pd.concat([session.scanpaths_by_stimuli() for session in self.sessions.values()], ignore_index=True)
    
    def correct_trials(self):
        correct_trials = pd.concat([session.correct_trials() for session in self.sessions.values()], ignore_index=True)
        correct_trials["subject_id"] = self.subject_id

        return correct_trials
    
    def accuracy(self):
        correct_trials = self.correct_trials()
        accuracy = correct_trials.groupby(["target_present", "memory_set_size"]).sum().reset_index()
        accuracy["accuracy"] = accuracy["correct_response"] / accuracy["total_trials"]
        accuracy.drop(columns=["correct_response", "total_trials"], inplace=True)
        accuracy["subject_id"] = self.subject_id

        return accuracy


    def remove_poor_accuracy_sessions(self, threshold=0.5):
        poor_accuracy_sessions = []
        keys = list(self.sessions.keys())
        for key in keys:
            session = self.sessions[key]
            if session.has_poor_accuracy(threshold):
                poor_accuracy_sessions.append(session.session_id)
                session.unlink_subject()
        if len(poor_accuracy_sessions) == len(keys):
            self.unlink_experiment()

    def cumulative_correct_trials(self, max_fixations=20):
        cumulative_correct = []
        for session in self.sessions.values():
            cumulative_correct.append(session.cumulative_correct_trials(max_fixations))

        cumulative_correct = pd.concat(cumulative_correct, ignore_index=True)
        total_trials = cumulative_correct[['memory_set_size','target_present','total_trials']].groupby(['memory_set_size','target_present']).sum().reset_index()
        cumulative_correct = cumulative_correct.drop(columns=["total_trials"]).groupby(["memory_set_size","target_present"]).sum().reset_index()
        cumulative_correct = cumulative_correct.merge(total_trials, on=["memory_set_size","target_present"])


        return cumulative_correct

    
    def cumulative_performance(self, max_fixations=20):
        cumulative_performance = self.cumulative_correct_trials(max_fixations)
        columns_starting_with_fix = [col for col in cumulative_performance.columns if col.startswith("fix")]
        cumulative_performance[columns_starting_with_fix] = cumulative_performance[columns_starting_with_fix] / cumulative_performance["total_trials"].values[:, None]
        return cumulative_performance
    
    def correct_trials_by_rt_bins(self, bin_end,bin_step):
        correct_trials = pd.concat([session.correct_trials_by_rt_bins(bin_end,bin_step) for session in self.sessions.values()], ignore_index=True)
        
        return correct_trials

       
class VisualSearchSession(Session):
    BEH_COLUMNS: list[str] = [
        "trial_number", "stimulus", "stimulus_coords", "memory_set", "memory_set_locations",
        "target_present", "target", "target_location", "correct_response"
    ]
    """
    Columns explanation:
    - trial_number: The number of the trial, in the order they were presented. They start from 0.
    - stimulus: The filename of the stimulus presented.
    - stimulus_coords: The coordinates of the stimulus presented. It should be a tuple containing the top-left corner of the stimulus and the bottom-right corner.
    - memory_set: The set of items memorized by the participant. It should be a list of strings. Each string should be the filename of the stimulus.
    - memory_set_locations: The locations of the items memorized by the participant. It should be a list of tuples. Each tuple should contain bounding
      boxes of the items memorized by the participant. The bounding boxes should be in the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and
      (x2, y2) is the bottom-right corner.
    - target_present: Whether one of the items is present in the stimulus. It should be a boolean.
    - target: The filename of the target item. It should be a string.
    - target_location: The location of the target item. It should be a tuple containing the bounding box of the target item. The bounding box should be in
      the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. If target_present is False, the value for this column will
      not be taken into account.
    - correct_response: The correct response for the trial. It should be a boolean.

    Notice that you can get the actual response of the user by using the "correct_response" and "target_present" columns.
    For all of the heights, widths and locations of the items, the values should be in pixels and according to the screen itself.
    """

    COLLECTION_COLUMNS: dict = {
        "stimulus_coords": tuple,           # Parse as a tuple
        "memory_set": list,                 # Parse as a list
        "memory_set_locations": list,       # Parse as a list of tuples
        "target_location": tuple          # Parse as a tuple
    }

    def __init__(
        self, 
        session_id: str, 
        subject: VisualSearchSubject, 
        session_path: Path, 
        behavior_path: Path,
        search_phase_name: str,
        memorization_phase_name: str,
        excluded_trials: list = None,
        export_format = FEATHER_EXPORT
    ):
        excluded_trials = [] if excluded_trials is None else excluded_trials
        super().__init__(session_id, subject, session_path, behavior_path, excluded_trials, export_format)
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name



    def load_behavior_data(self):
        # Get the name of the only csv file in the behavior path
        behavior_files = list(self.behavior_path.glob("*.csv"))
        
        if len(behavior_files) != 1:
            raise ValueError(
                f"There should only be one CSV file in the behavior path for session {self.session_id} "
                f"of subject {self.subject.subject_id}. Found files: {[file.name for file in behavior_files]}"
            )

        # Load the CSV file
        name = behavior_files[0].name
        self.behavior_data = pd.read_csv(
            self.behavior_path / name,
            dtype={
                "trial_number": int,
                "stimulus": str,
                "target_present": bool,
                "target": str,
                "correct_response": bool
            }
        )

        # Validate that all required columns are present
        missing_columns = set(self.BEH_COLUMNS) - set(self.behavior_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in behavior data: {missing_columns}")

    def _init_trials(self,samples,fix,sacc,blink,events_path):
        self._trials = {trial:
            VisualSearchTrial(trial, self, samples, fix, sacc, blink, events_path, self.behavior_data,self._search_phase_name,self._memorization_phase_name)
            for trial in samples["trial_number"].unique() 
            if trial != -1 and trial not in self.excluded_trials and trial in self.behavior_data["trial_number"].values
        }
    
    def load_data(self, detection_algorithm: str):
        self.load_behavior_data()
        super().load_data(detection_algorithm)
        del self.behavior_data

    def correct_trials(self):
        correct_trials = []
        for trial in self.trials.values():
            correct_trials.append({
                "target_present": trial.target_present,
                "memory_set_size": trial.memory_set_size,
                "correct_response": trial.correct_response
            })
        total_trials_per_memory_set_size = pd.DataFrame(correct_trials).groupby(["target_present", "memory_set_size"]).size().reset_index().rename(columns={0: "total_trials"})
        correct_trials = pd.DataFrame(correct_trials).groupby(["target_present", "memory_set_size"]).sum().reset_index()
        correct_trials["session_id"] = self.session_id
        correct_trials = correct_trials.merge(total_trials_per_memory_set_size, on=["target_present", "memory_set_size"])
        return correct_trials
    
    def correct_trials_by_rt_bins(self,bin_end,bin_step):
        bins = pd.interval_range(start=0, end=bin_end, freq=bin_step)
        rts = self.get_rts()
        rts = rts[rts["phase"] == self._search_phase_name].reset_index(drop=True)
        rts["rt"] = rts["rt"]/1000
        rts["rt_bin"] = pd.cut(rts["rt"], bins)
        # Map bin to the first element
        rts["rt_bin"] = rts["rt_bin"].apply(lambda x: x.left)
        return rts

    
    def accuracy(self):
        # Accuracy should be grouped by target present and memory set size
        correct_trials = self.correct_trials()
        accuracy = correct_trials.groupby(["target_present", "memory_set_size"]).sum().reset_index()
        accuracy["accuracy"] = accuracy["correct_response"] / accuracy["total_trials"]
        accuracy.drop(columns=["correct_response", "total_trials"], inplace=True)

        return accuracy

    def has_poor_accuracy(self, threshold=0.5):
        correct_trials = self.correct_trials()
        accuracy = correct_trials["correct_response"].sum() / correct_trials["total_trials"].sum()
        return accuracy < threshold
    
    def cumulative_correct_trials(self, max_fixations=20):
        correct_trials = np.zeros((len(self.trials), max_fixations+2))
        for i,trial in enumerate(self.trials.values()):
            scanpath_length = len(trial.search_fixations())
            if trial.correct_response and scanpath_length <= max_fixations:
                correct_trials[i, scanpath_length:] = 1
            correct_trials[i, -2] = trial.target_present
            correct_trials[i, -1] = trial.memory_set_size

        cumulative_correct = pd.DataFrame(correct_trials, columns=[f"fix_{i}" for i in range(1, max_fixations+1)] + ["target_present", "memory_set_size"])
        
        total_trials = cumulative_correct.groupby(["target_present", "memory_set_size"]).size().reset_index().rename(columns={0: "total_trials"})
        cumulative_correct = cumulative_correct.groupby(["target_present", "memory_set_size"]).sum().reset_index()
        cumulative_correct = cumulative_correct.merge(total_trials, on=["target_present", "memory_set_size"])
        return cumulative_correct
    
    def cumulative_performance(self, max_fixations=20):
        cumulative_performance = self.cumulative_correct_trials(max_fixations)
        columns_starting_with_fix = [col for col in cumulative_performance.columns if col.startswith("fix")]
        cumulative_performance[columns_starting_with_fix] = cumulative_performance[columns_starting_with_fix] / cumulative_performance["total_trials"].values[:, None]
        return cumulative_performance
    
    def scanpaths_by_stimuli(self):
        return pd.DataFrame([trial.scanpath_by_stimuli() for trial in self.trials.values()], columns=["fixations", "stimulus", "correct_response", "target_present", "memory_set_size"])

class VisualSearchTrial(Trial):

    def __init__(self, trial_number, session, samples, fix, sacc, blink, events_path, behavior_data,search_phase_name, memorization_phase_name,):
        super().__init__(trial_number, session, samples, fix, sacc, blink, events_path)
        self._target_present = behavior_data.loc[behavior_data["trial_number"] == trial_number, "target_present"].values[0]
        self._target = behavior_data.loc[behavior_data["trial_number"] == trial_number, "target"].values[0]
        if self._target_present:            
            self._target_location = ast.literal_eval(behavior_data.loc[behavior_data["trial_number"] == trial_number, "target_location"].values[0])

        self._correct_response = behavior_data.loc[behavior_data["trial_number"] == trial_number, "correct_response"].values[0]
        self._stimulus = behavior_data.loc[behavior_data["trial_number"] == trial_number, "stimulus"].values[0]
        self._stimulus_coords = ast.literal_eval(behavior_data.loc[behavior_data["trial_number"] == trial_number, "stimulus_coords"].values[0])
       
        self._memory_set = ast.literal_eval(behavior_data.loc[behavior_data["trial_number"] == trial_number, "memory_set"].values[0])
        self._memory_set_locations = ast.literal_eval(behavior_data.loc[behavior_data["trial_number"] == trial_number, "memory_set_locations"].values[0])
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name

    @property
    def target_present(self):
        return self._target_present
    
    @property
    def correct_response(self):
        return self._correct_response
    
    @property
    def memory_set_size(self):
        return len(self._memory_set)
    
    @property
    def stimulus(self):
        return self._stimulus

    def save_rts(self):
        if hasattr(self, "rts"):
            return
        rts = self._samples[self._samples["phase"] != ""].groupby(["phase"])["tSample"].agg(lambda x: x.iloc[-1] - x.iloc[0])
        self.rts = rts.reset_index().rename(columns={"tSample": "rt"})
        self.rts["trial_number"] = self.trial_number
        self.rts["memory_set_size"] = len(self._memory_set)
        self.rts["target_present"] = self._target_present
        self.rts["correct_response"] = self._correct_response
        # Make sure the values are of the correct type


    def search_fixations(self):
        return self._fix[self._fix["phase"] == self._search_phase_name].sort_values(by="tStart")
    
    def memorization_fixations(self):
        return self._fix[self._fix["phase"] == self._memorization_phase_name].sort_values(by="tStart")
    
    def search_saccades(self):
        return self._sacc[self._sacc["phase"] == self._search_phase_name].sort_values(by="tStart")
    
    def memorization_saccades(self):
        return self._sacc[self._sacc["phase"] == self._memorization_phase_name].sort_values(by="tStart")
    
    def search_samples(self):
        return self._samples[self._samples["phase"] == self._search_phase_name].sort_values(by="tSample")
    
    def memorization_samples(self):
        return self._samples[self._samples["phase"] == self._memorization_phase_name].sort_values(by="tSample")
    
    def scanpath_by_stimuli(self):
        return [self.search_fixations(), self._stimulus,self._correct_response,self._target_present,len(self._memory_set)]
    
    def plot_scanpath(self, screen_height, screen_width, **kwargs):
        '''
        Plots the scanpath of the trial. The scanpath will be plotted in two phases: the search phase and the memorization phase.
        The search phase will be plotted with the stimulus and the memorization phase will be plotted with the items memorized by the participant.
        The search phase will have the fixations and saccades of the trial, while the memorization phase will only have the fixations.
        The names of the phases should be the same ones used in the computation of the derivatives.
        If you don't really care about the memorization phase, you can pass None as an argument.

        '''
        vis = Visualization(self.events_path, self.detection_algorithm)
        (self.events_path / "plots").mkdir(parents=True, exist_ok=True)

        
        phase_data = {self._search_phase_name:{}, self._memorization_phase_name:{}}
        phase_data[self._search_phase_name]["img_paths"] = [self.session.subject.experiment.dataset_path.parent / STIMULI_FOLDER / self._stimulus]
        phase_data[self._search_phase_name]["img_plot_coords"] = [self._stimulus_coords]
        if self._memorization_phase_name is not None:
            phase_data[self._memorization_phase_name]["img_paths"] = [self.session.subject.experiment.dataset_path.parent / ITEMS_FOLDER / img for img in self._memory_set]
            phase_data[self._memorization_phase_name]["img_plot_coords"] = self._memory_set_locations

        # If the target is present add the "bbox" to the search_phase phase as a key-value pair
        if self._target_present:
            phase_data[self._search_phase_name]["bbox"] = self._target_location
        vis.scanpath(fixations=self._fix,phase_data=phase_data, saccades=self._sacc, samples=self._samples, screen_height=screen_height, screen_width=screen_width, 
                      folder_path=self.events_path / "plots", **kwargs)