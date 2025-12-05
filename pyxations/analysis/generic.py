from pathlib import Path
import polars as pl
from pyxations.visualization.visualization import Visualization
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
from pyxations.export import FEATHER_EXPORT, get_exporter
from math import hypot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
import weakref
import warnings
import multimatch_gaze as mm

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

def _parse_validations(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse EyeLink `!CAL VALIDATION …` lines that are stored in df["line"].
    Returns a tidy DataFrame with numeric columns ready for plotting.
    """
    df = df.filter(pl.col("line").str.contains("CAL VALIDATION")).select(["line","Calib_index"])
    # column "line" does not contain "ABORTED"

    # 0 · remove the "ABORTED" lines (if any)
    df = df.filter(~pl.col("line").str.contains("ABORTED"))

    # 1 · pull the pieces out with .str.extract
    parsed = (
        df
        .with_columns([
            # time‑stamp after the initial MSG token
            pl.col("line")
              .str.extract(r"MSG\s+(\d+)", 1)
              .cast(pl.Int64)
              .alias("timestamp"),

            # eye label (LEFT / RIGHT)
            pl.col("line")
              .str.extract(r"\s(LEFT|RIGHT)\s", 1)
              .alias("eye"),

            # average and maximum error (deg)
            pl.col("line")
              .str.extract(r"ERROR\s+([\d.]+)\s+avg", 1)
              .cast(pl.Float64)
              .alias("avg_error"),

            pl.col("line")
              .str.extract(r"avg\.\s+([\d.]+)\s+max", 1)
              .cast(pl.Float64)
              .alias("max_error"),

            # total offset (deg)
            pl.col("line")
              .str.extract(r"OFFSET\s+([\d.]+)\s+deg", 1)
              .cast(pl.Float64)
              .alias("offset_deg"),

            # X / Y pixel offsets  (two separate capture groups)
            pl.col("line")
              .str.extract(r"deg\.\s+(-?[\d.]+),(-?[\d.]+)", 1)
              .cast(pl.Float64)
              .alias("offset_x"),

            pl.col("line")
              .str.extract(r"deg\.\s+(-?[\d.]+),(-?[\d.]+)", 2)
              .cast(pl.Float64)
              .alias("offset_y"),
        ])
    )

    # 2 · create a validation index (0‑based) within each calibration block
    parsed = (
        parsed
        .with_columns(
            pl.col("line")                 # ← a column to operate on
            .cum_count()                 # running 0, 1, 2, …
            .over(["Calib_index", "eye"])# reset counter per calibration × eye
            .alias("validation_id")
        )
        .drop("line")
        .sort(["Calib_index", "eye", "validation_id"])
    )

    return parsed
class Experiment:

    def __init__(self, dataset_path: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(self.dataset_path.name + "_derivatives")
        self.metadata = pl.read_csv(self.dataset_path / "participants.tsv", separator="\t", 
                                    schema_overrides={"subject_id": pl.Utf8, "old_subject_id": pl.Utf8})
        self.subjects = { subject_id:
            Subject(subject_id, old_subject_id, self, 
                     excluded_sessions.get(subject_id, []), excluded_trials.get(subject_id, {}),export_format)
            for subject_id, old_subject_id in zip(self.metadata.select("subject_id").to_series(),
                                                  self.metadata.select("old_subject_id").to_series())
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
        fixations = pl.concat([subject.fixations() for subject in self.subjects.values()])
        saccades = pl.concat([subject.saccades() for subject in self.subjects.values()])

        vis = Visualization(self.derivatives_path, self.detection_algorithm)
        vis.plot_multipanel(fixations, saccades, display)

    def filter_fixations(self, min_fix_dur=50, print_flag=True):
        amount_fix = self.fixations().shape[0]
        for subject in self.subjects.values():
            subject.filter_fixations(min_fix_dur)

        if print_flag:
            print(f"Removed {amount_fix - self.fixations().shape[0]} fixations shorter than {min_fix_dur} ms.")
    def collapse_fixations(self, threshold_px: float, print_flag=True):
        amount_fix = self.fixations().shape[0]
        for subject in self.subjects.values():
            subject.collapse_fixations(threshold_px)
        if print_flag:
            print(f"Removed {amount_fix - self.fixations().shape[0]} fixations that were merged.")

    def drop_trials_with_nan_threshold(self, phase, threshold=0.1,print_flag=True):
        amount_trials_total = self.rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.drop_trials_with_nan_threshold(phase,threshold,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.rts().shape[0]} trials with NaN values.")

    def drop_trials_longer_than(self, seconds,phase, print_flag=True):
        amount_trials_total = self.rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.drop_trials_longer_than(seconds,phase,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.rts().shape[0]} trials longer than {seconds} seconds.")
    
    def plot_scanpaths(self,screen_height,screen_width,display: bool = False):
        with ProcessPoolExecutor(8) as executor:
            futures = [executor.submit(subject.plot_scanpaths,screen_height,screen_width,display) for subject in self.subjects.values()]
            for future in as_completed(futures):
                future.result()

    def drop_poor_or_non_calibrated_trials(self, threshold=1.0, print_flag=True):
        '''
        Drop trials that are not calibrated or have a poor calibration.
        A trial is considered not calibrated if there is no validation data for its calibration index.
        A trial is considered poorly calibrated if the average error is greater than the threshold.
        '''
        amount_trials_total = self.rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.drop_poor_or_non_calibrated_trials(threshold,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.rts().shape[0]} trials with poor calibration.")

    def rts(self):
        rts = [subject.rts() for subject in self.subjects.values()]
        return pl.concat(rts)

    def get_subject(self, subject_id):
        return self.subjects[subject_id]
    
    def get_session(self, subject_id, session_id):
        subject = self.get_subject(subject_id)
        return subject.get_session(session_id)
    
    def get_trial(self, subject_id, session_id, trial_number):
        session = self.get_session(subject_id, session_id)
        return session.get_trial(trial_number)
    
    def fixations(self):
        return pl.concat([subject.fixations() for subject in self.subjects.values()])

    def saccades(self):
        return pl.concat([subject.saccades() for subject in self.subjects.values()])

    def samples(self):
        return pl.concat([subject.samples() for subject in self.subjects.values()])
    
    def remove_subject(self, subject_id):
        if subject_id in self.subjects:
            del self.subjects[subject_id]

    def calib_data(self):
        calib_data = [subject.calib_data() for subject in self.subjects.values()]
        calib_indexes = pl.concat([calib_data[1] for calib_data in calib_data])
        calib_data = pl.concat([calib_data[0] for calib_data in calib_data])
        return calib_data, calib_indexes

    def plot_calib_data(self):
        # Step 0: Load and preprocess
        calib_data = self.calib_data()
        trial_numbers = calib_data[1]
        calib_data = calib_data[0].select([
            "subject_id", "session_id", "Calib_index", "eye", "avg_error", "validation_id"
        ])

        # Step 1: Get only rows with max validation_id per group
        max_vals = (
            calib_data
            .group_by(["subject_id", "session_id", "Calib_index", "eye"])
            .agg(pl.col("validation_id").max().alias("max_validation_id"))
        )

        calib_data = (
            calib_data
            .join(max_vals, on=["subject_id", "session_id", "Calib_index", "eye"])
            .filter(pl.col("validation_id") == pl.col("max_validation_id"))
            .drop(["max_validation_id", "validation_id"])
        )

        # Step 2: Choose best eye (lowest avg_error) per calibration
        best_eyes = (
            calib_data
            .group_by(["subject_id", "session_id", "Calib_index"])
            .agg(pl.col("avg_error").min().alias("best_eye_error"))
        )

        calib_data = (
            calib_data
            .join(best_eyes, on=["subject_id", "session_id", "Calib_index"])
            .filter(pl.col("avg_error") == pl.col("best_eye_error"))
            .drop(["eye", "best_eye_error"])
        )

        # Step 3: Add trial number and clean up
        calib_data = (
            calib_data
            .join(trial_numbers, on=["subject_id", "session_id", "Calib_index"],how="right")
            .drop("Calib_index")
        )
        # Replace nans in avg_error with -1
        calib_data = calib_data.with_columns(
            pl.when(pl.col("avg_error").is_null()).then(-1).otherwise(pl.col("avg_error")).alias("avg_error")
        )

        # Step 4: Combine the columns "subject_id" and "session_id" into a single column
        calib_data = (
            calib_data
            .with_columns(
                (pl.col("subject_id").cast(pl.Utf8) + "_" + pl.col("session_id").cast(pl.Utf8)).alias("subject_id")
            )
            .drop("session_id")
        )
        # Create a copy of the colormap and set 'under' color for -1s
        cmap = colormaps["rocket_r"].copy()
        cmap.set_under("yellow")  # or any color you prefer, e.g., "black", "white"

        heatmap_data = (
            calib_data
            .pivot(
                values="avg_error",
                index="subject_id",
                on="trial_number",
                aggregate_function="first"  # safe if unique per cell
            )
            .sort("subject_id")
            .to_pandas()
            .set_index("subject_id")
        )
        heatmap_data = heatmap_data[sorted(heatmap_data.columns, key=lambda x: int(x))]
        
        # Step 5: Plot with adaptive sizing
        n_subjects = heatmap_data.shape[0]
        n_trials = heatmap_data.shape[1]

        # Define a base size per cell, then scale it
        cell_width = 0.5   # width per trial column
        cell_height = 0.2  # height per subject row

        # Limit extremes so it doesn’t explode with huge data
        fig_width = max(10, min(cell_width * n_trials, 40))
        fig_height = max(8, min(cell_height * n_subjects, 40))

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            center=0.5,
            vmin=0,
            linewidths=0.3,
            linecolor="grey",
            cbar_kws=dict(label="Avg. error (°)")
        )
        plt.xlabel("Trial #", fontsize=14)
        plt.ylabel("Subject", fontsize=14)
        plt.title("Calibration Error per Subject and Trial", fontsize=16)

        # Rotate labels
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, ha="right", va="center", fontsize=10)  # horizontal y labels

        plt.tight_layout()
        plt.show()
        plt.close()

class Subject:

    def __init__(self, subject_id: str, old_subject_id: str, experiment: Experiment,
                 excluded_sessions: list = [], excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.subject_id = subject_id
        self.old_subject_id = old_subject_id
        self.experiment = weakref.ref(experiment)
        self._sessions = None  # Lazy load sessions
        self.excluded_sessions = excluded_sessions
        self.excluded_trials = excluded_trials
        self.subject_dataset_path = self.experiment().dataset_path / f"sub-{self.subject_id}"
        self.subject_derivatives_path = self.experiment().derivatives_path / f"sub-{self.subject_id}"
        self.export_format = export_format

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the parent *id* instead of the weakref itself
        exp = state.pop("experiment", None)
        state["_experiment_id"] = id(exp()) if exp else None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # In a worker the real Experiment instance isn’t available
        # – keep a placeholder or rebuild if you can look it up.
        self.experiment = lambda: None      # callable that returns None

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
        return f"Subject = '{self.subject_id}', " + self.experiment().__repr__()
    
    def __next__(self):
        return next(self.sessions)
    

    def remove_session(self, session_id):
        if self._sessions and session_id in self._sessions:
            del self._sessions[session_id]
            if len(self._sessions) == 0:
                exp = self.experiment()
                if exp:
                    exp.remove_subject(self.subject_id)
                self._sessions = None
                self.experiment = lambda: None
    
    def load_data(self, detection_algorithm: str):
        self.detection_algorithm = detection_algorithm
        for session in self.sessions.values():
            session.load_data(detection_algorithm)


    def filter_fixations(self, min_fix_dur=50):
        for session in self.sessions.values():
            session.filter_fixations(min_fix_dur)

    def collapse_fixations(self, threshold_px: float):
        for session in self.sessions.values():
            session.collapse_fixations(threshold_px)

    def drop_trials_with_nan_threshold(self,phase, threshold=0.1, print_flag=True):
        total_sessions = len(self.sessions)
        amount_trials_total = self.rts().shape[0]
        for session in list(self.sessions.values()):
            session.drop_trials_with_nan_threshold(phase,threshold,False)
        bad_sessions_count = total_sessions - len(self.sessions)


        # If the proportion of bad sessions exceeds the threshold, remove all sessions
        if bad_sessions_count / total_sessions > threshold:
            self.experiment().remove_subject(self.subject_id)
        
        if print_flag:
            print(f"Removed {amount_trials_total - self.rts().shape[0]} trials with NaN values.")

    def drop_poor_or_non_calibrated_trials(self, threshold=1.0, print_flag=True):
        '''
        Drop trials that are not calibrated or have a poor calibration.
        A trial is considered not calibrated if there is no validation data for its calibration index.
        A trial is considered poorly calibrated if the average error is greater than the threshold.
        '''
        amount_trials_total = self.rts().shape[0]
        for session in list(self.sessions.values()):
            session.drop_poor_or_non_calibrated_trials(threshold,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.rts().shape[0]} trials with poor calibration.")

    def drop_trials_longer_than(self, seconds,phase, print_flag=True):
        amount_trials_total = self.rts().shape[0]
        for session in list(self.sessions.values()):
            session.drop_trials_longer_than(seconds,phase,False)
        if print_flag:
            print(f"Removed {amount_trials_total - self.rts().shape[0]} trials longer than {seconds} seconds.")

    def plot_scanpaths(self,screen_height,screen_width, display: bool = False):
        for session in self.sessions.values():
            session.plot_scanpaths(screen_height,screen_width,display)
 
    def rts(self):
        rts = [session.rts() for session in self.sessions.values()]
        rts = pl.concat(rts).with_columns([
            (pl.lit(self.subject_id)).alias("subject_id"),])
        return rts

    def get_session(self, session_id):
        return self.sessions[session_id]

    def get_trial(self, session_id, trial_number):
        session = self.get_session(session_id)
        return session.get_trial(trial_number)
    
    
    def fixations(self):
        df = pl.concat([session.fixations() for session in self.sessions.values()]).with_columns([
            (pl.lit(self.subject_id)).alias("subject_id"),])
        return df
    
    def saccades(self):
        df = pl.concat([session.saccades() for session in self.sessions.values()]).with_columns([
            (pl.lit(self.subject_id)).alias("subject_id"),])
        return df

    def samples(self):
        df = pl.concat([session.samples() for session in self.sessions.values()]).with_columns([
            (pl.lit(self.subject_id)).alias("subject_id"),])
        return df

    def calib_data(self):
        calib_data = [session.calib_data() for session in self.sessions.values()]
        calib_indexes = pl.concat([calib_data[1] for calib_data in calib_data]).with_columns([
            (pl.lit(self.subject_id)).alias("subject_id"),])
        calib_data = pl.concat([calib_data[0] for calib_data in calib_data]).with_columns([
            (pl.lit(self.subject_id)).alias("subject_id"),])
        return calib_data, calib_indexes

class Session():
    
    def __init__(self, session_id: str, subject: Subject, excluded_trials: list = [],export_format = FEATHER_EXPORT):
        self.session_id = session_id
        self.subject = weakref.ref(subject)
        self.excluded_trials = excluded_trials
        self.session_dataset_path = self.subject().subject_dataset_path / f"ses-{self.session_id}"
        self.session_derivatives_path = self.subject().subject_derivatives_path / f"ses-{self.session_id}"
        self._trials = None  # Lazy load trials
        self.export_format = export_format

        if not self.session_derivatives_path.exists():
            raise FileNotFoundError(f"Session path not found: {self.session_derivatives_path}")
        
    def __getstate__(self):
        state = self.__dict__.copy()
        sess_parent = state.pop("subject", None)
        state["_subject_id"] = id(sess_parent()) if sess_parent else None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.subject = lambda: None

    @property
    def trials(self):
        if self._trials is None:
            raise ValueError("Trials not loaded. Please load data first.")
        return self._trials

    def __repr__(self):
        return f"Session = '{self.session_id}', " + self.subject().__repr__()
    
    def drop_trials_with_nan_threshold(self, phase, threshold=0.1, print_flag=True):
        total_trials = len(self.trials)
        # Filter bad trials

        bad_trials = [trial for trial in self.trials.keys() if self.trials[trial].is_trial_bad(phase, threshold)]
        if len(bad_trials)/total_trials > threshold:
            self.subject().remove_session(self.session_id)

        
        if print_flag:
            print(f"Removed {len(bad_trials)} trials with NaN values.")
    
    def drop_poor_or_non_calibrated_trials(self, threshold=1.0, print_flag=True):
        '''
        Drop trials that are not calibrated or have a poor calibration.
        A trial is considered not calibrated if there is no validation data for its calibration index.
        A trial is considered poorly calibrated if the average error is greater than the threshold.
        '''
        trial_numbers = [trial for trial in self.trials.keys()]
        # Step 1: Get only rows with max validation_id per group
        calib_data, trial_numbers = self.calib_data()
        calib_data = calib_data.drop("session_id")
        max_vals = (
            calib_data
            .group_by(["Calib_index", "eye"])
            .agg(pl.col("validation_id").max().alias("max_validation_id"))
        )

        calib_data = (
            calib_data
            .join(max_vals, on=["Calib_index", "eye"])
            .filter(pl.col("validation_id") == pl.col("max_validation_id"))
            .drop(["max_validation_id", "validation_id"])
        )

        # Step 2: Choose best eye (lowest avg_error) per calibration
        best_eyes = (
            calib_data
            .group_by(["Calib_index"])
            .agg(pl.col("avg_error").min().alias("best_eye_error"))
        )

        calib_data = (
            calib_data
            .join(best_eyes, on=["Calib_index"])
            .filter(pl.col("avg_error") == pl.col("best_eye_error"))
            .drop(["eye", "best_eye_error"])
        )

        calib_data = (
            calib_data
            .join(trial_numbers, on=["Calib_index"],how="right")
            .drop("Calib_index")
        )
        # Bad trials are those with avg_error > threshold, or those that have NaN values in avg_error
        bad_trials = calib_data.filter((pl.col("avg_error") > threshold) | (pl.col("avg_error").is_null())).select("trial_number").to_series().unique().to_list()

        for trial in bad_trials:
            self.remove_trial(trial)
        
        if print_flag:
            print(f"Removed {len(bad_trials)} trials with poor calibration.")

    def drop_trials_longer_than(self, seconds,phase, print_flag=True):

        # Filter bad trials

        bad_trials = [trial for trial in self.trials.keys() if self.trials[trial].is_trial_longer_than(seconds,phase)]
        for trial in bad_trials:
            self.remove_trial(trial)
                      
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
        self._calib_data = _parse_validations(exporter.read(self.session_derivatives_path, "calib")) if (self.session_derivatives_path / ("calib" + file_extension)).exists() else None
   
        # Initialize trials
        self._init_trials(samples,fix,sacc,blink,events_path)

    def calib_data(self):
        if self._calib_data is None:
            raise ValueError(f"Calibration data for session {self.session_id} and subject {self.subject().subject_id} not loaded. Please load data first.")
        
        calib_indexes = [(trial.trial_number,trial.calib_index) for trial in self.trials.values() if trial.calib_index is not None]
        calib_indexes = pl.DataFrame(calib_indexes, schema=["trial_number","Calib_index"],orient="row").with_columns([
            (pl.lit(self.session_id)).alias("session_id")])
        return self._calib_data.with_columns([
            (pl.lit(self.session_id)).alias("session_id")]), calib_indexes

    def _init_trials(self,samples,fix,sacc,blink,events_path):
        cosas = [trial for trial in samples.select("trial_number").to_series().unique() if trial != -1 and trial not in self.excluded_trials]
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

    def filter_fixations(self, min_fix_dur=50):
        for trial in self.trials.values():
            trial.filter_fixations(min_fix_dur)

    def collapse_fixations(self, threshold_px: float):
        for trial in self.trials.values():
            trial.collapse_fixations(threshold_px)


    def rts(self):
        rts = [trial.rts() for trial in self.trials.values()]
        rts = pl.concat(rts).with_columns([
            (pl.lit(self.session_id)).alias("session_id"),])
        return rts
    

    def fixations(self):
        df = pl.concat([trial.fixations() for trial in self.trials.values()]).with_columns([
            (pl.lit(self.session_id)).alias("session_id"),])
        return df

    def saccades(self):
        df = pl.concat([trial.saccades() for trial in self.trials.values()]).with_columns([
            (pl.lit(self.session_id)).alias("session_id"),])
        return df
        

    def samples(self):
        df = pl.concat([trial.samples() for trial in self.trials.values()]).with_columns([
            (pl.lit(self.session_id)).alias("session_id"),])
        return df

    def remove_trial(self, trial_number):
        if self._trials and trial_number in self._trials:
            del self._trials[trial_number]
            if len(self._trials) == 0:
                subj = self.subject()
                if subj:
                    subj.remove_session(self.session_id)
                self._trials = None
                self.subject = lambda: None
class Trial:

    def __init__(self, trial_number: int, session: Session, samples: pl.DataFrame, fix: pl.DataFrame, 
                sacc: pl.DataFrame, blink: pl.DataFrame, events_path: Path):
        self.trial_number = trial_number
        self.session = session

        # Filter per trial
        # If "Calib_index" is a column in samples, set self._calib_index to the value of that column
        
        self._calib_index = samples.filter(pl.col("trial_number") == trial_number).select("Calib_index").to_series()[0] if "Calib_index" in samples.columns else None
        

        self._samples = samples.filter(pl.col("trial_number") == trial_number).drop("Calib_index",strict=False)
        self._fix = fix.filter(pl.col("trial_number") == trial_number).drop("Calib_index",strict=False)
        self._sacc = sacc.filter(pl.col("trial_number") == trial_number).drop("Calib_index",strict=False)
        self._blink = blink.filter(pl.col("trial_number") == trial_number).drop("Calib_index",strict=False) if blink is not None else None

        # Get the start time
        start_time = self._samples.select("tSample").to_series()[0]

        # Time normalization
        self._samples = self._samples.with_columns([
            (pl.col("tSample") - start_time).alias("tSample")
        ])

        self._fix = self._fix.with_columns([
            (pl.col("tStart") - start_time).alias("tStart"),
            (pl.col("tEnd") - start_time).alias("tEnd")
        ])

        self._sacc = self._sacc.with_columns([
            (pl.col("tStart") - start_time).alias("tStart"),
            (pl.col("tEnd") - start_time).alias("tEnd")
        ])

        if self._blink is not None:
            self._blink = self._blink.with_columns([
                (pl.col("tStart") - start_time).alias("tStart"),
                (pl.col("tEnd") - start_time).alias("tEnd")
            ])

        self.events_path = events_path
        self.detection_algorithm = events_path.name[:-7]

 
    def fixations(self):
        return self._fix
    
    @property
    def calib_index(self):
        return self._calib_index
    

    def saccades(self):
        return self._sacc
    

    def samples(self):
        return self._samples

    def __repr__(self):
        return f"Trial = '{self.trial_number}', " + self.session.__repr__()

    def plot_scanpath(self,screen_height,screen_width, **kwargs):
        vis = Visualization(self.events_path, self.detection_algorithm)
        (self.events_path / "plots").mkdir(parents=True, exist_ok=True)
        vis.scanpath(fixations=self._fix, saccades=self._sacc, samples=self._samples, screen_height=screen_height, screen_width=screen_width, 
                      folder_path=self.events_path / "plots", **kwargs)

    def plot_animation(self, screen_height, screen_width, video_path=None, background_image_path=None, **kwargs):
        """
        Create an animated visualization of eye-tracking data for this trial.

        When a video is provided, the animation syncs gaze samples with video frames.
        When no video is provided, gaze points are animated on a grey background or
        a provided background image, using the sample timestamps for timing.

        Parameters
        ----------
        screen_height, screen_width
            Stimulus resolution in pixels.
        video_path
            Path to a video file. If provided, gaze is overlaid on video frames.
        background_image_path
            Path to a background image. Only used when video_path is None.
            If both are None, a grey background is used.
        **kwargs
            Additional arguments passed to Visualization.plot_animation():
            - folder_path: Directory to save the animation
            - tmin, tmax: Time window in ms
            - seconds_to_show: Limit animation to first N seconds
            - scale_factor: Resolution scaling (default 0.5)
            - gaze_radius: Gaze point radius in pixels
            - gaze_color: RGB tuple for gaze color
            - fps: Animation frames per second
            - output_format: "html" (default), "mp4", "gif", or "matplotlib"
            - display: If True, return HTML for notebook display

        Returns
        -------
        IPython.display.HTML or None
            Returns HTML animation if display=True and output_format="html".
            For output_format="matplotlib", displays in a GUI window and returns None.
        """
        vis = Visualization(self.events_path, self.detection_algorithm)
        (self.events_path / "plots").mkdir(parents=True, exist_ok=True)
        
        return vis.plot_animation(
            samples=self._samples,
            screen_height=screen_height,
            screen_width=screen_width,
            video_path=video_path,
            background_image_path=background_image_path,
            **kwargs
        )

    def filter_fixations(self, min_fix_dur: int = 50):
        """
        1.  Delete fixations shorter than `min_fix_dur` (ms).
        2.  Merge the two saccades that flank each deleted fixation
            into one longer saccade, always staying inside a single
            (“phase”, “eye”) stream.

        Returns
        -------
        self   # so you can do:  trial.filter_fixations().is_trial_bad()
        """
        # ─────────────────────── 0 · split keep / drop ──────────────────────
        short_fix = self._fix.filter(pl.col("duration") < min_fix_dur)
        keep_fix  = self._fix.filter(pl.col("duration") >= min_fix_dur)

        if short_fix.is_empty():
            return                                # nothing to do

        # ─────────────────────── 1 · prepare saccades ───────────────────────
        sacc = (self._sacc       # add an integer key that survives every shuffle
                .with_row_count("idx")
                .sort(["phase", "eye", "tStart"]))

        prev_src = sacc.select(["idx", "phase", "eye",
                                pl.col("tEnd").alias("t")])
        next_src = sacc.select(["idx", "phase", "eye",
                                pl.col("tStart").alias("t")])

        # ─────────────────────── 2 · find neighbour IDs ─────────────────────
        short_fix = short_fix.rename({"tStart": "tStart_fix",
                                    "tEnd":   "tEnd_fix"})



        short_fix = short_fix.sort(["phase", "eye", "tStart_fix"])
        prev_src  = prev_src.sort(["phase", "eye", "t"])
        next_src  = next_src.sort(["phase", "eye", "t"])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sortedness of columns cannot be checked when 'by' groups provided",
                category=UserWarning,
            )

            short_fix = (
                short_fix
                .join_asof(
                    prev_src,
                    left_on="tStart_fix",
                    right_on="t",
                    by=["phase", "eye"],
                    strategy="backward"
                )
                .rename({"idx": "idx_prev"})
                .drop("t")
                .join_asof(
                    next_src,
                    left_on="tEnd_fix",
                    right_on="t",
                    by=["phase", "eye"],
                    strategy="forward"
                )
                .rename({"idx": "idx_next"})
                .drop("t")
            )

        # only keep rows where we found BOTH neighbours
        short_fix_pairs = short_fix.select(["idx_prev", "idx_next"]).drop_nulls()
        if short_fix_pairs.is_empty():
            # we could not build any (prev,next) pair → only delete fixations
            self._fix = keep_fix.sort(["phase", "tStart"])
            return self

        # ───────────────────── 3 · join the two saccades ────────────────────
        pair_df = (short_fix_pairs.unique()
                .join(sacc, left_on="idx_prev", right_on="idx", how="inner")
                .join(sacc, left_on="idx_next", right_on="idx", suffix="_nxt"))

        # keep **prev** row plus ONLY the four _nxt columns that we still need
        prev_cols = [c for c in pair_df.columns if not c.endswith("_nxt")]
        need_nxt  = ["tEnd_nxt", "xEnd_nxt", "yEnd_nxt", "vPeak_nxt"]
        merged = pair_df.select(prev_cols + need_nxt)

        # ───────── overwrite / derive fields that span both flanks ──────────
        merged = merged.with_columns([
            pl.col("tEnd_nxt").alias("tEnd"),
            (pl.col("tEnd_nxt") - pl.col("tStart")).alias("duration"),
            pl.col("xEnd_nxt").alias("xEnd"),
            pl.col("yEnd_nxt").alias("yEnd"),
            pl.max_horizontal("vPeak", "vPeak_nxt").alias("vPeak"),
            (
                (pl.col("xEnd_nxt") - pl.col("xStart"))**2
            + (pl.col("yEnd_nxt") - pl.col("yStart"))**2
            ).sqrt().alias("ampDeg"),
        ])

        # drop helper columns that end in _nxt (no longer needed)
        merged = merged.drop([c for c in merged.columns if c.endswith("_nxt")])

        # 4 · bring schema in line with original  --------------------------------
        base_cols = sacc.drop("idx").columns

        for col in base_cols:
            if col not in merged.columns:
                if f"{col}_nxt" in pair_df.columns:
                    merged = merged.with_columns(pl.col(f"{col}_nxt").alias(col))
                else:
                    merged = merged.with_columns(
                        pl.lit(None).cast(sacc[col].dtype).alias(col)
                    )

        # --- NEW: make sure every dtype matches the canonical sacc table ----
        for col in base_cols:
            if merged[col].dtype != sacc[col].dtype:
                merged = merged.with_columns(pl.col(col).cast(sacc[col].dtype))

        merged = merged.select(base_cols)

        # ───────────────────── 5 · build the final saccade table ────────────
        to_drop = pl.concat([short_fix_pairs["idx_prev"],
                            short_fix_pairs["idx_next"]]).unique()
        new_sacc = (sacc
                    .filter(~pl.col("idx").is_in(to_drop))
                    .drop("idx")          # helper column gone
                    .vstack(merged)       # add fused rows
                    .sort(["phase", "eye", "tStart"]))

        # ───────────────────── 6 · store back and return ────────────────────
        self._fix  = keep_fix.sort(["phase", "tStart"])
        self._sacc = new_sacc
        


    def collapse_fixations(self, threshold_px: float) -> None:
        """
        Collapse consecutive fixations that lie ≤ `threshold_px` apart
        *within each phase separately*.  Saccades whose whole time‑span
        falls between the first and last fixation of a pool are discarded.
        The saccade immediately before the pool has its (xEnd, yEnd)
        adjusted to the merged‑fixation centroid; the saccade immediately
        after the pool has its (xStart, yStart) adjusted likewise.

        After running:
            self._fix   → collapsed fixations
            self._sacc  → original saccades minus the discarded ones,
                        plus the updated coordinates for the two
                        bordering saccades.
        """

        # ────────────────── 0 · prepare helpers ──────────────────
        fix = self._fix.sort("tStart").with_row_count("fix_idx")
        sac = self._sacc.sort("tStart").with_row_count("sac_idx")

        new_fix_rows: list[dict] = []
        drop_sac_idx: set[int]   = set()
        mod_sac: dict[int, dict] = {}          # idx → partial‑row updates

        # ────────────────── 1 · loop over phases ─────────────────
        for phase_val in fix["phase"].unique():               # ① per phase
            # Loop over eyes if needed
            for eye in fix["eye"].unique():
                fix_p = fix.filter((pl.col("phase") == phase_val) & (pl.col("eye") == eye))
                sac_p = sac.filter((pl.col("phase") == phase_val) & (pl.col("eye") == eye))

                i, n_fix = 0, len(fix_p)
                while i < n_fix:

                    # ── grow one pool ───────────────────────────────
                    pool = [fix_p.row(i, named=True)]
                    j = i + 1
                    while j < n_fix:
                        dx = fix_p["xAvg"][j] - fix_p["xAvg"][j - 1]
                        dy = fix_p["yAvg"][j] - fix_p["yAvg"][j - 1]
                        if hypot(dx, dy) <= threshold_px:
                            pool.append(fix_p.row(j, named=True))
                            j += 1
                        else:
                            break

                    # ── pool of size 1: keep as‑is ──────────────────
                    if len(pool) == 1:
                        new_fix_rows.append(pool[0].copy())        # unchanged
                        i = j
                        continue

                    # ── merge the pool (>1 fix) ─────────────────────
                    first_fix, last_fix = pool[0], pool[-1]

                    merged_fix = first_fix.copy()
                    merged_fix.update({
                        "tEnd":     last_fix["tEnd"],
                        "duration": sum(f["duration"] for f in pool),
                        "xAvg":     np.mean([f["xAvg"] for f in pool]),
                        "yAvg":     np.mean([f["yAvg"] for f in pool]),
                        "pupilAvg": np.mean([f["pupilAvg"] for f in pool]),
                    })
                    new_fix_rows.append(merged_fix)

                    # ── identify & drop fully‑internal saccades ─────
                    inside = sac_p.filter(
                        (pl.col("tStart") >= first_fix["tEnd"]) &
                        (pl.col("tEnd")   <= last_fix["tStart"])
                    )
                    drop_sac_idx.update(inside["sac_idx"].to_list())

                    # ── adjust bordering saccades ───────────────────
                    merged_x = merged_fix["xAvg"]
                    merged_y = merged_fix["yAvg"]

                    # previous saccade (ends at first_fix.tStart)
                    prev_df = sac_p.filter(pl.col("tEnd") <= first_fix["tStart"]).tail(1)
                    if prev_df.height:
                        prev = prev_df.row(0, named=True)
                        idx  = prev["sac_idx"]
                        upd  = {
                            "xEnd": merged_x,
                            "yEnd": merged_y,
                            "dx":   merged_x - prev["xStart"],
                            "dy":   merged_y - prev["yStart"],
                        }
                        upd["amplitude"] = hypot(upd["dx"], upd["dy"])
                        mod_sac.setdefault(idx, {}).update(upd)

                    # next saccade (starts at last_fix.tEnd)
                    next_df = sac_p.filter(pl.col("tStart") >= last_fix["tEnd"]).head(1)
                    if next_df.height:
                        nxt = next_df.row(0, named=True)
                        idx = nxt["sac_idx"]
                        upd = {
                            "xStart": merged_x,
                            "yStart": merged_y,
                            "dx":     nxt["xEnd"] - merged_x,
                            "dy":     nxt["yEnd"] - merged_y,
                        }
                        upd["amplitude"] = hypot(upd["dx"], upd["dy"])
                        mod_sac.setdefault(idx, {}).update(upd)

                    i = j                                         # advance

        # ────────────────── 2 · rebuild tables ──────────────────
        # 2‑a  fixations
        new_fix = (
            pl.DataFrame(new_fix_rows,
                        schema=fix.drop("fix_idx").schema,
                        orient="row")
            .sort(["phase", "tStart"])
        )

        # 2‑b  saccades: drop + modify in one pass
        new_sac_rows = []
        for row in sac.iter_rows(named=True):
            idx = row["sac_idx"]
            if idx in drop_sac_idx:
                continue                                     # discard
            if idx in mod_sac:                               # apply edits
                row.update(mod_sac[idx])
                # re‑compute amplitude in case only dx/dy were provided
                if "amplitude" not in mod_sac[idx]:
                    row["amplitude"] = hypot(row["dx"], row["dy"])
            new_sac_rows.append({k: v for k, v in row.items() if k != "sac_idx"})

        new_sac = (
            pl.DataFrame(new_sac_rows,
                        schema=sac.drop("sac_idx").schema,
                        orient="row")
            .sort(["phase", "tStart"])
        )

        # ────────────────── 3 · store back ──────────────────────
        self._fix  = new_fix
        self._sacc = new_sac

    def save_rts(self):
        if hasattr(self, "_rts"):
            return

        # Filter out empty phase rows
        filtered = self._samples.filter(pl.col("phase") != "")

        # Calculate RT as the difference between last and first tSample per phase
        rts = (
            filtered
            .group_by("phase")
            .agg([
                (pl.col("tSample").max() - pl.col("tSample").min()).alias("rt")
            ])
            .with_columns([
                pl.lit(self.trial_number).alias("trial_number")
            ])
        )

        self._rts = rts


    def rts(self):
        if not hasattr(self, "_rts"):
            self.save_rts()
        return self._rts
    
    def is_trial_bad(self, phase, threshold=0.1):
        # Filter samples for the given phase
        samples = self._samples.filter(pl.col("phase") == phase)

        # Remove samples during blinks
        if self._blink is not None and self._blink.height > 0:
            for blink in self._blink.iter_rows(named=True):
                start, end = blink["tStart"], blink["tEnd"]
                samples = samples.filter(~((pl.col("tSample") > start) & (pl.col("tSample") < end)))

        total_samples = samples.height
        if total_samples == 0:
            return True  # If no samples remain, consider it bad

        # Count total NaNs across all columns
        nan_counts = samples.select([pl.col(c).is_null().sum().alias(c) for c in samples.columns])
        nan_total = sum(nan_counts.row(0))

        # Count "bad" values
        bad_values = samples.select(pl.col("bad").sum()).item()

        bad_and_nan_percentage = (nan_total + bad_values) / total_samples

        return bad_and_nan_percentage > threshold

    
    def is_trial_longer_than(self, seconds, phase):
        rt_row = self.rts().filter(pl.col("phase") == phase)
        if rt_row.is_empty():
            return False  # Or True if no data should be considered long
        return rt_row.select("rt").item() > seconds * 1000.0

    def compute_multimatch(self,other_trial: "Trial",screen_height,screen_width):
        trial_scanpath = self.search_fixations().to_pandas()
        trial_to_compare_scanpath = other_trial.search_fixations().to_pandas()
        # Turn trial scanpath into list of tuples
        trial_scanpath = [tuple(row) for row in trial_scanpath[["xAvg", "yAvg", "duration"]].values]
        trial_to_compare_scanpath = [tuple(row) for row in trial_to_compare_scanpath[["xAvg", "yAvg", "duration"]].values]

        # Convert the list of tuples into a numpy array with the format needed for the multimatch function
        trial_scanpath = np.array(trial_scanpath, dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
        trial_to_compare_scanpath = np.array(trial_to_compare_scanpath, dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

        return mm.docomparison(trial_scanpath, trial_to_compare_scanpath, (screen_width, screen_height))