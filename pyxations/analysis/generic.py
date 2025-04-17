from pathlib import Path
import polars as pl
from pyxations.visualization.visualization import Visualization
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
from pyxations.export import FEATHER_EXPORT, get_exporter
from math import hypot
import numpy as np

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
        self.metadata = pl.read_csv(self.dataset_path / "participants.tsv", separator="\t", 
                                    dtypes={"subject_id": pl.Utf8, "old_subject_id": pl.Utf8})
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


    def filter_fixations(self, min_fix_dur=50):
        for session in self.sessions.values():
            session.filter_fixations(min_fix_dur)

    def collapse_fixations(self, threshold_px: float):
        for session in self.sessions.values():
            session.collapse_fixations(threshold_px)

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

    def get_rts(self):
        rts = [trial.get_rts() for trial in self.trials.values()]
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
        del self._trials[trial_number]

class Trial:

    def __init__(self, trial_number: int, session: Session, samples: pl.DataFrame, fix: pl.DataFrame, 
                sacc: pl.DataFrame, blink: pl.DataFrame, events_path: Path):
        self.trial_number = trial_number
        self.session = session

        # Filter per trial
        self._samples = samples.filter(pl.col("trial_number") == trial_number)
        self._fix = fix.filter(pl.col("trial_number") == trial_number)
        self._sacc = sacc.filter(pl.col("trial_number") == trial_number)
        self._blink = blink.filter(pl.col("trial_number") == trial_number) if blink is not None else None

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

        short_fix = (short_fix
                    .join_asof(prev_src,
                                left_on="tStart_fix", right_on="t",
                                by=["phase", "eye"],
                                strategy="backward")
                    .rename({"idx": "idx_prev"})
                    .drop("t")
                    .join_asof(next_src,
                                left_on="tEnd_fix", right_on="t",
                                by=["phase", "eye"],
                                strategy="forward")
                    .rename({"idx": "idx_next"})
                    .drop("t"))

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
            fix_p = fix.filter(pl.col("phase") == phase_val)
            sac_p = sac.filter(pl.col("phase") == phase_val)

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
        if hasattr(self, "rts"):
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

        self.rts = rts


    def get_rts(self):
        if not hasattr(self, "rts"):
            self.save_rts()
        return self.rts
    
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
        rt_row = self.get_rts().filter(pl.col("phase") == phase)
        if rt_row.is_empty():
            return False  # Or True if no data should be considered long
        return rt_row.select("rt").item() > seconds * 1000.0
