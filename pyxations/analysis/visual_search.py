from pathlib import Path
import polars as pl
from pyxations.visualization.visualization import Visualization
from pyxations.export import FEATHER_EXPORT
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pyxations.analysis.generic import Experiment, Subject, Session, Trial, _find_fixation_cutoff, STIMULI_FOLDER, ITEMS_FOLDER

def _as(obj, typ):
    if isinstance(obj, typ): return obj
    return ast.literal_eval(obj)
class VisualSearchExperiment(Experiment):
    def __init__(self, dataset_path: str,search_phase_name: str,memorization_phase_name: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(self.dataset_path.name + "_derivatives")
        self.metadata = pl.read_csv(self.dataset_path / "participants.tsv", separator="\t", 
                                    dtypes={"subject_id": pl.Utf8, "old_subject_id": pl.Utf8})
        self.subjects = { subject_id:
            VisualSearchSubject(subject_id, old_subject_id, self, search_phase_name, memorization_phase_name,
                     excluded_sessions.get(subject_id, []), excluded_trials.get(subject_id, {}),export_format)
            for subject_id, old_subject_id in zip(self.metadata["subject_id"], self.metadata["old_subject_id"])
            if subject_id not in excluded_subjects and old_subject_id not in excluded_subjects
        }
        self.export_format = export_format
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name

    def accuracy(self):
        accuracy = pl.concat([subject.accuracy() for subject in self.subjects.values()])

        return accuracy
    
    def plot_accuracy_by_subject(self):
        
        correct_responses = self.search_rts()
        # Sort by the sum of correct responses of each subject
        correct_responses_aux = (
            correct_responses
            .group_by(["subject_id", "memory_set_size", "target_present"])
            .agg(pl.col("correct_response").mean().alias("correct_response_mean"))
        ).select(["subject_id", "memory_set_size", "target_present", "correct_response_mean"])
        # Merge the correct_responses with the correct_responses_aux
        correct_responses = correct_responses.join(
            correct_responses_aux,
            on=["subject_id", "memory_set_size", "target_present"],
            how="left"
        ).sort(by=["memory_set_size", "target_present","correct_response_mean"])

        correct_responses = correct_responses.to_pandas()
        # target present to bool
        correct_responses["target_present"] = correct_responses["target_present"].astype(bool)
        # There should be an ax for each memory set size

        mem_set_sizes = correct_responses["memory_set_size"].unique()
        mem_set_sizes.sort()

        width_size = max(0.25 * len(correct_responses["subject_id"].unique()),10)

        n_rows = len(mem_set_sizes)
        fig, axs = plt.subplots(n_rows, 1, figsize=(width_size, 5 * n_rows),sharey=True)

        if n_rows == 1:
            axs = np.array([axs])

        for i, row in enumerate(mem_set_sizes):
            data = correct_responses[(correct_responses["memory_set_size"] == row)]
            sns.lineplot(x='subject_id',y='correct_response',data=data,hue='target_present',errorbar='se',ax=axs[i],estimator='mean')
            axs[i].set_title(f"Memory Set Size {row}")
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].set_xlabel("Subject ID")
            axs[i].set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()
        plt.close()
    
    def plot_accuracy_by_stimulus(self):
        # Convert to pandas for Seaborn
        correct_responses = self.search_rts()
        correct_responses_aux = (
            correct_responses
            .group_by(["stimulus", "memory_set_size", "target_present"])
            .agg(pl.col("correct_response").mean().alias("correct_response_mean"))
        ).select(["stimulus", "memory_set_size", "target_present", "correct_response_mean"])

        # Merge the correct_responses with the correct_responses_aux
        correct_responses = correct_responses.join(
            correct_responses_aux,
            on=["stimulus", "memory_set_size", "target_present"],
            how="left"
        ).to_pandas().sort_values(by=["memory_set_size", "target_present", "correct_response_mean"])

        # Convert target_present to bool (in case it's int 0/1)
        correct_responses["target_present"] = correct_responses["target_present"].astype(bool)

        # One subplot per memory set size
        mem_set_sizes = sorted(correct_responses["memory_set_size"].unique())
        n_rows = len(mem_set_sizes)

        width_size = max(0.25 * len(correct_responses["stimulus"].unique()),10)

        fig, axs = plt.subplots(n_rows, 1, figsize=(width_size, 5 * n_rows), sharey=True)

        if n_rows == 1:
            axs = np.array([axs])

        for i, mem_size in enumerate(mem_set_sizes):
            data = correct_responses[correct_responses["memory_set_size"] == mem_size]
            sns.lineplot(x='stimulus',y='correct_response',data=data,hue='target_present',errorbar='se',ax=axs[i],estimator='mean')
            axs[i].set_title(f"Memory Set Size {mem_size}")
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].set_xlabel("Stimulus")
            axs[i].set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()
        plt.close()

    def search_rts(self):
        rts = self.rts().filter(pl.col("phase") == self._search_phase_name)
        return rts
    
    def search_saccades(self):
        saccades = self.saccades().filter(pl.col("phase") == self._search_phase_name)
        return saccades

    def search_fixations(self):
        fixations = self.fixations().filter(pl.col("phase") == self._search_phase_name)
        return fixations

    def plot_speed_accuracy_tradeoff_by_subject(self):
        # 1) Aggregate the data
        speed_accuracy = (
            self.search_rts()
            .group_by(["target_present", "memory_set_size", "subject_id"])
            .agg([
                pl.col("rt").mean().alias("rt"),
                pl.col("correct_response").mean().alias("accuracy")
            ])
            .with_columns([
                pl.col("rt") / 1000,  # Convert to seconds
                pl.col("target_present").cast(pl.Boolean)
            ])
            .sort("memory_set_size")
        ).to_pandas()

        # 2) Unique memory set sizes
        mem_set_sizes = np.sort(speed_accuracy["memory_set_size"].unique())
        n_rows = len(mem_set_sizes)

        # 3) Prepare grid layout
        fig = plt.figure(figsize=(6, 1 + 6 * n_rows))
        gs = fig.add_gridspec(
            2 * n_rows, 2,
            width_ratios=(4, 1),
            height_ratios=[1, 4] * n_rows,
            left=0.1, right=0.9, bottom=0.07, top=0.85,
            wspace=0.05, hspace=0.05
        )

        # 4) Loop over memory set sizes
        for i, mem_size in enumerate(mem_set_sizes):
            data = speed_accuracy[speed_accuracy["memory_set_size"] == mem_size]

            row_top = 2 * i
            row_bottom = 2 * i + 1

            ax = fig.add_subplot(gs[row_bottom, 0])
            ax_histx = fig.add_subplot(gs[row_top, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[row_bottom, 1], sharey=ax)

            # (A) Scatter plot
            sns.scatterplot(
                x="accuracy",
                y="rt",
                data=data,
                hue="target_present",
                ax=ax,
                palette="deep"
            )

            # (B) Connection lines per subject
            for subj_id in data["subject_id"].unique():
                subj_data = data[data["subject_id"] == subj_id]
                if len(subj_data) != 2:
                    continue
                p0 = subj_data[subj_data["target_present"] == False]
                p1 = subj_data[subj_data["target_present"] == True]
                if not p0.empty and not p1.empty:
                    ax.plot(
                        [p0["accuracy"].values[0], p1["accuracy"].values[0]],
                        [p0["rt"].values[0], p1["rt"].values[0]],
                        color="black", alpha=0.3, linewidth=0.5, zorder=0
                    )

            # (C) Marginal histograms
            ax_histx.hist(data["accuracy"], bins=np.linspace(0, 1, 21), color="gray")
            ax_histy.hist(data["rt"], bins=20, orientation='horizontal', color="gray")

            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)

            ax_histx.set_title(f"Memory Set Size {mem_size}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, speed_accuracy["rt"].max() * 1.1)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Mean RT (s)")

        plt.suptitle("Speed-Accuracy Tradeoff by Subject", fontsize=14)
        plt.show()
        plt.close()

    def plot_speed_accuracy_tradeoff_by_stimulus(self):
        # 1) Aggregate the data
        speed_accuracy = (
            self.search_rts()
            .group_by(["target_present", "memory_set_size", "stimulus"])
            .agg([
                pl.col("rt").mean().alias("rt"),
                pl.col("correct_response").mean().alias("accuracy")
            ])
            .with_columns([
                (pl.col("rt") / 1000).alias("rt"),  # convert ms → s
                pl.col("target_present").cast(pl.Boolean)
            ])
            .sort("memory_set_size")
            .to_pandas()
        )

        # 2) Unique memory set sizes
        mem_set_sizes = np.sort(speed_accuracy["memory_set_size"].unique())
        n_rows = len(mem_set_sizes)

        # 3) Prepare grid layout
        fig = plt.figure(figsize=(6, 1 + 6 * n_rows))
        gs = fig.add_gridspec(
            2 * n_rows, 2,
            width_ratios=(4, 1),
            height_ratios=[1, 4] * n_rows,
            left=0.1, right=0.9, bottom=0.07, top=0.85,
            wspace=0.05, hspace=0.05
        )

        # 4) Loop over memory set sizes
        for i, mem_size in enumerate(mem_set_sizes):
            data = speed_accuracy[speed_accuracy["memory_set_size"] == mem_size]

            row_top = 2 * i
            row_bottom = 2 * i + 1

            ax = fig.add_subplot(gs[row_bottom, 0])
            ax_histx = fig.add_subplot(gs[row_top, 0], sharex=ax)
            ax_histy = fig.add_subplot(gs[row_bottom, 1], sharey=ax)

            # (A) Main scatter plot
            sns.scatterplot(
                x="accuracy",
                y="rt",
                data=data,
                hue="target_present",
                ax=ax,
                palette="deep"
            )

            # (B) Connect stimulus points (False → True)
            for stim in data["stimulus"].unique():
                stim_data = data[data["stimulus"] == stim]
                if len(stim_data) != 2:
                    continue
                p0 = stim_data[stim_data["target_present"] == False]
                p1 = stim_data[stim_data["target_present"] == True]
                if not p0.empty and not p1.empty:
                    ax.plot(
                        [p0["accuracy"].values[0], p1["accuracy"].values[0]],
                        [p0["rt"].values[0], p1["rt"].values[0]],
                        color="black", alpha=0.3, linewidth=0.5, zorder=0
                    )

            # (C) Marginal histograms
            ax_histx.hist(data["accuracy"], bins=np.linspace(0, 1, 21), color="gray")
            ax_histy.hist(data["rt"], bins=20, orientation='horizontal', color="gray")

            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)

            # (D) Titles, limits, labels
            ax_histx.set_title(f"Memory Set Size {mem_size}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, speed_accuracy["rt"].max() * 1.1)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Mean RT (s)")

        # 5) Final touch
        plt.suptitle("Speed-Accuracy Tradeoff by Stimulus", fontsize=14)
        plt.show()
        plt.close()

    def remove_non_answered_trials(self, print_flag=True):
        amount_trials_before_removal = self.search_rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.remove_non_answered_trials(False)

        if print_flag:
            print(f"Removed {amount_trials_before_removal - self.search_rts().shape[0]} non answered trials")

    def remove_poor_accuracy_sessions(self, threshold=0.5, print_flag=True):
        amount_sessions_total = sum([len(subject.sessions) for subject in self.subjects.values()])
        for subject in list(self.subjects.keys()):
            self.subjects[subject].remove_poor_accuracy_sessions(threshold,False)

        if print_flag:
            print(f"Removed {amount_sessions_total - sum([len(subject.sessions) for subject in self.subjects.values()])} sessions with poor accuracy")                



    def scanpaths_by_stimuli(self):
        return pl.concat([subject.scanpaths_by_stimuli() for subject in self.subjects.values()])



    def find_fixation_cutoff(self, percentile=1.0):
        # 1. Gather fixation counts
        fix_counts = [
            {
                "fix_count": trial.search_fixations().height,
                "target_present": trial.target_present,
                "memory_set_size": trial.memory_set_size
            }
            for subject in self.subjects.values()
            for session in subject.sessions.values()
            for trial in session.trials.values()
        ]
        fix_counts = pl.DataFrame(fix_counts)

        # 2. Get all unique group keys
        group_keys = fix_counts.select(["target_present", "memory_set_size"]).unique().to_dicts()

        # 3. Compute cutoff per group
        rows = []
        for group in group_keys:
            tp = group["target_present"]
            mem_size = group["memory_set_size"]

            group_df = fix_counts.filter(
                (pl.col("target_present") == tp) &
                (pl.col("memory_set_size") == mem_size)
            )

            fix_counts_list = group_df["fix_count"].to_list()
            total_fixations = sum(fix_counts_list)
            threshold = total_fixations * percentile
            max_possible = max(fix_counts_list)

            fix_cutoff = _find_fixation_cutoff(
                fix_count_list=fix_counts_list,
                threshold=threshold,
                max_possible=max_possible
            )

            rows.append({
                "target_present": tp,
                "memory_set_size": mem_size,
                "fix_cutoff": fix_cutoff
            })

        return pl.DataFrame(rows)


    def remove_trials_for_stimuli(self,stimuli,print_flag=True):
        '''
        Remove trials for stimuli that are in the list of stimuli.
        Parameters:
            - stimuli: list of stimuli to remove
            - print_flag: if True, print the number of trials removed
        '''
        # Get the trials for the stimuli to remove
        amount_trials_removed = 0
        subj_keys = list(self.subjects.keys())
        for subject_key in subj_keys:
            subject = self.subjects[subject_key]
            session_keys = list(subject.sessions.keys())
            for session_key in session_keys:
                session = subject.sessions[session_key]
                trial_keys = list(session.trials.keys())
                for trial_key in trial_keys:
                    trial = session.trials[trial_key]
                    if trial.stimulus in stimuli:
                        session.remove_trial(trial_key)
                        amount_trials_removed += 1
        if print_flag:
            print(f"Removed {amount_trials_removed} trials for stimuli {stimuli}")



    def remove_trials_for_stimuli_with_poor_accuracy(self, threshold=0.5, print_flag=True):
        '''For now this will be done without grouping by target_present'''
        scanpaths_by_stimuli = self.scanpaths_by_stimuli()
        grouped = scanpaths_by_stimuli.group_by(["stimulus", "memory_set_size"])
        poor_accuracy_stimuli = (
                                grouped.agg(pl.col("correct_response").mean().alias("accuracy"))
                                .filter(pl.col("accuracy") < threshold)
                            )
        # Get the stimulus and memory set size of poor_accuracy_stimuli into a list of tuples
        poor_accuracy_stimuli = poor_accuracy_stimuli.select(pl.col("stimulus"), pl.col("memory_set_size")).to_dicts()
        poor_accuracy_stimuli = [(stimulus["stimulus"], stimulus["memory_set_size"]) for stimulus in poor_accuracy_stimuli]
        amount_trials_removed = 0
        subj_keys = list(self.subjects.keys())
        for subject_key in subj_keys:
            subject = self.subjects[subject_key]
            session_keys = list(subject.sessions.keys())
            for session_key in session_keys:
                session = subject.sessions[session_key]
                trial_keys = list(session.trials.keys())
                for trial_key in trial_keys:
                    trial = session.trials[trial_key]
                    if (trial.stimulus, trial.memory_set_size) in poor_accuracy_stimuli:
                        session.remove_trial(trial_key)
                        amount_trials_removed += 1
        if print_flag:
            print(f"Removed {amount_trials_removed} trials from stimuli with less than {threshold} accuracy.")
    
    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_correct = pl.concat([subject.cumulative_correct_trials_by_fixation(group_cutoffs) for subject in self.subjects.values()])

        return cumulative_correct


    
    def plot_cumulative_performance(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_performance = self.cumulative_correct_trials_by_fixation(group_cutoffs).join(
            group_cutoffs,
            on=["target_present", "memory_set_size"],
            how="left"
        )

        tp_ta = cumulative_performance.select(pl.col("target_present")).unique().to_series()
        tp_ta.sort()
        mem_set_sizes = cumulative_performance.select(pl.col("memory_set_size")).unique().to_series()
        mem_set_sizes.sort()

        # Convert to pandas for Seaborn
        cumulative_performance = cumulative_performance.to_pandas()

        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),sharey=True)
        fig.suptitle("Cumulative Performance")
        if n_cols == 1:
            axs = np.array([axs])

        if n_rows == 1:
            axs = np.array([axs])

        # For each fixation number (i.e. first "max_fixations" columns), we need the mean and the standard error
        # The X axis will be the fixation number, the Y axis will be the accuracy
        # The area around the mean will be the standard error
        
        for i, row in enumerate(mem_set_sizes):
            for j, col in enumerate(tp_ta):
                # Get the max fix for the current group, groups_cutoff is in polars

                data = cumulative_performance[(cumulative_performance["memory_set_size"] == row) & (cumulative_performance["target_present"] == col)]
                max_fix = int(data["fix_cutoff"].iloc[0])

                # 1. Trim every array to the same length (optional but handy)
                trimmed = data["cumulative_correct"].apply(lambda arr: arr[:max_fix])

                # 2. Turn the Series-of-lists into long form
                exploded = trimmed.explode().reset_index(drop=True).to_frame("cumulative_correct")

                # 3. Add a 1-based fixation index
                exploded["fixation_number"] = (np.tile(np.arange(1, max_fix + 1), len(data))  # repeat 1..max_fix for every original row
    )
                sns.lineplot(
                    x="fixation_number",
                    y="cumulative_correct",
                    data=exploded,
                    ax=axs[i, j],
                    errorbar='se',
                    estimator='mean',
                    color="black"
                )
                axs[i, j].set_title(f"Memory Set Size {int(row)}, Target Present {bool(col)}")
                # Ticks every 5 fixations
                axs[i, j].set_xticks(range(0, max_fix, 5))
                axs[i, j].set_xticklabels(range(1, max_fix+1, 5))
                axs[i, j].set_xlabel("Fixation Number")
                axs[i, j].set_ylabel("Accuracy")

        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        plt.close()

    def trials_by_rt_bins(self, bin_end, bin_step):
        # 1. Get and filter RTs
        rts = self.rts().filter(pl.col("phase") == self._search_phase_name)
        rts = rts.with_columns([
            (pl.col("rt") / 1000).alias("rt")
        ])

        # 2. Compute bin edges
        bin_edges = np.arange(0, bin_end + bin_step, bin_step)

        # 3. Bin RTs using numpy (returns indices)
        bin_indices = np.digitize(rts["rt"].to_numpy(), bin_edges, right=False)

        # 4. Convert to left edge values
        rt_bin_labels = [bin_edges[i - 1] if i > 0 and i < len(bin_edges) else None for i in bin_indices]

        # 5. Assign back to the DataFrame
        rts = rts.with_columns([
            pl.Series("rt_bin", rt_bin_labels)
        ])

        return rts


    def plot_correct_trials_by_rt_bins(self, bin_end, bin_step):
        # Get relevant trial info with binned RTs
        correct_trials_per_bin = (
            self.trials_by_rt_bins(bin_end, bin_step)
            .select(["rt_bin", "target_present", "memory_set_size", "correct_response"])
            .group_by(["rt_bin", "target_present", "memory_set_size"])
            .agg(pl.col("correct_response").sum().alias("correct_response"))
            .sort(["memory_set_size", "target_present", "rt_bin"])
        ).to_pandas()

        # Ensure sorted and unique values
        tp_ta = sorted(correct_trials_per_bin["target_present"].unique())
        mem_set_sizes = sorted(correct_trials_per_bin["memory_set_size"].unique())

        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True, sharex=True)
        fig.suptitle("Correct Trials by RT Bins")

        # Normalize axis shape for consistent indexing
        if n_cols == 1:
            axs = np.expand_dims(axs, axis=1)
        if n_rows == 1:
            axs = np.expand_dims(axs, axis=0)

        for i, mem_size in enumerate(mem_set_sizes):
            for j, tp in enumerate(tp_ta):
                data = correct_trials_per_bin[
                    (correct_trials_per_bin["memory_set_size"] == mem_size) &
                    (correct_trials_per_bin["target_present"] == tp)
                ]
                sns.barplot(x="rt_bin", y="correct_response", data=data, ax=axs[i, j])
                axs[i, j].set_title(f"Memory Set Size {mem_size}, Target Present {bool(tp)}")
                axs[i, j].set_xlabel("RT Bins (s)")
                axs[i, j].set_ylabel("Correct Trials")
                axs[i, j].set_xticks(range(0, int(bin_end/bin_step)+3, 3))

        plt.tight_layout()
        plt.show()
        plt.close()
        
    def plot_incorrect_trials_by_rt_bins(self, bin_end, bin_step):
        # Get RT binned trial info
        incorrect_trials_per_bin = (
            self.trials_by_rt_bins(bin_end, bin_step)
            .select(["rt_bin", "target_present", "memory_set_size", "correct_response"])
            .with_columns([
                (1 - pl.col("correct_response")).alias("incorrect_response")
            ])
            .group_by(["rt_bin", "target_present", "memory_set_size"])
            .agg(pl.col("incorrect_response").sum().alias("incorrect_response"))
            .sort(["memory_set_size", "target_present", "rt_bin"])
            .to_pandas()
        )

        # Setup for plotting
        tp_ta = sorted(incorrect_trials_per_bin["target_present"].unique())
        mem_set_sizes = sorted(incorrect_trials_per_bin["memory_set_size"].unique())

        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True, sharex=True)
        fig.suptitle("Incorrect Trials by RT Bins")

        # Normalize shape for subplots
        if n_cols == 1:
            axs = np.expand_dims(axs, axis=1)
        if n_rows == 1:
            axs = np.expand_dims(axs, axis=0)

        for i, mem_size in enumerate(mem_set_sizes):
            for j, tp in enumerate(tp_ta):
                data = incorrect_trials_per_bin[
                    (incorrect_trials_per_bin["memory_set_size"] == mem_size) &
                    (incorrect_trials_per_bin["target_present"] == tp)
                ]
                sns.barplot(x="rt_bin", y="incorrect_response", data=data, ax=axs[i, j])
                axs[i, j].set_title(f"Memory Set Size {mem_size}, Target Present {bool(tp)}")
                axs[i, j].set_xlabel("RT Bins (s)")
                axs[i, j].set_ylabel("Incorrect Trials")
                axs[i, j].set_xticks(range(0, int(bin_end/bin_step)+3, 3))

        plt.tight_layout()
        plt.show()
        plt.close()
    
    def plot_probability_of_deciding_by_rt_bin(self, bin_end, bin_step):
        # Get RT-binned trials
        trials = (
            self.trials_by_rt_bins(bin_end, bin_step)
            .select(["rt_bin", "target_present", "memory_set_size", "correct_response"])
        )

        # Unique labels
        tp_ta = sorted(trials["target_present"].unique().to_list())
        mem_set_sizes = sorted(trials["memory_set_size"].unique().to_list())

        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)

        # Count occurrences per bin
        grouped = (
            trials
            .group_by(["rt_bin", "target_present", "correct_response", "memory_set_size"])
            .agg(pl.count().alias("count"))
            .sort(["correct_response", "target_present", "memory_set_size", "rt_bin"])
        )

        # Compute totals per (correctness, target, memory)
        totals = (
            grouped
            .group_by(["correct_response", "target_present", "memory_set_size"])
            .agg(pl.col("count").sum().alias("total_per_group"))
        )

        # Merge total counts back
        grouped = grouped.join(
            totals,
            on=["correct_response", "target_present", "memory_set_size"],
            how="left"
        )

        # Compute cumulative sums within groups
        grouped = (
            grouped
            .with_columns([
                pl.col("count").cum_sum().over(["correct_response", "target_present", "memory_set_size"]).alias("cumsum"),
            ])
            .with_columns([
                (pl.col("total_per_group") - pl.col("cumsum") + pl.col("count")).alias("total_per_bin"),
                (pl.col("count") / (pl.col("total_per_group") - pl.col("cumsum") + pl.col("count"))).alias("count_normalized"),
                pl.col("correct_response").cast(pl.Boolean)
            ])
        )

        # Convert to pandas for seaborn
        grouped_pd = grouped.to_pandas()

        # Plot setup
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True, sharex=True)
        fig.suptitle("Probability of Deciding by RT Bins")

        if n_cols == 1:
            axs = np.expand_dims(axs, axis=1)
        if n_rows == 1:
            axs = np.expand_dims(axs, axis=0)

        for i, mem_size in enumerate(mem_set_sizes):
            for j, tp in enumerate(tp_ta):
                data = grouped_pd[
                    (grouped_pd["memory_set_size"] == mem_size) &
                    (grouped_pd["target_present"] == tp)
                ]
                sns.barplot(x="rt_bin", y="count_normalized", hue="correct_response", data=data, ax=axs[i, j])
                axs[i, j].set_title(f"Memory Set Size {mem_size}, Target Present {bool(tp)}")
                axs[i, j].set_xlabel("RT Bins (s)")
                axs[i, j].set_ylabel("Probability of Deciding")
                axs[i, j].set_xticks(range(0, int(bin_end/bin_step)+3, 3))

        plt.tight_layout()
        plt.show()
        plt.close()


class VisualSearchSubject(Subject):
    def __init__(self, subject_id: str, old_subject_id: str, experiment: VisualSearchExperiment, search_phase_name, memorization_phase_name,
                 excluded_sessions: list = [], excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        super().__init__(subject_id, old_subject_id, experiment, excluded_sessions, excluded_trials, export_format)
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name

    @property
    def sessions(self):
        if self._sessions is None:
            self._sessions = { session_folder.name.split("-")[-1] :
                VisualSearchSession(session_folder.name.split("-")[-1], self,self._search_phase_name, self._memorization_phase_name,
                        self.excluded_trials.get(session_folder.name.split("-")[-1], {}),self.export_format) 
                for session_folder in self.subject_derivatives_path.glob("ses-*") 
                if session_folder.name.split("-")[-1] not in self.excluded_sessions
            }
        return self._sessions
    
    def scanpaths_by_stimuli(self):
        return pl.concat([session.scanpaths_by_stimuli() for session in self.sessions.values()])
    
    def search_rts(self):
        rts = self.rts().filter(pl.col("phase") == self._search_phase_name)
        return rts
    
    def search_saccades(self):
        saccades = self.saccades().filter(pl.col("phase") == self._search_phase_name)
        return saccades
    
    def search_fixations(self):
        fixations = self.fixations().filter(pl.col("phase") == self._search_phase_name)
        return fixations
    
    def accuracy(self):
        # Accuracy should be grouped by target present and memory set size
        correct_trials = self.search_rts()[["target_present", "correct_response", "memory_set_size"]]
        accuracy = correct_trials.group_by(["target_present", "memory_set_size"]).agg(
            pl.col("correct_response").mean().alias("accuracy")
        )
        # Add the self.subject_id to a new column
        accuracy = accuracy.with_columns(pl.lit(self.subject_id).alias("subject_id"))

        return accuracy
    
    def remove_non_answered_trials(self, print_flag=True):
        # Remove non answered trials from all sessions
        amount_trials_before_removal = self.search_rts().height
        for session in list(self.sessions.values()):
            session.remove_non_answered_trials(False)

        if print_flag:
            print(f"Removed {amount_trials_before_removal - self.search_rts().height} non answered trials from subject {self.subject_id}")



    def find_fixation_cutoff(self, percentile=1.0):
        # 1. Gather fixation counts
        fix_counts = [
            {
                "fix_count": trial.search_fixations().height,
                "target_present": trial.target_present,
                "memory_set_size": trial.memory_set_size
            }
            for session in self.sessions.values()
            for trial in session.trials.values()
        ]
        fix_counts = pl.DataFrame(fix_counts)

        # 2. Get all unique group keys
        group_keys = fix_counts.select(["target_present", "memory_set_size"]).unique().to_dicts()

        # 3. Compute cutoff per group
        rows = []
        for group in group_keys:
            tp = group["target_present"]
            mem_size = group["memory_set_size"]

            group_df = fix_counts.filter(
                (pl.col("target_present") == tp) &
                (pl.col("memory_set_size") == mem_size)
            )

            fix_counts_list = group_df["fix_count"].to_list()
            total_fixations = sum(fix_counts_list)
            threshold = total_fixations * percentile
            max_possible = max(fix_counts_list)

            fix_cutoff = _find_fixation_cutoff(
                fix_count_list=fix_counts_list,
                threshold=threshold,
                max_possible=max_possible
            )

            rows.append({
                "target_present": tp,
                "memory_set_size": mem_size,
                "fix_cutoff": fix_cutoff
            })

        return pl.DataFrame(rows)

    def remove_poor_accuracy_sessions(self, threshold=0.5, print_flag=True):
        poor_accuracy_sessions = 0
        keys = list(self.sessions.keys())
        for key in keys:
            session = self.sessions[key]
            if session.has_poor_accuracy(threshold):
                poor_accuracy_sessions+=1
                self.remove_session(key)

        if print_flag:
            print(f"Removed {poor_accuracy_sessions} sessions with poor accuracy from subject {self.subject_id}")

    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()

        cumulative_correct = pl.concat([session.cumulative_correct_trials_by_fixation(group_cutoffs) for session in self.sessions.values()])
        return cumulative_correct
    
      
class VisualSearchSession(Session):
    BEH_COLUMNS: list[str] = [
        "trial_number", "stimulus", "stimulus_coords", "memory_set", "memory_set_locations",
        "target_present", "target", "target_location", "correct_response", "was_answered"
    ]
    """
    Columns explanation:
    - trial_number: The number of the trial, in the order they were presented. They start from 0.
    - stimulus: The filename of the stimulus presented.
    - stimulus_coords: The coordinates of the stimulus presented. It should be a tuple containing the x, y of the top-left corner of the stimulus and the x, y of the bottom-right corner.
    - memory_set: The set of items memorized by the participant. It should be a list of strings. Each string should be the filename of the stimulus.
    - memory_set_locations: The locations of the items memorized by the participant. It should be a list of tuples. Each tuple should contain bounding
      boxes of the items memorized by the participant. The bounding boxes should be in the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and
      (x2, y2) is the bottom-right corner.
    - target_present: Whether one of the items is present in the stimulus. It should be a boolean.
    - target: The filename of the target item. It should be a string. If target_present is False, the value for this column will
      not be taken into account.
    - target_location: The location of the target item. It should be a tuple containing the bounding box of the target item. The bounding box should be in
      the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. If target_present is False, the value for this column will
      not be taken into account.
    - correct_response: The correct response for the trial. It should be a boolean.
    - was_answered: Whether the trial was answered by the participant. It should be a boolean.

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
        search_phase_name: str,
        memorization_phase_name: str,
        excluded_trials: list = None,
        export_format = FEATHER_EXPORT
    ):
        excluded_trials = [] if excluded_trials is None else excluded_trials
        super().__init__(session_id, subject, excluded_trials, export_format)
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name
        self.behavior_data = None



    def load_behavior_data(self):
        # Get the name of the only csv file in the behavior path
        behavior_path = self.session_dataset_path / "behavioral"

        behavior_files = list(behavior_path.glob("*.csv"))
        
        if len(behavior_files) != 1:
            raise ValueError(
                f"There should only be one CSV file in the behavior path for session {self.session_id} "
                f"of subject {self.subject.subject_id}. Found files: {[file.name for file in behavior_files]}"
            )

        # Load the CSV file
        name = behavior_files[0].name
        self.behavior_data = pl.read_csv(
            behavior_path / name,
            dtypes={
                "trial_number": pl.Int32,
                "stimulus": pl.Utf8,
                "target_present": pl.Int32,
                "target": pl.Utf8,
                "correct_response": pl.Int32,
                "was_answered": pl.Int32
            }
        )

        # Validate that all required columns are present
        missing_columns = set(self.BEH_COLUMNS) - set(self.behavior_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in behavior data: {missing_columns} for session {self.session_id} of subject {self.subject.subject_id}")

    def _init_trials(self,samples,fix,sacc,blink,events_path):
        self._trials = {trial:
            VisualSearchTrial(trial, self, samples, fix, sacc, blink, events_path, self.behavior_data,self._search_phase_name,self._memorization_phase_name)
            for trial in samples["trial_number"].unique() 
            if trial != -1 and trial not in self.excluded_trials and trial in self.behavior_data.select(pl.col("trial_number")).to_series().to_list()
        }
    
    def load_data(self, detection_algorithm: str):
        self.load_behavior_data()
        super().load_data(detection_algorithm)


    def search_rts(self):
        rts = self.rts().filter(pl.col("phase") == self._search_phase_name)
        return rts
    
    def search_saccades(self):
        saccades = self.saccades().filter(pl.col("phase") == self._search_phase_name)
        return saccades

    def search_fixations(self):
        fixations = self.fixations().filter(pl.col("phase") == self._search_phase_name)
        return fixations

    def accuracy(self):
        # Accuracy should be grouped by target present and memory set size
        correct_trials = self.search_rts()[["target_present", "correct_response", "memory_set_size"]]
        accuracy = correct_trials.groupby(["target_present", "memory_set_size"]).mean().reset_index()
        # Change the column name to accuracy
        accuracy.rename(columns={"correct_response": "accuracy"}, inplace=True)
        accuracy["session_id"] = self.session_id

        return accuracy

    def remove_non_answered_trials(self,print_flag=True):
        # Remove trials that were not answered
        non_answered_trials = [trial for trial in self.trials if not self.trials[trial].was_answered]
        for trial in non_answered_trials:
            self.remove_trial(trial)
        if print_flag:
            print(f"Removed {len(non_answered_trials)} non answered trials from session {self.session_id}")

    def has_poor_accuracy(self, threshold=0.5):
        correct_trials = self.search_rts()[["target_present", "correct_response", "memory_set_size"]]
        accuracy = correct_trials["correct_response"].sum() / correct_trials["correct_response"].count()
        return accuracy < threshold
    
    def find_fixation_cutoff(self, percentile=1.0):
        # 1. Gather fixation counts
        fix_counts = [
            {
                "fix_count": trial.search_fixations().height,
                "target_present": trial.target_present,
                "memory_set_size": trial.memory_set_size
            }

            for trial in self.trials.values()
        ]
        fix_counts = pl.DataFrame(fix_counts)

        # 2. Get all unique group keys
        group_keys = fix_counts.select(["target_present", "memory_set_size"]).unique().to_dicts()

        # 3. Compute cutoff per group
        rows = []
        for group in group_keys:
            tp = group["target_present"]
            mem_size = group["memory_set_size"]

            group_df = fix_counts.filter(
                (pl.col("target_present") == tp) &
                (pl.col("memory_set_size") == mem_size)
            )

            fix_counts_list = group_df["fix_count"].to_list()
            total_fixations = sum(fix_counts_list)
            threshold = total_fixations * percentile
            max_possible = max(fix_counts_list)

            fix_cutoff = _find_fixation_cutoff(
                fix_count_list=fix_counts_list,
                threshold=threshold,
                max_possible=max_possible
            )

            rows.append({
                "target_present": tp,
                "memory_set_size": mem_size,
                "fix_cutoff": fix_cutoff
            })

        return pl.DataFrame(rows)
    

    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()  # this should return a pl.DataFrame

        records = []

        for trial in self.trials.values():
            scanpath_length = len(trial.search_fixations())

            # ✅ Filter the appropriate fix_cutoff value
            fix_cutoff = (
                group_cutoffs
                .filter(
                    (pl.col("memory_set_size") == trial.memory_set_size) &
                    (pl.col("target_present") == trial.target_present)
                )
                .select("fix_cutoff")
                .item()
            )

            cumulative_correct = np.zeros(fix_cutoff)

            if trial.correct_response and scanpath_length - 1 <= fix_cutoff:
                cumulative_correct[scanpath_length - 1:] = 1

            records.append({
                "cumulative_correct": cumulative_correct,
                "target_present": trial.target_present,
                "memory_set_size": trial.memory_set_size,
            })

        df = pl.DataFrame(records)
        return df
   
    def scanpaths_by_stimuli(self):
        return pl.DataFrame([trial.scanpath_by_stimuli() for trial in self.trials.values()])

class VisualSearchTrial(Trial):

    def __init__(self, trial_number, session, samples, fix, sacc, blink, events_path, behavior_data, search_phase_name, memorization_phase_name):
        super().__init__(trial_number, session, samples, fix, sacc, blink, events_path)

        trial_data = behavior_data.filter(pl.col("trial_number") == trial_number)

        self._target_present = trial_data.select("target_present").item()
        self._target = trial_data.select("target").item()
        
        if self._target_present:
            self._target_location = _as(trial_data.select("target_location").item(), tuple)

        self._correct_response = trial_data.select("correct_response").item()
        self._stimulus = trial_data.select("stimulus").item()
        self._stimulus_coords = _as(trial_data.select("stimulus_coords").item(), tuple)

        self._memory_set = _as(trial_data.select("memory_set").item(), list)
        self._memory_set_locations = _as(trial_data.select("memory_set_locations").item(), list)
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name
        self._was_answered = trial_data.select("was_answered").item()

    @property
    def target(self):
        return self._target

    @property
    def target_location(self):
        return self._target_location

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
    def memory_set_locations(self):
        return self._memory_set_locations
    
    @property
    def memory_set(self):
        return self._memory_set
    
    @property
    def stimulus(self):
        return self._stimulus
    
    @property
    def stimulus_coords(self):
        return self._stimulus_coords
    
    @property
    def was_answered(self):
        return self._was_answered

    def save_rts(self):
        if hasattr(self, "_rts"):
            return

        # Filter out empty phase rows
        filtered = self._samples.filter(pl.col("phase") != "")

        # Calculate RT as the difference between last and first tSample per phase
        self._rts = (
            filtered
            .group_by("phase")
            .agg((pl.col("tSample").max() - pl.col("tSample").min()).alias("rt"))
            .with_columns([
                pl.lit(self.trial_number).alias("trial_number"),
                pl.lit(len(self._memory_set)).alias("memory_set_size"),
                pl.lit(self._target_present).alias("target_present"),
                pl.lit(self._correct_response).alias("correct_response"),
                pl.lit(self._stimulus).alias("stimulus"),
                pl.lit(self._target).alias("target"),
                pl.lit(self._was_answered).alias("was_answered"),
            ])
        )


    def fixations(self):
        fixations = super().fixations().with_columns([
            pl.lit(self._target_present).alias("target_present"),
            pl.lit(self._correct_response).alias("correct_response"),
            pl.lit(self._stimulus).alias("stimulus"),
            pl.lit(self._target).alias("target"),
            pl.lit(self._memory_set).alias("memory_set"),
        ])
        return fixations


    def saccades(self):
        saccades = super().saccades().with_columns([
            pl.lit(self._target_present).alias("target_present"),
            pl.lit(self._correct_response).alias("correct_response"),
            pl.lit(self._stimulus).alias("stimulus"),
            pl.lit(self._target).alias("target"),
            pl.lit(self._memory_set).alias("memory_set"),
        ])
        return saccades



    def search_fixations(self):
        return self.fixations().filter(pl.col("phase") == self._search_phase_name).sort(by="tStart")
    
    def memorization_fixations(self):
        return self.fixations().filter(pl.col("phase") == self._memorization_phase_name).sort(by="tStart")

    def search_saccades(self):
        return self.saccades().filter(pl.col("phase") == self._search_phase_name).sort(by="tStart")
    
    def memorization_saccades(self):
        return self.saccades().filter(pl.col("phase") == self._memorization_phase_name).sort(by="tStart")
    
    def search_samples(self):
        return self.samples().filter(pl.col("phase") == self._search_phase_name).sort(by="tSample")
    
    def memorization_samples(self):
        return self.samples().filter(pl.col("phase") == self._memorization_phase_name).sort(by="tSample")
    
    def scanpath_by_stimuli(self):
        return {"fixations": self.search_fixations(), "stimulus": self._stimulus,"correct_response":self._correct_response,"target_present":self._target_present,"memory_set_size":len(self._memory_set)}
    
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
        dataset_parent_folder = self.events_path.parent.parent.parent.parent
        phase_data[self._search_phase_name]["img_paths"] = [dataset_parent_folder / STIMULI_FOLDER / self._stimulus]
        phase_data[self._search_phase_name]["img_plot_coords"] = [self._stimulus_coords]
        if self._memorization_phase_name is not None:
            phase_data[self._memorization_phase_name]["img_paths"] = [dataset_parent_folder / ITEMS_FOLDER / img for img in self._memory_set]
            phase_data[self._memorization_phase_name]["img_plot_coords"] = self._memory_set_locations

        # If the target is present add the "bbox" to the search_phase phase as a key-value pair
        if self._target_present:
            phase_data[self._search_phase_name]["bbox"] = self._target_location
        vis.scanpath(fixations=self._fix,phase_data=phase_data, saccades=self._sacc, samples=self._samples, screen_height=screen_height, screen_width=screen_width, 
                      folder_path=self.events_path / "plots", **kwargs)

    def plot_animation(self, screen_height, screen_width, video_path=None, background_image_path=None, **kwargs):
        """
        Create an animated visualization of eye-tracking data for this trial.

        When a video is provided, the animation syncs gaze samples with video frames.
        When no video is provided, gaze points are animated on a grey background,
        or a provided background image (e.g., the stimulus image), using the sample 
        timestamps for timing.

        Parameters
        ----------
        screen_height, screen_width
            Stimulus resolution in pixels.
        video_path
            Path to a video file. If provided, gaze is overlaid on video frames.
        background_image_path
            Path to a background image. Only used when video_path is None.
            If None and no video, uses the search stimulus as background if available,
            otherwise uses a grey background.
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

        # Set default folder_path if not provided
        if 'folder_path' not in kwargs:
            kwargs['folder_path'] = self.events_path / "plots"

        # If no background image provided and no video, try to use the stimulus
        if video_path is None and background_image_path is None:
            dataset_parent_folder = self.events_path.parent.parent.parent.parent
            stimulus_path = dataset_parent_folder / STIMULI_FOLDER / self._stimulus
            if stimulus_path.exists():
                background_image_path = stimulus_path

        return vis.plot_animation(
            samples=self._samples,
            screen_height=screen_height,
            screen_width=screen_width,
            video_path=video_path,
            background_image_path=background_image_path,
            **kwargs
        )