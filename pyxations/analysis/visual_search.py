from pathlib import Path
import pandas as pd
from pyxations.visualization.visualization import Visualization
from pyxations.export import FEATHER_EXPORT
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pyxations.analysis.generic import Experiment, Subject, Session, Trial, _find_fixation_cutoff, STIMULI_FOLDER, ITEMS_FOLDER
import multimatch_gaze as mm


class VisualSearchExperiment(Experiment):
    def __init__(self, dataset_path: str,search_phase_name: str,memorization_phase_name: str, excluded_subjects: list = [], excluded_sessions: dict = {}, excluded_trials: dict = {}, export_format = FEATHER_EXPORT):
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(self.dataset_path.name + "_derivatives")
        self.metadata = pd.read_csv(self.dataset_path / "participants.tsv", sep="\t", 
                                    dtype={"subject_id": str, "old_subject_id": str})
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
        accuracy = pd.concat([subject.accuracy() for subject in self.subjects.values()], ignore_index=True)

        return accuracy
    
    def plot_accuracy_by_subject(self):
        accuracy = self.accuracy().sort_values(by=["memory_set_size", "target_present","accuracy"])
        # target present to bool
        accuracy["target_present"] = accuracy["target_present"].astype(bool)
        # There should be an ax for each memory set size

        mem_set_sizes = accuracy["memory_set_size"].unique()
        mem_set_sizes.sort()

        n_rows = len(mem_set_sizes)
        fig, axs = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows),sharey=True)

        if n_rows == 1:
            axs = np.array([axs])

        for i, row in enumerate(mem_set_sizes):
            data = accuracy[(accuracy["memory_set_size"] == row)]
            sns.barplot(x="subject_id", y="accuracy", data=data, ax=axs[i],estimator="mean", hue="target_present")
            axs[i].set_title(f"Memory Set Size {row}")
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].set_xlabel("Subject ID")
            axs[i].set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()
        plt.close()
    
    def plot_accuracy_by_stimulus(self):
        accuracy = self.get_search_rts().groupby(["target_present", "memory_set_size","stimulus"])[["rt","correct_response"]].mean().reset_index().sort_values(by=["memory_set_size", "target_present","correct_response"])
        # target present to bool
        accuracy["target_present"] = accuracy["target_present"].astype(bool)
        # correct_response name to accuracy
        accuracy = accuracy.rename(columns={"correct_response": "accuracy"})
        # There should be an ax for each memory set size

        mem_set_sizes = accuracy["memory_set_size"].unique()
        mem_set_sizes.sort()

        n_rows = len(mem_set_sizes)
        fig, axs = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows),sharey=True)

        if n_rows == 1:
            axs = np.array([axs])

        for i, row in enumerate(mem_set_sizes):
            data = accuracy[(accuracy["memory_set_size"] == row)]
            sns.barplot(x="stimulus", y="accuracy", data=data, ax=axs[i],estimator="mean", hue="target_present")
            axs[i].set_title(f"Memory Set Size {row}")
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].set_xlabel("Stimulus")
            axs[i].set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()
        plt.close()

    def get_search_rts(self):
        rts = self.get_rts()
        return rts[rts["phase"] == self._search_phase_name]
    
    def get_search_saccades(self):
        saccades = self.saccades()
        return saccades[saccades["phase"] == self._search_phase_name]

    def get_search_fixations(self):
        fixations = self.fixations()
        return fixations[fixations["phase"] == self._search_phase_name]

    def plot_speed_accuracy_tradeoff_by_subject(self):
        # 1) Aggregate the data (as you already do).
        speed_accuracy = (
            self.get_search_rts()
                .groupby(["target_present", "memory_set_size", "subject_id"])[["rt","correct_response"]]
                .mean()
                .reset_index()
        )
        # Convert booleans and rename
        speed_accuracy["target_present"] = speed_accuracy["target_present"].astype(bool)
        speed_accuracy = speed_accuracy.rename(columns={"correct_response": "accuracy"})
        
        # Convert RT from ms to seconds (if needed)
        speed_accuracy["rt"] = speed_accuracy["rt"] / 1000.0

        # Unique memory set sizes
        mem_set_sizes = np.sort(speed_accuracy["memory_set_size"].unique())
        n_rows = len(mem_set_sizes)

        # 2) Prepare a figure that has 2 rows per memory-set size:
        #    - top row:  x-hist
        #    - bottom:   scatter + y-hist
        fig = plt.figure(figsize=(6,1 + 6 * n_rows))
        gs = fig.add_gridspec(
            2 * n_rows, 2, 
            width_ratios=(4, 1), 
            height_ratios=[1, 4]*n_rows,  # repeat [1,4] for each row
            left=0.1, right=0.9, bottom=0.07, top=0.85,
            wspace=0.05, hspace=0.05
        )

        # 3) Loop over each memory-set size, building a “scatter+hist” layout
        for i, mem_size in enumerate(mem_set_sizes):
            data = speed_accuracy[speed_accuracy["memory_set_size"] == mem_size]

            # Indices in the GridSpec
            row_top    = 2*i     # hist of x
            row_bottom = 2*i + 1 # scatter + hist of y

            # Create the three Axes
            ax       = fig.add_subplot(gs[row_bottom, 0])  # main scatter
            ax_histx = fig.add_subplot(gs[row_top,    0], sharex=ax)
            ax_histy = fig.add_subplot(gs[row_bottom, 1], sharey=ax)

            # ------------------------------------
            # (A) The main scatter plot (color by `target_present`)
            #     Using Seaborn the same way you did:
            sns.scatterplot(
                x="accuracy", 
                y="rt", 
                data=data, 
                hue="target_present",
                ax=ax
            )

            # (B) Connect the same `subject_id` pairs with lines
            #     between target_present=False and target_present=True
            for stim in data["subject_id"].unique():
                stim_data = data[data["subject_id"] == stim]
                init_point  = stim_data[stim_data["target_present"] == False]
                final_point = stim_data[stim_data["target_present"] == True]
                if len(init_point) == 0 or len(final_point) == 0:
                    continue
                ax.plot(
                    [init_point["accuracy"].values[0], final_point["accuracy"].values[0]],
                    [init_point["rt"].values[0],       final_point["rt"].values[0]],
                    color="black", alpha=0.3, linewidth=0.5, zorder=0
                )

            # (C) Histograms on top (ax_histx) and on the right (ax_histy).
            #     We'll just do a simple histogram here; 
            #     you can adapt binning to your data range.
            ax_histx.hist(data["accuracy"], bins=np.linspace(0, 1, 21), color="gray")
            ax_histy.hist(data["rt"], bins=20, orientation='horizontal', color="gray")

            # Turn off tick labels for the marginal plots
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)

            # (D) Titles, axes limits, etc.
            ax_histx.set_title(f"Memory Set Size {mem_size}")
            ax.set_xlim(0, 1)
            # Force RT >= 0
            y_max = speed_accuracy["rt"].max() * 1.1
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Mean RT (s)")

        # 4) Final figure touches
        plt.suptitle("Speed-Accuracy Tradeoff by Subject", fontsize=14)
        plt.show()
        plt.close()

    def plot_speed_accuracy_tradeoff_by_stimulus(self):
        # 1) Aggregate the data (as you already do).
        speed_accuracy = (
            self.get_search_rts()
                .groupby(["target_present", "memory_set_size", "stimulus"])[["rt","correct_response"]]
                .mean()
                .reset_index()
        )
        # Convert booleans and rename
        speed_accuracy["target_present"] = speed_accuracy["target_present"].astype(bool)
        speed_accuracy = speed_accuracy.rename(columns={"correct_response": "accuracy"})
        
        # Convert RT from ms to seconds (if needed)
        speed_accuracy["rt"] = speed_accuracy["rt"] / 1000.0

        # Unique memory set sizes
        mem_set_sizes = np.sort(speed_accuracy["memory_set_size"].unique())
        n_rows = len(mem_set_sizes)

        # 2) Prepare a figure that has 2 rows per memory-set size:
        #    - top row:  x-hist
        #    - bottom:   scatter + y-hist
        fig = plt.figure(figsize=(6,1 + 6 * n_rows))
        gs = fig.add_gridspec(
            2 * n_rows, 2, 
            width_ratios=(4, 1), 
            height_ratios=[1, 4]*n_rows,  # repeat [1,4] for each row
            left=0.1, right=0.9, bottom=0.07, top=0.85,
            wspace=0.05, hspace=0.05
        )

        # 3) Loop over each memory-set size, building a “scatter+hist” layout
        for i, mem_size in enumerate(mem_set_sizes):
            data = speed_accuracy[speed_accuracy["memory_set_size"] == mem_size]

            # Indices in the GridSpec
            row_top    = 2*i     # hist of x
            row_bottom = 2*i + 1 # scatter + hist of y

            # Create the three Axes
            ax       = fig.add_subplot(gs[row_bottom, 0])  # main scatter
            ax_histx = fig.add_subplot(gs[row_top,    0], sharex=ax)
            ax_histy = fig.add_subplot(gs[row_bottom, 1], sharey=ax)

            # ------------------------------------
            # (A) The main scatter plot (color by `target_present`)
            #     Using Seaborn the same way you did:
            sns.scatterplot(
                x="accuracy", 
                y="rt", 
                data=data, 
                hue="target_present",
                ax=ax
            )

            # (B) Connect the same `stimulus` pairs with lines
            #     between target_present=False and target_present=True
            for stim in data["stimulus"].unique():
                stim_data = data[data["stimulus"] == stim]
                init_point  = stim_data[stim_data["target_present"] == False]
                final_point = stim_data[stim_data["target_present"] == True]
                if len(init_point) == 0 or len(final_point) == 0:
                    continue
                ax.plot(
                    [init_point["accuracy"].values[0], final_point["accuracy"].values[0]],
                    [init_point["rt"].values[0],       final_point["rt"].values[0]],
                    color="black", alpha=0.3, linewidth=0.5, zorder=0
                )

            # (C) Histograms on top (ax_histx) and on the right (ax_histy).
            #     We'll just do a simple histogram here; 
            #     you can adapt binning to your data range.
            ax_histx.hist(data["accuracy"], bins=np.linspace(0, 1, 21), color="gray")
            ax_histy.hist(data["rt"], bins=20, orientation='horizontal', color="gray")

            # Turn off tick labels for the marginal plots
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)

            # (D) Titles, axes limits, etc.
            ax_histx.set_title(f"Memory Set Size {mem_size}")
            ax.set_xlim(0, 1)
            # Force RT >= 0
            y_max = speed_accuracy["rt"].max() * 1.1
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Mean RT (s)")

        # 4) Final figure touches
        plt.suptitle("Speed-Accuracy Tradeoff by Stimulus", fontsize=14)
        plt.show()
        plt.close()

    def remove_poor_accuracy_sessions(self, threshold=0.5):
        keys = list(self.subjects.keys())
        dict_removed = {"subject":[],"session":[]}
        for subject in keys:
            dict_subject = self.subjects[subject].remove_poor_accuracy_sessions(threshold)
            for session in dict_subject["session"]:
                dict_removed["session"].append((subject,session))
            for sub in dict_subject["subject"]:
                dict_removed["subject"].append(sub)
        return dict_removed

    def scanpaths_by_stimuli(self):
        return pd.concat([subject.scanpaths_by_stimuli() for subject in self.subjects.values()], ignore_index=True)



    def find_fixation_cutoff(self, percentile=1.0):
         # 1. Gather fixation counts
        fix_counts = [(trial.search_fixations().shape[0],trial.target_present, trial.memory_set_size) for subject in self.subjects.values() for session in subject.sessions.values() for trial in session.trials.values()]
        fix_counts = pd.DataFrame(fix_counts, columns=["fix_count", "target_present", "memory_set_size"])
        # Group fix_counts by target_present and memory_set_size
        grouped = fix_counts.groupby(["target_present", "memory_set_size"])["fix_count"] \
                   .apply(list) \
                   .reset_index(name="fix_counts_list")

        # 2. Compute total fixations & threshold per group
        grouped["total_fixations"] = grouped["fix_counts_list"].apply(sum)
        grouped["threshold"] = grouped["total_fixations"] * percentile
        grouped["max_possible"] = grouped["fix_counts_list"].apply(max)
        grouped["fix_cutoff"] = grouped.apply(
            lambda row: _find_fixation_cutoff(
                fix_count_list=row["fix_counts_list"], 
                threshold=row["threshold"], 
                max_possible=row["max_possible"]
            ), 
            axis=1
)

        return grouped[["target_present", "memory_set_size", "fix_cutoff"]]



    def remove_trials_for_stimuli_with_poor_accuracy(self, threshold=0.5):
        '''For now this will be done without grouping by target_present'''
        scanpaths_by_stimuli = self.scanpaths_by_stimuli()
        grouped = scanpaths_by_stimuli.groupby(["stimulus", "memory_set_size"])
        poor_accuracy_stimuli = grouped["correct_response"].mean() < threshold
        poor_accuracy_stimuli = poor_accuracy_stimuli[poor_accuracy_stimuli].index
        subj_keys = list(self.subjects.keys())
        dict_removed= {"subject":[],"session":[],"trial":[]}
        for subject_key in subj_keys:
            subject = self.subjects[subject_key]
            session_keys = list(subject.sessions.keys())
            for session_key in session_keys:
                session = subject.sessions[session_key]
                trial_keys = list(session.trials.keys())
                for trial_key in trial_keys:
                    trial = session.trials[trial_key]
                    if (trial.stimulus, trial.memory_set_size) in poor_accuracy_stimuli:
                        dict_removed["trial"].append((subject_key,session_key,trial_key))
                        trial.unlink_session()
                if len(session.trials) == 0:
                    dict_removed["session"].append((subject_key,session_key))
                    session.unlink_subject()
            if len(subject.sessions) == 0:
                dict_removed["subject"].append(subject_key)
                subject.unlink_experiment()
        return dict_removed
    
    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_correct = pd.concat([subject.cumulative_correct_trials_by_fixation(group_cutoffs) for subject in self.subjects.values()], ignore_index=True)

        return cumulative_correct

    def cumulative_performance_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_correct = self.cumulative_correct_trials_by_fixation(group_cutoffs)
        cumulative_performance = cumulative_correct.groupby(["memory_set_size", "target_present"])["cumulative_correct"].apply(lambda x: np.mean(x.values, axis=0)).reset_index()
        cumulative_performance_sem = cumulative_correct.groupby(["memory_set_size", "target_present"])["cumulative_correct"].apply(lambda x: np.std(x.values, axis=0) / np.sqrt(len(x))).reset_index()
        return cumulative_performance, cumulative_performance_sem
    
    def plot_cumulative_performance(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_performance, cumulative_performance_sem = self.cumulative_performance_by_fixation(group_cutoffs)

        tp_ta = cumulative_performance["target_present"].unique()
        tp_ta.sort()
        mem_set_sizes = cumulative_performance["memory_set_size"].unique()
        mem_set_sizes.sort()
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
                max_fix_for_group = group_cutoffs[(group_cutoffs["memory_set_size"] == row) & (group_cutoffs["target_present"] == col)]["fix_cutoff"].values[0]
                data_mean = cumulative_performance[(cumulative_performance["memory_set_size"] == row) & (cumulative_performance["target_present"] == col)]["cumulative_correct"].values[0]
                data_sem = cumulative_performance_sem[(cumulative_performance_sem["memory_set_size"] == row) & (cumulative_performance_sem["target_present"] == col)]["cumulative_correct"].values[0]
                axs[i, j].plot(data_mean,color="black")
                axs[i, j].fill_between(np.arange(0, max_fix_for_group), data_mean - data_sem, data_mean + data_sem, color="gray", alpha=0.5)
                axs[i, j].set_title(f"Memory Set Size {int(row)}, Target Present {bool(col)}")
                # Ticks every 5 fixations
                axs[i, j].set_xticks(range(0, max_fix_for_group, 5))
                axs[i, j].set_xticklabels(range(1, max_fix_for_group+1, 5))
                axs[i, j].set_xlabel("Fixation Number")
                axs[i, j].set_ylabel("Accuracy")

        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        plt.close()

    def trials_by_rt_bins(self, bin_end,bin_step):
        bins = pd.interval_range(start=0, end=bin_end, freq=bin_step)
        rts = self.get_rts()
        rts = rts[rts["phase"] == self._search_phase_name].reset_index(drop=True)
        rts["rt"] = rts["rt"]/1000
        rts["rt_bin"] = pd.cut(rts["rt"], bins)
        # Map bin to the first element
        rts["rt_bin"] = rts["rt_bin"].apply(lambda x: x.left)
        return rts

    def plot_correct_trials_by_rt_bins(self, bin_end,bin_step):
        correct_trials_per_bin = self.trials_by_rt_bins(bin_end,bin_step)[["rt_bin","target_present","memory_set_size","correct_response"]]
        correct_trials_per_bin = correct_trials_per_bin.groupby(["rt_bin","target_present","memory_set_size"],observed=False).sum().reset_index()
        tp_ta = correct_trials_per_bin["target_present"].unique()
        tp_ta.sort()
        mem_set_sizes = correct_trials_per_bin["memory_set_size"].unique()
        mem_set_sizes.sort()
        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),sharey=True,sharex=True)
        fig.suptitle("Correct Trials by RT Bins")
        if n_cols == 1:
            axs = np.array([axs])
        if n_rows == 1:
            axs = np.array([axs])
        for i, row in enumerate(mem_set_sizes):
            for j, col in enumerate(tp_ta):
                data = correct_trials_per_bin[(correct_trials_per_bin["memory_set_size"] == row) & (correct_trials_per_bin["target_present"] == col)]
                sns.barplot(x="rt_bin", y="correct_response", data=data, ax=axs[i, j])
                axs[i, j].set_title(f"Memory Set Size {int(row)}, Target Present {bool(col)}")
                axs[i, j].set_xlabel("RT Bins")
                axs[i, j].set_ylabel("Correct Trials")
                # Ticks every 5 bins
                axs[i, j].set_xticks(range(0, int(bin_end/bin_step)+3, 3))

        plt.tight_layout()
        plt.show()
        plt.close()
        
    def plot_incorrect_trials_by_rt_bins(self, bin_end,bin_step):
        incorrect_trials_per_bin = self.trials_by_rt_bins(bin_end,bin_step)[["rt_bin","target_present","memory_set_size","correct_response"]]
        incorrect_trials_per_bin["correct_response"] = 1 - incorrect_trials_per_bin["correct_response"]
        incorrect_trials_per_bin = incorrect_trials_per_bin.groupby(["rt_bin","target_present","memory_set_size"],observed=False).sum().reset_index()
        tp_ta = incorrect_trials_per_bin["target_present"].unique()
        tp_ta.sort()
        mem_set_sizes = incorrect_trials_per_bin["memory_set_size"].unique()
        mem_set_sizes.sort()
        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),sharey=True,sharex=True)
        fig.suptitle("Incorrect Trials by RT Bins")
        if n_cols == 1:
            axs = np.array([axs])
        if n_rows == 1:
            axs = np.array([axs])
        for i, row in enumerate(mem_set_sizes):
            for j, col in enumerate(tp_ta):
                data = incorrect_trials_per_bin[(incorrect_trials_per_bin["memory_set_size"] == row) & (incorrect_trials_per_bin["target_present"] == col)]
                sns.barplot(x="rt_bin", y="correct_response", data=data, ax=axs[i, j])
                axs[i, j].set_title(f"Memory Set Size {int(row)}, Target Present {bool(col)}")
                axs[i, j].set_xlabel("RT Bins")
                axs[i, j].set_ylabel("Incorrect Trials")
                # Ticks every 5 bins
                axs[i, j].set_xticks(range(0, int(bin_end/bin_step)+3, 3))

        plt.tight_layout()
        plt.show()
        plt.close()
    
    def plot_probability_of_deciding_by_rt_bin(self, bin_end,bin_step):
        # Grouped by rt bins and target_present
        correct_trials_per_bin = self.trials_by_rt_bins(bin_end,bin_step)[["rt_bin","target_present","memory_set_size","correct_response"]]
        tp_ta = correct_trials_per_bin["target_present"].unique()
        tp_ta.sort()
        mem_set_sizes = correct_trials_per_bin["memory_set_size"].unique()
        mem_set_sizes.sort()
        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)
        grouped = correct_trials_per_bin.groupby(["rt_bin", "target_present","correct_response","memory_set_size"],observed=False).size().reset_index(name="count")
        grouped["rt_bin"] = grouped["rt_bin"].astype(float)

        # Get the amount of unfinished trials per RT bin, which is cumulatively substracting the amount of trials in the previous bins
        totals = grouped.groupby(["correct_response", "target_present","memory_set_size"])["count"].sum()
        cumsums = grouped.groupby(["correct_response", "target_present","memory_set_size"])["count"].cumsum()

        # Change index of grouped to a MultiIndex of target_present and correct_response
        grouped.set_index(["correct_response","target_present","memory_set_size"], inplace=True)
        grouped["total_per_bin"] = totals
        grouped = grouped.reset_index()
        grouped["total_per_bin"] -= cumsums - grouped["count"]

        grouped_agg = grouped.groupby(["correct_response","target_present","rt_bin","memory_set_size"]).agg({"count": "sum", "total_per_bin": "sum"}).reset_index()
        grouped_agg["count_normalized"] = grouped_agg["count"] / grouped_agg["total_per_bin"]
        # correct_response as bool
        grouped_agg["correct_response"] = grouped_agg["correct_response"].astype(bool)

        # Plot the waiting per RT bin (1st plot for correct trials, 2nd plot for incorrect trials)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),sharey=True,sharex=True)
        fig.suptitle("Probability of Deciding by RT Bins")
        if n_cols == 1:
            axs = np.array([axs])
        if n_rows == 1:
            axs = np.array([axs])  

        for i, row in enumerate(mem_set_sizes):
            for j, col in enumerate(tp_ta):
                data = grouped_agg[(grouped_agg["memory_set_size"] == row) & (grouped_agg["target_present"] == col)]
                sns.barplot(x="rt_bin", y="count_normalized", hue="correct_response", data=data, ax=axs[i, j])
                axs[i, j].set_title(f"Memory Set Size {int(row)}, Target Present {bool(col)}")
                axs[i, j].set_xlabel("RT Bins")
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
        return pd.concat([session.scanpaths_by_stimuli() for session in self.sessions.values()], ignore_index=True)
    
    def get_search_rts(self):
        rts = self.get_rts()
        return rts[rts["phase"] == self._search_phase_name]
    
    def get_search_saccades(self):
        saccades = self.saccades()
        return saccades[saccades["phase"] == self._search_phase_name]
    
    def get_search_fixations(self):
        fixations = self.fixations()
        return fixations[fixations["phase"] == self._search_phase_name]
    
    def accuracy(self):
        # Accuracy should be grouped by target present and memory set size
        correct_trials = self.get_search_rts()[["target_present", "correct_response", "memory_set_size"]]
        accuracy = correct_trials.groupby(["target_present", "memory_set_size"]).mean().reset_index()
        # Change the column name to accuracy
        accuracy.rename(columns={"correct_response": "accuracy"}, inplace=True)
        accuracy["subject_id"] = self.subject_id

        return accuracy

    def find_fixation_cutoff(self, percentile=1.0):
         # 1. Gather fixation counts
        fix_counts = [(trial.search_fixations().shape[0],trial.target_present, trial.memory_set_size) for session in self.sessions.values() for trial in session.trials.values()]
        fix_counts = pd.DataFrame(fix_counts, columns=["fix_count", "target_present", "memory_set_size"])
        # Group fix_counts by target_present and memory_set_size
        grouped = fix_counts.groupby(["target_present", "memory_set_size"])["fix_count"] \
                   .apply(list) \
                   .reset_index(name="fix_counts_list")

        # 2. Compute total fixations & threshold per group
        grouped["total_fixations"] = grouped["fix_counts_list"].apply(sum)
        grouped["threshold"] = grouped["total_fixations"] * percentile
        grouped["max_possible"] = grouped["fix_counts_list"].apply(max)
        grouped["fix_cutoff"] = grouped.apply(
            lambda row: _find_fixation_cutoff(
                fix_count_list=row["fix_counts_list"], 
                threshold=row["threshold"], 
                max_possible=row["max_possible"]
            ), 
            axis=1
)

        return grouped[["target_present", "memory_set_size", "fix_cutoff"]]    

    def remove_poor_accuracy_sessions(self, threshold=0.5):
        poor_accuracy_sessions = []
        keys = list(self.sessions.keys())
        dict_removed = {"subject":[],"session":[]}
        for key in keys:
            session = self.sessions[key]
            if session.has_poor_accuracy(threshold):
                poor_accuracy_sessions.append(session.session_id)
                dict_removed["session"].append(session.session_id)
                session.unlink_subject()
        if len(poor_accuracy_sessions) == len(keys):
            dict_removed["subject"].append(self.subject_id)
            self.unlink_experiment()
        return dict_removed

    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()

        cumulative_correct = pd.concat([session.cumulative_correct_trials_by_fixation(group_cutoffs) for session in self.sessions.values()], ignore_index=True)
        return cumulative_correct

    
    def cumulative_performance_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_correct = self.cumulative_correct_trials_by_fixation(group_cutoffs)
        cumulative_performance = cumulative_correct.groupby(["memory_set_size", "target_present"])["cumulative_correct"].apply(lambda x: np.mean(x.values, axis=0)).reset_index()
        cumulative_performance_sem = cumulative_correct.groupby(["memory_set_size", "target_present"])["cumulative_correct"].apply(lambda x: np.std(x.values, axis=0) / np.sqrt(len(x))).reset_index()
        return cumulative_performance, cumulative_performance_sem
    
    def trials_by_rt_bins(self, bin_end,bin_step):
        bins = pd.interval_range(start=0, end=bin_end, freq=bin_step)
        rts = self.get_rts()
        rts = rts[rts["phase"] == self._search_phase_name].reset_index(drop=True)
        rts["rt"] = rts["rt"]/1000
        rts["rt_bin"] = pd.cut(rts["rt"], bins)
        # Map bin to the first element
        rts["rt_bin"] = rts["rt_bin"].apply(lambda x: x.left)
        return rts


       
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
        self.behavior_data = pd.read_csv(
            behavior_path / name,
            dtype={
                "trial_number": int,
                "stimulus": str,
                "target_present": bool,
                "target": str,
                "correct_response": bool,
                "was_answered": bool
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


    
    def trials_by_rt_bins(self,bin_end,bin_step):
        bins = pd.interval_range(start=0, end=bin_end, freq=bin_step)
        rts = self.get_rts()
        rts = rts[rts["phase"] == self._search_phase_name].reset_index(drop=True)
        rts["rt"] = rts["rt"]/1000
        rts["rt_bin"] = pd.cut(rts["rt"], bins)
        # Map bin to the first element
        rts["rt_bin"] = rts["rt_bin"].apply(lambda x: x.left)
        return rts

    def get_search_rts(self):
        rts = self.get_rts()
        return rts[rts["phase"] == self._search_phase_name]
    
    def get_search_saccades(self):
        saccades = self.saccades()
        return saccades[saccades["phase"] == self._search_phase_name]

    def get_search_fixations(self):
        fixations = self.fixations()
        return fixations[fixations["phase"] == self._search_phase_name]

    def accuracy(self):
        # Accuracy should be grouped by target present and memory set size
        correct_trials = self.get_search_rts()[["target_present", "correct_response", "memory_set_size"]]
        accuracy = correct_trials.groupby(["target_present", "memory_set_size"]).mean().reset_index()
        # Change the column name to accuracy
        accuracy.rename(columns={"correct_response": "accuracy"}, inplace=True)
        accuracy["session_id"] = self.session_id

        return accuracy

    def has_poor_accuracy(self, threshold=0.5):
        correct_trials = self.get_search_rts()[["target_present", "correct_response", "memory_set_size"]]
        accuracy = correct_trials["correct_response"].sum() / correct_trials["correct_response"].count()
        return accuracy < threshold
    
    def find_fixation_cutoff(self, percentile=1.0):
         # 1. Gather fixation counts
        fix_counts = [(trial.search_fixations().shape[0],trial.target_present, trial.memory_set_size) for trial in self.trials.values()]
        fix_counts = pd.DataFrame(fix_counts, columns=["fix_count", "target_present", "memory_set_size"])
        # Group fix_counts by target_present and memory_set_size
        grouped = fix_counts.groupby(["target_present", "memory_set_size"])["fix_count"] \
                   .apply(list) \
                   .reset_index(name="fix_counts_list")

        # 2. Compute total fixations & threshold per group
        grouped["total_fixations"] = grouped["fix_counts_list"].apply(sum)
        grouped["threshold"] = grouped["total_fixations"] * percentile
        grouped["max_possible"] = grouped["fix_counts_list"].apply(max)
        grouped["fix_cutoff"] = grouped.apply(
            lambda row: _find_fixation_cutoff(
                fix_count_list=row["fix_counts_list"], 
                threshold=row["threshold"], 
                max_possible=row["max_possible"]
            ), 
            axis=1
)

        return grouped[["target_present", "memory_set_size", "fix_cutoff"]]


    def cumulative_correct_trials_by_fixation(self,group_cutoffs=None):

        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()  

        records = []
        for trial in self.trials.values():
            scanpath_length = len(trial.search_fixations())
            fix_cuttoff = group_cutoffs[(group_cutoffs["memory_set_size"] == trial.memory_set_size) & (group_cutoffs["target_present"] == trial.target_present)]["fix_cutoff"].values[0]
            cumulative_correct = np.zeros(fix_cuttoff)

            if trial.correct_response and scanpath_length-1 <= fix_cuttoff:
                cumulative_correct[scanpath_length-1:] = 1


            records.append({
                "cumulative_correct" : cumulative_correct,
                "target_present": trial.target_present,
                "memory_set_size": trial.memory_set_size,
            })

        df = pd.DataFrame(records)


        return df
    
    def cumulative_performance_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_correct = self.cumulative_correct_trials_by_fixation(group_cutoffs)
        # the numpy arrays in cumulative correct should be "meaned" along the column for each row, so that for each group a single numpy array is obtained
        cumulative_performance = cumulative_correct.groupby(["memory_set_size", "target_present"])["cumulative_correct"].apply(lambda x: np.mean(x.values, axis=0)).reset_index()
        cumulative_performance_sem = cumulative_correct.groupby(["memory_set_size", "target_present"])["cumulative_correct"].apply(lambda x: np.std(x.values, axis=0) / np.sqrt(len(x))).reset_index()
        return cumulative_performance, cumulative_performance_sem
    
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
        self._was_answered = behavior_data.loc[behavior_data["trial_number"] == trial_number, "was_answered"].values[0]

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
        self.rts["stimulus"] = self._stimulus
        self.rts["target"] = self._target
        self.rts["was_answered"] = self._was_answered
        # Make sure the values are of the correct type


    def fixations(self):
        fixations = super().fixations()
        fixations["target_present"] = self._target_present
        fixations["correct_response"] = self._correct_response
        fixations["stimulus"] = self._stimulus
        fixations["memory_set_size"] = len(self._memory_set)
        fixations["target"] = self._target
        return fixations
    

    def saccades(self):
        saccades = super().saccades()
        saccades["target_present"] = self._target_present
        saccades["correct_response"] = self._correct_response
        saccades["stimulus"] = self._stimulus
        saccades["memory_set_size"] = len(self._memory_set)
        saccades["target"] = self._target
        return saccades

    def compute_multimatch(self,other_trial: "VisualSearchTrial",screen_height,screen_width):
        trial_scanpath = self.search_fixations()
        trial_to_compare_scanpath = other_trial.search_fixations()
        # Turn trial scanpath into list of tuples
        trial_scanpath = [tuple(row) for row in trial_scanpath[["xAvg", "yAvg", "duration"]].values]
        trial_to_compare_scanpath = [tuple(row) for row in trial_to_compare_scanpath[["xAvg", "yAvg", "duration"]].values]

        # Convert the list of tuples into a numpy array with the format needed for the multimatch function
        trial_scanpath = np.array(trial_scanpath, dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
        trial_to_compare_scanpath = np.array(trial_to_compare_scanpath, dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

        return mm.docomparison(trial_scanpath, trial_to_compare_scanpath, (screen_width, screen_height))

    def search_fixations(self):
        return self.fixations()[self._fix["phase"] == self._search_phase_name].sort_values(by="tStart")
    
    def memorization_fixations(self):
        return self.fixations()[self._fix["phase"] == self._memorization_phase_name].sort_values(by="tStart")
    
    def search_saccades(self):
        return self.saccades()[self._sacc["phase"] == self._search_phase_name].sort_values(by="tStart")
    
    def memorization_saccades(self):
        return self.saccades()[self._sacc["phase"] == self._memorization_phase_name].sort_values(by="tStart")
    
    def search_samples(self):
        return self.samples()[self._samples["phase"] == self._search_phase_name].sort_values(by="tSample")
    
    def memorization_samples(self):
        return self.samples()[self._samples["phase"] == self._memorization_phase_name].sort_values(by="tSample")
    
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