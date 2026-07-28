import ast
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pyxations.analysis.generic import (
    ITEMS_FOLDER,
    STIMULI_FOLDER,
    Experiment,
    Session,
    Subject,
    Trial,
    _find_fixation_cutoff,
    _partition_trials,
)
from pyxations.tables import read_tsv
from pyxations.visualization.visualization import Visualization


def _as(obj, typ):
    if isinstance(obj, typ):
        return obj
    return ast.literal_eval(obj)


def _grouped_accuracy(
    rts: pl.DataFrame,
    *,
    identifier: tuple[str, object] | None = None,
) -> pl.DataFrame:
    """Compute visual-search accuracy using the canonical Polars table."""

    accuracy = rts.group_by(["target_present", "memory_set_size"]).agg(
        pl.col("correct_response").mean().alias("accuracy")
    )
    if identifier is not None:
        column, value = identifier
        accuracy = accuracy.with_columns(pl.lit(value).alias(column))
    return accuracy


def _fixation_cutoffs(trials, percentile: float) -> pl.DataFrame:
    """Compute fixation-count cutoffs for visual-search trial groups."""

    if not 0 < percentile <= 1:
        raise ValueError("percentile must be greater than 0 and at most 1")

    rows = [
        {
            "fix_count": trial.search_fixations().height,
            "target_present": trial.target_present,
            "memory_set_size": trial.memory_set_size,
        }
        for trial in trials
    ]
    if not rows:
        return pl.DataFrame(
            schema={
                "target_present": pl.Boolean,
                "memory_set_size": pl.Int64,
                "fix_cutoff": pl.Int64,
            }
        )

    fixation_counts = pl.DataFrame(rows)
    cutoffs = []
    for group in fixation_counts.partition_by(
        ["target_present", "memory_set_size"], maintain_order=True
    ):
        counts = group.get_column("fix_count").to_list()
        cutoffs.append(
            {
                "target_present": group.get_column("target_present")[0],
                "memory_set_size": group.get_column("memory_set_size")[0],
                "fix_cutoff": _find_fixation_cutoff(
                    counts,
                    threshold=sum(counts) * percentile,
                    max_possible=max(counts, default=0),
                ),
            }
        )
    return pl.DataFrame(cutoffs)


def _cumulative_correct_by_fixation(
    trials, group_cutoffs: pl.DataFrame
) -> pl.DataFrame:
    """Build one cumulative-correctness sequence per trial."""

    records = []
    for trial in trials:
        cutoff_rows = group_cutoffs.filter(
            (pl.col("memory_set_size") == trial.memory_set_size)
            & (pl.col("target_present") == trial.target_present)
        )
        if cutoff_rows.height != 1:
            raise ValueError(
                "group_cutoffs must contain exactly one row for every trial group"
            )
        fixation_cutoff = int(cutoff_rows.get_column("fix_cutoff")[0])
        scanpath_length = trial.search_fixations().height
        cumulative_correct = np.zeros(fixation_cutoff)
        if (
            trial.correct_response
            and scanpath_length > 0
            and scanpath_length <= fixation_cutoff
        ):
            cumulative_correct[scanpath_length - 1 :] = 1
        records.append(
            {
                "cumulative_correct": cumulative_correct,
                "target_present": trial.target_present,
                "memory_set_size": trial.memory_set_size,
            }
        )
    return pl.DataFrame(records)


def _plot_grouped_mean_with_se(
    ax,
    data: pl.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
) -> None:
    """Plot grouped means with standard-error bands using Polars and Matplotlib.

    Category and group order follow their first appearance in ``data``. This
    intentionally preserves the ordering produced by the existing visual-search
    plotting methods.
    """
    if data.is_empty():
        return

    x_order = data.get_column(x).unique(maintain_order=True).to_list()
    group_order = data.get_column(group).unique(maintain_order=True).to_list()
    x_positions = {value: position for position, value in enumerate(x_order)}

    summary = (
        data.group_by([group, x], maintain_order=True)
        .agg(
            pl.col(y).mean().alias("mean"),
            pl.col(y).std(ddof=1).alias("std"),
            pl.len().alias("n"),
        )
        .with_columns((pl.col("std") / pl.col("n").sqrt()).alias("se"))
    )

    for group_value in group_order:
        group_data = summary.filter(pl.col(group) == group_value)
        group_summary = {
            row[x]: (row["mean"], row["se"]) for row in group_data.iter_rows(named=True)
        }
        present_categories = [value for value in x_order if value in group_summary]
        positions = np.asarray(
            [x_positions[value] for value in present_categories],
            dtype=float,
        )
        means = np.asarray(
            [group_summary[value][0] for value in present_categories],
            dtype=float,
        )
        standard_errors = np.asarray(
            [
                np.nan if group_summary[value][1] is None else group_summary[value][1]
                for value in present_categories
            ],
            dtype=float,
        )

        (line,) = ax.plot(positions, means, label=str(group_value))
        valid_error = np.isfinite(standard_errors)
        if valid_error.any():
            ax.fill_between(
                positions[valid_error],
                means[valid_error] - standard_errors[valid_error],
                means[valid_error] + standard_errors[valid_error],
                alpha=0.2,
                color=line.get_color(),
            )

    ax.set_xticks(np.arange(len(x_order)))
    ax.set_xticklabels([str(value) for value in x_order])
    ax.legend(title=group)


def _plot_cumulative_mean_with_se(
    ax,
    data: pl.DataFrame,
    *,
    max_fixations: int,
    values_column: str = "cumulative_correct",
) -> None:
    """Plot cumulative mean performance with standard-error bands.

    The input column is expected to contain one cumulative-performance sequence
    per trial or participant. Sequences are trimmed to ``max_fixations`` and
    shorter sequences are padded with missing values so they do not distort
    later fixation estimates.
    """
    if data.is_empty() or max_fixations <= 0:
        return

    sequences = []
    for values in data.get_column(values_column).to_list():
        if values is None:
            continue
        sequence = np.asarray(values, dtype=float).reshape(-1)[:max_fixations]
        if sequence.size == 0:
            continue
        padded = np.full(max_fixations, np.nan, dtype=float)
        padded[: sequence.size] = sequence
        sequences.append(padded)

    if not sequences:
        return

    matrix = np.vstack(sequences)
    valid_counts = np.sum(np.isfinite(matrix), axis=0)
    means = np.divide(
        np.nansum(matrix, axis=0),
        valid_counts,
        out=np.full(max_fixations, np.nan, dtype=float),
        where=valid_counts > 0,
    )

    standard_errors = np.full(max_fixations, np.nan, dtype=float)
    for index in np.flatnonzero(valid_counts > 1):
        standard_errors[index] = np.nanstd(matrix[:, index], ddof=1) / np.sqrt(
            valid_counts[index]
        )

    fixation_numbers = np.arange(1, max_fixations + 1, dtype=float)
    ax.plot(fixation_numbers, means, color="black")

    valid_error = np.isfinite(standard_errors) & np.isfinite(means)
    if valid_error.any():
        ax.fill_between(
            fixation_numbers[valid_error],
            means[valid_error] - standard_errors[valid_error],
            means[valid_error] + standard_errors[valid_error],
            color="black",
            alpha=0.2,
        )


def _plot_speed_accuracy_tradeoff(
    data: pl.DataFrame,
    *,
    entity_column: str,
    title: str,
) -> None:
    """Plot speed-accuracy points, paired lines, and marginal histograms."""
    if data.is_empty():
        return

    memory_set_sizes = data.get_column("memory_set_size").unique().sort().to_list()
    target_presence_values = (
        data.get_column("target_present").unique(maintain_order=True).to_list()
    )
    maximum_rt = data.get_column("rt").max()

    n_rows = len(memory_set_sizes)
    fig = plt.figure(figsize=(6, 1 + 6 * n_rows))
    grid = fig.add_gridspec(
        2 * n_rows,
        2,
        width_ratios=(4, 1),
        height_ratios=[1, 4] * n_rows,
        left=0.1,
        right=0.9,
        bottom=0.07,
        top=0.85,
        wspace=0.05,
        hspace=0.05,
    )

    for row_index, memory_set_size in enumerate(memory_set_sizes):
        subset = data.filter(pl.col("memory_set_size") == memory_set_size)
        top_row = 2 * row_index
        bottom_row = top_row + 1

        ax = fig.add_subplot(grid[bottom_row, 0])
        ax_hist_x = fig.add_subplot(grid[top_row, 0], sharex=ax)
        ax_hist_y = fig.add_subplot(grid[bottom_row, 1], sharey=ax)

        for target_present in target_presence_values:
            group_data = subset.filter(pl.col("target_present") == target_present)
            if group_data.is_empty():
                continue
            ax.scatter(
                group_data.get_column("accuracy").to_list(),
                group_data.get_column("rt").to_list(),
                label=str(target_present),
            )

        for entity_value in subset.get_column(entity_column).unique().to_list():
            entity_data = subset.filter(pl.col(entity_column) == entity_value)
            absent = entity_data.filter(pl.col("target_present") == False)
            present = entity_data.filter(pl.col("target_present") == True)
            if absent.height != 1 or present.height != 1:
                continue
            ax.plot(
                [absent.get_column("accuracy")[0], present.get_column("accuracy")[0]],
                [absent.get_column("rt")[0], present.get_column("rt")[0]],
                color="black",
                alpha=0.3,
                linewidth=0.5,
                zorder=0,
            )

        ax_hist_x.hist(
            subset.get_column("accuracy").to_list(),
            bins=np.linspace(0, 1, 21),
            color="gray",
        )
        ax_hist_y.hist(
            subset.get_column("rt").to_list(),
            bins=20,
            orientation="horizontal",
            color="gray",
        )

        ax_hist_x.tick_params(axis="x", labelbottom=False)
        ax_hist_y.tick_params(axis="y", labelleft=False)
        ax_hist_x.set_title(f"Memory Set Size {memory_set_size}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, maximum_rt * 1.1)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Mean RT (s)")
        ax.legend(title="target_present")

    plt.suptitle(title, fontsize=14)
    plt.show()
    plt.close()


def _plot_rt_bin_bars(
    ax,
    data: pl.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    hue_column: str | None = None,
    tick_stride: int = 3,
) -> None:
    """Plot RT-bin summaries using Polars and Matplotlib.

    RT bins are treated as ordered categories so their positions remain evenly
    spaced. When ``hue_column`` is provided, bars are grouped side by side and
    missing combinations are represented by zero-height bars.
    """
    if data.is_empty():
        return

    data = data.filter(pl.col("rt_bin").is_not_null())
    if data.is_empty():
        return

    rt_bins = data.get_column("rt_bin").unique().sort().to_list()
    positions = np.arange(len(rt_bins), dtype=float)

    if hue_column is None:
        values_by_bin = {
            row["rt_bin"]: row[value_column]
            for row in data.select(["rt_bin", value_column]).iter_rows(named=True)
        }
        heights = [values_by_bin.get(rt_bin, 0) for rt_bin in rt_bins]
        ax.bar(positions, heights)
    else:
        hue_values = data.get_column(hue_column).unique(maintain_order=True).to_list()
        bar_width = 0.8 / max(len(hue_values), 1)

        for hue_index, hue_value in enumerate(hue_values):
            hue_data = data.filter(pl.col(hue_column) == hue_value)
            values_by_bin = {
                row["rt_bin"]: row[value_column]
                for row in hue_data.select(["rt_bin", value_column]).iter_rows(
                    named=True
                )
            }
            heights = [values_by_bin.get(rt_bin, 0) for rt_bin in rt_bins]
            offset = (hue_index - (len(hue_values) - 1) / 2) * bar_width
            ax.bar(
                positions + offset,
                heights,
                width=bar_width,
                label=str(hue_value),
            )

        ax.legend(title=hue_column)

    tick_positions = np.arange(0, len(rt_bins), max(tick_stride, 1))
    tick_labels = [f"{rt_bins[index]:g}" for index in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("RT Bins (s)")
    ax.set_ylabel(ylabel)


class VisualSearchExperiment(Experiment):
    def __init__(
        self,
        dataset_path: str,
        search_phase_name: str,
        memorization_phase_name: str,
        excluded_subjects: list | None = None,
        excluded_sessions: dict | None = None,
        excluded_trials: dict | None = None,
    ):
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name
        super().__init__(
            dataset_path,
            excluded_subjects,
            excluded_sessions,
            excluded_trials,
        )

    def _create_subject(
        self,
        subject_id: str,
        old_subject_id: str,
        excluded_sessions: list,
        excluded_trials: dict,
    ):
        return VisualSearchSubject(
            subject_id,
            old_subject_id,
            self,
            self._search_phase_name,
            self._memorization_phase_name,
            excluded_sessions,
            excluded_trials,
        )

    def accuracy(self):
        accuracy = pl.concat([subject.accuracy() for subject in self.subjects.values()])

        return accuracy

    def plot_accuracy_by_subject(self):
        correct_responses = self.search_rts()
        correct_responses_aux = (
            correct_responses.group_by(
                ["subject_id", "memory_set_size", "target_present"]
            )
            .agg(pl.col("correct_response").mean().alias("correct_response_mean"))
            .select(
                [
                    "subject_id",
                    "memory_set_size",
                    "target_present",
                    "correct_response_mean",
                ]
            )
        )
        correct_responses = (
            correct_responses.join(
                correct_responses_aux,
                on=["subject_id", "memory_set_size", "target_present"],
                how="left",
            )
            .with_columns(pl.col("target_present").cast(pl.Boolean))
            .sort(by=["memory_set_size", "target_present", "correct_response_mean"])
        )

        mem_set_sizes = (
            correct_responses.get_column("memory_set_size").unique().sort().to_list()
        )
        width_size = max(
            0.25 * correct_responses.get_column("subject_id").n_unique(), 10
        )

        n_rows = len(mem_set_sizes)
        _, axs = plt.subplots(n_rows, 1, figsize=(width_size, 5 * n_rows), sharey=True)

        if n_rows == 1:
            axs = np.array([axs])

        for i, memory_set_size in enumerate(mem_set_sizes):
            data = correct_responses.filter(
                pl.col("memory_set_size") == memory_set_size
            )
            _plot_grouped_mean_with_se(
                axs[i],
                data,
                x="subject_id",
                y="correct_response",
                group="target_present",
            )
            axs[i].set_title(f"Memory Set Size {memory_set_size}")
            axs[i].tick_params(axis="x", rotation=90)
            axs[i].set_xlabel("Subject ID")
            axs[i].set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_accuracy_by_stimulus(self):
        correct_responses = self.search_rts()
        correct_responses_aux = (
            correct_responses.group_by(
                ["stimulus", "memory_set_size", "target_present"]
            )
            .agg(pl.col("correct_response").mean().alias("correct_response_mean"))
            .select(
                [
                    "stimulus",
                    "memory_set_size",
                    "target_present",
                    "correct_response_mean",
                ]
            )
        )
        correct_responses = (
            correct_responses.join(
                correct_responses_aux,
                on=["stimulus", "memory_set_size", "target_present"],
                how="left",
            )
            .with_columns(pl.col("target_present").cast(pl.Boolean))
            .sort(by=["memory_set_size", "target_present", "correct_response_mean"])
        )

        mem_set_sizes = (
            correct_responses.get_column("memory_set_size").unique().sort().to_list()
        )
        n_rows = len(mem_set_sizes)
        width_size = max(0.25 * correct_responses.get_column("stimulus").n_unique(), 10)

        _, axs = plt.subplots(n_rows, 1, figsize=(width_size, 5 * n_rows), sharey=True)

        if n_rows == 1:
            axs = np.array([axs])

        for i, memory_set_size in enumerate(mem_set_sizes):
            data = correct_responses.filter(
                pl.col("memory_set_size") == memory_set_size
            )
            _plot_grouped_mean_with_se(
                axs[i],
                data,
                x="stimulus",
                y="correct_response",
                group="target_present",
            )
            axs[i].set_title(f"Memory Set Size {memory_set_size}")
            axs[i].tick_params(axis="x", rotation=90)
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
        speed_accuracy = (
            self.search_rts()
            .group_by(["target_present", "memory_set_size", "subject_id"])
            .agg(
                pl.col("rt").mean().alias("rt"),
                pl.col("correct_response").mean().alias("accuracy"),
            )
            .with_columns(
                (pl.col("rt") / 1000).alias("rt"),
                pl.col("target_present").cast(pl.Boolean),
            )
            .sort("memory_set_size")
        )
        _plot_speed_accuracy_tradeoff(
            speed_accuracy,
            entity_column="subject_id",
            title="Speed-Accuracy Tradeoff by Subject",
        )

    def plot_speed_accuracy_tradeoff_by_stimulus(self):
        speed_accuracy = (
            self.search_rts()
            .group_by(["target_present", "memory_set_size", "stimulus"])
            .agg(
                pl.col("rt").mean().alias("rt"),
                pl.col("correct_response").mean().alias("accuracy"),
            )
            .with_columns(
                (pl.col("rt") / 1000).alias("rt"),
                pl.col("target_present").cast(pl.Boolean),
            )
            .sort("memory_set_size")
        )
        _plot_speed_accuracy_tradeoff(
            speed_accuracy,
            entity_column="stimulus",
            title="Speed-Accuracy Tradeoff by Stimulus",
        )

    def remove_non_answered_trials(self, print_flag=True):
        amount_trials_before_removal = self.search_rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.remove_non_answered_trials(False)

        if print_flag:
            print(
                f"Removed {amount_trials_before_removal - self.search_rts().shape[0]} non answered trials"
            )

    def remove_poor_accuracy_sessions(self, threshold=0.5, print_flag=True):
        amount_sessions_total = sum(
            [len(subject.sessions) for subject in self.subjects.values()]
        )
        for subject in list(self.subjects.keys()):
            self.subjects[subject].remove_poor_accuracy_sessions(threshold, False)

        if print_flag:
            print(
                f"Removed {amount_sessions_total - sum([len(subject.sessions) for subject in self.subjects.values()])} sessions with poor accuracy"
            )

    def scanpaths_by_stimuli(self):
        return pl.concat(
            [subject.scanpaths_by_stimuli() for subject in self.subjects.values()]
        )

    def find_fixation_cutoff(self, percentile=1.0):
        return _fixation_cutoffs(
            (
                trial
                for subject in self.subjects.values()
                for session in subject.sessions.values()
                for trial in session.trials.values()
            ),
            percentile,
        )

    def remove_trials_for_stimuli(self, stimuli, print_flag=True):
        """
        Remove trials for stimuli that are in the list of stimuli.
        Parameters:
            - stimuli: list of stimuli to remove
            - print_flag: if True, print the number of trials removed
        """
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

    def remove_trials_for_stimuli_with_poor_accuracy(
        self, threshold=0.5, print_flag=True
    ):
        scanpaths_by_stimuli = self.scanpaths_by_stimuli()
        grouped = scanpaths_by_stimuli.group_by(
            ["stimulus", "memory_set_size", "target_present"]
        )
        poor_accuracy_stimuli = grouped.agg(
            pl.col("correct_response").mean().alias("accuracy")
        ).filter(pl.col("accuracy") < threshold)
        poor_accuracy_stimuli = poor_accuracy_stimuli.select(
            "stimulus", "memory_set_size", "target_present"
        ).iter_rows()
        poor_accuracy_stimuli = set(poor_accuracy_stimuli)
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
                    if (
                        trial.stimulus,
                        trial.memory_set_size,
                        trial.target_present,
                    ) in poor_accuracy_stimuli:
                        session.remove_trial(trial_key)
                        amount_trials_removed += 1
        if print_flag:
            print(
                f"Removed {amount_trials_removed} trials from stimuli with less than {threshold} accuracy."
            )

    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        cumulative_correct = pl.concat(
            [
                subject.cumulative_correct_trials_by_fixation(group_cutoffs)
                for subject in self.subjects.values()
            ]
        )

        return cumulative_correct

    def plot_cumulative_performance(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()

        cumulative_performance = self.cumulative_correct_trials_by_fixation(
            group_cutoffs
        ).join(
            group_cutoffs,
            on=["target_present", "memory_set_size"],
            how="left",
        )

        target_presence_values = (
            cumulative_performance.select("target_present")
            .unique()
            .to_series()
            .to_list()
        )
        memory_set_sizes = (
            cumulative_performance.select("memory_set_size")
            .unique()
            .to_series()
            .to_list()
        )

        n_cols = len(target_presence_values)
        n_rows = len(memory_set_sizes)
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            sharey=True,
        )
        fig.suptitle("Cumulative Performance")
        axs = np.asarray(axs, dtype=object).reshape(n_rows, n_cols)

        for row_index, memory_set_size in enumerate(memory_set_sizes):
            for col_index, target_present in enumerate(target_presence_values):
                data = cumulative_performance.filter(
                    (pl.col("memory_set_size") == memory_set_size)
                    & (pl.col("target_present") == target_present)
                )
                if data.is_empty():
                    continue

                max_fixations = int(data.get_column("fix_cutoff")[0])
                ax = axs[row_index, col_index]
                _plot_cumulative_mean_with_se(
                    ax,
                    data,
                    max_fixations=max_fixations,
                )
                ax.set_title(
                    f"Memory Set Size {int(memory_set_size)}, "
                    f"Target Present {bool(target_present)}"
                )
                ax.set_xticks(range(0, max_fixations, 5))
                ax.set_xticklabels(range(1, max_fixations + 1, 5))
                ax.set_xlabel("Fixation Number")
                ax.set_ylabel("Accuracy")

        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        plt.close()

    def trials_by_rt_bins(self, bin_end, bin_step):
        if bin_end <= 0:
            raise ValueError("bin_end must be greater than zero")
        if bin_step <= 0:
            raise ValueError("bin_step must be greater than zero")

        # 1. Get and filter RTs
        rts = self.rts().filter(pl.col("phase") == self._search_phase_name)
        rts = rts.with_columns([(pl.col("rt") / 1000).alias("rt")])

        # 2. Compute bin edges
        bin_edges = np.arange(0, bin_end + bin_step, bin_step)

        # 3. Bin RTs using numpy (returns indices)
        bin_indices = np.digitize(rts["rt"].to_numpy(), bin_edges, right=False)

        # 4. Convert to left edge values
        rt_bin_labels = [
            bin_edges[i - 1] if i > 0 and i < len(bin_edges) else None
            for i in bin_indices
        ]

        # 5. Assign back to the DataFrame
        rts = rts.with_columns([pl.Series("rt_bin", rt_bin_labels)])

        return rts

    def plot_correct_trials_by_rt_bins(self, bin_end, bin_step):
        correct_trials_per_bin = (
            self.trials_by_rt_bins(bin_end, bin_step)
            .select(["rt_bin", "target_present", "memory_set_size", "correct_response"])
            .group_by(["rt_bin", "target_present", "memory_set_size"])
            .agg(pl.col("correct_response").sum().alias("correct_response"))
            .sort(["memory_set_size", "target_present", "rt_bin"])
        )

        tp_ta = sorted(
            correct_trials_per_bin.get_column("target_present").unique().to_list()
        )
        mem_set_sizes = sorted(
            correct_trials_per_bin.get_column("memory_set_size").unique().to_list()
        )

        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            sharey=True,
            sharex=True,
        )
        fig.suptitle("Correct Trials by RT Bins")
        axs = np.asarray(axs, dtype=object).reshape(n_rows, n_cols)

        for i, mem_size in enumerate(mem_set_sizes):
            for j, tp in enumerate(tp_ta):
                data = correct_trials_per_bin.filter(
                    (pl.col("memory_set_size") == mem_size)
                    & (pl.col("target_present") == tp)
                )
                _plot_rt_bin_bars(
                    axs[i, j],
                    data,
                    value_column="correct_response",
                    ylabel="Correct Trials",
                )
                axs[i, j].set_title(
                    f"Memory Set Size {mem_size}, Target Present {bool(tp)}"
                )

        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_incorrect_trials_by_rt_bins(self, bin_end, bin_step):
        incorrect_trials_per_bin = (
            self.trials_by_rt_bins(bin_end, bin_step)
            .select(["rt_bin", "target_present", "memory_set_size", "correct_response"])
            .with_columns((1 - pl.col("correct_response")).alias("incorrect_response"))
            .group_by(["rt_bin", "target_present", "memory_set_size"])
            .agg(pl.col("incorrect_response").sum().alias("incorrect_response"))
            .sort(["memory_set_size", "target_present", "rt_bin"])
        )

        tp_ta = sorted(
            incorrect_trials_per_bin.get_column("target_present").unique().to_list()
        )
        mem_set_sizes = sorted(
            incorrect_trials_per_bin.get_column("memory_set_size").unique().to_list()
        )

        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            sharey=True,
            sharex=True,
        )
        fig.suptitle("Incorrect Trials by RT Bins")
        axs = np.asarray(axs, dtype=object).reshape(n_rows, n_cols)

        for i, mem_size in enumerate(mem_set_sizes):
            for j, tp in enumerate(tp_ta):
                data = incorrect_trials_per_bin.filter(
                    (pl.col("memory_set_size") == mem_size)
                    & (pl.col("target_present") == tp)
                )
                _plot_rt_bin_bars(
                    axs[i, j],
                    data,
                    value_column="incorrect_response",
                    ylabel="Incorrect Trials",
                )
                axs[i, j].set_title(
                    f"Memory Set Size {mem_size}, Target Present {bool(tp)}"
                )

        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_probability_of_deciding_by_rt_bin(self, bin_end, bin_step):
        trials = self.trials_by_rt_bins(bin_end, bin_step).select(
            ["rt_bin", "target_present", "memory_set_size", "correct_response"]
        )

        tp_ta = sorted(trials.get_column("target_present").unique().to_list())
        mem_set_sizes = sorted(trials.get_column("memory_set_size").unique().to_list())

        n_cols = len(tp_ta)
        n_rows = len(mem_set_sizes)

        grouped = (
            trials.group_by(
                ["rt_bin", "target_present", "correct_response", "memory_set_size"]
            )
            .agg(pl.len().alias("count"))
            .sort(["correct_response", "target_present", "memory_set_size", "rt_bin"])
        )

        totals = grouped.group_by(
            ["correct_response", "target_present", "memory_set_size"]
        ).agg(pl.col("count").sum().alias("total_per_group"))

        grouped = grouped.join(
            totals,
            on=["correct_response", "target_present", "memory_set_size"],
            how="left",
        )

        grouped = grouped.with_columns(
            pl.col("count")
            .cum_sum()
            .over(["correct_response", "target_present", "memory_set_size"])
            .alias("cumsum")
        ).with_columns(
            (pl.col("total_per_group") - pl.col("cumsum") + pl.col("count")).alias(
                "total_per_bin"
            ),
            (
                pl.col("count")
                / (pl.col("total_per_group") - pl.col("cumsum") + pl.col("count"))
            ).alias("count_normalized"),
            pl.col("correct_response").cast(pl.Boolean),
        )

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 5 * n_rows),
            sharey=True,
            sharex=True,
        )
        fig.suptitle("Probability of Deciding by RT Bins")
        axs = np.asarray(axs, dtype=object).reshape(n_rows, n_cols)

        for i, mem_size in enumerate(mem_set_sizes):
            for j, tp in enumerate(tp_ta):
                data = grouped.filter(
                    (pl.col("memory_set_size") == mem_size)
                    & (pl.col("target_present") == tp)
                )
                _plot_rt_bin_bars(
                    axs[i, j],
                    data,
                    value_column="count_normalized",
                    ylabel="Probability of Deciding",
                    hue_column="correct_response",
                )
                axs[i, j].set_title(
                    f"Memory Set Size {mem_size}, Target Present {bool(tp)}"
                )

        plt.tight_layout()
        plt.show()
        plt.close()


class VisualSearchSubject(Subject):
    def __init__(
        self,
        subject_id: str,
        old_subject_id: str,
        experiment: VisualSearchExperiment,
        search_phase_name,
        memorization_phase_name,
        excluded_sessions: list | None = None,
        excluded_trials: dict | None = None,
    ):
        super().__init__(
            subject_id, old_subject_id, experiment, excluded_sessions, excluded_trials
        )
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name

    def _create_session(self, session_id: str):
        return VisualSearchSession(
            session_id,
            self,
            self._search_phase_name,
            self._memorization_phase_name,
            self.excluded_trials.get(session_id, {}),
        )

    def scanpaths_by_stimuli(self):
        return pl.concat(
            [session.scanpaths_by_stimuli() for session in self.sessions.values()]
        )

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
        return _grouped_accuracy(
            self.search_rts(),
            identifier=("subject_id", self.subject_id),
        )

    def remove_non_answered_trials(self, print_flag=True):
        # Remove non answered trials from all sessions
        amount_trials_before_removal = self.search_rts().height
        for session in list(self.sessions.values()):
            session.remove_non_answered_trials(False)

        if print_flag:
            print(
                f"Removed {amount_trials_before_removal - self.search_rts().height} non answered trials from subject {self.subject_id}"
            )

    def find_fixation_cutoff(self, percentile=1.0):
        return _fixation_cutoffs(
            (
                trial
                for session in self.sessions.values()
                for trial in session.trials.values()
            ),
            percentile,
        )

    def remove_poor_accuracy_sessions(self, threshold=0.5, print_flag=True):
        poor_accuracy_sessions = 0
        keys = list(self.sessions.keys())
        for key in keys:
            session = self.sessions[key]
            if session.has_poor_accuracy(threshold):
                poor_accuracy_sessions += 1
                self.remove_session(key)

        if print_flag:
            print(
                f"Removed {poor_accuracy_sessions} sessions with poor accuracy from subject {self.subject_id}"
            )

    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()

        cumulative_correct = pl.concat(
            [
                session.cumulative_correct_trials_by_fixation(group_cutoffs)
                for session in self.sessions.values()
            ]
        )
        return cumulative_correct


class VisualSearchSession(Session):
    BEH_COLUMNS: ClassVar[list[str]] = [
        "trial_number",
        "stimulus",
        "stimulus_coords",
        "memory_set",
        "memory_set_locations",
        "target_present",
        "target",
        "target_location",
        "correct_response",
        "was_answered",
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

    COLLECTION_COLUMNS: ClassVar[dict[str, type]] = {
        "stimulus_coords": tuple,  # Parse as a tuple
        "memory_set": list,  # Parse as a list
        "memory_set_locations": list,  # Parse as a list of tuples
        "target_location": tuple,  # Parse as a tuple
    }

    def __init__(
        self,
        session_id: str,
        subject: VisualSearchSubject,
        search_phase_name: str,
        memorization_phase_name: str,
        excluded_trials: list | None = None,
    ):
        excluded_trials = [] if excluded_trials is None else excluded_trials
        super().__init__(session_id, subject, excluded_trials)
        self._search_phase_name = search_phase_name
        self._memorization_phase_name = memorization_phase_name
        self.behavior_data = None

    def load_behavior_data(self):
        behavior_path = self.session_dataset_path / "beh"
        behavior_files = sorted(behavior_path.glob("*_events.tsv"))
        if not behavior_files:
            raise ValueError(
                f"No BIDS events.tsv file was found for session "
                f"{self.session_id} of subject {self.subject().subject_id}."
            )
        tables = [
            read_tsv(
                path,
                has_header=True,
                schema_overrides={
                    "trial_number": pl.Int32,
                    "stimulus": pl.Utf8,
                    "target_present": pl.Int32,
                    "target": pl.Utf8,
                    "correct_response": pl.Int32,
                    "was_answered": pl.Int32,
                },
            )
            for path in behavior_files
        ]
        self.behavior_data = (
            pl.concat(tables, how="diagonal_relaxed") if len(tables) > 1 else tables[0]
        )

        # Validate that all required columns are present
        missing_columns = set(self.BEH_COLUMNS) - set(self.behavior_data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing columns in BIDS events data: {missing_columns} "
                f"for session {self.session_id} of subject "
                f"{self.subject().subject_id}"
            )

    def _init_trials(self, samples, fix, sacc, blink, events_path):
        sample_trials = _partition_trials(samples)
        fixation_trials = _partition_trials(fix)
        saccade_trials = _partition_trials(sacc)
        blink_trials = _partition_trials(blink)
        behavior_trials = _partition_trials(self.behavior_data)
        empty_fix = fix.head(0)
        empty_sacc = sacc.head(0)
        empty_blink = blink.head(0) if blink is not None else None
        self._trials = {
            trial: VisualSearchTrial(
                trial,
                self,
                sample_rows,
                fixation_trials.get(trial, empty_fix),
                saccade_trials.get(trial, empty_sacc),
                blink_trials.get(trial, empty_blink),
                events_path,
                behavior_trials[trial],
                self._search_phase_name,
                self._memorization_phase_name,
                prefiltered=True,
            )
            for trial, sample_rows in sample_trials.items()
            if (
                trial != -1
                and trial not in self.excluded_trials
                and trial in behavior_trials
            )
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
        return _grouped_accuracy(
            self.search_rts(),
            identifier=("session_id", self.session_id),
        )

    def remove_non_answered_trials(self, print_flag=True):
        # Remove trials that were not answered
        non_answered_trials = [
            trial for trial in self.trials if not self.trials[trial].was_answered
        ]
        for trial in non_answered_trials:
            self.remove_trial(trial)
        if print_flag:
            print(
                f"Removed {len(non_answered_trials)} non answered trials from session {self.session_id}"
            )

    def has_poor_accuracy(self, threshold=0.5):
        responses = self.search_rts().get_column("correct_response")
        return responses.is_empty() or responses.mean() < threshold

    def find_fixation_cutoff(self, percentile=1.0):
        return _fixation_cutoffs(self.trials.values(), percentile)

    def cumulative_correct_trials_by_fixation(self, group_cutoffs=None):
        if group_cutoffs is None:
            group_cutoffs = self.find_fixation_cutoff()
        return _cumulative_correct_by_fixation(self.trials.values(), group_cutoffs)

    def scanpaths_by_stimuli(self):
        return pl.DataFrame(
            [trial.scanpath_by_stimuli() for trial in self.trials.values()]
        )


class VisualSearchTrial(Trial):
    def __init__(
        self,
        trial_number,
        session,
        samples,
        fix,
        sacc,
        blink,
        events_path,
        behavior_data,
        search_phase_name,
        memorization_phase_name,
        prefiltered=False,
    ):
        super().__init__(
            trial_number,
            session,
            samples,
            fix,
            sacc,
            blink,
            events_path,
            prefiltered=prefiltered,
        )

        trial_data = (
            behavior_data
            if prefiltered
            else behavior_data.filter(pl.col("trial_number") == trial_number)
        )

        self._target_present = bool(trial_data.select("target_present").item())
        self._target = trial_data.select("target").item()

        self._target_location = None
        if self._target_present:
            self._target_location = _as(
                trial_data.select("target_location").item(), tuple
            )

        self._correct_response = bool(trial_data.select("correct_response").item())
        self._stimulus = trial_data.select("stimulus").item()
        self._stimulus_coords = _as(trial_data.select("stimulus_coords").item(), tuple)

        self._memory_set = _as(trial_data.select("memory_set").item(), list)
        self._memory_set_locations = _as(
            trial_data.select("memory_set_locations").item(), list
        )
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
            filtered.group_by("phase")
            .agg((pl.col("tSample").max() - pl.col("tSample").min()).alias("rt"))
            .with_columns(
                [
                    pl.lit(self.trial_number).alias("trial_number"),
                    pl.lit(len(self._memory_set)).alias("memory_set_size"),
                    pl.lit(self._target_present).alias("target_present"),
                    pl.lit(self._correct_response).alias("correct_response"),
                    pl.lit(self._stimulus).alias("stimulus"),
                    pl.lit(self._target).alias("target"),
                    pl.lit(self._was_answered).alias("was_answered"),
                ]
            )
        )

    def fixations(self):
        fixations = (
            super()
            .fixations()
            .with_columns(
                [
                    pl.lit(self._target_present).alias("target_present"),
                    pl.lit(self._correct_response).alias("correct_response"),
                    pl.lit(self._stimulus).alias("stimulus"),
                    pl.lit(self._target).alias("target"),
                    pl.lit(self._memory_set).alias("memory_set"),
                ]
            )
        )
        return fixations

    def saccades(self):
        saccades = (
            super()
            .saccades()
            .with_columns(
                [
                    pl.lit(self._target_present).alias("target_present"),
                    pl.lit(self._correct_response).alias("correct_response"),
                    pl.lit(self._stimulus).alias("stimulus"),
                    pl.lit(self._target).alias("target"),
                    pl.lit(self._memory_set).alias("memory_set"),
                ]
            )
        )
        return saccades

    def search_fixations(self):
        return (
            self.fixations()
            .filter(pl.col("phase") == self._search_phase_name)
            .sort(by="tStart")
        )

    def _multimatch_fixations(self) -> pl.DataFrame:
        return self.search_fixations()

    def memorization_fixations(self):
        return (
            self.fixations()
            .filter(pl.col("phase") == self._memorization_phase_name)
            .sort(by="tStart")
        )

    def search_saccades(self):
        return (
            self.saccades()
            .filter(pl.col("phase") == self._search_phase_name)
            .sort(by="tStart")
        )

    def memorization_saccades(self):
        return (
            self.saccades()
            .filter(pl.col("phase") == self._memorization_phase_name)
            .sort(by="tStart")
        )

    def search_samples(self):
        return (
            self.samples()
            .filter(pl.col("phase") == self._search_phase_name)
            .sort(by="tSample")
        )

    def memorization_samples(self):
        return (
            self.samples()
            .filter(pl.col("phase") == self._memorization_phase_name)
            .sort(by="tSample")
        )

    def scanpath_by_stimuli(self):
        return {
            "fixations": self.search_fixations(),
            "stimulus": self._stimulus,
            "correct_response": self._correct_response,
            "target_present": self._target_present,
            "memory_set_size": len(self._memory_set),
        }

    def plot_scanpath(self, screen_height, screen_width, **kwargs):
        """
        Plots the scanpath of the trial. The scanpath will be plotted in two phases: the search phase and the memorization phase.
        The search phase will be plotted with the stimulus and the memorization phase will be plotted with the items memorized by the participant.
        The search phase will have the fixations and saccades of the trial, while the memorization phase will only have the fixations.
        The names of the phases should be the same ones used in the computation of the derivatives.
        If you don't really care about the memorization phase, you can pass None as an argument.

        """
        vis = Visualization(self.events_path, self.detection_algorithm)
        self.events_path.mkdir(parents=True, exist_ok=True)

        phase_data = {self._search_phase_name: {}, self._memorization_phase_name: {}}
        dataset_parent_folder = self.session.session_dataset_path.parents[1]
        phase_data[self._search_phase_name]["img_paths"] = [
            dataset_parent_folder / STIMULI_FOLDER / self._stimulus
        ]
        phase_data[self._search_phase_name]["img_plot_coords"] = [self._stimulus_coords]
        if self._memorization_phase_name is not None:
            phase_data[self._memorization_phase_name]["img_paths"] = [
                dataset_parent_folder / ITEMS_FOLDER / img for img in self._memory_set
            ]
            phase_data[self._memorization_phase_name]["img_plot_coords"] = (
                self._memory_set_locations
            )

        # If the target is present add the "bbox" to the search_phase phase as a key-value pair
        if self._target_present:
            phase_data[self._search_phase_name]["bbox"] = self._target_location
        vis.scanpath(
            fixations=self._fix,
            phase_data=phase_data,
            saccades=self._sacc,
            samples=self._samples,
            screen_height=screen_height,
            screen_width=screen_width,
            folder_path=self.events_path,
            **kwargs,
        )

    def plot_animation(
        self,
        screen_height,
        screen_width,
        video_path=None,
        background_image_path=None,
        **kwargs,
    ):
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
            - output_format: "matplotlib" (default), "html", "mp4", or "gif"
            - display: If True, return HTML for notebook display

        Returns
        -------
        IPython.display.HTML or None
            Returns HTML animation if display=True and output_format="html".
            For output_format="matplotlib", displays in a GUI window and returns None.
        """
        vis = Visualization(self.events_path, self.detection_algorithm)
        self.events_path.mkdir(parents=True, exist_ok=True)

        # Set default folder_path if not provided
        if "folder_path" not in kwargs:
            kwargs["folder_path"] = self.events_path

        # If no background image provided and no video, try to use the stimulus
        if video_path is None and background_image_path is None:
            dataset_parent_folder = self.session.session_dataset_path.parents[1]
            stimulus_path = dataset_parent_folder / STIMULI_FOLDER / self._stimulus
            if stimulus_path.exists():
                background_image_path = stimulus_path

        return vis.plot_animation(
            samples=self._samples,
            screen_height=screen_height,
            screen_width=screen_width,
            video_path=video_path,
            background_image_path=background_image_path,
            **kwargs,
        )
