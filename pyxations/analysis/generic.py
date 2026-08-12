import warnings
import weakref
from dataclasses import dataclass
from importlib import import_module
from math import hypot
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm

from pyxations.export.bids import BIDSDerivativeExport
from pyxations.tables import read_tsv
from pyxations.visualization.visualization import Visualization

STIMULI_FOLDER = "stimuli"
ITEMS_FOLDER = "items"


@dataclass(frozen=True)
class SessionQualityAssessment:
    """Bad-trial assessment made before a session is modified."""

    bad_trials: tuple[object, ...]
    total_trials: int

    @property
    def bad_trial_fraction(self) -> float:
        """Fraction of assessed trials that were flagged as bad.

        Returns
        -------
        float
            ``len(bad_trials) / total_trials``, or ``0.0`` when the session
            holds no trials.
        """
        if self.total_trials == 0:
            return 0.0
        return len(self.bad_trials) / self.total_trials


@dataclass(frozen=True)
class QualityFilterResult:
    """Changes made by a bad-trial and bad-session quality filter."""

    bad_trials_removed: int = 0
    sessions_removed: int = 0
    subjects_removed: int = 0
    trials_discarded_with_sessions: int = 0

    def __add__(self, other):
        if not isinstance(other, QualityFilterResult):
            return NotImplemented
        return QualityFilterResult(
            bad_trials_removed=self.bad_trials_removed + other.bad_trials_removed,
            sessions_removed=self.sessions_removed + other.sessions_removed,
            subjects_removed=self.subjects_removed + other.subjects_removed,
            trials_discarded_with_sessions=(
                self.trials_discarded_with_sessions
                + other.trials_discarded_with_sessions
            ),
        )


_MULTIMATCH_DTYPE = np.dtype(
    [
        ("start_x", "<f8"),
        ("start_y", "<f8"),
        ("duration", "<f8"),
    ]
)


def _load_multimatch():
    """Import the optional MultiMatch dependency with an actionable error."""
    try:
        return import_module("multimatch_gaze")
    except ImportError as exc:
        raise ImportError(
            "MultiMatch support is optional. Install it with "
            "`pip install 'pyxations[multimatch]'`."
        ) from exc


def _to_multimatch_scanpath(fixations: pl.DataFrame) -> np.ndarray:
    """Convert fixation data to MultiMatch's structured NumPy format."""
    required_columns = ("xAvg", "yAvg", "duration")
    missing_columns = [
        column for column in required_columns if column not in fixations.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Cannot compute MultiMatch: missing fixation columns: {missing}."
        )

    scanpath = np.empty(fixations.height, dtype=_MULTIMATCH_DTYPE)
    for source, target in zip(
        required_columns, ("start_x", "start_y", "duration"), strict=True
    ):
        scanpath[target] = fixations.get_column(source).cast(pl.Float64).to_numpy()

    return scanpath


def _collect_frames(
    children,
    accessor: str,
    *,
    identifier: tuple[str, object] | None = None,
) -> pl.DataFrame:
    """Collect one table from a hierarchy level without Python row copies."""

    frames = [getattr(child, accessor)() for child in children]
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    if identifier is not None:
        column, value = identifier
        combined = combined.with_columns(pl.lit(value).alias(column))
    return combined


def _partition_trials(
    frame: pl.DataFrame | None,
) -> dict[object, pl.DataFrame]:
    """Partition a table once and index the results by trial number."""

    if frame is None or frame.is_empty():
        return {}
    return {
        group.get_column("trial_number")[0]: group
        for group in frame.partition_by("trial_number", maintain_order=True)
    }


def _find_fixation_cutoff(fix_count_list, threshold, max_possible):
    """Return the smallest fixation count that reaches ``threshold``.

    Each trial contributes at most the candidate number of fixations. The
    returned value is a count, not a zero-based index.
    """

    counts = [max(0, int(count)) for count in fix_count_list]
    if not counts or max_possible <= 0:
        return 0

    maximum = max(0, int(max_possible))
    target = max(0.0, float(threshold))
    for candidate in range(1, maximum + 1):
        if sum(min(count, candidate) for count in counts) >= target:
            return candidate
    return maximum


def _parse_validations(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse EyeLink `!CAL VALIDATION …` lines that are stored in df["line"].
    Returns a tidy DataFrame with numeric columns ready for plotting.
    """
    df = df.filter(pl.col("line").str.contains("CAL VALIDATION")).select(
        ["line", "Calib_index"]
    )
    # column "line" does not contain "ABORTED"

    # 0 · remove the "ABORTED" lines (if any)
    df = df.filter(~pl.col("line").str.contains("ABORTED"))

    # 1 · pull the pieces out with .str.extract
    parsed = df.with_columns(
        [
            # time‑stamp after the initial MSG token
            pl.col("line")
            .str.extract(r"MSG\s+(\d+)", 1)
            .cast(pl.Int64)
            .alias("timestamp"),
            # eye label (LEFT / RIGHT)
            pl.col("line").str.extract(r"\s(LEFT|RIGHT)\s", 1).alias("eye"),
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
        ]
    )

    # 2 · create a validation index (0‑based) within each calibration block
    parsed = (
        parsed.with_columns(
            pl.col("line")
            .cum_count()  # running 0, 1, 2, …
            .over(["Calib_index", "eye"])  # reset counter per calibration × eye
            .alias("validation_id")
        )
        .drop("line")
        .sort(["Calib_index", "eye", "validation_id"])
    )

    return parsed


class Experiment:
    """Top level of the analysis hierarchy for one BIDS dataset.

    An ``Experiment`` reads ``participants.tsv`` from the raw BIDS dataset and
    builds a :class:`Subject` for every participant that is not excluded. The
    matching ``*_derivatives`` dataset is located automatically from
    ``dataset_path``; the derivative tables themselves are only read once
    :meth:`load_data` is called.

    The hierarchy is ``Experiment -> Subject -> Session -> Trial``. Table
    accessors such as :meth:`fixations` are available at every level and always
    return a Polars DataFrame with the identifiers of that level attached.

    Parameters
    ----------
    dataset_path : str
        Path to the **raw** BIDS dataset. The derivatives dataset is expected
        as a sibling directory with the ``_derivatives`` suffix.
    excluded_subjects : list, optional
        Subject identifiers to skip. Matched against both the BIDS
        ``subject_id`` and the original ``old_subject_id``.
    excluded_sessions : dict, optional
        Mapping of ``subject_id`` to a list of session identifiers to skip.
    excluded_trials : dict, optional
        Mapping of ``subject_id`` to a ``{session_id: [trial_number, ...]}``
        mapping of trials to skip.

    Attributes
    ----------
    dataset_path : pathlib.Path
        Root of the raw BIDS dataset.
    derivatives_path : pathlib.Path
        Root of the linked BIDS Derivatives dataset.
    metadata : polars.DataFrame
        Contents of ``participants.tsv``.
    subjects : dict
        Mapping of ``subject_id`` to :class:`Subject`.

    Examples
    --------
    >>> exp = Experiment(dataset_path="generated/example_dataset")
    >>> exp.load_data("eyelink")
    >>> exp.fixations().height  # doctest: +SKIP
    1234
    """

    def __init__(
        self,
        dataset_path: str,
        excluded_subjects: list | None = None,
        excluded_sessions: dict | None = None,
        excluded_trials: dict | None = None,
    ):
        excluded_subjects = excluded_subjects or []
        excluded_sessions = excluded_sessions or {}
        excluded_trials = excluded_trials or {}
        self.dataset_path = Path(dataset_path)
        self.derivatives_path = self.dataset_path.with_name(
            self.dataset_path.name + "_derivatives"
        )
        self.metadata = read_tsv(
            self.dataset_path / "participants.tsv",
            has_header=True,
            schema_overrides={"subject_id": pl.Utf8, "old_subject_id": pl.Utf8},
        )
        self.subjects = {
            subject_id: self._create_subject(
                subject_id,
                old_subject_id,
                excluded_sessions.get(subject_id, []),
                excluded_trials.get(subject_id, {}),
            )
            for subject_id, old_subject_id in self.metadata.select(
                "subject_id", "old_subject_id"
            ).iter_rows()
            if subject_id not in excluded_subjects
            and old_subject_id not in excluded_subjects
        }

    def _create_subject(
        self,
        subject_id: str,
        old_subject_id: str,
        excluded_sessions: list,
        excluded_trials: dict,
    ):
        return Subject(
            subject_id,
            old_subject_id,
            self,
            excluded_sessions,
            excluded_trials,
        )

    def __iter__(self):
        return iter(self.subjects)

    def __getitem__(self, index):
        return self.subjects[index]

    def __len__(self):
        return len(self.subjects)

    def __repr__(self):
        return f"Experiment = '{self.dataset_path.name}'"

    def load_data(self, detection_algorithm: str):
        """Load derivative tables for every subject in the experiment.

        Must be called once before any table accessor or plotting method. The
        algorithm name selects which set of derivatives to read, so it has to
        match the ``detection_algorithm`` used in
        :func:`~pyxations.compute_derivatives_for_dataset`.

        Parameters
        ----------
        detection_algorithm : str
            Name of the eye-movement detection algorithm whose derivatives
            should be loaded, such as ``"eyelink"``, ``"engbert"`` or
            ``"remodnav"``.
        """
        self.detection_algorithm = detection_algorithm
        for subject in self.subjects.values():
            subject.load_data(detection_algorithm)

    def plot_multipanel(self, display: bool):
        """Plot summary panels of fixations and saccades for the whole dataset.

        Renders fixation duration, saccade amplitude, saccade direction and
        main-sequence panels from the pooled tables of every subject. The
        figure is written under ``<derivatives>/figures/group/``.

        Parameters
        ----------
        display : bool
            Whether to show the figure interactively in addition to saving it.
        """
        visualization_root = self.derivatives_path / "figures" / "group"
        vis = Visualization(visualization_root, self.detection_algorithm)
        vis.plot_multipanel(self.fixations(), self.saccades(), display)

    def filter_fixations(self, min_fix_dur=50, print_flag=True):
        """Drop short fixations from every trial in the experiment.

        Modifies the loaded tables in place. Saccades adjacent to the removed
        fixations are updated accordingly at the trial level.

        Parameters
        ----------
        min_fix_dur : int, default 50
            Minimum fixation duration to keep, in milliseconds. Fixations
            shorter than this are removed.
        print_flag : bool, default True
            Whether to print how many fixations were removed.
        """
        amount_fix = self.fixations().shape[0]
        for subject in self.subjects.values():
            subject.filter_fixations(min_fix_dur)

        if print_flag:
            print(
                f"Removed {amount_fix - self.fixations().shape[0]} fixations shorter than {min_fix_dur} ms."
            )

    def collapse_fixations(self, threshold_px: float, print_flag=True):
        """Merge consecutive fixations that fall close together in space.

        Consecutive fixations separated by less than ``threshold_px`` are
        merged into a single fixation whose duration spans both. Modifies the
        loaded tables in place.

        Parameters
        ----------
        threshold_px : float
            Maximum distance, in pixels, between two consecutive fixations for
            them to be merged.
        print_flag : bool, default True
            Whether to print how many fixations were merged away.
        """
        amount_fix = self.fixations().shape[0]
        for subject in self.subjects.values():
            subject.collapse_fixations(threshold_px)
        if print_flag:
            print(
                f"Removed {amount_fix - self.fixations().shape[0]} fixations that were merged."
            )

    def remove_bad_trials_and_sessions(
        self,
        phase,
        trial_nan_threshold=0.1,
        session_bad_trial_threshold=0.1,
        print_flag=True,
    ):
        """Remove poor sessions, or only their bad trials when recoverable.

        Every session is assessed before it is modified. A session is removed
        when its fraction of bad trials is greater than
        ``session_bad_trial_threshold``. Otherwise, only its bad trials are
        removed. Subjects left without sessions are removed from the
        experiment.

        Parameters
        ----------
        phase : str
            Name of the trial phase to assess, as defined by the
            ``start_msgs``/``end_msgs`` used during segmentation.
        trial_nan_threshold : float, default 0.1
            Maximum fraction of bad samples a trial may contain before it is
            considered bad.
        session_bad_trial_threshold : float, default 0.1
            Maximum fraction of bad trials a session may contain before the
            whole session is removed instead of only its bad trials.
        print_flag : bool, default True
            Whether to print a summary of what was removed.

        Returns
        -------
        QualityFilterResult
            Counts of the trials, sessions and subjects that were removed.
        """

        result = QualityFilterResult()
        for subject_id, subject in list(self.subjects.items()):
            result += subject.remove_bad_trials_and_sessions(
                phase,
                trial_nan_threshold,
                session_bad_trial_threshold,
                False,
            )
            if subject_id not in self.subjects:
                result += QualityFilterResult(subjects_removed=1)
        if print_flag:
            print(
                f"Removed {result.bad_trials_removed} bad trials and "
                f"{result.sessions_removed} sessions across "
                f"{result.subjects_removed} removed subjects, discarding "
                f"{result.trials_discarded_with_sessions} trials with those sessions."
            )
        return result

    def drop_trials_longer_than(self, seconds, phase, print_flag=True):
        """Remove trials whose duration exceeds a limit.

        Useful for discarding trials in which the participant did not respond
        or the recording ran on past the end of the task.

        Parameters
        ----------
        seconds : float
            Maximum trial duration to keep, in seconds.
        phase : str
            Name of the trial phase whose duration is measured, as defined by
            the ``start_msgs``/``end_msgs`` used during segmentation.
        print_flag : bool, default True
            Whether to print how many trials were removed.
        """
        amount_trials_total = self.rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.drop_trials_longer_than(seconds, phase, False)
        if print_flag:
            print(
                f"Removed {amount_trials_total - self.rts().shape[0]} trials longer than {seconds} seconds."
            )

    def plot_scanpaths(self, screen_height, screen_width, display: bool = False):
        """Plot the scanpath of every trial of every subject.

        Figures are written under ``<derivatives>/figures/`` following the
        subject and session hierarchy.

        Parameters
        ----------
        screen_height : int
            Height of the stimulus screen in pixels, used to set the plot
            limits.
        screen_width : int
            Width of the stimulus screen in pixels, used to set the plot
            limits.
        display : bool, default False
            Whether to show each figure interactively in addition to saving it.
        """
        for subject in self.subjects.values():
            subject.plot_scanpaths(screen_height, screen_width, display)

    def drop_poor_or_non_calibrated_trials(self, threshold=1.0, print_flag=True):
        """Drop trials that are uncalibrated or poorly calibrated.

        A trial is considered uncalibrated when no validation data exists for
        its calibration index, and poorly calibrated when the average
        validation error exceeds ``threshold``.

        Parameters
        ----------
        threshold : float, default 1.0
            Maximum average validation error to keep, in degrees of visual
            angle.
        print_flag : bool, default True
            Whether to print how many trials were removed.
        """
        amount_trials_total = self.rts().shape[0]
        for subject in list(self.subjects.values()):
            subject.drop_poor_or_non_calibrated_trials(threshold, False)
        if print_flag:
            print(
                f"Removed {amount_trials_total - self.rts().shape[0]} trials with poor calibration."
            )

    def rts(self):
        """Return response times for every trial in the experiment.

        Returns
        -------
        polars.DataFrame
            One row per trial, with ``subject_id`` and ``session_id``
            identifying its origin.
        """
        return _collect_frames(self.subjects.values(), "rts")

    def get_subject(self, subject_id):
        """Return one subject by identifier.

        Parameters
        ----------
        subject_id : str
            BIDS subject identifier, without the ``sub-`` prefix.

        Returns
        -------
        Subject
            The requested subject.

        Raises
        ------
        KeyError
            If the subject is not part of the experiment, for instance because
            it was excluded at construction time.
        """
        return self.subjects[subject_id]

    def get_session(self, subject_id, session_id):
        """Return one session by subject and session identifier.

        Parameters
        ----------
        subject_id : str
            BIDS subject identifier, without the ``sub-`` prefix.
        session_id : str
            BIDS session identifier, without the ``ses-`` prefix.

        Returns
        -------
        Session
            The requested session.
        """
        subject = self.get_subject(subject_id)
        return subject.get_session(session_id)

    def get_trial(self, subject_id, session_id, trial_number):
        """Return one trial by subject, session and trial number.

        Parameters
        ----------
        subject_id : str
            BIDS subject identifier, without the ``sub-`` prefix.
        session_id : str
            BIDS session identifier, without the ``ses-`` prefix.
        trial_number : int
            Zero-based trial index within the session.

        Returns
        -------
        Trial
            The requested trial.
        """
        session = self.get_session(subject_id, session_id)
        return session.get_trial(trial_number)

    def fixations(self):
        """Return detected fixations from every loaded subject.

        Returns
        -------
        polars.DataFrame
            Pooled fixation table. Empty if :meth:`load_data` has not been
            called or no fixations were detected.
        """
        return _collect_frames(self.subjects.values(), "fixations")

    def saccades(self):
        """Return detected saccades from every loaded subject.

        Returns
        -------
        polars.DataFrame
            Pooled saccade table. Empty if :meth:`load_data` has not been
            called or no saccades were detected.
        """
        return _collect_frames(self.subjects.values(), "saccades")

    def blinks(self):
        """Return blink events from every loaded subject.

        Returns
        -------
        polars.DataFrame
            Pooled blink-event table.
        """
        return _collect_frames(self.subjects.values(), "blinks")

    def pupil_samples(self):
        """Return samples containing pupil measurements from every subject.

        Returns
        -------
        polars.DataFrame
            Pooled sample table containing at least one valid pupil value per
            row.
        """
        return _collect_frames(self.subjects.values(), "pupil_samples")

    def samples(self):
        """Return processed gaze samples from every loaded subject.

        Returns
        -------
        polars.DataFrame
            Pooled sample-level table. This is the largest table in the
            hierarchy; prefer accessing it from a single session or trial when
            working with big datasets.
        """
        return _collect_frames(self.subjects.values(), "samples")

    def remove_subject(self, subject_id):
        """Drop a subject from the experiment.

        Does nothing if the subject is not present, so it is safe to call
        repeatedly.

        Parameters
        ----------
        subject_id : str
            BIDS subject identifier, without the ``sub-`` prefix.
        """
        if subject_id in self.subjects:
            del self.subjects[subject_id]

    def calib_data(self):
        """Return parsed calibration validations for every subject.

        Only meaningful for recordings that report calibration blocks, such as
        EyeLink ``!CAL VALIDATION`` messages.

        Returns
        -------
        calib_data : polars.DataFrame
            One row per validation, with average and maximum error, offsets and
            the recorded eye.
        calib_indexes : polars.DataFrame
            Mapping of each trial to the calibration block that applies to it.
        """
        calib_data = [subject.calib_data() for subject in self.subjects.values()]
        calib_indexes = pl.concat([calib_data[1] for calib_data in calib_data])
        calib_data = pl.concat([calib_data[0] for calib_data in calib_data])
        return calib_data, calib_indexes

    def plot_calib_data(self):
        """Plot a heatmap of calibration error per subject and trial.

        Each cell shows the average validation error, in degrees, of the
        best-calibrated eye for the calibration block that applies to that
        trial. Trials without validation data are drawn in the ``under`` colour
        so that uncalibrated stretches of a session are visible at a glance.

        The figure is shown interactively and not saved to disk.
        """
        # Step 0: Load and preprocess
        calib_data = self.calib_data()
        trial_numbers = calib_data[1]
        calib_data = calib_data[0].select(
            [
                "subject_id",
                "session_id",
                "Calib_index",
                "eye",
                "avg_error",
                "validation_id",
            ]
        )

        # Step 1: Get only rows with max validation_id per group
        max_vals = calib_data.group_by(
            ["subject_id", "session_id", "Calib_index", "eye"]
        ).agg(pl.col("validation_id").max().alias("max_validation_id"))

        calib_data = (
            calib_data.join(
                max_vals, on=["subject_id", "session_id", "Calib_index", "eye"]
            )
            .filter(pl.col("validation_id") == pl.col("max_validation_id"))
            .drop(["max_validation_id", "validation_id"])
        )

        # Step 2: Choose best eye (lowest avg_error) per calibration
        best_eyes = calib_data.group_by(
            ["subject_id", "session_id", "Calib_index"]
        ).agg(pl.col("avg_error").min().alias("best_eye_error"))

        calib_data = (
            calib_data.join(best_eyes, on=["subject_id", "session_id", "Calib_index"])
            .filter(pl.col("avg_error") == pl.col("best_eye_error"))
            .drop(["eye", "best_eye_error"])
        )

        # Step 3: Add trial number and clean up
        calib_data = calib_data.join(
            trial_numbers, on=["subject_id", "session_id", "Calib_index"], how="right"
        ).drop("Calib_index")
        # Replace nans in avg_error with -1
        calib_data = calib_data.with_columns(
            pl.when(pl.col("avg_error").is_null())
            .then(-1)
            .otherwise(pl.col("avg_error"))
            .alias("avg_error")
        )

        # Step 4: Combine the columns "subject_id" and "session_id" into a single column
        calib_data = calib_data.with_columns(
            (
                pl.col("subject_id").cast(pl.Utf8)
                + "_"
                + pl.col("session_id").cast(pl.Utf8)
            ).alias("subject_id")
        ).drop("session_id")
        heatmap_data = calib_data.pivot(
            values="avg_error",
            index="subject_id",
            on="trial_number",
            aggregate_function="first",  # safe if unique per cell
        ).sort("subject_id")

        trial_columns = sorted(
            (column for column in heatmap_data.columns if column != "subject_id"),
            key=lambda value: int(value),
        )
        subject_labels = heatmap_data.get_column("subject_id").to_list()
        heatmap_values = heatmap_data.select(trial_columns).to_numpy()

        # Step 5: Plot with adaptive sizing
        n_subjects, n_trials = heatmap_values.shape

        # Define a base size per cell, then scale it
        cell_width = 0.5  # width per trial column
        cell_height = 0.2  # height per subject row

        # Limit extremes so it doesn’t explode with huge data
        fig_width = max(10, min(cell_width * n_trials, 40))
        fig_height = max(8, min(cell_height * n_subjects, 40))

        # Use a high-contrast reversed sequential map for calibration error.
        cmap = colormaps["magma_r"].with_extremes(under="yellow", bad="white")

        valid_values = heatmap_values[
            np.isfinite(heatmap_values) & (heatmap_values >= 0)
        ]
        vmax = max(0.500001, float(valid_values.max())) if valid_values.size else 1.0
        norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=vmax)
        masked_values = np.ma.masked_invalid(heatmap_values)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        mesh = ax.pcolormesh(
            masked_values,
            cmap=cmap,
            norm=norm,
            edgecolors="grey",
            linewidth=0.3,
            shading="flat",
        )
        ax.invert_yaxis()

        ax.set_xticks(np.arange(n_trials) + 0.5, labels=trial_columns)
        ax.set_yticks(np.arange(n_subjects) + 0.5, labels=subject_labels)
        ax.set_xlabel("Trial #", fontsize=14)
        ax.set_ylabel("Subject", fontsize=14)
        ax.set_title("Calibration Error per Subject and Trial", fontsize=16)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right", va="center", fontsize=10)

        colorbar = fig.colorbar(mesh, ax=ax)
        colorbar.set_label("Avg. error (°)")

        fig.tight_layout()
        plt.show()
        plt.close()


class Subject:
    """One participant of an :class:`Experiment`.

    Sessions are discovered lazily from the subject's directory in the
    derivatives dataset the first time :attr:`sessions` is accessed, so
    constructing a ``Subject`` does not touch the filesystem beyond building
    its paths.

    Subjects are normally created by :class:`Experiment` rather than directly.

    Parameters
    ----------
    subject_id : str
        BIDS subject identifier, without the ``sub-`` prefix.
    old_subject_id : str
        Identifier the subject had in the original vendor recording, preserved
        during conversion so results can be traced back to the source files.
    experiment : Experiment
        Parent experiment. Held as a weak reference to avoid a reference cycle.
    excluded_sessions : list, optional
        Session identifiers to skip.
    excluded_trials : dict, optional
        Mapping of ``session_id`` to a list of trial numbers to skip.

    Attributes
    ----------
    subject_dataset_path : pathlib.Path
        Subject directory inside the raw BIDS dataset.
    subject_derivatives_path : pathlib.Path
        Subject directory inside the derivatives dataset.
    """

    def __init__(
        self,
        subject_id: str,
        old_subject_id: str,
        experiment: Experiment,
        excluded_sessions: list | None = None,
        excluded_trials: dict | None = None,
    ):
        excluded_sessions = excluded_sessions or []
        excluded_trials = excluded_trials or {}
        self.subject_id = subject_id
        self.old_subject_id = old_subject_id
        self.experiment = weakref.ref(experiment)
        self._sessions = None  # Lazy load sessions
        self.excluded_sessions = excluded_sessions
        self.excluded_trials = excluded_trials
        self.subject_dataset_path = (
            self.experiment().dataset_path / f"sub-{self.subject_id}"
        )
        self.subject_derivatives_path = (
            self.experiment().derivatives_path / f"sub-{self.subject_id}"
        )

    @property
    def sessions(self):
        """Sessions of this subject, discovered lazily on first access.

        Returns
        -------
        dict
            Mapping of ``session_id`` to :class:`Session`, ordered by the
            ``ses-*`` directory names found in the derivatives dataset and
            excluding any session listed in ``excluded_sessions``.
        """
        if self._sessions is None:
            session_folders = sorted(self.subject_derivatives_path.glob("ses-*"))
            self._sessions = {
                session_id: self._create_session(session_id)
                for session_folder in session_folders
                if (session_id := session_folder.name.removeprefix("ses-"))
                not in self.excluded_sessions
            }
        return self._sessions

    def _create_session(self, session_id: str):
        return Session(
            session_id,
            self,
            self.excluded_trials.get(session_id, {}),
        )

    def __iter__(self):
        return iter(self.sessions)

    def __getitem__(self, index):
        return self.sessions[index]

    def __len__(self):
        return len(self.sessions)

    def __repr__(self):
        return f"Subject = '{self.subject_id}', " + self.experiment().__repr__()

    def remove_session(self, session_id):
        """Drop a session from this subject.

        When the last session is removed, the subject removes itself from its
        parent experiment, since a subject without sessions carries no data.

        Parameters
        ----------
        session_id : str
            BIDS session identifier, without the ``ses-`` prefix. Ignored if
            the session is not present.
        """
        if self._sessions and session_id in self._sessions:
            del self._sessions[session_id]
            if len(self._sessions) == 0:
                exp = self.experiment()
                if exp:
                    exp.remove_subject(self.subject_id)
                self._sessions = None
                self.experiment = lambda: None

    def load_data(self, detection_algorithm: str):
        """Load derivative tables for every session of this subject.

        Parameters
        ----------
        detection_algorithm : str
            Name of the eye-movement detection algorithm whose derivatives
            should be loaded, such as ``"eyelink"``, ``"engbert"`` or
            ``"remodnav"``.
        """
        self.detection_algorithm = detection_algorithm
        for session in self.sessions.values():
            session.load_data(detection_algorithm)

    def filter_fixations(self, min_fix_dur=50):
        """Drop short fixations from every session of this subject.

        Parameters
        ----------
        min_fix_dur : int, default 50
            Minimum fixation duration to keep, in milliseconds.
        """
        for session in self.sessions.values():
            session.filter_fixations(min_fix_dur)

    def collapse_fixations(self, threshold_px: float):
        """Merge nearby consecutive fixations in every session of this subject.

        Parameters
        ----------
        threshold_px : float
            Maximum distance, in pixels, between two consecutive fixations for
            them to be merged.
        """
        for session in self.sessions.values():
            session.collapse_fixations(threshold_px)

    def remove_bad_trials_and_sessions(
        self,
        phase,
        trial_nan_threshold=0.1,
        session_bad_trial_threshold=0.1,
        print_flag=True,
    ):
        """Apply the session-or-trial quality policy to this subject.

        Each session is assessed before it is modified. A session whose
        fraction of bad trials exceeds ``session_bad_trial_threshold`` is
        removed entirely; otherwise only its bad trials are dropped.

        Parameters
        ----------
        phase : str
            Name of the trial phase to assess.
        trial_nan_threshold : float, default 0.1
            Maximum fraction of bad samples a trial may contain before it is
            considered bad.
        session_bad_trial_threshold : float, default 0.1
            Maximum fraction of bad trials a session may contain before the
            whole session is removed.
        print_flag : bool, default True
            Whether to print a summary of what was removed.

        Returns
        -------
        QualityFilterResult
            Counts of the trials and sessions that were removed.

        Raises
        ------
        ValueError
            If ``session_bad_trial_threshold`` is outside ``[0, 1]``.
        """

        if not 0 <= session_bad_trial_threshold <= 1:
            raise ValueError(
                "session_bad_trial_threshold must be between 0 and 1 inclusive"
            )

        result = QualityFilterResult()
        for session_id, session in list(self.sessions.items()):
            assessment = session.assess_trial_quality(phase, trial_nan_threshold)
            if assessment.bad_trial_fraction > session_bad_trial_threshold:
                result += QualityFilterResult(
                    sessions_removed=1,
                    trials_discarded_with_sessions=assessment.total_trials,
                )
                self.remove_session(session_id)
            else:
                removed = session._remove_assessed_bad_trials(assessment)
                result += QualityFilterResult(bad_trials_removed=removed)

        if print_flag:
            print(
                f"Removed {result.bad_trials_removed} bad trials and "
                f"{result.sessions_removed} sessions from subject {self.subject_id}."
            )
        return result

    def drop_poor_or_non_calibrated_trials(self, threshold=1.0, print_flag=True):
        """Drop trials that are uncalibrated or poorly calibrated.

        A trial is considered uncalibrated when no validation data exists for
        its calibration index, and poorly calibrated when the average
        validation error exceeds ``threshold``.

        Parameters
        ----------
        threshold : float, default 1.0
            Maximum average validation error to keep, in degrees of visual
            angle.
        print_flag : bool, default True
            Whether to print how many trials were removed.
        """
        amount_trials_total = self.rts().shape[0]
        for session in list(self.sessions.values()):
            session.drop_poor_or_non_calibrated_trials(threshold, False)
        if print_flag:
            print(
                f"Removed {amount_trials_total - self.rts().shape[0]} trials with poor calibration."
            )

    def drop_trials_longer_than(self, seconds, phase, print_flag=True):
        """Remove trials of this subject whose duration exceeds a limit.

        Parameters
        ----------
        seconds : float
            Maximum trial duration to keep, in seconds.
        phase : str
            Name of the trial phase whose duration is measured.
        print_flag : bool, default True
            Whether to print how many trials were removed.
        """
        amount_trials_total = self.rts().shape[0]
        for session in list(self.sessions.values()):
            session.drop_trials_longer_than(seconds, phase, False)
        if print_flag:
            print(
                f"Removed {amount_trials_total - self.rts().shape[0]} trials longer than {seconds} seconds."
            )

    def plot_scanpaths(self, screen_height, screen_width, display: bool = False):
        """Plot the scanpath of every trial of this subject.

        Parameters
        ----------
        screen_height : int
            Height of the stimulus screen in pixels.
        screen_width : int
            Width of the stimulus screen in pixels.
        display : bool, default False
            Whether to show each figure interactively in addition to saving it.
        """
        for session in self.sessions.values():
            session.plot_scanpaths(screen_height, screen_width, display)

    def rts(self):
        """Return response times with the subject identifier attached.

        Returns
        -------
        polars.DataFrame
            One row per trial, with a ``subject_id`` column.
        """
        return _collect_frames(
            self.sessions.values(),
            "rts",
            identifier=("subject_id", self.subject_id),
        )

    def get_session(self, session_id):
        """Return one session by identifier.

        Parameters
        ----------
        session_id : str
            BIDS session identifier, without the ``ses-`` prefix.

        Returns
        -------
        Session
            The requested session.

        Raises
        ------
        KeyError
            If the session is not part of this subject.
        """
        return self.sessions[session_id]

    def get_trial(self, session_id, trial_number):
        """Return one trial by session identifier and trial number.

        Parameters
        ----------
        session_id : str
            BIDS session identifier, without the ``ses-`` prefix.
        trial_number : int
            Zero-based trial index within the session.

        Returns
        -------
        Trial
            The requested trial.
        """
        session = self.get_session(session_id)
        return session.get_trial(trial_number)

    def fixations(self):
        """Return fixations with the subject identifier attached.

        Returns
        -------
        polars.DataFrame
            Fixation table pooled across sessions, with a ``subject_id``
            column.
        """
        return _collect_frames(
            self.sessions.values(),
            "fixations",
            identifier=("subject_id", self.subject_id),
        )

    def saccades(self):
        """Return saccades with the subject identifier attached.

        Returns
        -------
        polars.DataFrame
            Saccade table pooled across sessions, with a ``subject_id`` column.
        """
        return _collect_frames(
            self.sessions.values(),
            "saccades",
            identifier=("subject_id", self.subject_id),
        )

    def blinks(self):
        """Return blink events with the subject identifier attached.

        Returns
        -------
        polars.DataFrame
            Blink-event table pooled across sessions, with a ``subject_id``
            column.
        """
        return _collect_frames(
            self.sessions.values(),
            "blinks",
            identifier=("subject_id", self.subject_id),
        )

    def pupil_samples(self):
        """Return pupil samples with the subject identifier attached.

        Returns
        -------
        polars.DataFrame
            Pupil-sample table pooled across sessions, with a ``subject_id``
            column.
        """
        return _collect_frames(
            self.sessions.values(),
            "pupil_samples",
            identifier=("subject_id", self.subject_id),
        )

    def samples(self):
        """Return processed gaze samples with the subject identifier attached.

        Returns
        -------
        polars.DataFrame
            Sample-level table pooled across sessions, with a ``subject_id``
            column.
        """
        return _collect_frames(
            self.sessions.values(),
            "samples",
            identifier=("subject_id", self.subject_id),
        )

    def calib_data(self):
        """Return parsed calibration validations for every session.

        Returns
        -------
        calib_data : polars.DataFrame
            One row per validation, with a ``subject_id`` column.
        calib_indexes : polars.DataFrame
            Mapping of each trial to its calibration block, with a
            ``subject_id`` column.
        """
        calib_data = [session.calib_data() for session in self.sessions.values()]
        calib_indexes = pl.concat(
            [calib_data[1] for calib_data in calib_data]
        ).with_columns(
            [
                (pl.lit(self.subject_id)).alias("subject_id"),
            ]
        )
        calib_data = pl.concat(
            [calib_data[0] for calib_data in calib_data]
        ).with_columns(
            [
                (pl.lit(self.subject_id)).alias("subject_id"),
            ]
        )
        return calib_data, calib_indexes


class Session:
    """One recording session of a :class:`Subject`.

    A session owns the derivative tables actually read from disk: processed
    samples, fixations, saccades, blinks and messages. :meth:`load_data` reads
    them once and splits them into :class:`Trial` objects, which then share
    slices of the same tables.

    Sessions are normally created by :class:`Subject` rather than directly.

    Parameters
    ----------
    session_id : str
        BIDS session identifier, without the ``ses-`` prefix.
    subject : Subject
        Parent subject. Held as a weak reference to avoid a reference cycle.
    excluded_trials : list, optional
        Trial numbers to skip when building the trial hierarchy.

    Attributes
    ----------
    session_dataset_path : pathlib.Path
        Session directory inside the raw BIDS dataset.
    session_derivatives_path : pathlib.Path
        Session directory inside the derivatives dataset.

    Raises
    ------
    FileNotFoundError
        If the session directory does not exist in the derivatives dataset.
    """

    def __init__(
        self,
        session_id: str,
        subject: Subject,
        excluded_trials: list | None = None,
    ):
        excluded_trials = excluded_trials or []
        self.session_id = session_id
        self.subject = weakref.ref(subject)
        self.excluded_trials = excluded_trials
        self.session_dataset_path = (
            self.subject().subject_dataset_path / f"ses-{self.session_id}"
        )
        self.session_derivatives_path = (
            self.subject().subject_derivatives_path / f"ses-{self.session_id}"
        )
        self._trials = None  # Lazy load trials

        if not self.session_derivatives_path.exists():
            raise FileNotFoundError(
                f"Session path not found: {self.session_derivatives_path}"
            )

    @property
    def trials(self):
        """Trials of this session, keyed by trial number.

        Returns
        -------
        dict
            Mapping of ``trial_number`` to :class:`Trial`.

        Raises
        ------
        ValueError
            If :meth:`load_data` has not been called yet.
        """
        if self._trials is None:
            raise ValueError("Trials not loaded. Please load data first.")
        return self._trials

    def __repr__(self):
        return f"Session = '{self.session_id}', " + self.subject().__repr__()

    def assess_trial_quality(
        self, phase, trial_nan_threshold=0.1
    ) -> SessionQualityAssessment:
        """Classify trials as good or bad without modifying the session.

        Assessing before removing keeps the bad-trial fraction meaningful: it
        is computed against the original number of trials rather than against a
        set that is shrinking as trials are dropped.

        Parameters
        ----------
        phase : str
            Name of the trial phase to assess.
        trial_nan_threshold : float, default 0.1
            Largest allowed fraction of invalid gaze samples in an individual
            trial.

        Returns
        -------
        SessionQualityAssessment
            The trials flagged as bad and the total number assessed.

        Raises
        ------
        ValueError
            If ``trial_nan_threshold`` is outside ``[0, 1]``.
        """

        if not 0 <= trial_nan_threshold <= 1:
            raise ValueError("trial_nan_threshold must be between 0 and 1 inclusive")

        bad_trials = tuple(
            trial_number
            for trial_number, trial in self.trials.items()
            if trial.is_trial_bad(phase, trial_nan_threshold)
        )
        return SessionQualityAssessment(
            bad_trials=bad_trials,
            total_trials=len(self.trials),
        )

    def _remove_assessed_bad_trials(self, assessment: SessionQualityAssessment) -> int:
        removed = 0
        for trial_number in assessment.bad_trials:
            if self._trials is None or trial_number not in self._trials:
                continue
            self.remove_trial(trial_number)
            removed += 1
        return removed

    def remove_bad_trials(self, phase, trial_nan_threshold=0.1, print_flag=True) -> int:
        """Remove individual bad trials without applying a session policy.

        Unlike :meth:`Subject.remove_bad_trials_and_sessions`, this never
        removes the session itself, however many trials turn out to be bad.

        Parameters
        ----------
        phase : str
            Name of the trial phase to assess.
        trial_nan_threshold : float, default 0.1
            Largest allowed fraction of invalid gaze samples in a trial.
        print_flag : bool, default True
            Whether to print how many trials were removed.

        Returns
        -------
        int
            Number of trials removed.
        """

        assessment = self.assess_trial_quality(phase, trial_nan_threshold)
        removed = self._remove_assessed_bad_trials(assessment)
        if print_flag:
            print(f"Removed {removed} bad trials.")
        return removed

    def drop_poor_or_non_calibrated_trials(self, threshold=1.0, print_flag=True):
        """Drop trials that are uncalibrated or poorly calibrated.

        A trial is considered uncalibrated when no validation data exists for
        its calibration index, and poorly calibrated when the average
        validation error exceeds ``threshold``.

        Parameters
        ----------
        threshold : float, default 1.0
            Maximum average validation error to keep, in degrees of visual
            angle.
        print_flag : bool, default True
            Whether to print how many trials were removed.
        """
        trial_numbers = list(self.trials)
        # Step 1: Get only rows with max validation_id per group
        calib_data, trial_numbers = self.calib_data()
        calib_data = calib_data.drop("session_id")
        max_vals = calib_data.group_by(["Calib_index", "eye"]).agg(
            pl.col("validation_id").max().alias("max_validation_id")
        )

        calib_data = (
            calib_data.join(max_vals, on=["Calib_index", "eye"])
            .filter(pl.col("validation_id") == pl.col("max_validation_id"))
            .drop(["max_validation_id", "validation_id"])
        )

        # Step 2: Choose best eye (lowest avg_error) per calibration
        best_eyes = calib_data.group_by(["Calib_index"]).agg(
            pl.col("avg_error").min().alias("best_eye_error")
        )

        calib_data = (
            calib_data.join(best_eyes, on=["Calib_index"])
            .filter(pl.col("avg_error") == pl.col("best_eye_error"))
            .drop(["eye", "best_eye_error"])
        )

        calib_data = calib_data.join(
            trial_numbers, on=["Calib_index"], how="right"
        ).drop("Calib_index")
        # Bad trials are those with avg_error > threshold, or those that have NaN values in avg_error
        bad_trials = (
            calib_data.filter(
                (pl.col("avg_error") > threshold) | (pl.col("avg_error").is_null())
            )
            .select("trial_number")
            .to_series()
            .unique()
            .to_list()
        )

        for trial in bad_trials:
            self.remove_trial(trial)

        if print_flag:
            print(f"Removed {len(bad_trials)} trials with poor calibration.")

    def drop_trials_longer_than(self, seconds, phase, print_flag=True):
        """Remove trials of this session whose duration exceeds a limit.

        Parameters
        ----------
        seconds : float
            Maximum trial duration to keep, in seconds.
        phase : str
            Name of the trial phase whose duration is measured.
        print_flag : bool, default True
            Whether to print how many trials were removed.
        """

        # Filter bad trials

        bad_trials = [
            trial
            for trial in self.trials
            if self.trials[trial].is_trial_longer_than(seconds, phase)
        ]
        for trial in bad_trials:
            self.remove_trial(trial)

        if print_flag:
            print(f"Removed {len(bad_trials)} trials longer than {seconds} seconds.")

    def load_data(self, detection_algorithm: str):
        """Read this session's derivative tables and build its trials.

        Reads the processed samples, fixations, saccades, blinks and
        calibration report written by
        :func:`~pyxations.compute_derivatives_for_dataset`, then partitions
        them by trial number into :class:`Trial` objects. Trials listed in
        ``excluded_trials`` and the ``-1`` bucket of samples that fall outside
        any trial are dropped.

        Parameters
        ----------
        detection_algorithm : str
            Name of the eye-movement detection algorithm whose derivatives
            should be loaded, such as ``"eyelink"``, ``"engbert"`` or
            ``"remodnav"``.
        """
        self.detection_algorithm = detection_algorithm
        bundle = BIDSDerivativeExport().read_session(
            self.session_derivatives_path, detection_algorithm
        )
        samples = bundle.samples
        fix = bundle.fixations
        sacc = bundle.saccades
        blink = bundle.blinks
        calibration = bundle.calibration
        self._calib_data = (
            _parse_validations(calibration)
            if (
                not calibration.is_empty()
                and "line" in calibration.columns
                and "Calib_index" in calibration.columns
            )
            else None
        )
        events_path = (
            self.session_derivatives_path.parents[1]
            / "figures"
            / self.session_derivatives_path.parent.name
            / self.session_derivatives_path.name
            / self.detection_algorithm
        )

        self._init_trials(samples, fix, sacc, blink, events_path)

    def calib_data(self):
        """Return parsed calibration validations for this session.

        Returns
        -------
        calibration : polars.DataFrame
            One row per validation, with average and maximum error, offsets,
            the recorded eye and a ``session_id`` column.
        calib_indexes : polars.DataFrame
            Mapping of each trial to the calibration block that applies to it.

        Raises
        ------
        ValueError
            If :meth:`load_data` has not been called, or if the recording
            contains no calibration report.
        """
        if self._calib_data is None:
            raise ValueError(
                f"Calibration data for session {self.session_id} and subject {self.subject().subject_id} not loaded. Please load data first."
            )

        calib_indexes = [
            (trial.trial_number, trial.calib_index)
            for trial in self.trials.values()
            if trial.calib_index is not None
        ]
        calib_indexes = pl.DataFrame(
            calib_indexes, schema=["trial_number", "Calib_index"], orient="row"
        ).with_columns([(pl.lit(self.session_id)).alias("session_id")])
        calibration = self._calib_data.with_columns(
            pl.col("Calib_index").cast(pl.Int64, strict=False),
            pl.lit(self.session_id).alias("session_id"),
        )
        calib_indexes = calib_indexes.with_columns(
            pl.col("Calib_index").cast(pl.Int64, strict=False)
        )
        return calibration, calib_indexes

    def _init_trials(self, samples, fix, sacc, blink, events_path):
        sample_trials = _partition_trials(samples)
        fixation_trials = _partition_trials(fix)
        saccade_trials = _partition_trials(sacc)
        blink_trials = _partition_trials(blink)
        empty_fix = fix.head(0)
        empty_sacc = sacc.head(0)
        empty_blink = blink.head(0) if blink is not None else None
        trial_numbers = [
            trial
            for trial in sample_trials
            if trial != -1 and trial not in self.excluded_trials
        ]
        self._trials = {
            trial: Trial(
                trial,
                self,
                sample_trials[trial],
                fixation_trials.get(trial, empty_fix),
                saccade_trials.get(trial, empty_sacc),
                blink_trials.get(trial, empty_blink),
                events_path,
                prefiltered=True,
            )
            for trial in trial_numbers
        }

    def plot_scanpaths(self, screen_height, screen_width, display: bool = False):
        """Plot the scanpath of every trial of this session.

        Parameters
        ----------
        screen_height : int
            Height of the stimulus screen in pixels.
        screen_width : int
            Width of the stimulus screen in pixels.
        display : bool, default False
            Whether to show each figure interactively in addition to saving it.
        """
        for trial in self.trials.values():
            trial.plot_scanpath(screen_height, screen_width, display=display)

    def __iter__(self):
        return iter(self.trials)

    def __getitem__(self, index):
        return self.trials[index]

    def __len__(self):
        return len(self.trials)

    def get_trial(self, trial_number):
        """Return one trial by number.

        Parameters
        ----------
        trial_number : int
            Zero-based trial index within the session.

        Returns
        -------
        Trial
            The requested trial.

        Raises
        ------
        KeyError
            If the trial is not part of this session, for instance because it
            was excluded or removed by a quality filter.
        """
        return self._trials[trial_number]

    def filter_fixations(self, min_fix_dur=50):
        """Drop short fixations from every trial of this session.

        Parameters
        ----------
        min_fix_dur : int, default 50
            Minimum fixation duration to keep, in milliseconds.
        """
        for trial in self.trials.values():
            trial.filter_fixations(min_fix_dur)

    def collapse_fixations(self, threshold_px: float):
        """Merge nearby consecutive fixations in every trial of this session.

        Parameters
        ----------
        threshold_px : float
            Maximum distance, in pixels, between two consecutive fixations for
            them to be merged.
        """
        for trial in self.trials.values():
            trial.collapse_fixations(threshold_px)

    def rts(self):
        """Return response times with the session identifier attached.

        Returns
        -------
        polars.DataFrame
            One row per trial, with a ``session_id`` column.
        """
        return _collect_frames(
            self.trials.values(),
            "rts",
            identifier=("session_id", self.session_id),
        )

    def fixations(self):
        """Return fixations with the session identifier attached.

        Returns
        -------
        polars.DataFrame
            Fixation table pooled across trials, with a ``session_id`` column.
        """
        return _collect_frames(
            self.trials.values(),
            "fixations",
            identifier=("session_id", self.session_id),
        )

    def saccades(self):
        """Return saccades with the session identifier attached.

        Returns
        -------
        polars.DataFrame
            Saccade table pooled across trials, with a ``session_id`` column.
        """
        return _collect_frames(
            self.trials.values(),
            "saccades",
            identifier=("session_id", self.session_id),
        )

    def blinks(self):
        """Return blink events with the session identifier attached.

        Returns
        -------
        polars.DataFrame
            Blink-event table pooled across trials, with a ``session_id``
            column.
        """
        return _collect_frames(
            self.trials.values(),
            "blinks",
            identifier=("session_id", self.session_id),
        )

    def pupil_samples(self):
        """Return pupil samples with the session identifier attached.

        Returns
        -------
        polars.DataFrame
            Pupil-sample table pooled across trials, with a ``session_id``
            column.
        """
        return _collect_frames(
            self.trials.values(),
            "pupil_samples",
            identifier=("session_id", self.session_id),
        )

    def samples(self):
        """Return processed gaze samples with the session identifier attached.

        Returns
        -------
        polars.DataFrame
            Sample-level table pooled across trials, with a ``session_id``
            column.
        """
        return _collect_frames(
            self.trials.values(),
            "samples",
            identifier=("session_id", self.session_id),
        )

    def remove_trial(self, trial_number):
        """Drop a trial from this session.

        When the last trial is removed, the session removes itself from its
        parent subject, which in turn may remove the subject from the
        experiment.

        Parameters
        ----------
        trial_number : int
            Zero-based trial index. Ignored if the trial is not present.
        """
        if self._trials and trial_number in self._trials:
            del self._trials[trial_number]
            if len(self._trials) == 0:
                subj = self.subject()
                if subj:
                    subj.remove_session(self.session_id)
                self._trials = None
                self.subject = lambda: None


class Trial:
    """One segmented trial of a :class:`Session`.

    A trial holds slices of its session's tables. All timestamps are
    normalized on construction so that the trial starts at ``t = 0``: the
    timestamp of the first sample is subtracted from ``tSample`` and from the
    ``tStart``/``tEnd`` of fixations, saccades and blinks. Values keep the
    units reported by the source eye tracker.

    Trials are normally created by :meth:`Session.load_data` rather than
    directly.

    Parameters
    ----------
    trial_number : int
        Zero-based trial index within the session.
    session : Session
        Parent session.
    samples : polars.DataFrame
        Processed gaze samples.
    fix : polars.DataFrame
        Detected fixations.
    sacc : polars.DataFrame
        Detected saccades.
    blink : polars.DataFrame or None
        Detected blinks, or ``None`` when the recording reports none.
    events_path : pathlib.Path
        Directory where figures for this trial are written.
    prefiltered : bool, default False
        Whether the tables already contain only this trial's rows. When
        ``False`` they are filtered by ``trial_number`` on construction.

    Attributes
    ----------
    trial_number : int
        Zero-based trial index within the session.
    detection_algorithm : str
        Name of the algorithm whose derivatives this trial was built from.
    """

    def __init__(
        self,
        trial_number: int,
        session: Session,
        samples: pl.DataFrame,
        fix: pl.DataFrame,
        sacc: pl.DataFrame,
        blink: pl.DataFrame | None,
        events_path: Path,
        *,
        prefiltered: bool = False,
    ):
        self.trial_number = trial_number
        self.session = session

        if prefiltered:
            sample_rows = samples
            fixation_rows = fix
            saccade_rows = sacc
            blink_rows = blink
        else:
            sample_rows = samples.filter(pl.col("trial_number") == trial_number)
            fixation_rows = fix.filter(pl.col("trial_number") == trial_number)
            saccade_rows = sacc.filter(pl.col("trial_number") == trial_number)
            blink_rows = (
                blink.filter(pl.col("trial_number") == trial_number)
                if blink is not None
                else None
            )

        self._calib_index = (
            sample_rows.get_column("Calib_index")[0]
            if "Calib_index" in sample_rows.columns and sample_rows.height
            else None
        )
        self._samples = sample_rows.drop("Calib_index", strict=False)
        self._fix = fixation_rows.drop("Calib_index", strict=False)
        self._sacc = saccade_rows.drop("Calib_index", strict=False)
        self._blink = (
            blink_rows.drop("Calib_index", strict=False)
            if blink_rows is not None
            else None
        )

        # Get the start time
        start_time = self._samples.select("tSample").to_series()[0]

        # Time normalization
        self._samples = self._samples.with_columns(
            [(pl.col("tSample") - start_time).alias("tSample")]
        )

        self._fix = self._fix.with_columns(
            [
                (pl.col("tStart") - start_time).alias("tStart"),
                (pl.col("tEnd") - start_time).alias("tEnd"),
            ]
        )

        self._sacc = self._sacc.with_columns(
            [
                (pl.col("tStart") - start_time).alias("tStart"),
                (pl.col("tEnd") - start_time).alias("tEnd"),
            ]
        )

        if self._blink is not None:
            self._blink = self._blink.with_columns(
                [
                    (pl.col("tStart") - start_time).alias("tStart"),
                    (pl.col("tEnd") - start_time).alias("tEnd"),
                ]
            )

        self.events_path = events_path
        self.detection_algorithm = events_path.name.removesuffix("_events")

    def fixations(self):
        """Return the fixations detected in this trial.

        Returns
        -------
        polars.DataFrame
            Fixation table with ``tStart``/``tEnd`` relative to the start of
            the trial.
        """
        return self._fix

    @property
    def calib_index(self):
        """Index of the calibration block that applies to this trial.

        Returns
        -------
        int or None
            The calibration index, or ``None`` when the recording reports no
            calibration.
        """
        return self._calib_index

    def saccades(self):
        """Return the saccades detected in this trial.

        Returns
        -------
        polars.DataFrame
            Saccade table with ``tStart``/``tEnd`` relative to the start of the
            trial.
        """
        return self._sacc

    def blinks(self):
        """Return blink events for this trial.

        Times and durations retain the units used by the source eye tracker.

        Returns
        -------
        polars.DataFrame
            Blink events for the trial, or an empty table with the canonical
            blink schema when the source contains no blink events.
        """
        if self._blink is None:
            return pl.DataFrame(
                schema={
                    "tStart": pl.Float64,
                    "tEnd": pl.Float64,
                    "duration": pl.Float64,
                }
            )
        return self._blink

    def pupil_samples(self):
        """Return samples with at least one recorded pupil-size value.

        Pupil values retain the units reported by the source eye tracker.
        Depending on the recording, the columns are ``Pupil`` or the
        eye-specific ``LPupil`` and ``RPupil``.

        Returns
        -------
        polars.DataFrame
            Rows from the trial sample table that contain at least one valid
            pupil measurement. If pupil data were not recorded, an empty table
            with the sample schema is returned.
        """
        pupil_columns = [
            column
            for column in ("Pupil", "LPupil", "RPupil", "pupil_size")
            if column in self._samples.columns
        ]
        if not pupil_columns:
            return self._samples.head(0)

        valid_pupil = pl.any_horizontal(
            [
                pl.col(column).is_not_null()
                & ~pl.col(column).cast(pl.Float64, strict=False).is_nan()
                for column in pupil_columns
            ]
        )
        return self._samples.filter(valid_pupil)

    def samples(self):
        """Return the processed gaze samples of this trial.

        Returns
        -------
        polars.DataFrame
            Sample-level table with ``tSample`` relative to the start of the
            trial.
        """
        return self._samples

    def __repr__(self):
        return f"Trial = '{self.trial_number}', " + self.session.__repr__()

    def plot_scanpath(self, screen_height, screen_width, **kwargs):
        """Plot the scanpath of this trial.

        The figure is written under the trial's ``events_path``, inside the
        derivatives ``figures/`` directory that the dataset's ``.bidsignore``
        excludes from validation.

        Parameters
        ----------
        screen_height : int
            Height of the stimulus screen in pixels.
        screen_width : int
            Width of the stimulus screen in pixels.
        **kwargs : object
            Extra keyword arguments forwarded to
            :meth:`~pyxations.Visualization.scanpath`, such as ``display`` or a
            background image.
        """
        vis = Visualization(self.events_path, self.detection_algorithm)
        self.events_path.mkdir(parents=True, exist_ok=True)
        vis.scanpath(
            fixations=self._fix,
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
        """Create an animated visualization of this trial's gaze data.

        When a video is provided, gaze samples are synced with its frames. When
        none is provided, gaze points are animated over a grey background or a
        supplied background image, timed by the sample timestamps.

        Requires the optional OpenCV dependency, installed with
        ``pip install 'pyxations[video]'``.

        Parameters
        ----------
        screen_height : int
            Height of the stimulus screen in pixels.
        screen_width : int
            Width of the stimulus screen in pixels.
        video_path : str or pathlib.Path, optional
            Video over which gaze is overlaid.
        background_image_path : str or pathlib.Path, optional
            Background image, used only when ``video_path`` is omitted. With
            neither, the background is grey.
        **kwargs : object
            Extra keyword arguments forwarded to
            :meth:`~pyxations.Visualization.plot_animation`:

            folder_path : str or pathlib.Path
                Directory in which the animation is saved.
            tmin, tmax : int
                Time window to animate, in milliseconds.
            seconds_to_show : float
                Limit the animation to the first N seconds.
            scale_factor : float, default 0.5
                Resolution scaling applied to the output.
            gaze_radius : int
                Radius of the gaze marker, in pixels.
            gaze_color : tuple of int
                RGB colour of the gaze marker.
            fps : int
                Frames per second of the animation.
            output_format : {"matplotlib", "html", "mp4", "gif"}
                Output format, ``"matplotlib"`` by default.
            display : bool
                Whether to return HTML for display in a notebook.

        Returns
        -------
        IPython.display.HTML or None
            An HTML animation when ``display=True`` and
            ``output_format="html"``. With ``output_format="matplotlib"`` the
            animation is shown in a GUI window and ``None`` is returned.
        """
        vis = Visualization(self.events_path, self.detection_algorithm)
        self.events_path.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("folder_path", self.events_path)

        return vis.plot_animation(
            samples=self._samples,
            screen_height=screen_height,
            screen_width=screen_width,
            video_path=video_path,
            background_image_path=background_image_path,
            **kwargs,
        )

    def filter_fixations(self, min_fix_dur: int = 50):
        """Delete short fixations and merge their flanking saccades.

        Processing stays within each phase and eye stream and modifies the
        trial's fixation and saccade tables in place.

        Parameters
        ----------
        min_fix_dur : int, default 50
            Minimum fixation duration to retain, in milliseconds.
        """
        # ─────────────────────── 0 · split keep / drop ──────────────────────
        short_fix = self._fix.filter(pl.col("duration") < min_fix_dur)
        keep_fix = self._fix.filter(pl.col("duration") >= min_fix_dur)

        if short_fix.is_empty():
            return  # nothing to do

        # ─────────────────────── 1 · prepare saccades ───────────────────────
        sacc = (
            self._sacc.with_row_index(  # add an integer key that survives every shuffle
                "idx"
            ).sort(["phase", "eye", "tStart"])
        )

        prev_src = sacc.select(["idx", "phase", "eye", pl.col("tEnd").alias("t")])
        next_src = sacc.select(["idx", "phase", "eye", pl.col("tStart").alias("t")])

        # ─────────────────────── 2 · find neighbour IDs ─────────────────────
        short_fix = short_fix.rename({"tStart": "tStart_fix", "tEnd": "tEnd_fix"})

        short_fix = short_fix.sort(["phase", "eye", "tStart_fix"])
        prev_src = prev_src.sort(["phase", "eye", "t"])
        next_src = next_src.sort(["phase", "eye", "t"])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sortedness of columns cannot be checked when 'by' groups provided",
                category=UserWarning,
            )

            short_fix = (
                short_fix.join_asof(
                    prev_src,
                    left_on="tStart_fix",
                    right_on="t",
                    by=["phase", "eye"],
                    strategy="backward",
                )
                .rename({"idx": "idx_prev"})
                .drop("t")
                .join_asof(
                    next_src,
                    left_on="tEnd_fix",
                    right_on="t",
                    by=["phase", "eye"],
                    strategy="forward",
                )
                .rename({"idx": "idx_next"})
                .drop("t")
            )

        # only keep rows where we found BOTH neighbours
        short_fix_pairs = short_fix.select(["idx_prev", "idx_next"]).drop_nulls()
        if short_fix_pairs.is_empty():
            # we could not build any (prev,next) pair → only delete fixations
            self._fix = keep_fix.sort(["phase", "tStart"])
            return

        # ───────────────────── 3 · join the two saccades ────────────────────
        pair_df = (
            short_fix_pairs.unique()
            .join(sacc, left_on="idx_prev", right_on="idx", how="inner")
            .join(sacc, left_on="idx_next", right_on="idx", suffix="_nxt")
        )

        # keep **prev** row plus ONLY the four _nxt columns that we still need
        prev_cols = [c for c in pair_df.columns if not c.endswith("_nxt")]
        need_nxt = ["tEnd_nxt", "xEnd_nxt", "yEnd_nxt", "vPeak_nxt"]
        merged = pair_df.select(prev_cols + need_nxt)

        # ───────── overwrite / derive fields that span both flanks ──────────
        merged = merged.with_columns(
            [
                pl.col("tEnd_nxt").alias("tEnd"),
                (pl.col("tEnd_nxt") - pl.col("tStart")).alias("duration"),
                pl.col("xEnd_nxt").alias("xEnd"),
                pl.col("yEnd_nxt").alias("yEnd"),
                pl.max_horizontal("vPeak", "vPeak_nxt").alias("vPeak"),
                (
                    (pl.col("xEnd_nxt") - pl.col("xStart")) ** 2
                    + (pl.col("yEnd_nxt") - pl.col("yStart")) ** 2
                )
                .sqrt()
                .alias("ampDeg"),
            ]
        )

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

        # Match the canonical saccade-table dtypes.
        for col in base_cols:
            if merged[col].dtype != sacc[col].dtype:
                merged = merged.with_columns(pl.col(col).cast(sacc[col].dtype))

        merged = merged.select(base_cols)

        # ───────────────────── 5 · build the final saccade table ────────────
        to_drop = pl.concat(
            [short_fix_pairs["idx_prev"], short_fix_pairs["idx_next"]]
        ).unique()
        new_sacc = (
            sacc.filter(~pl.col("idx").is_in(to_drop.implode()))
            .drop("idx")  # helper column gone
            .vstack(merged)  # add fused rows
            .sort(["phase", "eye", "tStart"])
        )

        # ───────────────────── 6 · store back and return ────────────────────
        self._fix = keep_fix.sort(["phase", "tStart"])
        self._sacc = new_sacc

    def collapse_fixations(self, threshold_px: float) -> None:
        """Collapse spatially adjacent fixations within each phase and eye.

        Saccades wholly between the first and last fixation in a merged group
        are discarded. The bordering saccades are adjusted to the merged
        fixation centroid. The trial's fixation and saccade tables are
        modified in place.

        Parameters
        ----------
        threshold_px : float
            Maximum Euclidean distance, in pixels, between consecutive
            fixations that should be merged.
        """

        # ────────────────── 0 · prepare helpers ──────────────────
        fix = self._fix.sort("tStart").with_row_index("fix_idx")
        sac = self._sacc.sort("tStart").with_row_index("sac_idx")

        new_fix_rows: list[dict] = []
        drop_sac_idx: set[int] = set()
        mod_sac: dict[int, dict] = {}  # idx → partial‑row updates

        # ────────────────── 1 · loop over phases ─────────────────
        for phase_val in fix["phase"].unique():  # ① per phase
            # Loop over eyes if needed
            for eye in fix["eye"].unique():
                fix_p = fix.filter(
                    (pl.col("phase") == phase_val) & (pl.col("eye") == eye)
                )
                sac_p = sac.filter(
                    (pl.col("phase") == phase_val) & (pl.col("eye") == eye)
                )

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
                        new_fix_rows.append(pool[0].copy())  # unchanged
                        i = j
                        continue

                    # ── merge the pool (>1 fix) ─────────────────────
                    first_fix, last_fix = pool[0], pool[-1]

                    merged_fix = first_fix.copy()
                    merged_fix.update(
                        {
                            "tEnd": last_fix["tEnd"],
                            "duration": sum(f["duration"] for f in pool),
                            "xAvg": np.mean([f["xAvg"] for f in pool]),
                            "yAvg": np.mean([f["yAvg"] for f in pool]),
                            "pupilAvg": np.mean([f["pupilAvg"] for f in pool]),
                        }
                    )
                    new_fix_rows.append(merged_fix)

                    # ── identify & drop fully‑internal saccades ─────
                    inside = sac_p.filter(
                        (pl.col("tStart") >= first_fix["tEnd"])
                        & (pl.col("tEnd") <= last_fix["tStart"])
                    )
                    drop_sac_idx.update(inside["sac_idx"].to_list())

                    # ── adjust bordering saccades ───────────────────
                    merged_x = merged_fix["xAvg"]
                    merged_y = merged_fix["yAvg"]

                    # previous saccade (ends at first_fix.tStart)
                    prev_df = sac_p.filter(pl.col("tEnd") <= first_fix["tStart"]).tail(
                        1
                    )
                    if prev_df.height:
                        prev = prev_df.row(0, named=True)
                        idx = prev["sac_idx"]
                        upd = {
                            "xEnd": merged_x,
                            "yEnd": merged_y,
                            "dx": merged_x - prev["xStart"],
                            "dy": merged_y - prev["yStart"],
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
                            "dx": nxt["xEnd"] - merged_x,
                            "dy": nxt["yEnd"] - merged_y,
                        }
                        upd["amplitude"] = hypot(upd["dx"], upd["dy"])
                        mod_sac.setdefault(idx, {}).update(upd)

                    i = j  # advance

        # ────────────────── 2 · rebuild tables ──────────────────
        # 2‑a  fixations
        new_fix = pl.DataFrame(
            new_fix_rows, schema=fix.drop("fix_idx").schema, orient="row"
        ).sort(["phase", "tStart"])

        # 2‑b  saccades: drop + modify in one pass
        new_sac_rows = []
        for row in sac.iter_rows(named=True):
            idx = row["sac_idx"]
            if idx in drop_sac_idx:
                continue  # discard
            if idx in mod_sac:  # apply edits
                row.update(mod_sac[idx])
                # re‑compute amplitude in case only dx/dy were provided
                if "amplitude" not in mod_sac[idx]:
                    row["amplitude"] = hypot(row["dx"], row["dy"])
            new_sac_rows.append({k: v for k, v in row.items() if k != "sac_idx"})

        new_sac = pl.DataFrame(
            new_sac_rows, schema=sac.drop("sac_idx").schema, orient="row"
        ).sort(["phase", "tStart"])

        # ────────────────── 3 · store back ──────────────────────
        self._fix = new_fix
        self._sacc = new_sac

    def save_rts(self):
        """Compute and cache the response time of each phase of this trial.

        The response time of a phase is the span between its first and last
        sample. Results are cached, so calling this repeatedly is cheap and
        subsequent calls do nothing.
        """
        if hasattr(self, "_rts"):
            return

        # Filter out empty phase rows
        filtered = self._samples.filter(pl.col("phase") != "")

        # Calculate RT as the difference between last and first tSample per phase
        rts = (
            filtered.group_by("phase")
            .agg([(pl.col("tSample").max() - pl.col("tSample").min()).alias("rt")])
            .with_columns([pl.lit(self.trial_number).alias("trial_number")])
        )

        self._rts = rts

    def rts(self):
        """Return the response time of each phase of this trial.

        Computes them on first access via :meth:`save_rts`.

        Returns
        -------
        polars.DataFrame
            One row per phase, with the ``phase`` name, its ``rt`` in the time
            units of the recording, and ``trial_number``.
        """
        if not hasattr(self, "_rts"):
            self.save_rts()
        return self._rts

    def is_trial_bad(self, phase, threshold=0.1):
        """Report whether a phase of this trial has too many invalid samples.

        Samples that fall inside a detected blink are excluded before counting,
        since a blink is expected data loss rather than a tracking failure. Of
        the remaining samples, one counts as bad when no gaze pair is finite or
        when the preprocessing step flagged it in the ``bad`` column.

        Parameters
        ----------
        phase : str
            Name of the trial phase to assess.
        threshold : float, default 0.1
            Maximum tolerated fraction of bad samples.

        Returns
        -------
        bool
            ``True`` when the bad-sample fraction exceeds ``threshold``, or
            when the phase contains no samples at all outside blinks.
        """
        samples = self._samples.filter(pl.col("phase") == phase)

        if self._blink is not None and self._blink.height > 0:
            for blink in self._blink.iter_rows(named=True):
                start, end = blink["tStart"], blink["tEnd"]
                samples = samples.filter(
                    ~((pl.col("tSample") > start) & (pl.col("tSample") < end))
                )

        total_samples = samples.height
        if total_samples == 0:
            return True

        gaze_pairs = [
            (x, y)
            for x, y in (("X", "Y"), ("LX", "LY"), ("RX", "RY"))
            if x in samples.columns and y in samples.columns
        ]
        valid_pair_expressions = [
            (
                pl.col(x).cast(pl.Float64, strict=False).is_finite()
                & pl.col(y).cast(pl.Float64, strict=False).is_finite()
            ).fill_null(False)
            for x, y in gaze_pairs
        ]
        invalid_gaze = (
            ~pl.any_horizontal(valid_pair_expressions)
            if valid_pair_expressions
            else pl.lit(False)
        )
        marked_bad = (
            pl.col("bad").cast(pl.Boolean, strict=False).fill_null(False)
            if "bad" in samples.columns
            else pl.lit(False)
        )
        bad_samples = samples.select(
            (invalid_gaze | marked_bad).sum().alias("count")
        ).item()
        return bad_samples / total_samples > threshold

    def is_trial_longer_than(self, seconds, phase):
        """Report whether a phase of this trial lasted longer than a limit.

        Parameters
        ----------
        seconds : float
            Duration limit, in seconds.
        phase : str
            Name of the trial phase to measure.

        Returns
        -------
        bool
            ``True`` when the phase lasted longer than ``seconds``. Trials
            without data for that phase are not considered long and return
            ``False``.
        """
        rt_row = self.rts().filter(pl.col("phase") == phase)
        if rt_row.is_empty():
            return False  # Or True if no data should be considered long
        return rt_row.select("rt").item() > seconds * 1000.0

    def _multimatch_fixations(self) -> pl.DataFrame:
        return self.fixations()

    def compute_multimatch(self, other_trial: "Trial", screen_height, screen_width):
        """Compare this trial's scanpath with another using MultiMatch.

        Requires the optional MultiMatch dependency, installed with
        ``pip install 'pyxations[multimatch]'``.

        Parameters
        ----------
        other_trial : Trial
            Trial whose scanpath is compared against this one.
        screen_height : int
            Height of the stimulus screen in pixels.
        screen_width : int
            Width of the stimulus screen in pixels.

        Returns
        -------
        list of float
            The five MultiMatch similarity dimensions: shape, direction,
            length, position and duration.

        Raises
        ------
        ImportError
            If MultiMatch is not installed.
        ValueError
            If either trial lacks the ``xAvg``, ``yAvg`` or ``duration``
            fixation columns.
        """
        trial_scanpath = _to_multimatch_scanpath(self._multimatch_fixations())
        trial_to_compare_scanpath = _to_multimatch_scanpath(
            other_trial._multimatch_fixations()
        )

        multimatch = _load_multimatch()
        return multimatch.docomparison(
            trial_scanpath,
            trial_to_compare_scanpath,
            (screen_width, screen_height),
        )
