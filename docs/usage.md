# Usage

This page walks through the full Pyxations pipeline: from raw eye-tracking recordings to per-trial fixations and saccades ready for analysis.

The pipeline has three stages:

1. **Convert** raw recordings into a BIDS-formatted dataset.
2. **Compute derivatives**: parse files, detect eye movements, split into trials.
3. **Analyze and visualize** the resulting data.

## 1. Convert raw recordings to BIDS

`dataset_to_bids` takes a folder of raw recordings and produces a BIDS-compliant dataset.

```python
import pyxations as pyx

bids_path = pyx.dataset_to_bids(
    target_folder_path="path/to/output",  # where the BIDS dataset will be created
    files_folder_path="path/to/raw/files",  # folder containing raw recordings
    dataset_name="my_experiment",
    format_name="eyelink",  # eyelink | tobii | gaze | webgazer
    task_name="visualsearch",  # fallback when task- is not in filenames
)
```

The resulting layout looks like:

```
path/to/output/my_experiment/
├── dataset_description.json
├── participants.tsv
├── sub-0001/
│   └── ses-<session>/
│       └── beh/
│           ├── sub-0001_ses-<session>_task-<task>_recording-eye1_physio.tsv.gz
│           ├── sub-0001_ses-<session>_task-<task>_recording-eye1_physio.json
│           ├── sub-0001_ses-<session>_task-<task>_recording-eye1_physioevents.tsv.gz
│           └── sub-0001_ses-<session>_task-<task>_events.tsv
└── sourcedata/
    └── ...                    # verbatim copy of path/to/raw/files
└── ...
```

Subject IDs are inferred from the source filenames (everything before the first
`_`) and re-numbered as zero-padded `sub-0001`, `sub-0002`, … The mapping to
your original IDs is preserved in `participants.tsv`. The session label comes
from the next part of the filename: use `session_substrings=N` to take more
underscore-separated tokens.

Pyxations writes one headerless `physio.tsv.gz` recording per eye and a JSON
sidecar containing the sample columns, eye, sampling frequency, coordinate
description, and pupil units. Tracker events are normalized to
`physioevents.tsv.gz`, and behavioral tables to `events.tsv`. The original
input tree is retained byte-for-byte under `sourcedata/` for provenance only;
derivative processing does not read it.

### Behavioral CSV, TSV, and PsychoPy logs

Behavioral CSV and TSV files associated with a recording are normalized into
BIDS `events.tsv`. If no behavioral CSV or TSV is present, Pyxations also reads
PsychoPy text logs containing standard `New trial` records. It retains:

- the TrialHandler condition mapping;
- the zero-based trial order and PsychoPy repetition/index values;
- relevant component updates such as displayed images, text, position, rating,
  and slider values;
- keypresses observed before the next trial.

PsychoPy component columns use explicit names such as
`trial_image_image`. Map these to experiment concepts only when their meaning
is known:

```python
bids_path = pyx.dataset_to_bids(
    target_folder_path="path/to/output",
    files_folder_path="path/to/source-recordings",
    dataset_name="my_experiment",
    format_name="eyelink",
    behavioral_column_map={
        "trial_image_image": "stimulus",
        "is_target_present": "target_present",
    },
)
```

`behavioral_column_map` applies equally to CSV, TSV, and PsychoPy inputs. It
maps source-specific names onto the experiment concepts expected by downstream
analysis; for example, `VisualSearchExperiment` expects fields including
`stimulus`, `target_present`, and `correct_response`.

When a behavioral CSV/TSV and a log are both present, the tabular file takes
precedence; the original log remains archived under `sourcedata/`. PsychoPy's
clock is not assumed to be synchronized with the eye tracker, so its timestamps
are stored as `psychopy_onset` while BIDS `onset` remains `n/a`. Call
`pyx.read_behavioral_events()` for the source-independent reader, or
`pyx.psychopy_log_to_events()` when you specifically need the unmodified
PsychoPy fields.

To validate an output dataset locally, install the official BIDS Validator or
Deno and run:

```python
pyx.validate_bids_dataset(bids_path)
```

The continuous-integration suite performs this validation for synthetic
EyeLink, Tobii, GazePoint, and WebGazer datasets.

## 2. Compute derivatives

Derivatives are the parsed, processed outputs of the pipeline: samples,
messages, detected fixations, saccades and blinks, split into trials. They are
stored in a sibling `*_derivatives/` folder next to the raw BIDS dataset,
preserving its subject/session layout. The derivative dataset follows the
general BIDS Derivatives conventions because BIDS 1.11.1 does not yet define a
domain-specific derivative schema for detected eye movements. Both folders are
checked without validation errors in continuous integration using the official
BIDS Validator 3.0.1.

The derivative step starts from raw BIDS `physio`, `physioevents`, and
`events` files. This keeps conversion work out of repeated analyses and means
the dataset remains fully processable after its archival `sourcedata/` copy is
removed.

BIDS `physio.tsv.gz`/JSON and `physioevents.tsv.gz`/JSON files are the canonical
default. The sidecars preserve preprocessing provenance and reversible mappings
to the Pyxations tables, so the analysis API continues to expose Polars
DataFrames rather than tying downstream code to TSV files.

```python
pyx.compute_derivatives_for_dataset(
    bids_dataset_folder="path/to/output/my_experiment",
    dataset_format="eyelink",  # "eyelink" | "tobii" | "gaze" | "webgazer"
    detection_algorithm="remodnav",  # "remodnav" | "engbert" | "eyelink"
    msg_keywords=["begin", "end", "press"],
    start_msgs={"search": ["beginning_of_stimuli"]},
    end_msgs={"search": ["end_of_stimuli"]},
    overwrite=True,
)
```

The returned derivatives path contains a `dataset_description.json` with
`DatasetType` set to `derivative`, `GeneratedBy` metadata for Pyxations, and a
link to the source dataset. Its TSV.GZ/JSON tables are the canonical inputs to
the analysis API. Processing is serial by default, which avoids worker startup
and table-serialization overhead for small datasets. For a large dataset with
many independent sessions, pass `num_processes=N` explicitly to process
sessions in parallel.

### Trial segmentation parameters

- **`msg_keywords`**: substrings used to filter which experimenter messages from the recording are kept in the parsed output. Anything not matching is discarded to keep the message table small.
- **`start_msgs`** / **`end_msgs`**: define how each trial is delimited based on messages logged during the recording.

Pyxations accepts one of three segmentation strategies, picked by which kwargs you pass:

1. `start_msgs` + `end_msgs`: trials run from a start message to an end message.
2. `start_msgs` + `durations`: fixed-duration trials anchored at each start message.
3. `start_times` + `end_times`: explicit per-trial timestamps (typically loaded from a behavioral log).

See [`pyxations.pre_processing`](api/preprocessing.md) for the full segmentation API.

### Detection algorithms

- **`remodnav`**: wraps the [REMoDNaV](https://github.com/psychoinformatics-de/remodnav) package.
- **`engbert`**: Python port of `detecteyemovements.m` from the [EYE-EEG toolbox](https://github.com/olafdimigen/eye-eeg/blob/master/detecteyemovements.m).
- **`eyelink`**: reuses EyeLink-reported events and is available only for
  EyeLink source recordings.

See [`pyxations.methods.eyemovement`](api/detection.md) for parameters specific to each algorithm.

## 3. Load and analyze derivatives

Once derivatives exist, the high-level `Experiment` API gives access to per-subject, per-session, per-trial data. Point it at the **BIDS dataset path** (not the derivatives folder); the sibling `*_derivatives/` is found automatically.

```python
from pyxations import Experiment

exp = Experiment(dataset_path="path/to/output/my_experiment")
exp.load_data("remodnav")  # must match the detection_algorithm you computed

for subject_id, subject in exp.subjects.items():
    for session_id, session in subject.sessions.items():
        fixations = session.fixations()  # polars.DataFrame
        saccades = session.saccades()  # polars.DataFrame
        blinks = session.blinks()  # polars.DataFrame
        samples = session.samples()  # polars.DataFrame (raw gaze)
        pupil = session.pupil_samples()  # rows with pupil measurements

# Access a specific trial
trial = exp.get_trial(subject_id="0001", session_id="second", trial_number=0)
trial.fixations()
trial.saccades()
trial.blinks()
trial.pupil_samples()
```

`exp.subjects` and `subject.sessions` are dicts keyed by ID strings (`"0001"`, …). Tables are returned as **polars** DataFrames.

The same `blinks()` and `pupil_samples()` accessors are available at the
experiment, subject, session, and trial levels. `pupil_samples()` preserves
the source tracker's pupil columns and units; consult the recording's BIDS
JSON sidecar to determine whether the values represent diameter or area and
which units were reported. Pyxations exposes these measurements for analysis;
pupillometry preprocessing such as deblinking, interpolation or baseline
correction is left to a dedicated package.

### Trial and session quality filtering

Trial-level and session-level exclusion are deliberately separate decisions.
First, `assess_trial_quality()` classifies every trial in a session from its
fraction of invalid gaze samples without modifying the hierarchy. You can use
`session.remove_bad_trials()` when you only want to remove those individual
trials.

For a combined policy, call:

```python
result = exp.remove_bad_trials_and_sessions(
    phase="search",
    trial_nan_threshold=0.1,
    session_bad_trial_threshold=0.25,
)
```

Each session is assessed before any trial is removed. If more than 25% of its
trials are bad in this example, the entire session is removed; otherwise only
the bad trials are removed. A subject whose last session is removed is also
removed explicitly from `exp.subjects`. The returned `QualityFilterResult`
reports removed trials, removed sessions, removed subjects, and trials
discarded as part of session removal.

For visual-search paradigms, `VisualSearchExperiment` adds helpers for target/distractor analyses.

## 4. Visualization

Each `Trial` knows how to plot itself:

```python
trial.plot_scanpath(screen_height=1080, screen_width=1920)
# Requires: pip install "pyxations[video]"
trial.plot_animation(screen_height=1080, screen_width=1920)
```

For aggregate plots across a session or experiment, use the `Visualization` class directly:

```python
from pyxations import Visualization

vis = Visualization(
    derivatives_folder_path=exp.derivatives_path,
    events_detection_algorithm="remodnav",
)
vis.fix_duration(session.fixations())
vis.sacc_amplitude(session.saccades())
vis.sacc_main_sequence(session.saccades())
```

See [`pyxations.visualization`](api/visualization.md) for the full plot catalog.

## Worked examples

The repository includes focused notebooks under
[`docs/tutorials/`](https://github.com/NeuroLIAA/pyxations/tree/main/docs/tutorials):

- `eyelink_example.ipynb`: full EyeLink pipeline on the bundled example dataset.
- `tobii_example.ipynb`: complete workflow for a Tobii tabular export.
- `gazepoint_example.ipynb`: complete workflow for a GazePoint CSV export.
- `multimatch_example.ipynb`: scanpath comparison with MultiMatch.
- `webgazer_example.ipynb`: webcam-based recordings.
- `driving_animation.ipynb`: visualization on a continuous task; the bundled
  video is included, but its eye-tracking dataset must be supplied separately.

Small vendor-format source recordings under `examples/` reproduce the
conversion, derivative, hierarchy, and plotting workflows end-to-end.
