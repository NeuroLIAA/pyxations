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
    target_folder_path="path/to/output",        # where the BIDS dataset will be created
    files_folder_path="path/to/raw/files",      # folder containing raw recordings
    dataset_name="my_experiment",
    format_name="eyelink",                      # eyelink | tobii | gaze | webgazer
    task_name="visualsearch",                   # fallback when task- is not in filenames
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
preserving its subject/session layout. Both folders are checked with the
official BIDS Validator.

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
    dataset_format="eyelink",            # "eyelink" | "tobii" | "gaze" | "webgazer"
    detection_algorithm="remodnav",      # "remodnav" | "engbert"
    msg_keywords=["begin", "end", "press"],
    start_msgs={"search": ["beginning_of_stimuli"]},
    end_msgs={"search": ["end_of_stimuli"]},
    overwrite=True,
)
```

The returned derivatives path contains a `dataset_description.json` with
`DatasetType` set to `derivative`, `GeneratedBy` metadata for Pyxations, and a
link to the source dataset. Feather and HDF5 exports are retained only for
backward compatibility and can be selected explicitly with `exp_format`.

### Trial segmentation parameters

- **`msg_keywords`**: substrings used to filter which experimenter messages from the recording are kept in the parsed output. Anything not matching is discarded to keep the message table small.
- **`start_msgs`** / **`end_msgs`**: define how each trial is delimited based on messages logged during the recording.

Pyxations accepts one of three segmentation strategies, picked by which kwargs you pass:

1. `start_msgs` + `end_msgs`: trials run from a start message to an end message.
2. `start_msgs` + `durations`: fixed-duration trials anchored at each start message.
3. `start_times` + `end_times`: explicit per-trial timestamps (typically loaded from a behavioral log).

See [`pyxations.pre_processing`](api_reference.md#pre_processing) for the full segmentation API.

### Detection algorithms

- **`remodnav`**: wraps the [REMoDNaV](https://github.com/psychoinformatics-de/remodnav) package.
- **`engbert`**: Python port of `detecteyemovements.m` from the [EYE-EEG toolbox](https://github.com/olafdimigen/eye-eeg/blob/master/detecteyemovements.m).

See [`pyxations.methods.eyemovement`](api_reference.md#methodseyemovement) for parameters specific to each algorithm.

## 3. Load and analyze derivatives

Once derivatives exist, the high-level `Experiment` API gives access to per-subject, per-session, per-trial data. Point it at the **BIDS dataset path** (not the derivatives folder); the sibling `*_derivatives/` is found automatically.

```python
from pyxations import Experiment

exp = Experiment(dataset_path="path/to/output/my_experiment")
exp.load_data("remodnav")   # must match the detection_algorithm you computed

for subject_id, subject in exp.subjects.items():
    for session_id, session in subject.sessions.items():
        fixations = session.fixations()   # polars.DataFrame
        saccades  = session.saccades()    # polars.DataFrame
        samples   = session.samples()     # polars.DataFrame (raw gaze)

# Access a specific trial
trial = exp.get_trial(subject_id="0001", session_id="second", trial_number=0)
trial.fixations()
trial.saccades()
```

`exp.subjects` and `subject.sessions` are dicts keyed by ID strings (`"0001"`, …). Tables are returned as **polars** DataFrames.

For visual-search paradigms, `VisualSearchExperiment` adds helpers for target/distractor analyses.

## 4. Visualization

Each `Trial` knows how to plot itself:

```python
trial.plot_scanpath(screen_height=1080, screen_width=1920)
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

See [`pyxations.visualization`](api_reference.md#visualization) for the full plot catalog.

## Worked examples

The repository includes runnable notebooks under [`notebooks/`](https://github.com/NeuroLIAA/pyxations/tree/main/notebooks):

- `Eyelink tutorial.ipynb`: full EyeLink pipeline on the bundled example dataset.
- `multimatch_example.ipynb`: scanpath comparison with MultiMatch.
- `webgazer_example.ipynb`: webcam-based recordings.
- `driving_animation.ipynb`: visualization on a continuous task.

Small vendor-format source recordings under `examples/` reproduce the
conversion, derivative, hierarchy, and plotting workflows end-to-end.
