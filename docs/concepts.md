# Concepts

This page explains the data layout Pyxations produces and the main analysis choices you'll make.

## BIDS dataset layout

`dataset_to_bids` converts supported recordings to the eye-tracking
physiological format in BIDS 1.11.1:

```
<dataset_name>/
├── dataset_description.json
├── participants.tsv
├── sub-0001/
│   └── ses-<session>/
│       └── beh/
│           ├── sub-0001_ses-<session>_task-<task>_recording-eye1_physio.tsv.gz
│           ├── sub-0001_ses-<session>_task-<task>_recording-eye1_physio.json
│           ├── sub-0001_ses-<session>_task-<task>_recording-eye1_physioevents.tsv.gz
│           ├── sub-0001_ses-<session>_task-<task>_events.tsv
│           └── ...           # one recording per tracked eye
├── sourcedata/
│   └── ...                   # byte-for-byte copy of the input folder
└── ...
```

Key points:

- **Subjects are renumbered**. Source filenames are scanned for an ID prefix (everything before the first `_`); they're sorted and re-issued as zero-padded `sub-0001`, `sub-0002`, … The original IDs are kept in `participants.tsv`.
- **Sessions come from the filename**. The next underscore-separated token after the subject ID becomes the session label. Use `session_substrings=N` if your session ID spans several tokens (e.g. `sub_2024-05-12_morning`).
- **Standardized samples live under `beh/`**. Each eye is represented by a
  headerless `physio.tsv.gz` table and its JSON metadata sidecar.
- **Raw events also live under `beh/`**. Tracker messages and reported
  fixations, saccades, and blinks use `physioevents.tsv.gz`; behavioral trial
  tables use BIDS `events.tsv`.
- **Original files live under `sourcedata/` only for provenance**. The complete
  input tree is copied verbatim. Derivative computation and analysis read the
  normalized raw BIDS files and continue to work if `sourcedata/` is absent.
- **Validation is executable**. The test suite runs the official BIDS Validator
  against generated datasets for every supported input format.

## Derivatives layout

`compute_derivatives_for_dataset` writes a validator-tested sibling
`<dataset_name>_derivatives/` dataset that mirrors the raw BIDS
subject/session tree:

```
<dataset_name>_derivatives/
├── dataset_description.json               # DatasetType: derivative
├── participants.tsv
├── participants.json
└── sub-0001/
    └── ses-<session>/
        └── beh/
            ├── sub-0001_ses-<session>_task-<task>_recording-eye1remodnav_physio.tsv.gz
            ├── sub-0001_ses-<session>_task-<task>_recording-eye1remodnav_physio.json
            ├── sub-0001_ses-<session>_task-<task>_recording-eye1remodnav_physioevents.tsv.gz
            └── sub-0001_ses-<session>_task-<task>_recording-eye1remodnav_physioevents.json
└── ...
```

Two preprocessing steps run for every input format. Samples whose gaze fell
outside the screen, or was never tracked, are flagged in a `bad` column, which
plotting skips; this needs the screen size, taken from the recording or from
the `screen_width`/`screen_height` arguments. Saccades gain `deg` and `dir`
columns classifying their direction, where the event table carries endpoint
coordinates. Recordings without synchronisation messages get both steps; only
trial segmentation is skipped.

The processed sample stream is stored as `physio.tsv.gz`; detected fixations,
saccades, blinks, and retained messages share its time axis in
`physioevents.tsv.gz`. Their JSON sidecars hold column definitions, the
detection algorithm, preprocessing recipe and provenance, calibration/header
payloads, and a reversible mapping to Pyxations' in-memory table columns.
`Experiment` reconstructs those Polars tables when loading the dataset, so
analysis and plotting remain independent of the on-disk format.

BIDS 1.11.1 does not define a domain-specific derivative schema for detected
eye movements. Pyxations therefore follows the general BIDS Derivatives
conventions, uses the physiological recording file types, and explicitly
documents its additional columns and processing provenance. The raw and
derivative example datasets are checked without validation errors using BIDS
Validator 3.0.1 in continuous integration.

Naming notes:

- The processing algorithm is included in the `recording-` entity because the
  current BIDS eye-tracking schema does not allow a `desc-` entity on
  `physio`/`physioevents`.
- BIDS TSV.GZ/JSON is the sole canonical persisted derivative format.
- Generated figures are kept below `figures/` in the derivative dataset.
  Pyxations lists that directory in the derivative dataset's `.bidsignore`, so
  plots stay near their provenance without becoming canonical BIDS data or
  invalidating the dataset.

## Detection algorithms

Pyxations ships two software detector integrations and can reuse EyeLink's
reported events. Pick one with `detection_algorithm=`.

- **`remodnav`**: wraps the [REMoDNaV](https://github.com/psychoinformatics-de/remodnav) package.
- **`engbert`**: Python port of the `detecteyemovements.m` routine from the [EYE-EEG toolbox](https://github.com/olafdimigen/eye-eeg/blob/master/detecteyemovements.m).
- **`eyelink`**: reuses fixation, saccade, and blink events reported by the
  EyeLink parser; it is available only for EyeLink source data.

When the source is EyeLink, the tracker's own event reports are retained in the
raw BIDS `physioevents.tsv.gz`, so they can be selected as the derivative event
source or compared with another detector without reading the ASC/EDF again.

`remodnav` needs the optional extra of the same name; `engbert` and `eyelink`
need nothing beyond the base install.

`remodnav` and `engbert` classify events from gaze velocity, which needs
several samples inside a saccade to work: a saccade lasts roughly 30 to 80 ms.
Pyxations warns when a recording is sampled below 50 Hz, since at that rate a
saccade falls between one or two samples and the detected events describe the
sampling more than the eye. Webcam recordings can land there, because their
rate is set by the participant's browser and machine rather than by the
experimenter. For such recordings, analyse the gaze samples directly, as
[`SampleVisualization`](api/visualization.md) does.

See [`pyxations.methods.eyemovement`](api/detection.md) for each algorithm's parameters and references.

## Supported input formats

`dataset_format=` identifies the source format and its normalized BIDS event
semantics. Derivative computation itself reads the raw BIDS dataset. Currently
supported:

- `eyelink`: EDF or ASC files; EDF conversion requires `edf2asc` (see
  [Requirements](requirements.md)).
- `tobii`: Tobii native exports.
- `gaze`: Gazepoint exports.
- `webgazer`: WebGazer.js recordings exported by jsPsych, where the gaze
  samples of each trial arrive as JSON in a `webgazer_data` column. Gorilla
  writes WebGazer data with a different structure and is not supported yet.

`trial_number` is always `0, 1, 2, ...` in presentation order, whatever the
source format. jsPsych numbers every screen it presents, including instructions
and calibration, so WebGazer trials are renumbered on conversion and the
original index is kept alongside them as `source_trial_index`, in both the gaze
samples and the behavioral events.

These tracker formats are independent of the behavioral source format.
Associated CSV and TSV tables are preferred when present. Otherwise, a
PsychoPy `.log` can supply trial conditions, displayed-component updates, and
keypresses for BIDS `events.tsv`; parsing the log does not require PsychoPy to
be installed. A single `behavioral_column_map` maps fields from any of these
sources onto the experiment-level schema used by downstream analysis.

Pupil Labs Neon is not currently supported as a `dataset_format`.
