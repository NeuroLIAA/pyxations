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
│           └── ...           # one recording per tracked eye
├── sourcedata/
│   └── sub-0001/
│       └── ses-<session>/
│           ├── ET/           # original vendor data
│           └── behavioral/   # original logs, when present
└── ...
```

Key points:

- **Subjects are renumbered**. Source filenames are scanned for an ID prefix (everything before the first `_`); they're sorted and re-issued as zero-padded `sub-0001`, `sub-0002`, … The original IDs are kept in `participants.tsv`.
- **Sessions come from the filename**. The next underscore-separated token after the subject ID becomes the session label. Use `session_substrings=N` if your session ID spans several tokens (e.g. `sub_2024-05-12_morning`).
- **Standardized samples live under `beh/`**. Each eye is represented by a
  headerless `physio.tsv.gz` table and its JSON metadata sidecar.
- **Original files live under `sourcedata/`**. Pyxations continues to use these
  vendor files when computing derivatives, without mixing non-BIDS filenames
  into the raw BIDS subject directories.
- **Validation is executable**. The test suite runs the official BIDS Validator
  against generated datasets for every supported input format.

## Derivatives layout

`compute_derivatives_for_dataset` writes a validator-checked sibling
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

The processed sample stream is stored as `physio.tsv.gz`; detected fixations,
saccades, blinks, and retained messages share its time axis in
`physioevents.tsv.gz`. Their JSON sidecars hold column definitions, the
detection algorithm, preprocessing recipe and provenance, calibration/header
payloads, and a reversible mapping to Pyxations' in-memory table columns.
`Experiment` reconstructs those Polars tables when loading the dataset, so
analysis and plotting remain independent of the on-disk format.

Naming notes:

- The processing algorithm is included in the `recording-` entity because the
  current BIDS eye-tracking schema does not allow a `desc-` entity on
  `physio`/`physioevents`.
- BIDS TSV.GZ/JSON is the canonical default. Feather and HDF5 remain available
  as explicit legacy exports through `exp_format`.
- Generated figures are kept below `docs/figures/`, an allowed opaque
  derivatives directory, so plots do not invalidate the dataset.

## Detection algorithms

Pyxations ships two pluggable eye-movement detectors. Pick one with `detection_algorithm=`.

- **`remodnav`**: wraps the [REMoDNaV](https://github.com/psychoinformatics-de/remodnav) package.
- **`engbert`**: Python port of the `detecteyemovements.m` routine from the [EYE-EEG toolbox](https://github.com/olafdimigen/eye-eeg/blob/master/detecteyemovements.m).

When the source is EyeLink, the tracker's own event reports are also written under `eyelink_events/`, so you can compare any algorithm's output against EyeLink's parser without re-running anything.

See [`pyxations.methods.eyemovement`](api_reference.md#methodseyemovement) for each algorithm's parameters and references.

## Supported input formats

`dataset_format=` selects the parser used to read raw recordings. Currently wired in:

- `eyelink`: EDF files; requires `edf2asc` (see [Requirements](requirements.md)).
- `tobii`: Tobii native exports.
- `gaze`: Gazepoint exports.
- `webgazer`: WebGazer.js browser-based recordings.

Pupil Labs Neon is not yet supported as a `dataset_format`. The `notebooks/` directory contains a manual example showing how to work with Neon recordings until first-class support lands.
