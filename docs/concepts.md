# Concepts

This page explains the data layout Pyxations produces and the main analysis choices you'll make.

## BIDS dataset layout

`dataset_to_bids` reorganizes raw recordings into a BIDS-inspired layout under `<target_folder_path>/<dataset_name>/`:

```
<dataset_name>/
├── participants.tsv          # maps new sub-XXXX IDs to your original IDs
├── sub-0001/
│   └── ses-<session>/
│       ├── ET/               # eye-tracking files (EDF, ASC, CSV, ...)
│       └── behavioral/       # optional behavioral logs (added manually)
├── sub-0002/
│   └── ses-<session>/
│       └── ET/
└── ...
```

Key points:

- **Subjects are renumbered**. Source filenames are scanned for an ID prefix (everything before the first `_`); they're sorted and re-issued as zero-padded `sub-0001`, `sub-0002`, … The original IDs are kept in `participants.tsv`.
- **Sessions come from the filename**. The next underscore-separated token after the subject ID becomes the session label. Use `session_substrings=N` if your session ID spans several tokens (e.g. `sub_2024-05-12_morning`).
- **`ET/` is the canonical eye-tracking folder**. Every input format (EyeLink, Tobii, Gazepoint, WebGazer) lands here. Adding `behavioral/` alongside is optional and only used if you pass `behavioral_columns=` when computing derivatives.

## Derivatives layout

`compute_derivatives_for_dataset` writes a sibling `<dataset_name>_derivatives/` folder mirroring the BIDS subject/session tree:

```
<dataset_name>_derivatives/
├── participants.tsv
├── sub-0001/
│   └── ses-<session>/
│       ├── header.feather                  # recording metadata
│       ├── calib.feather                   # calibration / validation lines
│       ├── msg.feather                     # filtered experimenter messages
│       ├── samples.feather                 # raw gaze samples (per-sample table)
│       ├── preprocessing_recipe.json       # parameters used for this session
│       ├── preprocessing_provenance.json   # versions, timestamps, file hashes
│       ├── eyelink_events/                 # events as reported by the tracker
│       │   ├── fix.feather
│       │   ├── sacc.feather
│       │   ├── blink.feather
│       │   └── plots/
│       └── remodnav_events/                # events from the chosen algorithm
│           ├── fix.feather
│           ├── sacc.feather
│           └── blink.feather
└── ...
```

Naming notes:

- The `<algorithm>_events/` folder name follows the `detection_algorithm` you passed (`remodnav_events/`, `engbert_events/`, …).
- `eyelink_events/` is only produced when the source format is EyeLink: it stores the events the tracker itself reported, useful as a baseline against the algorithmic detection.
- Tables are written as Apache Feather by default (fast, language-agnostic). HDF5 is also available via `exp_format=HDF_EXPORT`.

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
