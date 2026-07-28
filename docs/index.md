# Pyxations

**Pyxations** is a Python library for analyzing eye-tracking data, from raw recordings to processed datasets. It standardizes data into BIDS layout, runs fixation/saccade detection with multiple algorithms, segments recordings into trials, and provides visualization utilities: so researchers can focus on analysis instead of file plumbing.

## Who is it for

Cognitive scientists, vision researchers and developers working with eye-tracking recordings (EyeLink, Tobii, Gazepoint, WebGazer) who want a reproducible pipeline in Python.

## Quick install

```bash
pip install "pyxations[remodnav]"
```

For EDF inputs you also need EyeLink's `edf2asc` tool on your `PATH`. See [Requirements](requirements.md) and [Installation](installation.md) for details.

## 60-second example

```python
import pyxations as pyx

# 1) Convert raw recordings into a BIDS dataset
bids_path = pyx.dataset_to_bids(
    target_folder_path="path/to/output",
    files_folder_path="path/to/raw/edf/files",
    dataset_name="my_experiment",
)

# 2) Compute derivatives (parse, detect fixations/saccades, split into trials)
pyx.compute_derivatives_for_dataset(
    bids_dataset_folder=bids_path,
    dataset_format="eyelink",
    detection_algorithm="remodnav",
    msg_keywords=["begin", "end", "press"],
    start_msgs={"search": ["beginning_of_stimuli"]},
    end_msgs={"search": ["end_of_stimuli"]},
    overwrite=True,
)
```

See [Usage](usage.md) for an end-to-end walkthrough.

## Features

- **Validated BIDS conversion**: write per-eye physiological recordings from
  EyeLink, Tobii, GazePoint, and webcam/WebGazer data, normalize tracker and
  behavioral events, and retain a verbatim archival copy under `sourcedata/`.
- **EyeLink import**: read ASC exports directly, or convert EDF files with
  `edf2asc` before extracting messages, calibration reports, tracker events,
  and gaze samples.
- **Multi-vendor support**: input formats for EyeLink, Tobii, Gazepoint and WebGazer.
- **Trial segmentation**: split continuous recordings using start/end messages, fixed durations or explicit timestamps.
- **Eye movement detection**: fixations and saccades with REMoDNaV,
  Engbert–Kliegl, or EyeLink-reported events.
- **Saccade direction classification**: right / left / up / down based on start–end coordinates.
- **Derivatives pipeline**: reproducible per-subject derivatives computed from
  raw BIDS without re-reading vendor files.
- **Visualization**: plots for samples, events and per-trial inspection.

## Where to go next

- [Requirements](requirements.md): Python version, `edf2asc`, dependencies.
- [Installation](installation.md): pip, uv, from source.
- [Usage](usage.md): end-to-end pipeline with output layout.
- [API reference](api_reference.md): public modules and functions.
- [Contributing](contributing.md): dev setup, tests, building the docs.

## Citation

If you use Pyxations in academic work, please cite the accompanying paper (see `paper.md` in the [repository](https://github.com/NeuroLIAA/pyxations)).
