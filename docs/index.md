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
- **Behavioral-log support**: normalize PsychoPy trial logs, condition fields,
  displayed-component updates, and keypresses into BIDS events tables.
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
- [API reference](api/index.md): public modules and functions.
- [Contributing](contributing.md): dev setup, tests, building the docs.

## Citation

If you use Pyxations in your work, please cite the paper:

> Ruarte, G., Juantorena, G., González, J., & Kamienkowski, J. E. (2026). Pyxations: organizing, parsing, and analysing eye-movement data using Python. *Journal of Open Source Software*. [https://doi.org/10.21105/joss.10899](https://doi.org/10.21105/joss.10899)

```bibtex
@article{Ruarte2026Pyxations,
  author  = {Ruarte, Gonzalo and Juantorena, Gustavo and Gonz{\'a}lez, Joaqu{\'\i}n and Kamienkowski, Juan Esteban},
  title   = {Pyxations: organizing, parsing, and analysing eye-movement data using Python},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {10.21105/joss.10899}
}
```

The archived source code of each release is available on Zenodo; version 0.4.0 is [10.5281/zenodo.23771267](https://doi.org/10.5281/zenodo.23771267).
