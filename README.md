# Pyxations

[![codecov](https://codecov.io/gh/NeuroLIAA/pyxations/branch/main/graph/badge.svg)](https://codecov.io/gh/NeuroLIAA/pyxations)

<div>
  <table>
    <tr>
       <td>
          <img src="https://raw.githubusercontent.com/NeuroLIAA/pyxations/main/docs/images/pyxations_improved_logo.png" alt="pyxations logo" width="180">
       </td>
    </tr>
</table>
 
  <p>
    <strong>Pyxations</strong> is a Python library designed to analyze eye-tracking data, whether you are working with raw eye-tracking data or processed datasets. It helps researchers and developers extract useful insights from complex eye movements using Python's robust ecosystem.
  </p>
</div>

[📘 Documentation](https://neuroliaa.github.io/pyxations/)  
## Features

- **Validated BIDS Conversion**: Convert EyeLink, Tobii, GazePoint, and
  webcam/WebGazer samples, tracker events, and behavioral tables to raw BIDS
  while preserving the input folder verbatim under archival `sourcedata/`.
- **EyeLink Import**: Read EyeLink ASC exports directly, or accept EDF files by
  first converting them with SR Research's `edf2asc` utility. Pyxations then
  extracts messages, calibration reports, tracker events, and gaze samples from
  the resulting ASC data.
- **Trial Segmentation**: Segment continuous eye-tracking data into trials using flexible methods, including start/end messages, fixed durations, or explicit start/end times.
- **Behavioral Input Adapters**: Normalize behavioral CSV, TSV, or standard
  PsychoPy `New trial` logs into BIDS `events.tsv`, with source-independent
  column mapping and no PsychoPy runtime dependency.
- **Derivative Computation**: Compute derivatives directly from the normalized
  raw BIDS dataset; the archived source files are not required at runtime.
- **Analysis and Visualization**: Load derivative tables through the experiment,
  subject, session, and trial hierarchy and generate gaze, scanpath, calibration,
  and task-specific plots.
- **Eye Movement Detection**: Use REMoDNaV, the Engbert–Kliegl implementation,
  or EyeLink-reported events.
- **Saccades Direction Classification**: Classify saccades based on their start and end coordinates into four primary directions: right, left, up, and down.

  
## Requirements

* `Python 3.11` or newer is required.
* EyeLink EDF input requires the `edf2asc` program from the EyeLink Developers
  Kit on `PATH`. Existing EyeLink ASC files can be read directly.

### Dependencies

The base installation contains only the shared runtime stack:

- `numpy`
- `polars`
- `matplotlib`

Feature-specific packages are optional:

- `pyxations[remodnav]` adds REMoDNaV detection.
- `pyxations[multimatch]` adds MultiMatch scanpath comparison.
- `pyxations[video]` adds OpenCV-backed gaze animation, with optional video or
  image backgrounds.
- `pyxations[all]` installs all three feature groups.

Test and documentation tools are kept in optional dependency groups.

Canonical raw and derivative data are stored as compressed BIDS TSV/JSON, and
the complete tabular pipeline uses Polars in memory.


## Installation

Install the base package with `uv` or `pip`:

```bash
uv pip install pyxations
# or
pip install pyxations
```

Install only the features you use. The example below uses REMoDNaV:

```bash
pip install "pyxations[remodnav]"
```

For every optional feature:

```bash
pip install "pyxations[all]"
```

## Documentation

#### Full documentation and API reference are available at https://neuroliaa.github.io/pyxations

## Usage
### Minimal example
```python
import pyxations as pyx

# 1) Convert raw files to BIDS
bids_path = pyx.dataset_to_bids(
    target_folder_path="path/to/output",
    files_folder_path="path/to/source-recordings",
    dataset_name="dataset_name",
    format_name="eyelink",
    task_name="visualsearch",
)

# 2) Compute derivatives using REMoDNaV
msg_keywords = ["begin", "end", "press"]
start_msgs = {"search": ["beginning_of_stimuli"]}
end_msgs = {"search": ["end_of_stimuli"]}

pyx.compute_derivatives_for_dataset(
    bids_path,
    dataset_format="eyelink",
    detection_algorithm="remodnav",
    msg_keywords=msg_keywords,
    start_msgs=start_msgs,
    end_msgs=end_msgs,
    overwrite=True,
)
```

This produces two sibling, validator-tested BIDS datasets: the raw dataset and
`dataset_name_derivatives`. Canonical derivative samples and eye-movement
annotations are stored as compressed BIDS TSV.GZ files with JSON sidecars.
Pyxations reconstructs the same in-memory analysis tables when they are loaded.

## Contributing

Contributions are welcome! Please check out the [issues](https://github.com/NeuroLIAA/pyxations/issues) and submit a pull request if you'd like to help.

### To develop locally

```bash
# Clone repository
git clone https://github.com/NeuroLIAA/pyxations.git
cd pyxations

# Create virtual environment and install
uv venv
uv pip install -e '.[dev]'

# To work on documentation
uv pip install -e '.[docs]'
```


## License

This project is licensed under the MIT License.
