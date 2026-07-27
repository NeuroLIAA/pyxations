# Quickstart

This page runs the full Pyxations pipeline on the small dataset bundled with the repository, so you can see what the inputs, outputs and APIs look like before pointing it at your own data.

## 0. Get the source example

Clone the repository:

```bash
git clone https://github.com/NeuroLIAA/pyxations.git
cd pyxations
```

The repository commits only small source recordings under `examples/`. Raw
BIDS datasets, derivative datasets, and figures are generated locally and are
not versioned.

## 1. Create raw BIDS and compute derivatives

```python
from pathlib import Path
import pyxations as pyx

repo = Path.cwd()
bids_path = pyx.dataset_to_bids(
    target_folder_path=repo / "generated",
    files_folder_path=repo / "examples" / "eyelink_visual_search",
    dataset_name="example_dataset",
    format_name="eyelink",
)

pyx.compute_derivatives_for_dataset(
    bids_dataset_folder=bids_path,
    dataset_format="eyelink",
    detection_algorithm="eyelink",
    msg_keywords=["begin", "end", "press"],
    start_msgs={"search": ["beginning_of_stimuli"]},
    end_msgs={"search": ["end_of_stimuli"]},
    overwrite=True,
)
```

This creates `generated/example_dataset/` and its BIDS-valid sibling
`generated/example_dataset_derivatives/`. Processed samples use
`physio.tsv.gz`/JSON and eye-movement annotations use
`physioevents.tsv.gz`/JSON.

## 2. Load and inspect

`Experiment` points at the **BIDS dataset path**; the matching `*_derivatives/` folder is found automatically. Call `load_data()` once with the same `detection_algorithm` you computed.

```python
from pyxations import Experiment

exp = Experiment(dataset_path=bids_path)
exp.load_data("eyelink")

print(list(exp.subjects.keys()))   # ['0001']

subject = exp.subjects["0001"]
session = subject.sessions["second"]
trial   = session.get_trial(0)

print(trial.fixations().head())
print(trial.saccades().head())
```

Tables come back as **polars** DataFrames.

## 3. Visualize one trial

```python
trial.plot_scanpath(screen_height=1080, screen_width=1920)
```

The plot is saved under
`generated/example_dataset_derivatives/figures/sub-0001/ses-second/eyelink/`.
The derivative dataset's `.bidsignore` excludes `figures/`, so the canonical
TSV.GZ/JSON outputs remain validator-compatible after plotting.

## Where to go next

- [Usage](usage.md): the same pipeline applied to your own data, with details on every parameter.
- [Concepts](concepts.md): what the BIDS and derivatives folders actually contain, and how to pick a detection algorithm.
- [API reference](api_reference.md): every public function and class.

For longer walkthroughs see the notebooks in [`notebooks/`](https://github.com/NeuroLIAA/pyxations/tree/main/notebooks):

- `Eyelink tutorial.ipynb`: full EyeLink pipeline.
- `multimatch_example.ipynb`: scanpath comparison with MultiMatch.
- `webgazer_example.ipynb`: webcam-based recordings.
- `driving_animation.ipynb`: visualization on a continuous task; it requires
  a separately supplied eye-tracking dataset.
