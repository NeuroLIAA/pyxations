# Quickstart

This page runs the full Pyxations pipeline on the small dataset bundled with the repository, so you can see what the inputs, outputs and APIs look like before pointing it at your own data.

## 0. Get the example dataset

Clone the repository (or download the `example_dataset/` folder from GitHub):

```bash
git clone https://github.com/NeuroLIAA/pyxations.git
cd pyxations
```

The bundled example uses Pyxations' legacy pre-BIDS layout: four subjects
(`sub-0001` … `sub-0004`), each with one session containing EyeLink recordings
under `ET/` and behavioral logs under `behavioral/`. It remains available for
reproducing the existing analysis tutorial. New datasets created with
`dataset_to_bids` use standardized BIDS `physio` files and retain the original
vendor layout under `sourcedata/`.

## 1. Compute derivatives

```python
import pyxations as pyx

pyx.compute_derivatives_for_dataset(
    bids_dataset_folder="example_dataset",
    dataset_format="eyelink",
    detection_algorithm="remodnav",
    msg_keywords=["begin", "end", "press"],
    start_msgs={"search": ["beginning_of_stimuli"]},
    end_msgs={"search": ["end_of_stimuli"]},
    overwrite=True,
)
```

This creates a BIDS-valid sibling dataset, `example_dataset_derivatives/`.
Processed samples use `physio.tsv.gz`/JSON and eye-movement annotations use
`physioevents.tsv.gz`/JSON.

A pre-computed legacy Feather copy of the derivatives ships in the repo
(`example_dataset_derivatives/`), so you can also skip this step and jump
straight to loading. `Experiment` detects that older layout automatically.

## 2. Load and inspect

`Experiment` points at the **BIDS dataset path**; the matching `*_derivatives/` folder is found automatically. Call `load_data()` once with the same `detection_algorithm` you computed.

```python
from pyxations import Experiment

exp = Experiment(dataset_path="example_dataset")
exp.load_data("remodnav")

print(list(exp.subjects.keys()))   # ['0001', '0002', '0003', '0004']

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

The plot is saved under `example_dataset_derivatives/sub-0001/ses-second/remodnav_events/plots/`.

## Where to go next

- [Usage](usage.md): the same pipeline applied to your own data, with details on every parameter.
- [Concepts](concepts.md): what the BIDS and derivatives folders actually contain, and how to pick a detection algorithm.
- [API reference](api_reference.md): every public function and class.

For longer, runnable walkthroughs see the notebooks in [`notebooks/`](https://github.com/NeuroLIAA/pyxations/tree/main/notebooks):

- `Eyelink tutorial.ipynb`: full EyeLink pipeline.
- `multimatch_example.ipynb`: scanpath comparison with MultiMatch.
- `webgazer_example.ipynb`: webcam-based recordings.
- `driving_animation.ipynb`: visualization on a continuous task.
