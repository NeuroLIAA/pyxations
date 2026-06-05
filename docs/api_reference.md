# API Reference

This page documents all public Pyxations modules and functions. Docstrings are rendered automatically from the source.

## Top-level entry points

The most common entry points are re-exported from the package root:

```python
from pyxations import (
    dataset_to_bids,
    compute_derivatives_for_dataset,
    PreProcessing,
    RemodnavDetection,
    EngbertDetection,
    Visualization,
    SampleVisualization,
    Experiment,
    VisualSearchExperiment,
    get_ordered_trials_from_psycopy_logs,
)
```

## bids_formatting

BIDS conversion and dataset-level derivatives computation.

::: pyxations.bids_formatting

## pre_processing

Per-recording parsing and trial segmentation.

::: pyxations.pre_processing

## methods.eyemovement

Fixation and saccade detection algorithms.

### eye_movement_detection

::: pyxations.methods.eyemovement.eye_movement_detection

### REMoDNaV

::: pyxations.methods.eyemovement.REMoDNaV

### Engbert–Kliegl

::: pyxations.methods.eyemovement.engbert

## analysis

High-level experiment objects for loading and iterating over derivatives.

### generic

::: pyxations.analysis.generic

### visual_search

::: pyxations.analysis.visual_search

## visualization

Plotting utilities for scanpaths, fixations, saccades and raw samples.

### visualization

::: pyxations.visualization.visualization

### samples

::: pyxations.visualization.samples

## formats

Vendor-specific input readers. Selected via the `dataset_format` argument of `compute_derivatives_for_dataset`.

::: pyxations.formats.generic

## export

Writers for persisting derivatives in different on-disk formats.

::: pyxations.export

## utils

Helpers for log alignment and miscellaneous utilities.

::: pyxations.utils
