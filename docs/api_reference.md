# API Reference

This page documents all public Pyxations modules and functions. Docstrings are rendered automatically from the source.

## Top-level entry points

The most common entry points are re-exported from the package root:

```python
from pyxations import (
    dataset_to_bids,
    read_behavioral_events,
    psychopy_log_to_events,
    compute_derivatives_for_dataset,
    validate_bids_dataset,
    BIDSValidationError,
    PreProcessing,
    EngbertDetection,
    Visualization,
    SampleVisualization,
    Experiment,
    VisualSearchExperiment,
    BIDSDerivativeExport,
    SessionTables,
)
```

`RemodnavDetection` is also available from the package root after installing
the `remodnav` extra:

```python
from pyxations import RemodnavDetection
```

## Behavioral input

Read and map behavioral CSV, TSV, or PsychoPy log data through a common
source-independent interface.

::: pyxations.behavior

PsychoPy log parsing does not require PsychoPy to be installed:

::: pyxations.psychopy

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

## export

The canonical BIDS TSV.GZ/JSON derivative reader and writer.

::: pyxations.export
