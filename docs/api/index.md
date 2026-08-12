# API reference

This section documents every public Pyxations module, class and function.
Docstrings are rendered automatically from the source, so what you read here is
what the installed package provides.

The pages follow the order of the pipeline itself: recordings are converted to
raw BIDS, segmented into trials, annotated with detected eye movements, and then
loaded back for analysis and plotting.

| Page | What it covers |
| --- | --- |
| [Conversion to BIDS](conversion.md) | Reading vendor recordings and behavioral tables, and writing the raw BIDS dataset |
| [Preprocessing](preprocessing.md) | Trial segmentation, bad-sample marking and provenance recipes |
| [Event detection](detection.md) | Fixation, saccade and blink detection algorithms |
| [Analysis](analysis.md) | The experiment, subject, session and trial hierarchy |
| [Visual search](visual_search.md) | The search-task specialization of that hierarchy |
| [Visualization](visualization.md) | Scanpaths, summary panels, calibration plots and animations |
| [Data storage](storage.md) | The canonical table container and the BIDS derivative reader and writer |

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
