"""Public Pyxations API.

Feature-specific dependencies are imported lazily so the base package can be
used without installing REMoDNaV, MultiMatch, or OpenCV.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .analysis.generic import Experiment
from .analysis.visual_search import VisualSearchExperiment
from .behavior import read_behavioral_events
from .bids import BIDSValidationError, validate_bids_dataset
from .bids_formatting import compute_derivatives_for_dataset, dataset_to_bids
from .export import BIDSDerivativeExport
from .methods.eyemovement.engbert import EngbertDetection
from .pre_processing import PreProcessing
from .psychopy import psychopy_log_to_events
from .tables import SessionTables
from .visualization.samples import SampleVisualization
from .visualization.visualization import Visualization

__all__ = [
    "BIDSDerivativeExport",
    "BIDSValidationError",
    "EngbertDetection",
    "Experiment",
    "PreProcessing",
    "RemodnavDetection",
    "SampleVisualization",
    "SessionTables",
    "VisualSearchExperiment",
    "Visualization",
    "compute_derivatives_for_dataset",
    "dataset_to_bids",
    "psychopy_log_to_events",
    "read_behavioral_events",
    "validate_bids_dataset",
]


def __getattr__(name: str) -> Any:
    """Load optional public objects only when they are requested."""
    if name == "RemodnavDetection":
        try:
            module = import_module(".methods.eyemovement.remodnav_detector", __name__)
        except ImportError as exc:
            if exc.name and exc.name.startswith("remodnav"):
                raise ImportError(
                    "REMoDNaV support is optional. Install it with "
                    "`pip install 'pyxations[remodnav]'`."
                ) from exc
            raise
        return module.RemodnavDetection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
