from pathlib import Path

import matplotlib
import pytest

from pyxations import compute_derivatives_for_dataset, dataset_to_bids

matplotlib.use("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"

EXAMPLE_CASES = {
    "eyelink": {
        "source": EXAMPLES_ROOT / "eyelink_visual_search",
        "dataset_name": "eyelink-visual-search",
        "algorithm": "eyelink",
        "session": "second",
        "compute_kwargs": {
            "force_best_eye": False,
            "msg_keywords": [
                "beginning_of_stimuli",
                "end_of_stimuli",
                "pressed",
            ],
            "start_msgs": {"search": ["beginning_of_stimuli"]},
            "end_msgs": {"search": ["end_of_stimuli"]},
        },
    },
    "webgazer": {
        "source": EXAMPLES_ROOT / "webgazer_antisaccade",
        "dataset_name": "webgazer-antisaccade",
        "algorithm": "remodnav",
        "session": "antisacadas",
        "compute_kwargs": {"screen_height": 768, "screen_width": 1024},
    },
    "tobii": {
        "source": EXAMPLES_ROOT / "tobii_sceneviewing",
        "dataset_name": "tobii-sceneviewing",
        "algorithm": "remodnav",
        "session": "sceneviewing",
        "compute_kwargs": {"screen_height": 1080, "screen_width": 1920},
    },
    "gaze": {
        "source": EXAMPLES_ROOT / "gazepoint_sart",
        "dataset_name": "gazepoint-sart",
        "algorithm": "remodnav",
        "session": "A",
        "compute_kwargs": {"screen_height": 1080, "screen_width": 1920},
    },
}


@pytest.fixture(scope="session")
def generated_datasets(tmp_path_factory):
    """Build raw and derivative BIDS datasets from committed source examples."""

    output_root = tmp_path_factory.mktemp("pyxations-generated")
    generated = {}
    for format_name, definition in EXAMPLE_CASES.items():
        raw = dataset_to_bids(
            output_root,
            definition["source"],
            definition["dataset_name"],
            format_name=format_name,
            authors=["Pyxations test suite", "NeuroLIAA"],
            overwrite=True,
        )
        derivatives = compute_derivatives_for_dataset(
            raw,
            format_name,
            definition["algorithm"],
            num_processes=1,
            overwrite=True,
            **definition["compute_kwargs"],
        )
        generated[format_name] = {
            **definition,
            "raw": raw,
            "derivatives": derivatives,
        }
    return generated
