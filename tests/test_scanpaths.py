import pytest

from pyxations import Experiment
from pyxations.visualization.visualization import Visualization


def test_visualization_accepts_optional_detector_names(tmp_path):
    visualization = Visualization(tmp_path, "remodnav")

    assert visualization.events_detection_folder.name == "remodnav"


@pytest.mark.parametrize("name", ["", "../outside"])
def test_visualization_rejects_invalid_detector_names(tmp_path, name):
    with pytest.raises(ValueError):
        Visualization(tmp_path, name)


def test_scanpath_is_written_to_ignored_figures_directory(generated_datasets):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])

    experiment.get_trial("0001", "second", 0).plot_scanpath(1080, 1920, display=False)

    figure = (
        case["derivatives"]
        / "figures"
        / "sub-0001"
        / "ses-second"
        / "eyelink"
        / "scanpath_0_search.png"
    )
    assert figure.is_file()
