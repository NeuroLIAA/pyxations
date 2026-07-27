import pytest

from pyxations import Experiment
from pyxations.bids import validate_bids_dataset, validator_command


def test_multipanel_is_written_without_invalidating_derivatives(
    generated_datasets,
):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    experiment.plot_multipanel(display=False)

    figure = (
        case["derivatives"] / "figures" / "group" / "eyelink" / "multipanel_search.png"
    )
    assert figure.is_file()

    command = validator_command()
    if command is None:
        pytest.skip("Official BIDS Validator or Deno is not installed")
    validate_bids_dataset(case["derivatives"], command=command)
