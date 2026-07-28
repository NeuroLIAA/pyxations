from pyxations import Experiment


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
