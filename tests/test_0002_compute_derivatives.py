import json

import pytest

import pyxations.analysis.generic as generic_analysis
from pyxations import Experiment, VisualSearchExperiment
from pyxations.bids import validate_bids_dataset, validator_command
from pyxations.export.bids import BIDSDerivativeExport


@pytest.mark.parametrize("format_name", ["eyelink", "webgazer", "tobii", "gaze"])
def test_examples_generate_canonical_bids_derivatives(generated_datasets, format_name):
    case = generated_datasets[format_name]
    derivatives = case["derivatives"]
    description = json.loads(
        (derivatives / "dataset_description.json").read_text(encoding="utf-8")
    )

    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "Pyxations"
    assert (derivatives / ".bidsignore").read_text(encoding="utf-8") == (
        "figures\nfigures/**\n"
    )
    assert not list(derivatives.rglob("*.feather"))
    assert not list(derivatives.rglob("*.hdf5"))

    session = derivatives / "sub-0001" / f"ses-{case['session']}"
    assert {path.name for path in session.iterdir()} == {"beh"}
    assert len(list((session / "beh").glob("*_physio.tsv.gz"))) == 1
    assert len(list((session / "beh").glob("*_physioevents.tsv.gz"))) == 1

    bundle = BIDSDerivativeExport().read_session(session, case["algorithm"])
    assert not bundle.samples.is_empty()
    assert {
        "samples",
        "fixations",
        "saccades",
        "blinks",
        "messages",
        "calibration",
        "header",
        "behavioral_events",
    } <= set(bundle.__dataclass_fields__)

    sample_start = bundle.samples.get_column("tSample").min()
    sample_end = bundle.samples.get_column("tSample").max()
    for events in (bundle.fixations, bundle.saccades):
        if events.is_empty():
            continue
        assert events.get_column("tStart").min() >= sample_start
        assert events.get_column("tEnd").max() <= sample_end


@pytest.mark.parametrize("format_name", ["eyelink", "webgazer", "tobii", "gaze"])
def test_generated_examples_pass_official_bids_validator(
    generated_datasets, format_name
):
    command = validator_command()
    if command is None:
        pytest.skip("Official BIDS Validator or Deno is not installed")

    case = generated_datasets[format_name]
    validate_bids_dataset(case["raw"], command=command)
    validate_bids_dataset(case["derivatives"], command=command)


def test_generic_analysis_hierarchy_uses_generated_bids(generated_datasets):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])

    assert len(experiment) == 1
    subject = experiment["0001"]
    assert len(subject) == 1
    session = subject["second"]
    assert len(session) == 2
    trial = session[0]

    assert trial.trial_number == 0
    assert not trial.samples().is_empty()
    assert not trial.fixations().is_empty()
    assert not trial.blinks().is_empty()
    assert not trial.pupil_samples().is_empty()
    assert {"LPupil", "RPupil"}.issubset(trial.pupil_samples().columns)
    assert set(experiment.samples()["subject_id"].unique()) == {"0001"}
    assert set(experiment.blinks()["subject_id"].unique()) == {"0001"}
    assert set(experiment.pupil_samples()["subject_id"].unique()) == {"0001"}


def test_pupil_accessor_does_not_invent_webcam_measurements(
    generated_datasets,
):
    case = generated_datasets["webgazer"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])

    assert experiment.pupil_samples().is_empty()


def test_webgazer_hierarchy_uses_sequential_trial_numbers(generated_datasets):
    """Raw jsPsych indices must not leak into the public trial lookup API."""
    case = generated_datasets["webgazer"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])

    session = experiment["0001"][case["session"]]
    trial = session.get_trial(0)

    assert trial.trial_number == 0
    assert not trial.samples().is_empty()
    assert trial.samples().get_column("source_trial_index").n_unique() == 1
    assert trial.samples().get_column("source_trial_index")[0] != 0


def test_generic_trials_support_optional_multimatch(generated_datasets, monkeypatch):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    session = experiment["0001"]["second"]

    class FakeMultimatch:
        @staticmethod
        def docomparison(first, second, screen):
            return len(first), len(second), first.dtype.names, screen

    monkeypatch.setattr(generic_analysis, "_load_multimatch", lambda: FakeMultimatch)
    result = session[0].compute_multimatch(session[1], 1080, 1920)

    assert result[0] > 0
    assert result[1] > 0
    assert result[2] == ("start_x", "start_y", "duration")
    assert result[3] == (1920, 1080)


def test_visual_search_hierarchy_uses_bids_events(generated_datasets):
    case = generated_datasets["eyelink"]
    experiment = VisualSearchExperiment(
        case["raw"],
        search_phase_name="search",
        memorization_phase_name="memorization",
    )
    experiment.load_data(case["algorithm"])

    subject = experiment["0001"]
    session = subject["second"]
    trial = session[0]

    assert len(session) == 2
    assert trial.stimulus.endswith(".jpg")
    assert isinstance(trial.target_present, bool)
    assert not trial.search_samples().is_empty()
