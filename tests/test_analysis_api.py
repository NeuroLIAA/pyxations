import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import pyxations.analysis.generic as generic_module
import pyxations.analysis.visual_search as visual_search_module
from pyxations import Experiment, VisualSearchExperiment
from pyxations.analysis.generic import (
    QualityFilterResult,
    SessionQualityAssessment,
    _find_fixation_cutoff,
    _to_multimatch_scanpath,
)
from pyxations.visualization.visualization import Visualization


def _visual_search_experiment(generated_datasets):
    case = generated_datasets["eyelink"]
    experiment = VisualSearchExperiment(
        case["raw"],
        search_phase_name="search",
        memorization_phase_name="memorization",
    )
    experiment.load_data(case["algorithm"])
    return experiment


def test_generic_hierarchy_public_api_and_calibration(generated_datasets, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])

    subject = experiment.get_subject("0001")
    session = experiment.get_session("0001", "second")
    trial = experiment.get_trial("0001", "second", 0)

    assert list(experiment) == ["0001"]
    assert list(subject) == ["second"]
    assert list(session) == [0, 1]
    assert experiment["0001"] is subject
    assert subject["second"] is session
    assert session[0] is trial
    assert len(experiment) == len(subject) == 1
    assert len(session) == 2
    assert "Experiment" in repr(experiment)
    assert "Subject" in repr(subject)
    assert "Session" in repr(session)
    assert "Trial" in repr(trial)

    assert not experiment.rts().is_empty()
    assert not subject.rts().is_empty()
    assert not session.rts().is_empty()
    assert not experiment.fixations().is_empty()
    assert not experiment.saccades().is_empty()
    assert not experiment.samples().is_empty()
    assert not experiment.blinks().is_empty()
    assert not experiment.pupil_samples().is_empty()

    calibration, indexes = experiment.calib_data()
    assert {"avg_error", "subject_id", "session_id"} <= set(calibration.columns)
    assert {"Calib_index", "trial_number"} <= set(indexes.columns)
    experiment.plot_calib_data()


def test_visual_search_statistics_and_trial_accessors(generated_datasets):
    experiment = _visual_search_experiment(generated_datasets)
    subject = experiment["0001"]
    session = subject["second"]
    absent_trial = session[0]
    present_trial = session[1]

    for accuracy in (
        experiment.accuracy(),
        subject.accuracy(),
        session.accuracy(),
    ):
        assert accuracy.get_column("accuracy").to_list() == [1.0, 1.0]

    assert session.accuracy().get_column("session_id").unique().to_list() == ["second"]
    assert subject.accuracy().get_column("subject_id").unique().to_list() == ["0001"]
    assert experiment.search_rts().height == 2
    assert not experiment.search_fixations().is_empty()
    assert not experiment.search_saccades().is_empty()
    assert not subject.search_rts().is_empty()
    assert not subject.search_fixations().is_empty()
    assert not subject.search_saccades().is_empty()
    assert not session.search_rts().is_empty()
    assert not session.search_fixations().is_empty()
    assert not session.search_saccades().is_empty()
    assert experiment.scanpaths_by_stimuli().height == 2

    cutoffs = experiment.find_fixation_cutoff()
    assert cutoffs.height == 2
    assert min(cutoffs.get_column("fix_cutoff")) > 0
    assert subject.find_fixation_cutoff().height == 2
    assert session.find_fixation_cutoff().height == 2
    with pytest.raises(ValueError, match="percentile"):
        experiment.find_fixation_cutoff(0)

    cumulative = experiment.cumulative_correct_trials_by_fixation(cutoffs)
    assert cumulative.height == 2
    assert all(
        np.asarray(values).size > 0
        for values in cumulative.get_column("cumulative_correct").to_list()
    )

    binned = experiment.trials_by_rt_bins(bin_end=60, bin_step=1)
    assert "rt_bin" in binned.columns
    with pytest.raises(ValueError, match="bin_end"):
        experiment.trials_by_rt_bins(0, 1)
    with pytest.raises(ValueError, match="bin_step"):
        experiment.trials_by_rt_bins(10, 0)

    assert absent_trial.target_present is False
    assert absent_trial.target == "rubik_cube.jpg"
    assert absent_trial.target_location is None
    assert present_trial.target_present is True
    assert len(present_trial.target_location) == 4
    assert absent_trial.memory_set_size == 1
    assert absent_trial.memory_set == ["rubik_cube.jpg"]
    assert len(absent_trial.memory_set_locations) == 1
    assert len(absent_trial.stimulus_coords) == 4
    assert absent_trial.was_answered
    assert absent_trial.correct_response
    assert not absent_trial.fixations().is_empty()
    assert not absent_trial.saccades().is_empty()
    assert not absent_trial.search_fixations().is_empty()
    assert set(absent_trial.memorization_fixations().get_column("phase").unique()) <= {
        "memorization"
    }
    assert not absent_trial.search_saccades().is_empty()
    assert set(absent_trial.memorization_saccades().get_column("phase").unique()) <= {
        "memorization"
    }
    assert not absent_trial.search_samples().is_empty()
    assert set(absent_trial.memorization_samples().get_column("phase").unique()) <= {
        "memorization"
    }
    assert absent_trial.scanpath_by_stimuli()["stimulus"] == absent_trial.stimulus
    assert not absent_trial._multimatch_fixations().is_empty()
    absent_trial.save_rts()
    assert absent_trial.is_trial_longer_than(0, "search")
    assert not absent_trial.is_trial_longer_than(0, "missing")


def test_visual_search_plotting_api(generated_datasets, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    experiment = _visual_search_experiment(generated_datasets)

    experiment.plot_accuracy_by_subject()
    experiment.plot_accuracy_by_stimulus()
    experiment.plot_speed_accuracy_tradeoff_by_subject()
    experiment.plot_speed_accuracy_tradeoff_by_stimulus()
    experiment.plot_cumulative_performance()
    experiment.plot_correct_trials_by_rt_bins(60, 1)
    experiment.plot_incorrect_trials_by_rt_bins(60, 1)
    experiment.plot_probability_of_deciding_by_rt_bin(60, 1)


def test_visual_search_trial_plot_delegation(generated_datasets, monkeypatch):
    trial = _visual_search_experiment(generated_datasets)["0001"]["second"][0]
    calls = {}

    def fake_scanpath(self, **kwargs):
        calls["scanpath"] = kwargs

    def fake_animation(self, **kwargs):
        calls["animation"] = kwargs
        return "animation"

    monkeypatch.setattr(Visualization, "scanpath", fake_scanpath)
    monkeypatch.setattr(Visualization, "plot_animation", fake_animation)

    trial.plot_scanpath(1080, 1920, display=False)
    result = trial.plot_animation(1080, 1920, display=False)

    assert "bbox" not in calls["scanpath"]["phase_data"]["search"]
    assert calls["scanpath"]["display"] is False
    assert calls["animation"]["folder_path"] == trial.events_path
    assert result == "animation"


def test_visual_search_removal_methods(generated_datasets, monkeypatch):
    experiment = _visual_search_experiment(generated_datasets)
    session = experiment["0001"]["second"]
    monkeypatch.setattr(session[0], "is_trial_bad", lambda phase, threshold: True)
    monkeypatch.setattr(session[1], "is_trial_bad", lambda phase, threshold: False)

    assessment = session.assess_trial_quality("search")
    assert assessment.bad_trials == (0,)
    assert assessment.total_trials == 2
    assert assessment.bad_trial_fraction == 0.5
    assert list(session.trials) == [0, 1]

    removed = session.remove_bad_trials("search", print_flag=False)
    assert removed == 1
    assert list(session.trials) == [1]

    experiment = _visual_search_experiment(generated_datasets)
    session = experiment["0001"]["second"]
    monkeypatch.setattr(session[0], "is_trial_bad", lambda phase, threshold: True)
    monkeypatch.setattr(session[1], "is_trial_bad", lambda phase, threshold: False)

    result = experiment.remove_bad_trials_and_sessions(
        "search",
        trial_nan_threshold=0.1,
        session_bad_trial_threshold=0.5,
        print_flag=False,
    )
    assert list(experiment["0001"]["second"].trials) == [1]
    assert result.bad_trials_removed == 1
    assert result.sessions_removed == 0
    assert result.trials_discarded_with_sessions == 0

    experiment = _visual_search_experiment(generated_datasets)
    session = experiment["0001"]["second"]
    monkeypatch.setattr(session[0], "is_trial_bad", lambda phase, threshold: True)
    monkeypatch.setattr(session[1], "is_trial_bad", lambda phase, threshold: False)

    result = experiment.remove_bad_trials_and_sessions(
        "search",
        trial_nan_threshold=0.1,
        session_bad_trial_threshold=0.4,
        print_flag=False,
    )
    assert "0001" not in experiment.subjects
    assert result.bad_trials_removed == 0
    assert result.sessions_removed == 1
    assert result.subjects_removed == 1
    assert result.trials_discarded_with_sessions == 2

    experiment = _visual_search_experiment(generated_datasets)
    experiment.remove_non_answered_trials(print_flag=False)
    assert len(experiment["0001"]["second"]) == 2
    experiment.remove_trials_for_stimuli(["0167.jpg"], print_flag=False)
    assert len(experiment["0001"]["second"]) == 1

    experiment = _visual_search_experiment(generated_datasets)
    experiment.remove_trials_for_stimuli_with_poor_accuracy(
        threshold=1.1, print_flag=False
    )
    assert len(experiment) == 0

    experiment = _visual_search_experiment(generated_datasets)
    experiment.remove_poor_accuracy_sessions(threshold=1.1, print_flag=False)
    assert len(experiment) == 0


def test_bad_trial_and_session_threshold_validation(generated_datasets):
    experiment = _visual_search_experiment(generated_datasets)
    session = experiment["0001"]["second"]

    with pytest.raises(ValueError, match="trial_nan_threshold"):
        session.assess_trial_quality("search", -0.1)
    with pytest.raises(ValueError, match="session_bad_trial_threshold"):
        experiment.remove_bad_trials_and_sessions(
            "search",
            session_bad_trial_threshold=1.1,
            print_flag=False,
        )


def test_trial_bad_sample_fraction_counts_rows(generated_datasets):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    trial = experiment["0001"]["second"][0]
    trial._samples = pl.DataFrame(
        {
            "tSample": [0.0, 1.0, 2.0],
            "phase": ["search", "search", "search"],
            "X": [1.0, None, 3.0],
            "Y": [1.0, None, 3.0],
            "bad": [False, False, True],
            "unrelated_metadata": [None, None, None],
        }
    )
    trial._blink = pl.DataFrame(
        schema={"tStart": pl.Float64, "tEnd": pl.Float64, "duration": pl.Float64}
    )

    assert trial.is_trial_bad("search", threshold=0.5)
    assert not trial.is_trial_bad("search", threshold=0.7)
    assert trial.is_trial_bad("missing", threshold=0.7)


def test_fixation_cutoff_is_a_count_not_an_index():
    assert _find_fixation_cutoff([3, 3], threshold=6, max_possible=3) == 3
    assert _find_fixation_cutoff([1, 3], threshold=3, max_possible=3) == 2
    assert _find_fixation_cutoff([], threshold=0, max_possible=0) == 0
    assert _find_fixation_cutoff([1], threshold=10, max_possible=3) == 3


def test_visual_search_empty_helpers_and_cutoff_validation():
    assert visual_search_module._as("[1, 2]", list) == [1, 2]
    assert visual_search_module._fixation_cutoffs([], 1).is_empty()

    trial = type(
        "TrialStub",
        (),
        {
            "memory_set_size": 1,
            "target_present": True,
            "correct_response": True,
            "search_fixations": lambda self: pl.DataFrame({"x": [1]}),
        },
    )()
    with pytest.raises(ValueError, match="exactly one row"):
        visual_search_module._cumulative_correct_by_fixation(
            [trial],
            pl.DataFrame(
                schema={
                    "memory_set_size": pl.Int64,
                    "target_present": pl.Boolean,
                    "fix_cutoff": pl.Int64,
                }
            ),
        )

    fig, ax = plt.subplots()
    visual_search_module._plot_grouped_mean_with_se(
        ax,
        pl.DataFrame(),
        x="x",
        y="y",
        group="group",
    )
    visual_search_module._plot_cumulative_mean_with_se(
        ax, pl.DataFrame(), max_fixations=0
    )
    visual_search_module._plot_rt_bin_bars(
        ax,
        pl.DataFrame(),
        value_column="value",
        ylabel="Value",
    )
    plt.close(fig)
    visual_search_module._plot_speed_accuracy_tradeoff(
        pl.DataFrame(),
        entity_column="subject",
        title="Empty",
    )


def test_quality_result_and_multimatch_validation(monkeypatch):
    assert SessionQualityAssessment((), 0).bad_trial_fraction == 0
    with pytest.raises(TypeError):
        QualityFilterResult() + 1
    with pytest.raises(ValueError, match="missing fixation columns"):
        _to_multimatch_scanpath(pl.DataFrame({"xAvg": [1.0]}))

    def missing_multimatch(name):
        raise ModuleNotFoundError("No module named 'multimatch_gaze'")

    monkeypatch.setattr(generic_module, "import_module", missing_multimatch)
    with pytest.raises(ImportError, match=r"pyxations\[multimatch\]"):
        generic_module._load_multimatch()


def test_generic_cleanup_and_plot_delegation(generated_datasets, monkeypatch, capsys):
    case = generated_datasets["eyelink"]

    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    experiment.filter_fixations(min_fix_dur=0)
    assert "shorter than 0 ms" in capsys.readouterr().out

    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    experiment.collapse_fixations(threshold_px=0)
    assert "fixations that were merged" in capsys.readouterr().out

    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    calls = []
    for trial in experiment["0001"]["second"].trials.values():
        monkeypatch.setattr(
            trial,
            "plot_scanpath",
            lambda *args, trial_number=trial.trial_number, **kwargs: calls.append(
                trial_number
            ),
        )
    experiment.plot_scanpaths(1080, 1920)
    assert calls == [0, 1]

    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    experiment.drop_trials_longer_than(10_000, "search")
    assert "longer than 10000 seconds" in capsys.readouterr().out

    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    experiment.drop_poor_or_non_calibrated_trials(100)
    assert "poor calibration" in capsys.readouterr().out


def test_generic_quality_prints_and_trial_edge_paths(
    generated_datasets, monkeypatch, capsys
):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    session = experiment["0001"]["second"]
    monkeypatch.setattr(session[0], "is_trial_bad", lambda phase, threshold: False)
    monkeypatch.setattr(session[1], "is_trial_bad", lambda phase, threshold: False)

    result = experiment.remove_bad_trials_and_sessions("search")
    assert result == QualityFilterResult()
    assert "0 bad trials" in capsys.readouterr().out
    session.remove_bad_trials("search")
    assert "Removed 0 bad trials" in capsys.readouterr().out

    trial = session[0]
    trial._blink = None
    assert trial.blinks().is_empty()
    unchanged = trial.fixations().clone()
    assert trial.filter_fixations(min_fix_dur=0) is None
    assert trial.fixations().equals(unchanged)

    trial._samples = pl.DataFrame(
        {
            "tSample": [0.0, 1.0, 2.0],
            "phase": ["search"] * 3,
            "X": [1.0, None, 2.0],
            "Y": [1.0, None, 2.0],
        }
    )
    trial._blink = pl.DataFrame({"tStart": [0.5], "tEnd": [1.5], "duration": [1.0]})
    assert not trial.is_trial_bad("search", threshold=0)


def test_generic_trial_animation_delegation(generated_datasets, monkeypatch):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    trial = experiment["0001"]["second"][0]
    captured = {}

    def fake_animation(self, **kwargs):
        captured.update(kwargs)
        return "animation"

    monkeypatch.setattr(Visualization, "plot_animation", fake_animation)
    assert trial.plot_animation(1080, 1920, display=False) == "animation"
    assert captured["folder_path"] == trial.events_path


def test_trial_filter_and_collapse_fixations(generated_datasets):
    case = generated_datasets["eyelink"]
    experiment = Experiment(case["raw"])
    experiment.load_data(case["algorithm"])
    trial = experiment["0001"]["second"][0]

    trial._fix = pl.DataFrame(
        {
            "phase": ["search", "search", "search"],
            "eye": ["L", "L", "L"],
            "tStart": [0.0, 20.0, 40.0],
            "tEnd": [10.0, 25.0, 50.0],
            "duration": [10.0, 5.0, 10.0],
            "xAvg": [0.0, 1.0, 100.0],
            "yAvg": [0.0, 1.0, 100.0],
            "pupilAvg": [2.0, 2.0, 2.0],
        }
    )
    trial._sacc = pl.DataFrame(
        {
            "phase": ["search", "search"],
            "eye": ["L", "L"],
            "tStart": [10.0, 25.0],
            "tEnd": [20.0, 40.0],
            "duration": [10.0, 15.0],
            "xStart": [0.0, 1.0],
            "yStart": [0.0, 1.0],
            "xEnd": [1.0, 100.0],
            "yEnd": [1.0, 100.0],
            "dx": [1.0, 99.0],
            "dy": [1.0, 99.0],
            "amplitude": [np.sqrt(2), np.sqrt(99**2 + 99**2)],
            "ampDeg": [1.0, 2.0],
            "vPeak": [10.0, 20.0],
        }
    )

    trial.filter_fixations(min_fix_dur=6)
    assert trial.fixations().height == 2
    assert trial.saccades().height == 1
    assert trial.saccades().get_column("tEnd")[0] == 40

    trial._fix = pl.DataFrame(
        {
            "phase": ["search", "search", "search"],
            "eye": ["L", "L", "L"],
            "tStart": [0.0, 20.0, 40.0],
            "tEnd": [10.0, 30.0, 50.0],
            "duration": [10.0, 10.0, 10.0],
            "xAvg": [0.0, 1.0, 100.0],
            "yAvg": [0.0, 1.0, 100.0],
            "pupilAvg": [2.0, 4.0, 6.0],
        }
    )
    trial._sacc = pl.DataFrame(
        {
            "phase": ["search", "search"],
            "eye": ["L", "L"],
            "tStart": [10.0, 30.0],
            "tEnd": [20.0, 40.0],
            "xStart": [0.0, 1.0],
            "yStart": [0.0, 1.0],
            "xEnd": [1.0, 100.0],
            "yEnd": [1.0, 100.0],
            "dx": [1.0, 99.0],
            "dy": [1.0, 99.0],
            "amplitude": [np.sqrt(2), np.sqrt(99**2 + 99**2)],
        }
    )

    trial.collapse_fixations(threshold_px=5)
    assert trial.fixations().height == 2
    assert trial.fixations().get_column("duration")[0] == 20
    assert trial.saccades().height == 1
