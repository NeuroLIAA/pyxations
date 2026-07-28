from types import SimpleNamespace

import polars as pl
import pytest

import pyxations.bids_formatting as formatting


class _SegmentationMethods:
    def split_all_into_trials(
        self, start_times, end_times, trial_labels=None, require_nonoverlap=True
    ):
        pass

    def split_all_into_trials_by_durations(
        self, start_msgs, durations, trial_labels=None
    ):
        pass

    def split_all_into_trials_by_msgs(
        self, start_msgs, end_msgs, use_regex=False, return_match_token=False
    ):
        pass


def test_segmentation_recipe_selects_supported_arguments():
    preprocessing = _SegmentationMethods()

    explicit = formatting._segmentation_recipe(
        preprocessing,
        {
            "start_times": {"search": [0]},
            "end_times": {"search": [1]},
            "trial_labels": ["a"],
            "ignored": True,
        },
    )
    durations = formatting._segmentation_recipe(
        preprocessing,
        {
            "start_msgs": {"search": ["start"]},
            "durations": {"search": 100},
            "end_msgs": {"search": ["end"]},
            "prefer_durations": True,
        },
    )
    messages = formatting._segmentation_recipe(
        preprocessing,
        {
            "start_msgs": {"search": ["start"]},
            "end_msgs": {"search": ["end"]},
            "use_regex": True,
        },
    )

    assert explicit == (
        "split_all_into_trials",
        {
            "start_times": {"search": [0]},
            "end_times": {"search": [1]},
            "trial_labels": ["a"],
        },
    )
    assert durations[0] == "split_all_into_trials_by_durations"
    assert messages == (
        "split_all_into_trials_by_msgs",
        {
            "start_msgs": {"search": ["start"]},
            "end_msgs": {"search": ["end"]},
            "use_regex": True,
        },
    )
    assert formatting._segmentation_recipe(preprocessing, {}) is None


def test_best_eye_selection_and_table_projection():
    calibration = pl.DataFrame(
        {
            "line": [
                "MSG 1 !CAL VALIDATION HV9 LEFT GOOD ERROR 0.50 avg.",
                "MSG 2 !CAL VALIDATION HV9 RIGHT GOOD ERROR 0.25 avg.",
            ]
        }
    )
    samples = pl.DataFrame(
        {
            "tSample": [0.0, 1.0],
            "LX": [1.0, 2.0],
            "LY": [3.0, 4.0],
            "LPupil": [5.0, 6.0],
            "RX": [7.0, 8.0],
            "RY": [9.0, 10.0],
            "RPupil": [11.0, 12.0],
            "Calib_index": [0, 0],
        }
    )
    events = pl.DataFrame(
        {
            "eye": ["L", "R"],
            "tStart": [0.0, 0.0],
            "tEnd": [1.0, 1.0],
        }
    )

    assert formatting._find_best_eye(calibration) == "R"
    projected = formatting._keep_eye("R", samples, events, events, events)
    assert {"X", "Y", "Pupil", "eye"} <= set(projected[0].columns)
    assert projected[0].get_column("eye").unique().to_list() == ["R"]
    assert all(table.height == 1 for table in projected[1:])

    assert formatting._find_best_eye(pl.DataFrame({"other": [1]})) == "M"
    assert (
        formatting._find_best_eye(
            pl.DataFrame(
                {
                    "line": [
                        "MSG 1 !CAL VALIDATION HV9 L ABORTED",
                        "MSG 2 !CAL VALIDATION HV9 RIGHT GOOD ERROR 0.4 avg.",
                    ]
                }
            )
        )
        == "R"
    )


def test_sample_detector_uses_recorded_rate_and_user_configuration(
    tmp_path, monkeypatch
):
    calls = {}

    class FakeDetector:
        def __init__(self, session_folder_path, samples):
            calls["samples"] = samples

        def run_eye_movement(
            self, gazex_data, gazey_data, sample_rate, savgol_length=0.19
        ):
            raise AssertionError("Only the adapter should invoke this method")

        def run_eye_movement_from_samples(self, sample_rate, config):
            calls["sample_rate"] = sample_rate
            calls["config"] = config
            empty = pl.DataFrame(
                schema={
                    "tStart": pl.Float64,
                    "tEnd": pl.Float64,
                    "duration": pl.Float64,
                }
            )
            return empty, empty

    monkeypatch.setattr(formatting, "_detector_type", lambda name: FakeDetector)
    raw = SimpleNamespace(
        samples=pl.DataFrame(
            {
                "tSample": [0.0, 5.0],
                "LX": [1.0, 2.0],
                "LY": [3.0, 4.0],
                "LPupil": [5.0, 6.0],
            }
        ),
        fixations=pl.DataFrame(),
        saccades=pl.DataFrame(),
        blinks=pl.DataFrame(
            schema={
                "tStart": pl.Float64,
                "tEnd": pl.Float64,
                "duration": pl.Float64,
            }
        ),
        sampling_frequency=120.0,
    )

    samples, _, _, _ = formatting._detect_from_bids(
        raw,
        dataset_format="tobii",
        detection_algorithm="remodnav",
        session_folder_path=tmp_path,
        kwargs={"sample_rate": 120.0, "savgol_length": 0.1, "ignored": 123},
    )

    assert {"X", "Y", "Pupil", "eye"} <= set(samples.columns)
    assert calls["sample_rate"] == 120.0
    assert calls["config"]["savgol_length"] == 0.1
    assert "ignored" not in calls["config"]


def test_detector_and_process_session_validation(tmp_path, monkeypatch):
    raw = SimpleNamespace(
        samples=pl.DataFrame(),
        fixations=pl.DataFrame(),
        saccades=pl.DataFrame(),
        blinks=pl.DataFrame(),
        sampling_frequency=60.0,
    )
    with pytest.raises(ValueError, match="requires tracker-reported"):
        formatting._detect_from_bids(
            raw,
            dataset_format="tobii",
            detection_algorithm="eyelink",
            session_folder_path=tmp_path,
            kwargs={},
        )
    with pytest.raises(ValueError, match="Unknown eye-movement detector"):
        formatting._detect_from_bids(
            raw,
            dataset_format="tobii",
            detection_algorithm="unknown",
            session_folder_path=tmp_path,
            kwargs={},
        )

    destination = tmp_path / "derivatives" / "sub-0001" / "ses-one"
    behavior = destination / "beh"
    behavior.mkdir(parents=True)
    (behavior / "sub-0001_recording-eye1engbert_physio.tsv.gz").touch()
    monkeypatch.setattr(
        formatting,
        "process_bids_session",
        lambda *args, **kwargs: pytest.fail("existing output should be skipped"),
    )
    formatting.process_session(
        tmp_path,
        "tobii",
        "engbert",
        destination,
        False,
        False,
    )

    with pytest.raises(ValueError, match="Dataset format"):
        formatting.process_session(
            tmp_path,
            "unsupported",
            "engbert",
            destination,
            False,
            True,
        )


def test_parallel_derivative_orchestration_forwards_session_options(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "raw"
    session = dataset / "sub-0001" / "ses-one"
    session.mkdir(parents=True)
    (dataset / "participants.tsv").write_text(
        "subject_id\told_subject_id\n0001\toriginal\n",
        encoding="utf-8",
    )
    calls = []

    class Finished:
        def result(self):
            return None

    class ImmediateExecutor:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args, **kwargs):
            calls.append((function, args, kwargs))
            return Finished()

    monkeypatch.setattr(formatting, "ProcessPoolExecutor", ImmediateExecutor)
    derivatives = formatting.compute_derivatives_for_dataset(
        dataset,
        "tobii",
        "engbert",
        num_processes=2,
        start_times={"original": {"one": {"search": [0]}}},
        end_times={"original": {"one": {"search": [10]}}},
        behavioral_columns=["condition"],
    )

    assert derivatives == tmp_path / "raw_derivatives"
    assert len(calls) == 1
    _, args, options = calls[0]
    assert args[0] == session
    assert options["start_times"] == {"search": [0]}
    assert options["end_times"] == {"search": [10]}
    assert options["behavioral_columns"] == ["condition"]


@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError)],
)
def test_derivative_worker_count_validation(tmp_path, value, error):
    with pytest.raises(error):
        formatting.compute_derivatives_for_dataset(
            tmp_path,
            "tobii",
            num_processes=value,
        )
