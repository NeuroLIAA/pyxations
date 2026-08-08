import warnings
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


def test_optional_detector_loading_and_errors(monkeypatch):
    assert formatting._detector_type("engbert") is formatting.EngbertDetection
    assert formatting._detector_type("missing") is None

    def missing_dependency(name):
        error = ModuleNotFoundError("No module named 'remodnav'")
        error.name = "remodnav"
        raise error

    monkeypatch.setattr(formatting, "import_module", missing_dependency)
    with pytest.raises(ImportError, match=r"pyxations\[remodnav\]"):
        formatting._detector_type("remodnav")

    def unrelated_dependency(name):
        error = ModuleNotFoundError("No module named 'unrelated'")
        error.name = "unrelated"
        raise error

    monkeypatch.setattr(formatting, "import_module", unrelated_dependency)
    with pytest.raises(ModuleNotFoundError, match="unrelated"):
        formatting._detector_type("remodnav")


def test_detector_sampling_and_engbert_forwarding(tmp_path, monkeypatch):
    calls = {}

    class FakeRemodnav:
        def __init__(self, **kwargs):
            pass

        def run_eye_movement(self, gazex_data, gazey_data, sample_rate):
            pass

        def run_eye_movement_from_samples(self, sample_rate, config):
            raise AssertionError("missing frequency must fail before detection")

    monkeypatch.setattr(formatting, "_detector_type", lambda name: FakeRemodnav)
    raw = SimpleNamespace(
        samples=pl.DataFrame({"tSample": [0.0], "X": [1.0], "Y": [2.0]}),
        fixations=pl.DataFrame(),
        saccades=pl.DataFrame(),
        blinks=pl.DataFrame(),
        sampling_frequency=None,
    )
    with pytest.raises(ValueError, match="Sampling frequency"):
        formatting._detect_from_bids(
            raw,
            dataset_format="eyelink",
            detection_algorithm="remodnav",
            session_folder_path=tmp_path,
            kwargs={},
        )

    class FakeEngbert:
        def __init__(self, session_folder_path, samples):
            calls["samples"] = samples

        def detect_eye_movements(self, vfac=5):
            calls["vfac"] = vfac
            return pl.DataFrame({"event": [1]}), pl.DataFrame({"event": [2]})

    monkeypatch.setattr(formatting, "_detector_type", lambda name: FakeEngbert)
    raw.samples = pl.DataFrame(
        {
            "tSample": [0.0],
            "RX": [1.0],
            "RY": [2.0],
            "RPupil": [3.0],
        }
    )
    samples, fixations, saccades, _ = formatting._detect_from_bids(
        raw,
        dataset_format="tobii",
        detection_algorithm="engbert",
        session_folder_path=tmp_path,
        kwargs={"vfac": 7, "ignored": True},
    )
    assert {"X", "Y", "Pupil", "eye"} <= set(samples.columns)
    assert samples["eye"].to_list() == ["R"]
    assert calls["vfac"] == 7
    assert fixations["event"].to_list() == [1]
    assert saccades["event"].to_list() == [2]


def test_best_eye_fallbacks_and_calibration_groups():
    tables = (
        pl.DataFrame({"Calib_index": [1], "LX": [1.0], "LY": [2.0]}),
        pl.DataFrame({"Calib_index": [1]}),
        pl.DataFrame({"Calib_index": [1]}),
        pl.DataFrame({"Calib_index": [1]}),
    )
    raw = SimpleNamespace(calibration=pl.DataFrame())
    assert formatting._choose_best_eye(raw, *tables) == tables

    raw.calibration = pl.DataFrame({"Calib_index": [1], "line": ["other"]})
    assert formatting._choose_best_eye(raw, *tables) == tables
    assert formatting._find_best_eye(
        pl.DataFrame(
            {
                "line": [
                    "CAL VALIDATION L ABORTED",
                    "CAL VALIDATION R ABORTED",
                ]
            }
        )
    ) == "M"
    assert formatting._find_best_eye(
        pl.DataFrame({"line": ["CAL VALIDATION LEFT GOOD"]})
    ) == "L"

    raw.calibration = pl.DataFrame(
        {
            "Calib_index": [1, 2],
            "line": [
                "CAL VALIDATION LEFT GOOD ERROR 0.1",
                "CAL VALIDATION L ABORTED",
            ],
        }
    )
    selected = formatting._choose_best_eye(raw, *tables)
    assert selected[0].height == 1
    assert selected[0]["X"].to_list() == [1.0]


def test_default_trial_assignment_preserves_existing_trials():
    preprocessing = SimpleNamespace(
        samples=pl.DataFrame(
            {
                "tSample": [0.0, 10.0, 20.0],
                "trial_number": [None, 2, 2],
                "phase": ["", "search", "search"],
            }
        ),
        fixations=pl.DataFrame({"tStart": [10.0], "tEnd": [20.0]}),
        saccades=pl.DataFrame(),
        blinks=pl.DataFrame(),
    )
    formatting._assign_default_trials(preprocessing)

    assert preprocessing.samples["trial_number"].to_list() == [0, 2, 2]
    assert preprocessing.fixations["trial_number"].to_list() == [2]
    assert {"trial_number", "phase"} <= set(preprocessing.saccades.columns)


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [
        (5.9, True),  # a webcam recording
        (49.9, True),
        (50.0, False),  # the threshold itself is acceptable
        (298.4, False),  # a Tobii recording
        (None, False),  # unknown rate: nothing to judge
        (0.0, False),  # invalid rate: reported elsewhere
    ],
)
def test_low_sampling_rate_warns_before_event_detection(frequency, expected):
    """Detected events mean little when a saccade spans barely one sample."""
    if expected:
        with pytest.warns(UserWarning, match="velocity-based detection"):
            formatting._warn_if_rate_is_too_low_to_detect(frequency, "remodnav")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        formatting._warn_if_rate_is_too_low_to_detect(frequency, "remodnav")


def test_low_rate_warning_names_the_detector_and_the_measured_rate():
    with pytest.warns(UserWarning) as records:
        formatting._warn_if_rate_is_too_low_to_detect(5.92, "engbert")

    message = str(records[0].message)
    assert "5.9 Hz" in message
    assert "engbert" in message
