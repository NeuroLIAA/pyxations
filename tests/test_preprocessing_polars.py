from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyxations.pre_processing import PreProcessing, SessionMetadata


def _messages() -> pl.DataFrame:
    return pl.DataFrame({"timestamp": [0], "message": ["start"]})


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema={"tStart": pl.Float64, "tEnd": pl.Float64})


def _values(frame: pl.DataFrame, column: str):
    return frame.get_column(column).to_list()


def _preprocessing(
    tmp_path: Path,
    *,
    samples: pl.DataFrame | None = None,
    fixations: pl.DataFrame | None = None,
    saccades: pl.DataFrame | None = None,
    blinks: pl.DataFrame | None = None,
    messages: pl.DataFrame | None = None,
    metadata: SessionMetadata | None = None,
) -> PreProcessing:
    return PreProcessing(
        samples=samples
        if samples is not None
        else pl.DataFrame(schema={"tSample": pl.Float64}),
        fixations=fixations if fixations is not None else _empty_events(),
        saccades=saccades if saccades is not None else _empty_events(),
        blinks=blinks if blinks is not None else _empty_events(),
        user_messages=messages if messages is not None else _messages(),
        session_path=tmp_path,
        metadata=metadata,
    )


def test_constructor_clones_polars_inputs_and_rejects_other_tables(tmp_path: Path):
    samples = pl.DataFrame({"tSample": [0], "X": [1.0]})
    pp = _preprocessing(tmp_path, samples=samples)

    assert isinstance(pp.samples, pl.DataFrame)
    assert pp.samples.equals(samples)
    assert pp.samples is not samples

    with pytest.raises(TypeError, match="must be a Polars DataFrame"):
        PreProcessing(
            samples={"tSample": [0]},  # type: ignore[arg-type]
            fixations=_empty_events(),
            saccades=_empty_events(),
            blinks=_empty_events(),
            user_messages=_messages(),
            session_path=tmp_path,
        )


def test_explicit_trial_segmentation_and_event_containment(tmp_path: Path):
    pp = _preprocessing(
        tmp_path,
        samples=pl.DataFrame(
            {
                "tSample": [0, 10, 20, 30, 40, 50],
                "X": [1, 2, 3, 4, 5, 6],
                "Y": [1, 2, 3, 4, 5, 6],
            }
        ),
        fixations=pl.DataFrame(
            {
                "tStart": [5, 15, 35],
                "tEnd": [15, 25, 45],
                "xAvg": [1, 2, 3],
                "yAvg": [1, 2, 3],
            }
        ),
        saccades=pl.DataFrame(
            {
                "tStart": [0, 30],
                "tEnd": [20, 50],
                "xStart": [1, 2],
                "yStart": [1, 2],
                "xEnd": [2, 3],
                "yEnd": [2, 3],
            }
        ),
        blinks=pl.DataFrame({"tStart": [19, 35], "tEnd": [21, 40]}),
    )

    pp.split_all_into_trials(
        {"search": [0, 30]},
        {"search": [20, 50]},
        {"search": ["first", "second"]},
    )

    assert _values(pp.samples, "phase") == ["search"] * 6
    assert _values(pp.samples, "trial_number") == [0, 0, 0, 1, 1, 1]
    assert _values(pp.samples, "trial_label") == [
        "first",
        "first",
        "first",
        "second",
        "second",
        "second",
    ]
    assert _values(pp.fixations, "trial_number") == [0, -1, 1]
    assert _values(pp.blinks, "trial_number") == [-1, 1]
    assert _values(pp.saccades, "trial_label") == ["first", "second"]


def test_open_final_trial_is_discarded(tmp_path: Path):
    pp = _preprocessing(
        tmp_path,
        samples=pl.DataFrame(
            {"tSample": [0, 10, 30, 40, 60], "X": [1] * 5, "Y": [1] * 5}
        ),
    )

    pp.split_all_into_trials(
        {"phase": [0, 30, 60]},
        {"phase": [20, 50]},
        {"phase": ["one", "two"]},
        allow_open_last=True,
    )

    assert _values(pp.samples, "trial_number") == [0, 0, 1, 1, -1]


def test_trial_segmentation_validates_phase_definitions(tmp_path: Path):
    pp = _preprocessing(
        tmp_path,
        samples=pl.DataFrame({"tSample": [0, 10, 20]}),
    )

    with pytest.raises(ValueError, match="Missing end-time definitions"):
        pp.split_all_into_trials({"search": [0]}, {})

    with pytest.raises(ValueError, match="Overlapping trials"):
        pp.split_all_into_trials(
            {"search": [0, 10]},
            {"search": [15, 20]},
        )

    with pytest.raises(ValueError, match="Non-positive interval"):
        pp.split_all_into_trials({"search": [10]}, {"search": [10]})

    with pytest.raises(ValueError, match="trial labels"):
        pp.split_all_into_trials(
            {"search": [0]},
            {"search": [10]},
            {"search": ["one", "extra"]},
        )


def test_bad_samples_handles_binocular_bounds_and_missing_values(tmp_path: Path):
    pp = _preprocessing(
        tmp_path,
        samples=pl.DataFrame(
            {
                "tSample": [0, 1, 2, 3, 4],
                "LX": [0.0, -1.0, 10.0, np.nan, 100.0],
                "LY": [0.0, 10.0, 10.0, 10.0, 100.0],
                "RX": [100.0, 10.0, 10.0, 10.0, 100.0],
                "RY": [100.0, 10.0, 101.0, 10.0, 100.0],
            }
        ),
        fixations=pl.DataFrame(
            {
                "tStart": [0, 10],
                "tEnd": [5, 15],
                "xAvg": [50.0, 101.0],
                "yAvg": [50.0, 50.0],
            }
        ),
        saccades=pl.DataFrame(
            {
                "tStart": [0, 10],
                "tEnd": [5, 15],
                "xStart": [0.0, 10.0],
                "yStart": [0.0, 10.0],
                "xEnd": [100.0, 10.0],
                "yEnd": [100.0, -1.0],
            }
        ),
        metadata=SessionMetadata(screen_width=100, screen_height=100),
    )

    pp.bad_samples()
    assert _values(pp.samples, "bad") == [False, True, True, True, False]
    assert _values(pp.fixations, "bad") == [False, True]
    assert _values(pp.saccades, "bad") == [False, True]
    assert "bad" not in pp.blinks.columns

    pp.bad_samples(mark_nan_as_bad=False)
    assert _values(pp.samples, "bad") == [False, True, True, False, False]

    pp.bad_samples(mark_nan_as_bad=False, inclusive_bounds=False)
    assert _values(pp.samples, "bad") == [True, True, True, False, True]


def test_bad_samples_without_coordinates_adds_false_flag(tmp_path: Path):
    pp = _preprocessing(
        tmp_path,
        samples=pl.DataFrame({"tSample": [0, 1]}),
        fixations=pl.DataFrame({"tStart": [0], "tEnd": [1]}),
        saccades=pl.DataFrame({"tStart": [0], "tEnd": [1]}),
    )

    pp.bad_samples(screen_width=100, screen_height=100)

    assert _values(pp.samples, "bad") == [False, False]
    assert _values(pp.fixations, "bad") == [False]
    assert _values(pp.saccades, "bad") == [False]


def test_bad_samples_validates_screen_dimensions(tmp_path: Path):
    pp = _preprocessing(tmp_path)

    with pytest.raises(ValueError, match="requires screen_height"):
        pp.bad_samples()
    with pytest.raises(ValueError, match="must be positive"):
        pp.bad_samples(screen_width=100, screen_height=0)


def _metadata_preprocessing(tmp_path: Path) -> PreProcessing:
    return _preprocessing(
        tmp_path,
        samples=pl.DataFrame(
            {
                "tSample": [30, 10, 20, 40],
                "trial_index": [2, 1, 1, 3],
                "condition": ["old", "old", "old", "old"],
            }
        ),
    )


def test_add_trial_metadata_preserves_order_and_replaces_columns(tmp_path: Path):
    pp = _metadata_preprocessing(tmp_path)
    metadata = pl.DataFrame(
        {
            "trial_index": [1, 2],
            "condition": ["target", "distractor"],
            "difficulty": ["easy", "hard"],
        }
    )

    pp.add_trial_metadata(metadata, ["condition", "difficulty"])

    assert _values(pp.samples, "tSample") == [30, 10, 20, 40]
    assert _values(pp.samples, "condition") == [
        "distractor",
        "target",
        "target",
        None,
    ]
    assert _values(pp.samples, "difficulty") == ["hard", "easy", "easy", None]


def test_add_trial_metadata_handles_duplicates_and_missing_columns(tmp_path: Path):
    pp = _metadata_preprocessing(tmp_path)
    metadata = pl.DataFrame(
        {
            "trial_index": [1, 1, 2],
            "condition": ["target", "target", "distractor"],
        }
    )

    with pytest.warns(RuntimeWarning, match="missing_column"):
        pp.add_trial_metadata(metadata, ["condition", "missing_column"])

    assert _values(pp.samples, "condition")[:3] == [
        "distractor",
        "target",
        "target",
    ]
    assert "missing_column" not in pp.samples.columns


def test_add_trial_metadata_rejects_conflicting_duplicates(tmp_path: Path):
    pp = _metadata_preprocessing(tmp_path)
    metadata = pl.DataFrame(
        {
            "trial_index": [1, 1],
            "condition": ["target", "distractor"],
        }
    )

    with pytest.raises(ValueError, match="Conflicting metadata"):
        pp.add_trial_metadata(metadata, ["condition"])


def test_add_trial_metadata_validates_tables_and_join_key(tmp_path: Path):
    pp = _metadata_preprocessing(tmp_path)

    with pytest.raises(TypeError, match="metadata_df must be a Polars DataFrame"):
        pp.add_trial_metadata({"trial_index": [1]}, ["condition"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="metadata_df must contain"):
        pp.add_trial_metadata(pl.DataFrame({"condition": ["target"]}), ["condition"])

    pp.samples = pl.DataFrame({"tSample": [0]})
    with pytest.raises(ValueError, match="samples must contain"):
        pp.add_trial_metadata(
            pl.DataFrame({"trial_index": [1], "condition": ["target"]}),
            ["condition"],
        )


def _message_processing(tmp_path: Path) -> PreProcessing:
    return _preprocessing(
        tmp_path,
        samples=pl.DataFrame(
            {
                "tSample": [100, 125, 150, 200, 300, 325, 350],
                "X": [10.0] * 7,
                "Y": [20.0] * 7,
            }
        ),
        fixations=pl.DataFrame(
            {
                "tStart": [110, 190, 310],
                "tEnd": [140, 210, 340],
                "xAvg": [10.0, 10.0, 10.0],
                "yAvg": [20.0, 20.0, 20.0],
            }
        ),
        saccades=pl.DataFrame(
            {
                "tStart": [100, 300],
                "tEnd": [150, 350],
                "xStart": [10.0, 10.0],
                "yStart": [20.0, 20.0],
                "xEnd": [11.0, 11.0],
                "yEnd": [21.0, 21.0],
            }
        ),
        blinks=pl.DataFrame({"tStart": [120, 320], "tEnd": [130, 330]}),
        messages=pl.DataFrame(
            {
                "timestamp": [300, 100, 350, 150, 500, 550, 700],
                "message": [
                    "TRIAL_START 2",
                    "trial_start 1",
                    "TRIAL_END 2",
                    "trial_end 1",
                    "literal[3]",
                    "LITERAL_END[3]",
                    None,
                ],
            }
        ),
    )


def test_message_matching_regex_literal_sorting_and_traceability(tmp_path: Path):
    pp = _message_processing(tmp_path)

    starts = pp.get_timestamps_from_messages(
        {"search": [r"TRIAL_START\s+\d+"]},
        return_match_token=True,
    )
    ends = pp.get_timestamps_from_messages(
        {"search": [r"TRIAL_END\s+\d+"]},
        return_match_token=True,
    )
    literal = pp.get_timestamps_from_messages(
        {"literal": ["literal[3]"]},
        use_regex=False,
    )

    assert starts == {"search": [100, 300]}
    assert ends == {"search": [150, 350]}
    assert literal == {"literal": [500]}
    assert _values(pp.user_messages, "matched_token") == [
        r"TRIAL_START\s+\d+",
        r"TRIAL_START\s+\d+",
        r"TRIAL_END\s+\d+",
        r"TRIAL_END\s+\d+",
        None,
        None,
        None,
    ]


def test_message_based_trial_segmentation(tmp_path: Path):
    pp = _message_processing(tmp_path)

    pp.split_all_into_trials_by_msgs(
        start_msgs={"search": [r"TRIAL_START\s+\d+"]},
        end_msgs={"search": [r"TRIAL_END\s+\d+"]},
        trial_labels={"search": ["one", "two"]},
        return_match_token=True,
    )

    assert _values(pp.samples, "trial_number") == [0, 0, 0, -1, 1, 1, 1]
    assert _values(pp.fixations, "trial_number") == [0, -1, 1]
    assert _values(pp.saccades, "trial_number") == [0, 1]
    assert _values(pp.blinks, "trial_number") == [0, 1]


def test_duration_based_trial_segmentation(tmp_path: Path):
    pp = _message_processing(tmp_path)

    pp.split_all_into_trials_by_durations(
        start_msgs={"search": [r"TRIAL_START\s+\d+"]},
        durations={"search": [50, 50]},
        trial_labels={"search": ["one", "two"]},
    )

    assert _values(pp.samples, "trial_number") == [0, 0, 0, -1, 1, 1, 1]
    assert _values(pp.fixations, "trial_number") == [0, -1, 1]


def test_message_matching_reports_invalid_inputs(tmp_path: Path):
    pp = _message_processing(tmp_path)

    with pytest.raises(ValueError, match="Invalid message pattern"):
        pp.get_timestamps_from_messages({"search": ["["]})
    with pytest.raises(ValueError, match="No timestamps found"):
        pp.get_timestamps_from_messages({"search": ["DOES_NOT_EXIST"]})
    with pytest.raises(ValueError, match="Empty token list"):
        pp.get_timestamps_from_messages({"search": []})
    with pytest.raises(ValueError, match="Provided 1 durations but found 2"):
        pp.split_all_into_trials_by_durations(
            start_msgs={"search": [r"TRIAL_START\s+\d+"]},
            durations={"search": [50]},
        )


def test_saccades_direction_classifies_cardinal_movements(tmp_path: Path):
    pp = _preprocessing(
        tmp_path,
        saccades=pl.DataFrame(
            {
                "event_id": [
                    "right",
                    "left",
                    "down",
                    "up",
                    "diagonal",
                    "zero",
                    "missing",
                ],
                "xStart": [0.0, 10.0, 0.0, 0.0, 0.0, 5.0, np.nan],
                "yStart": [0.0, 0.0, 0.0, 10.0, 0.0, 5.0, 0.0],
                "xEnd": [10.0, 0.0, 0.0, 0.0, 10.0, 5.0, 1.0],
                "yEnd": [0.0, 0.0, 10.0, 0.0, 10.0, 5.0, 1.0],
            }
        ),
    )

    pp.saccades_direction()

    assert _values(pp.saccades, "dir") == [
        "right",
        "left",
        "down",
        "up",
        "",
        "right",
        "",
    ]
    degrees = np.asarray(_values(pp.saccades, "deg"), dtype=float)
    np.testing.assert_allclose(
        degrees[:6],
        [0.0, 180.0, 90.0, -90.0, 45.0, 0.0],
        atol=1e-12,
    )
    assert np.isnan(degrees[6])


def test_saccades_direction_custom_tolerance_and_validation(tmp_path: Path):
    angle_deg = 10.0
    pp = _preprocessing(
        tmp_path,
        saccades=pl.DataFrame(
            {
                "xStart": [0.0],
                "yStart": [0.0],
                "xEnd": [float(np.cos(np.radians(angle_deg)))],
                "yEnd": [float(np.sin(np.radians(angle_deg)))],
            }
        ),
    )

    pp.saccades_direction(tol_deg=5.0)
    assert _values(pp.saccades, "dir") == [""]
    pp.saccades_direction(tol_deg=15.0)
    assert _values(pp.saccades, "dir") == ["right"]

    with pytest.raises(ValueError, match="between 0 and 90"):
        pp.saccades_direction(tol_deg=100)

    pp.saccades = pl.DataFrame({"xStart": [0.0], "yStart": [0.0], "xEnd": [1.0]})
    with pytest.raises(ValueError, match="yEnd"):
        pp.saccades_direction()


def test_process_writes_recipe_and_provenance(tmp_path: Path):
    pp = _preprocessing(
        tmp_path,
        samples=pl.DataFrame({"tSample": [0], "X": [5.0], "Y": [5.0]}),
    )

    pp.process(
        {"bad_samples": {"screen_width": 10, "screen_height": 10}},
    )

    assert _values(pp.samples, "bad") == [False]
    recipe = json.loads((tmp_path / "preprocessing_recipe.json").read_text())
    provenance = json.loads((tmp_path / "preprocessing_provenance.json").read_text())
    assert recipe["tool_version"] == PreProcessing.VERSION
    assert provenance["completed_recipe"] == ["bad_samples"]

    with pytest.raises(AttributeError, match="Unknown preprocessing function"):
        pp.process({"does_not_exist": {}}, log_recipe=False)
    with pytest.raises(TypeError, match="must be a dict"):
        pp.process({"bad_samples": []}, log_recipe=False)  # type: ignore[dict-item]
