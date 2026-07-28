import json
from pathlib import Path

import polars as pl
import pytest

from pyxations import (
    dataset_to_bids,
    psychopy_log_to_events,
    read_behavioral_events,
)
from pyxations.tables import read_tsv

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


def _write_log(path: Path) -> None:
    path.write_text(
        """1.0000\tEXP\tinstruction_text: autoDraw = True
2.0000\tEXP\tNew trial (rep=0, index=4): OrderedDict([('condition_image', 'planned.png'), ('is_target_present', 1)])
2.1000\tEXP\ttrial_image: image = 'actual.png'
2.2000\tEXP\ttrial_image: autoDraw = True
2.3000\tDATA\tKeypress: right
2.4000\tEXP\tslider: markerPos = None
2.5000\tEXP\tslider: markerPos = 4.0
5.0000\tEXP\tNew trial (rep=0, index=8): {'condition_image': 'second.png', 'is_target_present': 0}
5.1000\tEXP\ttrial_image: image = 'second-actual.png'
5.2000\tDATA\tKeypress: left
""",
        encoding="utf-8",
    )


def test_psychopy_log_to_events_preserves_conditions_and_actual_updates(tmp_path):
    log = tmp_path / "subject_session.log"
    _write_log(log)

    parsed = psychopy_log_to_events(log)
    assert "trial_image_image" in parsed

    events = read_behavioral_events(
        log,
        column_map={
            "trial_image_image": "stimulus",
            "is_target_present": "target_present",
        },
    )

    assert events.height == 2
    assert events["trial_number"].to_list() == [0, 1]
    assert events["trial_index"].to_list() == [0, 1]
    assert events["psychopy_index"].to_list() == [4, 8]
    assert events["condition_image"].to_list() == ["planned.png", "second.png"]
    assert events["stimulus"].to_list() == ["actual.png", "second-actual.png"]
    assert events["target_present"].to_list() == [1, 0]
    assert events["keypresses"].to_list() == [["right"], ["left"]]
    assert events["slider_markerpos"].to_list() == [4.0, None]
    assert events["psychopy_trial_interval"].to_list() == [3.0, None]
    assert events["onset"].null_count() == 2
    assert events["duration"].null_count() == 2
    assert "trial_image_autodraw" not in events


def test_psychopy_log_handles_malformed_payloads_and_mapping_errors(tmp_path):
    malformed = tmp_path / "malformed.log"
    malformed.write_text(
        """not a PsychoPy log record
1.0\tEXP\tNew trial (rep=unknown, index=[1]): ['not a mapping']
2.0\tEXP\tNew trial (rep=0, index=1): {'onset': 30, 'nested': [1], '!!!': 'blank', '12 field': 'number'}
""",
        encoding="utf-8",
    )
    events = psychopy_log_to_events(malformed)
    assert events.item(0, "psychopy_condition_payload") == "['not a mapping']"
    assert events.item(0, "psychopy_rep") == "unknown"
    assert events.item(0, "psychopy_index") == "[1]"
    assert events.item(1, "psychopy_condition_onset") == 30
    assert events.item(1, "nested") == "[1]"
    assert events.item(1, "psychopy_field") == "blank"
    assert events.item(1, "psychopy_12_field") == "number"
    assert events.item(1, "psychopy_onset") == 2.0

    empty = tmp_path / "empty.log"
    empty.write_text("1.0\tEXP\tunrelated message\n", encoding="utf-8")
    assert psychopy_log_to_events(empty).is_empty()

    with pytest.raises(ValueError, match="not found"):
        read_behavioral_events(malformed, column_map={"missing": "value"})
    with pytest.raises(ValueError, match="duplicate"):
        read_behavioral_events(
            malformed,
            column_map={"psychopy_condition_payload": "trial_type"},
        )


@pytest.mark.parametrize(("suffix", "separator"), [(".csv", ","), (".tsv", "\t")])
def test_behavioral_reader_maps_tabular_sources(tmp_path, suffix, separator):
    path = tmp_path / f"behavior{suffix}"
    path.write_text(
        f"source trial{separator}answer\n4{separator}left\n",
        encoding="utf-8",
    )

    events = read_behavioral_events(
        path,
        column_map={"source trial": "trial_number"},
    )

    assert events.to_dicts() == [{"trial_number": 4, "answer": "left"}]

    unsupported = tmp_path / "behavior.json"
    with pytest.raises(ValueError, match="Unsupported behavioral file format"):
        read_behavioral_events(unsupported)


def test_target_absent_example_log_retains_planned_and_displayed_values():
    log = (
        EXAMPLES_ROOT
        / "eyelink_target_absent"
        / "ab01_first_half_2023-09-11_11h32.09.057.log"
    )
    events = psychopy_log_to_events(log)

    assert events.height == 100
    assert events.item(0, "t_image") == "stimuli/0000.jpg"
    assert events.item(0, "trial_image_image") == "stimuli/0089.jpg"
    assert events.item(0, "t_target") == "targets/carton.jpg"
    assert events.item(0, "trial_target_image") == "targets/clock.jpg"
    assert events.item(0, "is_target_present") == 0
    assert events["keypresses"].to_list()[0] == ["down"]


def test_dataset_to_bids_uses_psychopy_log_when_no_behavior_table_exists(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "s01_A_task-look.asc").write_text(
        """** VERSION: EYELINK II 1
SAMPLES GAZE LEFT RATE 1000.00 TRACKING CR FILTER 2
!MODE RECORD CR 1000 2 1 L
1000 100.0 200.0 500.0
1001 101.0 201.0 501.0
""",
        encoding="utf-8",
    )
    _write_log(source / "s01_A_task-look.log")

    dataset = dataset_to_bids(
        tmp_path,
        source,
        "psychopy-log",
        behavioral_column_map={
            "trial_image_image": "stimulus",
            "is_target_present": "target_present",
        },
    )

    behavior = dataset / "sub-0001" / "ses-A" / "beh"
    event_path = next(behavior.glob("*_events.tsv"))
    events = read_tsv(event_path, has_header=True)
    assert events.height == 2
    assert events["stimulus"].to_list() == ["actual.png", "second-actual.png"]
    assert events["source_file"].unique().to_list() == ["s01_A_task-look.log"]
    assert events["onset"].null_count() == 2
    assert (dataset / "sourcedata" / "s01_A_task-look.log").is_file()
    sidecar = json.loads(event_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["psychopy_onset"]["Units"] == "s"
    assert "not assumed to be synchronized" in sidecar["psychopy_onset"]["Description"]


def test_behavioral_csv_takes_precedence_over_psychopy_log(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "s01_A_task-look.asc").write_text(
        """** VERSION: EYELINK II 1
SAMPLES GAZE LEFT RATE 1000.00 TRACKING CR FILTER 2
1000 100.0 200.0 500.0
""",
        encoding="utf-8",
    )
    _write_log(source / "s01_A_task-look.log")
    pl.DataFrame(
        {
            "trial_number": [0],
            "native_stimulus": ["canonical.csv"],
        }
    ).write_csv(source / "s01_A_task-look_behavior.csv")

    dataset = dataset_to_bids(
        tmp_path,
        source,
        "csv-precedence",
        behavioral_column_map={"native_stimulus": "stimulus"},
    )
    event_path = next((dataset / "sub-0001" / "ses-A" / "beh").glob("*_events.tsv"))
    events = read_tsv(event_path, has_header=True)
    assert events.height == 1
    assert events.item(0, "stimulus") == "canonical.csv"
    assert events.item(0, "source_file") == "s01_A_task-look_behavior.csv"
