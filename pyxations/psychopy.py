"""PsychoPy text-log parsing without requiring PsychoPy at runtime."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from .behavior import _column_name
from .tables import json_value

_LOG_LINE = re.compile(
    r"^\s*(?P<timestamp>[+-]?\d+(?:\.\d+)?)\s*\t\s*"
    r"(?P<level>[^\t]+?)\s*\t\s*(?P<message>.*)$"
)
_NEW_TRIAL = re.compile(r"^New trial\s*\((?P<context>[^)]*)\):\s*(?P<payload>.*)$")
_CONTEXT_VALUE = re.compile(r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^,]+)")
_COMPONENT_VALUE = re.compile(
    r"^(?P<component>[A-Za-z_][\w .-]*):\s*"
    r"(?P<attribute>[A-Za-z_][\w .-]*)\s*=\s*(?P<value>.*)$"
)
_KEYPRESS = re.compile(r"^Keypress:\s*(?P<value>.*)$")
_TRACKED_COMPONENT_ATTRIBUTES = {
    "image",
    "markerpos",
    "pos",
    "rating",
    "text",
    "value",
}
_RESERVED_COLUMNS = {
    "duration",
    "onset",
    "trial_index",
    "trial_number",
    "trial_type",
}


def _literal(value: str) -> Any:
    """Parse a scalar safely, retaining unsupported expressions as text."""

    value = value.strip()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    if parsed is None or isinstance(parsed, (str, bool, int, float)):
        return parsed
    return json.dumps(json_value(parsed), ensure_ascii=False)


def _condition_values(payload: str) -> dict[str, Any]:
    """Parse the mapping printed by PsychoPy's TrialHandler."""

    candidate = payload.strip()
    try:
        if candidate.startswith("OrderedDict(") and candidate.endswith(")"):
            pairs = ast.literal_eval(candidate[len("OrderedDict(") : -1])
            values = dict(pairs)
        else:
            values = ast.literal_eval(candidate)
            if not isinstance(values, Mapping):
                return {}
    except (SyntaxError, ValueError, TypeError):
        return {}

    result = {}
    for name, value in values.items():
        column = _column_name(str(name), prefix="psychopy")
        if column in _RESERVED_COLUMNS:
            column = f"psychopy_condition_{column}"
        result[column] = (
            value
            if value is None or isinstance(value, (str, bool, int, float))
            else json.dumps(json_value(value), ensure_ascii=False)
        )
    return result


def psychopy_log_to_events(
    log_file_path: str | Path,
) -> pl.DataFrame:
    """Parse PsychoPy ``.log`` trials into a BIDS-ready Polars table.

    The parser creates one row for every ``New trial`` record. It retains the
    TrialHandler condition mapping, subsequent component property updates, and
    keypresses observed before the next trial. Component updates use columns
    such as ``trial_image_image``; repeated updates retain their last value.

    PsychoPy timestamps use a separate clock that is not necessarily
    synchronized with the eye tracker. They are therefore stored as
    ``psychopy_onset`` and ``psychopy_trial_interval``. Canonical BIDS
    ``onset`` and ``duration`` remain missing rather than claiming a false
    synchronization.

    Parameters
    ----------
    log_file_path : str or pathlib.Path
        PsychoPy text log to parse.

    Returns
    -------
    polars.DataFrame
        One row per logged trial, or an empty table when no trial markers are
        present.
    """

    path = Path(log_file_path)
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_keypresses: list[str] = []

    def finish(next_timestamp: float | None = None) -> None:
        nonlocal current, current_keypresses
        if current is None:
            return
        if next_timestamp is not None:
            current["psychopy_trial_interval"] = (
                next_timestamp - current["psychopy_onset"]
            )
        if current_keypresses:
            current["keypresses"] = current_keypresses.copy()
        rows.append(current)
        current = None
        current_keypresses = []

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            match = _LOG_LINE.match(raw_line.rstrip("\r\n"))
            if match is None:
                continue
            timestamp = float(match.group("timestamp"))
            message = match.group("message").strip()

            trial_match = _NEW_TRIAL.match(message)
            if trial_match is not None:
                finish(timestamp)
                trial_number = len(rows)
                current = {
                    "onset": None,
                    "duration": None,
                    "trial_type": "psychopy_trial",
                    "trial_number": trial_number,
                    "trial_index": trial_number,
                    "psychopy_onset": timestamp,
                }
                for context in _CONTEXT_VALUE.finditer(trial_match.group("context")):
                    current[
                        f"psychopy_{_column_name(context.group('name'), prefix='psychopy')}"
                    ] = _literal(context.group("value"))
                condition_values = _condition_values(trial_match.group("payload"))
                if condition_values:
                    current.update(condition_values)
                elif trial_match.group("payload"):
                    current["psychopy_condition_payload"] = trial_match.group("payload")
                continue

            if current is None:
                continue

            keypress_match = _KEYPRESS.match(message)
            if keypress_match is not None:
                current_keypresses.append(keypress_match.group("value").strip())
                continue

            component_match = _COMPONENT_VALUE.match(message)
            if component_match is not None:
                if (
                    component_match.group("attribute").strip().lower()
                    not in _TRACKED_COMPONENT_ATTRIBUTES
                ):
                    continue
                column = _column_name(
                    f"{component_match.group('component')}_"
                    f"{component_match.group('attribute')}",
                    prefix="psychopy",
                )
                current[column] = _literal(component_match.group("value"))

    finish()
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(rows, infer_schema_length=None, strict=False)
