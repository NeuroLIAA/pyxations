from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


Number = Union[int, float]
PathLike = Union[str, Path]


@dataclass
class SessionMetadata:
    """Lightweight metadata container saved alongside derivatives."""
    coords_unit: str = "px"          # 'px' or 'deg'
    time_unit: str = "ms"            # 'ms'
    pupil_unit: str = "arbitrary"
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    extra: Dict[str, Union[str, int, float, bool, None]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "coords_unit": self.coords_unit,
            "time_unit": self.time_unit,
            "pupil_unit": self.pupil_unit,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "extra": self.extra,
        }


class PreProcessing:
    """
    Pyxations preprocessing: trial segmentation, quality flags, and saccade direction.
    All mutating functions are safe (copy-aware) and validate required columns.

    Tables (pd.DataFrame) expected:
        samples:   typically contains 'tSample' (ms), gaze columns (e.g., 'LX','LY','RX','RY' or 'X','Y')
        fixations: typically contains 'tStart','tEnd' and optional 'xAvg','yAvg'
        saccades:  typically contains 'tStart','tEnd','xStart','yStart','xEnd','yEnd'
        blinks:    typically contains 'tStart','tEnd' (optional)
        user_messages: must contain 'timestamp','message'

    New columns created:
        - All tables after trialing: 'phase', 'trial_number', 'trial_label' (optional)
        - samples/fixations/saccades: 'bad' (bool) after bad_samples()
        - saccades: 'deg' (float degrees), 'dir' (str) after saccades_direction()
    """

    VERSION = "0.2.0"

    def __init__(
        self,
        samples: pd.DataFrame,
        fixations: pd.DataFrame,
        saccades: pd.DataFrame,
        blinks: pd.DataFrame,
        user_messages: pd.DataFrame,
        session_path: PathLike,
        metadata: Optional[SessionMetadata] = None,
    ):
        self.samples = samples.copy()
        self.fixations = fixations.copy()
        self.saccades = saccades.copy()
        self.blinks = blinks.copy()
        self.user_messages = user_messages.copy()
        self.session_path = Path(session_path)
        self.metadata = metadata or SessionMetadata()

        # Normalize dtypes where possible (strings for messages)
        if "message" in self.user_messages.columns:
            self.user_messages["message"] = self.user_messages["message"].astype(str)

    # ------------------------------- Utilities ------------------------------- #

    @staticmethod
    def _require_columns(
        df: pd.DataFrame, cols: Sequence[str], context: str
    ) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{context}] Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

    @staticmethod
    def _assert_nonoverlap(starts: Sequence[int], ends: Sequence[int], key: str, session: Path) -> None:
        if len(starts) != len(ends):
            raise ValueError(
                f"[{key}] start_times and end_times must have the same length, "
                f"got {len(starts)} vs {len(ends)} in session: {session}"
            )
        for i, (s, e) in enumerate(zip(starts, ends)):
            if not (s < e):
                raise ValueError(
                    f"[{key}] Non-positive interval at trial {i}: start={s}, end={e} "
                    f"in session: {session}"
                )
            if i < len(starts) - 1:
                if e > starts[i + 1]:
                    raise ValueError(
                        f"[{key}] Overlapping trials {i}–{i+1}: end[i]={e} > start[i+1]={starts[i+1]} "
                        f"in session: {session}"
                    )

    @staticmethod
    def _ensure_columns_exist(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
        """Return the subset of 'cols' that actually exist in df."""
        return [c for c in cols if c in df.columns]

    def _save_json_sidecar(self, obj: dict, filename: str) -> None:
        outdir = self.session_path
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / filename, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    # ---------------------------- Public API: Meta ---------------------------- #

    def set_metadata(
        self,
        coords_unit: Optional[str] = None,
        time_unit: Optional[str] = None,
        pupil_unit: Optional[str] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        **extra,
    ) -> None:
        """Update session-level metadata used in bounds checks and documentation."""
        if coords_unit is not None:
            self.metadata.coords_unit = coords_unit
        if time_unit is not None:
            self.metadata.time_unit = time_unit
        if pupil_unit is not None:
            self.metadata.pupil_unit = pupil_unit
        if screen_width is not None:
            self.metadata.screen_width = screen_width
        if screen_height is not None:
            self.metadata.screen_height = screen_height
        self.metadata.extra.update(extra)

    def save_metadata(self, filename: str = "metadata.json") -> None:
        """Persist metadata next to derivatives for reproducibility."""
        self._save_json_sidecar(self.metadata.to_dict(), filename)

    # ----------------------- Public API: Message Parsing ---------------------- #

    def get_timestamps_from_messages(
        self,
        messages_dict: Dict[str, List[str]],
        *,
        case_insensitive: bool = True,
        use_regex: bool = True,
        return_match_token: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Extract ordered timestamps per phase by matching message substrings/patterns.

        Parameters
        ----------
        messages_dict : dict
            e.g., {'trial': ['TRIAL_START', 'BEGIN_TRIAL'], 'stim': ['STIM_ONSET']}
        case_insensitive : bool
            If True, ignore case during matching.
        use_regex : bool
            If True, treat entries as regex patterns joined by '|'; otherwise escape literals.
        return_match_token : bool
            If True, also creates/updates a 'matched_token' column with the first matched pattern.

        Returns
        -------
        Dict[str, List[int]]
            Ordered timestamps in ms for each key.
        """
        df = self.user_messages
        self._require_columns(df, ["timestamp", "message"], "get_timestamps_from_messages")

        timestamps_dict: Dict[str, List[int]] = {}
        flags = re.I if case_insensitive else 0

        # Prepare an optional matched_token column for traceability
        if return_match_token and "matched_token" not in df.columns:
            df = df.copy()
            df["matched_token"] = pd.Series([None] * len(df), index=df.index)

        for key, tokens in messages_dict.items():
            if not tokens:
                raise ValueError(f"[{key}] Empty token list passed to get_timestamps_from_messages.")
            parts = tokens if use_regex else [re.escape(t) for t in tokens]
            pat = re.compile("|".join(parts), flags=flags)

            hits = df[df["message"].str.contains(pat, regex=True, na=False)].copy()
            hits.sort_values(by="timestamp", inplace=True)

            if return_match_token and not hits.empty:
                # record which token matched first for each hit
                def _which(m: str) -> Optional[str]:
                    for t in tokens:
                        if (re.search(t, m, flags=flags) if use_regex else re.search(re.escape(t), m, flags=flags)):
                            return t
                    return None

                hits["matched_token"] = hits["message"].apply(_which)
                # write back those rows (optional traceability)
                df.loc[hits.index, "matched_token"] = hits["matched_token"]

            stamps = hits["timestamp"].astype(int).tolist()
            if len(stamps) == 0:
                raise ValueError(
                    f"[{key}] No timestamps found for messages {tokens} "
                    f"in session: {self.session_path}"
                )
            timestamps_dict[key] = stamps

        # Persist updated matched_token if requested
        if return_match_token:
            self.user_messages = df

        return timestamps_dict

    # ---------------------- Public API: Trial Segmentation -------------------- #

    def split_all_into_trials(
        self,
        start_times: Dict[str, List[int]],
        end_times: Dict[str, List[int]],
        trial_labels: Optional[Dict[str, List[str]]] = None,
        *,
        allow_open_last: bool = True,
        require_nonoverlap: bool = True,
    ) -> None:
        """Segment samples/fixations/saccades/blinks using explicit times."""
        for df in (self.samples, self.fixations, self.saccades, self.blinks):
            self._split_into_trials_df(
                df, start_times, end_times, trial_labels,
                allow_open_last=allow_open_last,
                require_nonoverlap=require_nonoverlap,
            )

    def split_all_into_trials_by_msgs(
        self,
        start_msgs: Dict[str, List[str]],
        end_msgs: Dict[str, List[str]],
        trial_labels: Optional[Dict[str, List[str]]] = None,
        **msg_kwargs,
    ) -> None:
        """Segment tables using start and end message patterns."""
        starts = self.get_timestamps_from_messages(start_msgs, **msg_kwargs)
        ends = self.get_timestamps_from_messages(end_msgs, **msg_kwargs)
        self.split_all_into_trials(starts, ends, trial_labels)

    def split_all_into_trials_by_durations(
        self,
        start_msgs: Dict[str, List[str]],
        durations: Dict[str, List[int]],
        trial_labels: Optional[Dict[str, List[str]]] = None,
        **msg_kwargs,
    ) -> None:
        """Segment using start message patterns and per-trial durations (ms)."""
        starts = self.get_timestamps_from_messages(start_msgs, **msg_kwargs)
        end_times: Dict[str, List[int]] = {}
        for key, durs in durations.items():
            s = starts.get(key, [])
            if len(durs) < len(s):
                raise ValueError(
                    f"[{key}] Provided {len(durs)} durations but found {len(s)} start times "
                    f"in session: {self.session_path}"
                )
            end_times[key] = [st + du for st, du in zip(s, durs)]
        self.split_all_into_trials(starts, end_times, trial_labels)

    def _split_into_trials_df(
        self,
        data: pd.DataFrame,
        start_times: Dict[str, List[int]],
        end_times: Dict[str, List[int]],
        trial_labels: Optional[Dict[str, List[str]]] = None,
        *,
        allow_open_last: bool = True,
        require_nonoverlap: bool = True,
    ) -> None:
        """
        Core segmentation for a single table. Works with 'tSample' OR ('tStart','tEnd').
        Adds 'phase', 'trial_number', 'trial_label'.
        """
        if data is self.samples:
            time_mode = "sample"
            self._require_columns(data, ["tSample"], "split_into_trials(samples)")
        else:
            # events (fixations/saccades/blinks)
            time_mode = "event"
            self._require_columns(data, ["tStart", "tEnd"], "split_into_trials(events)")

        df = data.copy()
        # Initialize columns deterministically
        df["phase"] = ""
        df["trial_number"] = -1
        df["trial_label"] = ""

        for key in start_times.keys():
            start_list = list(start_times[key])
            end_list = list(end_times[key])

            # Discard starts after last end (common partial last-trial artifact)
            if allow_open_last and end_list:
                last_end = end_list[-1]
                start_list = [st for st in start_list if st < last_end]

            # Sanity checks
            if require_nonoverlap:
                self._assert_nonoverlap(start_list, end_list, key, self.session_path)
            elif len(start_list) != len(end_list):
                raise ValueError(
                    f"[{key}] start_times and end_times length mismatch: {len(start_list)} vs {len(end_list)} "
                    f"in session: {self.session_path}"
                )

            labels = trial_labels.get(key) if (trial_labels and key in trial_labels) else None
            if labels is not None and len(labels) != len(start_list):
                raise ValueError(
                    f"[{key}] Computed {len(start_list)} trials but got {len(labels)} trial labels "
                    f"in session: {self.session_path}"
                )

            # Apply segmentation
            if time_mode == "sample":
                t = df["tSample"].values
                for i, (st, en) in enumerate(zip(start_list, end_list)):
                    mask = (t >= st) & (t <= en)
                    if not np.any(mask):
                        continue
                    df.loc[mask, "trial_number"] = i
                    df.loc[mask, "phase"] = str(key)
                    if labels is not None:
                        df.loc[mask, "trial_label"] = labels[i]
            else:
                t0 = df["tStart"].values
                t1 = df["tEnd"].values
                for i, (st, en) in enumerate(zip(start_list, end_list)):
                    mask = (t0 >= st) & (t1 <= en)
                    if not np.any(mask):
                        continue
                    df.loc[mask, "trial_number"] = i
                    df.loc[mask, "phase"] = str(key)
                    if labels is not None:
                        df.loc[mask, "trial_label"] = labels[i]

        # Commit
        if data is self.samples:
            self.samples = df
        elif data is self.fixations:
            self.fixations = df
        elif data is self.saccades:
            self.saccades = df
        elif data is self.blinks:
            self.blinks = df

    # ------------------------- Public API: QC / Flags ------------------------- #

    def bad_samples(
        self,
        screen_height: Optional[int] = None,
        screen_width: Optional[int] = None,
        *,
        mark_nan_as_bad: bool = True,
        inclusive_bounds: bool = True,
    ) -> None:
        """
        Mark rows as 'bad' if any available coordinate falls outside screen bounds.
        Applies to samples, fixations, saccades. (Blinks unaffected.)

        If width/height not provided, will use metadata.screen_* if available.
        """
        H = screen_height if screen_height is not None else self.metadata.screen_height
        W = screen_width if screen_width is not None else self.metadata.screen_width
        if H is None or W is None:
            raise ValueError(
                "bad_samples requires screen_height and screen_width (either passed "
                "or set via set_metadata())."
            )

        def _mark(df: pd.DataFrame) -> pd.DataFrame:
            d = df.copy()

            # Gather candidate coordinate columns if present
            coord_cols = self._ensure_columns_exist(
                d,
                [
                    "LX", "LY", "RX", "RY", "X", "Y",
                    "xStart", "xEnd", "yStart", "yEnd", "xAvg", "yAvg",
                ],
            )
            if not coord_cols:
                # If no coords present, default to 'not bad'
                if "bad" not in d.columns:
                    d["bad"] = False
                return d

            xcols = [c for c in coord_cols if c.lower().startswith("x")]
            ycols = [c for c in coord_cols if c.lower().startswith("y")]

            # Validity masks for each axis
            if inclusive_bounds:
                valid_w = np.logical_and.reduce([d[c].ge(0) & d[c].le(W) for c in xcols]) if xcols else True
                valid_h = np.logical_and.reduce([d[c].ge(0) & d[c].le(H) for c in ycols]) if ycols else True
            else:
                valid_w = np.logical_and.reduce([d[c].gt(0) & d[c].lt(W) for c in xcols]) if xcols else True
                valid_h = np.logical_and.reduce([d[c].gt(0) & d[c].lt(H) for c in ycols]) if ycols else True

            bad = ~(valid_w & valid_h)
            if mark_nan_as_bad:
                bad |= d[coord_cols].isna().any(axis=1)

            d["bad"] = bad.values
            return d

        self.samples = _mark(self.samples)
        self.fixations = _mark(self.fixations)
        self.saccades = _mark(self.saccades)

    # ---------------------- Public API: Saccade Direction --------------------- #

    def saccades_direction(self, tol_deg: float = 15.0) -> None:
        """
        Compute saccade angle (deg) and cardinal direction with tolerance bands.

        Parameters
        ----------
        tol_deg : float
            Half-width of the acceptance band around 0°, ±90°, and ±180°
            for classifying right/left/up/down.
        """
        df = self.saccades.copy()
        self._require_columns(
            df, ["xStart", "xEnd", "yStart", "yEnd"], "saccades_direction"
        )

        x_dif = df["xEnd"].astype(float) - df["xStart"].astype(float)
        y_dif = df["yEnd"].astype(float) - df["yStart"].astype(float)
        deg = np.degrees(np.arctan2(y_dif.to_numpy(), x_dif.to_numpy()))
        df["deg"] = deg.astype(float)

        # Tolerant direction bins
        right = (-tol_deg < df["deg"]) & (df["deg"] < tol_deg)
        left = (df["deg"] > 180 - tol_deg) | (df["deg"] < -180 + tol_deg)
        down = ((90 - tol_deg) < df["deg"]) & (df["deg"] < (90 + tol_deg))
        up = ((-90 - tol_deg) < df["deg"]) & (df["deg"] < (-90 + tol_deg))

        df["dir"] = ""
        df.loc[right, "dir"] = "right"
        df.loc[left, "dir"] = "left"
        df.loc[down, "dir"] = "down"
        df.loc[up, "dir"] = "up"

        self.saccades = df

    # -------------------------- Public API: Orchestrator ---------------------- #

    def process(
        self,
        functions_and_params: Dict[str, Dict],
        *,
        log_recipe: bool = True,
        recipe_filename: str = "preprocessing_recipe.json",
        provenance_filename: str = "preprocessing_provenance.json",
    ) -> None:
        """
        Run a declarative preprocessing recipe, e.g.:
            pp.process({
                "split_all_into_trials_by_msgs": {
                    "start_msgs": {"trial": ["TRIAL_START"]},
                    "end_msgs": {"trial": ["TRIAL_END"]},
                },
                "bad_samples": {"screen_height": 1080, "screen_width": 1920},
                "saccades_direction": {"tol_deg": 15},
            })

        Unknown function names raise a helpful error.
        """
        # Optional: save the declared recipe for exact reproducibility
        if log_recipe:
            recipe_obj = {
                "declared_recipe": functions_and_params,
                "tool_version": self.VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "session_path": str(self.session_path),
            }
            self._save_json_sidecar(recipe_obj, recipe_filename)

        for func_name, params in functions_and_params.items():
            if not hasattr(self, func_name):
                raise AttributeError(
                    f"Unknown preprocessing function '{func_name}'. "
                    f"Available: {[m for m in dir(self) if not m.startswith('_')]}"
                )
            fn = getattr(self, func_name)
            if not isinstance(params, dict):
                raise TypeError(
                    f"Parameters for '{func_name}' must be a dict, got {type(params)}"
                )
            fn(**params)

        # Save lightweight provenance after successful run
        if log_recipe:
            prov = {
                "completed_recipe": list(functions_and_params.keys()),
                "tool_version": self.VERSION,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": self.metadata.to_dict(),
            }
            self._save_json_sidecar(prov, provenance_filename)
