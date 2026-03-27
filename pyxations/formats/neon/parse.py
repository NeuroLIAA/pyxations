import inspect
import numpy as np
import pandas as pd
from pathlib import Path

from pupil_labs.neon_recording import load as load_neon

from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing


def _find_recording_folder(eye_tracking_data_path: Path) -> Path:
    """Return the Neon recording folder (the one that contains info.json)."""
    if (eye_tracking_data_path / "info.json").exists():
        return eye_tracking_data_path
    candidates = [d for d in eye_tracking_data_path.iterdir()
                  if d.is_dir() and (d / "info.json").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No Neon recording folder (with info.json) found in {eye_tracking_data_path}"
        )
    return candidates[0]


def process_session(eye_tracking_data_path, detection_algorithm, session_folder_path,
                    force_best_eye, keep_ascii, overwrite, exp_format, **kwargs):
    recording_folder = _find_recording_folder(Path(eye_tracking_data_path))
    NeonParse(session_folder_path, exp_format).parse(
        recording_folder, detection_algorithm, overwrite, **kwargs
    )


class NeonParse(BidsParse):

    def parse(self, recording_folder: Path, detection_algorithm: str, overwrite: bool, **kwargs):
        from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT

        rec = load_neon(Path(recording_folder))
        start_ns    = rec.start_time
        sample_rate = rec.info.get("gaze_frequency", 200)

        def _ms(arr):
            return (arr - start_ns) / 1e6

        # ── Samples ──────────────────────────────────────────────────────────
        dfSample = pd.DataFrame({
            "tSample": _ms(rec.gaze["time"]),
            "X":       rec.gaze["point_x"].astype("float64"),
            "Y":       rec.gaze["point_y"].astype("float64"),
        })

        dfPupil = pd.DataFrame({
            "tSample": _ms(rec.pupil["time"]),
            "Pupil":   ((rec.pupil["diameter_left"] + rec.pupil["diameter_right"]) / 2).astype("float64"),
        })

        dfSample = pd.merge_asof(
            dfSample.sort_values("tSample"),
            dfPupil.sort_values("tSample"),
            on="tSample",
            direction="nearest",
        )

        dfSample["Eyes_recorded"] = "LR"
        dfSample["Rate_recorded"] = float(sample_rate)
        dfSample = dfSample.reset_index(drop=True)
        dfSample["Line_number"] = dfSample.index

        # ── Blinks ───────────────────────────────────────────────────────────
        dfBlink = pd.DataFrame({
            "tStart": _ms(rec.blinks["start_time"]),
            "tEnd":   _ms(rec.blinks["stop_time"]),
        })
        dfBlink["duration"] = dfBlink["tEnd"] - dfBlink["tStart"]
        dfBlink["eye"]      = "LR"

        # ── Messages (Neon events) ────────────────────────────────────────────
        events = rec.events
        if len(events) > 0:
            dfMsg = pd.DataFrame({
                "timestamp": _ms(events["time"]),
                "message":   events["event"],
            })
        else:
            dfMsg = pd.DataFrame(columns=["timestamp", "message"])

        # ── Fixations & Saccades ──────────────────────────────────────────────
        self.detection_algorithm = detection_algorithm

        if detection_algorithm == "neon":
            fx   = rec.fixations
            sacc = rec.saccades

            dfFix = pd.DataFrame({
                "eye":      "LR",
                "tStart":   _ms(fx["start_time"]),
                "tEnd":     _ms(fx["stop_time"]),
                "duration": _ms(fx["stop_time"]) - _ms(fx["start_time"]),
                "xAvg":     fx["mean_gaze_x"].astype("float64"),
                "yAvg":     fx["mean_gaze_y"].astype("float64"),
                "pupilAvg": np.nan,
            }).reset_index(drop=True)

            dfSacc = pd.DataFrame({
                "eye":      "LR",
                "tStart":   _ms(sacc["start_time"]),
                "tEnd":     _ms(sacc["stop_time"]),
                "duration": _ms(sacc["stop_time"]) - _ms(sacc["start_time"]),
                "xStart":   sacc["start_gaze_x"].astype("float64"),
                "yStart":   sacc["start_gaze_y"].astype("float64"),
                "xEnd":     sacc["stop_gaze_x"].astype("float64"),
                "yEnd":     sacc["stop_gaze_y"].astype("float64"),
                "ampDeg":   sacc["amplitude_angle"].astype("float64"),
                # max_velocity from Neon is in px/s — convert to deg/s.
                # Default FOV: 1600px / 103° (Neon scene camera horizontal FOV).
                "vPeak":    sacc["max_velocity"].astype("float64") / kwargs.get("px_per_deg", 1600 / 103),
            }).reset_index(drop=True)

        else:
            eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](
                session_folder_path=self.session_folder_path,
                samples=dfSample,
            )
            config = kwargs.pop("detector_config", {})
            dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(
                sample_rate, config=config
            )

        # ── PreProcessing ─────────────────────────────────────────────────────
        pre_processing = PreProcessing(dfSample, dfFix, dfSacc, dfBlink, dfMsg,
                                       self.session_folder_path)

        if "screen_width" in kwargs and "screen_height" in kwargs:
            pre_processing.set_metadata(
                screen_width=kwargs["screen_width"],
                screen_height=kwargs["screen_height"],
            )

        prefer_durations   = kwargs.get("prefer_durations", False)
        have_explicit      = ("start_times" in kwargs) and ("end_times" in kwargs)
        have_durations     = ("start_msgs"  in kwargs) and ("durations" in kwargs)
        have_message_times = ("start_msgs"  in kwargs) and ("end_msgs"  in kwargs)

        if not (have_explicit or have_durations or have_message_times):
            # No segmentation parameters provided — treat the whole recording as one trial.
            for df_attr in ("samples", "fixations", "saccades", "blinks"):
                df = getattr(pre_processing, df_attr)
                if "trial_number" not in df.columns:
                    df["trial_number"] = 0
                    df["trial_label"]  = "recording"
                    df["phase"]        = "recording"
                    setattr(pre_processing, df_attr, df)
        else:
            if have_explicit:
                seg_func_name = "split_all_into_trials"
            elif have_durations and (prefer_durations or not have_message_times):
                seg_func_name = "split_all_into_trials_by_durations"
            else:
                seg_func_name = "split_all_into_trials_by_msgs"

            seg_func = getattr(pre_processing, seg_func_name)
            seg_sig  = inspect.signature(seg_func).parameters
            candidate_keys = {
                "trial_labels",
                "start_times", "end_times", "allow_open_last", "require_nonoverlap",
                "start_msgs", "end_msgs",
                "durations",
                "case_insensitive", "use_regex", "return_match_token",
            }
            seg_params = {k: v for k, v in kwargs.items() if k in candidate_keys and k in seg_sig}
            pre_processing.process({seg_func_name: seg_params})

        # ── Saccade direction ─────────────────────────────────────────────────
        dir_params = {}
        dir_sig = inspect.signature(pre_processing.saccades_direction).parameters
        if "tol_deg" in kwargs and "tol_deg" in dir_sig:
            dir_params["tol_deg"] = kwargs["tol_deg"]
        pre_processing.saccades_direction(**dir_params)

        # ── Save ──────────────────────────────────────────────────────────────
        events_folder = self.session_folder_path / f"{detection_algorithm}_events"
        events_folder.mkdir(parents=True, exist_ok=True)

        self.save_dataframe(pre_processing.samples,    self.session_folder_path, "samples", key="samples")
        self.save_dataframe(pre_processing.fixations,  events_folder, "fix",   key="fix")
        self.save_dataframe(pre_processing.saccades,   events_folder, "sacc",  key="sacc")
        self.save_dataframe(pre_processing.blinks,     events_folder, "blink", key="blink")
        if not dfMsg.empty:
            self.save_dataframe(dfMsg, self.session_folder_path, "msg", key="msg")
