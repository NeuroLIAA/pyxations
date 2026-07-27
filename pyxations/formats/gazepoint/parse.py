'''
Created on Nov 7, 2024

@author: placiana
'''
import pandas as pd
from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing
import inspect


def process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, force_best_eye, keep_ascii, overwrite, exp_format, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.csv']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)


    GazePointParse(session_folder_path, exp_format).parse(edf_file_path, detection_algorithm, force_best_eye, overwrite, **kwargs)

class GazePointParse(BidsParse):

    def parse(self, file_path, detection_algorithm, force_best_eye, overwrite, **kwargs):
        from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT
        
        df = pd.read_csv(file_path)

        # Gazepoint records both eyes plus its own combined "Best Point Of Gaze" (BPOG)
        requested_eye = kwargs.get('eye')
        if isinstance(requested_eye, str) and requested_eye.upper() in ('L', 'R'):
            eye = requested_eye.upper()
        else:
            eye = 'B'
        gaze_prefix = {'L': 'LPOG', 'R': 'RPOG', 'B': 'BPOG'}[eye]
        pupil_col = {'L': 'LPD', 'R': 'RPD', 'B': 'LPD'}[eye]
        eye_tag = {'L': 'L', 'R': 'R', 'B': 'Best'}[eye]

        dfSample = df.reset_index().rename(columns={
            "index": "line_number", 
            "TIME": "tSample",
            f"{gaze_prefix}X": "X",
            f"{gaze_prefix}Y": "Y",
            pupil_col: "Pupil"
        })

        # Convert timestamp so samples and detected events share one absolute ms base 
        # compatible with eyelink and remodnav integration
        dfSample['tSample'] = dfSample['tSample'] * 1000.0

        dfBlink = df[df['BKDUR'] > 0].reset_index().rename(columns={
            "index": "line_number", 
            "TIME": "tEnd",
            "BKDUR": "duration"
        })
        # Same seconds -> milliseconds conversion for blink event times.
        dfBlink['tEnd'] = dfBlink['tEnd'] * 1000.0
        dfBlink['duration'] = dfBlink['duration'] * 1000.0
        dfBlink['tStart'] = dfBlink['tEnd'] - dfBlink['duration']

        # Estimate the sampling rate 
        time_s = df['TIME'].to_numpy()
        sample_rate = (len(time_s) - 1) / (time_s[-1] - time_s[0])

        eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=self.session_folder_path,samples=dfSample)

        # Provide the detector inputs
        config = {
            'eye': eye_tag
        }
        detector_params = set(inspect.signature(eye_movement_detector.run_eye_movement).parameters)
        config.update({k: v for k, v in kwargs.items() if k in detector_params and k != 'eye'})
        self.detection_algorithm = detection_algorithm
        dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(sample_rate, config=config)
        
        # Gazepoint has no user messages: use an empty, correctly-typed message table 
        dfMsg = pd.DataFrame(columns=["timestamp", "message"])

        pre_processing = PreProcessing(dfSample, dfFix, dfSacc, dfBlink, dfMsg, self.session_folder_path)

        # Gazepoint reports gaze in normalized 0-1 screen coordinates, so the screen bounds are simply (1, 1)
        screen_res = kwargs.get("screen_res")
        if isinstance(screen_res, (tuple, list)) and len(screen_res) == 2:
            pre_processing.set_metadata(screen_width=screen_res[0], screen_height=screen_res[1])
        elif kwargs.get("screen_width") and kwargs.get("screen_height"):
            pre_processing.set_metadata(screen_width=kwargs["screen_width"], screen_height=kwargs["screen_height"])
        else:
            pre_processing.set_metadata(screen_width=1, screen_height=1)

        bad_sig = inspect.signature(pre_processing.bad_samples).parameters
        bad_params = {
            k: kwargs[k]
            for k in ("screen_height", "screen_width", "mark_nan_as_bad", "inclusive_bounds")
            if k in kwargs and k in bad_sig
        }

        # Forward the optional saccade-direction tolerance, mirroring eyelink.
        dir_params = {}
        if "tol_deg" in kwargs and "tol_deg" in inspect.signature(pre_processing.saccades_direction).parameters:
            dir_params["tol_deg"] = kwargs["tol_deg"]

        # ---- Decide which trialing API to use ----
        prefer_durations = kwargs.get("prefer_durations", False)

        have_explicit_times = ("start_times" in kwargs) and ("end_times" in kwargs)
        have_durations     = ("start_msgs" in kwargs) and ("durations" in kwargs)
        have_message_times = ("start_msgs" in kwargs) and ("end_msgs" in kwargs)

        if not (have_explicit_times or have_durations or have_message_times):
            # No trial markers/times supplied: treat the whole recording as a single trial
            pre_processing.process({
                "bad_samples": bad_params,
                "split_all_into_single_trial": {"phase_name": "complete_recording"},
                "saccades_direction": dir_params,
            })
        else:
            if have_explicit_times:
                seg_func_name = "split_all_into_trials"
            elif have_durations and (prefer_durations or not have_message_times):
                seg_func_name = "split_all_into_trials_by_durations"
            else:
                seg_func_name = "split_all_into_trials_by_msgs"

            seg_func = getattr(pre_processing, seg_func_name)
            seg_sig = inspect.signature(seg_func).parameters
            allowed = set(seg_sig.keys())

            # superset of possible keys across the three APIs
            candidate_keys = {
                # common
                "trial_labels",
                # explicit times
                "start_times", "end_times", "allow_open_last", "require_nonoverlap",
                # messages (both _by_msgs and _by_durations use start_msgs)
                "start_msgs", "end_msgs",
                # durations
                "durations",
                # message-matching extras
                "case_insensitive", "use_regex", "return_match_token",
            }

            seg_params = {k: v for k, v in kwargs.items() if (k in candidate_keys and k in allowed)}

            # Run via the declarative orchestrator (writes recipe/provenance JSONs)
            pre_processing.process({
                "bad_samples": bad_params,
                seg_func_name: seg_params,
                "saccades_direction": dir_params,
            })
        dfSample = pre_processing.samples
        dfBlink = pre_processing.blinks
        dfFix = pre_processing.fixations
        dfSacc = pre_processing.saccades
        
        self.store_dataframes(dfSample, dfBlink=dfBlink, dfFix=dfFix, dfSacc=dfSacc)

    