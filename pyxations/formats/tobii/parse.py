'''
Created on Nov 7, 2024

@author: placiana
'''
import pandas as pd
import numpy as np
from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing
import inspect


def process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, force_best_eye, overwrite, exp_format, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.txt']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)

    TobiiParse(session_folder_path, exp_format).parse(
        edf_file_path, detection_algorithm, force_best_eye, overwrite, **kwargs)


class TobiiParse(BidsParse):

    def parse(self, file_path, detection_algorithm, force_best_eye, overwrite, **kwargs):
        from pyxations.bids_formatting import find_besteye, EYE_MOVEMENT_DETECTION_DICT, keep_eye
        
        # Convert EDF to ASCII (only if necessary)
        # ascii_file_path = convert_edf_to_ascii(edf_file_path, session_folder_path)
        df = pd.read_csv(file_path, sep="\t")

        df = df.rename(columns={'Eyetracker timestamp': 'tSample'})

        # Convert timestamp so samples and detected events share one absolute ms base 
        # compatible with eyelink and remodnav integration
        df['tSample'] = pd.to_numeric(df['tSample'], errors='coerce')

        # Drop rows without a timestamp: they carry no time and cannot sit on the
        # uniform sampling grid that remodnav's time reconstruction relies on.
        df = df[df['tSample'].notna()].reset_index(drop=True)
        df['tSample'] = df['tSample'] / 1000.0

        # set the invalid samples to NaN as Tobii encodes an untracked
        # eye as 0 across all gaze channels
        gaze_cols = [c for c in df.columns if c.startswith(('Gaze', 'Eyepos', 'PupilDiam'))]
        for c in gaze_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Pick the eye with the most valid (non-(0,0), non-NaN) gaze samples
        def _valid_count(eye_name):
            x, y = df[f'Gaze2d_{eye_name}.x'], df[f'Gaze2d_{eye_name}.y']
            return int((~((x == 0) & (y == 0)) & x.notna() & y.notna()).sum())

        requested_eye = kwargs.get('eye')
        if isinstance(requested_eye, str) and requested_eye.upper() in ('L', 'R'):
            eye = requested_eye.upper()
        elif force_best_eye or (isinstance(requested_eye, str) and requested_eye.lower() == 'best'):
            eye = 'L' if _valid_count('Left') >= _valid_count('Right') else 'R'
        else:
            eye = 'L'
        eye_name = 'Left' if eye == 'L' else 'Right'

        # Preserve every timestamped row but mark tracking-loss samples (encoded by Tobii as gaze (0,0)) of the chosen eye as missing
        invalid = (df[f'Gaze2d_{eye_name}.x'] == 0) & (df[f'Gaze2d_{eye_name}.y'] == 0)
        df.loc[invalid, gaze_cols] = np.nan

        # Expose the chosen eye under the canonical X/Y/Pupil names
        df['X'] = df[f'Gaze2d_{eye_name}.x']
        df['Y'] = df[f'Gaze2d_{eye_name}.y']
        df['Pupil'] = df[f'PupilDiam_{eye_name}']

        dfSample = df.reset_index().rename(columns={"index": "line_number"})

        # Estimate the true sampling rate from the timestamps
        t_ms = dfSample['tSample'].to_numpy()
        sample_rate = (len(t_ms) - 1) / ((t_ms[-1] - t_ms[0]) / 1000.0)

        # Eye movement detect
        eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=self.session_folder_path, samples=dfSample)

        # Provide the detector inputs
        config = {
            'eyes_recorded': eye,
            'eye': eye,
            'pupil_data': dfSample['Pupil'],
        }
        detector_params = set(inspect.signature(eye_movement_detector.run_eye_movement).parameters)
        # Forward detector tunables from kwargs
        config.update({k: v for k, v in kwargs.items()
                       if k in detector_params and k not in ('eye', 'eyes_recorded')})
        self.detection_algorithm = detection_algorithm
        # Detect on the 0-1 screen gaze (Gaze2d) of the chosen eye
        dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(
            sample_rate, config=config, )
        

        # Split into trials
        dfMsg = pd.DataFrame(columns=["timestamp", "message"])
        dfBlink = pd.DataFrame(columns=["tStart", "tEnd", "duration"])

        pre_processing = PreProcessing(dfSample, dfFix, dfSacc, dfBlink, dfMsg, self.session_folder_path)

        # Optionally flag out-of-screen / NaN gaze as 'bad'
        # resolution of the recording screen.
        screen_res = kwargs.get("screen_res")
        if isinstance(screen_res, (tuple, list)) and len(screen_res) == 2:
            pre_processing.set_metadata(screen_width=screen_res[0], screen_height=screen_res[1])
        elif kwargs.get("screen_width") and kwargs.get("screen_height"):
            pre_processing.set_metadata(screen_width=kwargs["screen_width"], screen_height=kwargs["screen_height"])

        run_bad_samples = (
            pre_processing.metadata.screen_width is not None
            and pre_processing.metadata.screen_height is not None
        )
        bad_params = {}
        if run_bad_samples:
            bad_sig = inspect.signature(pre_processing.bad_samples).parameters
            for k in ("screen_height", "screen_width", "mark_nan_as_bad", "inclusive_bounds"):
                if k in kwargs and k in bad_sig:
                    bad_params[k] = kwargs[k]

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
            # No trial markers/times supplied: treat the whole recording as a
            # single trial so downstream trial-based analysis works.
            recipe = {}
            if run_bad_samples:
                recipe["bad_samples"] = bad_params
            recipe["split_all_into_single_trial"] = {"phase_name": "complete_recording"}
            recipe["saccades_direction"] = dir_params
            pre_processing.process(recipe)
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
            recipe = {}
            if run_bad_samples:
                recipe["bad_samples"] = bad_params
            recipe[seg_func_name] = seg_params
            recipe["saccades_direction"] = dir_params
            pre_processing.process(recipe)
        
        dfSample = pre_processing.samples
        dfFix = pre_processing.fixations
        dfSacc = pre_processing.saccades        

        
        
        self.store_dataframes(dfSample, dfFix=dfFix, dfSacc=dfSacc, dfBlink=pre_processing.blinks)

        return df
