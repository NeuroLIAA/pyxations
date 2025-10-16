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


    GazePointParse(session_folder_path, exp_format).parse(edf_file_path, detection_algorithm, overwrite, **kwargs)

class GazePointParse(BidsParse):

    def parse(self, file_path, detection_algorithm, overwrite, **kwargs):
        from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT
        
        df = pd.read_csv(file_path)
        
        dfSample = df.reset_index().rename(columns={
            "index": "line_number", 
            "TIME": "tSample",
            "BPOGX": "X",
            "BPOGY": "Y",
            "LPD": "Pupil"
        })
        
        dfBlink = df[df['BKDUR'] > 0].reset_index().rename(columns={
            "index": "line_number", 
            "TIME": "tEnd",
            "BKDUR": "duration"
        })
        dfBlink['tStart'] = dfBlink['tEnd'] - dfBlink['duration']

        eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=self.session_folder_path,samples=dfSample)
        config = {
            'savgol_length': 0.19,
            'max_pso_dur': 0.4
        }
        self.detection_algorithm = detection_algorithm
        dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(60, config=config, eye='Best')
        
        #placeholder
        dfMsg = pd.DataFrame(columns=dfSample.columns)

        pre_processing = PreProcessing(dfSample, dfFix, dfSacc, dfBlink, dfMsg, self.session_folder_path)

        # ---- Decide which trialing API to use ----
        prefer_durations = kwargs.get("prefer_durations", False)

        have_explicit_times = ("start_times" in kwargs) and ("end_times" in kwargs)
        have_durations     = ("start_msgs" in kwargs) and ("durations" in kwargs)
        have_message_times = ("start_msgs" in kwargs) and ("end_msgs" in kwargs)

        if not (have_explicit_times or have_durations or have_message_times):
            print(
                "Skipping preprocessing: not enough parameters for trial segmentation "
                "(need (start_times & end_times) or (start_msgs & durations) or (start_msgs & end_msgs))."
            )
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
                seg_func_name: seg_params
            })
        dfSample = pre_processing.samples
        dfBlink = pre_processing.blinks
        dfFix = pre_processing.fixations
        dfSacc = pre_processing.saccades
        
        self.store_dataframes(dfSample, dfBlink=dfBlink, dfFix=dfFix, dfSacc=dfSacc)

    