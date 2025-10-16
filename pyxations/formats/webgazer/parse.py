'''
Created on Oct 31, 2024

@author: placiana
'''
import pandas as pd
import json
from pyxations.formats.generic import BidsParse
from pyxations.pre_processing import PreProcessing
import inspect


def process_session(eye_tracking_data_path, detection_algorithm, session_folder_path, overwrite, exp_format, **kwargs):
    csv_files = [file for file in eye_tracking_data_path.iterdir() if file.suffix.lower() == '.csv']
    if len(csv_files) > 1:
        print(f"More than one csv file found in {eye_tracking_data_path}. Skipping folder.")
        return
    edf_file_path = csv_files[0]
    (session_folder_path / 'events').mkdir(parents=True, exist_ok=True)
    
    WebGazerParse(session_folder_path, exp_format).parse(edf_file_path, detection_algorithm,
                         overwrite, **kwargs)


class WebGazerParse(BidsParse):

    def parse(self, file_path, detection_algorithm, overwrite, **kwargs):
        # Convert EDF to ASCII (only if necessary)
        # ascii_file_path = convert_edf_to_ascii(edf_file_path, session_folder_path)
        from pyxations.bids_formatting import find_besteye, EYE_MOVEMENT_DETECTION_DICT, keep_eye
        df = pd.read_csv(file_path)
        
        session_folder_path = self.session_folder_path
        
        df['line_number'] = df.index
        # columna importante 
        dfSample = df[df['webgazer_data'].notna()].reset_index()
        dfSample['data'] = dfSample['webgazer_data'].apply(json.loads)
        df_exploded = dfSample.explode('data')
        
        df_exploded['data'] = df_exploded.apply(
            lambda row: {**row['data'], 't_acum': row['data']['t'] + row['time_elapsed']}, axis=1
        )
        
        expanded_df = pd.json_normalize(df_exploded['data'])
        expanded_df = pd.concat(
        [df_exploded[['line_number', 'trial_index', 'time_elapsed']].reset_index(drop=True),  # Keep desired columns
         expanded_df],                    # Expand the data
        axis=1
        )
        
        dfSample = expanded_df.rename(columns={"x": "X", "y": "Y", 't': 'tSample'})
    
        # Calibration messages    
        dfCalib = df[df['rastoc-type'] == 'calibration-stimulus']
    
        # Eye movement
        eye_movement_detector = EYE_MOVEMENT_DETECTION_DICT[detection_algorithm](session_folder_path=session_folder_path,samples=dfSample)
        config = {
            'savgol_length': 0.195,
            'max_pso_dur': 0.1
        }
        
        dfFix, dfSacc = eye_movement_detector.run_eye_movement_from_samples(30, config=config)

        dfBlink = pd.DataFrame(columns=dfSample.columns)
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


    

        self.detection_algorithm = detection_algorithm

        pp = pre_processing
        self.store_dataframes(pp.samples, dfCalib, pp.fixations, pp.saccades, pp.blinks, pp.user_messages)
            


def get_samples_for_remodnav(df_sample, rate_recorded=60, r_pupil=1, l_pupil=1):
    df_sample['Rate_recorded'] = rate_recorded
    df_sample['LX'] = df_sample['X'] 
    df_sample['RX'] = df_sample['X']
    df_sample['LY'] = df_sample['Y']
    df_sample['RY'] = df_sample['Y']
    df_sample['LPupil'] = l_pupil
    df_sample['RPupil'] = r_pupil
    df_sample['Calib_index'] = 1
    df_sample['Eyes_recorded'] = 'LR'

    return df_sample
        
        