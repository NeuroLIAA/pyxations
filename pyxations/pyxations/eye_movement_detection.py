import pandas as pd
import numpy as np
import math
import os
import copy
from tqdm import tqdm

class EyeMovementDetection():
    def __init__(self, session_folder_path):
        self.session_folder_path = session_folder_path

    def detect_eye_movements(self,*args,**kwargs):
        pass
    
    def save_eye_movements(self,fixations,saccades,times):
        #TODO: Implement save_eye_movements method, following the guidelines from the eyelink ascii parser.
        pass


class RemodnavDetection(EyeMovementDetection):
    def __init__(self, session_folder_path):
        super().__init__(session_folder_path)
    
    def detect_eye_movements(self,min_pursuit_dur:float=10., max_pso_dur:float=0.0, min_fix_dur:float=0.05, 
                                 sac_max_vel:float=1000., fix_max_amp:float=1.5, sac_time_thresh:float=0.002,
                                 drop_fix_from_blink:bool=True, sfreq:float=1000,
                                 screen_size:float=38., screen_resolution:int=1920, screen_distance:float=60,
                                 out_fname:str='events', out_folder:str='Remodnav_detection/'):
        
        """
        Detects fixations and saccades from eye-tracking data for both left and right eyes using REMoDNaV, a velocity based eye movement event detection algorithm 
        that is based on, but extends the adaptive Nyström & Holmqvist algorithm (Nyström & Holmqvist, 2010).

        Parameters
        ----------
        min_pursuit_dur : float, optional
            Minimum pursuit duration in seconds for Remodnav detection (default is 10.0).
        max_pso_dur : float, optional
            Maximum post-saccadic oscillation duration in seconds for Remodnav detection (default is 0.0 -No PSO events detection-).
        min_fix_dur : float, optional
            Minimum fixation duration in seconds for Remodnav detection (default is 0.05).
        sac_max_vel : float, optional
            Maximum saccade velocity in deg/s (default is 1000.0).
        fix_max_amp : float, optional
            Maximum fixation amplitude in deg (default is 1.5).
        sac_time_thresh : float, optional
            Time threshold in seconds to consider a saccade as neighboring a fixation (default is 0.002).
        drop_fix_from_blink : bool, optional
            Whether to drop fixations that do not have a previous saccade within the time threshold (default is True).
        sfreq : float, optional
            Sampling frequency of the eye-tracking data in Hz (default is 1000).
        screen_size : float, optional
            Size of the screen in cm (default is 38.0).
        screen_resolution : int, optional
            Horizontal resolution of the screen in pixels (default is 1920).
        screen_distance : float, optional
            Distance from the screen to the participant's eyes in cm (default is 60).
        out_fname : str, optional
            Name of the output file (without extension) to save Remodnav detection results (default is 'events').
        out_folder : str, optional
            Name of the folder to save the output files (default is 'Remodnav_detection/').

        Returns
        -------
        fixations : dict
            Dictionary containing DataFrames of detected fixations for both left and right eyes with additional columns for mean x, y positions, and pupil size.
        saccades : dict
            Dictionary containing DataFrames of detected saccades for both left and right eyes.
        times : np.ndarray
            Array of time points corresponding to the samples.

        Raises
        ------
        ValueError
            If Remodnav detection fails.
        """
        out_folder = os.path.join(self.session_folder_path, out_folder)
        # Move eye data, detections file and image to subject results directory
        os.makedirs(out_folder, exist_ok=True)

        samples = pd.read_hdf(path_or_buf=os.path.join(self.session_folder_path, 'samples.hdf5'))
        
        times = np.arange(stop=len(samples) / sfreq, step=1/sfreq)

        # Dictionaries to store fixations and saccades dataframes from each eye
        fixations = {}
        saccades = {}

        # TODO: Adapt to one-eye data
        for gazex_data, gazey_data, pupil_data, eye in zip((samples['LX'], samples['RX']), 
                                                        (samples['LY'], samples['RY']), 
                                                        (samples['LPupil'], samples['RPupil']), 
                                                        ('left', 'right')):

            # If not pre run data, run
            print(f'\nRunning eye movements detection for {eye} eye...')

            # Define data to save to excel file needed to run the saccades detection program Remodnav
            eye_data = {'x': gazex_data, 'y': gazey_data}
            df = pd.DataFrame(eye_data)

            # Remodnav parameters
            eye_samples_fname = f'eye_samples_{eye}.csv'  # eye data file to use as input for Remodnav 
            px2deg = math.degrees(math.atan2(.5 * screen_size, screen_distance)) / (.5 * screen_resolution)  # Pixel to degree conversion
            results_fname = out_fname + f'_{eye}.tsv'  # Output results filename 
            image_fname = out_fname + f'_{eye}.png'  # Output image filename

            # Save csv file
            df.to_csv(eye_samples_fname, sep='\t', header=False, index=False)

            # Run Remodnav not considering pursuit class and min fixations 50 ms
            command = f'remodnav {eye_samples_fname} {results_fname} {px2deg} {sfreq} --min-pursuit-duration {min_pursuit_dur} ' \
                        f'--max-pso-duration {max_pso_dur} --min-fixation-duration {min_fix_dur} --max-vel {sac_max_vel}'
            failed = os.system(command)

            # Move et data file
            os.replace(eye_samples_fname, os.path.join(out_folder, eye_samples_fname))

            # Raise error if events detection with Remodnav failed
            if failed:
                raise ValueError('Remodnav detection failed')

            # Read results file with detections
            sac_fix = pd.read_csv(results_fname, sep='\t')



            # Move results file
            os.replace(results_fname, os.path.join(out_folder,results_fname))
            # Move results image
            os.replace(image_fname, os.path.join(out_folder, image_fname))

            # Get saccades and fixations
            saccades_eye_all = copy.copy(sac_fix.loc[(sac_fix['label'] == 'SACC') | (sac_fix['label'] == 'ISAC')])
            fixations_eye_all = copy.copy(sac_fix.loc[sac_fix['label'] == 'FIXA'])

            # Drop saccades and fixations based on conditions
            print(f'Dropping saccades with average vel > {sac_max_vel} deg/s, and fixations with amplitude > {fix_max_amp} deg')

            fixations_eye = copy.copy(fixations_eye_all[(fixations_eye_all['amp'] <= fix_max_amp)])
            saccades_eye = copy.copy(saccades_eye_all[saccades_eye_all['peak_vel'] <= sac_max_vel])

            print(f'Kept {len(fixations_eye)} out of {len(fixations_eye_all)} fixations')
            print(f'Kept {len(saccades_eye)} out of {len(saccades_eye_all)} saccades')

            # Variables to save fixations features
            mean_x = []
            mean_y = []
            pupil_size = []
            prev_sac = []
            next_sac = []

            # Identify neighbour saccades to each fixation (considering sac_time_thresh)
            print('Finding previous and next saccades')

            for fix_idx, fixation in tqdm(fixations_eye.iterrows(), total=len(fixations_eye)):

                fix_time = fixation['onset']
                fix_dur = fixation['duration']

                # Previous and next saccades
                try:
                    sac0 = saccades_eye.loc[(saccades_eye['onset'] + saccades_eye['duration'] > fix_time - sac_time_thresh) & (
                                saccades_eye['onset'] + saccades_eye['duration'] < fix_time + sac_time_thresh)].index.values[-1]
                except:
                    sac0 = None
                prev_sac.append(sac0)

                try:
                    sac1 = saccades_eye.loc[(saccades_eye['onset'] > fix_time + fix_dur - sac_time_thresh) & (
                                saccades_eye['onset'] < fix_time + fix_dur + sac_time_thresh)].index.values[0]
                except:
                    sac1 = None
                next_sac.append(sac1)


            # Add columns
            fixations_eye['prev_sac'] = prev_sac
            fixations_eye['next_sac'] = next_sac

            # Drop when no previous saccade detected in sac_time_thresh
            if drop_fix_from_blink:
                fixations_eye.dropna(subset=['prev_sac'], inplace=True)
                print(f'\nKept {len(fixations_eye)} fixations with previous saccade')


            # Fixations features
            print('Computing average pupil size, and x and y position')

            for fix_idx, fixation in tqdm(fixations_eye.iterrows(), total=len(fixations_eye)):

                fix_time = fixation['onset']
                fix_dur = fixation['duration']

                # Average pupil size, x and y position
                fix_time_idx = np.where(np.logical_and(times > fix_time, times < fix_time + fix_dur))[0]

                pupil_data_fix = pupil_data[fix_time_idx]
                gazex_data_fix = gazex_data[fix_time_idx]
                gazey_data_fix = gazey_data[fix_time_idx]

                pupil_size.append(np.nanmean(pupil_data_fix))
                mean_x.append(np.nanmean(gazex_data_fix))
                mean_y.append(np.nanmean(gazey_data_fix))

            fixations_eye['mean_x'] = mean_x
            fixations_eye['mean_y'] = mean_y
            fixations_eye['pupil'] = pupil_size
            fixations_eye = fixations_eye.astype({'mean_x': float, 'mean_y': float, 'pupil': float, 'prev_sac': 'Int64', 'next_sac': 'Int64'})

            # Add tEnd column
            fixations_eye['tEnd'] = fixations_eye['onset'] + fixations_eye['duration']
            saccades_eye['tEnd'] = saccades_eye['onset'] + saccades_eye['duration']
            
            # Rename columns to match samples columns names
            fixations_eye.rename(columns={'start_x': 'xStart', 'start_y': 'yStart', 'end_x': 'xEnd', 'end_y': 'yEnd', 'onset': 'tStart'}, inplace=True)
            saccades_eye.rename(columns={'start_x': 'xStart', 'start_y': 'yStart', 'end_x': 'xEnd', 'end_y': 'yEnd', 'onset': 'tStart'}, inplace=True)

            # Save to dictionary
            fixations[eye] = fixations_eye
            saccades[eye] = saccades_eye

        return fixations, saccades, times
    
