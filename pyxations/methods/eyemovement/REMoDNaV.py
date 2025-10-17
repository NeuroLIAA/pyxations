import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from remodnav.clf import EyegazeClassifier
from pyxations.methods.eyemovement.eye_movement_detection import EyeMovementDetection


# TODO: force screen_size, screen_width and screen_distance to be present when calling detect_eye_movements or run_eye_movement_from_samples
class RemodnavDetection(EyeMovementDetection):

    def __init__(self, session_folder_path, samples):
        self.session_folder_path = session_folder_path
        self.out_folder = (session_folder_path / 'remodnav_events')
        self.samples = samples

    def detect_eye_movements(self, min_pursuit_dur:float=10., max_pso_dur:float=0.0, min_fix_dur:float=0.05,
                                 sac_max_vel:float=1000., fix_max_amp:float=1.5, sac_time_thresh:float=0.002,
                                 drop_fix_from_blink:bool=True, screen_size:float=38., screen_width:int=1920, screen_distance:float=60,
                                 savgol_length=0.195):
        
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
        screen_size : float, optional
            Size of the screen in cm (default is 38.0).
        screen_width : int, optional
            Horizontal resolution of the screen in pixels (default is 1920).
        screen_distance : float, optional
            Distance from the screen to the participant's eyes in cm (default is 60).

        Returns
        -------
        fixations : dict
            Dictionary containing DataFrames of detected fixations for both left and right eyes with additional columns for mean x, y positions, and pupil size.
        saccades : dict
            Dictionary containing DataFrames of detected saccades for both left and right eyes.

        Raises
        ------
        ValueError
            If Remodnav detection fails.
        """

        # Move eye data, detections file and image to subject results directory
        self.out_folder.mkdir(parents=True, exist_ok=True)

        # Dictionaries to store fixations and saccades DataFrames
        fixations = []
        saccades = []

        # Divide samples by continuous data chunks
        # If the difference in the column "tSample" from row i to row i+1 is greater than 1000 / sample_rate, then it is a new chunk

        # Logic to find chunks
        chunks_start_indexes = np.where(np.diff(self.samples['tSample']) > (1000 / self.samples['Rate_recorded'][0]))[0] + 1

        chunks = np.zeros(len(self.samples))
        chunks[chunks_start_indexes] = 1
        chunks = np.cumsum(chunks)

        # Divide samples by chunks and apply to each chunk the detection algorithm
        for chunk in np.unique(chunks): 
            chunk_samples = self.samples[chunks == chunk].reset_index(drop=True)
            fixations_chunk, saccades_chunk = self.detect_on_chunk(chunk_samples, min_pursuit_dur, max_pso_dur, min_fix_dur, 
                            sac_max_vel, fix_max_amp, sac_time_thresh, drop_fix_from_blink, screen_size, screen_width, screen_distance,
                            savgol_length)
            fixations.append(fixations_chunk)
            saccades.append(saccades_chunk)

        return pd.concat(fixations).sort_values(by='tEnd', ignore_index=True), pd.concat(saccades).sort_values(by='tEnd', ignore_index=True)



    def run_eye_movement_from_samples(self, sample_rate, x_label='X', y_label='Y', config={}, **kwargs):
        '''
        Recieves a pandas dataframe, a sample rate, and optional configuration
        :param dfSamples: Pandas dataframe including x and y columns
        :param sample_rate: an integer representing the data sample rate.
        :param x_label: X column name
        :param y_label: Y column name
        :param config:
        '''
        starting_time = self.samples['tSample'].min()
        eye_data = {
            'x': self.samples[x_label], 
            'y': self.samples[y_label]
        }
        eye_data = np.rec.fromarrays(list(eye_data.values()), names=list(eye_data.keys()))

        if 'pupil_data' not in config.keys():
            config['pupil_data'] = pd.Series([0]* len(eye_data['x']))
        times = np.arange(stop=len(eye_data['x'])) / sample_rate
        
        return self.run_eye_movement(eye_data['x'], eye_data['y'], sample_rate, 
            times=times, starting_time=starting_time, **config, **kwargs)
        

    def run_eye_movement(self, gazex_data, gazey_data, sample_rate,
             min_pursuit_dur:float=10., max_pso_dur:float=0.0, min_fix_dur:float=0.05,
             min_saccade_duration=0.04,
             sac_max_vel:float=1000., fix_max_amp:float=1.5, sac_time_thresh:float=0.002,
             drop_fix_from_blink:bool=True, screen_size:float=38., screen_width:int=1920, 
             screen_distance:float=60, calib_index=0, savgol_length=0.19,
             eyes_recorded=None, starting_time=None, times=None, pupil_data=None, eye=None):

        # If not pre run data, run
        print(f'\nRunning eye movements detection for {eye} eye...')
        
        # Define data to save to excel file needed to run the saccades detection program Remodnav
        eye_data = {'x':gazex_data, 'y':gazey_data}  # eye_data to numpy record array
        eye_data = np.rec.fromarrays(list(eye_data.values()), names=list(eye_data.keys()))
        
        # Remodnav parameters
        px2deg = math.degrees(math.atan2(.5 * screen_size, screen_distance)) / (.5 * screen_width)  # Pixel to degree conversion
        
        # Run Remodnav not considering pursuit class and min fixations 50 ms
        clf = EyegazeClassifier(px2deg=px2deg,
            sampling_rate=sample_rate,
            min_pursuit_duration=min_pursuit_dur,
            max_pso_duration=max_pso_dur,
            min_fixation_duration=min_fix_dur,
            min_saccade_duration=min_saccade_duration)
        pp = clf.preproc(
            eye_data,
            savgol_length=savgol_length
        )
        
        
        sac_fix = clf(pp, classify_isp=True, sort_events=True)
               
       
        # sac_fix to pandas DataFrame
        sac_fix = pd.DataFrame(sac_fix, columns=['start_time', 'end_time', 'label', 'start_x', 'start_y', 'end_x', 'end_y', 'amp', 'peak_vel', 'med_vel', 'avg_vel'])
        sac_fix['duration'] = sac_fix['end_time'] - sac_fix['start_time']
        sac_fix.rename(columns={'start_time':'onset'}, inplace=True)
        # Get saccades and fixations
        saccades_eye_all = sac_fix.loc[(sac_fix['label'] == 'SACC') | (sac_fix['label'] == 'ISAC')]
        fixations_eye_all = sac_fix.loc[sac_fix['label'] == 'FIXA']
        # Drop saccades and fixations based on conditions
        print(f'Dropping saccades with average vel > {sac_max_vel} deg/s, and fixations with amplitude > {fix_max_amp} deg')
        fixations_eye = fixations_eye_all[fixations_eye_all['amp'] <= fix_max_amp].sort_values(by='start_x', ignore_index=True)
        saccades_eye = saccades_eye_all[saccades_eye_all['peak_vel'] <= sac_max_vel].sort_values(by='start_x', ignore_index=True)
        fixations_eye.drop(columns=['label'], inplace=True)
        saccades_eye.drop(columns=['label'], inplace=True)
        print(f'Kept {len(fixations_eye)} out of {len(fixations_eye_all)} fixations')
        print(f'Kept {len(saccades_eye)} out of {len(saccades_eye_all)} saccades')
        # Variables to save fixations features
        mean_x = []
        mean_y = []
        pupil_size = []
        # Drop when no previous saccade detected in sac_time_thresh
        if drop_fix_from_blink:
            prev_sac = []
            # Identify neighbour saccades to each fixation (considering sac_time_thresh)
            print('Finding previous and next saccades')
            for fix_idx, fixation in tqdm(fixations_eye.iterrows(), total=len(fixations_eye)):
                fix_time = fixation['onset']
                fix_dur = fixation['duration']
                # Previous and next saccades
                try:
                    sac0 = saccades_eye.loc[(saccades_eye['onset'] + saccades_eye['duration'] > fix_time - sac_time_thresh) & (saccades_eye['onset'] + saccades_eye['duration'] < fix_time + sac_time_thresh)].index.values[-1]
                except:
                    sac0 = -1
                prev_sac.append(sac0)
            
            # Add columns
            fixations_eye['prev_sac'] = prev_sac
            fixations_eye.drop(fixations_eye[fixations_eye['prev_sac'] == -1].index, inplace=True)
            print(f'\nKept {len(fixations_eye)} fixations with previous saccade')
            fixations_eye.drop(columns=['prev_sac'], inplace=True)
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
        
        fixations_eye['xAvg'] = mean_x
        fixations_eye['yAvg'] = mean_y
        fixations_eye['pupilAvg'] = pupil_size
        fixations_eye = fixations_eye.astype({'xAvg':float, 'yAvg':float, 'pupilAvg':float})
        fixations_eye['onset'] = fixations_eye['onset'] * 1000 + starting_time  # Convert to ms
        fixations_eye['end_time'] = fixations_eye['end_time'] * 1000 + starting_time  # Convert to ms
        fixations_eye['duration'] = fixations_eye['duration'] * 1000  # Convert to ms
        saccades_eye['onset'] = saccades_eye['onset'] * 1000 + starting_time  # Convert to ms
        saccades_eye['end_time'] = saccades_eye['end_time'] * 1000 + starting_time  # Convert to ms
        saccades_eye['duration'] = saccades_eye['duration'] * 1000  # Convert to ms
        # Rename columns to match samples columns names
        fixations_eye.rename(columns={'start_x':'xStart', 'start_y':'yStart', 'end_x':'xEnd', 'end_y':'yEnd', 'onset':'tStart', 'end_time':'tEnd', 'amp':'ampDeg', 'peak_vel':'vPeak'}, inplace=True)
        saccades_eye.rename(columns={'start_x':'xStart', 'start_y':'yStart', 'end_x':'xEnd', 'end_y':'yEnd', 'onset':'tStart', 'end_time':'tEnd', 'amp':'ampDeg', 'peak_vel':'vPeak'}, inplace=True)
        fixations_eye['Calib_index'] = calib_index
        saccades_eye['Calib_index'] = calib_index
        fixations_eye['Eyes_recorded'] = eyes_recorded
        saccades_eye['Eyes_recorded'] = eyes_recorded
        fixations_eye['Rate_recorded'] = sample_rate
        saccades_eye['Rate_recorded'] = sample_rate
        fixations_eye['eye'] = eye
        saccades_eye['eye'] = eye  # Save to dictionary
        
        return fixations_eye, saccades_eye

    def detect_on_chunk(self, chunk, min_pursuit_dur:float=10., max_pso_dur:float=0.0, min_fix_dur:float=0.05,
                                 sac_max_vel:float=1000., fix_max_amp:float=1.5, sac_time_thresh:float=0.002,
                                 drop_fix_from_blink:bool=True, screen_size:float=38., screen_width:int=1920, screen_distance:float=60,
                                 savgol_length:float=0.19):

        sample_rate = chunk["Rate_recorded"].iloc[0]
        calib_index = chunk["Calib_index"].iloc[0]
        eyes_recorded = chunk["Eyes_recorded"].iloc[0]
        starting_time = chunk["tSample"].iloc[0]

        # Check that the columns Rate_recorded, Calib_index and Eyes_recorded are the same for all the samples in the chunk
        assert (chunk["Rate_recorded"] == sample_rate).all()
        assert (chunk["Calib_index"] == calib_index).all()
        assert (chunk["Eyes_recorded"] == eyes_recorded).all()
        
        times = np.arange(stop=len(chunk)) / sample_rate

        fixations = []
        saccades = []
        
        min_saccade_duration=0.04

        for gazex_data, gazey_data, pupil_data, eye in zip((chunk['LX'], chunk['RX']),
                                                        (chunk['LY'], chunk['RY']),
                                                        (chunk['LPupil'], chunk['RPupil']),
                                                        ('L', 'R')):

            fixations_eye, saccades_eye = self.run_eye_movement(
                gazex_data, gazey_data, sample_rate,
                min_pursuit_dur, max_pso_dur, min_fix_dur, 
                min_saccade_duration,
                sac_max_vel, fix_max_amp, sac_time_thresh,
                drop_fix_from_blink, screen_size, screen_width, screen_distance, 
                calib_index, savgol_length, eyes_recorded, starting_time, times, pupil_data, eye)


            fixations.append(fixations_eye)
            saccades.append(saccades_eye)

        return pd.concat(fixations), pd.concat(saccades)

