import numpy as np
import pandas as pd


class PreProcessing:
    def __init__(self, samples, fixations,saccades,blinks, user_messages):
        self.samples = samples
        self.fixations = fixations
        self.saccades = saccades
        self.blinks = blinks
        self.user_messages = user_messages

    def split_all_into_trials(self,start_times: dict[list[int]], end_times: dict[list[int]],trial_labels:dict[list[str]] = None):
        self.split_into_trials(self.samples,start_times,end_times,trial_labels)
        self.split_into_trials(self.fixations,start_times,end_times,trial_labels)
        self.split_into_trials(self.saccades,start_times,end_times,trial_labels)
        self.split_into_trials(self.blinks,start_times,end_times,trial_labels)
        
    def split_all_into_trials_by_msgs(self, start_msgs: dict[list[str]], end_msgs: dict[list[str]],trial_labels:dict[list[str]] = None):
        start_times = self.get_timestamps_from_messages(start_msgs)
        end_times = self.get_timestamps_from_messages(end_msgs)
        self.split_into_trials(self.samples,start_times,end_times,trial_labels)
        self.split_into_trials(self.fixations,start_times,end_times,trial_labels)
        self.split_into_trials(self.saccades,start_times,end_times,trial_labels)
        self.split_into_trials(self.blinks,start_times,end_times,trial_labels)

    def split_all_into_trials_by_durations(self, start_msgs: dict[list[str]], durations: dict[list[int]],trial_labels:dict[list[str]] = None):
        # Get the start times for each trial
        start_times = self.get_timestamps_from_messages(start_msgs)

        end_times = {}
        for key in durations.keys():
            if len(durations[key]) < len(start_times[key]):
                raise ValueError("The amount of durations provided is less than the amount of start messages.")
            end_times[key] = ([(start_time + duration) for start_time, duration in zip(start_times[key],durations[key])])

        self.split_into_trials(self.samples,start_times,end_times,trial_labels)
        self.split_into_trials(self.fixations,start_times,end_times,trial_labels)
        self.split_into_trials(self.saccades,start_times,end_times,trial_labels)
        self.split_into_trials(self.blinks,start_times,end_times,trial_labels)



    def process(self,functions_and_params:dict):
        for function,params in functions_and_params.items():
            getattr(self,function)(**params)


    def bad_samples(self, screen_height:int, screen_width:int):
        """
        Classifies samples as 'bad' if they fall outside the screen boundaries.

        This function processes a DataFrame containing gaze samples data,
        and classifying samples as 'bad' if they fall outside the screen boundaries.
        If there are NaN values in a given row, they are skipped.

        Parameters:
        samples (pd.DataFrame): DataFrame containing gaze samples data with the following columns:
                                'LX', 'LY', 'RX', 'RY'.
        height (int): Height of the screen in pixels.
        width (int): Width of the screen in pixels.

        """
        for df in [self.samples,self.fixations,self.saccades]:            
            columns = [cols for cols in df.columns if cols in ['LX', 'LY', 'RX', 'RY', 'X', 'Y','xStart', 'xEnd', 'yStart', 'yEnd','xAvg','yAvg']]
            width_columns = [cols for cols in df.columns if cols in ['LX', 'RX', 'X','xStart', 'xEnd','xAvg']]
            height_columns = [cols for cols in df.columns if cols in ['LY', 'RY', 'Y','yStart', 'yEnd','yAvg']]


            df['bad'] = (df[columns] < 0).any(axis=1)

            # Add width filter to the filter list
            df['bad'] = df['bad'] | (df[width_columns] > screen_width).any(axis=1)
            
            # Add height filter to the filter list
            df['bad'] = df['bad'] | (df[height_columns] > screen_height).any(axis=1)



    def saccades_direction(self):
        """
        Classifies saccades into directional categories based on their start and end coordinates.

        This function processes a DataFrame containing saccade data, filling missing values,
        converting coordinate columns to float, computing saccade amplitudes in the x and y
        directions, mapping these to the complex plane, calculating the saccade angles in degrees,
        and finally classifying the direction of each saccade as 'right', 'left', 'up', or 'down'.

        Parameters:
        saccades (pd.DataFrame): DataFrame containing saccade data with the following columns:
                                'xStart', 'xEnd', 'yStart', 'yEnd'.

        """

        # Saccades amplitude in x and y
        x_dif = self.saccades['xEnd'] - self.saccades['xStart']
        y_dif = self.saccades['yEnd'] - self.saccades['yStart']
        
        # Take to complex plane
        z = x_dif + 1j * y_dif

        # Saccades degrees
        self.saccades['deg'] = np.angle(z, deg=True)

        # Classify in right / left / up / down
        self.saccades['dir'] = [''] * len(self.saccades)

        self.saccades.loc[(-15 < self.saccades['deg']) & (self.saccades['deg'] < 15), 'dir'] = 'right'
        self.saccades.loc[(75 < self.saccades['deg']) & (self.saccades['deg'] < 105), 'dir'] = 'down'
        self.saccades.loc[(165 < self.saccades['deg']) | (self.saccades['deg'] < -165), 'dir'] = 'left'
        self.saccades.loc[(-105 < self.saccades['deg']) & (self.saccades['deg'] < -75), 'dir'] = 'up'
    
    
    def get_timestamps_from_messages(self, messages_dict: dict[list[str]]):
        """
        Get the timestamps for a list of messages from the user messages DataFrame.
        The idea is to get the rows which have any of the messages in the list as a substring in the value of their 'message' column.

        Parameters:
        user_messages (pd.DataFrame): DataFrame containing user messages data with the following columns:
                                        'timestamp', 'message'.
        messages (dict[list[str]]): Dict of the name of the epochs as keys and value of list of strings to identify the messages.

        Returns:
        list[int]: List of timestamps for the messages.
        """
        timestamps_dict = {}
        for key,messages in messages_dict.items():
            # Get the timestamps for the messages and the samples rates
            timestamps_and_rates = self.user_messages[self.user_messages['message'].str.contains('|'.join(messages))][['timestamp','Rate_recorded']].sort_values(by='timestamp')
            timestamps = timestamps_and_rates['timestamp'].tolist()


            # Raise exception if no timestamps are found
            if len(timestamps) == 0:
                raise ValueError("No timestamps found for the messages: {}, in the session path: {}".format(messages))
            timestamps_dict[key] = timestamps

        return timestamps_dict
    
    def split_into_trials(self,data:pd.DataFrame, start_times: dict[list[int]], end_times: dict[list[int]],trial_labels:dict[list[str]] = None):
        data['phase'] = [''] * len(data)
        data['trial_number'] = [-1] * len(data)
        for key in start_times.keys():
            start_times_list = start_times[key]
            end_times_list = end_times[key]
            trial_labels_list = trial_labels[key] if trial_labels and key in trial_labels else None
            # It is somewhat common that the last trial is not closed, so we will discard starting times that are greater than the last ending time
            start_times_list = [start_time for start_time in start_times_list if start_time < end_times_list[-1]]

            # Check that both lists have the same length
            if len(start_times_list) != len(end_times_list):
                data.drop('trial_number', axis=1, inplace=True)
                data.drop('phase', axis=1, inplace=True)
                raise ValueError("start_times and end_times for {} must have the same length, but they have lengths {} and {} respectively.".format(key,len(start_times_list), len(end_times_list)))

            # Check that the length of ordered_trials_ids is the same as the number of trials
            if trial_labels_list and len(trial_labels_list) != len(start_times_list):
                data.drop('trial_number', axis=1, inplace=True)
                data.drop('phase', axis=1, inplace=True)
                raise ValueError("The amount of computed trials is {} while the amount of ordered trial ids is {} for key {}.".format(len(start_times_list), len(trial_labels_list)), key)



            # Divide in trials according to start_times_list and end_times_list
            if 'tSample' in data.columns:
                for i in range(len(start_times_list)):
                    data.loc[(data['tSample'] >= start_times_list[i]) & (data['tSample'] <= end_times_list[i]), 'trial_number'] = i
                    data.loc[(data['tSample'] >= start_times_list[i]) & (data['tSample'] <= end_times_list[i]), 'phase'] = key
            elif 'tStart' in data.columns and 'tEnd' in data.columns:
                for i in range(len(start_times_list)):
                    data.loc[(data['tStart'] >= start_times_list[i]) & (data['tEnd'] <= end_times_list[i]), 'trial_number'] = i
                    data.loc[(data['tStart'] >= start_times_list[i]) & (data['tEnd'] <= end_times_list[i]), 'phase'] = key
            else:
                data.drop('trial_number', axis=1, inplace=True)
                data.drop('phase', axis=1, inplace=True)
                raise ValueError("The DataFrame must contain either the 'tSample' column or the 'tStart' and 'tEnd' columns.")

            if trial_labels_list:
                data['trial_label'] = [''] * len(data)
                for i in range(len(start_times_list)):
                    data.loc[data['trial_number'] == i, 'trial_label'] = trial_labels_list[i]