# no se me ocurrió otro nombre para el archivo xd

import numpy as np
import pandas as pd
from os import path

class PostProcessing:
    def __init__(self, session_folder_path,events_detection_algorithm):
        self.session_folder_path = session_folder_path
        self.events_detection_folder = events_detection_algorithm+'_events'

    def bad_samples(self,samples:pd.DataFrame, height:int, width:int):
        """
        Classifies samples as 'bad' if they fall outside the screen boundaries.

        This function processes a DataFrame containing gaze samples data, filling missing values,
        and classifying samples as 'bad' if they fall outside the screen boundaries.

        Parameters:
        samples (pd.DataFrame): DataFrame containing gaze samples data with the following columns:
                                'LX', 'LY', 'RX', 'RY'.
        height (int): Height of the screen in pixels.
        width (int): Width of the screen in pixels.

        Returns:
        pd.DataFrame: The original DataFrame with an additional column:
                    - 'bad': Whether the sample is 'bad' or not.
        """

        # Fill '.' values with 0
        samples[['LX', 'LY', 'RX', 'RY']] = samples[['LX', 'LY', 'RX', 'RY']].replace('.', 0)

        # Convert columns to float
        samples[['LX', 'LY', 'RX', 'RY']] = samples[['LX', 'LY', 'RX', 'RY']].astype(float)

        # Classify samples as 'bad' if they fall outside the screen boundaries
        samples['bad'] = (samples['LX'] < 0) | (samples['LY'] < 0) | (samples['RX'] < 0) | (samples['RY'] < 0) | (samples['LX'] > width) | (samples['LY'] > height) | (samples['RX'] > width) | (samples['RY'] > height)

        samples.to_hdf(path_or_buf=path.join(self.session_folder_path, "samples.hdf5"), key='samples', mode='w')
        return samples


    def saccades_direction(self,saccades:pd.DataFrame):
        """
        Classifies saccades into directional categories based on their start and end coordinates.

        This function processes a DataFrame containing saccade data, filling missing values,
        converting coordinate columns to float, computing saccade amplitudes in the x and y
        directions, mapping these to the complex plane, calculating the saccade angles in degrees,
        and finally classifying the direction of each saccade as 'right', 'left', 'up', or 'down'.

        Parameters:
        saccades (pd.DataFrame): DataFrame containing saccade data with the following columns:
                                'xStart', 'xEnd', 'yStart', 'yEnd'.

        Returns:
        pd.DataFrame: The original DataFrame with additional columns:
                    - 'deg': The angle of each saccade in degrees.
                    - 'dir': The direction of each saccade ('right', 'left', 'up', 'down').
        """

        # Saccades amplitude in x and y
        x_dif = saccades['xEnd'] - saccades['xStart']
        y_dif = saccades['yEnd'] - saccades['yStart']
        
        # Take to complex plane
        z = x_dif + 1j * y_dif

        # Saccades degrees
        saccades['deg'] = np.angle(z, deg=True)

        # Classify in right / left / up / down
        saccades['dir'] = [''] * len(saccades)

        saccades.loc[(-15 < saccades['deg']) & (saccades['deg'] < 15), 'dir'] = 'right'
        saccades.loc[(75 < saccades['deg']) & (saccades['deg'] < 105), 'dir'] = 'down'
        saccades.loc[(165 < saccades['deg']) | (saccades['deg'] < -165), 'dir'] = 'left'
        saccades.loc[(-105 < saccades['deg']) & (saccades['deg'] < -75), 'dir'] = 'up'
    
        saccades.to_hdf(path_or_buf=path.join(self.session_folder_path,self.events_detection_folder, "sacc.hdf5"), key="sacc", mode='w')
        return saccades
    
    def get_timestamps_from_messages(self, user_messages:pd.DataFrame, messages: list[str]):
        """
        Get the timestamps for a list of messages from the user messages DataFrame.
        The idea is to get the rows which have any of the messages in the list as a substring in the value of their 'text' column.

        Parameters:
        user_messages (pd.DataFrame): DataFrame containing user messages data with the following columns:
                                        'time', 'text'.
        messages (list[str]): List of strings to identify the messages.

        Returns:
        list[int]: List of timestamps for the messages.
        """
        
        # Get the timestamps for the messages
        timestamps = user_messages[user_messages['text'].str.contains('|'.join(messages))]['time'].to_numpy(dtype=int)

        # Raise exception if no timestamps are found
        if len(timestamps) == 0:
            raise ValueError("No timestamps found for the messages: {}, in the session path: {}".format(messages, self.session_folder_path))

        return list(timestamps)

    def split_into_trials(self,data:pd.DataFrame,filename:str,trial_labels:list[str] = None, user_messages:pd.DataFrame=None,start_msgs: list[str]=None, end_msgs: list[str]=None,duration: float=None, start_times: list[int]=None, end_times: list[int]=None):
        """
        There are three ways of splitting the samples into trials:
        1) Using the start and end messages.
        2) Using the start messages and a fixed duration.
        3) Using the start and end times.

        Parameters:
        data (pd.DataFrame): DataFrame that must contain either the 'tSample' column or the 'tStart' and 'tEnd' columns.
        trial_labels (list[str]): List of trial labels to assign to each trial.
        user_messages (pd.DataFrame): DataFrame containing user messages data with the following columns:
                                    'time', 'text'.
        start_msgs (list[str]): List of strings to identify the start of a trial.
        end_msgs (list[str]): List of strings to identify the end of a trial.
        duration (float): Duration of each trial in seconds.
        start_times (list[int]): List of start times for each trial.
        end_times (list[int]): List of end times for each trial.

        In every case the time is measured based on the sample rate of the eye tracker.

        Returns:
        pd.DataFrame: The original DataFrame with an additional column:
                    - 'trial': The trial id for each sample.
        """

        # TODO: Turn duration (in seconds) to duration (in samples) using the sample rate of the eye tracker
        # If start_msgs and end_msgs are provided, use them to split the samples
        if start_msgs is not None and end_msgs is not None and user_messages is not None:
            # Get the start and end times for each trial
            start_times = self.get_timestamps_from_messages(user_messages, start_msgs)
            end_times = self.get_timestamps_from_messages(user_messages, end_msgs)

        # If start_msgs and duration are provided, use them to split the samples
        elif start_msgs is not None and duration is not None and user_messages is not None:
            # Get the start times for each trial
            start_times = self.get_timestamps_from_messages(user_messages, start_msgs)
            # Calculate the end times for each trial
            end_times = start_times + duration

        # If start_times and end_times are provided, use them to split the samples
        elif start_times is not None and end_times is not None:
            pass
        # If none of the above conditions are met, raise an exception
        else:
            raise ValueError("Either start_msgs and end_msgs, start_msgs and duration, or start_times and end_times must be provided.")

        # Check that both lists have the same length
        if len(start_times) != len(end_times):
            raise ValueError("start_times and end_times must have the same length, but they have lengths {} and {} respectively.".format(len(start_times), len(end_times)))

        # Check that the length of ordered_trials_ids is the same as the number of trials
        if trial_labels and len(trial_labels) != len(start_times):
            raise ValueError("The amount of computed trials is {} while the amount of ordered trial ids is {}.".format(len(start_times), len(trial_labels)))

        # Create a list of trial ids for each sample
        data['trial_number'] = [-1] * len(data)


        # Divide in trials according to start_times and end_times
        if 'tSample' in data.columns:
            for i in range(len(start_times)):
                data.loc[(data['tSample'] >= start_times[i]) & (data['tSample'] <= end_times[i]), 'trial_number'] = i
        elif 'tStart' in data.columns and 'tEnd' in data.columns:
            for i in range(len(start_times)):
                data.loc[(data['tStart'] >= start_times[i]) & (data['tEnd'] <= end_times[i]), 'trial_number'] = i
        else:
            raise ValueError("The DataFrame must contain either the 'tSample' column or the 'tStart' and 'tEnd' columns.")

        if trial_labels:
            data['trial_label'] = [''] * len(data)
            for i in range(len(start_times)):
                data.loc[data['trial_number'] == i, 'trial_label'] = trial_labels[i]

        file_path = path.join(self.session_folder_path,self.events_detection_folder, filename) if filename != "samples.hdf5" else path.join(self.session_folder_path, filename)
        data.to_hdf(path_or_buf=file_path, key=filename[:-5], mode='w')
        return data
