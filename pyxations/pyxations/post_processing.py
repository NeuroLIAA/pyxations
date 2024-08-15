# no se me ocurrió otro nombre para el archivo xd

import numpy as np
import pandas as pd


def saccades_direction(saccades):
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

    # Fill '.' values with 0
    # This should't be happening. There must be an error in parsing the edf because no saccade should have missing data in the start or end coordinates
    saccades[['xStart', 'xEnd', 'yStart', 'yEnd']] = saccades[['xStart', 'xEnd', 'yStart', 'yEnd']].replace('.', 0)

    # Convert start and end columns to float
    saccades[['xStart', 'xEnd', 'yStart', 'yEnd']] = saccades[['xStart', 'xEnd', 'yStart', 'yEnd']].astype(float)

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


    return saccades

def get_timestamps_from_messages(user_messages:pd.DataFrame, messages: list[str]):
    """
    Get the timestamps for a list of messages from the user messages DataFrame.
    The idea is to get the rows which have any of the messages in the list as a substring in the value of their 'text' column.

    Parameters:
    user_messages (pd.DataFrame): DataFrame containing user messages data with the following columns:
                                    'time', 'text'.
    messages (list[str]): List of strings to identify the messages.

    Returns:
    list[float]: List of timestamps for the messages.
    """
    
    # Get the timestamps for the messages
    timestamps = user_messages[user_messages['text'].str.contains('|'.join(messages))]['time']

    return timestamps

def split_into_trials(samples:pd.Dataframe,ordered_trials_ids:list, user_messages:pd.Dataframe=None,start_msgs: list[str]=None, end_msgs: list[str]=None,duration: float=None, start_times: list[float]=None, end_times: list[float]=None):
    """
    There are three ways of splitting the samples into trials:
    1) Using the start and end messages.
    2) Using the start messages and a fixed duration.
    3) Using the start and end times.

    Parameters:
    samples (pd.DataFrame): DataFrame containing samples data with the following columns:
                             'tSample', 'x', 'y', 'trial'.
    ordered_trials_ids (list): List of ordered trial ids.
    user_messages (pd.DataFrame): DataFrame containing user messages data with the following columns:
                                  'time', 'text'.
    start_msgs (list[str]): List of strings to identify the start of a trial.
    end_msgs (list[str]): List of strings to identify the end of a trial.
    duration (float): Duration of each trial.
    start_times (list[float]): List of start times for each trial.
    end_times (list[float]): List of end times for each trial.

    In every case the time is measured based on the sample rate of the eye tracker.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column:
                  - 'trial': The trial id for each sample.
    """


    # If start_msgs and end_msgs are provided, use them to split the samples
    if start_msgs is not None and end_msgs is not None and user_messages is not None:
        # Get the start and end times for each trial
        start_times = get_timestamps_from_messages(user_messages, start_msgs)
        end_times = get_timestamps_from_messages(user_messages, end_msgs)

    # If start_msgs and duration are provided, use them to split the samples
    elif start_msgs is not None and duration is not None and user_messages is not None:
        # Get the start times for each trial
        start_times = get_timestamps_from_messages(user_messages, start_msgs)
        # Calculate the end times for each trial
        end_times = start_times + duration

    # If start_times and end_times are provided, use them to split the samples
    elif start_times is not None and end_times is not None:
        pass
    # If none of the above conditions are met, return None
    else:
        return None

    # Check that both lists have the same length
    if len(start_times) != len(end_times):
        raise ValueError("start_times and end_times must have the same length, but they have lengths {} and {} respectively.".format(len(start_times), len(end_times)))

    # Check that the length of ordered_trials_ids is the same as the number of trials
    if len(ordered_trials_ids) != len(start_times):
        raise ValueError("The amount of computed trials is {} while the amount of ordered trial ids is {}.".format(len(start_times), len(ordered_trials_ids)))

    # Create a list of trial ids for each sample
    samples['trial'] = [np.nan] * len(samples)

    # Divide in trials according to start_times and end_times
    for i in range(len(start_times)):
        samples.loc[(samples['tSample'] >= start_times[i]) & (samples['tSample'] <= end_times[i]), 'trial'] = ordered_trials_ids[i]

    return samples