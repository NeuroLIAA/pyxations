# no se me ocurrió otro nombre para el archivo xd

import numpy as np


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