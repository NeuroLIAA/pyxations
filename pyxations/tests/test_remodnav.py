import unittest
import os
from pyxations import RemodnavDetection
import pandas as pd
from pathlib import Path
import pyxations.formats.webgazer.parse as webgazer_parse
from remodnav.clf import EyegazeClassifier
import numpy as np


class TestRemodnav(unittest.TestCase):

    def test_remodnav(self):
        return None
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        path_to_derivatives = os.path.join(current_folder, "example_dataset_derivatives")
        
        # Columns required: 'LX', 'RX', 'LY', 'RY', 'LPupil','RPupil', 'Rate_recorded'
        df_samples = pd.read_hdf(os.path.join(path_to_derivatives, 'sub-0001/ses-second/samples.hdf5'))
        
        remodnav = RemodnavDetection(
            Path(os.path.join(path_to_derivatives, 'sub-0001', 'ses-second')),
            df_samples
        )
        
        result = remodnav.detect_eye_movements()
        
        self.assertTrue(result)
        

    def test_remodnav_with_eyelink_dataset(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        session_folder_path = os.path.join(current_folder, "example_dataset_derivatives", 'sub-0001/ses-second')
        dfSamples = pd.read_hdf(os.path.join(session_folder_path, 'samples.hdf5'))
        dfSamples = dfSamples[dfSamples['trial_number'] == 1]

        eye_data = {
            'x': dfSamples['X'], 
            'y': dfSamples['Y']
        }
        
        eye_data = np.rec.fromarrays(list(eye_data.values()), names=list(eye_data.keys()))
        
        params = {'px2deg': 0.018303394560752525, 'sampling_rate': 1000.0, 'min_pursuit_duration': 10.0, 'max_pso_duration': 0.0, 'min_fixation_duration': 0.05}
        
        sample_rate = 1000
        times = np.arange(stop=len(eye_data['x'])) / sample_rate

        remodnav = RemodnavDetection(
            Path(session_folder_path),
            dfSamples
        )
        
        remodnav.run_eye_movement(eye_data['x'], eye_data['y'], sample_rate, times=times)



    def test_webgazer_remodnav(self):
        return None
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        path_to_derivatives = os.path.join(current_folder, "antisacadas_dataset_derivatives")
        
        # Columns required: 'LX', 'RX', 'LY', 'RY', 'LPupil','RPupil', 'Rate_recorded', 'Calib_index', ''Eyes_recorded'
        df_samples = pd.read_hdf(os.path.join(path_to_derivatives, 'sub-0001/ses-antisacadas/samples.hdf5'))
        df_samples = webgazer_parse.get_samples_for_remodnav(df_samples)
        
        pass 
        remodnav = RemodnavDetection(
            Path(os.path.join(path_to_derivatives, 'sub-0001', 'ses-antisacadas')),
            df_samples
        )
        
        result = remodnav.detect_eye_movements()
        
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
