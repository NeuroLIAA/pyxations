import unittest
import os
from pyxations import RemodnavDetection
import pandas as pd
from pathlib import Path
import pyxations.formats.webgazer.parse as webgazer_parse



current_path = Path(__file__).resolve()
current_folder = current_path.parent.parent

current_path = Path(__file__).resolve()
data_folder = os.path.join(current_path.parent, 'data')


class TestRemodnav(unittest.TestCase):

    # def test_remodnav(self):
    #     path_to_derivatives = os.path.join(current_folder, "example_dataset_derivatives")
    #
    #     # Columns required: 'LX', 'RX', 'LY', 'RY', 'LPupil','RPupil', 'Rate_recorded'
    #     df_samples = pd.read_hdf(os.path.join(path_to_derivatives, 'sub-0001/ses-second/samples.hdf5'))
    #
    #     remodnav = RemodnavDetection(
    #         Path(os.path.join(path_to_derivatives, 'sub-0001', 'ses-second')),
    #         df_samples
    #     )
    #
    #     result = remodnav.detect_eye_movements()
    #
    #     self.assertTrue(result)
        

    def test_remodnav_with_eyelink_dataset(self):
        session_folder_path = os.path.join(data_folder, "derivatives", "example_dataset_derivatives", 'sub-0001/ses-second')
        dfSamples = pd.read_feather(os.path.join(session_folder_path, 'samples.feather'))
        dfSamples = dfSamples[dfSamples['trial_number'] == 1]

        sample_rate = 1000

        remodnav = RemodnavDetection(
            Path(session_folder_path),
            dfSamples
        )

        config = {
            'savgol_length': 0.195
        }

        saccades, fixations = remodnav.run_eye_movement_from_samples(dfSamples, sample_rate, config=config)

        self.assertFalse(saccades.empty)
        self.assertFalse(fixations.empty)


    def test_webgazer_remodnav(self):

        path_to_derivatives = os.path.join(data_folder, "derivatives", "antisacadas_dataset_derivatives")
        
        # Columns required: 'LX', 'RX', 'LY', 'RY', 'LPupil','RPupil', 'Rate_recorded', 'Calib_index', ''Eyes_recorded'
        df_samples = pd.read_feather(os.path.join(path_to_derivatives, 'sub-0001/ses-antisacadas/samples.feather'))
        df_samples = webgazer_parse.get_samples_for_remodnav(df_samples)
        
        remodnav = RemodnavDetection(
            Path(os.path.join(path_to_derivatives, 'sub-0001', 'ses-antisacadas')),
            df_samples
        )
        
        config = {
            'savgol_length': 0.195
        }
        
        result = remodnav.run_eye_movement_from_samples(df_samples, 30, config=config)
        
        self.assertTrue(result)

    def test_tobii_remodnav(self):
        df_samples = pd.read_hdf(os.path.join(data_folder, "derivatives", 'tobii_dataset_derivatives', 'sub-0001/ses-sceneviewing/samples.hdf5'))
        
        remodnav = RemodnavDetection(
            Path(current_folder),
            df_samples
        )
        
        config = {
            'savgol_length': 0.195,
            'eyes_recorded': 'L',
            'eye': 'L',
            'pupil_data': df_samples['PupilDiam_Left']
        }
        
        saccades, fixations = remodnav.run_eye_movement_from_samples(
            df_samples,  60,
            x_label='Gaze3d_Left.x', y_label='Gaze3d_Left.y', config=config)
        
        self.assertFalse(saccades.empty)
        self.assertFalse(fixations.empty)

        


if __name__ == "__main__":
    unittest.main()
