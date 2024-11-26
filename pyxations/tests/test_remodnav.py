import unittest
import os
from pyxations import RemodnavDetection
import pandas as pd
from pathlib import Path
import pyxations.formats.webgazer.parse as webgazer_parse
class TestRemodnav(unittest.TestCase):
    def test_remodnav(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        path_to_derivatives = os.path.join(current_folder, "example_dataset_derivatives")
        
        # Columns required: 'LX', 'RX', 'LY', 'RY', 'LPupil','RPupil', 'Rate_recorded'
        df_samples = pd.read_hdf(os.path.join(path_to_derivatives,'sub-0001/ses-second/samples.hdf5'))
        
        remodnav = RemodnavDetection(
            Path(os.path.join(path_to_derivatives,'sub-0001','ses-second')),
            df_samples
        )
        
        result = remodnav.detect_eye_movements()
        
        self.assertTrue(result)


    def test_webgazer_remodnav(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        path_to_derivatives = os.path.join(current_folder, "antisacadas_dataset_derivatives")
        
        # Columns required: 'LX', 'RX', 'LY', 'RY', 'LPupil','RPupil', 'Rate_recorded', 'Calib_index', ''Eyes_recorded'
        df_samples = pd.read_hdf(os.path.join(path_to_derivatives,'sub-0001/ses-antisacadas/samples.hdf5'))
        df_samples = webgazer_parse.get_samples_for_remodnav(df_samples)
        
        remodnav = RemodnavDetection(
            Path(os.path.join(path_to_derivatives,'sub-0001','ses-antisacadas')),
            df_samples
        )
        
        result = remodnav.detect_eye_movements()
        
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()