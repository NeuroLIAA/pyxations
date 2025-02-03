'''
Created on Nov 25, 2024

@author: placiana
'''
import unittest
import os
from pyxations.bids_formatting import compute_derivatives_for_dataset
from pathlib import Path

current_path = Path(__file__).resolve()
current_folder = current_path.parent.parent

class Test(unittest.TestCase):


    def test_compute_derivatives_webgazer(self):
        bids_dataset_folder = os.path.join(current_folder,"antisacadas_dataset")
        dataset_format = 'webgazer'
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, dataset_format, detection_algorithm,
                                        screen_height=768, screen_width=1024)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.hdf5")))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()