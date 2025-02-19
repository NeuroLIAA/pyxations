'''
Created on Nov 25, 2024

@author: placiana
'''
import unittest
import os
from pyxations.bids_formatting import compute_derivatives_for_dataset


class Test(unittest.TestCase):


    def test_compute_derivatives_webgazer(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
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