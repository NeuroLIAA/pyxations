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
        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'webgazer'
        compute_derivatives_for_dataset(bids_dataset_folder, msg_keywords, detection_algorithm, start_msgs=start_msgs, end_msgs=end_msgs)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.hdf5")))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()