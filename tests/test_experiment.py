'''
Created on Jan 8, 2025

@author: placiana
'''

import unittest
import os
from pyxations.bids_formatting import compute_derivatives_for_dataset
from pyxations.post_processing import Experiment
from pyxations.export import FEATHER_EXPORT
from pathlib import Path


current_path = Path(__file__).resolve()
current_folder = current_path.parent.parent

class Test(unittest.TestCase):

    def test_compute_derivatives_webgazer_remodnav(self):
        bids_dataset_folder = os.path.join(current_folder,"antisacadas_dataset")
        start_times = {
            0: [100, 501, 1001],
        }
        end_times = {
            0: [500, 1000, 2000],
        }
        trial_labels = {0:['first', 'second', 'third'], 1: ['fourth']}
        
        # detection_algorithm = 'remodnav'
        # compute_derivatives_for_dataset(
        #     bids_dataset_folder, 'webgazer', detection_algorithm, overwrite=True, 
        #     exp_format=FEATHER_EXPORT, screen_height=768, screen_width=1024,
        #     start_times=start_times, end_times=end_times, trial_labels=trial_labels)
        #

        experiment = Experiment(bids_dataset_folder)
        experiment.load_data('remodnav')
        print('subjects:', experiment.subjects)






if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()