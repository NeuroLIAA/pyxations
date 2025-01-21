'''
Created on Jan 8, 2025

@author: placiana
'''

import unittest
import os
from pyxations.bids_formatting import compute_derivatives_for_dataset
from pyxations.post_processing import Experiment
from pyxations.export import FEATHER_EXPORT


class Test(unittest.TestCase):

    def test_compute_derivatives_webgazer_remodnav(self):
        current_folder = '/home/placiana/workspace/pyxations/'
        bids_dataset_folder = os.path.join(current_folder,"antisacadas_dataset")
        start_times = {
            0: [100, 501, 1001],
        }
        end_times = {
            0: [500, 1000, 2000],
        }
        trial_labels = {0:['first', 'second', 'third'], 1: ['fourth']}
        
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(
            bids_dataset_folder, 'webgazer', detection_algorithm, overwrite=True, 
            exp_format=FEATHER_EXPORT, screen_height=768, screen_width=1024,
            start_times=start_times, end_times=end_times, trial_labels=trial_labels)
        
        experiment = Experiment('/home/placiana/workspace/pyxations/antisacadas_dataset')
        experiment.load_data('remodnav')
        print('subjects:', experiment.subjects)


    def not_test_compute_derivatives_eyelink_remodnav(self):
        experiment = Experiment('/home/placiana/workspace/pyxations/example_dataset')
        experiment.load_data('remodnav')
        
        
        self.assertTrue(len(experiment.subjects) == 10)
        
        
        a_subject = experiment.subjects['0001']
        
        session = a_subject.get_session('second')
        
        a_trial = session.get_trial(1)
        
        print(a_subject)





if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()