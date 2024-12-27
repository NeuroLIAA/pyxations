import unittest
from pyxations import PreProcessing
import os
import pandas as pd

class TestSplitIntoTrials(unittest.TestCase):
    def test_split_into_trials(self):
        # TODO: Placeholder for future implementation
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        session_path = os.path.join(current_folder,"antisacadas_dataset_derivatives/sub-0001/ses-antisacadas")
        
        samples = pd.read_feather(os.path.join(session_path, 'samples.feather'))
        fixations = saccades = blinks = samples.copy(True) # whatever
        user_messages = None
        
        
        pp = PreProcessing(samples, fixations, saccades, blinks, user_messages, session_path)
        
        start_times = {
            0: [100, 501, 1001],
            1: [2001]
        }
        end_times = {
            0: [500, 1000, 2000],
            1: [2500]
        }
        trial_labels = {0:['first', 'second', 'third'], 1: ['fourth']}
        
        pp.split_all_into_trials(start_times, end_times, trial_labels)
        
        
        self.assertTrue('trial_number' in samples.columns)
        self.assertTrue('phase' in samples.columns)
        self.assertTrue('trial_label' in samples.columns)        
        self.assertTrue(0 in samples.phase.unique())
        self.assertTrue(1 in samples.phase.unique())
        
        self.assertTrue('second' in samples.trial_label.unique())
        self.assertTrue('fourth' in samples.trial_label.unique())

if __name__ == "__main__":
    unittest.main()
