import unittest
from pyxations import PreProcessing
import os
import pandas as pd
from pathlib import Path

current_path = Path(__file__).resolve()
current_folder = current_path.parent.parent

class TestSplitIntoTrials(unittest.TestCase):
    def test_split_into_trials(self):
        # TODO: Placeholder for future implementation
        session_path = os.path.join(current_folder,"antisacadas_dataset_derivatives/sub-0001/ses-antisacadas")
        
        samples = pd.read_feather(os.path.join(session_path, 'samples.feather'))
        fixations = saccades = blinks = samples.copy(True) # whatever
        user_messages = None
        
        
        pp = PreProcessing(samples, fixations, saccades, blinks, user_messages, session_path)
        
        start_times = {
            0: [100, 501, 1001],
            1: [2001, 2600]
        }
        end_times = {
            0: [500, 1000, 2000],
            1: [2500, 2800]
        }
        #trial_labels = {0:['first', 'second', 'third'], 
        #                1: ['first', 'second', 'third']}


        
        trial_labels = ['first', 'second', 'third']
        pp.split_all_into_trials(start_times, end_times, trial_labels)
        
        
        self.assertTrue('trial_number' in samples.columns)
        self.assertTrue('phase' in samples.columns)
        self.assertTrue('trial_label' in samples.columns)        
        self.assertTrue('0' in samples.phase.unique())
        self.assertTrue('1' in samples.phase.unique())
        
        #self.assertTrue('second' in samples.trial_label.unique())
        #self.assertTrue('fourth' in samples.trial_label.unique())


    def test_split_into_trials_time(self):

        session_path = os.path.join(current_folder,"antisacadas_dataset_derivatives/sub-0001/ses-antisacadas")
        
        samples = pd.read_feather(os.path.join(session_path, 'samples.feather'))
        fixations = saccades = blinks = samples.copy(True) # whatever
        user_messages = None
        
        
        pp = PreProcessing(samples, fixations, saccades, blinks, user_messages, session_path)
        
        trials_delimiters = [
            {'start': 100, 'end': 500},
            {'start': 501, 'end': 1000},
            
        ]
        
        start_times = {
            0: [100, 501, 1001],
            1: [2001, 2501]
        }
        end_times = {
            0: [500, 1000, 2000],
            1: [2500, 2700]
        }
        #trial_labels = {0:['first', 'second', 'third'], 
        #                1: ['first', 'second', 'third']}


        
        trial_labels = ['first', 'second', 'third']
        pp.split_all_into_trials(start_times, end_times, trial_labels)


    def test_split_into_trials_by_duration(self):
       
        # Eyelink dataset samples
        session_path = os.path.join(current_folder, "example_dataset_derivatives/sub-0001/ses-second/")
        samples = pd.read_feather(os.path.join(session_path, 'samples.feather'))
        
        fixations = saccades = blinks = samples.copy(True) # whatever
        user_messages = pd.read_feather(os.path.join(session_path, 'msg.feather'))
        
        
        pp = PreProcessing(samples, fixations, saccades, blinks, user_messages, session_path)
        
        start_msgs = {'search':['beginning_of_stimuli']}
        
        durations = {'search': [500 for x in user_messages[user_messages['message']== 'beginning_of_stimuli'].iterrows()]}
        
        trial_labels = ['first', 'second', 'third']
        ##pp.split_all_into_trials(start_times, end_times, trial_labels)
        pp.split_all_into_trials_by_durations(start_msgs, durations, trial_labels)
        
        
        self.assertTrue('trial_number' in samples.columns)
        self.assertTrue('phase' in samples.columns)
        self.assertTrue('trial_label' in samples.columns)        
        #self.assertTrue('1' in samples.phase.unique())



if __name__ == "__main__":
    unittest.main()
