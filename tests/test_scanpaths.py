import os
import unittest
from pyxations import Experiment
from pathlib import Path

current_path = Path(__file__).resolve()
data_folder = os.path.join(current_path.parent, 'data')

class TestScanpaths(unittest.TestCase):
    def test_scanpaths(self):
        
        # Create an experiment
        exp = Experiment(os.path.join(data_folder, "example_dataset"))
        exp.load_data("eyelink")
        exp.subjects['0001'].get_trial(session_id='second', trial_number=0).plot_scanpath(1080, 1920, display=False)

        # Assert that the scanpath file exists
        path_to_derivatives = os.path.join(data_folder, "example_dataset_derivatives")
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'sub-0001','ses-second','eyelink_events','plots', "scanpath_0_search.png")))



if __name__ == "__main__":
    unittest.main()