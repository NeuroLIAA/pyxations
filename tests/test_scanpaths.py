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
        exp.plot_scanpaths(screen_height=1080, screen_width=1920)

        # Assert that the scanpath file exists
        path_to_derivatives = os.path.join(data_folder, "example_dataset_derivatives")
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'sub-0001','ses-second','eyelink_events','plots', "scanpath_0_search.png")))

""" 
    def test_scanpaths_webgazer(self):
        path_to_derivatives = os.path.join(data_folder, "antisacadas_dataset_derivatives")
        # Create an experiment
        exp = Experiment(os.path.join(data_folder, "antisacadas_dataset"))
        exp.load_data("remodnav")
        exp.plot_scanpaths(screen_height=1080, screen_width=1920)

        # Assert that the scanpath file exists
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'sub-0001','ses-antisacadas','remodnav_events','plots', "scanpath_0_search.png")))
 """



if __name__ == "__main__":
    unittest.main()
