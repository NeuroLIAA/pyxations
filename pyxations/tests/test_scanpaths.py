import os
import unittest
from pyxations import Experiment


class TestScanpaths(unittest.TestCase):
    def test_scanpaths(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        path_to_derivatives = os.path.join(current_folder, "example_dataset_derivatives")
        # Create an experiment
        exp = Experiment(os.path.join(current_folder,"example_dataset"))
        exp.load_data("eyelink")
        exp.plot_scanpaths(screen_height=1080, screen_width=1920)

        # Assert that the scanpath file exists
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'sub-0001','ses-second','eyelink_events','plots', "scanpath_0_search.png")))

if __name__ == "__main__":
    unittest.main()
