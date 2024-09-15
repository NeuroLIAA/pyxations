import os
import unittest
from pyxations import Visualization

class TestGlobalPlots(unittest.TestCase):
    def test_multipanel(self):

        # Get path to samples from parsed edf
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)

        path_to_derivatives = os.path.join(current_folder, "example_dataset_derivatives")

        visualization = Visualization(path_to_derivatives,'eyelink')

        # Plot multipanel
        visualization.global_plots(16)
        # Assert that the file multipanel.png was created
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'eyelink_events','plots', "multipanel.png")))
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'sub-0001','ses-second','eyelink_events','plots', "scanpath_0.png")))


if __name__ == "__main__":
    unittest.main()
