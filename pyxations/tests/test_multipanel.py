import os
import unittest
from pyxations import Visualization, PostProcessing
import pandas as pd

class TestMultipanel(unittest.TestCase):
    def test_multipanel(self):

        # Get path to samples from parsed edf
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)

        path_to_session = os.path.join(current_folder, "example_dataset_derivatives", "sub-ab01", "ses-second_half")

        visualization = Visualization(path_to_session,'eyelink')
        post_processing = PostProcessing(path_to_session,'eyelink')
        fixations = pd.read_hdf(path_or_buf=os.path.join(path_to_session,'eyelink_events', "fix.hdf5"))
        saccades = pd.read_hdf(path_or_buf=os.path.join(path_to_session,'eyelink_events', "sacc.hdf5"))
        saccades = post_processing.saccades_direction(saccades)
        # Plot multipanel
        visualization.plot_multipanel(fixations, saccades)

        # Assert that the file multipanel.png was created
        self.assertTrue(os.path.exists(os.path.join(path_to_session,'eyelink_events','plots', "multipanel.png")))


if __name__ == "__main__":
    unittest.main()
