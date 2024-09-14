import os
import unittest
from pyxations import Visualization
import pandas as pd

class TestMultipanel(unittest.TestCase):
    def test_multipanel(self):

        # Get path to samples from parsed edf
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)

        path_to_session = os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second")
        fixations = pd.read_hdf(path_or_buf=os.path.join(path_to_session,'eyelink_events', "fix.hdf5"))
        saccades = pd.read_hdf(path_or_buf=os.path.join(path_to_session,'eyelink_events', "sacc.hdf5"))
        
        visualization = Visualization(path_to_session,'eyelink',fixations, saccades,None)

        # Plot multipanel
        visualization.plot_multipanel()
        # Assert that the file multipanel.png was created
        self.assertTrue(os.path.exists(os.path.join(path_to_session,'eyelink_events','plots', "multipanel.png")))


if __name__ == "__main__":
    unittest.main()
