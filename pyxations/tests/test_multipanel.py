import os
import unittest
from pyxations import Visualization

class TestVisualization(unittest.TestCase):
    def test_visualization(self):

        # Get path to samples from parsed edf
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)

        path_to_session = os.path.join(current_folder, "example_dataset_derivatives", "sub-ab01", "ses-second_half")

        visualization = Visualization(path_to_session)

        # Plot multipanel
        visualization.plot_multipanel()

        # Assert that the file multipanel.png was created
        self.assertTrue(os.path.exists(os.path.join(path_to_session, "multipanel.png")))


if __name__ == "__main__":
    unittest.main()
