import os
import unittest
from pyxations import Experiment
from pathlib import Path

current_path = Path(__file__).resolve()
current_folder = current_path.parent.parent

class TestMultipanel(unittest.TestCase):
    def test_multipanel(self):
        path_to_derivatives = os.path.join(current_folder, "example_dataset_derivatives")
        # Create an experiment
        exp = Experiment(os.path.join(current_folder,"example_dataset"))
        exp.load_data("eyelink")
        exp.plot_multipanel(display=False)

        # Assert that the file multipanel.png was created
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'eyelink_events','plots', "multipanel_search.png")))

if __name__ == "__main__":
    unittest.main()
