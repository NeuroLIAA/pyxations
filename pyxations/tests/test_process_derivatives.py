import unittest
from pyxations import process_derivatives
import os

class TestProcessDerivatives(unittest.TestCase):
    def test_process_derivatives(self):

        start_msgs = ['beginning_of_stimuli']
        end_msgs = ['end_of_stimuli']

        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)

        derivatives_folder_path = os.path.join(current_folder, "example_dataset_derivatives")
        process_derivatives(derivatives_folder_path,start_msgs,end_msgs)
        
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
