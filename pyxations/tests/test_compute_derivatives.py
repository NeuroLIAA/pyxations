import unittest
from pyxations import compute_derivatives_for_dataset
import os

class TestComputeDerivatives(unittest.TestCase):
    def test_compute_derivatives(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"example_dataset")
        msg_keywords = ["begin","end","press"]
        start_msgs = ['beginning_of_stimuli']
        end_msgs = ['end_of_stimuli']
        detection_algorithm = 'eyelink'
        compute_derivatives_for_dataset(bids_dataset_folder, msg_keywords, detection_algorithm, start_msgs=start_msgs, end_msgs=end_msgs)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second_half")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second_half", "samples.hdf5")))

if __name__ == "__main__":
    unittest.main()