import unittest
from pyxations.bids_formatting import compute_derivatives_for_dataset
import os

class TestComputeDerivatives(unittest.TestCase):
    def test_compute_derivatives(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"example_dataset")
        msg_keywords = ["begin","end","press"]
        compute_derivatives_for_dataset(bids_dataset_folder, msg_keywords)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-ab01")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-ab01", "ses-second_half")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-ab01", "ses-second_half", "samples.hdf5")))

if __name__ == "__main__":
    unittest.main()