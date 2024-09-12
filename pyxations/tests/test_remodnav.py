import unittest
import os
from pyxations import RemodnavDetection

class TestRemodnav(unittest.TestCase):
    def test_remodnav(self):

        # Get path to samples from parsed edf
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)

        path_to_session = os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second_half")

        eye_detection = RemodnavDetection(path_to_session)

        # Run eye movements detection and save results
        fixations, saccades, time = eye_detection.detect_eye_movements()

        # Assert that there are fixations and saccades
        self.assertTrue(len(fixations) > 0)
        self.assertTrue(len(saccades) > 0)

if __name__ == "__main__":
    unittest.main()