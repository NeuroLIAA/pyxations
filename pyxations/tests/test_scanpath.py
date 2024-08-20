import os
import unittest
from pyxations import Visualization
import pandas as pd

class TestScanpath(unittest.TestCase):
    def test_scanpath(self):

        # Get path to samples from parsed edf
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)

        path_to_session = os.path.join(current_folder, "example_dataset_derivatives", "sub-ab01", "ses-second_half")

        visualization = Visualization(path_to_session,'eyelink')

        # Load samples.hdf5 file
        path_to_samples = os.path.join(path_to_session, "samples.hdf5")
        path_to_fixations = os.path.join(path_to_session,'eyelink_events', "fix.hdf5")
        path_to_saccades = os.path.join(path_to_session,'eyelink_events', "sacc.hdf5")

        samples = pd.read_hdf(path_or_buf=path_to_samples)
        fixations = pd.read_hdf(path_or_buf=path_to_fixations)
        saccades = pd.read_hdf(path_or_buf=path_to_saccades)

        img_path = os.path.join('..','example_images','test_img.jpg')
        
        # Plot Scanpath
        visualization.scanpath(fixations=fixations,tmin=samples['tSample'][100000], tmax=samples['tSample'][110000], img_path=img_path, saccades=saccades, samples=samples)

        # Assert that the file multipanel.png was created
        self.assertTrue(os.path.exists(os.path.join(path_to_session,'eyelink_events', "scanpath.png")))


if __name__ == "__main__":
    unittest.main()
