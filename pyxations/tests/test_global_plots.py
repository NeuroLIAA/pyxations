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
        image_path_mapper = {}
        for subject in [fol for fol in os.listdir(path_to_derivatives) if fol.startswith('sub-')]:
            image_path_mapper[subject] = {}
            for session in [fol for fol in os.listdir(os.path.join(path_to_derivatives,subject)) if fol.startswith('ses-')]:
                image_path_mapper[subject][session] = {}
                for i in range(100):
                    image_path_mapper[subject][session][i] = os.path.join(current_folder,"example_images","test_img.jpg")
        

        # Plot multipanel
        visualization.global_plots(16,image_path_mapper)
        # Assert that the file multipanel.png was created
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'eyelink_events','plots', "multipanel.png")))
        self.assertTrue(os.path.exists(os.path.join(path_to_derivatives,'sub-0001','ses-second','eyelink_events','plots', "scanpath_0.png")))


if __name__ == "__main__":
    unittest.main()
