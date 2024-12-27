import unittest
from pyxations import compute_derivatives_for_dataset
import os
from pyxations.export import FEATHER_EXPORT

class TestComputeDerivatives(unittest.TestCase):
    def test_compute_derivatives_eyelink(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"example_dataset")
        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'eyelink'
        compute_derivatives_for_dataset(bids_dataset_folder, 'eyelink', detection_algorithm, msg_keywords=msg_keywords, 
                                        start_msgs=start_msgs, end_msgs=end_msgs, overwrite=True)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second", "samples.hdf5")))


    def test_compute_derivatives_webgazer(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"antisacadas_dataset")
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, 'webgazer', detection_algorithm, screen_height=768, screen_width=1024)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.hdf5")))



    def test_compute_derivatives_tobii(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"tobii_dataset")

        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, 'tobii', detection_algorithm, )
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives", "sub-0001", "ses-sceneviewing")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives", "sub-0001", "ses-sceneviewing", "samples.hdf5")))

    def test_compute_derivatives_gazepoint(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"gazepoint_dataset")

        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, 'gaze', detection_algorithm, overwrite=True)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A", "samples.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A", "blink.hdf5")))

    def test_compute_derivatives_feather_format(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"example_dataset")
        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'eyelink'
        dataset_type = 'eyelink'
        compute_derivatives_for_dataset(bids_dataset_folder, dataset_type, detection_algorithm, 
                                        msg_keywords=msg_keywords,start_msgs=start_msgs, 
                                        end_msgs=end_msgs, overwrite=True, export_format=FEATHER_EXPORT)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second", "samples.feather")))

    def test_compute_derivatives_webgazer_feather(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        bids_dataset_folder = os.path.join(current_folder,"antisacadas_dataset")

        detection_algorithm = 'webgazer'
        compute_derivatives_for_dataset(bids_dataset_folder, detection_algorithm, 
                                        export_format=FEATHER_EXPORT, screen_height=768, screen_width=1024)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.feather")))


if __name__ == "__main__":
    unittest.main()