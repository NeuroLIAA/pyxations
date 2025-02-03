import unittest
from pyxations import compute_derivatives_for_dataset
import os
from pyxations.export import FEATHER_EXPORT, HDF5_EXPORT
from pathlib import Path

current_path = Path(__file__).resolve()
current_folder = current_path.parent.parent

class TestComputeDerivatives(unittest.TestCase):
    def test_compute_derivatives_eyelink(self):
        bids_dataset_folder = os.path.join(current_folder,"example_dataset")
        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'eyelink'
        compute_derivatives_for_dataset(bids_dataset_folder, 'eyelink', detection_algorithm, msg_keywords=msg_keywords, 
                                        start_msgs=start_msgs, end_msgs=end_msgs, overwrite=False, exp_format=HDF5_EXPORT)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second", "samples.hdf5")))


    def test_compute_derivatives_eyelink_remodnav(self):
        bids_dataset_folder = os.path.join(current_folder,"example_dataset")
        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(
            bids_dataset_folder, 'eyelink', detection_algorithm, msg_keywords=msg_keywords, 
            start_msgs=start_msgs, end_msgs=end_msgs, overwrite=True,
            max_pso_dur=0, min_fix_dur=0, sac_max_vel=999, savgol_length= 0.195
        )
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "example_dataset_derivatives", "sub-0001", "ses-second", "samples.feather")))


    def test_compute_derivatives_webgazer(self):
        bids_dataset_folder = os.path.join(current_folder,"antisacadas_dataset")
        start_times = {
            0: [100, 501, 1001],
        }
        end_times = {
            0: [500, 1000, 2000],
        }
        trial_labels = {0:['first', 'second', 'third'], 1: ['fourth']}
        
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(
            bids_dataset_folder, 'webgazer', detection_algorithm, overwrite=True, 
            exp_format=HDF5_EXPORT, screen_height=768, screen_width=1024,
            start_times=start_times, end_times=end_times, trial_labels=trial_labels)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.hdf5")))


    def test_compute_derivatives_tobii(self):
        bids_dataset_folder = os.path.join(current_folder,"tobii_dataset")

        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, 'tobii', detection_algorithm, exp_format=HDF5_EXPORT, overwrite=True)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives", "sub-0001", "ses-sceneviewing")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "tobii_dataset_derivatives", "sub-0001", "ses-sceneviewing", "samples.hdf5")))

    def test_compute_derivatives_gazepoint(self):
        bids_dataset_folder = os.path.join(current_folder,"gazepoint_dataset")

        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, 'gaze', detection_algorithm, 
                                        overwrite=True, exp_format=HDF5_EXPORT)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A", "samples.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A", "remodnav_events", "blink.hdf5")))

    def test_compute_derivatives_feather_format(self):
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
        bids_dataset_folder = os.path.join(current_folder,"antisacadas_dataset")

        dataset_type = 'webgazer'
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, dataset_type, detection_algorithm, 
                                        export_format=FEATHER_EXPORT, screen_height=768, screen_width=1024)
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.feather")))


if __name__ == "__main__":
    unittest.main()