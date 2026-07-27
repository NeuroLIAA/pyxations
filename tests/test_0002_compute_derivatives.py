import unittest
from pyxations import compute_derivatives_for_dataset
import os
from pyxations.export import FEATHER_EXPORT, HDF5_EXPORT
from pathlib import Path
import shutil


current_path = Path(__file__).resolve()
data_folder = os.path.join(current_path.parent, 'data')

HAS_EDF2ASC = shutil.which("edf2asc") is not None

class TestComputeDerivatives(unittest.TestCase):
    @unittest.skipUnless(HAS_EDF2ASC, "edf2asc (EyeLink Developers Kit) not available on this platform")
    def test_compute_derivatives_eyelink(self):
        bids_dataset_folder = os.path.join(data_folder,"example_dataset")
        derivatives_path = os.path.join(data_folder, "example_dataset_derivatives")
        
        # Remove the target directory to ensure a clean test environment.
        if os.path.exists(derivatives_path):
            shutil.rmtree(derivatives_path)

        
        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'eyelink'
        compute_derivatives_for_dataset(bids_dataset_folder, 'eyelink', detection_algorithm, msg_keywords=msg_keywords, 
                                        start_msgs=start_msgs, end_msgs=end_msgs, overwrite=False, exp_format=HDF5_EXPORT)
        self.assertTrue(os.path.exists(derivatives_path))
        self.assertTrue(os.path.exists(os.path.join(derivatives_path, "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(derivatives_path, "sub-0001", "ses-second")))
        self.assertTrue(os.path.exists(os.path.join(derivatives_path, "sub-0001", "ses-second", "samples.hdf5")))


    @unittest.skipUnless(HAS_EDF2ASC, "edf2asc (EyeLink Developers Kit) not available on this platform")
    def test_compute_derivatives_eyelink_remodnav(self):
        bids_dataset_folder = os.path.join(data_folder,"example_dataset")
        derivatives_path = os.path.join(data_folder, "example_dataset_derivatives")
        
        # Remove the target directory to ensure a clean test environment.
        if os.path.exists(derivatives_path):
            shutil.rmtree(derivatives_path)



        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(
            bids_dataset_folder, 'eyelink', detection_algorithm, msg_keywords=msg_keywords, 
            start_msgs=start_msgs, end_msgs=end_msgs, overwrite=True,
            max_pso_dur=0, min_fix_dur=0, sac_max_vel=999, savgol_length= 0.195
        )
        self.assertTrue(os.path.exists(derivatives_path))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "example_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "example_dataset_derivatives", "sub-0001", "ses-second")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "example_dataset_derivatives", "sub-0001", "ses-second", "samples.feather")))


    def test_compute_derivatives_webgazer(self):
        bids_dataset_folder = os.path.join(data_folder,"antisacadas_dataset")
        derivatives_path = os.path.join(data_folder, "antisacadas_dataset_derivatives")
        
        # Remove the target directory to ensure a clean test environment.
        if os.path.exists(derivatives_path):
            shutil.rmtree(derivatives_path)

        
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
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.hdf5")))

    def test_compute_derivatives_tobii(self):
        bids_dataset_folder = os.path.join(data_folder,"tobii_dataset")
        derivatives_path = os.path.join(data_folder, "tobii_dataset_derivatives")
        
        # Remove the target directory to ensure a clean test environment.
        if os.path.exists(derivatives_path):
            shutil.rmtree(derivatives_path)
        
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(
            bids_dataset_folder, 'tobii', detection_algorithm, exp_format=HDF5_EXPORT, overwrite=True)
        self.assertTrue(os.path.exists(os.path.join(data_folder, "tobii_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "tobii_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "tobii_dataset_derivatives", "sub-0001", "ses-sceneviewing")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "tobii_dataset_derivatives", "sub-0001", "ses-sceneviewing", "samples.hdf5")))

    def test_compute_derivatives_gazepoint(self):
        bids_dataset_folder = os.path.join(data_folder,"gazepoint_dataset")
        derivatives_path = os.path.join(data_folder, "gazepoint_dataset_derivatives")

        # Remove the target directory to ensure a clean test environment.
        if os.path.exists(derivatives_path):
            shutil.rmtree(derivatives_path)

        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, 'gaze', detection_algorithm, 
                                        overwrite=True, exp_format=HDF5_EXPORT)
        self.assertTrue(os.path.exists(os.path.join(data_folder, "gazepoint_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "gazepoint_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A", "samples.hdf5")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "gazepoint_dataset_derivatives", "sub-0001", "ses-ses-A", "remodnav_events", "blink.hdf5")))

    @unittest.skipUnless(HAS_EDF2ASC, "edf2asc (EyeLink Developers Kit) not available on this platform")
    def test_compute_derivatives_feather_format(self):
        bids_dataset_folder = os.path.join(data_folder,"example_dataset")
        msg_keywords = ["begin","end","press"]
        start_msgs = {'search':['beginning_of_stimuli']}
        end_msgs = {'search':['end_of_stimuli']}
        detection_algorithm = 'eyelink'
        dataset_type = 'eyelink'
        compute_derivatives_for_dataset(bids_dataset_folder, dataset_type, detection_algorithm, 
                                        msg_keywords=msg_keywords,start_msgs=start_msgs, 
                                        end_msgs=end_msgs, overwrite=True, export_format=FEATHER_EXPORT)

        self.assertTrue(os.path.exists(os.path.join(data_folder, "example_dataset_derivatives", "sub-0001", "ses-second", "samples.feather")))

    def test_compute_derivatives_webgazer_feather(self):
        data_folder = os.path.join(current_path.parent, 'data')
        bids_dataset_folder = os.path.join(data_folder,"antisacadas_dataset")
        derivatives_path = os.path.join(data_folder, "antisacadas_dataset_derivatives")
        
        # Remove the target directory to ensure a clean test environment.
        if os.path.exists(derivatives_path):
            shutil.rmtree(derivatives_path)

        
        dataset_type = 'webgazer'
        detection_algorithm = 'remodnav'
        compute_derivatives_for_dataset(bids_dataset_folder, dataset_type, detection_algorithm, 
                                        export_format=FEATHER_EXPORT, screen_height=768, screen_width=1024)
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives", "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas")))
        self.assertTrue(os.path.exists(os.path.join(data_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.feather")))


    def test_compute_derivatives_webgazer_behavioral_columns(self):
        data_folder = os.path.join(current_path.parent, 'data')
        bids_dataset_folder = os.path.join(data_folder, "antisacadas_dataset")
        derivatives_path = os.path.join(data_folder, "antisacadas_dataset_derivatives")

        if os.path.exists(derivatives_path):
            shutil.rmtree(derivatives_path)

        session_path = os.path.join(
            derivatives_path, "sub-0001", "ses-antisacadas"
        )

        compute_derivatives_for_dataset(
            bids_dataset_folder, 'webgazer', 'remodnav',
            overwrite=True, exp_format=FEATHER_EXPORT,
            screen_height=768, screen_width=1024,
            behavioral_columns=['typeOfSaccade', 'cueShownAtLeft', 'rt'],
        )

        import pandas as pd

        samples_path = os.path.join(session_path, "samples.feather")
        self.assertTrue(os.path.exists(samples_path), "samples.feather not found")

        df = pd.read_feather(samples_path)
        self.assertIn("typeOfSaccade", df.columns, "typeOfSaccade not propagated to samples")
        self.assertIn("cueShownAtLeft", df.columns, "cueShownAtLeft not propagated to samples")

        events_path = os.path.join(session_path, "events.tsv")
        self.assertTrue(os.path.exists(events_path), "events.tsv not created")

        df_events = pd.read_csv(events_path, sep="\t")
        self.assertIn("onset", df_events.columns)
        self.assertIn("typeOfSaccade", df_events.columns)


if __name__ == "__main__":
    unittest.main()