import unittest
#from pyxations import pyx.dataset_to_bids
import os

import pyxations as pyx
from pathlib import Path

current_path = Path(__file__).resolve()
data_folder = os.path.join(current_path.parent, 'data')

class TestDatasetToBids(unittest.TestCase):
    def test_dataset_to_bids(self):
        files_folder_path = os.path.join(data_folder,"example_files")
        bids_dataset_folder = pyx.dataset_to_bids(data_folder,files_folder_path,"example_dataset")
        #bids_dataset_folder = pyx.dataset_to_bids(current_folder,files_folder_path,"antisacadas_dataset", format_name='eyelink')
        self.assertTrue(os.path.exists(bids_dataset_folder))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001", "ses-second")))


    def test_webgazer_dataset_to_bids(self):
        files_folder_path = os.path.join(data_folder,"antisacadas_files")
        bids_dataset_folder = pyx.dataset_to_bids(data_folder, files_folder_path,"antisacadas_dataset", format_name='webgazer')
        self.assertTrue(os.path.exists(bids_dataset_folder))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001")))
        #self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001", "ses-second")))


    def test_tobii_dataset_to_bids(self):
        files_folder_path = os.path.join(data_folder,"tobii_files")
        bids_dataset_folder = pyx.dataset_to_bids(data_folder,files_folder_path,"tobii_dataset", format_name='tobii')
        self.assertTrue(os.path.exists(bids_dataset_folder))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001")))
        #self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001", "ses-second")))

    def test_gaze_dataset_to_bids(self):
        files_folder_path = os.path.join(data_folder,"gazepoint_files")
        bids_dataset_folder = pyx.dataset_to_bids(data_folder,files_folder_path,"gazepoint_dataset", format_name='gaze')
        self.assertTrue(os.path.exists(bids_dataset_folder))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001")))
        #self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001", "ses-second")))


if __name__ == "__main__":
    unittest.main()