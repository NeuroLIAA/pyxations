import unittest
from pyxations import dataset_to_bids
import os

class TestDatasetToBids(unittest.TestCase):
    def test_dataset_to_bids(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        files_folder_path = os.path.join(current_folder,"example_files")
        bids_dataset_folder = dataset_to_bids(current_folder,files_folder_path,"example_dataset")
        #bids_dataset_folder = dataset_to_bids(current_folder,files_folder_path,"antisacadas_dataset", format_name='eyelink')
        self.assertTrue(os.path.exists(bids_dataset_folder))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001")))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001", "ses-second")))


    def test_webgazer_dataset_to_bids(self):
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        files_folder_path = os.path.join(current_folder,"antisacadas_files")
        bids_dataset_folder = dataset_to_bids(current_folder,files_folder_path,"antisacadas_dataset", format_name='webgazer')
        self.assertTrue(os.path.exists(bids_dataset_folder))
        self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001")))
        #self.assertTrue(os.path.exists(os.path.join(bids_dataset_folder, "sub-0001", "ses-second")))


if __name__ == "__main__":
    unittest.main()