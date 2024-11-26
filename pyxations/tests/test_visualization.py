'''
Created on Nov 22, 2024

@author: placiana
'''
import unittest
from pyxations.visualization.samples import SampleVisualization
import pandas as pd
import os


class Test(unittest.TestCase):


    def testName(self):
        print(os.getcwd())
        df = pd.read_hdf('../../gazepoint_dataset_derivatives/sub-0001/ses-ses-A/samples.hdf5')
        SampleVisualization(df)
        vis  = SampleVisualization(df, 1024, 768)
        vis.animate(display=False, out_file='anim.gif', in_percent=True)
        self.assertTrue(os.path.exists('anim.gif'))

    def testAnimWebgazer(self):
        print(os.getcwd())
        current_folder = os.getcwd()
        current_folder = os.path.dirname(current_folder)
        df_path = os.path.join(current_folder, "..", "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.hdf5")
        df = pd.read_hdf(df_path)
        SampleVisualization(df)
        vis  = SampleVisualization(df, screen_width=1366, screen_height=768)
        vis.animate(display=False, out_file='animWebgazer.gif')
        self.assertTrue(os.path.exists('animWebgazer.gif'))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()