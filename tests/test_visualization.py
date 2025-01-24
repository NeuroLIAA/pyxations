'''
Created on Nov 22, 2024

@author: placiana
'''
import unittest
from pyxations.visualization.samples import SampleVisualization
import pandas as pd
import os
from pathlib import Path


current_path = Path(__file__).resolve()
current_folder = current_path.parent.parent

class Test(unittest.TestCase):

    def testPlot(self):
        df = pd.read_hdf(
            os.path.join(current_folder, 'gazepoint_dataset_derivatives/sub-0001/ses-ses-A/samples.hdf5'))
        
        vis  = SampleVisualization(df, 1024, 768)
        vis.plot(in_percent=True)
        self.assertTrue(os.path.exists('scanpath.png'))

    def testName(self):
        df = pd.read_hdf(
            os.path.join(current_folder, 'gazepoint_dataset_derivatives/sub-0001/ses-ses-A/samples.hdf5'))
        
        vis  = SampleVisualization(df, 1024, 768)
        vis.animate(display=False, out_file='anim.gif', in_percent=True)
        self.assertTrue(os.path.exists('anim.gif'))

    def testAnimWebgazer(self):

        df_path = os.path.join(current_folder, "antisacadas_dataset_derivatives", "sub-0001", "ses-antisacadas", "samples.hdf5")
        df = pd.read_hdf(df_path)
        
        vis  = SampleVisualization(df, screen_width=1366, screen_height=768)
        vis.animate(display=False, out_file='animWebgazer.gif')
        self.assertTrue(os.path.exists('animWebgazer.gif'))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()