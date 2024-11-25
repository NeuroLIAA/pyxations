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
        vis  = SampleVisualization(df)
        vis.animate(display=False, out_file='anim.gif')
        self.assertTrue(os.path.exists('anim.gif'))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()