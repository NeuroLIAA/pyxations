'''
Created on 2 dic 2024

@author: placiana
'''

import pandas as pd


class HDFExport(object):
    def save(self, df, path, data_name, key, **kwargs):
        df.to_hdf((path / f'{data_name}.hdf5'), key=key, mode='w')
    
    def read(self, path):
        return pd.read_hdf(path, memory_map=True)
    
    def extension(self):
        return 'hdf5'
    
    #dfHeader.to_hdf((session_folder_path / 'header.hdf5'), key='header', mode='w')