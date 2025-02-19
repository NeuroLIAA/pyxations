'''
Created on 2 dic 2024

@author: placiana
'''

import pandas as pd
import pyarrow

class FeatherExport(object):
    def save(self, df, path, data_name, *args, **kwargs):
        try:
            df.to_feather((path / f'{data_name}.feather'), )
        except ValueError as e:
            df.reset_index().to_feather((path / f'{data_name}.feather'), )

    def read(self, path, data_name):
        return pd.read_feather((path / f'{data_name}.feather'))

    def extension(self):
        return '.feather'