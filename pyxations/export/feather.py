'''
Created on 2 dic 2024

@author: placiana
'''

import polars as pl

class FeatherExport(object):
    def save(self, df, path, data_name, *args, **kwargs):
        df.write_ipc((path / f'{data_name}.feather'), ) if isinstance(df, pl.DataFrame) else df.reset_index(drop=True).to_feather((path / f'{data_name}.feather'))


    def read(self, path, data_name):
        return pl.read_ipc((path / f'{data_name}.feather'),memory_map=False).drop(["__index_level_0__","line_number"],strict=False)

    def extension(self):
        return '.feather'